from typing import Optional, Dict, Union
import torch
from torch import nn
from mindspeed_mm.models.vlm_model import VLMModel

from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from mindspeed_mm.models.vla_modules.modeling_scaledp import ScaleDP
from mindspeed_mm.models.vla_modules.configuration_scaledp import ScaleDPPolicyConfig

from mindspeed_mm.models.action_head import MLPResNet


class VLADP(nn.Module):

    def __init__(self, vlm_config, action_head_config=None):
        super().__init__()
        self.config = core_transformer_config_from_args(get_args())
        self.vlm_model = VLMModel(vlm_config)
        self.pre_process = self.vlm_model.text_decoder.pre_process
        self.post_process = self.vlm_model.text_decoder.post_process
        self.share_embeddings_and_output_weights = not getattr(vlm_config.text_decoder, 'untie_embeddings_and_output_weights', True)
        self.vlm_model.text_decoder.post_process = False
        self.stop_token = vlm_config.stop_token
        self.action_start_token_id = vlm_config.action_start_token_id
        self.state_dim = action_head_config['state_dim']
        self.cond_vit_embeds = False
        self.cond_vit_memory = False
        if action_head_config is not None:
            print(action_head_config)
            if 'cond_vit_embeds' in action_head_config.keys():
                self.cond_vit_embeds = action_head_config.pop('cond_vit_embeds')
            if 'cond_vit_memory' in action_head_config.keys():
                self.cond_vit_memory = action_head_config.pop('cond_vit_memory')
                if not "memory_config" in action_head_config.keys():
                    action_head_config["memory_config"] = "mlp"
                if action_head_config['memory_config'] == 'mlp':
                    self.vit_memory = MLPResNet(1, action_head_config['cond_dim'], action_head_config['cond_dim'], action_head_config['cond_dim'])
                elif action_head_config['memory_config'] == "linear":
                    self.vit_memory = nn.Linear(action_head_config['cond_dim'], action_head_config['cond_dim'])
                self.cond_vit_embeds = True
            scale_dp_config = ScaleDPPolicyConfig(**action_head_config)
            self.action_head = ScaleDP(scale_dp_config)
            # self.action_head = AutoModel.from_config(config=action_head_config)
        
        total_params = sum(p.numel() for p in self.parameters())
        vlm_params = sum(p.numel() for p in self.vlm_model.parameters())
        head_params = sum(p.numel() for p in self.action_head.parameters())
        print("Total number of parameters: ", total_params, vlm_params, head_params)

    def forward(self,
                input_ids: torch.Tensor,
                input_embeds: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.Tensor] = None,
                image_grid_thw: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                states: Optional[torch.Tensor] = None,
                ground_truth_actions: Optional[torch.Tensor] = None,
                noisy_actions: Optional[torch.Tensor] = None,
                inference_params: Optional[InferenceParams] = None,
                decoder_input: Optional[torch.FloatTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                packed_seq_params: Optional[PackedSeqParams] = None,
                extra_block_kwargs: Optional[dict] = None,
                cache_position: Optional[torch.LongTensor] = None,
                rope_deltas: Optional[torch.LongTensor] = None,
                image_flags: Optional[torch.LongTensor] = None,
                *args,
                **kwargs) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        # shape convert to: (action_dim * action_chunk, bsz)
        bsz = input_ids.shape[0]
        if self.state_dim == 0:
            states = None
        vlm_output = self.vlm_model(input_ids, input_embeds, pixel_values, image_grid_thw,
                                    attention_mask, labels, inference_params,
                                    decoder_input, position_ids,
                                    packed_seq_params, extra_block_kwargs,
                                    cache_position, rope_deltas, image_flags, return_vit_embeds=self.cond_vit_embeds,
                                    *args, **kwargs)
        if self.post_process:
            if self.cond_vit_embeds or self.cond_vit_memory:
                vlm_output, vit_embeds = vlm_output
                if self.cond_vit_memory:
                    vit_embeds = self.vit_memory(vit_embeds)
                vlm_output = torch.cat((vit_embeds.unsqueeze(1), vlm_output), dim=0)
            # change to shape (N, seq_len, D)
            action_hidden_states = vlm_output.transpose(0, 1)
            is_pad = torch.zeros((bsz, ground_truth_actions.shape[1])).to(device='cuda', dtype=torch.bool)
            ret = self.action_head(actions=ground_truth_actions, hidden_states=action_hidden_states, states=states, is_pad=is_pad)
            return ret
        return vlm_output


    def set_input_tensor(self, input_tensor):
        self.vlm_model.set_input_tensor(input_tensor)

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        state_dict = {'vlm': self.vlm_model.state_dict_for_save_checkpoint(prefix, keep_vars)}
        action_head_state = self.action_head.state_dict(prefix=prefix, keep_vars=keep_vars)
        state_dict['action_head'] = {k: v.clone() for k, v in action_head_state.items()}
        if self.cond_vit_memory:
            state_dict['vit_memory'] = self.vit_memory.state_dict(prefix=prefix, keep_vars=keep_vars)
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        if 'vlm' in state_dict.keys() and 'action_head' in state_dict.keys():
            ret = self.vlm_model.load_state_dict(state_dict['vlm'], strict=strict)
            self.action_head.load_state_dict(state_dict['action_head'], strict=strict)
            if "vit_memory" in state_dict.keys():
                self.vit_memory.load_state_dict(state_dict["vit_memory"], strict=strict)
            return ret
        return super().load_state_dict(state_dict, strict)
