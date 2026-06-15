from typing import Optional, Dict, Union
import torch
from torch import nn
from mindspeed_mm.models.vlm_model import VLMModel

from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from mindspeed_mm.models.action_head import L1RegressionActionHead


class VLARegressionModel(nn.Module):

    def __init__(self, vlm_config, action_head_config=None):
        super().__init__()
        self.config = core_transformer_config_from_args(get_args())
        self.vlm_model = VLMModel(vlm_config)
        self.vlm_model.text_decoder.post_process = False
        self.action_start_token_id = vlm_config.action_start_token_id
        self.stop_token = vlm_config.stop_token
        if action_head_config is not None:
            self.action_normalization = action_head_config.pop('action_normalization')
            self.action_head = L1RegressionActionHead(**action_head_config)
        total_params = sum(p.numel() for p in self.parameters())
        vlm_params = sum(p.numel() for p in self.vlm_model.parameters())
        head_params = sum(p.numel() for p in self.action_head.parameters())
        print("Total number of parameters: ", total_params, vlm_params, head_params)

    def forward(self,
                input_ids: torch.Tensor,
                pixel_values: Optional[torch.Tensor] = None,
                image_grid_thw: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
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
        action_dim = ground_truth_actions.shape[-1]
        action_mask = (input_ids >= self.action_start_token_id).to(device=input_ids.device)
        input_ids[action_mask] = self.action_start_token_id
        action_mask = action_mask.transpose(0, 1)
        vlm_output = self.vlm_model(input_ids, None, pixel_values, image_grid_thw,
                                    attention_mask, labels, inference_params,
                                    decoder_input, position_ids,
                                    packed_seq_params, extra_block_kwargs,
                                    cache_position, rope_deltas, image_flags,
                                    *args, **kwargs)
        action_hidden = vlm_output[action_mask].unsqueeze(0)
        action_predict = self.action_head.predict_action(action_hidden)
        regression_loss = nn.functional.l1_loss(action_predict,
                                                ground_truth_actions,
                                                reduction="mean")
        return {'loss': regression_loss}


    def set_input_tensor(self, input_tensor):
        self.vlm_model.set_input_tensor(input_tensor)

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        state_dict = {'vlm': self.vlm_model.state_dict_for_save_checkpoint(prefix, keep_vars)}
        state_dict['action_head'] = self.action_head.state_dict(prefix=prefix, keep_vars=keep_vars)
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        if 'vlm' in state_dict.keys() and 'action_head' in state_dict.keys():
            ret = self.vlm_model.load_state_dict(state_dict['vlm'])
            self.action_head.load_state_dict(state_dict['action_head'])
            return ret
        return super().load_state_dict(state_dict, strict)

    def prepare_input_for_action_prediction(self, input_ids, attention_mask, NUM_ACTIONS_CHUNK=1, ACTION_DIM=7):
        """Prepares input for action prediction by adding necessary tokens"""
        # Add (ACTION_DIM * NUM_ACTIONS_CHUNK) placeholder tokens to input_ids to simulate action tokens
        actual_input_len = attention_mask.sum(-1)
        pad_len = (actual_input_len + ACTION_DIM + 7) // 8 * 8
        tail_len = pad_len - actual_input_len - ACTION_DIM
        pad_ids = (torch.ones((1, tail_len)) * self.stop_token).to(dtype=input_ids.dtype, device=input_ids.device)
        
        placeholder_action_token_ids = (
            (torch.ones((input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK)) * self.action_start_token_id).to(input_ids.device).to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids[:, :actual_input_len], placeholder_action_token_ids, pad_ids], dim=-1)

        # Extend the attention mask to fit the new shape of input
        # Note: Only batch size == 1 supported right now
        mask_extension = (
            torch.ones((attention_mask.shape[0], pad_len - actual_input_len))
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        mask_extension[:, -tail_len:] = 0
        attention_mask = torch.cat([attention_mask[:, :actual_input_len], mask_extension], dim=-1)
        action_mask = torch.zeros_like(attention_mask).to(attention_mask.dtype).to(attention_mask.device)
        action_mask[:, actual_input_len:(actual_input_len + ACTION_DIM)] = 1
        return input_ids, attention_mask, action_mask.bool()