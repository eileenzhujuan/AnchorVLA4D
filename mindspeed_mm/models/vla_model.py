from typing import Optional, Dict, Union
import torch
from torch import nn
from mindspeed_mm.models.vlm_model import VLMModel

from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from mindspeed_mm.models.action_head import DiffusionActionHead


class VLAModel(nn.Module):

    def __init__(self, vlm_config, action_head_config=None):
        super().__init__()
        self.config = core_transformer_config_from_args(get_args())
        self.vlm_model = VLMModel(vlm_config)
        self.vlm_model.text_decoder.post_process = False
        self.stop_token = vlm_config.stop_token
        self.action_start_token_id = vlm_config.action_start_token_id
        if action_head_config is not None:
            self.action_normalization = action_head_config.pop('action_normalization')
            self.action_head = DiffusionActionHead(**action_head_config)
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
        if ground_truth_actions is not None:
            assert self.action_normalization['done_normalized'] # do it before training
            noisy_dict = self.action_head.sample_noisy_actions(
                ground_truth_actions, sample_method='normal')
            noise, noisy_actions, _ = (
                noisy_dict["noise"],
                noisy_dict["noisy_actions"],
                noisy_dict["diffusion_timestep_embeddings"],
            )
        else:
            assert noisy_actions is not None # inference mode

        # shape convert to: (action_dim * action_chunk, bsz)
        bsz = input_ids.shape[0]
        noisy_actions = noisy_actions.view(bsz, -1).transpose(0, 1)
        input_embeds = self.vlm_model.text_decoder.embedding(input_ids=input_ids, position_ids=position_ids).clone()

        action_embeds = self.action_head.action_projector(noisy_actions.unsqueeze(-1))
        action_mask = (input_ids >= self.action_start_token_id).to(device=input_ids.device).transpose(0, 1)
        input_embeds.masked_scatter_(action_mask.unsqueeze(-1).expand_as(input_embeds), action_embeds)

        vlm_output = self.vlm_model(input_ids, input_embeds, pixel_values, image_grid_thw,
                                    attention_mask, labels, inference_params,
                                    decoder_input, position_ids,
                                    packed_seq_params, extra_block_kwargs,
                                    cache_position, rope_deltas, image_flags,
                                    *args, **kwargs)
        action_hidden_states = vlm_output[action_mask].view(bsz, -1)
        noise_pred = self.action_head.predict_noise(action_hidden_states)
        # Get diffusion noise prediction MSE loss
        noise_pred = noise_pred.reshape(noise.shape)

        diffusion_loss = nn.functional.mse_loss(noise_pred,
                                                noise,
                                                reduction="mean")

        return {'loss': diffusion_loss}


    def set_input_tensor(self, input_tensor):
        self.vlm_model.set_input_tensor(input_tensor)

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        state_dict = {'vlm': self.vlm_model.state_dict_for_save_checkpoint(prefix, keep_vars)}
        action_head_state = self.action_head.state_dict(prefix=prefix, keep_vars=keep_vars)
        state_dict['action_head'] = {k: v.clone() for k, v in action_head_state.items()}
        return state_dict


    def predict_action(self, inputs, NUM_ACTIONS_CHUNK=1):
        self.action_head.noise_scheduler.set_timesteps(num_inference_steps=1)
        input_ids = inputs.pop("input_ids")
        NUM_PROMPT_TOKENS = input_ids.shape[-1]
        attention_mask = inputs['attention_mask'].bool()
        input_ids, inputs['attention_mask'], action_mask = self.prepare_input_for_action_prediction(
            input_ids, attention_mask, NUM_ACTIONS_CHUNK, self.action_head.action_dim)
        input_embeddings=self.vlm_model.get_input_embeddings()(input_ids)
        inputs["inputs_embeds"] = input_embeddings
        inputs['output_hidden_states'] = True
        # Sample random noise with shape equal to output action, used as the starting state for reverse diffusion
        curr_noisy_actions = torch.randn(
            size=(1, NUM_ACTIONS_CHUNK, self.action_head.action_dim), device=input_embeddings.device, dtype=input_embeddings.dtype
        )
        print(f'{curr_noisy_actions.shape = }')
        # action_projection = ProprioProjector(2048, 1).to(dtype=input_embeddings.dtype, device=input_embeddings.device)
        # Reverse diffusion: Iteratively denoise to generate action prediction
        for t in self.action_head.noise_scheduler.timesteps:
            with torch.no_grad():
                # noisy_embeds = action_projection(curr_noisy_actions.transpose(1, 2))
                # inputs['inputs_embeds'] = replace_input_embeddings(input_embeddings, action_mask, noisy_embeds)
                last_hidden_states = self.vlm_model.forward(**inputs)['hidden_states'][-1]
            # Extract hidden states for action portion of response
            actions_hidden_states = last_hidden_states[
                :,
                NUM_PROMPT_TOKENS : NUM_PROMPT_TOKENS + self.action_head.action_dim * NUM_ACTIONS_CHUNK,
                :,
            ]  # (B, act_chunk_len, D)
            # Predict noise and update noisy actions: x_t -> x_{t-1}
            noise_pred = self.action_head.predict_noise(actions_hidden_states, NUM_ACTIONS_CHUNK)
            curr_noisy_actions = self.action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

        curr_noisy_actions = curr_noisy_actions.reshape(NUM_ACTIONS_CHUNK, self.action_head.action_dim)

        # Return final actions
        return curr_noisy_actions.float().cpu().detach().numpy()#, actions_hidden_states
    
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


def unormalize_action(action, scale, low, high):
    action = np.clip(action, -scale, scale)
    action = ((action / scale + 1) / 2 * (high - low + 1e-8) + low)
    action[6] = np.where(action[6] < 0.5, 0, 1)
    return action

def normalize_action(action, scale, low, high):
    action = (2 * (action - low) / (high - low + 1e-8) - 1) * scale
    action = np.clip(action, -scale, scale)
    return action

def prepare_labels_for_action_prediction(labels, input_ids, NUM_ACTIONS_CHUNK=1, ACTION_DIM=7, ACTION_TOKEN_BEGIN_IDX=151679, STOP_INDEX=151645):
    """Creates labels tensor for action prediction if not provided"""
    # Extend labels tensor with fake action labels
    ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
    labels_extension = (
        torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
        * ARBITRARY_ACTION_TOKEN_IDX
    )
    labels = torch.cat([labels, labels_extension], dim=-1)

    # Replace last label token with stop token
    labels[:, -1] = STOP_INDEX

    return labels


def replace_input_embeddings(input_embeddings, all_actions_mask, noisy_action_features):
    """
    Replace embeddings in input_embeddings at positions where all_actions_mask is True
    with embeddings from noisy_action_features, using vectorized operations.

    Args:
        input_embeddings: Tensor of shape (B, S, D)
        all_actions_mask: Boolean tensor of shape (B, S)
        noisy_action_features: Tensor of shape (B, K, D) where K is the number of True values in mask per sample

    Returns:
        Modified input_embeddings tensor
    """
    # Clone input to avoid modifying the original tensor
    new_input_embeddings = input_embeddings.clone()

    # Create a tensor with the same shape of input_embeddings to hold the noisy action features
    repositioned_noisy_action_features = torch.zeros_like(input_embeddings)

    # Create batch indices for splicing
    batch_indices = torch.arange(input_embeddings.shape[0], device=input_embeddings.device)
    batch_indices = batch_indices.unsqueeze(1).expand(-1, noisy_action_features.shape[1])

    # Get indices where mask is True for each sample
    masked_indices = torch.stack([torch.where(mask)[0] for mask in all_actions_mask])

    # Move the noisy action features into their correct positions
    repositioned_noisy_action_features[batch_indices, masked_indices] = noisy_action_features

    # Combine original input embeddings and noisy action embeddings using the mask
    new_input_embeddings = torch.where(
        all_actions_mask.unsqueeze(-1), repositioned_noisy_action_features, new_input_embeddings
    )

    return new_input_embeddings
