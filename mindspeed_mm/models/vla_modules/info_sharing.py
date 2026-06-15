import torch
from torch import nn
from uniception.models.info_sharing.alternating_attention_transformer import MultiViewAlternatingAttentionTransformerIFR
from uniception.models.info_sharing.base import MultiViewTransformerInput

class InfoSharing(nn.Module):

    def __init__(self, vit_embed_dim=1024, input_embed_dim=1024, output_dim=768):
        super().__init__()
        self.input_embed_dim = input_embed_dim
        self.scale_token = nn.Parameter(torch.zeros(vit_embed_dim))
        torch.nn.init.trunc_normal_(self.scale_token, std=0.02)
        self.info_sharing = MultiViewAlternatingAttentionTransformerIFR(
                input_embed_dim=input_embed_dim,
                name='aat_24_layers_ifr',
                indices=[11, 17],
                norm_intermediate=True,
                size='24_layers',
                depth=24,
                distinguish_ref_and_non_ref_views=True,
                gradient_checkpointing=False)
        self.final_projection = nn.Linear(self.info_sharing.dim, output_dim)


    def forward(self, all_encoder_features_across_views):
        batch_size_per_view = all_encoder_features_across_views[0].shape[0]
        # Expand the scale token to match the batch size
        input_scale_token = (
            self.scale_token.unsqueeze(0)
            .unsqueeze(-1)
            .repeat(batch_size_per_view, 1, 1)
        )  # (B, C, 1)
        # Combine all images into view-centric representation
        # Output is a list containing the encoded features for all N views after information sharing.
        info_sharing_input = MultiViewTransformerInput(
            features=all_encoder_features_across_views,
            additional_input_tokens=input_scale_token,
        )
        final_info, intermediate_info = self.info_sharing(info_sharing_input)
        stacked_final_features = torch.cat(
            final_info.features, dim=0
        )
        stacked_intermediate_features_0 = torch.cat(
            intermediate_info[0].features, dim=0
        )
        stacked_intermediate_features_1 = torch.cat(
            intermediate_info[1].features, dim=0
        )
        all_features = [stacked_final_features, stacked_intermediate_features_0, stacked_intermediate_features_1]
        all_features = torch.cat(all_features, dim=0)
        all_features = self.final_projection(all_features.permute(0,2,3,1).view(-1, batch_size_per_view, self.info_sharing.dim))
        return all_features


    def load_state_dict(self, state_dict, *args, **kwargs):
        def skip_key(keys_name, param_shapes):
            for key_name, param_shape in zip(keys_name, param_shapes):
                if key_name in state_dict.keys():
                    state_dict_shape = state_dict[key_name].shape
                    if state_dict_shape != param_shape:
                        state_dict.pop(key_name)
                        # logger.info(f'state_dict of {key_name} has a mismatch shape being {state_dict_shape},'
                        print(f'state_dict of {key_name} has a mismatch shape being {state_dict_shape},'
                                    f'different from current weight shape: {param_shape}, skipped loading')
        strict = skip_key(['scale_token', 'info_sharing.proj_embed.weight', 'final_projection.weight',
                          ], [
                              self.scale_token.shape, self.info_sharing.proj_embed.weight.shape,
                              self.final_projection.weight.shape,
                          ])
        kwargs.update({"strict": strict})
        return super().load_state_dict(state_dict, *args, **kwargs)

