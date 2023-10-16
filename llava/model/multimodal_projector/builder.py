import torch
import torch.nn as nn
import re
from transformers import Blip2QFormerModel, Blip2Config


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class QFormerProj(nn.Module):
    def __init__(self, mm_hidden_size, hidden_size, config=Blip2Config):
        super().__init__()
        config.qformer_config.hidden_size = mm_hidden_size
        config.text_config.hidden_size = hidden_size
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        self.post_init()

    def forward(self, image_features):
        query_tokens = self.query_tokens.expand(image_features.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_features,
            # encoder_attention_mask=image_attention_mask,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        return language_model_inputs


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == 'q_former':
        q_former_proj = QFormerProj(config.mm_hidden_size, config.hidden_size)
        return q_former_proj
    
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
