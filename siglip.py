from typing import Optional, Tuple
import torch
import torch.nn as nn

'''
Basically ViT - vision transformer
'''


class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding='valid'  # no padding added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(
            num_embeddings=self.num_positions, embedding_dim=self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # (batch_size, channels, height, width)
        _, _, height, width = pixel_values.shape
        # (batch_size, embed_dim, num_patches_h, num_patches_w)
        # num_patches_h = height // patchsize, num_patches_w = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)
        # (batch_size, embed_dim, num_patches_h, num_patches_w) -> (batch_size, embed_dim, num_patches)
        # num_patches = num_patches_h * num_patches_w
        embeddings = patch_embeds.flatten(2)
        # (batch_size, embed_dim, num_patches) -> (batch_size, num_patches, embed_dim)
        embeddings = embeddings.transpose(-1, -2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # (batch_size, num_patches, embed_dim)
        return embeddings


class SiglipAttention(nn.Module):
    '''
    Multi-head attention
    '''

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5  # 1 / sqrt(head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # W_o matrix in original attention paper
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # (batch_size, num_patches, embed_dim)
        batch_size, seq_len, _ = hidden_states.size()

        # (batch_size, num_patches, embed_dim) -> (batch_size, num_patches, embed_dim)
        query_states = self.q_proj(hidden_states)
        # (batch_size, num_patches, embed_dim) -> (batch_size, num_patches, embed_dim)
        key_states = self.k_proj(hidden_states)
        # (batch_size, num_patches, embed_dim) -> (batch_size, num_patches, embed_dim)
        value_states = self.v_proj(hidden_states)

        # For all
        # (batch_size, num_patches, embed_dim) -> (batch_size, num_patches, num_heads, head_dim)
        # -> (batch_size, num_heads, num_patches, head_dim) << each head 'matrix' can now be treated independently/parallel
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Q * K^T / sqrt(d_k)
        # (batch_size, num_heads, num_patches, head_dim)
        attn_weights = (torch.matmul(
            query_states, key_states.transpose(2, 3)) * self.scale)
        if (attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len)):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is {attn_weights.size()}")
        attn_weights = attn_weights.softmax(
            dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, self.dropout)  # training is default true, ignore for now
        # (batch_size, num_heads, num_patches, head_dim)
        attn_output = torch.matmul(attn_weights, value_states)

        # (batch_size, num_heads, num_patches, head_dim) -> (batch_size, num_patches, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # (batch_size, num_patches, num_heads, head_dim) -> (batch_size, num_patches, embed_dim)
        attn_output = attn_output.reshape(
            batch_size, seq_len, self.embed_dim)
        # Mixes the concat head outputs via W_o
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(in_features=config.hidden_size,
                             out_features=config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # (batch_size, num_patches, embed_dim) -> (batch_size, num_patches, intermediate_size)
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')
        # (batch_size, num_patches, intermediate_size) -> (batch_size, num_patches, embed_dim)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(
            self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(
            self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # residual: (batch_size, num_patches, embed_dim)
        residual1 = hidden_states
        # (batch_size, num_patches, embed_dim) -> (batch_size, num_patches, embed_dim)
        hidden_states = self.layer_norm1(hidden_states)
        # (batch_size, num_patches, embed_dim) -> (batch_size, num_patches, embed_dim)
        # self.attn returns attn_output, attn_weights tuple
        hidden_states, _ = self.self_attn(hidden_states)
        # (batch_size, num_patches, embed_dim) + (batch_size, num_patches, embed_dim) -> (batch_size, num_patches, embed_dim)
        hidden_states = hidden_states + residual1

        # residual: (batch_size, num_patches, embed_dim)
        residual2 = hidden_states
        # (batch_size, num_patches, embed_dim) -> (batch_size, num_patches, embed_dim)
        hidden_states = self.layer_norm2(hidden_states)
        # (batch_size, num_patches, embed_dim) -> (batch_size, num_patches, embed_dim)
        hidden_states = self.mlp(hidden_states)
        # (batch_size, num_patches, embed_dim) + (batch_size, num_patches, embed_dim) -> (batch_size, num_patches, embed_dim)
        hidden_states = hidden_states + residual2
        # residual: (batch_size, num_patches, embed_dim)
        return hidden_states


class SiglipEncoder(nn.Module):
    '''
    Contextualized patch embeddings
    '''

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        # (batch_size, num_patches, embed_dim) -> (batch_size, num_patches, embed_dim)
        hidden_states = input_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layer_norm = nn.LayerNorm(
            embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> Tuple:
        # (batch_size, channels, height, width) -> (batch_size, num_patches, embed_dim)
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(input_embeds=hidden_states)
        last_hidden_state = self.post_layer_norm(last_hidden_state)
        return last_hidden_state


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> Tuple:
        # (batch_size, channels, height, width) -> (batch_size, num_patches, embed_dim)
        return self.vision_model(pixel_values=pixel_values)
