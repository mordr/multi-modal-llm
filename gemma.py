import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from siglip import SiglipVisionConfig, SiglipVisionModel


class GemmaConfig:
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class PaliGemmaConfig:
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(
            **text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (
            self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim


class KVCache():
    pass

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size, config.projection_dim, bias=True)

    def forward(self, image_features: torch.Tensor):
        # (batch_size, num_patches, embed_dim) -> (batch_size, num_patches, projection_dim)
        return self.linear(image_features)


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        pass


class GemmaRMSNorm(nn.Module):
    pass


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.paddixng_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.paddixng_idx)
        self.layers = nn.ModuleList([GemmaDecoderLayer(
            config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None
    ) -> torch.FloatTensor:
        # (batch_size, seq_len, hidden_size)
        hidden_states = input_embeds
        # (batch_size, seq_len, hidden_size)
        normalizer = torch.tensor(
            self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            # (batch_size, seq_len, hidden_size)
            hidden_states = decoder_layer(
                hidden_states, attention_mask, position_ids, kv_cache)

        # (batch_size, seq_len, hidden_size)
        hidden_states = self.norm(hidden_states)

        # (batch_size, seq_len, hidden_size)
        return hidden_states


class GemmaForCausalLM(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None
    ) -> Tuple:

        # input_embeds: (batch_size, seq_len, hidden_size)
        # outputs: (batch_size, seq_len, hidden_size)
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        output_data = {
            'logits': logits
        }

        if kv_cache is not None:
            output_data['kv_cache'] = kv_cache

        return output_data

class KVCache(nn.Module):
    pass


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(
            config.multi_modal_projector_config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.language_model_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(self,
                                             image_features: torch.Tensor,
                                             input_embeds: torch.Tensor,
                                             input_ids: torch.Tensor,
                                             attention_mask: torch.Tensor,
                                             kv_cache: Optional[KVCache] = None):
        _, _, embed_dim = image_features.shape
        batch_size, seq_len = input_ids.shape
        dtype, device = input_embeds.dtype, input_embeds.device

        # (batch_size, seq_len, hidden_dim)
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        # Combine embeddings of image tokens, text tokens, and mask out all padding tokens
        final_embedding = torch.zeros(batch_size,
                                      seq_len,
                                      embed_dim,
                                      dtype=dtype,
                                      device=device)
        # (batch_size, seq_len)
        text_mask = (input_ids != self.config.image_token_index) and (
            input_ids != self.pad_token_id)
        # (batch_size, seq_len)
        image_mask = (input_ids == self.config.image_token_index)
        # (batch_size, seq_len)
        pad_mask = (input_ids == self.pad_token_id)

        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(
            -1).expand(-1, -1, embed_dim)

        # Add text embeddings
        final_embedding = torch.where(
            text_mask_expanded, input_embeds, final_embedding)
        # Add image embeddings
        final_embedding = final_embedding.masked_scatter(
            image_mask_expanded, scaled_image_features)
        # Add padding mask
        final_embedding = final_embedding.masked_fill(
            pad_mask_expanded, 0.0)

        # KV cache - pre-filling and generation
        dtype, device, input_embeds.dtype, input_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = input_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # prefill stage
            # no masking, paliGemma design
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device)
        else:
            # generation stage
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also do not mask anything, paliGemma design
            # 1 x kv_len
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device)

        # Add head dimension
        # (batch_size, q_len, kv_len) -> (batch_size, num_heads_q, q_len, kv_len)
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # position of query is last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # create position ids based on size of the attention mask
            # for masked tokens, use 1 as position
            position_ids = (attention_mask.cumsum(-1)
                            ).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids


    def forward(self,
                input_ids: torch.LongTensor = None,
                pixel_values: torch.FloatTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None
                ) -> Tuple:
        assert torch.all(attention_mask == 1), 'Input cannot be padded'

        # Extract input embeddings
        # (batch_size, seq_len, hidden_size)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Merge text and images
        # (batch_size, channel, height, width) -> (batch_size, num_patches, embed_dim)
        selected_image_feature = self.vision_tower(
            pixel_values.to(input_embeds.dtype))

        # (batch_size, num_patches, embed_dim) -> (batch_size, num_patches, hidden_size)
        image_features = self.multi_modal_projector(selected_image_feature)

        # Merge embeddings of text and image tokens
        input_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features, input_embeds, input_ids, attention_mask, kv_cache)
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache
        )

        return outputs
