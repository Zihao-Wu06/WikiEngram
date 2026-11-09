# Reference copy of the PLE (n-gram memory) pieces of transformers
# src/transformers/models/qwen4_exp/modeling_qwen4_exp.py (main @ b25568817a6cb458f6d8b180a32698c2ab11fd26),
# used only by tests/test_variant_parity.py. The classes and functions below are copied verbatim; the only
# change is this header, the import block and a `Cache` placeholder (no cache object is passed in the tests).
#
# Copyright 2026 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn.functional as F
from torch import nn

Cache = object  # placeholder for transformers.cache_utils.Cache (type annotations only)
Qwen4ExpTextConfig = object  # placeholder for the config class (type annotations only)


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(unigram_vocab_size, ngram_size, ple_layer_index, seed: int) -> torch.Tensor:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(multipliers, dtype=torch.long)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


class Qwen4ExpTextRMSNorm(nn.Module):
    def __init__(self, dim: int, group_size: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        self.group_size = group_size
        if group_size is not None and dim % group_size != 0:
            raise ValueError(f"hidden_size ({dim}) must be divisible by group_size ({group_size}).")

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.group_size is not None:
            x = x.reshape(*x.shape[:-1], -1, self.group_size)
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.flatten(-2) if self.group_size is not None else out

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Qwen4ExpText is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


# NOTE: the FLA package does not re-cast to `input_dtype` in its implementation, maybe we should do the same


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    # NOTE: attention mask is a 2D boolean tensor
    if attention_mask is not None:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


class Qwen4ExpTextNGramEmbedding(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig, embedding_dim: int, layer_idx: int, ple_layer_index: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.ngram_size = config.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = ple_layer_index
        self.unigram_vocab_size = config.vocab_size
        self.ngram_vocab_size_base = config.ngram_vocab_size_base
        head_dim_per_ngram = embedding_dim // self.ngram_heads
        self.seed = config.seed
        # CODEPATH: @ArthurZucker fix flagging for no reason here
        self.eos_token_id = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id

        self.head_vocab_sizes = []
        self.head_offsets = []
        self.total_vocab_size = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = _find_nth_prime_after(self.ngram_vocab_size_base - 1, global_head_idx + 1)
            self.head_vocab_sizes.append(size)
            self.head_offsets.append(self.total_vocab_size)
            self.total_vocab_size += size

        self.layer_multipliers = nn.Buffer(
            _build_layer_multipliers(self.unigram_vocab_size, self.ngram_size, self.ple_layer_index, self.seed)
        )
        self.ngram_heads_vocab_sizes = nn.Buffer(torch.tensor(self.head_vocab_sizes, dtype=torch.long))
        self.ngram_heads_offsets = nn.Buffer(torch.tensor(self.head_offsets, dtype=torch.long))
        ngram_vocab_divisor = config.make_ngram_vocab_size_divisible_by
        padded_vocab_size = math.ceil(self.total_vocab_size / ngram_vocab_divisor) * ngram_vocab_divisor
        self.ngram_embedding = nn.Embedding(padded_vocab_size, head_dim_per_ngram)

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat([eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1)
        segment_start = previous_eos + 1
        position_in_segment = positions.unsqueeze(0) - segment_start
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def forward(self, input_ids: torch.Tensor, past_key_values: Cache | None) -> torch.Tensor:
        input_ids = input_ids.long()
        # This is a trick to store the previous N=self.context_len `input_ids` - indeed the manipulations are identical to storing
        # a past conv_state, so we can use an additional conv_states inside the Cache for it
        cached_ctx = (
            past_key_values.get_conv_state(self.layer_idx, state_idx=2) if past_key_values is not None else None
        )

        if cached_ctx is not None:
            previous_context = cached_ctx.clone()
        else:
            previous_context = input_ids.new_full((input_ids.shape[0], self.context_len), self.eos_token_id)

        # Store the current input_ids for the next forward
        if past_key_values is not None:
            input_ids_to_cache = input_ids
            # In the case where `input_ids` would be smaller than `self.context_len`, the `update_conv_state` will pad with zeros, whereas
            # here we want to pad with eos, so we do it explicitly
            if cached_ctx is None and input_ids.shape[1] < self.context_len:
                input_ids_to_cache = torch.nn.functional.pad(
                    input_ids_to_cache, (self.context_len - input_ids.shape[1], 0), value=self.eos_token_id
                )
            _ = past_key_values.update_conv_state(
                input_ids_to_cache, self.layer_idx, state_idx=2, conv_kernel_size=self.context_len
            )

        # Get full token history
        token_history = torch.cat([previous_context, input_ids], dim=-1)
        shifted_tokens = [self._shift_right_ignore_eos(token_history, shift) for shift in range(self.ngram_size)]

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start_idx = (ngram - 2) * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(
                    mixed_ids,
                    shifted_tokens[position] * self.layer_multipliers[position],
                )
            head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
            head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
            ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
            blocks.append(ngram_ids + head_offsets.view(1, 1, -1))

        ngram_ids = torch.cat(blocks, dim=-1)[:, -input_ids.shape[1] :]
        # We need explicit device placement here, as the embedding may be skipped from device_map completely (we just need to be
        # careful in the case of offloading to disk)
        execution_device = (
            self.ngram_embedding.weight.device if self.ngram_embedding.weight.device.type != "meta" else None
        )
        return self.ngram_embedding(ngram_ids.to(execution_device)).to(ngram_ids.device).flatten(-2)


class Qwen4ExpTextPLELayer(nn.Module):
    """Inject hashed n-gram features into every hyper-connection stream.

    PLE projects each token's concatenated n-gram embedding to a shared value and one key per residual stream. The
    normalized stream activations gate those values, then a dilated depthwise convolution adds local lexical context.
    The returned tensor has shape `(batch_size, sequence_length, hc_count * hidden_size)`.
    """

    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int, ple_layer_index: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        ple_embed_dim = config.ple_embed_dim
        hc_hidden_size = self.hidden_size * self.hc_count
        self.ple_embedding = Qwen4ExpTextNGramEmbedding(config, ple_embed_dim, layer_idx, ple_layer_index)
        conv_kernel_size = config.ple_conv_kernel_size
        conv_dilation = config.ngram_size
        self.short_conv_state_len = (conv_kernel_size - 1) * conv_dilation
        self.key_proj = nn.Linear(ple_embed_dim, hc_hidden_size, bias=False)
        self.value_proj = nn.Linear(ple_embed_dim, self.hidden_size, bias=False)
        self.norm_key = Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.norm_query = Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.norm_conv = Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.conv1d = nn.Conv1d(
            hc_hidden_size,
            hc_hidden_size,
            kernel_size=conv_kernel_size,
            groups=hc_hidden_size,
            dilation=conv_dilation,
            bias=False,
        )

    def _short_conv(self, hidden_states: torch.Tensor, past_key_values: Cache | None) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states.transpose(1, 2)

        if past_key_values is not None:
            hidden_states = past_key_values.update_conv_state(
                hidden_states, self.layer_idx, state_idx=1, conv_kernel_size=self.short_conv_state_len
            )

        # We always pad and slice due to the dilation in the conv, to make sure we have enough states
        hidden_states = F.pad(hidden_states, (self.short_conv_state_len, 0))
        hidden_states = hidden_states[..., -(self.short_conv_state_len + seq_len) :]

        # We cannot use the usual functions/kernels here for the short conv as the conv1d has dilation
        hidden_states = F.silu(self.conv1d(hidden_states))

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        past_key_values: Cache | None,
        conv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.ple_embedding(input_ids, past_key_values)
        key_normed = self.norm_key(self.key_proj(embeddings)).unflatten(-1, (self.hc_count, self.hidden_size))
        value = self.value_proj(embeddings)
        query_normed = self.norm_query(hidden_states).unflatten(-1, (self.hc_count, self.hidden_size))
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_normed = self.norm_conv(gated_value.flatten(-2))
        gated_value = gated_value.flatten(-2)
        if conv_mask is not None:
            gated_value = apply_mask_to_padding_states(gated_value, conv_mask)
            gated_value_normed = apply_mask_to_padding_states(gated_value_normed, conv_mask)
        output = gated_value + self._short_conv(gated_value_normed, past_key_values)
        return output
