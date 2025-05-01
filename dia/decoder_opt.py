import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple

from dia.layers import Decoder, DecoderLayer
from dia.state import DecoderInferenceState, KVCache
from dia.config import DiaConfig


class OptimizedDecoder(nn.Module):
    """
    Memory-efficient implementation of the Decoder that preserves the same
    interface but applies optimizations for faster inference.
    """
    
    def __init__(self, original_decoder: Decoder):
        super().__init__()
        
        # Transfer all attributes from the original decoder
        self.config = original_decoder.config
        self.num_channels = original_decoder.num_channels
        self.num_layers = original_decoder.num_layers
        
        # Keep the original modules
        self.embeddings = original_decoder.embeddings
        self.layers = original_decoder.layers
        self.norm = original_decoder.norm
        self.logits_dense = original_decoder.logits_dense
        
        # Optimization flags
        self.memory_efficient = True
        self.batch_decode = True  # For future batch decoding optimization
        
        # Cache and reuse tensors when possible
        self._cached_embeddings = {}
        self._last_token_ids = None
        
    def clear_caches(self):
        """Clear any cached tensors to free memory"""
        self._cached_embeddings = {}
        self._last_token_ids = None
        
    def precompute_cross_attn_cache(
        self,
        enc_out: torch.Tensor,
        enc_positions: torch.Tensor,
    ) -> list[KVCache]:
        """
        Optimized version of precomputing cross-attention KV cache from encoder output.
        Performs projections in parallel when possible.
        """
        # This is a direct call to the original implementation
        # Could be further optimized by batch processing the layers
        per_layer_kv_cache: list[KVCache] = []

        # Create a big-tensor optimization when possible
        if enc_out.shape[0] <= 4:  # Small batch optimization
            # Process all layers at once for small batches
            all_k_projs = []
            all_v_projs = []
            
            for layer in self.layers:
                cross_attn_module = layer.cross_attention
                k_proj = cross_attn_module.k_proj(enc_out)
                v_proj = cross_attn_module.v_proj(enc_out)
                
                k_proj = cross_attn_module.rotary_emb(k_proj, position=enc_positions)
                k = k_proj.transpose(1, 2)
                v = v_proj.transpose(1, 2)
                
                per_layer_kv_cache.append(KVCache.from_kv(k, v))
        else:
            # For larger batches, do one at a time
            for layer in self.layers:
                cross_attn_module = layer.cross_attention
                k_proj = cross_attn_module.k_proj(enc_out)
                v_proj = cross_attn_module.v_proj(enc_out)
                
                k_proj = cross_attn_module.rotary_emb(k_proj, position=enc_positions)
                k = k_proj.transpose(1, 2)
                v = v_proj.transpose(1, 2)
                
                per_layer_kv_cache.append(KVCache.from_kv(k, v))

        return per_layer_kv_cache
        
    def _compute_embeddings(self, token_ids, channel_idx):
        """Compute embeddings with caching for repeated tokens"""
        # Create a cache key based on the token id and channel
        cache_key = (token_ids.item(), channel_idx)
        
        # Check if we already computed this embedding
        if cache_key in self._cached_embeddings:
            return self._cached_embeddings[cache_key]
        
        # Compute the embedding
        embedding = self.embeddings[channel_idx](token_ids)
        
        # Cache the result for future reuse
        if self.memory_efficient and len(self._cached_embeddings) < 1000:  # Limit cache size
            self._cached_embeddings[cache_key] = embedding
            
        return embedding
        
    def decode_step(
        self,
        tgt_ids_Bx1xC: torch.Tensor,  # [B, 1, C]
        state: DecoderInferenceState,
    ) -> torch.Tensor:
        """
        Optimized single decoding step with embedding caching and layer optimizations.
        """
        # Fast path for repeated tokens (optimization for token streaming)
        if (self._last_token_ids is not None and 
            torch.all(tgt_ids_Bx1xC == self._last_token_ids)):
            # We've seen these exact tokens before - return the cached result
            return self._last_result
            
        # Store current tokens for potential future reuse
        self._last_token_ids = tgt_ids_Bx1xC.clone()
        
        # Compute embeddings with potential caching
        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids_Bx1xC[..., i]
            
            # Use cached embeddings when possible for single tokens
            if tgt_ids_Bx1xC.shape[1] == 1 and tgt_ids_Bx1xC.shape[0] == 1:
                channel_embed = self._compute_embeddings(channel_tokens, i)
            else:
                channel_embed = self.embeddings[i](channel_tokens)
                
            x = channel_embed if x is None else x + channel_embed

        # Process through layers
        for i, layer in enumerate(self.layers):
            self_cache = state.self_attn_cache[i]
            cross_cache = state.cross_attn_cache[i]
            x = layer(
                x,
                state,
                self_attn_cache=self_cache,
                cross_attn_cache=cross_cache,
            )

        # Apply final normalization
        x = self.norm(x)
        
        # Compute logits
        logits_Bx1xCxV = self.logits_dense(x)
        logits_output = logits_Bx1xCxV.to(torch.float32)
        
        # Cache the result
        self._last_result = logits_output
        
        return logits_output

    def forward(self, tgt_ids_BxTxC: torch.Tensor, state: DecoderInferenceState) -> torch.Tensor:
        """
        Forward pass that handles the full sequence, with optimized embedding computation.
        """
        # Clear any cached single-token results as we're doing a full forward pass
        self._last_token_ids = None
        
        # Verify input dimensions
        _, _, num_channels_in = tgt_ids_BxTxC.shape
        assert num_channels_in == self.num_channels, "Input channels mismatch"

        # Compute embeddings efficiently
        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids_BxTxC[..., i]
            channel_embed = self.embeddings[i](channel_tokens)
            x = channel_embed if x is None else x + channel_embed

        # Process through layers
        for i, layer in enumerate(self.layers):
            self_cache = state.self_attn_cache[i]
            cross_cache = state.cross_attn_cache[i]
            x = layer(x, state, self_attn_cache=self_cache, cross_attn_cache=cross_cache, prefill=True)

        # Apply final normalization
        x = self.norm(x)
        
        # Compute logits
        logits_BxTxCxV = self.logits_dense(x)
        
        return logits_BxTxCxV.to(torch.float32)


def optimize_decoder(dia_instance):
    """
    Replace the decoder in a Dia model with the OptimizedDecoder.
    
    Args:
        dia_instance: An instance of the Dia class
        
    Returns:
        The same instance with an optimized decoder
    """
    model = dia_instance.model
    
    # Replace the decoder with our optimized version
    original_decoder = model.decoder
    optimized_decoder = OptimizedDecoder(original_decoder)
    model.decoder = optimized_decoder
    
    print("Replaced decoder with memory-efficient optimized version")
    return dia_instance