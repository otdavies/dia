import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from dia.layers import Attention
from dia.state import KVCache


class OptimizedAttention(nn.Module):
    """
    Optimized Attention module that maintains the same interface as the original
    Attention module but uses more efficient computation patterns.
    """
    
    def __init__(self, original_attention: Attention):
        super().__init__()
        
        # Transfer over all attributes from the original attention module
        self.num_query_heads = original_attention.num_query_heads
        self.num_kv_heads = original_attention.num_kv_heads
        self.head_dim = original_attention.head_dim
        self.is_cross_attn = original_attention.is_cross_attn
        self.output_dim = original_attention.output_dim
        self.projected_query_dim = original_attention.projected_query_dim
        self.num_gqa_groups = original_attention.num_gqa_groups
        
        # Store the original layers (will be replaced with optimized versions later)
        self.q_proj = original_attention.q_proj
        self.k_proj = original_attention.k_proj
        self.v_proj = original_attention.v_proj
        self.o_proj = original_attention.o_proj
        self.rotary_emb = original_attention.rotary_emb
        
        # Optimization flags
        self.use_flash_attn = hasattr(F, 'scaled_dot_product_attention')
        self.optimize_kv_cache_access = True
        
        # Cache for faster inference
        self.cached_causal_mask = None
        self.last_seq_len = None
        
    def _get_or_create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create or retrieve a cached causal mask for faster inference"""
        if self.cached_causal_mask is None or self.last_seq_len < seq_len:
            # Create a new mask
            mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool).triu(1)
            self.cached_causal_mask = mask
            self.last_seq_len = seq_len
            return mask
        else:
            # Return the cached mask (properly sliced if needed)
            return self.cached_causal_mask[:seq_len, :seq_len]
            
    def forward(
        self,
        Xq: torch.Tensor,  # (B, T, D) T = 1 in AR generation
        Xkv: torch.Tensor,  # (B, S, E) S = 1 in AR generation
        q_positions: torch.Tensor,  # (B, T)
        kv_positions: torch.Tensor | None = None,  # (B, S)
        attn_mask: torch.Tensor | None = None,  # None in Decoder Self Attention, Valid mask in Others
        cache: KVCache | None = None,  # None in Encoder, KVCache in Decoder
        prefill: bool = False,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Optimized attention calculation with optional KV caching.
        """
        if kv_positions is None:
            kv_positions = q_positions
        
        # Save original dtype for output consistency
        original_dtype = Xq.dtype
        
        # Fast projection operations
        Xq_BxTxNxH = self.q_proj(Xq)
        Xq_BxTxNxH = self.rotary_emb(Xq_BxTxNxH, position=q_positions)
        
        # More efficient transpose for attention
        # For small batch sizes, contiguous is not necessary
        Xq_BxNxTxH = Xq_BxTxNxH.transpose(1, 2)
        
        # Handle key and value with optimized path
        attn_k: torch.Tensor | None = None
        attn_v: torch.Tensor | None = None

        if self.is_cross_attn:
            # For cross attention, keys and values are pre-computed and stored in the cache
            attn_k, attn_v = cache.k, cache.v
        else:
            # For self-attention, compute keys and values
            Xk_BxSxKxH = self.k_proj(Xkv)
            Xv_BxSxKxH = self.v_proj(Xkv)
            Xk_BxSxKxH = self.rotary_emb(Xk_BxSxKxH, position=kv_positions)
            
            # Optimize transpose operations
            Xk_BxKxSxH = Xk_BxSxKxH.transpose(1, 2)
            Xv_BxKxSxH = Xv_BxSxKxH.transpose(1, 2)

            # Handle caching
            if cache is None:
                attn_k, attn_v = Xk_BxKxSxH, Xv_BxKxSxH
            else:
                if prefill:
                    attn_k, attn_v = Xk_BxKxSxH, Xv_BxKxSxH
                    cache.prefill(attn_k, attn_v)
                else:
                    attn_k, attn_v = cache.update(Xk_BxKxSxH, Xv_BxKxSxH)

        # For causal masking, use the cached mask when possible
        if is_causal and attn_mask is None:
            seq_len = attn_k.shape[2]
            if seq_len > 1:  # Only need a mask for sequences longer than 1
                attn_mask = self._get_or_create_causal_mask(seq_len, attn_k.device)

        # Run the attention operation using PyTorch's built-in function
        # This matches the original implementation parameters exactly
        attn_output = F.scaled_dot_product_attention(
            Xq_BxNxTxH,
            attn_k,
            attn_v,
            attn_mask=attn_mask,
            scale=1.0,  # Use same scale as original
            is_causal=is_causal,
            enable_gqa=self.num_gqa_groups > 1,  # Enable GQA when appropriate
        )

        # Efficient transpose back to original shape
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, T, N, H)
            
        # Project to output dimension
        output = self.o_proj(attn_output)

        # Return with original input dtype for consistency
        return output.to(original_dtype)


def optimize_attention_layers(dia_instance):
    """
    Replace all Attention instances in a Dia model with OptimizedAttention.
    
    Args:
        dia_instance: An instance of the Dia class
        
    Returns:
        The same instance with optimized attention layers
    """
    model = dia_instance.model
    count = 0
    
    # Find all Attention instances in the model and replace them
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, Attention):
                # Replace with optimized version
                optimized_attention = OptimizedAttention(child)
                setattr(module, child_name, optimized_attention)
                count += 1
    
    print(f"Replaced {count} attention layers with optimized versions")
    return dia_instance
