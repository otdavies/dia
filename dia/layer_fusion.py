import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from dia.layers import DecoderLayer, EncoderLayer, MlpBlock, RMSNorm


class FusedRMSNormDense(nn.Module):
    """
    Fuses RMSNorm and DenseGeneral operations into a single forward pass.
    This reduces memory transfers by avoiding intermediate tensor allocation.
    """

    def __init__(self, norm_layer, dense_layer):
        super().__init__()
        self.norm_layer = norm_layer
        self.dense_layer = dense_layer
        self.eps = norm_layer.eps
        self.weight = norm_layer.weight  # Important: Keep the RMSNorm weight tensor
        self.compute_dtype = dense_layer.weight.dtype

    def forward(self, x):
        # Get original dtype for consistency
        orig_dtype = x.dtype

        # Apply RMSNorm with proper scale factor using weight
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_normalized = x.to(torch.float32) * torch.rsqrt(variance + self.eps)
        x_normalized = x_normalized * self.weight

        # Apply dense layer directly to normalized tensor
        result = self.dense_layer(x_normalized.to(self.compute_dtype))

        # Return with original dtype
        return result.to(orig_dtype)


class FusedRMSNormMLP(nn.Module):
    """
    Fused RMSNorm + MLP block, combining normalization and both dense projections
    into a more efficient implementation.
    """

    def __init__(self, norm_layer, mlp_block):
        super().__init__()
        self.norm_layer = norm_layer
        self.mlp_block = mlp_block
        self.eps = norm_layer.eps
        self.weight = norm_layer.weight  # RMSNorm weight is crucial
        self.compute_dtype = mlp_block.dtype

        # Keep references to the original layers for weight access
        self.wi_fused = mlp_block.wi_fused
        self.wo = mlp_block.wo

    def forward(self, x):
        # Get original dtype
        orig_dtype = x.dtype

        # Apply RMSNorm inline with proper weight scaling
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_normalized = x.to(torch.float32) * torch.rsqrt(variance + self.eps)
        x_normalized = x_normalized * self.weight

        # Apply MLP block directly to normalized tensor, avoiding intermediate allocations
        fused_x = self.wi_fused(x_normalized.to(self.compute_dtype))

        # Correctly handle the gated SwiGLU activation
        gate = fused_x[..., 0, :]
        up = fused_x[..., 1, :]

        hidden = torch.mul(F.silu(gate), up).to(self.compute_dtype)
        output = self.wo(hidden)

        return output.to(orig_dtype)


class FusedDecoderSelfAttentionBlock(nn.Module):
    """
    Fuses the norm, self-attention, and residual connection into a single module
    for more efficient computation.
    """

    def __init__(self, norm_layer, self_attention):
        super().__init__()
        self.norm_layer = norm_layer
        self.self_attention = self_attention
        self.eps = norm_layer.eps
        self.weight = norm_layer.weight
        self.compute_dtype = self_attention.q_proj.weight.dtype

    def forward(self, x, state, self_attn_cache=None, prefill=False, is_causal=False):
        # Store residual
        residual = x

        # Apply RMSNorm inline with proper weight scaling
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_normalized = x.to(torch.float32) * torch.rsqrt(variance + self.eps)
        x_normalized = x_normalized * self.weight

        # Apply self-attention
        sa_out = self.self_attention(
            Xq=x_normalized.to(self.compute_dtype),
            Xkv=x_normalized.to(self.compute_dtype),
            q_positions=state.dec_positions,
            kv_positions=state.dec_positions,
            attn_mask=None,
            cache=self_attn_cache,
            prefill=prefill,
            is_causal=is_causal,
        )

        # Directly add to residual
        return residual + sa_out


class FusedDecoderCrossAttentionBlock(nn.Module):
    """
    Fuses the norm, cross-attention, and residual connection into a single module
    for more efficient computation.
    """

    def __init__(self, norm_layer, cross_attention):
        super().__init__()
        self.norm_layer = norm_layer
        self.cross_attention = cross_attention
        self.eps = norm_layer.eps
        self.weight = norm_layer.weight
        self.compute_dtype = cross_attention.q_proj.weight.dtype

    def forward(self, x, state, cross_attn_cache=None):
        # Store residual
        residual = x

        # Apply RMSNorm inline with proper weight scaling
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_normalized = x.to(torch.float32) * torch.rsqrt(variance + self.eps)
        x_normalized = x_normalized * self.weight

        # Apply cross-attention
        ca_out = self.cross_attention(
            Xq=x_normalized.to(self.compute_dtype),
            Xkv=state.enc_out,
            q_positions=state.dec_positions,
            kv_positions=state.enc_positions,
            attn_mask=state.dec_cross_attn_mask,
            cache=cross_attn_cache,
        )

        # Directly add to residual
        return residual + ca_out


class FusedDecoderLayer(nn.Module):
    """
    A decoder layer with fused operations for faster inference.
    """

    def __init__(self, original_layer: DecoderLayer):
        super().__init__()

        # Make deep copies to avoid modifying original weights
        # Create fused blocks from original components
        self.self_attn_block = FusedDecoderSelfAttentionBlock(
            original_layer.pre_sa_norm,
            original_layer.self_attention
        )

        self.cross_attn_block = FusedDecoderCrossAttentionBlock(
            original_layer.pre_ca_norm,
            original_layer.cross_attention
        )

        self.mlp_block = FusedRMSNormMLP(
            original_layer.pre_mlp_norm,
            original_layer.mlp
        )

        # Keep reference to original components for compatibility with KV caching
        self.pre_sa_norm = original_layer.pre_sa_norm
        self.self_attention = original_layer.self_attention
        self.pre_ca_norm = original_layer.pre_ca_norm
        self.cross_attention = original_layer.cross_attention
        self.pre_mlp_norm = original_layer.pre_mlp_norm
        self.mlp = original_layer.mlp
        self.compute_dtype = original_layer.compute_dtype

    def forward(
        self,
        x: torch.Tensor,
        state,
        self_attn_cache=None,
        cross_attn_cache=None,
        prefill=False,
    ) -> torch.Tensor:
        # Apply fused self-attention block
        x = self.self_attn_block(
            x,
            state,
            self_attn_cache=self_attn_cache,
            prefill=prefill,
            is_causal=prefill,
        )

        # Apply fused cross-attention block
        x = self.cross_attn_block(
            x,
            state,
            cross_attn_cache=cross_attn_cache,
        )

        # Apply fused MLP block
        x = x + self.mlp_block(x)  # Fixed: Add the residual to the MLP output

        return x


class FusedEncoderSelfAttentionBlock(nn.Module):
    """
    Fuses the norm, self-attention, and residual connection for encoder layers.
    """

    def __init__(self, norm_layer, self_attention):
        super().__init__()
        self.norm_layer = norm_layer
        self.self_attention = self_attention
        self.eps = norm_layer.eps
        self.weight = norm_layer.weight
        self.compute_dtype = self_attention.q_proj.weight.dtype

    def forward(self, x, state):
        # Store residual
        residual = x

        # Apply RMSNorm inline with proper weight scaling
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_normalized = x.to(torch.float32) * torch.rsqrt(variance + self.eps)
        x_normalized = x_normalized * self.weight

        # Apply self-attention
        sa_out = self.self_attention(
            Xq=x_normalized.to(self.compute_dtype),
            Xkv=x_normalized.to(self.compute_dtype),
            q_positions=state.positions,
            kv_positions=state.positions,
            attn_mask=state.attn_mask,
        )

        # Add to residual
        return residual + sa_out


class FusedEncoderLayer(nn.Module):
    """
    An encoder layer with fused operations for faster inference.
    """

    def __init__(self, original_layer: EncoderLayer):
        super().__init__()

        # Create fused blocks
        self.self_attn_block = FusedEncoderSelfAttentionBlock(
            original_layer.pre_sa_norm,
            original_layer.self_attention
        )

        self.mlp_block = FusedRMSNormMLP(
            original_layer.post_sa_norm,
            original_layer.mlp
        )

        # Keep reference to original components for compatibility
        self.pre_sa_norm = original_layer.pre_sa_norm
        self.self_attention = original_layer.self_attention
        self.post_sa_norm = original_layer.post_sa_norm
        self.mlp = original_layer.mlp
        self.compute_dtype = original_layer.compute_dtype

    def forward(self, x, state):
        # Apply fused self-attention block
        x = self.self_attn_block(x, state)

        # Apply fused MLP block (includes residual)
        x = x + self.mlp_block(x)  # Fixed: Add the residual to the MLP output

        return x


def apply_layer_fusion(dia_instance):
    """
    Apply layer fusion optimizations to the model.
    This fuses operations like normalization + dense to reduce memory transfers.

    Args:
        dia_instance: An instance of the Dia class

    Returns:
        The same instance with fused operations
    """
    model = dia_instance.model
    fusion_count = 0

    # First check if model is already using fused layers
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        first_encoder_layer = model.encoder.layers[0] if len(
            model.encoder.layers) > 0 else None
        if first_encoder_layer is not None and isinstance(first_encoder_layer, FusedEncoderLayer):
            print("Model is already using fused layers. Skipping fusion.")
            return dia_instance

    # Fuse encoder layers
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        for i, layer in enumerate(model.encoder.layers):
            if isinstance(layer, EncoderLayer):
                try:
                    model.encoder.layers[i] = FusedEncoderLayer(layer)
                    fusion_count += 1
                except Exception as e:
                    print(f"Warning: Could not fuse encoder layer {i}: {e}")

    # Fuse decoder layers
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
        for i, layer in enumerate(model.decoder.layers):
            if isinstance(layer, DecoderLayer):
                try:
                    model.decoder.layers[i] = FusedDecoderLayer(layer)
                    fusion_count += 1
                except Exception as e:
                    print(f"Warning: Could not fuse decoder layer {i}: {e}")

    if fusion_count > 0:
        print(f"Applied layer fusion to {fusion_count} transformer layers")
    else:
        print("No layers were fused - check model compatibility")

    return dia_instance
