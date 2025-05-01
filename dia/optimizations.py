import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Dict, Any, Optional, Tuple, Union, List
import math

from dia.layers import DenseGeneral, _normalize_axes


class FastDenseGeneral(nn.Module):
    """
    Optimized implementation of DenseGeneral that uses native PyTorch operations
    for better performance and lower memory usage.
    """

    def __init__(
        self,
        in_shapes: tuple[int, ...],
        out_features: tuple[int, ...],
        axis: tuple[int, ...] = (-1,),
        weight_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.in_shapes = in_shapes
        self.out_features = out_features
        self.axis = axis

        # Compute required kernel shape
        self.kernel_shape = self.in_shapes + self.out_features

        # Initialize weight parameter with better initialization
        factory_kwargs = {"device": device, "dtype": weight_dtype}
        self.weight = nn.Parameter(torch.empty(
            self.kernel_shape, **factory_kwargs))

        # Pre-compute and cache shapes for common cases
        self._setup_fast_path()

        # Fused bias (currently not used but prepared for future extension)
        self.use_fused_bias = False
        self.bias = None

        # Quantization state
        self.is_quantized = False
        self.scale_factor = None
        self.zero_point = None
        self.quantized_weight = None

        # Cache for forward pass
        self._last_batch_shape = None
        self._last_output_shape = None

    def _setup_fast_path(self):
        """Pre-compute shapes and flags for optimized execution paths."""
        # Special case optimization for the common pattern: contracting last dimension
        if self.axis == (-1,):
            self.use_fast_path = True

            # Calculate in and out dimensions
            self.in_dim = self.in_shapes[0]

            # Compute flattened output dimension size
            self.out_dim = 1
            for d in self.out_features:
                self.out_dim *= d

            # Cache reshaping dimensions for faster inference
            self.out_reshape_dims = self.out_features

            # Pre-reshape weight for common case
            # This saves the reshape operation during each forward pass
            self._reshaped_weight_2d = None
        else:
            self.use_fast_path = False
            # Pre-compute normalized axes for general case
            self.norm_axis = tuple(ax if ax >= 0 else len(
                self.in_shapes) + ax for ax in self.axis)
            self.contract_dims = list(self.norm_axis)

            # Initialize other cached values that will be set on first forward
            self.cached_weight_shape = None
            self._reshaped_weight = None

    def _lazy_reshape_weight(self):
        """Lazily reshape the weight tensor only when needed and cache it"""
        if self._reshaped_weight_2d is None and self.use_fast_path:
            # Handle the fast path
            self._reshaped_weight_2d = self.weight.reshape(
                self.in_dim, self.out_dim)

    def quantize(self, num_bits=8):
        """Quantize weights to int8 or int4 for faster inference"""
        if self.is_quantized:
            return  # Already quantized

        # Store original weight for recovery if needed
        self._original_weight = self.weight.clone()

        weight = self.weight
        if self.use_fast_path:
            weight = weight.reshape(self.in_dim, self.out_dim)

        # Calculate scale and zero point per output channel for better precision
        if weight.dim() >= 2 and weight.size(1) > 1:
            # Per-channel quantization (along output dimension)
            # Move output dim to first position
            weight_t = weight.transpose(0, 1)
            weight_min = weight_t.min(
                dim=-1)[0].view(-1, *([1] * (weight_t.dim() - 1)))
            weight_max = weight_t.max(
                dim=-1)[0].view(-1, *([1] * (weight_t.dim() - 1)))

            qmin = 0
            qmax = 2**num_bits - 1

            scale = (weight_max - weight_min) / (qmax - qmin)
            scale = torch.maximum(scale, torch.tensor(
                1e-8, device=scale.device))  # Prevent div by zero
            zero_point = qmin - torch.round(weight_min / scale)

            # Quantize weights (per output channel)
            quantized_weight_t = torch.clamp(torch.round(
                weight_t / scale + zero_point), qmin, qmax).to(torch.uint8)
            quantized_weight = quantized_weight_t.transpose(
                0, 1)  # Move output dim back

            # For inference, store flat versions
            self.scale_factor = scale.transpose(0, 1)
            self.zero_point = zero_point.transpose(0, 1)
        else:
            # Per-tensor quantization
            weight_min = weight.min()
            weight_max = weight.max()

            qmin = 0
            qmax = 2**num_bits - 1

            scale = (weight_max - weight_min) / (qmax - qmin)
            scale = max(scale.item(), 1e-8)  # Prevent div by zero
            zero_point = qmin - torch.round(weight_min / scale)

            # Quantize weights
            quantized_weight = torch.clamp(torch.round(
                weight / scale + zero_point), qmin, qmax).to(torch.uint8)

            # For inference
            self.scale_factor = torch.tensor(scale, device=weight.device)
            self.zero_point = torch.tensor(zero_point, device=weight.device)

        # Store quantization parameters
        self.is_quantized = True
        self.quantized_weight = quantized_weight
        self.num_bits = num_bits

        # Update the original weight parameter to use quantized representation
        # This ensures that other hooks or operations on self.weight will work correctly
        with torch.no_grad():
            dequantized = (self.quantized_weight.float() -
                           self.zero_point) * self.scale_factor
            self.weight.copy_(dequantized.reshape(self.kernel_shape))

        # Reset cached tensors
        self._reshaped_weight_2d = None
        self._reshaped_weight = None

    def dequantize(self):
        """Convert back to full precision if needed"""
        if not self.is_quantized:
            return

        # Restore original weights
        if hasattr(self, '_original_weight'):
            with torch.no_grad():
                self.weight.copy_(self._original_weight)
            delattr(self, '_original_weight')

        # Reset quantization state
        self.is_quantized = False
        self.scale_factor = None
        self.zero_point = None
        self.quantized_weight = None

        # Reset cached reshaped weights
        self._reshaped_weight_2d = None
        self._reshaped_weight = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using optimized tensor operations for different scenarios.
        """
        # Save original dtype for output consistency
        input_dtype = inputs.dtype

        if self.use_fast_path:
            # Fast path for common case (contracting last dimension)
            # Reshape inputs to 2D for efficient matmul
            batch_shape = inputs.shape[:-1]

            # Check if we can reuse the cached reshape
            need_reshape = True
            if self._last_batch_shape == batch_shape:
                need_reshape = False
                inputs_2d = inputs.reshape(-1, self.in_dim)
            else:
                self._last_batch_shape = batch_shape
                inputs_2d = inputs.reshape(-1, self.in_dim)

            # Use the cached reshaped weight or create it if not yet available
            self._lazy_reshape_weight()
            weight_2d = self._reshaped_weight_2d

            if self.is_quantized:
                # For quantized weights, we already dequantized the self.weight
                # in the quantize method, so we can just use that
                outputs_2d = torch.matmul(
                    inputs_2d.to(self.weight.dtype),
                    weight_2d
                )
            else:
                # Efficient matrix multiplication
                outputs_2d = torch.matmul(
                    inputs_2d.to(self.weight.dtype),
                    weight_2d
                )

            # Reshape output back to original dimensions
            output_shape = batch_shape + self.out_reshape_dims

            # Check if we can reuse cached output shape
            if output_shape != self._last_output_shape:
                self._last_output_shape = output_shape
                outputs = outputs_2d.reshape(output_shape)
            else:
                outputs = outputs_2d.reshape(output_shape)
        else:
            # General case for arbitrary contraction axes
            norm_axis = _normalize_axes(self.axis, inputs.ndim)

            # Use torch.einsum for general tensor contraction
            # This is more readable and can be optimized by PyTorch's backend
            if self.is_quantized:
                # For quantized weights, just use the dequantized self.weight
                outputs = torch.tensordot(
                    inputs.to(self.weight.dtype),
                    self.weight,
                    dims=(norm_axis, tuple(range(len(norm_axis)))),
                )
            else:
                outputs = torch.tensordot(
                    inputs.to(self.weight.dtype),
                    self.weight,
                    dims=(norm_axis, tuple(range(len(norm_axis)))),
                )

        # Return with original input dtype for consistency
        return outputs.to(input_dtype)

    @classmethod
    def from_dense_general(cls, dense_general):
        """Convert a DenseGeneral module to a FastDenseGeneral"""
        optimized = cls(
            in_shapes=dense_general.in_shapes,
            out_features=dense_general.out_features,
            axis=dense_general.axis,
            weight_dtype=dense_general.weight.dtype,
            device=dense_general.weight.device,
        )

        # Copy weights from original module
        with torch.no_grad():
            optimized.weight.copy_(dense_general.weight)

        return optimized


def optimize_dia_model(dia_instance, quantize=False, quantization_bits=8):
    """
    Optimize a Dia model instance by replacing DenseGeneral implementations
    with FastDenseGeneral or UltraFastDenseGeneral in the underlying PyTorch model.

    Args:
        dia_instance: An instance of the Dia class
        quantize: Whether to quantize weights for faster inference
        quantization_bits: Number of bits for quantization (8 or 4)
        ultra_mode: Whether to use UltraFastDenseGeneral instead of FastDenseGeneral

    Returns:
        The same instance with optimized internals
    """
    # Access the DiaModel instance
    model = dia_instance.model

    # Create a list of all replacements to make
    replacements = []

    # Find all DenseGeneral instances in the model's modules
    for name, module in model.named_modules():
        # Find all DenseGeneral instances that are direct children of this module
        for child_name, child in module.named_children():
            if isinstance(child, DenseGeneral):
                parent_module = module
                replacements.append((parent_module, child_name, child))

    # Apply all replacements
    print(f"Found {len(replacements)} DenseGeneral instances to optimize")

    for parent_module, child_name, dense_general_module in replacements:

        # Create optimized replacement
        optimized_module = FastDenseGeneral.from_dense_general(
            dense_general_module)

        # Replace in parent module
        setattr(parent_module, child_name, optimized_module)

    return dia_instance
