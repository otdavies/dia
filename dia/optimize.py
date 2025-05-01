import time
from typing import Dict, Any, Optional
from dia.model import Dia
from dia.optimizations import optimize_dia_model
from dia.attention_opt import optimize_attention_layers
from dia.decoder_opt import optimize_decoder
from dia.layer_fusion import apply_layer_fusion
from dia.quantization import apply_dynamic_quantization


def optimize_dia_for_inference(
    dia_instance: Dia,
    optimize_dense: bool = True,
    optimize_attention: bool = False,
    optimize_decoder_mem: bool = False,
    apply_fusion: bool = False,
    apply_quantization: bool = False,
    verbose: bool = True,
) -> Dia:
    """
    Apply a comprehensive set of optimizations to a Dia model for faster inference.

    Args:
        dia_instance: A Dia model instance
        optimize_dense: Whether to optimize DenseGeneral layers
        optimize_attention: Whether to optimize Attention layers
        optimize_decoder_mem: Whether to optimize the Decoder for memory efficiency
        apply_fusion: Whether to apply layer fusion 
        use_cuda_kernels: Whether to use custom CUDA kernels (when available)
        apply_quantization: Whether to apply quantization
        quantization_config: Additional configuration for quantization
        ultra_mode: Whether to use more aggressive optimizations
        verbose: Whether to print optimization details

    Returns:
        The optimized Dia model instance
    """
    start_time = time.time()

    if verbose:
        print("Applying inference optimizations to Dia model...")

    # Apply optimizations in order of dependency
    if optimize_dense:
        if verbose:
            print("Optimizing dense layers...")
        dia_instance = optimize_dia_model(
            dia_instance,
            quantize=apply_quantization,  # We'll handle quantization separately
        )

    if optimize_attention:
        if verbose:
            print("Optimizing attention mechanisms...")
        dia_instance = optimize_attention_layers(dia_instance)

    if optimize_decoder_mem:
        if verbose:
            print("Optimizing decoder for memory efficiency...")
        dia_instance = optimize_decoder(dia_instance)

    if apply_fusion:
        if verbose:
            print("Applying layer fusion optimizations...")
        dia_instance = apply_layer_fusion(dia_instance)

    # Apply quantization last, after all other optimizations
    if apply_quantization:
        if verbose:
            print("Applying model quantization...")

        dia_instance = apply_dynamic_quantization(
            dia_instance,
        )

    if verbose:
        elapsed = time.time() - start_time
        print(f"All optimizations applied in {elapsed:.2f} seconds")

    return dia_instance
