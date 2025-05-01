import torch
import time
from dia.config import DiaConfig
from dia.model import Dia, ComputeDtype
from dia.optimizations import (
    optimize_dia_model,
    create_quantized_layers,
    apply_layer_fusion,
)
from dia.attention_opt import optimize_attention_layers
from dia.decoder_opt import optimize_decoder


def benchmark_inference(model, text_prompt, n_runs=5):
    """Benchmark inference speed"""
    # Warmup run
    _ = model.generate(text_prompt, max_tokens=100, verbose=False)

    # Timed runs
    times = []
    for i in range(n_runs):
        start_time = time.time()
        _ = model.generate(text_prompt, max_tokens=100, verbose=False)
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Run {i+1}/{n_runs}: {times[-1]:.4f}s")

    avg_time = sum(times) / len(times)
    print(f"Average inference time: {avg_time:.4f}s")
    return avg_time


def main():
    # Load model (from local or HF hub)
    model = Dia.from_pretrained(
        "nari-labs/Dia-1.6B",
        compute_dtype=ComputeDtype.FLOAT16,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Benchmark original model
    print("=== Original Model Performance ===")
    text_prompt = "Please convert this text to speech with a cheerful tone."
    original_time = benchmark_inference(model, text_prompt)

    # Apply FastDenseGeneral optimizations
    print("\n=== Applying FastDenseGeneral Optimizations ===")
    optimized_model = optimize_dia_model(
        model, quantize=False, ultra_mode=True)
    dense_opt_time = benchmark_inference(optimized_model, text_prompt)

    # Apply attention-specific optimizations
    print("\n=== Applying Attention-Specific Optimizations ===")
    optimized_model = optimize_attention_layers(optimized_model)
    attn_opt_time = benchmark_inference(optimized_model, text_prompt)

    # Apply decoder-specific optimizations
    print("\n=== Applying Decoder-Specific Optimizations ===")
    optimized_model = optimize_decoder(optimized_model)
    decoder_opt_time = benchmark_inference(optimized_model, text_prompt)

    # Apply layer fusion optimizations (placeholder)
    print("\n=== Applying Layer Fusion Optimizations ===")
    optimized_model = apply_layer_fusion(optimized_model)
    fused_time = benchmark_inference(optimized_model, text_prompt)

    # Print improvement statistics
    print("\n=== Performance Improvement Summary ===")
    print(f"Original model:          {original_time:.4f}s")
    print(
        f"Optimized dense layers:  {dense_opt_time:.4f}s ({original_time/dense_opt_time:.2f}x speedup)")
    print(
        f"Optimized attention:     {attn_opt_time:.4f}s ({original_time/attn_opt_time:.2f}x speedup)")
    print(
        f"Optimized decoder:       {decoder_opt_time:.4f}s ({original_time/decoder_opt_time:.2f}x speedup)")
    print(
        f"With layer fusion:       {fused_time:.4f}s ({original_time/fused_time:.2f}x speedup)")


if __name__ == "__main__":
    main()
