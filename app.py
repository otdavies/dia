import argparse
import gc
import tempfile
import time
import random
import io
import contextlib
import re
from pathlib import Path
from typing import Optional, Tuple, List

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from dia.model import Dia


# --- Global Setup ---
parser = argparse.ArgumentParser(description="Gradio interface for Nari TTS")
parser.add_argument("--device", type=str, default=None,
                    help="Force device (e.g., 'cuda', 'mps', 'cpu')")
parser.add_argument("--share", action="store_true",
                    help="Enable Gradio sharing")

args = parser.parse_args()


# Determine device
if args.device:
    device = torch.device(args.device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
# Simplified MPS check for broader compatibility
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # Basic check is usually sufficient, detailed check can be problematic
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load Nari model and config
print("Loading Nari model...")
try:
    # Load model with enhanced 8-bit quantization for better performance
    def load_optimized_model(device, verbose=True):
        if verbose:
            print("Loading Dia model with optimizations...")

        try:
            # Step 1: Load model normally
            model = Dia.from_pretrained(
                "nari-labs/Dia-1.6B",
                compute_dtype="float16",
                device=device
            )

            # Step 2: Apply dynamic quantization with expanded coverage
            quantized_model = torch.quantization.quantize_dynamic(
                model.model,
                {torch.nn.Linear, torch.nn.LSTM},
                dtype=torch.qint8
            )

            # Step 3: Dereference the original
            model.model = None
            torch.cuda.empty_cache()

            # Step 4: Replace with quantized
            model.model = quantized_model

            if verbose:
                print("Model loaded with optimized quantization")

            return model

        except Exception as e:
            print(f"Error loading Dia model: {e}")
            raise

    # Load the model
    model = load_optimized_model(device)

except Exception as e:
    print(f"Error loading Nari model: {e}")
    raise


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_effective_length(text):
    """Counts effective length treating [S1] and [S2] as single characters."""
    return len(text.replace("[S1]", "¤").replace("[S2]", "¤"))


def improved_text_chunking(text, max_effective_chars=64):
    """
    Splits text into chunks by sentence boundaries while respecting special
    tokens and max size constraints. Improves quality by not cutting mid-sentence.
    """
    # Protect special tokens from splitting
    protected_text = text.replace("[S1]", " [S1] ")
    protected_text = protected_text.replace("[S2]", " [S2] ")

    # Split on sentence boundaries (., !, ?)
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, protected_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If single sentence exceeds max chars, resort to word splitting
        if count_effective_length(sentence) > max_effective_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # Fall back to word splitting for this long sentence
            word_chunks = split_by_words_respecting_special_tokens(
                sentence, max_effective_chars)
            chunks.extend(word_chunks)
            continue

        # Try adding this sentence to current chunk
        tentative_chunk = (current_chunk + " " +
                           sentence).strip() if current_chunk else sentence

        if count_effective_length(tentative_chunk) > max_effective_chars:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = tentative_chunk

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Ensure we don't have empty chunks
    chunks = [c for c in chunks if count_effective_length(c) > 0]

    return chunks


def split_by_words_respecting_special_tokens(text, max_effective_chars=64):
    """Splits text into chunks close to max_effective_chars, preserving full words and [S1]/[S2] markers."""
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        tentative_chunk = (current_chunk + " " +
                           word).strip() if current_chunk else word
        if count_effective_length(tentative_chunk) > max_effective_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                chunks.append(word)
                current_chunk = ""
        else:
            current_chunk = tentative_chunk

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def auto_adjust_chunk_size(text, user_chunk_size):
    """Auto-adjusts chunk size for optimal balance between speed and quality."""
    effective_chars = count_effective_length(text)
    if user_chunk_size > 0:
        # If user explicitly sets a chunk size, respect it
        return int(user_chunk_size)
    else:
        # Auto-tune based on input size - larger chunks for longer text
        if effective_chars <= 1024:
            return 48
        elif effective_chars <= 4096:
            return 64  # Increased for better sentence grouping
        else:
            return 128  # Increased for better efficiency


def optimized_batch_chunks(chunks, batch_size=4, max_tokens_per_batch=3072):
    """
    Create optimized batches that balance between batch size and token count.
    This avoids processing very small or overly large batches.
    """
    # First, estimate token count per chunk (rough approximation)
    # ~0.9 tokens per character
    estimated_tokens = [len(chunk) * 0.9 for chunk in chunks]

    current_batch = []
    current_token_count = 0
    batches = []

    for i, (chunk, token_est) in enumerate(zip(chunks, estimated_tokens)):
        # If adding this chunk would exceed token limit and we have chunks already,
        # or if we've reached batch size, yield the current batch
        if ((current_token_count + token_est > max_tokens_per_batch and current_batch) or
                len(current_batch) >= batch_size):
            batches.append(current_batch)
            current_batch = [chunk]
            current_token_count = token_est
        else:
            current_batch.append(chunk)
            current_token_count += token_est

    # Don't forget the last batch
    if current_batch:
        batches.append(current_batch)

    return batches


def run_inference(
    text_input: str,
    audio_prompt_text_input: str,
    audio_prompt_input: Optional[Tuple[int, np.ndarray]],
    max_new_tokens: int,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int,
    speed_factor: float,
    chunk_size: int,
    seed: Optional[int] = None,
):
    """
    Runs Nari inference using the globally loaded model and provided inputs.
    Supports dynamic chunking and token scaling.
    """

    global model, device  # Access global model, config, device
    console_output_buffer = io.StringIO()

    with contextlib.redirect_stdout(console_output_buffer):
        # Validation
        if not text_input or text_input.isspace():
            raise gr.Error("Text input cannot be empty.")

        if audio_prompt_input and (not audio_prompt_text_input or audio_prompt_text_input.isspace()):
            raise gr.Error("Audio Prompt Text input cannot be empty.")

        # Set and Display Generation Seed
        if seed is None or seed < 0:
            seed = random.randint(0, 2**32 - 1)
            print(f"\nNo seed provided, generated random seed: {seed}\n")
        else:
            print(f"\nUsing user-selected seed: {seed}\n")
        set_seed(seed)

        # Preprocess audio prompt
        temp_audio_prompt_path = None
        output_audio = (44100, np.zeros(1, dtype=np.float32))
        prompt_path_for_generate = None

        try:
            if audio_prompt_input is not None:
                sr, audio_data = audio_prompt_input
                if audio_data is None or audio_data.size == 0 or audio_data.max() == 0:
                    gr.Warning(
                        "Audio prompt seems empty or silent, ignoring prompt.")
                else:
                    with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f_audio:
                        temp_audio_prompt_path = f_audio.name

                        if np.issubdtype(audio_data.dtype, np.integer):
                            max_val = np.iinfo(audio_data.dtype).max
                            audio_data = audio_data.astype(
                                np.float32) / max_val
                        elif not np.issubdtype(audio_data.dtype, np.floating):
                            try:
                                audio_data = audio_data.astype(np.float32)
                            except Exception as conv_e:
                                raise gr.Error(
                                    f"Failed to convert audio prompt: {conv_e}")

                        if audio_data.ndim > 1:
                            audio_data = np.mean(audio_data, axis=-1)
                            audio_data = np.ascontiguousarray(audio_data)

                        sf.write(temp_audio_prompt_path,
                                 audio_data, sr, subtype="FLOAT")
                        prompt_path_for_generate = temp_audio_prompt_path
                        print(
                            f"Created temporary audio prompt file: {temp_audio_prompt_path} (orig sr: {sr})")

            # --- Improved Chunking ---
            chunk_size = auto_adjust_chunk_size(text_input, chunk_size)
            print(
                f"Auto-selected chunk size: {chunk_size} effective characters per chunk.")

            # Use improved sentence-aware chunking
            chunks = improved_text_chunking(
                text_input, max_effective_chars=chunk_size)

            print(
                f"Chunked into {len(chunks)} chunks (respecting sentence boundaries).")

            audio_segments = []

            start_time = time.time()

            # Use optimized adaptive batching
            batches = optimized_batch_chunks(
                chunks, batch_size=8, max_tokens_per_batch=3072)

            for batch_idx, chunk_batch in enumerate(batches):
                print(
                    f"Generating batch {batch_idx + 1}/{len(batches)} with {len(chunk_batch)} chunks...")

                batch_input_text = "\n".join(
                    chunk.strip() for chunk in chunk_batch).strip()

                if not batch_input_text:
                    raise gr.Error(
                        "All chunks in this batch were empty after trimming. Cannot generate.")
                effective_chars = count_effective_length(batch_input_text)
                scaling_factor = effective_chars / chunk_size
                adjusted_tokens = int(max_new_tokens * scaling_factor)
                adjusted_tokens = max(256, adjusted_tokens)

                with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    generated_batch_audio = model.generate(
                        batch_input_text,
                        max_tokens=adjusted_tokens,
                        cfg_scale=cfg_scale,
                        temperature=temperature,
                        top_p=top_p,
                        cfg_filter_top_k=cfg_filter_top_k,
                        use_torch_compile=True,
                        audio_prompt=prompt_path_for_generate,
                        audio_prompt_text=audio_prompt_text_input,
                        use_offloading=True,  # Enable selective offloading for memory efficiency
                    )

                if generated_batch_audio is not None:
                    audio_segments.append(generated_batch_audio)

                    # Add a small silence buffer **after the batch** (but NOT after the last batch)
                    if batch_idx < len(batches) - 1:
                        silence_duration_sec = 0.2
                        silence_samples = int(44100 * silence_duration_sec)
                        silence = np.zeros(silence_samples, dtype=np.float32)
                        audio_segments.append(silence)

                # Force release memory after each batch
                torch.cuda.empty_cache()
                gc.collect()

            if not audio_segments:
                output_audio_np = None
            else:
                output_audio_np = np.concatenate(audio_segments)

            end_time = time.time()
            print(
                f"Generation finished in {end_time - start_time:.2f} seconds.\n")

            # --- Postprocessing ---
            if output_audio_np is not None:
                output_sr = 44100

                # Slowdown if needed
                original_len = len(output_audio_np)
                speed_factor = max(0.1, min(speed_factor, 5.0))
                target_len = int(original_len / speed_factor)

                if target_len != original_len and target_len > 0:
                    x_original = np.arange(original_len)
                    x_resampled = np.linspace(0, original_len - 1, target_len)
                    resampled_audio_np = np.interp(
                        x_resampled, x_original, output_audio_np)
                    output_audio = (
                        output_sr, resampled_audio_np.astype(np.float32))
                    print(
                        f"Resampled audio from {original_len} to {target_len} samples for {speed_factor:.2f}x speed.")
                else:
                    output_audio = (output_sr, output_audio_np)
                    print(
                        f"Skipping audio speed adjustment (factor: {speed_factor:.2f}).")

                # Final output conversion
                if output_audio[1].dtype in (np.float32, np.float64):
                    audio_for_gradio = np.clip(output_audio[1], -1.0, 1.0)
                    audio_for_gradio = (
                        audio_for_gradio * 32767).astype(np.int16)
                    output_audio = (output_sr, audio_for_gradio)
                    print("Converted audio to int16 for Gradio output.")
            else:
                print("\nGeneration finished, but no valid tokens were produced.")
                gr.Warning("Generation produced no output.")

        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            raise gr.Error(f"Inference failed: {e}")

        finally:
            # Clean up temp files
            if temp_audio_prompt_path and Path(temp_audio_prompt_path).exists():
                try:
                    Path(temp_audio_prompt_path).unlink()
                    print(
                        f"Deleted temporary audio prompt file: {temp_audio_prompt_path}")
                except Exception as cleanup_e:
                    print(
                        f"Warning: Error deleting temporary audio prompt file: {cleanup_e}")

        console_output = console_output_buffer.getvalue()

    try:
        torch.cuda.empty_cache()
        gc.collect()
    except:
        print("No cache to clear or garbage to collect")
    finally:
        print("Generation completed")

    return output_audio, seed, console_output


# --- Create Gradio Interface ---
css = """
#col-container {max-width: 90%; margin-left: auto; margin-right: auto;}
"""
# Attempt to load default text from example.txt
default_text = "[S1] Dia is an open weights text to dialogue model. \n[S2] You get full control over scripts and voices. \n[S1] Wow. Amazing. (laughs) \n[S2] Try it now on Git hub or Hugging Face."
example_txt_path = Path("./example.txt")
if example_txt_path.exists():
    try:
        default_text = example_txt_path.read_text(encoding="utf-8").strip()
        if not default_text:  # Handle empty example file
            default_text = "Example text file was empty."
    except Exception as e:
        print(f"Warning: Could not read example.txt: {e}")


# Build Gradio UI
with gr.Blocks(css=css, theme="gradio/dark") as demo:
    gr.Markdown("# Nari Text-to-Speech Synthesis")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            with gr.Accordion("Audio Reference Prompt (Optional)", open=False):
                audio_prompt_input = gr.Audio(
                    label="Audio Prompt (Optional)",
                    show_label=True,
                    sources=["upload", "microphone"],
                    type="numpy",
                )
                audio_prompt_text_input = gr.Textbox(
                    label="Transcript of Audio Prompt (Required if using Audio Prompt)",
                    placeholder="Enter text here...",
                    value="",
                    lines=5,
                )
            text_input = gr.Textbox(
                label="Text To Generate",
                placeholder="Enter text here...",
                value=default_text,
                lines=5,
            )
            with gr.Accordion("Generation Parameters", open=False):
                chunk_size = gr.Number(
                    label="Chunk Size (Effective Characters)",
                    minimum=0,
                    value=0,
                    precision=0,
                    step=1,
                    info="If 0, auto-selects chunk size for optimal speed. Otherwise, set number of effective characters per generation chunk."
                )
                max_new_tokens = gr.Slider(
                    label="Max New Tokens (Audio Length)",
                    minimum=860,
                    maximum=3072,
                    value=model.config.data.audio_length,
                    step=50,
                    info="Controls the maximum length of the generated audio (more tokens = longer audio).",
                )
                cfg_scale = gr.Slider(
                    label="CFG Scale (Guidance Strength)",
                    minimum=1.0,
                    maximum=5.0,
                    value=3.0,
                    step=0.1,
                    info="Higher values increase adherence to the text prompt.",
                )
                temperature = gr.Slider(
                    label="Temperature (Randomness)",
                    minimum=1.0,
                    maximum=1.5,
                    value=1.3,
                    step=0.05,
                    info="Lower values make the output more deterministic, higher values increase randomness.",
                )
                top_p = gr.Slider(
                    label="Top P (Nucleus Sampling)",
                    minimum=0.80,
                    maximum=1.0,
                    value=0.95,
                    step=0.01,
                    info="Filters vocabulary to the most likely tokens cumulatively reaching probability P.",
                )
                cfg_filter_top_k = gr.Slider(
                    label="CFG Filter Top K",
                    minimum=15,
                    maximum=50,
                    value=30,
                    step=1,
                    info="Top k filter for CFG guidance.",
                )
                speed_factor_slider = gr.Slider(
                    label="Speed Factor",
                    minimum=0.8,
                    maximum=1.0,
                    value=0.94,
                    step=0.02,
                    info="Adjusts the speed of the generated audio (1.0 = original speed).",
                )
                seed_input = gr.Number(
                    label="Generation Seed (Optional)",
                    value=-1,
                    precision=0,
                    step=1,
                    interactive=True,
                    info="Set a generation seed for reproducible outputs. Leave empty or -1 for random seed.",
                )

            run_button = gr.Button("Generate Audio", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Generated Audio",
                type="numpy",
                autoplay=False,
            )
            seed_output = gr.Textbox(
                label="Generation Seed",
                interactive=False
            )
            console_output = gr.Textbox(
                label="Console Output Log", lines=10, interactive=False
            )

    # Link button click to function
    run_button.click(
        fn=run_inference,
        inputs=[
            text_input,
            audio_prompt_text_input,
            audio_prompt_input,
            max_new_tokens,
            cfg_scale,
            temperature,
            top_p,
            cfg_filter_top_k,
            speed_factor_slider,
            chunk_size,
            seed_input,
        ],
        outputs=[
            audio_output,
            seed_output,
            console_output,
        ],
        api_name="generate_audio",
    )

    # Note: Example section omitted as requested

# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio interface...")

    # set `GRADIO_SERVER_NAME`, `GRADIO_SERVER_PORT` env vars to override default values
    # use `GRADIO_SERVER_NAME=0.0.0.0` for Docker
    demo.launch(share=args.share)
