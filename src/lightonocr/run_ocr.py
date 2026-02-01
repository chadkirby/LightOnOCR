import torch
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor, TextIteratorStreamer
from threading import Thread
import pypdfium2 as pdfium
from PIL import Image
import os
import sys
import argparse
import re
import time
import datetime

def humanize_count(n):
    """Humanizes an integer count (e.g. 1500 -> 1.5k, 1250000 -> 1.25m)."""
    if n < 1000:
        return str(n)
    if n < 1000000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1000000:.2f}m"

def parse_pages(pages_str, num_pages, offset=0):
    """Parses a page range string into a list of indices."""
    if not pages_str:
        return list(range(num_pages))

    indices = set()
    parts = pages_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                indices.update(range(max(0, start - offset), min(num_pages, end - offset + 1)))
            except ValueError:
                print(f"Warning: Invalid range '{part}' ignored.", file=sys.stderr)
        else:
            try:
                val = int(part)
                idx = val - offset
                if 0 <= idx < num_pages:
                    indices.add(idx)
                else:
                    print(f"Warning: Page value {val} out of range ignored.", file=sys.stderr)
            except ValueError:
                print(f"Warning: Invalid page value '{part}' ignored.", file=sys.stderr)
    return sorted(list(indices))

def main():
    description = """
    LightOnOCR CLI - Fast, local PDF OCR for Mac (Apple Silicon optimized).
    Converts PDF pages into clean, naturally ordered Markdown text using the LightOnOCR-2-1B model.
    """

    epilog = """
    Examples:
      # Process all pages and save to a file
      lighton-ocr document.pdf -o output.md

      # Process specific 1-based page numbers (Human style)
      lighton-ocr document.pdf --pages 1,3,5-10

      # Process specific 0-based indices (Programmer style)
      lighton-ocr document.pdf --indices 0,2,4-9
    """

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pdf", help="Path to the PDF file to process.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-p", "--pages", help="1-based page numbers (e.g. '1', '1-5', '1,3,5'). Matching human/PDF viewer numbering.")
    group.add_argument("-i", "--indices", help="0-based page indices (e.g. '0', '0-4', '0,2,4'). Matching programmer/array numbering.")

    parser.add_argument("-o", "--output", help="Output file path. Defaults to stdout (standard output).")
    parser.add_argument("--temp-file", help="Temporary file path to use during processing. If not provided, a .tmp suffix is added to the output path.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum new tokens to generate per page (default: 4096).")

    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"Error: File not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    # Automatic device selection
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "mps" else torch.bfloat16

    output_path = args.output
    temp_path = None
    output_stream = None
    overall_start_time = time.time()
    total_tokens = 0

    try:
        if output_path:
            temp_path = args.temp_file or f"{output_path}.tmp"
            output_stream = open(temp_path, "w")
        else:
            output_stream = sys.stdout

        print(f"Loading Model... (Device: {device}, Dtype: {dtype})", file=sys.stderr)
        try:
            model = LightOnOcrForConditionalGeneration.from_pretrained("lightonai/LightOnOCR-2-1B", torch_dtype=dtype).to(device)
            processor = LightOnOcrProcessor.from_pretrained("lightonai/LightOnOCR-2-1B")
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            sys.exit(1)

        with pdfium.PdfDocument(args.pdf) as pdf:
            num_pages = len(pdf)

            if args.indices:
                target_indices = parse_pages(args.indices, num_pages, offset=0)
            elif args.pages:
                target_indices = parse_pages(args.pages, num_pages, offset=1)
            else:
                target_indices = list(range(num_pages))

            print(f"Processing {len(target_indices)} pages from '{args.pdf}'...", file=sys.stderr)

            for i, page_idx in enumerate(target_indices):
                print(f"[{i+1}/{len(target_indices)}] Rendering page {page_idx + 1} (index {page_idx})...", file=sys.stderr, end="", flush=True)

                page = pdf[page_idx]
                bitmap = page.render(scale=2)
                pil_image = bitmap.to_pil()

                print(" Done.", file=sys.stderr)

                conversation = [{"role": "user", "content": [{"type": "image", "image": pil_image}]}]

                inputs = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}

                print(f"      Generating OCR output...", file=sys.stderr, end="", flush=True)
                start_gen = time.time()

                output_stream.write(f"<!-- PAGE {page_idx + 1} -->\n")
                output_stream.flush()

                streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=args.max_tokens)
                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()

                spinner = ["|", "/", "-", "\\"]
                token_count = 0
                for new_text in streamer:
                    output_stream.write(new_text)
                    output_stream.flush()
                    token_count += 1

                    # Animate a spinner and show token count on stderr
                    s_idx = token_count % len(spinner)
                    h_count = humanize_count(token_count)
                    print(f"\r      Generating OCR output... {spinner[s_idx]} ({h_count} tokens)", file=sys.stderr, end="", flush=True)

                total_tokens += token_count
                output_stream.write("\n\n")
                output_stream.flush()

                gen_duration = time.time() - start_gen
                h_count = humanize_count(token_count)
                print(f"\r      Generating OCR output... Done. ({gen_duration:.1f}s, {h_count} tokens)", file=sys.stderr)

        # If we successfully reached here and were writing to a file, prepend metadata and move it
        if output_path:
            output_stream.close()

            # Read back the partial results
            with open(temp_path, "r") as f:
                content = f.read()

            # Prepare metadata
            duration = time.time() - overall_start_time
            iso_date = datetime.datetime.now().astimezone().isoformat()
            rel_pdf = os.path.relpath(args.pdf)

            metadata = f"""---
Date: {iso_date}
PDF_File: {rel_pdf}
Page_Count: {len(target_indices)}
Token_Count: {total_tokens}
Duration: {duration:.1f}s
---

"""
            # Write metadata + content to the final destination
            with open(output_path, "w") as f:
                f.write(metadata)
                f.write(content)

            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            print(f"OCR results saved to: {output_path}", file=sys.stderr)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...", file=sys.stderr)
        if output_path and output_stream:
            try:
                output_stream.close()
            except:
                pass
            if temp_path and os.path.exists(temp_path):
                print(f"Note: Incomplete results kept in: {temp_path}", file=sys.stderr)
        sys.exit(130)
    finally:
        if output_stream and not output_stream.closed and output_stream != sys.stdout:
            output_stream.close()

    print("Done!", file=sys.stderr)

if __name__ == "__main__":
    main()
