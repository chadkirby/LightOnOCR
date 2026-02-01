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

def format_ranges(nums):
    """Converts a list of numbers into a concise range string (e.g. [1, 2, 3, 5] -> '1-3, 5')."""
    if not nums:
        return ""
    nums = sorted(list(set(nums)))
    ranges = []
    if not nums:
        return ""

    start = nums[0]
    for i in range(len(nums)):
        if i + 1 == len(nums) or nums[i+1] != nums[i] + 1:
            if start == nums[i]:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{nums[i]}")
            if i + 1 < len(nums):
                start = nums[i+1]
    return ", ".join(ranges)

def humanize_count(n):
    """Humanizes an integer count (e.g. 1500 -> 1.5k, 1250000 -> 1.25m)."""
    if n < 1000:
        return str(n)
    if n < 1000000:
        return f"{n / 1000:.2f}k"
    return f"{n / 1000000:.3f}m"

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
    parser.add_argument("--resume", action="store_true", help="Resume OCR from an existing output file by skipping already-processed pages.")

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

            requested_pages_str = args.pages or args.indices or f"1-{num_pages}"
            included_pages = []
            total_tokens = 0
            overall_start_time = time.time()
            base_content = ""

            # Resume logic: check existing output file and skip completed pages
            if args.resume and output_path and os.path.exists(output_path):
                print(f"Checking existing output for progress: {output_path}", file=sys.stderr)
                try:
                    with open(output_path, "r") as f:
                        file_text = f.read()

                    if file_text.startswith("---"):
                        parts = file_text.split("---", 2)
                        if len(parts) >= 3:
                            header = parts[1]
                            base_content = parts[2].lstrip()

                            # Extract OCR_Pages
                            m_pages = re.search(r"OCR_Pages:\s*(.*)", header)
                            if m_pages:
                                ocr_pages_str = m_pages.group(1).strip()
                                completed_indices = parse_pages(ocr_pages_str, num_pages, offset=1)
                                target_indices = [idx for idx in target_indices if idx not in completed_indices]
                                included_pages = [p + 1 for p in sorted(completed_indices)]
                                print(f"  Found {len(completed_indices)} completed pages. {len(target_indices)} pages remaining.", file=sys.stderr)

                            # Extract cumulative stats
                            m_tokens = re.search(r"Token_Count:\s*(\d+)", header)
                            if m_tokens:
                                total_tokens = int(m_tokens.group(1))

                            m_dur = re.search(r"Duration:\s*([\d\.]+)s", header)
                            if m_dur:
                                prev_dur = float(m_dur.group(1))
                                overall_start_time -= prev_dur
                except Exception as e:
                    print(f"  Warning: Could not parse existing output for resume: {e}", file=sys.stderr)

            if not target_indices:
                print("All requested pages are already processed. Done!", file=sys.stderr)
                if output_path and temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
                return

            # If we have base content (from resume), write it to the temp file first
            if base_content:
                output_stream.write(base_content)
                output_stream.flush()

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
                included_pages.append(page_idx + 1)
                output_stream.write("\n\n")
                output_stream.flush()

                gen_duration = time.time() - start_gen
                h_count = humanize_count(token_count)
                print(f"\r      Generating OCR output... Done. ({gen_duration:.1f}s, {h_count} tokens)", file=sys.stderr)

                # Update the final output file incrementally after each page
                if output_path:
                    # Flush the temp file to ensure we read everything
                    output_stream.flush()

                    with open(temp_path, "r") as f:
                        content = f.read()

                    current_duration = time.time() - overall_start_time
                    iso_date = datetime.datetime.now().astimezone().isoformat()
                    rel_pdf = os.path.relpath(args.pdf)
                    included_pages_str = format_ranges(included_pages)

                    metadata = f"""---
Date: {iso_date}
PDF_File: {rel_pdf}
Total_Pages: {num_pages}
OCR_Pages: {included_pages_str}
Token_Count: {total_tokens}
Duration: {current_duration:.1f}s
---

"""
                    # Write to a secondary temp file then move atomically to final destination
                    interim_temp = f"{output_path}.new"
                    with open(interim_temp, "w") as f:
                        f.write(metadata)
                        f.write(content)
                    os.replace(interim_temp, output_path)

            # Cleanup temp file on full success
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
