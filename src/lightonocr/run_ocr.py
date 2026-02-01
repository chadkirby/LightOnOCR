import torch
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
import pypdfium2 as pdfium
from PIL import Image
import os
import sys
import argparse
import re

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

    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"Error: File not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    # Automatic device selection
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "mps" else torch.bfloat16

    print(f"Loading Model... (Device: {device}, Dtype: {dtype})", file=sys.stderr)

    try:
        model = LightOnOcrForConditionalGeneration.from_pretrained("lightonai/LightOnOCR-2-1B", torch_dtype=dtype).to(device)
        processor = LightOnOcrProcessor.from_pretrained("lightonai/LightOnOCR-2-1B")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    output_stream = open(args.output, "w") if args.output else sys.stdout

    try:
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
                print(f"[{i+1}/{len(target_indices)}] Processing page {page_idx + 1} (index {page_idx})...", file=sys.stderr)

                page = pdf[page_idx]
                bitmap = page.render(scale=2)
                pil_image = bitmap.to_pil()

                conversation = [{"role": "user", "content": [{"type": "image", "image": pil_image}]}]

                inputs = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}

                output_ids = model.generate(**inputs, max_new_tokens=4096)
                generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
                output_text = processor.decode(generated_ids, skip_special_tokens=True)

                output_stream.write(f"<!-- PAGE {page_idx} -->\n")
                output_stream.write(output_text)
                output_stream.write("\n\n")
                output_stream.flush()

    finally:
        if args.output:
            output_stream.close()
            print(f"OCR results saved to: {args.output}", file=sys.stderr)

    print("Done!", file=sys.stderr)

if __name__ == "__main__":
    main()
