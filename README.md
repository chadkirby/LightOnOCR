# LightOnOCR CLI

A fast, local PDF OCR tool for Mac, optimized for Apple Silicon. This tool uses the `lightonai/LightOnOCR-2-1B` vision-language model to convert PDF pages into clean, naturally ordered Markdown text.

## Features

- **Fast & Local**: Runs entirely on your machine (no API calls).
- **Apple Silicon Optimized**: Leverages MPS (Metal Performance Shaders) for high-speed inference on MacBooks.
- **Smart Formatting**: Outputs clean Markdown with structural elements (headers, lists, tables).
- **Dual Numbering**: Supports both 1-based page numbers (Human style) and 0-based page indices (Programmer style).
- **Flexible Output**: Print to stdout or save directly to a file.

## Installation

This project is designed to be used with [uv](https://github.com/astral-sh/uv).

### 1. Install via `uv tool` (Recommended)

To make the `lighton-ocr` command available system-wide:

```bash
uv tool install --editable .
```

*Note: The `--editable` flag allows any changes made to the source code in this directory to be immediately reflected in the global command.*

### 2. Manual Run

Alternatively, run it using `uv` from within the project directory:

```bash
uv run run_ocr.py [PDF_PATH]
```

## Usage

```bash
# Process all pages and save to a file
lighton-ocr document.pdf -o output.md

# Process specific 1-based page numbers (e.g., Pages 1, 3, and 5 through 10)
lighton-ocr document.pdf --pages 1,3,5-10

# Process specific 0-based indices (Programmer style)
lighton-ocr document.pdf --indices 0,2,4-9
```

### Options

- `-p, --pages`: Specify 1-based page numbers.
- `-i, --indices`: Specify 0-based page indices.
- `-o, --output`: Path to save the OCR results (defaults to stdout).
- `--help`: Show detailed help and examples.

## License

Apache 2.0 (matching the model's license).
