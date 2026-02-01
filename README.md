# LightOnOCR-2-1B CLI

Fast, local PDF OCR for Mac (Apple Silicon optimized). Converts PDF pages into clean, naturally ordered Markdown text using the `lightonai/LightOnOCR-2-1B` model.

## Features

- **Apple Silicon Optimized**: Uses `mps` (Metal Performance Shaders) for fast, local inference.
- **Atomic & Incremental Saving**: Updates your output file page-by-page. Never lose progress and never end up with a corrupt file.
- **Rich Metadata**: Automatically prepends YAML frontmatter with processing time, token counts, and page tracking.
- **Human-Friendly CLI**: Supports both 1-based page numbers (`--pages 1-5`) and 0-based indices (`--indices 0-4`).
- **Live Feedback**: Real-time generation feedback via an animated ASCII spinner and humanized token counts.

## Installation

Install using `uv`:

```bash
uv tool install lightonocr
```

Or for development (editable mode):

```bash
uv tool install --editable .
```

## Usage

```bash
# Process all pages and save to a file
lighton-ocr document.pdf -o output.md

# Process specific pages (1-based)
lighton-ocr document.pdf --pages 1,3,5-10

# Process specific indices (0-based)
lighton-ocr document.pdf --indices 0,2,4-9
```

## Output Format

The tool generates Markdown files starting with a YAML header:

```yaml
---
Date: 2026-02-01T14:13:22-08:00
PDF_File: financials/sept_2021.pdf
Total_Pages: 22
OCR_Pages: 1-5
Token_Count: 1250
Duration: 25.8s
---

<!-- PAGE 1 -->
# Document Title
...
```

## Troubleshooting

- **Performance**: OCR generation can take 15-45 seconds per page depending on content density and your Mac's hardware.
- **Stuck Generation**: Use `--max-tokens` to limit the generation length if the model gets stuck in a loop on complex tables.
- **Memory**: The model requires approximately 4-6GB of system memory.

## License

MIT
