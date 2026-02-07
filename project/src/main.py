"""Command-line entry for the image processor.

Usage:
    python -m src.main

This module only sets input/output directories and calls the processor.
"""
from pathlib import Path
import sys

try:
    from . import processor
    # Allow running this file directly: adjust sys.path so `src` package can be imported
except Exception:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import src.processor as processor


def main() -> None:
    project_root = Path.cwd()
    input_dir = project_root / "input"
    output_dir = project_root / "output"

    print(f"[main] Input directory: {input_dir}")
    print(f"[main] Output directory: {output_dir}")

    processor.process_all(input_dir, output_dir)


if __name__ == "__main__":
    main()
