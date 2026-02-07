from pathlib import Path
import sys

try:
    from . import processor
except Exception:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import src.processor as processor

def main() -> None:
    script_dir = Path(__file__).resolve().parents[2]  # Go up 2 levels to root
    input_dir = script_dir / "input"
    output_dir = script_dir / "output"


    # Ensure output folder exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[main] Input directory: {input_dir}")
    print(f"[main] Output directory: {output_dir}")

    processor.process_all(input_dir, output_dir)

if __name__ == "__main__":
    main()
