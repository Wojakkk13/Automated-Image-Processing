from pathlib import Path
from datetime import datetime
import sys

try:
    from . import processor
except Exception:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import src.processor as processor

def main() -> None:
    # Repo root
    script_dir = Path(__file__).resolve().parent.parent
    input_dir = script_dir / "input"
    output_base = script_dir / "output"

    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[main] Input directory: {input_dir}")
    print(f"[main] Output directory: {output_dir}")

    processor.process_all(input_dir, output_dir)

if __name__ == "__main__":
    main()