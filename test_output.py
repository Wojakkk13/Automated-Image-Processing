from pathlib import Path
import pytest

# Change this if your output folder is different
output_dir = Path(__file__).parent / "output"

def test_images_processed():
    """
    Verify that at least one processed image exists in output folder.
    Works when images are saved directly into output/ (no subfolders).
    """
    # Only look directly in output/
    files = [p for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    print(f"[test] Looking in: {output_dir}")
    print(f"[test] Found {len(files)} image(s): {[f.name for f in files]}")

    # Must have at least one processed image
    assert len(files) > 0, f"No processed images found in {output_dir}! Make sure main.py ran correctly."
