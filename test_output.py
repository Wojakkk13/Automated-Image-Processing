import pytest
from pathlib import Path

output_dir = Path("output")

def test_images_processed():
    """
    Verify that at least one processed image exists in output.
    Works with images directly in output folder.
    """
    files = list(output_dir.glob("*.jpg")) + \
            list(output_dir.glob("*.jpeg")) + \
            list(output_dir.glob("*.png"))

    print(f"[test] Looking in: {output_dir}")
    print(f"[test] Found {len(files)} image(s): {[f.name for f in files]}")

    assert len(files) > 0, f"No processed images found in {output_dir}! Make sure main.py ran correctly."
