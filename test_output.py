import os
from pathlib import Path

# Root output folder
output_dir = Path("output")

def test_images_processed():
    """
    Check that at least one processed image exists in output subfolders.
    Works with timestamped folders like output/run_YYYYMMDD_HHMMSS.
    """
    # recursively find all image files in output/*
    files = list(output_dir.rglob("*.[jJ][pP][gG]")) + \
            list(output_dir.rglob("*.[pP][nN][gG]")) + \
            list(output_dir.rglob("*.[jJ][pP][eE][gG]"))

    assert len(files) > 0, f"No processed images found in {output_dir}! Make sure main.py ran correctly."
