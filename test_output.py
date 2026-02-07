from pathlib import Path

output_dir = Path("output")

def test_images_processed():
    # recursively find all image files in output and subfolders
    files = list(output_dir.rglob("*.jpg")) + \
            list(output_dir.rglob("*.jpeg")) + \
            list(output_dir.rglob("*.png"))

    assert len(files) > 0, f"No processed images found in {output_dir}! Make sure main.py ran correctly."
