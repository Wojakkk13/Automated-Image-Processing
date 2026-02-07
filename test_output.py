from pathlib import Path

# Define the output folder relative to this test file
output_dir = Path(__file__).resolve().parents[0] / "output"

def test_images_processed():
    """
    Verify that at least one processed image exists in output or its subfolders.
    Works with images in timestamped subfolders like output/run_YYYYMMDD_HHMMSS.
    """
    # recursively find all images
    files = list(output_dir.rglob("*.jpg")) + \
            list(output_dir.rglob("*.jpeg")) + \
            list(output_dir.rglob("*.png"))

    # Debug info
    print(f"[test] Looking in: {output_dir}")
    print(f"[test] Found {len(files)} image(s): {[f.name for f in files]}")

    # Check that we have at least one image
    assert len(files) > 0, f"No processed images found in {output_dir}! Make sure main.py ran correctly."
