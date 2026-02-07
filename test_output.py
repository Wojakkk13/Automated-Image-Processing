from pathlib import Path

output_dir = Path("output")  # root/output

def test_images_processed():
    files = list(output_dir.glob("*.jpg")) + \
            list(output_dir.glob("*.jpeg")) + \
            list(output_dir.glob("*.png"))
    print(f"[test] Looking in: {output_dir}")
    print(f"[test] Found {len(files)} image(s): {[f.name for f in files]}")
    assert len(files) > 0, f"No processed images found in {output_dir}! Make sure main.py ran correctly."
