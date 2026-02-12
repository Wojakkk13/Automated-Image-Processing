from pathlib import Path
import cv2

output_dir = Path("output")
input_dir = Path("input")

def test_images_processed():
    """
    Verify that at least one processed image exists in root/output/.
    """
    files = list(output_dir.glob("*.jpg")) + \
            list(output_dir.glob("*.jpeg")) + \
            list(output_dir.glob("*.png"))

    print(f"[test] Looking in: {output_dir}")
    print(f"[test] Found {len(files)} image(s): {[f.name for f in files]}")

    assert len(files) > 0, f"No processed images found in {output_dir}! Make sure main.py ran correctly."


def test_output_folder_exists():
    assert output_dir.exists(), "Output folder does not exist!"


def test_files_not_empty():
    for f in output_dir.glob("*"):
        if f.name == ".gitkeep":
            continue
        assert f.stat().st_size > 1000, f"{f.name} is empty or corrupted!"


def test_expected_filters_present():
    names = [f.name for f in output_dir.glob("*")]

    assert any("mirror" in n for n in names), "Mirror filter missing"
    assert any("thermal" in n for n in names), "Thermal filter missing"
    assert any("unsharp" in n for n in names), "Unsharp filter missing"


def test_images_loadable():
    for f in output_dir.glob("*.jpg"):
        img = cv2.imread(str(f))
        assert img is not None, f"{f.name} cannot be opened!"


def test_each_input_has_output():
    inputs = list(input_dir.glob("*"))
    outputs = list(output_dir.glob("*"))

    for i in inputs:
        assert any(i.stem in o.name for o in outputs), f"No output generated for {i.name}"