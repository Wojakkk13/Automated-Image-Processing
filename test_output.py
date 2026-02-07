import os
import pytest

output_dir = "output"

def test_output_folder_exists():
    assert os.path.isdir(output_dir), "Output folder missing!"

def test_images_processed():
    files = [f for f in os.listdir(output_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    assert len(files) > 0, "No processed images found!"
