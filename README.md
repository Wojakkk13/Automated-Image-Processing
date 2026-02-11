# Automated Image Processing

## Project Overview
Automated Image Processing is a Python-based image processing application that demonstrates the integration of computer vision techniques with DevOps automation. The system processes images using multiple image processing operations and ensures reliability through automated testing and Continuous Integration (CI) using GitHub Actions.

---

## Learning Objectives
- Apply image processing techniques using Python and OpenCV  
- Implement automated testing using PyTest  
- Configure and utilize a Continuous Integration (CI) pipeline with GitHub Actions  
- Understand DevOps workflows and automation concepts  

---

## Tools and Technologies
- Python 3  
- OpenCV (opencv-python)  
- NumPy  
- PyTest  
- Git and GitHub  
- GitHub Actions  

---

## Installation

### 1. Clone the repository:
- git clone https://github.com/Wojakkk13/Automated-Image-Processing.git
### 2. Navigate to project folder:
- cd Automated-Image-Processing

### 3. Install dependencies:
- The pipeline will automatically install all the needed dependencies

---

## Run Locally
```
python -m project.src.main
```
---

## Project Structure 
```text
Automated-Image-Processing/
│
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions pipeline and Project Dependencies
│
├── input/                        #  Drop images here (trigger CI)
│   └── sample.jpg
│
├── output/                       #  Processed images appear here
│   └── sample_thermal.jpg       
│
├── project/
│   └── models/
│       └── deploy.prototxt
│   └── src/
│       ├── __init__.py
│       ├── main.py              # Entry point
│       └── processor.py        # Image processing logic
│
├── test_output.py               # Pytest validations
│       
│
└── README.md

```
---

## CI/CD Pipeline (GitHub Actions)

The CI pipeline automatically runs when code or images are pushed.

## Steps:
1. Checks out the repository
2. Sets up Python 3.11
3. Installs dependencies
4. Runs main.py to process images
5. Executes PyTest validation
6. Uploads output images as artifacts


---

## Implemented Image Processing Features

### Thermal Heat Mapping
Applies a thermal color map to visualize intensity variations within the image.

### Morphological Gradient
Highlights object boundaries using morphological image processing techniques.

### Half-Image Mirror
Creates a symmetrical image by mirroring one half of the original image.

### Unsharp Masking (Image Sharpening)
Enhances image details by emphasizing high-frequency components.

### Face Blurring
Detects human faces using Haar Cascade classifiers and applies blurring to protect privacy.

---

## How to Use the Automated Image Processing System

### 1. Upload/Input Images
- Place the image or images you want to process inside the `input` folder.  
- Supported image formats include `.jpg`, `.png`, and `.jpeg`.

### 2. Program Start
- Once you upload an image or images, it will execute the script.
- The system automatically processes all images found in the `input` folder.

### 3. Automated Processing
Each input image undergoes the following operations:
- Thermal Heat Mapping  
- Morphological Gradient  
- Half-Image Mirror  
- Unsharp Masking (Image Sharpening)  
- Face Blurring (if faces are detected)

### 4. Output Results
- All processed images are automatically saved in the `output` folder.  
- Each image processing feature generates its own output file for easy comparison and validation.

### 5. Automation and Continuous Integration
- Automated tests are executed using PyTest.  
- Every push or pull request triggers the GitHub Actions CI pipeline to ensure code quality and reliability.

---
## Automated Testing

### PyTest validates:

- Output folder exists
- Images are generated
- Files are not empty
- Each input image produces outputs
- Images can be opened

This ensures processing reliability.

---

## How to Trigger Automation

Simply push images into the input/ folder.

### GitHub Actions will:
- Process images
- Run tests
- Generate outputs

---


## Team Members
- **Kenneth V. Sarmiento - Project Lead / Presenter**
- **Adrian Paul D. Gonzales - DevOps Engineer / Documenter**
- **Aaron Kurt G. Singson - Tester / Documenter**



