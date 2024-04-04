<h1>License Plate Detection</h1>

A super simple prototype of a license plate detection application.

Built with:

- PySide6 - GUI
- OpenCV - For image transformations
- YoloV5 - For detections (Might upgrade to YoloV8)
- Tesserocr - for OCR detection
- SQLite - Data storage

<h2>Table of Contents</h2>

- [📝 Running Requirements](#%F0%9F%93%9D-running-requirements)
- [🏁 Getting Started](#%F0%9F%8F%81-getting-started)
- [Resources](#resources)
- [Notes](#notes)

## 📝 Running Requirements

- Install [Python 3.12.2](https://www.python.org/downloads/)
- Install [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/Installation.html) - `brew install tesseract` (on Mac)

## 🏁 Getting Started

1. Create Virtual Env:

```sh
python -m venv .venv
```

2. Activate Virtual Env:

```sh
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

3. Install Deps

```sh
# Install deps for the project.
pip install -r requirements.txt

# Install deps for the model.
git clone https://github.com/ultralytics/yolov5
pip install -r yolov5/requirements.txt
```

## Resources

https://realpython.com/python-pyqt-gui-calculator/

## Notes

- If issues with tesserocr:
  - `pip uninstall tesserocr`
  - `pip install --no-binary :all: tesserocr`
