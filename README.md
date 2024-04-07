<h1>License Plate Detection</h1>

A super simple prototype of a license plate detection application.

Built with:

- PySide6 - GUI
- OpenCV - For image transformations
- YoloV5 - For detections (Might upgrade to YoloV8)
- EasyOCR - for OCR detection
- SQLite - Data storage

<h2>Table of Contents</h2>

- [📝 Running Requirements](#%F0%9F%93%9D-running-requirements)
- [🏁 Getting Started](#%F0%9F%8F%81-getting-started)
- [Resources](#resources)
- [Notes](#notes)

## 📝 Running Requirements

- Install [Git](https://git-scm.com/downloads) - On Mac it's already installed.
- Install [Python 3.12.2](https://www.python.org/downloads/)

## 🏁 Getting Started

Open your terminal and do the following:

1. Git clone and go to directory

```sh
git clone https://github.com/Blankeos/license-plate-detection
cd license-plate-detection
```

2. Create Virtual Env:

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

4. Download the model [here](https://github.com/KALYAN1045/Automatic-Number-Plate-Recognition-using-YOLOv5/blob/main/Weights/best.pt). Click on [•••] > Download. Copy paste `best.pt` into the `license-plate-detection` folder (should be the same directory as your `main.py`)

5. Run the app (It will take a while the first run)

```sh
python main.py
```

## Resources

https://realpython.com/python-pyqt-gui-calculator/

## Notes
