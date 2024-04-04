import datetime
from typing import List
from PySide6.QtWidgets import (QApplication, QListWidget, QMainWindow, QWidget, QHeaderView, QTableView, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QSizePolicy, QMenuBar, QMenu)
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtCore import Qt, QTimer, QSize

# Import for OpenCV
import cv2
import numpy as np
import json

# Torch
import torch
from pathlib import Path

# src
from LicensePlateDetection.tablemodels.licenseplates_tablemodel import LicensePlatesTableModel
from LicensePlateDetection.utils.formatDate import formatDate
from LicensePlateDetection.utils.validation import validate_license_plate
from LicensePlateDetection.widgets.flow_layout import FlowLayout

# MODEL_PATH = Path("./cgjj_best.pt").resolve()
MODEL_PATH = Path("./best.pt").resolve()
YOLO_PATH = Path("./yolov5").resolve()

# Model Instance
model = torch.hub.load(YOLO_PATH.as_posix(), 'custom', MODEL_PATH.as_posix(), source='local')  # local repo
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s') # remote repo using a pre-trained model.

# OCR
from LicensePlateDetection.utils.ocrReader.base_ocr import BaseOCR
from LicensePlateDetection.utils.ocrReader.easy_ocr import EasyOCR
# from utils.ocrReader.pytesseract_ocr import PytesseractOCR
# from utils.ocrReader.tesserocr_ocr import TesserOCR



class LicensePlateDetection(QMainWindow):
    def __init__(self):
        super().__init__()

        # CONFIG
        self.ocrReader: BaseOCR = EasyOCR()
        self.detect_interval_counter = 1
        self.DETECT_INTERVAL = 20

        # Set up the main window
        self.setWindowTitle("License Plate Detection")
        self.setMinimumSize(QSize(980, 500))

        # Create the menu bar
        menubar = QMenuBar(self)
        menubar.setNativeMenuBar(True)
        self.setMenuBar(menubar)

        # Create the Quit menu
        quit_menu = QMenu("\0Options", self)
        menubar.addMenu(quit_menu)

        # Create the Quit action
        quit_action = QAction(" &Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        quit_menu.addAction(quit_action)

        ### -----

        # Create main layout (horizontal)
        self.main_layout = QHBoxLayout()
        self.setCentralWidget(QWidget())
        self.centralWidget().setLayout(self.main_layout)

        # Image display area (left side)
        self.left_layout = QVBoxLayout()
        self.image_label = QLabel()
        # self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.startButton = QPushButton("Detecting ON")
        self.startButton.setStyleSheet("background-color: green;")
        self.startButton.clicked.connect(self.toggleDetecting)

        self.left_layout.addWidget(self.image_label)
        self.left_layout.addStretch()
        self.left_layout.addWidget(self.startButton)
        self.main_layout.addLayout(self.left_layout)

        # ----------------------------------------------------------------------

        # Right Layout (right side)
        self.right_layout = QVBoxLayout()
        

        self.currentDetectionsHeader = QLabel("<h4>Current Detections</h4>")
        self.right_layout.addWidget(self.currentDetectionsHeader)

        self.right_flow_layout = FlowLayout()
        # flow_layout.addWidget(QPushButton("Short"))
        # flow_layout.addWidget(QPushButton("Longer"))
        # flow_layout.addWidget(QPushButton("Different text"))
        # flow_layout.addWidget(QPushButton("More text"))
        # flow_layout.addWidget(QPushButton("Even longer button text"))

        self.right_layout.addLayout(self.right_flow_layout)

        # Currently Detected
        self.right_layout.addWidget(QLabel("<h4>License Detection Logs</h4>"))

        # Table of the Recently Detected
        self.table_widget = QTableView()
        self.table_widget.setMaximumWidth(500)
        self.license_plates_model = LicensePlatesTableModel(self, [("8192 DAG", formatDate(datetime.datetime.now() ) ) ], ["License Plate", "Date Detected"])
        self.table_widget.setModel(self.license_plates_model)
        self.table_widget.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.right_layout.addWidget(self.table_widget)
        self.license_plates_model.addRows([("CARLO 123", formatDate(datetime.datetime.now()) )])
        self.license_plates_model.insertRows(0, [("TEST 0812", formatDate(datetime.datetime.now()) ), ("TESLA", formatDate(datetime.datetime.now()) )])

        # self.list_widget = QListWidget()
        # self.list_widget.addItems(["Apple", "Grapes", "Banana", "Nice"])
        # self.list_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        # self.list_widget.setMaximumHeight(200)
        # self.right_layout.addWidget(self.list_widget)

        # self.right_layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetFixedSize)
        # self.right_layout.setAlignment(Qt.AlignmentFlag.AlignBaseline)
        # self.right_layout.setContentsMargins(20,20,20,20)

        # sizer_layout = QVBoxLayout(); sizer_layout.setContentsMargins(200, 0, 0, 200); 
        # sizer_layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetFixedSize)
        # self.right_layout.addLayout(sizer_layout)

        # text_label = QLabel("This is some text information.")
        # # text_label.setWordWrap(True)
        # text_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # self.right_layout.addWidget(text_label) # ADD
        

        # another_label = QLabel("This is another text")
        # another_label.setWordWrap(True)
        # another_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # self.right_layout.addWidget(another_label) # ADD
        # self.text_label.setStyleSheet('background: yellow')

        # fruit_list_box = QVBoxLayout()
        # fruit_list_box.setSizeConstraint(QVBoxLayout.SizeConstraint.SetFixedSize)
        # fruit_labels = ["Apple", "Grapes", "Oranges"]
        # for label in fruit_labels:
        #     label_widget = QLabel(label)
        #     label_widget.setWordWrap(True)
        #     label_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        #     fruit_list_box.addWidget(label_widget)
        # self.right_layout.addLayout(fruit_list_box) # ADD

        self.right_layout.addStretch()



        self.main_layout.addLayout(self.right_layout) # ADD

        ###----
        self.shown = False


        # Setup webcam and timer
        self.cap = cv2.VideoCapture(0)  # Change index for multiple cameras
        self.is_grayscale = False
        self.is_detecting = True
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(0)  # Update every 10 milliseconds

    def setCurrentDetectionsCount(self, count: int):
        self.currentDetectionsHeader.setText(f"<h4>Current Detections ({count})</h4>")
        
    def update_frame(self):
        ret, frame = self.cap.read()

        # Only perform the detection after a certain interval (for performance)
        if (self.is_detecting and self.detect_interval_counter >= self.DETECT_INTERVAL):
            results, crops, boundingBoxes = self.detect(frame)
            self.setCurrentDetectionsCount(len(boundingBoxes))
            self.displayCroppedImages(crops)

            results.render()
            frame = results.ims[0]

            self.detect_interval_counter = 1
        
        if ret:
            # Convert frame to RGB for PyQt
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply grayscale if enabled
            if self.is_grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                # Must convert back to color for PyQt (When gray, it loses its channel in CV)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            # Resize frame so it fits the window
            scale_factor = 1.0

            if self.width() < frame.shape[1] or self.height() < frame.shape[0]:
                scale_factor = min(self.left_layout.geometry().width() / frame.shape[1], self.left_layout.geometry().height() / frame.shape[0])
            frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

            # Display frame
            height, width = frame.shape[:2]

            # Convert frame to QImage
            qImg = QImage(frame, width, height, int(frame.strides[0]), QImage.Format.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qImg))

        self.detect_interval_counter += 1
    # Display cropped detection if available

    # Draw on Image
    def detect(self, image: np.ndarray) -> tuple[np.ndarray, List[dict], int]:
        results = model(image)

        # Get cropped values.
        crops = results.crop(save=False)
        
        return (results, crops, results.xyxy[0])

    # def readLicensePlates(self, crops: list[dict]):
    #     for crop in crops:
    #         gray = cv2.cvtColor(crop['im'], cv2.COLOR_BGR2GRAY)


    #     pass
    def displayCroppedImages(self, crops: list[dir]):
        # 1. Delete All
        while (child := self.right_flow_layout.takeAt(0)) != None:
            child.widget().deleteLater()

        # 2. Add the new images
        for crop in crops:
            # Preprocess the image

            # 1. Convert the image to grayscale (For label)
            gray = cv2.cvtColor(crop['im'], cv2.COLOR_BGR2GRAY)
            # 2. Increase the contrast of the gray image
            gray = cv2.convertScaleAbs(gray, alpha=0.5, beta=50)
            # 3. Threshold the gray image
            # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # gray = 255 - thresh


            # ---- UI -----
            v_widget = QWidget()
            v_layout = QVBoxLayout()
            v_widget.setLayout(v_layout)

            # Create label with image
            crop_label = QLabel()
            # Resize crop to 128x128
            # resized_crop = cv2.resize(crop['im'], (128, 128))
            resized_crop = cv2.resize(gray, (128, 128))
            resized_crop = cv2.cvtColor(resized_crop, cv2.COLOR_GRAY2RGB)

            # Convert to QImage
            im = QImage(resized_crop, 128, 128, QImage.Format_RGB888)
            # Add label to layout
            crop_label.setPixmap(QPixmap.fromImage(im))

            ocr_text, score = self.ocrReader.read(gray)

            label_text = f"{crop['label']}\n{ocr_text} {score:.2f}" if ocr_text != None else crop['label']
            
            label_widget = QLabel(label_text)
            if (validate_license_plate(ocr_text)):
                label_widget.setStyleSheet("color: green;")
            else:
                label_widget.setStyleSheet("color: red;")

            v_layout.addWidget(crop_label)
            v_layout.addWidget(label_widget)

            self.right_flow_layout.addWidget(v_widget)

    def toggle_grayscale(self):
        self.is_grayscale = not self.is_grayscale

    def toggleDetecting(self):
        if self.is_detecting:
            # self.timer.stop()
            self.is_detecting = False
            self.startButton.setText('Detecting OFF')
            self.startButton.setStyleSheet("background-color: red;")
        else:
            # self.timer.start()
            self.is_detecting = True
            self.startButton.setText('Detecting ON')
            self.startButton.setStyleSheet("background-color: green;")