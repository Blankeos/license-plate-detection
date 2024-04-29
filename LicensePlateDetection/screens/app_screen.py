import datetime
from typing import List

# Qt
from PySide6.QtWidgets import (QApplication, QListWidget, QMainWindow, QWidget, QHeaderView, QTableView, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QSizePolicy, QMenuBar, QMenu)
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtCore import Qt, QTimer, QSize

# Import for OpenCV
import cv2
import numpy as np

# Torch
import torch

# Firestore
from LicensePlateDetection.database.firestore import getAllLicensePlates, insertNewLicensePlate

# src
from LicensePlateDetection.tablemodels.licenseplates_tablemodel import LicensePlatesTableModel
from LicensePlateDetection.utils.datastructure.hashqueue import HashQueue
from LicensePlateDetection.utils.formatDate import formatDate
from LicensePlateDetection.utils.validation import validate_license_plate
from LicensePlateDetection.widgets.flow_layout import FlowLayout

# MODEL_PATH
from pathlib import Path
MODEL_PATH = Path("./best.pt").resolve() # A yolov5 model
YOLO_PATH = Path("./yolov5").resolve()

# Model Instance
# 1. Load the model
model = torch.hub.load(YOLO_PATH.as_posix(), 'custom', MODEL_PATH.as_posix(), source='local', force_reload=True)  # local repo

# 2. Change to mps or cuda
if (torch.backends.mps.is_available()):
    model.to(torch.device("mps")) # ✅ This is what worked.
elif (torch.cuda.is_available()):
    model.to(torch.device("cuda"))

# Check if running on cuda/mps/cpu
print("[Model] is cuda (nvidia GPU, faster)??", next(model.parameters()).is_cuda) # returns boolean
print("[Model] is cpu (no GPU, slower)??", next(model.parameters()).is_cpu) # returns boolean
print("[Model] is mps (mac GPU, faster)??", next(model.parameters()).is_mps) # returns boolean

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
        self.DETECT_INTERVAL = 50

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

        quit_action = QAction(" &Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        quit_menu.addAction(quit_action)

        # ----------------------------------------------------------------------
        # Main Layout
        # ----------------------------------------------------------------------

        # Create main layout (horizontal)
        self.main_layout = QHBoxLayout()
        self.setCentralWidget(QWidget())
        self.centralWidget().setLayout(self.main_layout)


        # ----------------------------------------------------------------------
        # Left Layout (Image and Start Button)
        # ----------------------------------------------------------------------

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
        # Right Layout (Detections & Detection Logs)
        # ----------------------------------------------------------------------

        self.right_layout = QVBoxLayout()

        # Current Detections
        self.currentDetectionsHeader = QLabel("<h4>Current Detections</h4>")
        self.right_layout.addWidget(self.currentDetectionsHeader)

        self.right_flow_layout = FlowLayout()
        self.right_layout.addLayout(self.right_flow_layout)

        # Currently Detected
        self.right_layout.addWidget(QLabel("<h4>License Detection Logs</h4>"))

        # Table of the Recently Detected
        self.table_widget = QTableView()
        self.table_widget.setMaximumWidth(500)
        self.table_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.table_widget.resize
        self.license_plates_model = LicensePlatesTableModel(self, [], ["License Plate", "Date Detected"])
        self.table_widget.setModel(self.license_plates_model)
        self.table_widget.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.right_layout.addWidget(self.table_widget)

        self.recentlyDetected = HashQueue()

        self.main_layout.addLayout(self.right_layout)

        # ----------------------------------------------------------------------
        # Data Intialization
        # ----------------------------------------------------------------------
        self.recentlyDetected = HashQueue()

        # Get existing data from the database.
        licensePlateEntities = getAllLicensePlates() 

        # Add it DB data to the TableView.
        rows_to_tablemodel = [(entity.license_plate, formatDate(entity.date_detected)) for entity in licensePlateEntities] 
        self.license_plates_model.addRows(rows_to_tablemodel)
        
        # ----------------------------------------------------------------------
        # Update Loop Initialization
        # ----------------------------------------------------------------------

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
            # Resize crop to 128x128 (For display only)
            resized_crop = cv2.resize(gray, (128, 128))
            resized_crop = cv2.cvtColor(resized_crop, cv2.COLOR_GRAY2RGB)

            # Convert to QImage
            im = QImage(resized_crop, 128, 128, QImage.Format_RGB888)
            # Add label to layout
            crop_label.setPixmap(QPixmap.fromImage(im))

            # Perform OCR
            ocr_text, score = self.ocrReader.read(gray)

            label_text = f"{crop['label']}\n{ocr_text} {score:.2f}" if ocr_text != None else crop['label']
            
            label_widget = QLabel(label_text)

            if (validate_license_plate(ocr_text)):
                label_widget.setStyleSheet("color: green;")
                self.addLicensePlateToTable(ocr_text)
            else:
                label_widget.setStyleSheet("color: red;")

            v_layout.addWidget(crop_label)
            v_layout.addWidget(label_widget)

            self.right_flow_layout.addWidget(v_widget)

    def addLicensePlateToTable(self, license_plate: str):
        if (self.recentlyDetected.has(license_plate)):
            return
        
        
        # Can now add!
        datetime_detected = datetime.datetime.now()
        formatted_datetime_detected = formatDate(datetime_detected)

        # 1. Insert Row
        self.license_plates_model.insertRows(0, [(license_plate, formatted_datetime_detected)])
        # 2. Add to Recently Detected to avoid spamming
        self.recentlyDetected.add(license_plate)       
        # 3. Add to the database
        insertNewLicensePlate(license_plate, datetime_detected)
        
    def toggle_grayscale(self):
        """
        DeprecationWarning: This method will be deprecated.
        """
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