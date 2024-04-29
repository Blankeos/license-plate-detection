import datetime
from typing import List

# Qt
from PySide6.QtWidgets import (QApplication, QListWidget, QLayout, QLineEdit, QMainWindow, QWidget, QHeaderView, QTableView, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QSizePolicy, QMenuBar, QMenu)
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtCore import Qt, QTimer, QSize

# Import for OpenCV
import cv2
import numpy as np

# Torch
import torch

# Firestore
from LicensePlateDetection.database.firestore import getAllDetections, getAllRegisteredLicensePlates, insertNewDetection, insertNewRegisteredLicensePlate

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

        # Table of the Recently Detected
        self.right_layout.addWidget(QLabel("<h4>License Detection Logs</h4>"))

        # - Table
        self.logs_table_widget = QTableView()
        self.logs_table_widget.setMaximumWidth(500)
        self.logs_table_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.license_plates_model = LicensePlatesTableModel(self, [], ["License Plate", "Date Detected", "Registered"])
        self.logs_table_widget.setModel(self.license_plates_model)
        self.logs_table_widget.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.right_layout.addWidget(self.logs_table_widget)

        # Table for Registered Detections   
        self.right_layout.addWidget(QLabel("<h4>Registered License Plates</h4>"))
        
        # - Input
        add_new_registration_layout = QHBoxLayout()

        self.registered_table_input = QLineEdit()
        self.registered_table_input.setMaximumWidth(200)
        self.registered_table_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.registered_table_input.setPlaceholderText("Enter License Plate")

        add_new_registered_button = QPushButton("Register New")
        add_new_registered_button.setMaximumWidth(150)
        add_new_registered_button.clicked.connect(self.addNewRegistration)

        add_new_registration_layout.addWidget(self.registered_table_input)
        add_new_registration_layout.addWidget(add_new_registered_button)
        self.right_layout.addLayout(add_new_registration_layout)

        # - Table
        self.registered_table_widget = QTableView()
        self.registered_table_widget.setMaximumWidth(500)
        self.registered_table_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.registered_license_plates_model = LicensePlatesTableModel(self, [], ["License Plate", "Date Registered"])
        self.registered_table_widget.setModel(self.registered_license_plates_model)
        self.registered_table_widget.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.right_layout.addWidget(self.registered_table_widget)

        # Fin: Add the RightLayout
        self.main_layout.addLayout(self.right_layout)

        # ----------------------------------------------------------------------
        # Data Intialization
        # ----------------------------------------------------------------------
        self.recentlyDetected = HashQueue()

        # Get existing data from the database.
        licensePlateEntities = getAllDetections()
        registeredLicensePlateEntities = getAllRegisteredLicensePlates()

        # Add it DB data to the TableView.
        rows_to_tablemodel = [(entity.license_plate, formatDate(entity.date_detected), entity.registered) for entity in licensePlateEntities] 
        self.license_plates_model.addRows(rows_to_tablemodel)

        rows_to_registeredtablemodel = [(entity.license_plate, formatDate(entity.date_registered)) for entity in registeredLicensePlateEntities]
        self.registered_license_plates_model.addRows(rows_to_registeredtablemodel)
        
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

        # 1. Add to the database (Can be improved to make this asynchronous maybe?)
        new_detection = insertNewDetection(license_plate, datetime_detected)

        # 2. Insert Row
        self.license_plates_model.insertRows(0, [(new_detection.license_plate, new_detection.formatted_date_detected, new_detection.registered)])
        
        # 3. Add to Recently Detected to avoid spamming
        self.recentlyDetected.add(license_plate)

        
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

    def addNewRegistration(self):
        if (self.registered_table_input.text() == ""):
            return
        
        license_plate = self.registered_table_input.text()
        registered_date = datetime.datetime.now()
        
        # 1. Insert license plate.
        insertNewRegisteredLicensePlate(license_plate, registered_date)

        # 2. Add to table view.
        self.registered_license_plates_model.insertRows(0, [(license_plate, formatDate(registered_date))])

        # 3. Clear text
        self.registered_table_input.setText("")