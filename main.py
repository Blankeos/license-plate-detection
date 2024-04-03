import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer


# Import for OpenCV
import cv2


class WebcamWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam Image Processing")

        # Create layout
        self.layout = QVBoxLayout()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(self.layout)

        # Create image label
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Grayscale button
        self.grayscale_btn = QPushButton("Grayscale")
        self.grayscale_btn.clicked.connect(self.toggle_grayscale)
        self.layout.addWidget(self.grayscale_btn)

        # Setup webcam and timer
        self.cap = cv2.VideoCapture(0)  # Change index for multiple cameras
        self.is_grayscale = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(0)  # Update every 10 milliseconds

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB for PyQt
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply grayscale if enabled
            if self.is_grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                # Must convert back to color for PyQt (When gray, it loses its channel in CV)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            # Resize frame so it fits the window
            scale_factor = min(self.image_label.width() / frame.shape[1], self.image_label.height() / frame.shape[0])
            frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

            # Display frame
            height, width = frame.shape[:2]

            qImg = QImage(frame, width, height, int(frame.strides[0]), QImage.Format.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qImg))

    def toggle_grayscale(self):
        self.is_grayscale = not self.is_grayscale
        # Update display on button click


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WebcamWindow()
    window.show()
    sys.exit(app.exec())
