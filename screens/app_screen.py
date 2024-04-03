from PySide6.QtWidgets import (QApplication, QListWidget, QMainWindow, QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QSizePolicy, QMenuBar, QMenu)
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtCore import Qt, QTimer, QSize

# Import for OpenCV
import cv2

# class ImagePanel(QWidget):
#     def __init__(self):
#         super().__init__()
        
class LicensePlateDetection(QMainWindow):
    def __init__(self):
        super().__init__()
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
        self.grayscale_btn = QPushButton("Grayscale")
        self.grayscale_btn.clicked.connect(self.toggle_grayscale)

        self.left_layout.addWidget(self.image_label)
        self.left_layout.addStretch()
        self.left_layout.addWidget(self.grayscale_btn)
        self.main_layout.addLayout(self.left_layout)

        # ------

        # Right Layout (right side)
        self.right_layout = QVBoxLayout()

        self.list_widget = QListWidget()
        self.list_widget.addItems(["Apple", "Grapes", "Banana", "Nice"])
        self.list_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.list_widget.setMaximumHeight(200)
        self.right_layout.addWidget(self.list_widget)

        # self.right_layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetFixedSize)
        # self.right_layout.setAlignment(Qt.AlignmentFlag.AlignBaseline)
        # self.right_layout.setContentsMargins(20,20,20,20)

        # sizer_layout = QVBoxLayout(); sizer_layout.setContentsMargins(200, 0, 0, 200); 
        # sizer_layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetFixedSize)
        # self.right_layout.addLayout(sizer_layout)

        text_label = QLabel("This is some text information.")
        # text_label.setWordWrap(True)
        text_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.right_layout.addWidget(text_label) # ADD
        

        another_label = QLabel("This is another text")
        another_label.setWordWrap(True)
        another_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.right_layout.addWidget(another_label) # ADD
        # self.text_label.setStyleSheet('background: yellow')

        fruit_list_box = QVBoxLayout()
        fruit_list_box.setSizeConstraint(QVBoxLayout.SizeConstraint.SetFixedSize)
        fruit_labels = ["Apple", "Grapes", "Oranges"]
        for label in fruit_labels:
            label_widget = QLabel(label)
            label_widget.setWordWrap(True)
            label_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            fruit_list_box.addWidget(label_widget)
        # self.right_layout.addLayout(fruit_list_box) # ADD

        self.right_layout.addStretch()



        self.main_layout.addLayout(self.right_layout) # ADD

        ###----

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
            scale_factor = 1.0

            if self.width() < frame.shape[1] or self.height() < frame.shape[0]:
                scale_factor = min(self.left_layout.geometry().width() / frame.shape[1], self.left_layout.geometry().height() / frame.shape[0])
            frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

            # Display frame
            height, width = frame.shape[:2]

            # Convert frame to QImage
            qImg = QImage(frame, width, height, int(frame.strides[0]), QImage.Format.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qImg))

    def handle_close_event(self, event):
        # Handle closing logic (e.g., prompt user to save)
        # Accept the close event to quit the application
        event.accept()

    def toggle_grayscale(self):
        self.is_grayscale = not self.is_grayscale
        # Update display on button click