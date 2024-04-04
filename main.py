import sys
from PySide6.QtWidgets import QApplication
from LicensePlateDetection.screens.app_screen import LicensePlateDetection


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LicensePlateDetection()
    window.show()
    sys.exit(app.exec())
