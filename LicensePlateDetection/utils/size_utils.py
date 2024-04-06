from PySide6.QtCore import QSize

def get_half_width(size: QSize):
    return int(size.height() / 2)

def get_half_height(size: QSize) -> int:
    return int(size.height() / 2)