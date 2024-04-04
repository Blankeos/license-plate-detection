from LicensePlateDetection.utils.ocrReader.base_ocr import BaseOCR

import pytesseract

class PytesseractOCR(BaseOCR):
    def __init__(self):
        super().__init__("pytesseract")

 
    def read(self, image):
        # it has score but I'm not sure how to implement.
        return pytesseract.image_to_string(image, lang="eng", config="--psm 6"), None