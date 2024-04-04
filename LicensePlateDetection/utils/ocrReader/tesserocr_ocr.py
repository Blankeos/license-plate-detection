from LicensePlateDetection.utils.ocrReader.base_ocr import BaseOCR

from tesserocr import PyTessBaseAPI
from PIL import Image

class TesserOCR(BaseOCR):
    def __init__(self):
        super().__init__("tesserocr")
        # OCR Instance
        self.ocrAPI = PyTessBaseAPI()

 
    def read(self, image):
        pilImage = Image.fromarray(image)
        self.ocrAPI.SetImage(pilImage)

        score = self.AllWordConfidences()
        text = self.ocrAPI.GetUTF8Text().upper().strip('!@#$%^&*()_+-=[]{}|;:",.<>/? \n\t').lstrip('-').rstrip('-')
        return text, score