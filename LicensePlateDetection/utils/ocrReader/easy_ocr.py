import numpy as np
from LicensePlateDetection.utils.ocrReader.base_ocr import BaseOCR
import re

# EasyOCR (OCR)
import easyocr
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

class EasyOCR(BaseOCR):
    def __init__(self):
        super().__init__("easy_ocr")
        self.reader = easyocr.Reader(['en'], gpu=False) # this needs to run only once to load the model into memory

 
    def read(self, image):
        detections = self.reader.readtext(image, batch_size=5)
        texts = []
        scores = []

        for i, detection in enumerate(detections):
            bbox, text, score = detection

            text: str = text.replace(" ", "")
            text = text.strip('!@#$%^&*()_+-=[]{}|;:",.<>/? \n\t').upper().lstrip('-').rstrip('-')
            
            # Special niche case we have (when the plate number has NCR, REGION 6, etc. It's usually always the third). 
            # for this, workaround is just removing the third item if it's not a number
            if ((i == 1 or i == 2) and not text.isdigit()):
                break
            texts.append(text)
            scores.append(score)
        
        text = ''.join(texts)
        
        scores_average = None
        if (len(scores) > 0):
            scores_average = np.average(scores)

        return text, scores_average