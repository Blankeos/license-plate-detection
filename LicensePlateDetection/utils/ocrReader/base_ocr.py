from abc import ABC, abstractmethod
import numpy as np

class BaseOCR(ABC):
    def __init__(self, reader_name):
        self.name = reader_name

    @abstractmethod
    def read(self, image: np.ndarray) -> tuple[str, float]:
        """
        Reads your Mat image and turns it into an array.

        # Returns:
        #     text?: the text read from the image
        #     score?: the confidence score of the reader (between 0 and 1) - Some readers don't have score. We just return None.
        """
        pass
