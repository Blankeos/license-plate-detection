import time
from PySide6.QtCore import QThread, Signal

class IntervalThread(QThread):
    """
    Usage
    ```
    # In your initialization
    self.thread = IntervalThread(interval=200)
    self.thread.any_signal.connect(self.doSomething)
    self.thread.start()

    # ...

    def doSomething(self):
        print("Hello World!")
    ```
    """

    any_signal = Signal()

    def __init__(self, interval: float, parent=None):
        """
        @param interval - Delay (in seconds) execution for a given number of seconds and will repeat at this interval. The argument may be a floating point number for subsecond precision.
        """
        
        super(IntervalThread, self).__init__(parent)
        self.interval = interval

    def run(self):
        while True:
            time.sleep(self.interval)
            self.any_signal.emit()
