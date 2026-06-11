import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication
from marvin.config import ASSISTANT_NAME
from marvin.gui import MarvinWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(ASSISTANT_NAME)
    app.setApplicationDisplayName(ASSISTANT_NAME)
    window = MarvinWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
