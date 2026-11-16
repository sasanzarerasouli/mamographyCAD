import os
import logging
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtCore import Qt

from src.ui.main_window import MammographyCADApp


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    font = QFont("Arial", 9)
    app.setFont(font)
    
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, QColor(240, 240, 240))
    palette.setColor(palette.ColorRole.WindowText, QColor(50, 50, 50))
    palette.setColor(palette.ColorRole.Base, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = MammographyCADApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()