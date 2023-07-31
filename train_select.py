from cProfile import label
from re import S
from PyQt5.QtWidgets import (QWidget, QApplication, QToolTip, QComboBox, 
    QHBoxLayout, QLabel, QAction, QPushButton, QTextEdit, QMessageBox, QDesktopWidget,
    QPlainTextEdit, QVBoxLayout, QSizePolicy, QButtonGroup, QSlider, QLineEdit,
    QShortcut, QRadioButton, QProgressBar, QFileDialog, QGridLayout)

from PyQt5.QtGui import QFont, QPixmap, QKeySequence, QImage, QTextCursor
from PyQt5.QtCore import Qt, QTimer, QCoreApplication
from PyQt5 import QtCore

class train_select(QWidget):
    def __init__(self, model):
        super().__init__()
        self.initUI()
        self.model = model
        

    def initUI(self):
        self.resize(400, 300)
        self.setWindowTitle('select train args')
        self.center()
        QToolTip.setFont(QFont('SansSerif', 10))

        self.label = QLabel('train command: ', self)
        #self.label.move(10, 10)

        self.default_btn = QPushButton('default', self)
        self.default_btn.resize(self.default_btn.sizeHint())

        self.choose_btn = QPushButton('choose own cmd', self)
        self.choose_btn.resize(self.choose_btn.sizeHint())

        self.cmd = QLineEdit(self)

        self.h0 = QHBoxLayout()
        self.h1 = QHBoxLayout()
        self.v0 = QVBoxLayout()

        self.h0.addWidget(self.label)
        self.h0.addWidget(self.default_btn)
        self.h0.addWidget(self.choose_btn)
        self.h0.addStretch(1)
        #self.h1.addStretch(0.3)
        self.h1.addWidget(self.cmd)
        self.v0.addLayout(self.h0)
        self.v0.addLayout(self.h1)

        self.setLayout(self.v0)

    
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())