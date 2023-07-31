from cProfile import label, run
from email.charset import QP
from json import load
from lib2to3.pgen2 import driver
from re import S
import sys
import os
from os import path
import functools
from argparse import ArgumentParser
from telnetlib import SE
from tkinter import SEL
import xdrlib

import cv2 
import time
from PIL import Image
import numpy as np
import torch
from collections import deque

from PyQt5.QtWidgets import (QWidget, QApplication, QToolTip, QComboBox, 
    QHBoxLayout, QLabel, QAction, QPushButton, QTextEdit, QMessageBox, QDesktopWidget,
    QPlainTextEdit, QVBoxLayout, QSizePolicy, QButtonGroup, QSlider, QLineEdit,
    QShortcut, QRadioButton, QProgressBar, QFileDialog, QGridLayout)

from PyQt5.QtGui import QFont, QPixmap, QKeySequence, QImage, QTextCursor
from PyQt5.QtCore import Qt, QTimer, QCoreApplication
from PyQt5 import QtCore
from PyQt5.Qt import QThread, pyqtSignal

from train_select import train_select
from qt_material import apply_stylesheet

#basic_path = ''

class my_thread(QThread):
    send_signal = pyqtSignal(str)
    send_signal_1 = pyqtSignal(bool)

    def run(self):
        while(True):
            output = ''
            with open('./output.txt', 'r') as f:
                for line in f.readlines():
                    output += line
            f.close()
            #print(output)
            self.send_signal.emit(output)
            self.send_signal_1.emit(False)
            if output.endswith('done!\n'):
                self.send_signal_1.emit(True)
                break
            time.sleep(1)

class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.load_ctf = False
        self.modelSet = [
            'unet', 'hrnet', 'setr', 'unet3p', 'segnet', 'hardnet', 'deeplab',
            'pspnet', 'danet', 'eanet', 'ocrnet', 'resunet', 'ocnet', 'attunet',
            'dense', 'dlink', 'ternaus', 'scseunet', 'r2', 'r2att', 'csnet', 'unetpp',
            'unetppbig', 'multires', 'u2net', 'u2netp', 'onenet', 'lightnet', 'cenet',
            'setr', 'hardalter', 'lrfea', 'simple'
        ]
        self.switch_table = {'data-XH' : 'xh', 'data-hard' : 'data_hard', 'Big' : 'zs_big', 'Small' : 'zs_small', 'data-pancreas' : 'pancreas'}
        #self.basic_path = ''
        self.model = 'unet'
        self.initUI()
    
    def initUI(self):
        
        #self.setGeometry(0, 0, 1920, 1080)
        self.resize(1200, 800)
        self.center()
        self.setWindowTitle('JMedSeg')

        QToolTip.setFont(QFont('SansSerif', 100))

        self.model_label = QLabel('model select: ', self)
        self.model_label.setFont(QFont('Arial', 10))

        self.model_combo = QComboBox(self)
        self.model_combo.addItems(self.modelSet)
        self.model_combo.activated[str].connect(self.modelSelect)

        self.IMGPlace = QLabel(self)
        self.IMGPlace.resize(self.size() * 0.5)
        #self.IMGPlace.move(100, 100)
        
        self.INTROPlace = QTextEdit(self)
        self.INTROPlace.resize(self.size() * 0.5)
        intro = '[INTRO]:\nThis is a GUI for JMedSeg, aiming to help people who want to use deep learning models for medical image segmentation\n\n[USAGE]:\nLOAD: Load one image in your dataset(sorted as JMedSeg ask)\nLABEL: If you have not labeled your data, this button is to use LabelMe for labeling. If you have labeled your data, just use your data for the next move\nRUN: Once you have load&label your data, this button is for training/testing'

        self.INTROPlace.setFont(QFont('Arial', 10))
        self.INTROPlace.setText(intro)

        self.OUTPUTPlace = QTextEdit(self)
        self.OUTPUTPlace.resize(self.size() * 0.5)
        output = 'Output Information'
        self.OUTPUTPlace.setText(output)

        self.LinePlace = QLineEdit(self)
        self.basic_path = ''

        self.load_btn = QPushButton('Load', self)
        self.load_btn.setToolTip('Load your own labeled img')
        self.load_btn.resize(self.load_btn.sizeHint())
        #self.load_btn.move(50, 50)

        self.test_btn = QPushButton('Test', self)
        self.test_btn.setToolTip('Label your own unlabeled img using LabelMe')
        self.test_btn.resize(self.test_btn.sizeHint())

        self.run_btn = QPushButton('More', self)
        self.run_btn.setToolTip('Run your img on pretrained model')
        self.run_btn.resize(self.run_btn.sizeHint())

        '''
        OpenFile = QAction('Open', self)
        OpenFile.setStatusTip('Open new File')
        OpenFile.triggered.connect(self.showDialog)
        '''
        #load_btn.addAction(OpenFile)
        self.load_btn.clicked.connect(self.showIMG)
        self.test_btn.clicked.connect(self.TEST)
        '''
        qbtn = QPushButton('Quit', self)
        qbtn.clicked.connect(QCoreApplication.instance().quit)
        qbtn.resize(qbtn.sizeHint())
        qbtn.move(150, 150)
        '''

        

        self.h0 = QHBoxLayout()
        self.v0 = QVBoxLayout()
        self.h1 = QHBoxLayout()
        self.h_model = QHBoxLayout()
        self.v_tmp = QVBoxLayout()
        #self.h0.addStretch(1)
        
        self.h_model.addWidget(self.model_label)
        self.h_model.addWidget(self.model_combo)
        self.v_tmp.addLayout(self.h_model)
        self.v_tmp.addWidget(self.IMGPlace)
        self.h0.addWidget(self.OUTPUTPlace)
        self.h0.addLayout(self.v_tmp)
        self.h0.addWidget(self.INTROPlace)
        #self.h1.addStretch(1)
        self.h1.addWidget(self.LinePlace)
        self.h1.addWidget(self.load_btn)
        self.h1.addWidget(self.test_btn)
        self.h1.addWidget(self.run_btn)
        self.v0.addLayout(self.h0)
        self.v0.addLayout(self.h1)

        self.setLayout(self.v0)
        
        #self.totalSignal()

        self.show()
        


    def showIMG(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        img_suffix = ['.jpg', '.png', '.jpeg']
        if any(fname.lower().endswith(s) for s in img_suffix):
            #print(fname)
            pixmap = QPixmap(fname)
            self.IMGPlace.setPixmap(pixmap)
            #self.resize(pixmap.size())
            path = fname.split('/')[:-4]
            #print(path)
            for i in path:
                self.basic_path += (i + '/')
            self.basic_path = self.basic_path[:-1]
            print(self.basic_path)
            self.LinePlace.setText('Your data dir path: ' + self.basic_path)
            #basic_path = self.basic_path
            self.load_ctf = True
            self.IMGPlace.adjustSize()
            self.LinePlace.adjustSize()

    '''
    def useLabelme(self):
        #f = os.popen("").readlines()
        f = os.popen("labelme")
        print(f)
    '''

    def slot(self, text):
        self.OUTPUTPlace.clear()
        self.OUTPUTPlace.append(text)

    def slot_1(self, flg):
        if flg:
            dir = './test_performance/{}-Adam-ce-50-test-stnFalse-pretrainFalse-augFalse/'.format(self.model)
            for obj in os.walk(dir):
                dir += obj[2][0]
                break
            img = QPixmap(dir)
            self.IMGPlace.setPixmap(img)

    def TEST(self):
        self.my_thread = my_thread()
        self.my_thread.send_signal.connect(self.slot)
        self.my_thread.send_signal_1.connect(self.slot_1)
        self.my_thread.start()

        dir_now = os.path.dirname(os.path.abspath(__file__))
        dir_now = dir_now[:-3].replace('\\', '/').replace('\r', '')
        self.dataset = self.switch_table[self.basic_path.split('/')[-1]]
        order = 'cd ' + dir_now + '\n' + 'python run.py --model {} --dataset {} --mode test --cuda -e 50 --loss ce -r test_result'.format(self.model, self.dataset)
        #cmd = os.popen(order)
        print(order)
        with open('./test.sh', 'w') as f:
            f.write('cd ')
            f.write(dir_now)
            f.write('\n')
            f.write('python run.py --model {} --dataset {} --mode test --cuda -e 50 --loss ce -r test_result'.format(self.model, self.dataset))
        f.close()
        #os.system('cd ' + dir_now)
        #os.system('python ./run.py --model {} --dataset {} --mode test --cuda -e 50 --loss ce -r test_result'.format(self.model, self.dataset))
        os.system("start powershell.exe cmd /k 'python ./run.py --model {} --dataset {} --mode test --cuda -e 50 --loss ce -r test_result'".format(self.model, self.dataset))
        
        


    def modelSelect(self, text):
        self.model = text
        print(self.model)

    def closeEvent(self, event):
        with open('./output.txt', 'w') as f:
            f.write('')
        f.close()

        reply = QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QMessageBox.Yes | 
            QMessageBox.No, QMessageBox.No)
 
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class training_GUI(QWidget):
    
    def __init__(self):
        super().__init__()
        self.modelSet = [
            'unet', 'hrnet', 'setr', 'unet3p', 'segnet', 'hardnet', 'deeplab',
            'pspnet', 'danet', 'eanet', 'ocrnet', 'resunet', 'ocnet', 'attunet',
            'dense', 'dlink', 'ternaus', 'scseunet', 'r2', 'r2att', 'csnet', 'unetpp',
            'unetppbig', 'multires', 'u2net', 'u2netp', 'onenet', 'lightnet', 'cenet',
            'setr', 'hardalter', 'lrfea', 'simple'
        ]
        self.basic_path = ''
        self.model = 'unet'
        self.initUI()

    def initUI(self):
        self.resize(1000, 600)
        self.center()
        self.setWindowTitle('Model Trainning')

        QToolTip.setFont(QFont('SansSerif', 10))

        self.model_label = QLabel('model select: ', self)

        self.model_combo = QComboBox(self)
        self.model_combo.addItems(self.modelSet)
        self.model_combo.activated[str].connect(self.modelSelect)

        self.data_path = QLineEdit(self)
        self.judge()

        self.process_bar = QProgressBar(self)
        self.process_label = QLabel('process: ', self)

        self.train_btn = QPushButton('train', self)
        self.train_btn.resize(self.train_btn.sizeHint())

        self.label_btn = QPushButton('label', self)
        self.label_btn.resize(self.label_btn.sizeHint())

        self.train_btn.clicked.connect(self.train_model)
        self.label_btn.clicked.connect(self.useLabelme)

        self.output_text = QTextEdit(self)
        self.output_text.setText('output information\n')

        self.h0 = QHBoxLayout()
        self.h1 = QHBoxLayout()
        self.h2 = QHBoxLayout()
        self.v0 = QVBoxLayout()
        self.h3 = QHBoxLayout()

        self.h0.addWidget(self.model_label)
        self.h0.addWidget(self.model_combo)
        self.h1.addWidget(self.process_label)
        self.h1.addWidget(self.process_bar)
        self.h2.addWidget(self.label_btn)
        self.h2.addWidget(self.train_btn)
        self.v0.addLayout(self.h0)
        self.v0.addWidget(self.data_path)
        self.v0.addLayout(self.h1)
        self.v0.addLayout(self.h2)
        self.h3.addWidget(self.output_text)
        self.h3.addLayout(self.v0)

        self.setLayout(self.h3)

    def train_model(self):
        '''
        TODO: train model cmd and design
        '''
        self.train_select = train_select(self.model)
        self.train_select.show()

    '''
    def test_model(self):
        dir_now = os.path.dirname(os.path.abspath(__file__))
        self.dataset = self.basic_path.split('/')[-1]
        order = 'cd ' + dir_now + '\n' + 'python3.7 run.py --model {} --dataset {} --mode test --cuda -e 50 --loss ce -r test_result'.format(self.model, self.dataset)
        #cmd = os.popen(order)
        print(order)
    '''
    def useLabelme(self):
        f = os.popen("labelme")
        print(f)

    def receive(self, basic_path):
        self.basic_path = basic_path

    def judge(self):
        if self.basic_path is not '':
            self.data_path.setText('Your data path: ' + self.basic_path)
        else:
            self.data_path.setText('WARNING! YOU SHOULD LOAD YOUR DATA')

    def modelSelect(self, text):
        #self.model_label.setText('model select?: ' + text)
        #self.model_label.adjustSize()
        self.model = text
        print(self.model)

        
        
    '''
    def closeEvent(self, event):
        
        reply = QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QMessageBox.Yes | 
            QMessageBox.No, QMessageBox.No)
 
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
    '''

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())



if __name__ ==  '__main__':

    #parser = ArgumentParser()

    #load pretrained model
    extra = {
        'font_family' : 'Roboto',
        'density_scale' : '2'
    }

    app = QApplication(sys.argv)
    ui = GUI()
    train_ui = training_GUI()
    ui.run_btn.clicked.connect(lambda:{print(ui.basic_path), train_ui.receive(ui.basic_path), train_ui.judge(), train_ui.data_path.adjustSize(), train_ui.show()})
    apply_stylesheet(app, theme='dark_teal.xml', extra=extra)
    sys.exit(app.exec_())
    