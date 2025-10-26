from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys


class MenuScreen(QMainWindow):
    def __init__(self):
        super(MenuScreen, self).__init__()
        self.setGeometry(10, 100, 300, 300)
        self.setWindowTitle("Test")
        self.initUI()

    def initUI(self):
       
        
        self.label = QtWidgets.QLabel(self)
        self.label.setText("hi")
        self.label.move(50, 50)

        self.b1 = QtWidgets.QPushButton(self)
        self.b1.setText("Open Folder")
        self.b1.clicked.connect(self.click_to_open)

    def click_to_open(self):
        self.label.setText("yayaya")






def window():
    app = QApplication(sys.argv)
    win = MenuScreen()
    

    
    win.show()
    sys.exit(app.exec_())

window()
