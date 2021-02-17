# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.uic import loadUi
import sys
import gold_counter



class Ui_Dialog(QDialog):


    
    def setupUi(self, Dialog):
        self.input_file = ""
        Dialog.setObjectName("Dialog")
        Dialog.resize(804, 621)
        self.verticalLayoutWidget = QtWidgets.QWidget(Dialog)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 20, 761, 591))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.drop_in = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(18)
        font.setBold(False)
        font.setWeight(50)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.drop_in.setFont(font)
        self.drop_in.setFrameShape(QtWidgets.QFrame.Panel)
        self.drop_in.setFrameShadow(QtWidgets.QFrame.Plain)
        self.drop_in.setLineWidth(1)
        self.drop_in.setTextFormat(QtCore.Qt.PlainText)
        self.drop_in.setAlignment(QtCore.Qt.AlignCenter)
        self.drop_in.setObjectName("drop_in")
        self.verticalLayout.addWidget(self.drop_in)
        self.buttonBox = QtWidgets.QDialogButtonBox(self.verticalLayoutWidget)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)
        
        self.drop_in.mousePressEvent = self.browserFiles

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(self.imageProcess)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def browserFiles(self, e):
        fname = QFileDialog.getOpenFileName(self, 'Open ScoreBoard', 'C:\\Users\\Bryan\\Documents')
        if fname[0] != '':
            self.drop_in.setPixmap(QtGui.QPixmap(fname[0]))
        self.input_file = fname[0]
    
    def imageProcess(self):
        try:
            gold_counter.main(self.input_file)
        except:
            print('errors')
            
        self.drop_in.setPixmap(QtGui.QPixmap('output.jpg'))

        


    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.drop_in.setText(_translate("Dialog", "Drop in Image or Click to open window Browser"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())