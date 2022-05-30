import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout

"""
구성
창의 가운데 아래에 두 개의 버튼을 배치
두개의 버튼은 창의 크기를 변화시켜도 같은 자리에 위치합니다. 
"""


class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        # 두개의 버튼 만들기
        okButton = QPushButton('ok')
        cancelButton = QPushButton('Cancel')

        """
            수평 박스를 하나 만들고, 두개의 버튼과 양쪽에 빈 공간 추가 합니다. 
            addStretch() 메서드는 신축성있는 빈공간을 제공합니다
            두 버튼 양쪽의 stretch factor 1로 같기 떄문에 이 두 빈공간의 크기는 창의 크기가 변화해도 항상 동일 
        """
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(okButton)
        hbox.addWidget(cancelButton)
        hbox.addStretch(1)

        vbox = QVBoxLayout()
        vbox.addStretch(3)
        vbox.addLayout(hbox)
        vbox.addStretch(1)

        # 최종적으로 수직 박스를 창의 메인 레이어아웃으로 설정
        self.setLayout(vbox)
        self.setWindowTitle("Box Layout")
        self.setGeometry(300, 300, 300, 200)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
