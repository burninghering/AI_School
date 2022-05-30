import sys
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp, QFileDialog,QPushButton,QLabel

# QtCore 모듈의 QCoreApplication 클래스 가져 오기
from PyQt5.QtCore import QCoreApplication


class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        # 아이콘 추가
        self.setWindowIcon(QIcon("./web.png"))

        # 프레임 만들기
        # setWindowTitle -> 타이틀바에 나타는 창의 제목
        self.setWindowTitle("My First Application !!")
        self.move(300, 300)
        self.resize(400, 200)

        #푸쉬버튼 만들기
        btn=QPushButton('Q')
        #상태창 테스트를 위한 label 추가 초기값 설정
        self.label=QLabel('00000000',self)
        self.move(40,40)
        #상태창 테스트를 위한 Start 버튼 생성 설정
        self.pd=QPushButton("Start",self)
        self.pd.clicked.connect(self.count_number)
        self.pd.move(150,40) #위치 잡아주기 


        #상태바 초기값 설정 (하단)
        self.statusBar().showMessage("준비 중....")

        # 메뉴바 만들기
        exitAction = QAction('&Exit', self)

        # 단축키
        exitAction.setShortcut('Ctrl+q')
        exitAction.setStatusTip("Exit application")
        exitAction.triggered.connect(qApp.quit)

        menuber = self.menuBar()

        # 추가
        loadefile = QAction('loade file ... ', self)
        savefile = QAction('save file....', self)
        loadefile.triggered.connect(self.add_open)
        savefile.triggered.connect(self.add_save)

        fileMenu = menuber.addMenu('&File')
        fileMenu.addAction(exitAction)
        fileMenu.addAction(loadefile)
        fileMenu.addAction(savefile)

        self.show()

    def add_open(self):
        FileOpen = QFileDialog.getOpenFileName(self, 'Open file ', './')
        print(FileOpen)

    def add_save(self):
        FileSave = QFileDialog.getSaveFileName(self, 'Save file', './')
        print(FileSave)

    def count_number(self):
        """
        상태바 테스트 함수
        """
        #상태바 생성 코드
        self.statusBar().showMessage("작업 중....")  
        """
        PyQt5를 쓰면서 데이터가 계속 업데이트 되지 않을 때 반드시 repaint() 부분을 추가해주세요..
        """
        self.statusBar().repaint()
        for i in range(1,1000):
            print(i)
            self.label.setText(str(i))
            self.label.repaint()
        self.statusBar().showMessage("준비 중....")



if __name__ == "__main__":
    # 모든 PyQt5 어플리케이션은 어플리케이션 객체 생성
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
