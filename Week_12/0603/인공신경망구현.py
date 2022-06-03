"""
인공신경망 구현 하고 학습 해보기 
사용할 데이터는 보스턴 집값 데이터 활용 
"""
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

# torch
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
# 데이터를 모델에 사용 할 수 있게 정리 해주는 라이브러리
import torch.nn.functional as F
# torch 내의 세부적인 기능 불러 라이브러리
import torch.nn as nn
# Loss
from sklearn.metrics import mean_squared_error
# regression 문제의 성능 측정을 위해서 MSE 라이브러리

import matplotlib.pyplot as plt

"""데이터 불러오기"""
bos = load_boston()             # bos.data 데이터 로드
df = pd.DataFrame(bos.data)     # bos.data 데이터 불러오기
df.columns = bos.feature_names  # bos.feature_names 컬럼명 불러오기
df['Price'] = bos.target        # bos.target : 정답지 값을 가져옴

"""데이터 스켈링 하기"""
"""데이터를 넘파이 배열로 만들기"""
# 데이터프레임에서 타겟값(Price)을 제외하고 넘파이 배열로 만들기 !!
X = df.drop('Price', axis=1).to_numpy()
Y = df['Price'].to_numpy().reshape((-1, 1))
# 데이터프레임 형태의 타겟값을 넘파이 배열로 만들기 !!

"""
데이터 스케일링 sklearn 에서 제공하는 MinMaxScaler
(X-min(X)/(max(X)-min(X)))을 계산
"""
scale = MinMaxScaler()
scale.fit(X)
X = scale.transform(X)
scale.fit(Y)
Y = scale.transform(Y)


"""pytorch 기초 문법에서 했던것 그대로 사용해서 텐서 데이터 와 배치를 만들기"""
"""Custom dataset"""


class Mycustom(Dataset):

    def __init__(self, x_data, y_data):
        self.x_data = torch.Tensor(x_data)
        self.y_data = torch.Tensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


"""
데이터셋 나누기
전체 데이터를 학습 데이터와 평가 데이터로 나누겠습니다
전체 데이터 X_data -> 253 + 253 Y_data 253 + 253
"""
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

"""학습 데이터 , 테스트 데이터 배치 형태로 구축"""
train_dataset = Mycustom(X_train, Y_train)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True)

test_dataset = Mycustom(X_test, Y_test)
test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False)

"""모델 구축"""


class Regressor(nn.Module):

    def __init__(self):
        """모델 연산 정의"""
        super().__init__()
        self.fc1 = nn.Linear(13, 50, bias=True)  # 입력층 13 -> 은닉층1 50으로 가는 연산
        self.fc2 = nn.Linear(50, 30, bias=True)  # 은닉층1 50 -> 은닉층2 30 으로 가는 연산
        self.fc3 = nn.Linear(30, 1, bias=True)   # 은닉층2 30 -> 출력층 1 으로 가는 연산
        # 연산이 될 때마다 20% 비율로 랜덤하게 노드 없앤다.
        self.dropout = nn.Dropout(0.2)

        pass

    def forward(self, x):
        """모델 연산의 순서 정의"""
        x = F.relu(self.fc1(x))  # Linear 계산 후 활성함수 Relu 적용
        x = self.dropout(F.relu(self.fc2(x)))  # 은닉층2에서 드랍아웃을 적용
        # 30개 노드 -> 20% dropout -> 6개 제외 계산 됩니다.
        x = F.relu(self.fc3(x))  # Linear 계산 후 활성함수 Relu 적용

        return x


"""드롭아웃 과적합 을 방지하기 위해 노드 일부를 배제하고 계산하는 방식이기 때문에 출력층에서 사용하시면 안됩니다."""

# 모델 선언
model = Regressor()
# 손실 함수
criterion = nn.MSELoss()
# 최적화
optimizer = optim.Adam(model.parameters(), lr=0.001)

"""학습 코드 작성"""
loss_list = []
n = len(train_dataloader)

for epoch in range(400):

    running_loss = 0.0

    for i, data in enumerate(train_dataloader, 0):  # 무작위로 섞인 32개의 데이터 들어 옵니다.
        inputs, values = data  # data X , Y

        """최적화 초기화"""
        optimizer.zero_grad()
        """모델 입력값을 넣고 예측값 산출"""
        outputs = model(inputs)
        """손실함수를 이용해서 error 계산"""
        loss = criterion(outputs, values)
        """손실함수를 기준으로 역전파 설정"""
        loss.backward()
        """역전파를 진행하고 가중치 업데이트"""
        optimizer.step()

        running_loss += loss.item()  # epoch 마다 평균 loss 계산하기위해 배치 loss 더한다.

    loss_list.append(running_loss / n)  # MSE(Mean Squared Error) 계산

plt.plot(loss_list)
plt.title("loss")
plt.xlabel("epoch")
plt.show()
