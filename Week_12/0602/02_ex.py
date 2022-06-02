"""
파이토치로 다층 퍼셉트론 구현하기 
"""
import torch
import torch.nn as nn
"""
# GPU 사용가능한 여부 파악 test code -> CPU 인텔 엔비디아 GPU / AMD 엔비디아 GPU 맥북 M1 
device = "cuda" if torch.cuda.is_available() else "cpu"
"""

# M1 사용중인 분들
device = torch.device("mps")
# str_device = str(device)
# print("device info >> ", type(str_device))

# seed
torch.manual_seed(777)

if device == "cuda":
    torch.cuda.manual_seed_all(777)

# 데이터 생성
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

# 데이터 텐서 변경
x = torch.tensor(x, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)

print("x", x)
print("y", y)
