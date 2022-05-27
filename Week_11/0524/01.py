import torch
import torchvision
import cv2
import numpy as np
import time

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from matplotlib import pyplot as plt

#pip install -U Albumentations
import albumentations
from albumentations.pytorch import ToTensorV2

class AlbumentationsDataset(Dataset): #torchvision 대신 다른 라이브러리 만듦
    def __init__(self, file_path, labels, transform=None): #커스텀 데이터셋 만듦
        self.file_path = file_path
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        label = self.labels[index]
        file_path = self.file_path[index]

        # Image open
        # image = Image.open(file_path)
        image = cv2.imread(file_path) #image 열어주는 방법

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #bgr로 저장되기 때문에 한번 convert해야함

        start_t = time.time()

        if self.transform: 
            augmented = self.transform(image=image)
            image = augmented['image']

        total_time = (time.time() - start_t)

        return image, label, total_time

    def __len__(self):
        return len(self.file_path)


albumentations_transform = albumentations.Compose([ #albumentations에 어떤 것을 먹일까?
    albumentations.Resize(256, 256),
    albumentations.RandomCrop(224, 224),
    albumentations.HorizontalFlip(),
    ToTensorV2() 
    #albumentations
])

albumentations_dataset = AlbumentationsDataset(
    file_path=["./data/image.jpeg"], #데이터 경로 넣어주기
    labels=[1], #데이터의 라벨은 이것이다
    transform=albumentations_transform #경로에서 가져온 것을 albumentation에 넣겠다
)

total_time = 0
for i in range(100): #100번 반복
    sample, _, transform_time = albumentations_dataset[0]
    total_time += transform_time

print("torchvision time / sample : {} ms ".format(total_time*10)) #하나가 처리되는 시간에 대해 total_time에 합쳤고 albumen이 얼마나 빠른지를 보고 싶은 것이다

#마지막 하나만 알려주게 함
plt.figure(figsize=(10, 10)) 
plt.imshow(transforms.ToPILImage()(sample))
plt.show()



# #custom dataset
# class TorchvisionDataset(Dataset):

#     def __init__(self,file_path,labels,transform=None):
#         #데이터셋의 전처리를 해주는 부분
#         self.file_path=file_path
#         self.labels=labels
#         self.transform=transform

#     def __getitem__(self, index):
#         #데이터셋에서 특정 1개의 샘플을 가져오는 함수
#         labels=self.labels[index]
#         file_path=self.file_path[index]

#         # Image open
#         image=Image.open(file_path)

#         start_t=time.time()

#         if self.transform:
#             image=self.transform(image)
#         total_time=(time.time()-start_t)

#         return image,labels,total_time


#     def __len__(self):
#         #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
#         return len(self.file_path)



# torchvision_transform=transforms.Compose([
#     transforms.Resize((256,256)),
#     transforms.RandomCrop((224,224)),
#     transforms.RandomHorizontalFlip(p=0.8), #80퍼센트만 일어나게 해라(비우면 기본이 0.5)
#     transforms.ToTensor(), #머신러닝 기능 바꾸기

# ])

# torchvision_dataset=TorchvisionDataset(
#     file_path=["./data/rose1.png"], #이미지 한 장에 대해 진행해보자
#     labels=[1],
#     transform=torchvision_transform

# )

# total_time=0
# for i in range(100):
#     sample, _, transform_time=torchvision_dataset[0] #리턴값이 3개인데 라벨에 대한 것은 굳이 쓸 필요 없기 때문에 _,는 생략한다는 의미이다
#     total_time+=transform_time 


# print("torchvision time / sample : {} ms".format(total_time*10))

# plt.figure(figsize=(10,10))
# plt.imshow(transforms.ToPILImage()(sample))
# plt.show()


