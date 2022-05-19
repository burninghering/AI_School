#분류 학습하는 모델
from PIL import Image
from torch.utils.data import Dataset
import os

#각각의 라벨에 대해 매칭
CLASS_NAME = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
              '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

class CustomDataset(Dataset):
    def __init__(self, image_root, transform,model='train'):
        #전처리 :
        #사용할 이미지들의 경로 / 라벨 관련된 내용을 모두 지정
        self.image_paths=get_img_paths(image_root) #이미지 경로 불러오기
        self.labels=[]
        
        if mode=='train': # 트레인일때는 정답지가 있지만, 없을 때는 정답지가 없으므로
            
            for image_path in self.image_paths:
                label=image_path.split('\\')[-2] #라벨에 넣어주기
                self.labels.append(CLASS_NAME[label])
                #매칭되는 라벨으로 찾아갈 것
        else: #라벨이 의미없는 값으로 채워지도록
            self.labels=[-1 for i in range(len(self.image_paths))] #이미지의 데이터셋 크기만큼 -1 채워넣기
        
        self.transform=transform
            
        
    def __len__(self): #데이터셋 크기 반환
        return len(self.image_paths) #0부터 데이터셋 길이 주기

    def __getitem__(self, idx):
        #이미지 읽어서 라벨과 반환 / 학습에 들어갈 정보 반환
        
        #인덱스 사용해서 이미지 경로 특정
        image_path=self.image_paths[idx] #이미지 경로 읽어와서 이미지 처리
        label=self.labes[idx] #라벨 읽어주기
        
        image=Image.open()
        image=self.transform(image)
        
        return {'label':label,'image':image,'path':image_path} #라벨, 이미지, 경로 정보 반환
        
        
#이미지 경로 저장해주는 함수    
def get_img_paths(image_root): 
    image_paths=[]
    for (path,dir,files) in os.walk(image_root): #os.walk 사용해 변수 만들어 사용
        for file in files: #이미지 아닌 파일을 검색해내자
            ext=file.split('.')[-1]
            if ext in ['jpg','png']:
                image_path=os.path.join(path,file)
                #경로를 찾아주고 image path에 붙이자

#-----------------------------------------------------

from PIL import Image
from torch.utils.data import Dataset
import os

CLASS_NAME = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
              '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
class CustomDataset(Dataset):
    def __init__(self, image_root, transform):
        # 전처리
        # 사용할 이미지들의 경로 / 라벨 관련된 내용을 모두 지정
        self.image_paths = get_img_paths(image_root)
        self.labels = []
        for image_path in self.image_paths:
            # C:\Users\user\ai_school\220516_20\18_minst\_dataset\train\0\image.png
            label = image_path.split('\\')[-2]
            self.labels.append(CLASS_NAME[label])

        self.transform = transform

    def __len__(self): # 데이터셋 크기 반환
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 이미지 읽어서 라벨과 반환 / 학습에 들어갈 정보 반환
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path)
        image = self.transform(image)

        return {'label': label, 'image': image, 'path': image_path}


def get_img_paths(image_root):
    image_paths = []
    for (path, dir, files) in os.walk(image_root):
        for file in files:
            ext = file.split('.')[-1]
            if ext in ['jpg', 'png']:
                image_path = os.path.join(path, file)
                image_paths.append(image_path)
    return image_paths
            