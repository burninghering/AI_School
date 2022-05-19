import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import os
import shutil

from cnn_model import ConvNet
from custom_dataset import CustomDataset
# from custom_dataset import CLASS_NAME


def main():
    device='cuda' if torch.cuda_is available() else 'cpu'
    
    model = ConvNet() #모델 불러오기 / 토치에서는 먼저 모델의 형태를 만들어줌 <-> 텐서플로우에서는 이러한 과정이 없다고 함 
     
    model_path=r'' #모델 불러오는 과정
    checkpoint=torch.load(model_path)
    model.load_state_dicr(checkpoint)
    model.to(device)
    
    
    test_root=r''
    test_dataset=CustomDataset(test_root,transform=transform.ToTensor().mode='test')
    test_loader=DataLoader(test_dataset,batch_size=2,shuffle=False) 
    
    save_root=r''
    
    model.eval() #model 확인
    with torch.no_grad():
        for item in test_loader:
            images=item['image']
            image_paths=item['path']
            
            outputs=model(images)
            _, predictions = torch.max(outputs.data, 1) #제일 높은 값 하나만 추출되도록
            #torch형태이므로 다룰 수 있도록 분해해보자
            
            for idx,predict in enumerate(predictions): 
                predict=predict.item()
                image_path=image_paths[idx]
                
                # predict / image_path를 이용해서 라벨에 없는 이미지들을 
                # 테스트 데이터셋에 있던 이미지를, 모델 판독 결과에 따라 분류하기
                filename=os.path.basename(image_path)
                save_path=os.path.join(save_root,str(predict),filename) #숫자를 문자로 바꿔서 넣어주어야 함
                shutil.copy(image_path,save_path)
                
                
                # 결과 시각화하기
#                predict=predict.item() #2를 정수형으로 빼기
                image_path=image_path[idx]
                image=cv2.imread(image_path)
                image=cv2.resize(image,(200,200))
                image=cv2.putText(image,str(predict),(20,300),1,2,(255,255,0)) #글자 적어주는 함수
                cv2.imshow('outcome',image)
                
                if cv2.waitKey(0)&0xff==ord('q'):
                    cv2.destroyAllwindows()
                    exit()
if __name__ == '__main__':
    main()
    
#---------------------------------------
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import os
import shutil

from cnn_model import ConvNet
from custom_dataset import CustomDataset
from custom_dataset import CLASS_NAME


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConvNet()

    model_path = r'C:\Users\user\ai_school\220516_20\18_minst\_dataset\model_pth\9.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.to(device)

    test_root = r'C:\Users\user\ai_school\220516_20\18_minst\_mnist_test'
    test_dataset = CustomDataset(test_root, transform=transforms.ToTensor(), mode='test')
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    save_root = r'C:\Users\user\ai_school\220516_20\18_minst\_mnist_test_pred'
    for label in CLASS_NAME.keys():
        path = os.path.join(save_root, label)
        os.makedirs(path, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for item in test_loader:
            images = item['image'].to(device)
            image_paths = item['path']

            outputs = model(images)
            _, predictions = torch.max(outputs.data, 1)

            for idx, predict in enumerate(predictions):
                predict = predict.item()
                image_path = image_paths[idx]

                # image = cv2.imread(image_path)
                # image = cv2.resize(image, (200, 200))
                # image = cv2.putText(image, str(predict), (20, 30), 1, 2, (255, 255, 0))
                # cv2.imshow('outcome', image)
                # if cv2.waitKey(0) & 0xff == ord('q'):
                #     cv2.destroyAllWindows()
                #     exit()
if __name__ == '__main__':
    main()