import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models


from cnn_model import ConvNet #ConvNet 가져오기
from custom_dataset import CustomDataset

def main():
    device='cuda' if torch.cuda.is_available() else 'cpu' 
    #모델에 개선된 환경을 맞춰주기 위해 위치를 맞춰줌 = 계산환경을 맞춰준다 
    #똑같은 디바이스 환경에 있기 위해 토치에 있던지, cpu에 있던지 둘 중 하나여야 함
    
    # model 지정하기
    model=ConvNet() #모델 클래스객체를 만들어 모델 학습함
    model=model.to(device)
    
    # 모델 가져오기

    # loss function, optimizer(learning rate) 지정하기
    loss_function=nn.CrossEntropyLoss() #손실함수
    optimizer=optim.SGD(model.parameters(),lr=0.01) #무난한 경사 하강법 (SGD) 사용하기 model.parameters():변형할 파라미터 / 모델에 사용되는 weight(lr)바꾸기
    
    # batch size, epoch 지정
    batch_size=64 #데이터를 한번에 몇개 불러올 것인지
    epoch=10 #한 에포크는 3개의 스텝으로 이루어짐
    
    # train용 dataset 불러오기
    train_root=r''
    train_dataset=CustomDataset(train_root,transform=transform.ToTensor()) #train코드에 있던 데이터를 학습에 이용할 수 있는 양식으로 바꿈
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True) #저장해줌, batch_size=batch_size만큼 읽어오겠다
    
    # valid용 dataset 불러오기
    valid_root=r''
    valid_dataset=(valid_root,transform=transform.ToTensor())
    valid_dataloader=(valid_dataset,batch_size=batch_size,shuffle=False) #shuffle=False 섞어주지 않겠다 (평가할때는 원본 그대로 사용해야하므로)
    
    # 학습 및 validation
    for epoch_idx in range(epoch): #이미지 전체에 대해 한 epoch마다 학습 1번
        model.train() #학습하는 용도
        losses=0
        for item in train_dataloader: #학습용 데이터 읽어와주기
            labels=item['label'].to(device) #커스텀데이터셋을 사용함
            images=item['image'].to(device)
            
            optimizer.zero_grad()
            outputs=model(images) #모델이 추론한 라벨 결과가 outputs에 기록됨 #모델에 들어갈 이미지정보
            
            loss=loss_function(outputs,labels) #그것을 정답과 비교해서
            loss.backward() # loss의 기준으로 거꾸로 가는 역전파
            
            optimizer.step() #모델의 파라미터/weight를 바꿔나가는 학습의 일련과정 끝
            
            losses+=loss #학습 종료
            
    # valid 모델 성능 평가하기
    model.eval() # 모델이 학습용이 아니라 평가용이라는 것 
    with torch.no_grad(): #모델의 파라미터가 변경되지 않도록 제어해줘야함
        corrcet=0
        total=0
        for item in valid_dataloader: #데이터 읽어와서
            labels=item['label'].to(device) 
            images=item['image'].to(device)
            
            outputs=model(images) #모델을 수행한 후 
            _,predictions=torch.max(outputs.data,1) #outputs 데이터 중 가장 높은 것을 하나 가져와서 변수에 저장
            
            total += labels.size(0)
            correct += (labels == predictions).sum() #이미지 결과를 매칭시켜서, sum해서 true인것만 다 더해주기
            
        avg_loss=losses / batch_size
        accuracy = correct / total * 100 #정확도 검사
        print(f'epoch: {epoch_idx} - Accuracy: {accuracy}, Avg_loss:{avg_loss}') # 한 epoch마다 이렇게 저장됨
        torch.save(model.state.dict(), f'{save_root}\') #save_root\epoch_idx.pth로 저장하기 
                   
        # 정확도가 제일 높은 모델 'best.pth' 이름으로 저장하기        
    if accuracy > best_accuracy:
                   best_accuracy=accuracy
                   torch.save(model.state_dict(), f'{save_root}/best.pth')
                   
                   
if __name__ == '__main__':
    main()
    
##-------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models


from cnn_model import ConvNet
from custom_dataset import CustomDataset

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model 지정하기
    model = ConvNet()
    model = model.to(device)

    # loss function, optimizer(learning rate) 지정하기
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # batch size, epoch 지정
    batch_size = 64
    epoch = 10
    # train용 dataset 불러오기
    train_root = r'C:\Users\user\ai_school\220516_20\18_minst\_dataset\train'
    train_dataset = CustomDataset(train_root, transform=transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # valid용 dataset 불러오기
    valid_root = r'C:\Users\user\ai_school\220516_20\18_minst\_dataset\valid'
    valid_dataset = CustomDataset(valid_root, transform=transforms.ToTensor())
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # 학습 및 validation
    for epoch_idx in range(epoch):

        # train
        model.train()
        losses=0
        for item in tqdm(train_dataloader):
            labels = item['label'].to(device)
            images = item['image'].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

        # valid
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for item in valid_dataloader:
                labels = item['label'].to(device)
                images = item['image'].to(device)

                outputs = model(images)
                _, predictions = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (labels == predictions).sum()
        
        avg_loss=losses / batch_size
        accuracy = correct / total * 100
        print(f'epoch: {epoch_idx} - Accuracy: {accuracy}, Avg_loss:{avg_loss}')
        torch.save(model.state.dict(), f'{save_root}\') #save_root\epoch_idx.pth로 저장하기 
                   
        # 정확도가 제일 높은 모델 'best.pth' 이름으로 저장하기


if __name__ == '__main__':
    main()