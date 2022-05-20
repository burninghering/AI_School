import json
import cv2
import os

mask_root = r'C:\Users\user\ai_school\220516\steel_masking\mask'

json_data = {}
for filename in os.listdir(mask_root):
    mask_path = os.path.join(mask_root, filename)
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    
    #json 파일 쓰는 작성하는 법
    
    # json annotation file 파일
    
    # { #딕셔너리
    #     filename: {
    #         'filename': 이미지 파일 이름,
    #         'width': 이미지 가로 길이(int),
    #         'height': 이미지 세로 길이(int),
    #         'anno': [
    #              [xmin, ymin, xmax, ymax] # int
    #             , ...
    #         ]
    #     }, ...
    # }

    height, width = mask.shape

    annos = []
    coutours, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #findContours 이미지 윤곽선 좌표 계산
    for coutour in coutours: #폴리곤으로 나오는데
        xs, ys = [], [] #bbox로 바꿔주기 위해 각각 값 저장

        for coord in coutour: 
            x, y = coord[0]
            xs.append(x)
            ys.append(y)

        xmin, xmax = min(xs), max(xs) #최소값
        ymin, ymax = min(ys), max(ys) #최대값 
        anno = [int(xmin), int(ymin), int(xmax), int(ymax)] 
        annos.append(anno) #어노테이션 정보 안에 bbox 정보 넣는 방식으로 

    json_image = { #이미지 하나에 대한 정보를 넣어주기
        'filename': filename,
        'width': width,
        'height': height,
        'anno': annos
    }
    json_data[filename] = json_image #전체 데이터 dict에다가 저장

sava_path = r'C:\Users\user\ai_school\220516\steel_masking\annotation.json'
with open(sava_path, 'w') as j: #w하는 식으로 파일 열어서 json으로 저장 
    json.dump(json_data, j, indent='\t')
