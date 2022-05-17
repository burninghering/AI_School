import os
import json
import cv2
import numpy as np

image_root = r'C:\Users\user\ai_school\220516_20\17_automotive_engine\automotive_engine\image' #이미지 파일 경로
json_root = r'C:\Users\user\ai_school\220516_20\17_automotive_engine\automotive_engine\json'

#이미지 파일과 json파일이 매칭되는 것이 서로 같은 이름으로 되어있기 때문에,

for filename in os.listdir(image_root):
     
    image_path = os.path.join(image_root, filename)
    # 이미지 경로 만들기 ^
    image = cv2.imread(image_path)

#파일 이름을 가져와서 jpg를 json으로 변경하자
    filename_json = filename.split('.')[0] + '.json'
    json_path = os.path.join(json_root, filename_json)
    # json 경로 만들기 ^

    with open(json_path, 'r') as j:
        json_data = json.load(j) #json으로 된 것 다 읽어오기

        

    annos = json_data['shapes'] #키값 가져와서 어노테이션 정보 읽어오고
    for anno in annos: #어노테이션에서 딕셔너리 하나씩 가지고 와서 읽음
        points = anno['points'] # [ [x1, y1], [x2, y2], ... ]
        points = np.array(points, np.int)

        image = cv2.polylines(image, [points], True, (255, 255, 0), 3)

    cv2.imshow('visual', image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()