#json을 읽어서 yolo형식인 txt로 만들기
import os
import json

save_root = r'C:\Users\user\ai_school\220516\steel_masking\anno'
json_path = r'C:\Users\user\ai_school\220516\steel_masking\annotation.json'

with open(json_path, 'r') as j: #with로 열면 자동으로 close
    json_data = json.load(j)

for filename in json_data.keys():
    filename_txt = filename.split('.')[0] + '.txt'
    save_path = os.path.join(save_root, filename_txt)

    f = open(save_path, 'w') #file을 오픈으로 열면 꼭 닫아줘야함

    json_image = json_data[filename]
    width = json_image['width']
    height = json_image['height']
    annos = json_image['anno']
    for anno in annos:
        xmin, ymin, xmax, ymax = anno

        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        bbox_w = xmax - xmin
        bbox_h = ymax - ymin

        x_center = x_center / width #비율에 대한 값으로 저장
        bbox_w = bbox_w / width

        y_center = y_center / height
        bbox_h = bbox_h / height

        write = f'0 {x_center} {y_center} {bbox_w} {bbox_h}\n' #클래스 정보 통일하기 
        f.write(write) #텍스트 파일 저장 
    f.close() #파일 안닫으면 파일이 손상될 수도 있음