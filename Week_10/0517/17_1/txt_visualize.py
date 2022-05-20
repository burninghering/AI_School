import os
import cv2

image_root = r'C:\Users\user\ai_school\220516_20\17_automotive_engine\automotive_engine\image'
txt_root = r'C:\Users\user\ai_school\220516_20\17_automotive_engine\automotive_engine\txt'

for filename in os.listdir(image_root):
    image_path = os.path.join(image_root, filename)
    image = cv2.imread(image_path)
    height, width, channel = image.shape

    filename_txt = filename.split('.')[0] + '.txt'
    txt_path = os.path.join(txt_root, filename_txt)

    with open(txt_path, 'r') as f: #file을 읽고 
        while True: 
            line = f.readline()[:-2] #한 줄씩 읽기
            if not line:
                break
            line = line.split(' ') #한 줄씩
            x_center, y_center, w, h = line[1:]
            x_center = float(x_center) #형 변환
            y_center = float(y_center)
            w = float(w)
            h = float(h)
               
            #Yolo 형식 : 전체 이미지에 대한 비율 x center/y center, w, h 어디에 위치한지 확인한 후 비율로 확인
            x_center = x_center * width #가로길이 곱하기
            y_center = y_center * height #세로길이 곱하기
            w = w * width
            h = h * height

            xmin = int(x_center - w/2) #섹터에서 반을 나눈 값을 빼줘야함
            ymin = int(y_center - h/2)
            xmax = int(x_center + w/2)
            ymax = int(y_center + h/2)
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 0), 3)

    cv2.imshow('visualize', image)
    if cv2.waitKey(0) & 0xff == ord('q'):
        cv2.destroyAllWindows()
        exit()
