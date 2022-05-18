import json
import os
import cv2


def main():
    # json 파일 읽어서 bbox 정보 가져와서 이미지 crop하기
    
    # 이미지 경로 지정
    image_root=r'C:\Users\hyerin\AI_School\Week_10\0518\18_airport\image'
    json_path=r'C:\Users\hyerin\AI_School\Week_10\0518\18_airport\annotation.json'
    save_root=r'C:\Users\hyerin\AI_School\Week_10\0518\18_airport\cropped'
    labels=['knife','gun']

    # 폴더 만들기
    for label in labels:
        save_forder=os.path.join(save_root,label) #label에 있는 요소들 폴더가 생김
        os.makedirs(save_forder,exist_ok=True)

    # json 파일 읽기
    with open(json_path,'r') as j:
        json_data=json.load(j)

    for filename in os.listdir(image_root):

        #파일 경로 불러오기
        file_path=os.path.join(image_root,filename)

        image=cv2.imread(file_path)

        json_image=json_data[filename]
        annos=json_image['anno']
        for idx, anno in enumerate(annos):
            lab=anno['label'].lower()
            xmin,ymin,xmax,ymax=anno['bbox']

            # 이미지 크롭하기(슬라이싱할 때 세로부터 해줌)
            image_crop=image[ymain:ymax,xmin:xmax]
            file = filename.split('.')[0]
            filename_new=f'{file}_{idx}.png'
            print(filename_new)
            exit()

            # 크롭한 이미지들 저장하기
            save_path=os.path.join(save_root,lab,filename)
            
            # 이미지 save
            cv2.imwrite(save_path,image_crop)

            cv2.imshow('crop',image_crop)
            if cv2.waitKey(0)&0xff==ord('q'):
                cv2.destroyAllwindows()

            

        print(filename)
        print(json_data[filename])
        exit()


if __name__=='__main__':
    main()