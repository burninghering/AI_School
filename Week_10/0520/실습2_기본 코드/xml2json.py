#example json 파일 확인하며 코드 작성하자

import os
from xml.etree import cElementTree as ET #xml 읽어오기
import json
import os

def main():
    #(3)
    xml_root=r''
    xml_paths=get_xml_paths(xml_root)

    #(9)json 데이터 저장할 곳 만들기
    json_data=[] #(22) xml이 몇개든 한 파일로 만들어서 실행될것임

    #(23) 저장하기
    json_path=r''


    #(4)
    for xml_path in xml_paths: 
        root=ET.parse(xml_path)
        images = root.findall('image')
        for image in images:
            #(5)하나의 꺽쇠 안에 있는 것은 attrib 사용해야함
            name=image.attrib['name']
            width=int(image.attrib['width']) #(6)문자를 숫자로 저장하자
            height=int(image.attrib['height'])

            #(12)json_annos 넣을 곳도 만들어주자
            json_annos={}

            #(7)polyline이 여러개 있으므로 findall로 가져오자
            annos=image.findall('polyline')
            for anno in annos:
                #(8)points 가져오기
                label=anno.attrib['label']
                points=anno.attrib['points']


                #(14) 점 4개 들어갈 곳 만들어주자
                json_points=[]
                for xy in points.split(';'):
                    x,y=xy.split(';') #(15) [x,y] 로 값이 나오는걸 각각 넣어주자
                    
                    #(16)
                    x=float(x)
                    y=float(y)

                    #(17)넣어주기
                    json_points.append(x)
                    json_points.append(y)

                json_annos[label]=json_points
#            print(json_annos) #(18) 중간 중간 프린트하며 확인해보자
        
                
                #print(points.split(';')) #(8)x와 y를 분리시키자
                #exit()
            
            #(10)json에 입력할 정보를 저장해주자
            json_image={
                'filename':name,
                'width':width,
                'height':height,
                #(11)annotation 정보 넣어주기
                #(13)
                'annotation':json_annos
            }

            #(21)
            json_data.append(json_image)


            #(20)
#            print(json_image)
 #           exit()

    #(24)읽기모드로 저장하기
    with open(json_path,'w') as j:
        json.dump(json_data,j,indent='\t') #indent는 줄 띄어주는것임, 안넣으면 일렬로 나와서 너무 보기 힘듦


#(2)이전에 썼던 함수 가져와서 조금만 고치기
def get_xml_paths(root):
    paths = []
    for (path, dir, files) in os.walk(root):
        for file in files:
            ext = file.split('.')[-1]
            if ext in ['xml']:
                file_path = os.path.join(path, file)
                paths.append(file_path)
    return paths

#(1)
if __name__:'__main__':
    main()



#--------------------------------------------
from xml.etree import ElementTree as ET
import json
import os

def main():
    xml_root = r'C:\Users\user\ai_school\220516_20\20_dataset\anno_file'
    xml_paths = get_xml_paths(xml_root)

    json_path = r'C:\Users\user\ai_school\220516_20\20_dataset\anno_file\annotations.json'
    json_data = []
    for xml_path in xml_paths:
        root = ET.parse(xml_path)
        images = root.findall('image')
        for image in images:
            name = image.attrib['name']
            width = int(image.attrib['width'])
            height = int(image.attrib['height'])

            json_annos = {}
            annos = image.findall('polyline')
            for anno in annos:
                label = anno.attrib['label']
                points = anno.attrib['points']

                json_points = []
                for xy in points.split(';'):
                    x, y = xy.split(',')
                    x = float(x)
                    y = float(y)

                    json_points.append(x)
                    json_points.append(y)
                json_annos[label] = json_points

            json_image = {
                'filename': name,
                'width': width,
                'height': height,
                'annotations': json_annos
            }
            json_data.append(json_image)

    with open(json_path, 'w') as j:
        json.dump(json_data, j, indent='\t')


def get_xml_paths(root):
    paths = []
    for (path, dir, files) in os.walk(root):
        for file in files:
            ext = file.split('.')[-1]
            if ext in ['xml']:
                file_path = os.path.join(path, file)
                paths.append(file_path)
    return paths

if __name__ == '__main__':
    main()