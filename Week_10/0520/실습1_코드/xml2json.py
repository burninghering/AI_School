from xml.etree import ElementTree as ET
import json
import os

def main():
    xml_root = r'C:\Users\user\ai_school\220516_20\20_dataset\anno_file'
    xml_paths = get_xml_paths(xml_root)

    save_path = r'C:\Users\user\ai_school\220516_20\20_dataset\anno_file\annotations.json'
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
                json_annos[label] = []

                points = anno.attrib['points']
                for xy in points.split(';'):
                    x, y = xy.split(',')
                    json_annos[label].extend([float(x), float(y)])

            json_image = {
                'filename': name,
                'width': width,
                'height': height,
                'annotations': json_annos
            }
            json_data.append(json_image)

    with open(save_path, 'w', encoding='utf-8') as j:
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