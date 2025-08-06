import cv2
import numpy as np 
import re
import os
import sys
import pathlib
from tqdm import tqdm

def main():
    omni_dir = pathlib.Path(sys.argv[1])
    text_paths = sorted(omni_dir.glob('labels/**/rgb_*.txt'))
    img_paths = sorted(omni_dir.glob('images/**/rgb_*.png'))
    for i,(text_path,img_path) in enumerate(tqdm(zip(text_paths,img_paths),total=len(text_paths),desc="処理中")):
        annotation_checker(text_path, img_path,omni_dir)


def annotation_checker(text_path,img_path,omni_dir):
    with open(text_path, "r") as txt_file:
        txt_data = txt_file.readlines()

    txt_split_data = []
    for txt in txt_data:
        txt_split_data.append(txt.split(' '))

    rect_points = [np.array([[float(element[1]),float(element[2])],
                           [float(element[5]),float(element[6])],
                           [float(element[7]),float(element[8])],
                           [float(element[3]),float(element[4])],]
                           ) for element in txt_split_data]
    
    rect_points = [np.clip(element,0,1) for element in rect_points]

    for element in rect_points:
        element[:,::2]  = element[:,::2] * 640
        element[:,1::2] = element[:,1::2] * 480

    rect_points = [element.astype(np.int32).reshape((-1,1,2)) for element in rect_points]

    img = cv2.imread(img_path)

    for rect_point in rect_points:
        cv2.polylines(img,[rect_point], isClosed=True,color=(0,255,0),thickness=1)
    
    num = int(re.findall(r"\d+",text_path.name)[0])
    
    if not os.path.exists(f"{omni_dir}/checker"):
        os.mkdir(f"{omni_dir}/checker")
    cv2.imwrite(f'{omni_dir}/checker/result_{num}.png',img)

if __name__ == "__main__":
    main()