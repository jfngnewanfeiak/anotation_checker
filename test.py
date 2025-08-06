import cv2
import numpy as np
import re
import os

# text file open
txt_data = ""
txt_path = "rgb_0014.txt"
with open(txt_path, "r") as txt_file:
    txt_data = txt_file.readlines()
    pass

# text data split
txt_split_data = []
for txt in txt_data:
    txt_split_data.append(txt.split(" "))



#
rect_points = [np.array([[float(element[1]),float(element[2])],
                           [float(element[5]),float(element[6])],
                           [float(element[7]),float(element[8])],
                           [float(element[3]),float(element[4])],]
                           ) for element in txt_split_data]

rect_points = [np.clip(element,0,1) for element in rect_points]

rect_points = [element for element in rect_points]
for element in rect_points:
    element[:,::2]  = element[:,::2]  * 640
    element[:,1::2] = element[:,1::2] * 480

rect_points = [element.astype(np.int32).reshape((-1,1,2)) for element in rect_points]


# img file open
img = cv2.imread("rgb_0014.png")


for rect_point in rect_points:
    cv2.polylines(img,[rect_point],isClosed=True,color=(0,255,0),thickness=1)

num = int(re.findall(r"\d+", txt_path)[0])
os.mkdir("checker")
cv2.imwrite(f'checker/result_{num}.png',img)