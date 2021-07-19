import cv2
import sys
from de import *

def fun(inp,name):
    img = cv2.imread(inp).astype('float32')

    dehazed_image = dehaze(img)
    dehazed_image = adjust(dehazed_image)

    save = inp.replace(name,"result.jpeg")
    cv2.imwrite(save,dehazed_image*255)
