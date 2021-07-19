import cv2
import sys
from de import *

inp = sys.argv[1]
name =  sys.argv[2]
img = cv2.imread(inp).astype('float32')

dehazed_image = dehaze(img)
dehazed_image = adjust(dehazed_image)

save = inp.replace(name,"result.png")
cv2.imwrite(save,dehazed_image*255)
