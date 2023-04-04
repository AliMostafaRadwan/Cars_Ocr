# import pytesseract
# from PIL import Image
# import os

# print(pytesseract.image_to_string(Image.open('pic/11.jpg'), lang="ara"))



# import cv2
# import numpy as np 
# import pandas as pd 
# import matplotlib.pyplot as plt 
# import imutils
# import easyocr


# # read the image
# img = cv2.imread('pic/9.jpg')
# ocrt = easyocr.Reader(['ar'])
# results = ocrt.readtext(img)
# numbers = results[1][-2]
# letters = results[2][1]
# print(f"car numbers = {numbers} , car letters {letters}")



from paddleocr import PaddleOCR
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='ar') # need to run only once to download and load model into memory
img_path = 'roi.jpg'
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line[1][0], line[1][1])


