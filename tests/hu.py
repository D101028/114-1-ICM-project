import cv2
import numpy as np

import time

def hu_similarity(img1, img2):
    # 轉灰階 & 二值化
    _, img1_bin = cv2.threshold(img1, 128, 255, cv2.THRESH_BINARY)
    _, img2_bin = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY)
    
    # 找輪廓
    contours1, _ = cv2.findContours(img1_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(img2_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 計算 Hu 矩
    hu1 = cv2.HuMoments(cv2.moments(contours1[0])).flatten()
    hu2 = cv2.HuMoments(cv2.moments(contours2[0])).flatten()
    
    # 計算對數距離
    # hu1 = -np.sign(hu1)*np.log10(np.abs(hu1)+1e-10)
    # hu2 = -np.sign(hu2)*np.log10(np.abs(hu2)+1e-10)
    
    distance = np.linalg.norm(hu1 - hu2)
    return distance

img1 = cv2.imread('latex_symbols/a.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('out_symbols/subcap.png', cv2.IMREAD_GRAYSCALE)

start = time.time()
score = hu_similarity(img1, img2)
print(time.time() - start)

print("Hu moment distance:", score)
