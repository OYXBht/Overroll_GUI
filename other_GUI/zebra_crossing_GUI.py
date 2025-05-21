import cv2
import numpy as np

kernel_Ero = np.ones((3,1),np.uint8)
kernel_Dia = np.ones((5,1),np.uint8)
img = cv2.imread("../images/img_7.png")
copy_img = img.copy()
copy_img = cv2.resize(copy_img,(1600,800))
cv2.imshow('copy_img',copy_img)
cv2.waitKey(0)
# 图像灰度化
gray=cv2.cvtColor(copy_img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
cv2.waitKey(0)
# 高斯滤波
imgblur=cv2.GaussianBlur(gray,(5,5),10)
cv2.imshow('imgblur',imgblur)
cv2.waitKey(0)
#阈值处理
ret,thresh=cv2.threshold(imgblur,200,255,cv2.THRESH_BINARY)
cv2.imshow('thresh',thresh)
cv2.waitKey(0)
#腐蚀
img_Ero=cv2.erode(thresh,kernel_Ero,iterations=3)
cv2.imshow('img_Ero',img_Ero)
cv2.waitKey(0)
#膨胀
img_Dia=cv2.dilate(img_Ero,kernel_Dia,iterations=1)
cv2.imshow('img_Dia',img_Dia)
cv2.waitKey(0)
#轮廓检测
contouts,h = cv2.findContours(img_Dia,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cnt = contouts
# img=cv2.drawContours(copy_img, cnt, -1, (0, 255, 0), 2)
# cv2.imshow('img',img)
# cv2.waitKey(0)
for i in cnt:
    #坐标赋值
    x,y,w,h = cv2.boundingRect(i)
    print(x,y,w,h)
    if x>1000 and w>50 and h>10:
        out=cv2.drawContours(copy_img,i,-1,(0,255,0),3)
cv2.imshow('out', out)
cv2.waitKey(0)
