# -*- coding: utf-8 -*-
import sys
import imp
import cv2

imp. reload(sys)
'''
读取图片

只需要给出待操作的图片的路径即可。
'''
image = cv2.imread(r'./timg.jpg')

'''
显示图像

编辑完的图像要么直接的被显示出来，要么就保存到物理的存储介质。
'''
cv2.imshow("MyImage",image)
cv2.waitKey(0)
print(1)

'''
灰度转换

灰度转换的作用就是：转换成灰度的图片的计算强度得以降低。
'''
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("MyImage",gray)
cv2.waitKey(0)
print(2)

'''
画图

opencv 的强大之处的一个体现就是其可以对图片进行任意编辑，处理。 
下面的这个函数最后一个参数指定的就是画笔的大小。
'''
#cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)

'''获取人脸识别训练数据

看似复杂，其实就是对于人脸特征的一些描述，这样opencv在读取完数据后很据训练中的样品数据，就可以感知读取到的图片上的特征，进而对图片进行人脸识别。
'''
face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')

'''
探测人脸

说白了，就是根据训练的数据来对新图片进行识别的过程。
'''

# 探测图片中的人脸

faces = face_cascade.detectMultiScale(
   gray,
   scaleFactor = 1.15,
   minNeighbors = 5,
   minSize = (5,5),
   flags = cv2.CAP_OPENNI_VALID_DEPTH_MASK
)

'''
我们可以随意的指定里面参数的值，来达到不同精度下的识别。返回值就是opencv对图片的探测结果的体现。

处理人脸探测的结果

结束了刚才的人脸探测，我们就可以拿到返回值来做进一步的处理了。但这也不是说会多么的复杂，无非添加点特征值罢了
'''

print( "发现{0}个人脸!".format(len(faces)))

for(x,y,w,h) in faces:
   cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)