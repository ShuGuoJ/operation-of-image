'''图像的几何变换：平移，水平翻转，垂直翻转，旋转'''
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class Image(object):
    def __init__(self):
        super(Image,self).__init__()

    def Move(self, img, delta_x, delta_y):
        transform = np.array([[1,0,delta_x],[0,1,delta_y]],dtype=np.float)
        h, w, c= img.shape
        # x_axi:[h,1]
        # y_axi:[1,w]
        x_axi = np.arange(w).reshape(1,w)
        y_axi = np.arange(h).reshape(h,1)
        x_axi, y_axi = np.repeat(x_axi, h, axis=0), np.repeat(y_axi, w, axis=1)
        # xy_axi:[h,w,3]
        xy_coordinates = np.stack([x_axi, y_axi, np.ones((h,w))], axis=-1)
        # coordinates:[h*w,3]
        coordinates = xy_coordinates.reshape(h*w, -1)
        # new_coordinates:[2,h*w]
        new_coordinates = transform@np.transpose(coordinates)
        # xy_coordinates = new_coordinates.t().reshape(h,w,-1)
        dst = np.zeros((h,w,c))
        for i in range(h*w):
            j, k = i//w, i%w
            x, y = new_coordinates[0, i], new_coordinates[1, i]
            if x>=0 and x<w and y>=0 and y<h:
                dst[int(y), int(x)] = img[j, k]
        return dst.astype(np.uint8)

    def Horizontal(self, img):
        h,w,c = img.shape
        transform = np.array([[-1,0,w],[0,1,0]])
        x_axi = np.arange(w).reshape(1,w)
        y_axi = np.arange(h).reshape(h,1)
        x_axi, y_axi = np.repeat(x_axi, h, axis=0), np.repeat(y_axi, w, axis=1)
        xy_coordinates = np.stack([x_axi, y_axi, np.ones((h,w))], axis=-1)
        coordinates = xy_coordinates.reshape(h*w, -1)
        new_coordinates = transform@np.transpose(coordinates)
        dst = np.zeros((h,w,c))
        for i in range(h*w):
            j, k = i//w, i%w
            x, y = new_coordinates[0,i], new_coordinates[1,i]
            if x>=0 and x<w and y>=0 and y<h:
                dst[int(y), int(x)] = img[j, k]
        return dst.astype(np.uint8)

    def Vertically(self, img):
        h,w,c = img.shape
        transform = np.array([[1,0,0],[0,-1,h]], dtype=np.float)
        x_axi = np.arange(w).reshape(1,w)
        y_axi = np.arange(h).reshape(h,1)
        x_axi, y_axi = np.repeat(x_axi,h,axis=0), np.repeat(y_axi,w,axis=1)
        xy_coordinates = np.stack([x_axi, y_axi, np.ones((h,w))], axis=-1)
        coordinates = xy_coordinates.reshape(h*w, -1)
        new_coordinates = transform@np.transpose(coordinates)
        dst = np.zeros((h,w,c))
        for i in range(h*w):
            j, k = i//w, i%w
            x, y = new_coordinates[0, i], new_coordinates[1, i]
            if x>=0 and x<w and y>=0 and y<h:
                dst[int(y), int(x)] = img[j, k]
        return dst.astype(np.uint8)

    '''由于旋转的过程中存在着像素点丢失的情况，所以导致了旋转后的图像中会出现黑点'''
    def Rotate(self, img, angle):
        beta = angle * np.pi / 180.
        h,w,c = img.shape
        transform = np.array([[np.cos(beta), np.sin(beta), 0],[-np.sin(beta), np.cos(beta), 0]], dtype=np.float)
        x_axi = np.arange(w).reshape(1,w)
        y_axi = np.arange(h).reshape(h,1)
        x_axi, y_axi = np.repeat(x_axi, h, axis=0), np.repeat(y_axi, w, axis=1)
        xy_coordinates = np.stack([x_axi, y_axi, np.ones((h,w))], axis=-1)
        coordinates = xy_coordinates.reshape(h*w,-1)
        new_coordinates = transform@np.transpose(coordinates)
        dst = np.zeros((h,w,c))
        for i in range(h*w):
            j, k = i//w, i%w
            x, y = int(new_coordinates[0,i]), int(new_coordinates[1,i])
            if x>=0 and x<w and y>=0 and y<h:
                dst[int(y), int(x)] = img[j, k]
        return dst.astype(np.uint8)


    '''双线性插入防止旋转过程中像素点的丢失而导致的图像黑点问题'''
    def Rotate_bilinear_interploate(self, img, angle):
        beta = angle*np.pi/180
        h,w,c = img.shape
        # transform:[2,2]
        transform = np.array([[np.cos(beta),-np.sin(beta)],[np.sin(beta),np.cos(beta)]], dtype=np.float)
        # x_axis:[1,w]
        # y_axis:[h,1]
        x_axis = np.arange(w).reshape((1,w))
        y_axis = np.arange(h).reshape((h,1))
        x_axis = np.repeat(x_axis,h,axis=0)
        y_axis = np.repeat(y_axis,w,axis=1)
        # xy_coordinates:[h,w,2]
        xy_coordinates = np.stack([x_axis, y_axis], axis=-1)
        coordinates = xy_coordinates.reshape((-1, 2))
        # new_coordinates:[2,h*w]
        new_coordinates = transform@coordinates.T
        Q_00 = np.floor(new_coordinates).astype(np.int)
        Q_00[0] = np.clip(Q_00[0], 0, w-2)
        Q_00[1] = np.clip(Q_00[1], 0, h-2)
        dst = np.zeros((h,w,c))
        for i in range(h*w):
            j, k = i//w, i%w
            x, y = new_coordinates[0, i], new_coordinates[1, i]
            if x>=0 and x<w and y>=0 and y<h:
                dst[j,k] = self.Bilinear_interploate(img, y, x, Q_00[1,i], Q_00[0,i])
        return dst.astype(np.uint8)

    def Bilinear_interploate(self, img, h, w, h_0, w_0):
        h_1, w_1 = h_0+1, w_0+1
        u,v = h-h_0, w-w_0
        return (1-u)*(1-v)*img[h_0, w_0] + (1-u)*v*img[h_0, w_1] + u*(1-v)*img[h_1,w_0] + u*v*img[h_1,w_1]

# t = Image()
# img = cv.imread('img/liuyifei_1.jpg')
# delta_x = 10
# delta_y = 20
# M = np.array([[1,0,delta_x],[0,1,delta_y]], dtype=np.float)
# dst_1 = t.Rotate(img, 15)
# # dst_1 = cv.warpAffine(img, M, tuple(img.shape[:2][::-1]))
# # dst_2 = t.Move(img, delta_x, delta_y)
# # dst_2 = t.Horizontal(img)
# # dst_2 = t.Vertically(img)
# dst_2 = t.Rotate_bilinear_interploate(img, 15)
# fig, axis = plt.subplots(1,2,figsize=(10,10))
# axis[0].imshow(dst_1[...,::-1])
# axis[1].imshow(dst_2[...,::-1])
# plt.show()


# t = Image()
# img = cv.imread('img/liuyifei_1.jpg')
# # img = np.ones((2,2,3),dtype=np.uint8)*255
# dst = t.Rotate_bilinear_interploate(img, 15)
# fig, axis = plt.subplots(1,2,figsize=(10,10))
# axis[0].imshow(img[...,::-1])
# axis[1].imshow(dst[...,::-1])
# plt.show()

# img = cv.imread('img/liuyifei_1.jpg')
# M = cv.getRotationMatrix2D((0,0),45,1)
# dst = cv.warpAffine(img, M, tuple(img.shape[:2])[::-1])
# fig, axis = plt.subplots(1,2,figsize=(10,10))
# axis[0].imshow(img[...,::-1])
# axis[1].imshow(dst[...,::-1])
# plt.show()

