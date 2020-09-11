'''implementation of bilinear interploate'''
import numpy as np

def bilinear_interploate(img, size):
    h,w,c = img.shape
    t_h, t_w = size[0], size[1]
    # 映射
    # hs_p:[t_h]
    # ws_p:[t_w]
    h_scale = h / t_h
    w_scale = w / t_w
    hs_p = (np.arange(t_h)+0.5)*h_scale-0.5
    ws_p = (np.arange(t_w)+0.5)*w_scale-0.5
    hs_p = np.clip(hs_p, 0, h-1)
    ws_p = np.clip(ws_p, 0, w-1)
    # 最近投影坐标
    hs_0 = np.clip(np.floor(hs_p), 0, h-2).astype(np.int)
    ws_0 = np.clip(np.floor(ws_p), 0, w-2).astype(np.int)

    dst = np.zeros((t_h, t_w, c))
    us = hs_p - hs_0
    vs = ws_p - ws_0
    for i in range(t_h):
        for j in range(t_w):
            h_0, h_1 = hs_0[i], hs_0[i]+1
            w_0, w_1 = ws_0[j], ws_0[j]+1
            u, v = us[i], vs[j]
            dst[i,j] = (1-u)*(1-v)*img[h_0,w_0] + (1-u)*v*img[h_0,w_1] + u*(1-v)*img[h_1,w_0] + u*v*img[h_1,w_1]
    return dst.astype(np.uint8)

# 矩阵运算代替循环
def bilinear_interploate_1(img, size):
    h, w, c = img.shape
    t_h, t_w = size[0], size[1]
    # 映射
    # hs_p:[t_h]
    # ws_p:[t_w]
    h_scale = h / t_h
    w_scale = w / t_w
    hs_p = (np.arange(t_h).reshape(t_h, 1) + 0.5) * h_scale - 0.5
    ws_p = (np.arange(t_w).reshape(1, t_w) + 0.5) * w_scale - 0.5
    hs_p = np.clip(hs_p, 0, h - 1)
    ws_p = np.clip(ws_p, 0, w - 1)
    # hs_p:[t_h, t_w]
    # ws_p:[t_h, t_w]
    hs_p = np.repeat(hs_p, t_w, axis=1)
    ws_p = np.repeat(ws_p, t_h, axis=0)
    # 最近投影坐标
    # [t_h, t_w]
    hs_0 = np.clip(np.floor(hs_p), 0, h - 2).astype(np.int)
    ws_0 = np.clip(np.floor(ws_p), 0, w - 2).astype(np.int)
    hs_1 = hs_0 + 1
    ws_1 = ws_0 + 1
    # 临近像素点
    # [t_h, t_w, channel]
    f00 = img[hs_0, ws_0]
    f01 = img[hs_0, ws_1]
    f10 = img[hs_1, ws_0]
    f11 = img[hs_1, ws_1]
    # 权重
    # [t_h, t_w]
    u = hs_p - hs_0
    v = ws_p - ws_0
    w00 = (1 - u) * (1 - v)
    w01 = (1 - u) * v
    w10 = u * (1 - v)
    w11 = u * v
    w00, w01, w10, w11 = np.expand_dims(w00, axis=-1), np.expand_dims(w01, axis=-1), np.expand_dims(w10,
                                                                                                    axis=-1), np.expand_dims(
        w11, axis=-1)
    dst = w00 * f00 + w01 * f01 + w10 * f10 + w11 * f11
    # for i in range(t_h):
    #     for j in range(t_w):
    #         h_0, h_1 = hs_0[i], hs_0[i] + 1
    #         w_0, w_1 = ws_0[j], ws_0[j] + 1
    #         u, v = us[i], vs[j]
    #         dst[i, j] = (1 - u) * (1 - v) * img[h_0, w_0] + (1 - u) * v * img[h_0, w_1] + u * (1 - v) * img[
    #             h_1, w_0] + u * v * img[h_1, w_1]
    return dst.astype(np.uint8)


import cv2 as cv
from matplotlib import pyplot as plt
import time
img = cv.imread('img/liuyifei_1.jpg')
begin = time.time()
size = (500,300)
dst_1 = cv.resize(img, size[::-1])
end = time.time()
print('1_consume:{}'.format(end-begin))
begin = time.time()
dst_2 = bilinear_interploate(img, size)
end = time.time()
print('2_consume:{}'.format(end-begin))
begin = time.time()
dst_3 = bilinear_interploate_1(img, size)
end = time.time()
print('3_consume:{}'.format(end-begin))
fig, axis = plt.subplots(1,4,figsize=(10,10))
axis[0].imshow(img[...,::-1])
axis[1].imshow(dst_1[...,::-1])
axis[2].imshow(dst_2[...,::-1])
axis[3].imshow(dst_3[...,::-1])
plt.show()

a = np.zeros((2,2))
