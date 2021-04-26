'''最近邻插值法'''
'''implementation of nearest interpolate'''
import numpy as np

def nearest_interpolate(img, size):
    h, w = img.shape[:2]
    t_h, t_w = size
    hs_p = (np.arange(t_h) + 0.5) * h // t_h
    ws_p = (np.arange(t_w) + 0.5) * w // t_w
    hs_p = hs_p.reshape((t_h, 1))
    ws_p = ws_p.reshape((1, t_w))
    hs_p = np.repeat(hs_p, t_w, axis=1).astype(np.int32)
    ws_p = np.repeat(ws_p, t_h, axis=0).astype(np.int32)
    # candidates = np.stack([hs_p, ws_p], axis=-1).astype(np.int32)
    return img[hs_p, ws_p]


import cv2 as cv
from matplotlib import pyplot as plt
import time
img = cv.imread('img/liuyifei.jpg')
begin = time.time()
size = (750,1000)
dst_1 = cv.resize(img, size[::-1])
end = time.time()
print('1_consume:{}'.format(end-begin))
begin = time.time()
dst_2 = nearest_interpolate(img, size)
end = time.time()
print('2_consume:{}'.format(end-begin))
# fig, axis = plt.subplots(1,3,figsize=(10,10))
# axis[0].imshow(img[...,::-1])
# axis[1].imshow(dst_1[...,::-1])
# axis[2].imshow(dst_2[...,::-1])
# plt.show()
plt.imshow(dst_1[...,::-1])
plt.savefig('dst1.jpg')
plt.clf()
plt.imshow(dst_2[...,::-1])
plt.savefig('dst2.jpg')

a = np.zeros((2,2))
