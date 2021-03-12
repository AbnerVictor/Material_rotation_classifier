import os
import cv2
import numpy as np

import csv
import codecs

dataset = './dataset/test_set_2'

data_label = []
data_label_file_path = os.path.join(dataset, 'data_label.csv')

for img_name in os.listdir(dataset):
    name, suffix = os.path.splitext(img_name)
    if suffix == '.jpg':

        img = cv2.imread(os.path.join(dataset, img_name), flags=cv2.IMREAD_GRAYSCALE)
        print(img.shape)

        path_0 = os.path.join(dataset, '0')
        path_90 = os.path.join(dataset, '90')
        path_180 = os.path.join(dataset, '180')
        path_270 = os.path.join(dataset, '270')

        def makedir(path):
            if not os.path.isdir(path):
                os.mkdir(path)

        makedir(path_0)
        makedir(path_90)
        makedir(path_180)
        makedir(path_270)

        (h, w) = img.shape
        center = (w//2, h//2)

        half_smaller_side = min(h, w) // 2

        cv2.imwrite(os.path.join(path_0, img_name), img[center[1]-half_smaller_side:center[1]+half_smaller_side, center[0]-half_smaller_side: center[0]+half_smaller_side])

        M = cv2.getRotationMatrix2D(center, 90, 1.0)
        img_90 = cv2.warpAffine(img, M, (w, h))[center[1]-half_smaller_side:center[1]+half_smaller_side, center[0]-half_smaller_side: center[0]+half_smaller_side]
        cv2.imwrite(os.path.join(path_90, img_name), img_90)

        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        img_180 = cv2.warpAffine(img, M, (w, h))[center[1]-half_smaller_side:center[1]+half_smaller_side, center[0]-half_smaller_side: center[0]+half_smaller_side]
        cv2.imwrite(os.path.join(path_180, img_name), img_180)

        M = cv2.getRotationMatrix2D(center, 270, 1.0)
        img_270 = cv2.warpAffine(img, M, (w, h))[center[1]-half_smaller_side:center[1]+half_smaller_side, center[0]-half_smaller_side: center[0]+half_smaller_side]
        cv2.imwrite(os.path.join(path_270, img_name), img_270)
