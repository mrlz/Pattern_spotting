import numpy as np
import scipy.io as sio
import time
import cv2
from sklearn import svm
from skimage.feature import local_binary_pattern
from os import listdir
import os
from sklearn.externals import joblib
from multiprocessing import Pool
import itertools
from sklearn import preprocessing

def copy_image_to_folder(img, destination):
    cv2.imwrite(destination, img)

def crop_image(img, bbox):
    return img[bbox[0]:bbox[2],bbox[1]:bbox[3]]

def print_box(X,Y):
    box = X[0]
    prediction = X[1]
    idx = X[2]
    if (prediction == 0):
        copy_image_to_folder(crop_image(Y[0],box),Y[1] % str(idx))
    else:
        copy_image_to_folder(crop_image(Y[0],box),Y[2] % str(idx))

def print_boxes(img, boxes, test_prediction ,obj_path, bg_path):
    ft = time.time()
    pool = Pool(processes=4)
    pool_result = pool.map(Evaluation(print_box, [img, obj_path, bg_path]), itertools.izip(boxes,test_prediction,range(boxes.shape[0])))
    pool.close()
    pool.join()

class Evaluation1(object):
    def __init__(self, f,color_dimension,c_size, lbp_size, cheap_feature_bg):
        self.ff = f
        self.color_dimension = color_dimension
        self.color_vector_size = c_size
        self.lbp_vector_size = lbp_size
        self.cfb = cheap_feature_bg
    def __call__(self, x):
        return self.ff(x, self.color_dimension,self.color_vector_size, self.lbp_vector_size, self.cfb)

def reasonable_object(img,box):
    box_area = (box[2]/1.0-box[0]/1.0)*(box[3]/1.0-box[1]/1.0)
    if ((img.shape[0]/1.0)*(img.shape[1]/1.0) > 5000000.0):
        if (box_area > 500000.0):
            return 0

    total_area = img.shape[0]/5.0*img.shape[1]/1.0
    if (box_area < total_area):
        return 1
    return 0

def make_cheap_feature(box,img,color_dimension,c_size,lbp_size):
    n = c_size + lbp_size
    Feature = np.zeros(n)
    crop = crop_image(img, box)
    im_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(im_gray,8,2,method='uniform')
    Feature[0:lbp_size] = np.bincount(np.array(lbp.ravel() , dtype = np.int64) , minlength=lbp_size)
    Feature[lbp_size:n] = cv2.calcHist(crop, [0,1,2], None, [color_dimension,color_dimension,color_dimension], [0,256,0,256,0,256]).flatten()
    return Feature

def make_cheap_features(boxes,img,color_dimension,c_size,lbp_size,cfb):
    imagecv = cv2.imread(img)
    features = []
    for box in boxes:
        if (reasonable_object(imagecv,box) == 0):
            features.append(cfb)
        else:
            features.append(make_cheap_feature(box,imagecv,color_dimension,c_size,lbp_size))
    features = np.array(features)
    features = preprocessing.normalize(features, norm = 'l2')
    return features

def make_filtered_boxes(X,color_dimension,c_size,lbp_size,cfb):
    SVC = joblib.load('background_svm_dump.pkl')
    d = './DocExplore_images/%s'
    tpre = time.time()
    boxes = X[0]
    object_estimate = SVC.predict(make_cheap_features(boxes,d % X[1],color_dimension,c_size,lbp_size,cfb))
    elements = boxes.shape[0]
    Acc = []
    j = 0
    while (j < elements):
        if (object_estimate[j] == 0):
            Acc.append(boxes[j])
        j = j + 1
    tpre = time.time() - tpre
    Acc = np.array(Acc)
    print("image:" + X[1] +" elements: " + str(boxes.shape[0]) +" clean: " + str(Acc.shape[0]) + " time: "+ str(tpre))
    return Acc

# save_obj_path = './segmentation/%s'
# save_bg_path = './segmentation/%s'

color_size = 8
color_vector_size = color_size**3

i = 0
radius = 2
no_points = 8
vector_size = no_points+2

mat_contents = sio.loadmat('boxescomplete.mat')
boxes = mat_contents['boxes'][0]

#load number of elements per box
with open('boxes_elements.txt','r') as f:
    sizes = np.array(f.read().splitlines())

with open('bg_cheap_feature','r') as f:
    cf = np.load(f)

t = time.time()
with open('docexplore_pages.txt', 'r') as f:
    lines = f.read().splitlines()
    clean_boxes = []
    t_it = time.time()

    # save_bg = save_bg_path % str(i+1) + "/bg/%s.png"
    # save_obj = save_obj_path % str(i+1) + "/obj/%s.png"

    print("Constructing clean boxes")
    ft = time.time()
    pool = Pool(processes=12)
    pool_result = pool.map(Evaluation1(make_filtered_boxes,color_size,color_vector_size, vector_size, cf), itertools.izip(boxes,lines))
    pool.close()
    pool.join()
    print("done creating clean boxes")

    for line, result in enumerate(pool_result):
        clean_boxes.append(result)
    ft = time.time() - ft

    clean_boxes = np.array(clean_boxes)
    with open('clean_boxes','w') as fboxes:
        np.save(fboxes,clean_boxes)

t = time.time() -t
print("building took: " + str(t))
