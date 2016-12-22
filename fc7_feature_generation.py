import numpy as np
import time
from os import listdir
import os
import caffe
from multiprocessing import Pool
import itertools
from sklearn import preprocessing

def crop_image(img, bbox):
    return img[bbox[0]:bbox[2],bbox[1]:bbox[3]]

def make_fc7_features(X):
    tpre = time.time()
    d = './DocExplore_images/%s'
    boxes = X[0]
    image = caffe.io.load_image(d % X[1])
    page = X[2]
    # image = caffe.io.load_image(img)
    net = caffe.Net('models/bvlc_reference_caffenet/deploy.prototxt',
                    'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                    caffe.TEST)

    # load input and configure preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    mu = np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy') # load the mean imagenet image
    mu = mu.mean(1).mean(1) # average over pixels to obtain the mean (BGR) pixel values
    transformer.set_mean('data', mu) #subtract the dataset-mean value in each channel
    transformer.set_transpose('data', (2,0,1)) #images are usually loaded as H x W x C, but we need them to be C x H x W
    transformer.set_channel_swap('data', (2,1,0)) #swap channels from RGB to BGR
    transformer.set_raw_scale('data', 255.0) #caffe.io.load_image loads in normalized form (0,1), we provide 255 to indicate max value so that it can be rescaled

    # print("DATA SHAPE:")
    # print(net.blobs['data'].data.shape)
    # #note we can change the batch size on-the-fly
    # #since we classify only one image, we change batch size from 10 to 1
    # #original 'data' size is (10,3,227,227): 10 images, 3 channels per image, 227 x 227 image size
    net.blobs['data'].reshape(1,3,227,227)
    elements = boxes.shape[0]
    Features = []
    j = 0
    while (j < elements):
        t = time.time()
        box = boxes[j]
        crop = crop_image(image, box)
        prepro = transformer.preprocess('data', crop)
        net.blobs['data'].data[...] = prepro
        #out = net.forward(data = prepro)
        out = net.forward()
        Features.append(net.blobs['fc7'].data[0])
        t = time.time() - t
        print("time: " + str(t) +" for " + str(j) + " in " +X[1] )
        j = j + 1

    Features = np.array(Features)

    with open('./features/%s' % str(page), 'w') as e:
        np.save(e,Features)
    tpre = time.time() - tpre
    print("image:" + X[1] +" elements: " + str(Features.shape[0]) + " time: "+ str(tpre))
    return Features

with open('full_clean_boxes') as cb:
    boxes = np.load(cb)

d = './DocExplore_images/%s'
i = 0
total_elements = 0

t = time.time()
with open('docexplore_pages.txt', 'r') as f:
    lines = f.read().splitlines()
    fc7_features = []

    ft = time.time()
    pool = Pool(processes=1)
    pool_result = pool.map(make_fc7_features, itertools.izip(boxes,lines,np.arange(1,1501)))
    pool.close()
    pool.join()
    ft = time.time() - ft
    print("done processing fc7 features, time: "+str(ft))

    for line, result in enumerate(pool_result):
        fc7_features.append(result)

t = time.time() -t
fc7 = np.array(fc7_features)
print("building took: " + str(t))
with open('fc7_full_features', 'w') as wfc:
    np.save(fc7)
