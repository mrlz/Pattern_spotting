from sklearn import svm
from os import listdir
from sklearn.externals import joblib
import numpy as np
import time
import cv2
from skimage.feature import local_binary_pattern
from sklearn import preprocessing
from multiprocessing import Pool
import itertools
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
import csv

class Evaluation(object):
    def __init__(self, f,X_train,y_train,X_test,y_test):
        self.ff = f
        self.X_train=X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def __call__(self, x):
        return self.ff(x, self.X_train,self.y_train,self.X_test,self.y_test)

def make_lbp(image_path, points, radius):
    expected = points + 2
    image = cv2.imread(image_path)
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
    return np.bincount(np.array(lbp.ravel() , dtype = np.int64) , minlength=expected)

def color_hist(image_path,size):
    img = cv2.imread(image_path)
    histr = cv2.calcHist(img,[0,1,2],None,[size,size,size],[0,256,0,256,0,256])
    return histr.flatten()

def get_specifics(y_pred, y_true):
    ans = []
    ans2 = []
    i = 0
    z = 0
    o = 0
    while(i < y_true.shape[0]):
        if (y_true[i] == 0):
            ans.append(y_pred[i])
            z = z + 1
        else:
            ans2.append(y_pred[i])
            o = o + 1
        i = i + 1
    return accuracy_score(np.zeros(z),ans, normalize=True), accuracy_score(np.ones(o), ans2, normalize=True)

def make_train_and_test_1_svm(c,g,X_train,y_train,X_test,y_test):
    SVC = svm.SVC(kernel='rbf', C = c, gamma = g)

    t = time.time()
    SVC.fit(X_train,y_train)
    t = time.time() - t

    t = time.time()
    train_prediction = SVC.predict(X_train)
    t = time.time() - t
    # print("train predict time "+ str(t))

    t = time.time()
    test_prediction = SVC.predict(X_test)
    t = time.time() - t
    # print("test predict time "+ str(t))

    obj_train, bg_train = get_specifics(train_prediction,y_train)
    obj_test, bg_test = get_specifics(test_prediction,y_test)
    acc_train = accuracy_score(y_train,train_prediction,normalize=True)
    acc_test = accuracy_score(y_test,test_prediction,normalize=True)
    mat_train = matthews_corrcoef(y_train, train_prediction)
    mat_test = matthews_corrcoef(y_test, test_prediction)
    Ans = []
    Ans.append(g)
    Ans.append(c)
    Ans.append(acc_train)
    Ans.append(acc_test)
    Ans.append(mat_train)
    Ans.append(mat_test)
    Ans.append(obj_train)
    Ans.append(bg_train)
    Ans.append(obj_test)
    Ans.append(bg_test)
    return np.array(Ans)

def make_train_and_test_svms(I,X_train,y_train,X_test,y_test):
    Gammas = I[1]
    c = I[0]
    Ans = []
    for g in Gammas:
        Ans.append(make_train_and_test_1_svm(c,g,X_train,y_train,X_test,y_test))
    return np.array(Ans)

bg_path = './background_svm_data/background/crops/%s'
backgrounds = np.array(listdir('./background_svm_data/background/crops'))
fg_path = './background_svm_data/foreground/crops/%s'
foregrounds = np.array(listdir('./background_svm_data/foreground/crops'))

bg_test_path = './background_svm_data/backgrounds_test/%s'
test_backgrounds = np.array(listdir('./background_svm_data/backgrounds_test'))
fg_test_path = './background_svm_data/foregrounds_test/%s'
test_foregrounds = np.array(listdir('./background_svm_data/foregrounds_test'))

n_backgrounds = backgrounds.shape[0]
n_foregrounds = foregrounds.shape[0]

n_test_backgrounds = test_backgrounds.shape[0]
n_test_foregrounds = test_foregrounds.shape[0]

color_size = 8
color_vector_size = color_size**3

i = 0
radius = 2
no_points = 8
vector_size = no_points+2

X_train = np.zeros((n_backgrounds+n_foregrounds,color_vector_size+vector_size))
y_train = np.zeros(n_backgrounds + n_foregrounds)

X_test = np.zeros((n_test_backgrounds+n_test_foregrounds,color_vector_size+vector_size))
y_test = np.zeros(n_test_backgrounds+n_test_foregrounds)


t = time.time()
while(i < n_backgrounds):
    X_train[i,0:vector_size] = make_lbp(bg_path % backgrounds[i], no_points, radius)
    X_train[i,vector_size:color_vector_size+vector_size] = color_hist(bg_path % backgrounds[i],color_size)
    y_train[i] = 1
    i = i + 1

track = i
i = 0
while(i < n_foregrounds):
    X_train[track + i,0:vector_size] = make_lbp(fg_path % foregrounds[i], no_points, radius)
    X_train[track + i,vector_size:color_vector_size+vector_size] = color_hist(fg_path % foregrounds[i],color_size)
    i = i + 1
t = time.time() - t
print("time: "+ str(t) + "\ncost by training element: " + str(t/(n_backgrounds+n_foregrounds)))

X_train = preprocessing.normalize(X_train, norm = 'l2')

i = 0
while(i < n_test_backgrounds):
    X_test[i,0:vector_size] = make_lbp(bg_test_path % test_backgrounds[i], no_points, radius)
    X_test[i,vector_size:color_vector_size+vector_size] = color_hist(bg_test_path % test_backgrounds[i],color_size)
    y_test[i] = 1
    i = i + 1

track = i
i = 0
while(i < n_test_foregrounds):
    X_test[track + i,0:vector_size] = make_lbp(fg_test_path % test_foregrounds[i], no_points, radius)
    X_test[track + i,vector_size:color_vector_size+vector_size] = color_hist(fg_test_path % test_foregrounds[i],color_size)
    i = i + 1

X_test = preprocessing.normalize(X_test, norm = 'l2')
print("time: "+ str(t) + "\ncost by testing element: " + str(t/(n_test_backgrounds+n_test_foregrounds)))

############# Perform cross-validation in parallel################
# # c = 1000
# # s1 = 0.4
# #gamma = 3.125
# # C = np.arange(1.0,12.1,0.1)
# C = np.array([1,4.5,4.8,4.9,5,5.1,5.2,5.3,8,9.5,9.8,9.9,10,10.1,10.2,10.3,100,1000])
# # C = [1,5,10,15,20,25,30,35,40,45,50]
# # Gammas = [2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6,4.8,5.0,5.2,5.4,5.6,5.8,6.0,6.2,6.4,6.6,6.8,7.0,7.2,7.4,7.6,7.8,8.0]
# Gammas = np.arange(4.8,6.1,0.1)
# # print("saving to disk")
# # with open("X_train", 'w') as f:
# #     np.save(f,X_train)
# t = time.time()
# pool = Pool(processes=12)
# pool_result = pool.map(Evaluation(make_train_and_test_svms,X_train,y_train,X_test,y_test), itertools.izip(C,itertools.repeat(Gammas)))
# pool.close()
# pool.join()
# with open("./svm_svc.csv", 'w') as csvfile:
#     fieldnames = ['G', 'C', 'Train_acc', 'Test_acc', 'M_train', 'M_test', 'Train_acc_obj', 'Train_acc_bg', 'Test_acc_obj', 'Test_acc_bg']
#     writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
#     writer.writeheader()
#     for line, result in enumerate(pool_result):
#         for g in result:
#             writer.writerow({'G':str(g[0]), 'C':str(g[1]), 'Train_acc':str(g[2]), 'Test_acc':str(g[3]), 'M_train':str(g[4]), 'M_test':str(g[5]), 'Train_acc_obj':str(g[6]), 'Train_acc_bg':str(g[7]), 'Test_acc_obj':str(g[8]), 'Test_acc_bg': str(g[9])})
# t = time.time() - t
# print("took: "+ str(t))

####################################################################



#############Sequential cross-validation - for reference#############
# with open("./experiment_svm/svm_svc.csv", 'w') as csvfile:
#     fieldnames = ['G', 'C', 'Train_acc', 'Test_acc', 'M_train', 'M_test', 'Train_acc_obj', 'Train_acc_bg', 'Test_acc_obj', 'Test_acc_bg']
#     writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
#     writer.writeheader()
#     for g in Gammas:
#         for c in C:
#             print("Gamma:"+str(g)+" C:" +str(c))
#             SVC = svm.SVC(kernel='rbf', C = c, gamma = g)
#
#             t = time.time()
#             SVC.fit(X_train,y_train)
#             t = time.time() - t
#             print("time "+ str(t))
#
#             # joblib.dump(SVC, 'background_svm_dump.pkl')
#
#             t = time.time()
#             train_prediction = SVC.predict(X_train)
#             t = time.time() - t
#             print("train predict time "+ str(t))
#
#             t = time.time()
#             test_prediction = SVC.predict(X_test)
#             t = time.time() - t
#             print("test predict time "+ str(t))
#
#             obj_train, bg_train = get_specifics(train_prediction,y_train)
#             obj_test, bg_test = get_specifics(test_prediction,y_test)
#             writer.writerow({'G':str(g), 'C':str(c), 'Train_acc':str(accuracy_score(y_train,train_prediction,normalize=True)), 'Test_acc':str(accuracy_score(y_test,test_prediction,normalize=True)), 'M_train':str(matthews_corrcoef(y_train, train_prediction)), 'M_test':str(matthews_corrcoef(y_test, test_prediction)), 'Train_acc_obj':str(obj_train), 'Train_acc_bg':str(bg_train), 'Test_acc_obj':str(obj_test), 'Test_acc_bg':str(bg_test)})

###############################################################################



################### Save model with desired hyper-parameters###################
c = 5.0
g = 5.0
SVC = svm.SVC(kernel='rbf', C = c, gamma = g)

t = time.time()
SVC.fit(X_train,y_train)
t = time.time() - t
print("time "+ str(t))

joblib.dump(SVC, 'background_svm_dump.pkl')

t = time.time()
train_prediction = SVC.predict(X_train)
t = time.time() - t
print("train predict time "+ str(t))

t = time.time()
test_prediction = SVC.predict(X_test)
t = time.time() - t
print("test predict time "+ str(t))

obj_train, bg_train = get_specifics(train_prediction,y_train)
obj_test, bg_test = get_specifics(test_prediction,y_test)
print("pred bgs train: " + str(np.sum(train_prediction)))
print("true bgs train: "+str(np.sum(y_train)))
print("pred bgs test: " + str(np.sum(test_prediction)))
print("true bgs test: "+str(np.sum(y_test)))
print("acc train: " + str(accuracy_score(y_train,train_prediction,normalize=True)))
print("acc test: " + str(accuracy_score(y_test,test_prediction,normalize=True)))
print("matthews train: " + str(matthews_corrcoef(y_train, train_prediction)))
print("matthews test: " + str(matthews_corrcoef(y_test, test_prediction)))
print("obj acc train: " + str(obj_train))
print("bg acc train: " + str(bg_train))
print("obj acc test: " + str(obj_test))
print("bg acc test: " + str(bg_test))

i = 0
while(i < 1000):
    if (y_test[i] == 1 and test_prediction[i] == 1):
        print("jackpot")
        with open('bg_cheap_feature', 'w') as fff:
            np.save(fff,X_test[i])
        break
    i = i + 1
##############################################################################
