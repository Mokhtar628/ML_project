# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 18:32:54 2021

@author: mohamed
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display

import pandas as pd
import numpy as np

# from PIL import Image

from skimage.feature import hog
from skimage.color import rgb2grey

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc

import cv2
from tqdm import tqdm

DATADIR = "our_data"



CATEGORIES = ["Ahmed Amr","Ali Habib","Mohamed Bebo","Mohamed Labib","Mohamed Mokhtar"]

training_data = []


def create_data():
    for category in CATEGORIES:  # do

        path = os.path.join(DATADIR,category)  # create path 
        class_num = CATEGORIES.index(category)  # get the classification

        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) #it may be not necessory but i put it as a precaution
                img_array=cv2.resize(img_array, (35, 35))
                img_array = hog(img_array, block_norm='L1')
                img_array = np.array(img_array)
                training_data.append([img_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
       
        

create_data()


y = []
features_train = []



for feature,label in training_data:
    y.append(label)
    features_train.append(feature)  



features_train = np.array(features_train)


print("Features loaded..\n")



X = pd.DataFrame(features_train)


Xtrain, Xtest, ytrain, ytest = train_test_split(X,
                                                    y,
                                                    test_size=.25,
                                                    random_state=1234123)




print("Start training..\n")



RF = SVC(kernel='rbf',decision_function_shape='ovo')
RF.fit(Xtrain, ytrain)

# import pickle
# # save the model to disk
# filename = 'svm_model2.sav'
# pickle.dump(linear, open(filename, 'wb'))


# generate predictions

rf_pred = RF.predict(Xtest)


print("End training..\n")
# calculate accuracy
rf_accuracy = accuracy_score(ytest, rf_pred)


print('rf model accuracy is: ', rf_accuracy)#99 93L 95P
'''

'''
'''
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
pred = RF.predict(Xtest)
pred_prop=RF.predict_proba(Xtest)
fpr = {}
tpr = {}
thresh ={}

n_class = 5

for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(ytest, pred_prop[:,i], pos_label=i)

# plotting
plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Class Ahmed Amr vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Class Ali Habib vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--',color='red', label='Class Mohamed Bebo vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--',color='blue', label='Class Mohamed Labib vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--',color='purple', label='Class Mohamed Mokhtar vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC',dpi=300);
'''
'''
from sklearn.model_selection import learning_curve
estimator = SVC(kernel='rbf',decision_function_shape='ovo')
train_sizes, train_scores, valid_scores = learning_curve(estimator=estimator,X=Xtest,y=ytest,scoring='accuracy')
plt.xlabel("Testing data")
plt.ylabel("Score")
plt.title("Accuracy of the model")
plt.plot(train_sizes,np.mean(valid_scores,axis=1))
'''
'''
from sklearn.metrics import det_curve
y_true = np.array(ytest)
y_scores = np.array(rf_pred)

fpr, fnr, thresholds = det_curve(y_true, y_scores,pos_label=1)
plt.plot(fpr,fnr)
plt.show()
'''
"""
from sklearn.model_selection import learning_curve
estimator = SVC(kernel='rbf',decision_function_shape='ovo')
train_sizes ,train_scores ,test_scores =learning_curve(estimator, Xtest, ytest, cv=5, scoring='accuracy' ,n_jobs=-1,verbose=1)

train_mean =np.mean(train_scores , axis=1)
train_mean

train_std =np.mean(train_scores , axis=1)   
train_std

test_mean =np.mean(test_scores , axis=1)
test_mean

test_std =np.mean(test_scores , axis=1)
test_std

plt.plot(train_sizes , train_mean ,label='training score')
plt.plot(train_sizes , test_mean ,label='test score')

plt.fill_between(train_sizes-train_std , train_mean+train_std ,color='#ffffff' )
plt.fill_between(train_sizes-test_std , test_mean+test_std ,color='#ffffff' )

plt.title('learning curve')
plt.xlabel('training size')
plt.ylabel('accuracy score')
plt.legend(loc = 'best')

"""