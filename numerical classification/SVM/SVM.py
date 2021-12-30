# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 22:11:38 2021

@author: mahmoud
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('cars.csv')
data.info()

#missing values
print("Total missing values:", data.isna().sum().sum())
print("Columns with missing values:", data.columns[data.isna().sum() > 0].values)

#fill null values
data['engine_capacity'] = data['engine_capacity'].fillna(data['engine_capacity'].mean())
print("Total missing values:", data.isna().sum().sum())

#encoding
{column: len(data[column].unique()) for column in data.columns if data.dtypes[column] == 'object'}
data=data.drop(['location_region','model_name','feature_0','feature_1','feature_2','feature_3','feature_4','feature_5','feature_6','feature_7','feature_8','feature_9'],axis=1) # not important for our model

for column in data.columns:
    if data.dtypes[column] == 'bool':
        data[column] = data[column].astype(int)

data['transmission'].unique()
transmission_mapping = {'automatic': 0, 'mechanical': 1}
data['transmission'] = data['transmission'].replace(transmission_mapping)

encode_x = LabelEncoder()
data['manufacturer_name'] = encode_x.fit_transform(data['manufacturer_name'])
data['color'] = encode_x.fit_transform(data['color'])
data['engine_fuel'] = encode_x.fit_transform(data['engine_fuel'])
data['body_type'] = encode_x.fit_transform(data['body_type'])
data['state'] = encode_x.fit_transform(data['state'])
data['drivetrain'] = encode_x.fit_transform(data['drivetrain'])

data['engine_type'].unique()
label_mapping = {
    'gasoline': 0,
    'diesel': 1,
    'electric': 2
}

data['engine_type'] = data['engine_type'].replace(label_mapping)

print("Remaining non-numeric columns:", (data.dtypes == 'object').sum())

data.info()

#Splitting/Scaling
y = data['engine_type'].copy()
X = data.drop('engine_type', axis=1).copy()
scaler = StandardScaler()

X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)


####Default hyperparameter means C=1.0, kernel=rbf and gamma=auto among other parameters.####
# import SVC classifier
from sklearn.svm import SVC


# import metrics to compute accuracy
from sklearn.metrics import accuracy_score


# instantiate classifier with default hyperparameters
svc=SVC() 



# instantiate classifier with rbf kernel and C=100
svc=SVC(C=100.0,probability=True) 


# fit classifier to training set
svc.fit(X_train,y_train)


# make predictions on test set
y_pred=svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with rbf kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# import pickle
# # save the model to disk
# filename = 'svm_model.sav'
# pickle.dump(svc, open(filename, 'wb'))






#some plots
"""
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
"""
"""
#ROC
pred = svc.predict(X_test)
pred_prop=svc.predict_proba(X_test)
fpr = {}
tpr = {}
thresh ={}

n_class = 3

for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prop[:,i], pos_label=i)
    
# plotting    
plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Class gasoline vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Class diesel vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='Class electric vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC',dpi=300);
"""

'''
#Accuracy
from sklearn.model_selection import learning_curve
import numpy as np

estimator = SVC(C=100.0)
train_sizes, train_scores, valid_scores = learning_curve(estimator=estimator,X=X_test,y=y_test,scoring='accuracy')
plt.xlabel("Testing data")
plt.ylabel("Score")
plt.title("Accuracy of the model")
plt.plot(train_sizes,np.mean(valid_scores,axis=1))


'''
"""
#confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
pred = svc.predict(X_test)
cm = confusion_matrix(y_test, pred)

cm_df = pd.DataFrame(cm,
                     index = ["0","1","2"], 
                     columns =["0","1","2"])

plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()
"""




"""
from sklearn.model_selection import learning_curve
import numpy as np

estimator = SVC(kernel='rbf',decision_function_shape='ovo')
train_sizes ,train_scores ,test_scores =learning_curve(estimator, X_test, y_test, cv=3, scoring='accuracy' ,n_jobs=-1,verbose=1)

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

