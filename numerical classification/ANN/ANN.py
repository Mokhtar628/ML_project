import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense ,Dropout
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

model = Sequential()
model.add(Dense(64,activation='relu'))#input layer
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))#output

# compile the keras model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
history = model.fit(X_train,  y_train, epochs=50,  batch_size=32,verbose=1, validation_split=0.2)

# evaluate the keras model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("accuracy ",test_acc)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#confusion matrix
from sklearn.metrics import confusion_matrix
prediction = model.predict(X_test).argmax(axis=1)
cm = confusion_matrix(y_test, prediction)

cm_df = pd.DataFrame(cm,
                     index = ["0","1","2"], 
                     columns =["0","1","2"])

plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True,fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

# roc curve for classes
from sklearn.metrics import roc_curve
pred = model.predict(X_test)
pred_prob = model.predict_proba(X_test)
fpr = {}
tpr = {}
thresh ={}

n_class = 3

for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:,i], pos_label=i)
    
# plotting    
plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Class gasoline vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Class diesel vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='Class electric vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC',dpi=300);

