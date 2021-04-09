

# !pip install lime
#
# !pip install tensorflow==2.3.0
#
# !pip uninstall neupy

"""####Libraries"""

import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dense
from tensorflow.keras import Sequential
from keras import regularizers
from keras.metrics import binary_accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score,precision_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve, auc, roc_curve, f1_score
from sklearn.metrics import matthews_corrcoef
from numpy import sqrt
from numpy import argmax 
import matplotlib.pyplot as plt

from tensorflow.python.keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.regularizers import l2

from keras.layers.core import Dense
from keras.optimizers import RMSprop
from rbflayer import RBFLayer, InitCentersRandom

# from neupy import algorithms
# import lime
# import lime.lime_tabular

data = pd.read_csv('train_data.txt', sep=",", header=None)


data.columns = ["Subject id", "Jitter (local)" ,"Jitter (local, absolute)" ,"Jitter (rap)" ,"Jitter (ppq5)", "Jitter (ddp)", "Shimmer (local)", "Shimmer (local, dB)" ,"Shimmer (apq3)", "Shimmer (apq5)", "Shimmer (apq11)", "Shimmer (dda)", "AC" ,"NTH" ,"HTN", "Median pitch", "Mean pitch", "Standard deviation", "Minimum pitch", "Maximum pitch", "Number of pulses", "Number of periods", "Mean period", "Standard deviation of period", "Fraction of locally unvoiced frames", "Number of voice breaks", "Degree of voice breaks", "UPDRS", "class information"]
data.shape

d = {}
for x in data["Subject id"]:
    d[x] = d.get(x, 0)+1
d

data.head()
data.isnull().sum()
data.info()

uniqueness = data.select_dtypes(include=['O'])
uniqueness.apply(pd.Series.nunique)

y_updrs = data.UPDRS # for regression
y_class_info = data["class information"]#for classification
data_x = data.drop(labels=['UPDRS', "Subject id", "class information"], axis=1, inplace = False)
print(data_x.shape, y_updrs.shape, y_class_info.shape)

def nature_updrs_class():
    classinfo_UPDRS = data[['UPDRS', 'class information']].copy()
    classinfo_UPDRS.iloc[1,1]
    d = {}
    for i in range(classinfo_UPDRS.shape[0]):
        class_ = classinfo_UPDRS.iloc[i,1]
        updrs  = classinfo_UPDRS.iloc[i,0]
        if class_ in d:
            if updrs not in d[class_]:
                d[class_].append(updrs)
        else:
            d[class_] = [updrs]
    for it in d:
        d[it].sort()
    print("no. of 0s = ",classinfo_UPDRS['class information'].to_list().count(0), "No of 1s = ",classinfo_UPDRS['class information'].to_list().count(1))
    return d

print(nature_updrs_class())
"""implementation of the same on NN"""

type(data_x)

"""###NN Model and using CV for evaluating models"""

batch_size = 20
no_epochs = 2000
optimizer_func = "adam"
loss_func = "binary_crossentropy"
valid_split = 0.20

fold_no = 1
total_folds = 10

def create_model():
    model = Sequential()
    model.add(Dense(52, kernel_initializer='normal', activation='relu', input_dim = 26,))
    model.add(Dense(26, kernel_initializer='normal', activation='relu'))
    model.add(Dense(13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    
    model.compile(optimizer = optimizer_func, loss = loss_func, metrics = ['accuracy'])
    
    return model

def create_rbf_model(X):

    model = Sequential()
    rbflayer = RBFLayer(13,
                        initializer=InitCentersRandom(X),
                        betas=2.0,
                        #input_shape=(1,)
                        )
    model.add(Dense(26, kernel_initializer='normal', activation='relu', input_dim = 26,))
    model.add(rbflayer)
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    model.compile(loss='mean_squared_error',
                  optimizer=RMSprop())

    return model

def create_pnn_model():
 
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3)

    pnn = algorithms.PNN(std=10, verbose=False)
    pnn.train(x_train, y_train)

    y_predicted = pnn.predict(x_test)
    metrics.accuracy_score(y_test, y_predicted)



#datacopying and normalization
data_x_nn = data_x.copy(deep=True)
target_y_nn = y_class_info.copy(deep=True)
nm = MinMaxScaler()
data_x_nn = nm.fit_transform(data_x_nn)


#Kfold Validator definition
kfold = KFold(n_splits= total_folds, shuffle=True, random_state=42)
acc_per_fold = []
loss_per_fold = []


for train_index, test_index in kfold.split(data_x_nn, target_y_nn):

    model = create_model()

    print('\nTraining for fold no ...  :' ,fold_no, "/",total_folds,"\n")

    history = model.fit(data_x_nn[train_index],
                        target_y_nn[train_index],
                        batch_size = batch_size, 
                        epochs = no_epochs,
                        verbose=2,
                        validation_split = valid_split)
    
    scores = model.evaluate(data_x_nn[test_index], target_y_nn[test_index], verbose=0)
    #print("Score for fold",fold_no,", ", model.metrics_names[0], " of " ,scores[0],"  ",model.metrics_names[1]," of, ",scores[1]*100,"%")
    
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    fold_no+=1
    
# neural_network = KerasClassifier(build_fn=create_model,epochs=1000, batch_size=10, verbose=2)
# cross_val_score(neural_network,X_train_nn, y_train_nn, cv=3)

# y_pred = model.predict(X_test_nn)
# y_predm = [1 if x>0.5 else 0 for x in y_pred]
# acc = accuracy_score(y_test_nn, y_predm)
# print('Test Accuracy = %.2f' % acc)

print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print("> Fold ",i+1," - Loss: ",loss_per_fold[i],"  - Accuracy: ", acc_per_fold[i],"%")
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print('> Accuracy: ',np.mean(acc_per_fold),' (+- ',np.std(acc_per_fold),')')
print('> Loss: ', np.mean(loss_per_fold))
print('------------------------------------------------------------------------')

# bacc = binary_accuracy(y_test_nn, y_pred, threshold=0.5)
# print('Test Accuracy = %.2f' % bacc*100)

path = "./Model_V2_Final_CV"
model.save(path+".h5")
model.save_weights(path+"_weights")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print("Saved")

"""#Developing the final model after CV evaluation """

batch_size = 10
nm = MinMaxScaler()
data_x_nn = data_x.copy(deep=True)

target_y_nn = y_class_info.copy(deep=True)
no_epochs = 2000
data_x_nn = nm.fit_transform(data_x_nn)

X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(data_x_nn, target_y_nn, test_size = 0.20, random_state = 42)
model_final = create_rbf_model(data_x_nn)

#one-hot encode target column
#y_train_nn = to_categorical(y_train_)

early_stopping_monitor = EarlyStopping(patience=10, restore_best_weights=True)

history_final = model_final.fit(X_train_nn,
                    y_train_nn,
                    #batch_size = batch_size, 
                    epochs = no_epochs,
                    verbose=2,
                    validation_split = valid_split,
                    callbacks=[early_stopping_monitor]
                    )

scores = model_final.evaluate(X_test_nn, y_test_nn, verbose=0)
predictions = model_final.predict(X_test_nn)
predictions_rounded = np.round_(predictions)

"""#####first column is the inverse the second column is the original value, in to_categorical

"""

print(predictions[:10], predictions_rounded[:10],accuracy_score(y_test_nn, predictions_rounded, normalize=True))

from sklearn.metrics import matthews_corrcoef
tn, fp, fn, tp = confusion_matrix(y_test_nn, predictions_rounded).ravel()

print("Average Accuracy : " ,(tp+tn)/(tp+tn+fp+fn))
print("Sensitivity: ",(tp)/(tp+fn))
print("Specificity: ",(tn)/(tn+fp))
print("MCC: ",matthews_corrcoef(y_test_nn, predictions_rounded))
confusion_matrix(y_test_nn, predictions_rounded, labels=None, sample_weight=None, normalize=None)

recall_score(y_test_nn, predictions_rounded, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')

precision_score(y_test_nn, predictions_rounded, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')

path = "./Model_V6_Final" #5 is with callback
model_final.save(path+".h5")
model_final.save_weights(path+"_weights")
model_json_final = model_final.to_json()
with open("model_final.json", "w") as json_file:
    json_file.write(model_json_final)
print("Saved")

model_final.summary()

model_final.weights[1]

"""##Visualization of training and validation losses and accuracies

"""

plt.figure(figsize=(15, 10))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.figure(figsize=(15, 10))
plt.plot(history_final.history['accuracy'])
plt.plot(history_final.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

"""##Details of the previous run"""

# batch_size = 20
# no_epochs = 1000
# optimizer_func = "sgd"
# loss_func = "binary_crossentropy"
# valid_split = 0.20

# hyperparameters for Kfold CV
# fold_no = 1
# total_folds = 10
#  model.add(Dense(52, kernel_initializer='normal', activation='relu', input_dim = 26))
#     model.add(Dense(26, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(26, kernel_initializer='normal', activation='relu'))
#     # Ouput layer
#     model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    
#     model.compile(optimizer = optimizer_func, loss = loss_func, metrics = ['accuracy'])

# Score per fold
# ------------------------------------------------------------------------
# > Fold  1  - Loss:  0.779529869556427   - Accuracy:  49.03846085071564 %
# ------------------------------------------------------------------------
# > Fold  2  - Loss:  0.6190924644470215   - Accuracy:  70.19230723381042 %
# ------------------------------------------------------------------------
# > Fold  3  - Loss:  0.7622536420822144   - Accuracy:  63.461536169052124 %
# ------------------------------------------------------------------------
# > Fold  4  - Loss:  0.6309367418289185   - Accuracy:  69.2307710647583 %
# ------------------------------------------------------------------------
# > Fold  5  - Loss:  0.7086811065673828   - Accuracy:  59.61538553237915 %
# ------------------------------------------------------------------------
# > Fold  6  - Loss:  0.8217067718505859   - Accuracy:  68.2692289352417 %
# ------------------------------------------------------------------------
# > Fold  7  - Loss:  0.63923579454422   - Accuracy:  62.5 %
# ------------------------------------------------------------------------
# > Fold  8  - Loss:  0.7599735260009766   - Accuracy:  56.7307710647583 %
# ------------------------------------------------------------------------
# > Fold  9  - Loss:  0.825320839881897   - Accuracy:  44.23076808452606 %
# ------------------------------------------------------------------------
# > Fold  10  - Loss:  0.7413977980613708   - Accuracy:  60.576921701431274 %
# ------------------------------------------------------------------------
# Average scores for all folds:
# > Accuracy:  60.3846150636673  (+-  8.088361972946457 )
# > Loss:  0.7288128554821014

# model_save_name = 'Parkinsons_V1' 
# #V1 = 2000epoch, adam , tanh, 2 hid layer, 28, 14, loss = 0.4458
# #V2 = 3000epoch, adam , tanh, 2 hid layer, 28, 14, loss = 0.4458
# #V3 = 2000epoch, adam , tanh, 2 hid layer, 26, 13, loss = 2.2358, removed subject id and class information
# #V3 = 1000epoch, adam , relu, 2 hid layer, 26, 13, loss = , removed subject id and class information
# #V4 = 1000epoch, adam , -,-,sigmoid, 2 hid layer, 26, 13, loss = 0.6220, acc = 0.66,
# #V5 = 1000epoch, adam , sig,sig,sigm, 2 hid layer, 26, 13, loss = 0.5220, acc = 0.68,
# #V5 = 1000epoch, adam , sig,sig,sigm, 2 hid layer, 26, 7, loss = 0.5482, acc = 0.72,
# #V6 = 1000epoch, adam , tanh,tanh,sigm, 2 hid layer, 26, 7, loss = 0.4024, val_acc = 0.8005, acc = 0.69
# #V7 = 1000epoch, adam,  tanh, tanh, 1 hid layer, 13, loss = 0.4810, val_acc = 0.7632, acc = 0.72
# #V8 = 400 epoch, adam,  tanh, tanh, 1 hid layer, 13, loss = 0.5334, val_acc = 0.6971, acc = 0.69 + regularization
# #V9 = 400 epoch, adam,  tanh, tanh, 1 hid layer, 13, loss = 0.3218, val_acc = 0.8486, acc = 0.63    ---overfit
# #V10 = 400 epoch, adam,  tanh, sigmoid, 1 hid layer, 26, loss = 0.1768, val_acc = 0.9507, acc = 0.68  ---overfit
# kernel_regularizer=regularizers.l2(0.01)



# path = "./Models"+model_save_name 
# model.save(path+'.h5')
# model.metrics_names
# # score = model.evaluate(X_test_nn, y_test_nn, verbose=0)
# # print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

import matplotlib.pyplot as plt
def scatter_graph(y_test_nn, y_pred):
    xaxis = [_ for _ in range(208)]
    plt.scatter(xaxis, y_test_nn,  color='gray',)
    plt.plot(xaxis, y_pred, color='red', )
    plt.show()

def bar_graph(y_test_nn, y_pred):
    y_real = y_test_nn.to_numpy().ravel()
    df = pd.DataFrame({'Actual': y_real, 'Predicted': y_pred.ravel()})
    df1 = df.head(50)
    df1.plot(kind='bar',figsize=(10,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()

"""##Loading the model
####Load path = path = "./Model_V2_Final"
####else
####path = "./Model_V2_Final_CV" - the one with cross val
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd drive/My\ Drive/ML/Parkinsons

from tensorflow.keras.models import model_from_json
path = "./Model_V6_Final" # latest = V6
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(path+".h5")
print("Loaded model from disk")

loaded_model.weights[1]

all(loaded_model.predict(X_test_nn) == predictions)

"""##ROC and PR Analysis"""

nm = MinMaxScaler()
data_x_nn = data_x.copy(deep=True)
target_y_nn = y_class_info.copy(deep=True)
data_x_nn = nm.fit_transform(data_x_nn)
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(data_x_nn, target_y_nn, test_size = 0.20, random_state = 42)

one = 0
zero = 0
total =0
for ele in y_train_nn:
    if ele == 1:
        one+=1
    elif ele == 0:
        zero+=1
    total+=1
print(one, zero, total, one+zero)

one = 0
zero = 0
total =0
for ele in y_test_nn:
    if ele == 1:
        one+=1
    elif ele == 0:
        zero+=1
    total+=1
print(one, zero, total, one+zero)

predictions = loaded_model.predict(X_test_nn)
predictions_rounded = np.round_(predictions)

import seaborn as sn

array = confusion_matrix(y_test_nn, predictions_rounded)
tn_thr, fp_thr, fn_thr, tp_thr = confusion_matrix(y_test_nn, predictions_rounded).ravel()

df_cm = pd.DataFrame(array, index = [i for i in ["non - PD", "PD"]],
                  columns = [i for i in ["non-PD","PD"]])
accuracy_thr = accuracy_score(y_test_nn, predictions_rounded)
print("New accuracy ",accuracy_thr)
print("Average Accuracy : " ,(tp_thr+tn_thr)/(tp_thr+tn_thr+fp_thr+fn_thr))
print("Sensitivity: ",(tp_thr)/(tp_thr+fn_thr))
print("Specificity: ",(tn_thr)/(tn_thr+fp_thr))
print("MCC: ",matthews_corrcoef(y_test_nn, predictions_rounded))
print("F-measure: ", f1_score(y_test_nn, predictions_rounded))

plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, cmap="YlGnBu",annot=True,annot_kws={"fontsize":48})
plt.savefig("NN_Conf_orig_matrix.jpg", bbox_inches = "tight")

print(len(y_test_nn), len(predictions))

fpr, tpr, thresholds = roc_curve(y_test_nn, predictions)
roc_auc = auc(fpr, tpr)
gmeans = sqrt(tpr * (1-fpr))
arg_g = argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[arg_g], gmeans[arg_g]))
best_threshold = thresholds[arg_g]
tn, fp, fn, tp = confusion_matrix(y_test_nn, predictions_rounded).ravel()
acc = accuracy_score(y_test_nn, predictions_rounded)
# tpr = tp/(tp+fn)
# fpr = fp/(fp+tn)
# tnr = tn/(fp+tn)
# fnr = fn/(tp+fn)

print("accuracy = ", acc, "\nroc_auc = ", roc_auc)

i=0
roc_table = pd.DataFrame(columns=["Threshold", "TPR (Sensitivity)",
                                  "FPR (Fall-out)", "Specificity",
                                  " (LR+)" ,"Youden index","Sensitivity + Specificity",
                                  "G-mean"], index=[_ for _ in range(len(thresholds))])

for fp_rate, tp_rate, thresh in zip(fpr, tpr, thresholds):
    spec = 1-fp_rate
    lrplus = tp_rate/fp_rate
    y_ind = tp_rate - fp_rate
    sen_sp = tp_rate + spec
    g_mean =  (tp_rate*spec)**(1/2)
    roc_table.iloc[i] = [thresh, tp_rate, fp_rate, spec, lrplus, y_ind, sen_sp, g_mean ]
    i+=1

# Commented out IPython magic to ensure Python compatibility.
# %cd ../../../..

pd.set_option("display.precision", 4)
roc_table
roc_table.to_csv("NN_ROC_Table.csv")

plt.subplots(1, figsize=(7,7))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, marker = "." ,label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'--', label = "No Skill")

plt.scatter(fpr[arg_g], tpr[arg_g], marker='o', color='black', label='Best')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(["ROC", "No Skill"], loc ="lower right")

plt.savefig("NN_ROC.jpg", bbox_inches="tight")
plt.show()

precision, recall, thresholds = precision_recall_curve(y_test_nn, predictions)

fscore_ = (2 * precision * recall) / (precision + recall)
# locate the index of the largest f score
ix = argmax(fscore_)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore_[ix]))

pr_table = pd.DataFrame(columns=["Threshold", "Precision",
                                  "Recall", "F-Measure"], 
                                index=[_ for _ in range(len(thresholds))])

i=0
for pre, rec, thresh in zip(precision, recall, thresholds):
    fscore = (2 * pre * rec) / (pre + rec)
    pr_table.iloc[i] = [thresh, pre, rec, fscore]
    i+=1

pr_table.to_csv("NN_PR_Table.csv")
pr_table

plt.subplots(1, figsize=(7,7))
plt.title('Precision-Recall Curve')
plt.plot(recall, precision)
plt.plot([0, 1], [0.725, 0.725], linestyle='--')
plt.legend(loc = 'lower right')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
# axis labels
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.legend(["PR","No Skill"], loc ="upper right")
plt.savefig("NN_PR.jpg", bbox_inches="tight")
plt.show()

print("F-measure : ", f1_score(y_test_nn, predictions_rounded))

y_pred_thr = [ 1 if _ >= best_threshold else 0 for _ in predictions]
print(best_threshold)
accuracy_thr = accuracy_score(y_test_nn, y_pred_thr)
print("New accuracy ",accuracy_thr)
tn_thr, fp_thr, fn_thr, tp_thr = confusion_matrix(y_test_nn, y_pred_thr).ravel()

print("Average Accuracy : " ,(tp_thr+tn_thr)/(tp_thr+tn_thr+fp_thr+fn_thr))
print("Sensitivity: ",(tp_thr)/(tp_thr+fn_thr))
print("Specificity: ",(tn_thr)/(tn_thr+fp_thr))
print("MCC: ",matthews_corrcoef(y_test_nn, y_pred_thr))
print("F-measure : ", f1_score(y_test_nn, y_pred_thr))

"""##Threshold tuning

##Confusion Matrix
"""

import seaborn as sn

nm = MinMaxScaler()
data_x_nn = data_x.copy(deep=True)
target_y_nn = y_class_info.copy(deep=True)
data_x_nn = nm.fit_transform(data_x_nn)

X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(data_x_nn, target_y_nn, test_size = 0.20, random_state = 42)
predictions_loaded_model  = loaded_model.predict(X_test_nn)

best_threshold = 0.5405
y_pred_thr = [ 1 if _ >= best_threshold else 0 for _ in predictions_loaded_model]
array = confusion_matrix(y_test_nn, y_pred_thr)
tn_thr, fp_thr, fn_thr, tp_thr = confusion_matrix(y_test_nn, y_pred_thr).ravel()

df_cm = pd.DataFrame(array, index = [i for i in ["non - PD", "PD"]],
                  columns = [i for i in ["non-PD","PD"]])
accuracy_thr = accuracy_score(y_test_nn, y_pred_thr)
print("New accuracy ",accuracy_thr)
print("Average Accuracy : " ,(tp_thr+tn_thr)/(tp_thr+tn_thr+fp_thr+fn_thr))
print("Sensitivity: ",(tp_thr)/(tp_thr+fn_thr))
print("Specificity: ",(tn_thr)/(tn_thr+fp_thr))
print("MCC: ",matthews_corrcoef(y_test_nn, y_pred_thr))

plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, cmap="YlGnBu",annot=True,annot_kws={"fontsize":48})
plt.savefig("NN_Conf_matrix.png", bbox_inches = "tight")

count=0
for i, j in zip(y_pred_thr, y_test_nn):
    if i==1 and j==0:
        count+=1
print(count)

"""##The LIME implementation"""

#instead going with model.predict()
def predict_proba(X):
    X_t = X[np.newaxis,:]
    val = model_final.predict(X_t)
    return np.array([1-val, val]).reshape(1,-1)

#Testing te above function
# X_t = X_test_nn.iloc[100]
# X_t = X_t[np.newaxis,:]
res = []
for i in range(X_test_nn.shape[0]):
    ans = predict_proba(X_test_nn.iloc[i])
    res.append(ans)
print(res)

X_test_nn = pd.DataFrame(data= X_test_nn, columns=features)

list(np.round_(loaded_model.predict(X_test_nn)).ravel()).count(0)

!pip install lime

data_x_nn = data_x.copy(deep=True)
target_y_nn = y_class_info.copy(deep=True)

features = data_x_nn.columns.to_list()
labels = [set(target_y_nn)]
print(len(features), data_x_nn.shape)

X_train_nn = pd.DataFrame(data= X_train_nn, columns=features)
X_train_nn.shape
# useful
# list(np.round_(loaded_model.predict(X_test_nn)).ravel()).count(1)
# list(y_test_nn.ravel()).count(1)
# np.round_(model.predict(X_test_nn.iloc[i].values[np.newaxis,:]))

loaded_model = model_final
list(np.round_(loaded_model.predict(X_test_nn)).ravel()).count(0)

X_test_nn = pd.DataFrame(data= X_test_nn, columns=features)
X_test_nn.shape

print(type(X_train_nn.values))
target_y_nn.unique()
np.array(["non-PD", "PD"])

import lime
from lime import lime_tabular
explainer = lime.lime_tabular.LimeTabularExplainer(X_train_nn.values, feature_names=features, class_names =np.array(["non-PD", "PD"]) , discretize_continuous=True)

exp = explainer.explain_instance(X_test_nn.iloc[i].values, loaded_model.predict, num_features=26, top_labels=26)

exp.show_in_notebook(show_table=True, show_all=True)

def add_to_top_five(res_):
    for weight, para in res_:
        top_five_gen[(para, weight)] = top_five_gen.get(para, 0)+1

def analyse_expl_map(t):
    t = exp.as_map()
    res = []

    for k,v in t.items():
        for ele in v:
            i,j = ele
            res.append((j,features[i]))

    res.sort(reverse = True)
    add_to_top_five(res[:])

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

top_five_gen = {}
corr = 0
num = 0
for i in range(X_test_nn.shape[0]):
    num += 1
    if num%10==0:
        print(num)
    pr = np.round_(loaded_model.predict(X_test_nn.iloc[i].values.reshape(1,-1)))
    real = y_test_nn.iloc[i]
    if pr[0][0]==real:
        corr += 1
        expl_ = explainer.explain_instance(X_test_nn.iloc[i].values, loaded_model.predict, num_features=26, top_labels=26)
        analyse_expl_map(expl_)

sort_top_five_gen = sorted(top_five_gen.items(), key=lambda x: x[1], reverse=True)
print(sort_top_five_gen)
print(corr, num, (corr/num*100))

pr = np.round_(model_final.predict(X_test_nn.iloc[i].values.reshape(1,-1)))
pr[0][0]

q=2
print(np.round_(model_final.predict(X_test_nn.iloc[q].values.reshape(1,-1)))[0][0])
print(y_test_nn.iloc[q])
expl_ = explainer.explain_instance(X_test_nn.iloc[q].values, model_final.predict, num_features=26, top_labels=26)
expl_.as_map()

import time
offset=0
print("It begins now at 20:15:00")
for i in range(0, 10):
    time.sleep(600)
    offset+=10
    print("+", offset, " minutes")

