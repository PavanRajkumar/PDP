import numpy as np
import matplotlib as mlt
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('train_data.txt', sep=",", header=None)
data.columns = ["Subject id", "Jitter (local)" ,"Jitter (local, absolute)" ,"Jitter (rap)" ,"Jitter (ppq5)", "Jitter (ddp)", "Shimmer (local)", "Shimmer (local, dB)" ,"Shimmer (apq3)", "Shimmer (apq5)", "Shimmer (apq11)", "Shimmer (dda)", "AC" ,"NTH" ,"HTN", "Median pitch", "Mean pitch", "Standard deviation", "Minimum pitch", "Maximum pitch", "Number of pulses", "Number of periods", "Mean period", "Standard deviation of period", "Fraction of locally unvoiced frames", "Number of voice breaks", "Degree of voice breaks", "UPDRS", "class information"]
data.shape

y_updrs = data.UPDRS # for regression
y_class_info = data["class information"]#for classification
data_x = data.drop(labels=['UPDRS', "Subject id", "class information"], axis=1, inplace = False)
print(data_x.shape, y_updrs.shape, y_class_info.shape)

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold


total_folds = 10
kfold = KFold(n_splits= total_folds, shuffle=True, random_state=42)

acc_per_fold = []
fold=0

for train_index, test_index in kfold.split(data_x, y_class_info):
    fold+=1
    X_train, X_test, y_train, y_test =  data_x.iloc[train_index], data_x.iloc[test_index], y_class_info.iloc[train_index], y_class_info.iloc[test_index]

    neigh = KNeighborsClassifier(n_neighbors=15)
    neigh.fit(X_train, y_train)

    y_pred = neigh.predict(X_test)
    print("\nFold : ",fold,"\n")
    print(neigh.score(X_test, y_test))

    acc_per_fold.append(neigh.score(X_test, y_test))

for i in range(total_folds):
    print("Fold ", i, "accuracy", acc_per_fold[i])

print("\nMean Accuracy = ",np.mean(acc_per_fold),'(+/-',np.std(acc_per_fold),')');

X_train, X_test, y_train, y_test =  train_test_split(data_x, y_class_info, test_size=0.20, random_state=42)
neigh = KNeighborsClassifier(n_neighbors=11)
neigh.fit(X_train, y_train)

y_pred = neigh.predict(X_test)
print(neigh.score(X_test, y_test))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("Average Accuracy : " ,(tp+tn)/(tp+tn+fp+fn))
print("Sensitivity: ",(tp)/(tp+fn))
print("Specificity: ",(tn)/(tn+fp))
print("MCC: ",matthews_corrcoef(y_test, y_pred))

