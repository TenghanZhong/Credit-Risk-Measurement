import numpy as np
import pandas as pd
import sklearn as skl
import warnings
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve as rc
from sklearn.metrics import auc
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.preprocessing import StandardScaler as ssr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.linear_model import LogisticRegression as lr

ordata = pd.read_csv("C:\\Users\\zth020906\\Desktop\\2023Zscore.csv", encoding='gbk')
ordata
type(ordata)
ordata.dropna(axis=0, how='any')
import re
ordata = ordata.rename(columns={'证券代码': 'IND', '证券简称': 'NAME'})
ordata
ordata = ordata.dropna(axis=0, how='any')
ordata
ordata.index = range(1, 4640)
ordata
ordata['NAME'] = ordata['NAME'].str.contains('S*ST')
ordata
ordata['NAME'].loc[63]
ordata['NAME'].sum()

STdata = ordata.loc[ordata['NAME'] == 1]
NSTdata = ordata.loc[ordata['NAME'] == 0]
STdata
NSTdata
NSTtrain = NSTdata.sample(75)
NSTtext = NSTdata.drop(NSTtrain.index, axis=0)
NSTtrain
NSTtext
STtrain = STdata.sample(75)
STtext = STdata.drop(STtrain.index, axis=0)
STtrain
STtext
traindata = NSTtrain.append(STtrain)
traindata
textdata = NSTtext.sample(50).append(STtext)
textdata

Zscore_train = traindata.iloc[:, 2:7]
Zscore_train
Zscore_train_lable = traindata.iloc[:, 1:2]
Zscore_train_lable
Zscore_train_S = ssr().fit_transform(Zscore_train)
Zscore_train_S

Zscore_text = textdata.iloc[:, 2:7]
Zscore_text
Zscore_text_lable = textdata.iloc[:, 1:2]
Zscore_text_lable
Zscore_text_S = ssr().fit_transform(Zscore_text)
Zscore_text_S

LDAM = lda(n_components=1).fit(Zscore_train, Zscore_train_lable)
LDAM.get_params()
LDAM.coef_
lotrain = LDAM.transform(Zscore_train)
Zscore_classes = LDAM.classes_
Zscore_label_pre = LDAM.predict(Zscore_text)
Zscore_score = LDAM.score(Zscore_text, Zscore_text_lable)
Zscore_score
Zscore_label_pre

fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
plt.scatter(Zscore_text.iloc[:, 0], Zscore_text.iloc[:, 1], Zscore_text.iloc[:, 2],
            c=Zscore_text_lable['NAME'])

plt.scatter(lotrain, lotrain, c=np.squeeze(Zscore_train_lable['NAME']))
plt.show()

Zscore_score

Zscore_pred_score = metrics.precision_score(Zscore_text_lable, Zscore_label_pre, average='macro')
Zscore_recall_score = metrics.recall_score(Zscore_text_lable, Zscore_label_pre, average='macro')
Zscore_f1_score = metrics.f1_score(Zscore_text_lable, Zscore_label_pre, average='macro')

Zscore_pred_score
Zscore_recall_score
Zscore_f1_score

Zscore_fpr, Zscore_tpr, Zscore_threshold = rc(Zscore_text_lable, LDAM.predict_proba(Zscore_text)[:, 1])
Zscore_auc_score = auc(Zscore_fpr, Zscore_tpr)
print('auc: ', Zscore_auc_score)
plt.figure(figsize=(8, 5))  # Only set the figure size here
plt.plot(Zscore_fpr, Zscore_tpr, 'b', label='AUC=%0.2f' % Zscore_auc_score)
plt.legend(loc='lower right', fontsize=14)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('TPR - True Positive Rate', fontsize=16)
plt.xlabel('FPR - False Positive Rate', fontsize=16)
plt.show()

# LOGISTIC
log = lr().fit(X=Zscore_train, y=Zscore_train_lable.values)

# In[56]:

log.score(Zscore_train, Zscore_train_lable.values)

log_label_pred = log.predict(Zscore_text)

log_pred_score = metrics.precision_score(Zscore_text_lable, log_label_pred, average='macro')
log_recall_score = metrics.recall_score(Zscore_text_lable, log_label_pred, average='macro')
log_f1_score = metrics.f1_score(Zscore_text_lable, Zscore_label_pre, average='macro')

log_pred_score
log_recall_score
log_f1_score

LR_fpr, LR_tpr, LR_threshold = rc(Zscore_text_lable, log.predict_proba(Zscore_text)[:, 1])
LR_auc_score = auc(LR_fpr, LR_tpr)
print('auc: ', LR_auc_score)
plt.figure(figsize=(8, 5))  # Only set the figure size here
plt.plot(LR_fpr, LR_tpr, 'b', label='AUC=%0.2f' % LR_auc_score)
plt.legend(loc='lower right', fontsize=14)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('TPR - True Positive Rate', fontsize=16)
plt.xlabel('FPR - False Positive Rate', fontsize=16)
plt.show()
