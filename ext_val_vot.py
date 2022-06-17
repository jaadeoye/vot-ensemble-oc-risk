import pandas as pd
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
import shap
shap.initjs()

#import data
df_train = pd.read_csv('/Users/jaadeoye/Desktop/ml_IADR/train.csv')
df_test = pd.read_csv('/Users/jaadeoye/Desktop/ml_IADR/test.csv')
features = ['V1','V2', 'V3', 'V4', 'V5', 'V6','V7','V8','V9','V10','V11','V12','V13','V14',
                 'V15','V16','V17','V18','V19','V21','V22','V23','V24',
              'V25','V26','V27','V28']
x = df_train[features]
y = df_train.O1
a = df_test[features]
b = df_test.O1
#train model
sm = SMOTEENN(random_state=0)
x_res, y_res = sm.fit_resample(x,y)

#models
m1 = KNeighborsClassifier(n_neighbors = 3)
m2 = ExtraTreesClassifier(random_state=0, max_depth=3)
m3 = DecisionTreeClassifier(random_state=0, max_depth = 3)
m4 = AdaBoostClassifier(random_state=0)
m5 = RandomForestClassifier(random_state=0, max_depth = 1)
m6 = LogisticRegression(random_state=0)
m7 = svm.SVC(kernel='rbf', probability = True)


ensemble = VotingClassifier(estimators=[('knn', m1), ('ert', m2), 
                                        ('ada', m4), ('rf', m5)], voting = 'soft')
ensemble = ensemble.fit(x_res,y_res)
scores = cross_val_score(ensemble, x_res, y_res, cv=10, scoring = "accuracy")
scores
print((scores.mean(), scores.std()))

#Prediction
pred=ensemble.predict(a)
pred1 = ensemble.predict_proba

#confusion matrix
cnf_matrix = metrics.confusion_matrix(b, pred)
cnf_matrix

#print
newb = b.to_numpy()
roc = roc_auc_score(newb, pred)
print("Accuracy:",metrics.accuracy_score(b,pred))
print("Recall:",metrics.recall_score(b,pred))
print("Precision:",metrics.precision_score(b,pred))
print("F1 score:",metrics.f1_score(b,pred))
print(cnf_matrix)
print(roc)
