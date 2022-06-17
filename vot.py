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
import pickle
import shap
shap.initjs()
df_train = pd.read_csv('/Users/jaadeoye/Desktop/ml_IADR/all1.csv')
features = ['AGE', 'OCC', 'EDU', 'SMK PK YRS', 'TOB CHEW', '2ND HAND SMK',
            'eCO LEVEL', 'ALC','LIFE ALC', 'BTL NUT', 'BRUSH', 'BRUSH FREQ',
            'GUM BLEED', 'RPD', 'FRUIT', 'FISH','RED MEAT', 'SPICE', 'SPICE SCORE',
            'DENT VISIT', 'MOUTHWASH', 'FAM CANCER', '1 DEG REL CANCER', '2ND DEG REL CANCER',
            'CLIN TYPE FAM CANCER', 'CCI', 'HTN']
x = df_train[features]
y = df_train.O3
#train model
sm = SMOTEENN(random_state=0)
x_res, y_res = sm.fit_resample(x,y)
#models
m1 = KNeighborsClassifier(n_neighbors = 3)
m2 = ExtraTreesClassifier(random_state=0, max_depth=3)
m4 = AdaBoostClassifier(random_state=0)
m5 = RandomForestClassifier(random_state=0, max_depth = 1)
#voting classifier
ensemble = VotingClassifier(estimators=[('knn', m1), ('ert', m2), 
                                        ('ada', m4), ('rf', m5)], voting = 'soft')
ensemble = ensemble.fit(x_res,y_res)
#shap
pred1 = ensemble.predict_proba
explainer = shap.Explainer(pred1, x)
#pickle
pickle.dump(ensemble, open('ensemble', 'wb'))
