from flask import Flask, render_template, url_for, request
import pandas as pd 
import numpy as np 
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN
import seaborn as sns
import shap
shap.initjs()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def analyze():
    if request.method == 'POST':
        V1 = request.form['AGE']
        V2 = request.form['OCC']
        V3 = request.form['EDU']
        V4 = request.form['SMK PK YRS']
        V5 = request.form['TOB CHEW']
        V6 = request.form['2ND HAND SMK']
        V7 = request.form['eCO LEVEL']
        V8 = request.form['ALC']
        V9 = request.form['LIFE ALC']
        V10 = request.form['BTL NUT']
        V11 = request.form['BRUSH']
        V12 = request.form['BRUSH FREQ']
        V13 = request.form['GUM BLEED']
        V14 = request.form['RPD']
        V15 = request.form['FRUIT']
        V16 = request.form['FISH']
        V17 = request.form['RED MEAT']
        V18 = request.form['SPICE']
        V19 = request.form['SPICE SCORE']
        V21 = request.form['DENT VISIT']
        V22 = request.form['MOUTHWASH']
        V23 = request.form['FAM CANCER']
        V24 = request.form['1 DEG REL CANCER']
        V25 = request.form['2ND DEG REL CANCER']
        V26 = request.form['CLIN TYPE FAM CANCER']
        V27 = request.form['CCI']
        V28 = request.form['HTN']
        
        model_choice = request.form['model_choice']

        sample_data = [V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
                       V11, V12, V13, V14, V15, V16, V17, V18, V19,
                       V21, V22, V23, V24, V25, V26, V27, V28]

        sample_data1 = [float(i) for i in sample_data]

        ex1=np.array(sample_data).reshape(1, -1)
        ex2=np.array(sample_data1).reshape(1, -1)
        

        if model_choice == 'Soft voting ensemble':
            df_train = pd.read_csv('static/all1.csv')
            features = ['AGE', 'OCC', 'EDU', 'SMK PK YRS', 'TOB CHEW', '2ND HAND SMK',
            'eCO LEVEL', 'ALC','LIFE ALC', 'BTL NUT', 'BRUSH', 'BRUSH FREQ',
            'GUM BLEED', 'RPD', 'FRUIT', 'FISH','RED MEAT', 'SPICE', 'SPICE SCORE',
            'DENT VISIT', 'MOUTHWASH', 'FAM CANCER', '1 DEG REL CANCER', '2ND DEG REL CANCER',
            'CLIN TYPE FAM CANCER', 'CCI', 'HTN']
            x = df_train[features]
            y = df_train.O3
            sm = SMOTEENN(random_state=0)
            x_res, y_res = sm.fit_resample(x,y)
            m1 = KNeighborsClassifier(n_neighbors = 3)
            m2 = ExtraTreesClassifier(random_state=0, max_depth=3)
            m4 = AdaBoostClassifier(random_state=0)
            m5 = RandomForestClassifier(random_state=0, max_depth = 1)
            ensemble = VotingClassifier(estimators=[('knn', m1), ('ert', m2), 
                                        ('ada', m4), ('rf', m5)], voting = 'soft')
            ensemble = ensemble.fit(x_res,y_res)
            result_prediction = ensemble.predict(ex1)
            prob_prediction = ensemble.predict_proba(ex1)
            pred1 = ensemble.predict_proba
            explainer = shap.Explainer(pred1, x)
            shap_test = explainer(ex2)
            shap.plots.bar(shap_test[:,:,1][0], show=False)
            plt.savefig('static/shap.jpg',  bbox_inches='tight')
            
        elif model_choice == 'Retrained soft voting ensemble':
            df_train = pd.read_csv('static/all1.csv')
            features = ['AGE', 'OCC', 'EDU', 'SMK PK YRS', 'TOB CHEW', '2ND HAND SMK',
            'eCO LEVEL', 'ALC','LIFE ALC', 'BTL NUT', 'BRUSH', 'BRUSH FREQ',
            'GUM BLEED', 'RPD', 'FRUIT', 'FISH','RED MEAT', 'SPICE', 'SPICE SCORE',
            'DENT VISIT', 'MOUTHWASH', 'FAM CANCER', '1 DEG REL CANCER', '2ND DEG REL CANCER',
            'CLIN TYPE FAM CANCER', 'CCI', 'HTN']
            x = df_train[features]
            y = df_train.O1
            sm = SMOTEENN(random_state=0)
            x_res, y_res = sm.fit_resample(x,y)
            m1 = KNeighborsClassifier(n_neighbors = 3)
            m2 = ExtraTreesClassifier(random_state=0, max_depth=3)
            m4 = AdaBoostClassifier(random_state=0)
            m5 = RandomForestClassifier(random_state=0, max_depth = 1)
            ensemble = VotingClassifier(estimators=[('knn', m1), ('ert', m2), 
                                        ('ada', m4), ('rf', m5)], voting = 'soft')
            ensemble = ensemble.fit(x_res,y_res)
            result_prediction = ensemble.predict(ex1)
            prob_prediction = ensemble.predict_proba(ex1)
            pred1 = ensemble.predict_proba
            explainer = shap.Explainer(pred1, x)
            shap_test = explainer(ex2)
            shap.plots.bar(shap_test[:,:,1][0], show=False)
            plt.savefig('static/shap.jpg',  bbox_inches='tight')

            
    return render_template('predict.html', result_prediction = result_prediction, url = 'static/shap.jpg', prob_prediction = prob_prediction, model_selected=model_choice)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=9020)
