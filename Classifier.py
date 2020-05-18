import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler


def classify():
    """
    Classifier for predicting target groups
    """
    dt = pd.read_excel('./m.xlsx')
    ds = pd.read_excel('./sample.xlsx')
    print(len(dt))
    ds.drop_duplicates(subset='web', keep = "first", inplace=True)
    print(len(ds))
    dt['label'] = ds['target_groups']

    singleLabel = []
    for i in list(dt['label']):
        j = i.split('|')
        singleLabel.append(j[0])
    print(singleLabel)
    dt['label'] = singleLabel

    dt['new_text'] = dt["Website"].astype(str) + '' + dt["clean_data"].astype(str)
    print(dt['new_text'])
    print(dt['label'])


    encoder = LabelEncoder()

    integer_encoded = encoder.fit_transform(dt["label"].astype(str)).reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(dt['new_text'], np.ravel(integer_encoded), test_size=0.15, random_state=11, stratify=integer_encoded)

    vectorizer = TfidfVectorizer(max_features=30000)
    xtrain = vectorizer.fit_transform(x_train)
    xval = vectorizer.transform(x_test)


    sampler = RandomOverSampler(sampling_strategy='minority')
    sm = SMOTE()

    xtrain, y_train = smt.fit_resample(xtrain, y_train)
    xtrain, y_train = sm.fit_resample(xtrain, y_train)

    print(np.bincount(y_train))

    final = LinearSVC()
    final.fit(xtrain, y_train)
    train_score = final.score(xtrain, y_train)
    print('Train score =', train_score)
    prediction = final.predict(xv)

    predictionList = encoder.inverse_transform(prediction)
    originalyList = encoder.inverse_transform(y_test)

    print("#####Predictions: ", predictionList)

    print('Report', classification_report(y_test, prediction))
    text = list(x_test)

    dq = pd.DataFrame(list(zip(predictionList, originalyList)), columns=['predicted label', 'original label'])
    print(dq)
    dq['text'] = text
    dq.to_csv('./final.csv')



