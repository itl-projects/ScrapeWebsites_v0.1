import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler


def classify():
    """
    Classifier for predicting target groups
    """
    frame = pd.read_excel('./initial.xlsx')
    singleLabel = []
    for i in list(frame['class']):
        j = i.split('|')
        singleLabel.append(j[0])
    print(singleLabel)
    frame['class'] = singleLabel
    frame['new_text'] = frame["Website"].astype(str) + '' + frame["clean_data"].astype(str)
    print(frame['new_text'])

    encoder = LabelEncoder()

    integer_encoded = encoder.fit_transform(frame["class"].astype(str)).reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(frame['new_text'], np.ravel(integer_encoded), test_size=0.10, random_state=11, stratify=integer_encoded)

    vectorizer = TfidfVectorizer(max_features=30000)
    xtrain = vectorizer.fit_transform(x_train)
    xval = vectorizer.transform(x_test)

    sampler_over = RandomOverSampler(sampling_strategy='minority')
    sampler = SMOTE()

    xtrain, y_train = sampler_over.fit_resample(xtrain, y_train)
    xtrain, y_train = sampler.fit_resample(xtrain, y_train)

    print(np.bincount(y_train))

    final = LinearSVC()
    final.fit(xtrain, y_train)
    train_score = final.score(xtrain, y_train)
    print('Train score =', train_score)
    prediction = final.predict(xval)
    predictionList = encoder.inverse_transform(prediction)
    original_y = encoder.inverse_transform(y_test)

    print("#Predictions: ", predictionList)

    print('Report', classification_report(y_test, prediction))
    text = list(x_test)

    final_frame = pd.DataFrame(list(zip(predictionList, original_y)), columns=['predicted label', 'original label'])
    print(final_frame)
    final_frame['text'] = text
    final_frame.to_csv('./final.csv')
classify()
