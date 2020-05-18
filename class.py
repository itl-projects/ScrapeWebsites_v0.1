import os
import string
from collections import Counter
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import gensim
from sklearn.preprocessing import LabelEncoder
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import class_weight
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split
dt = pd.read_excel('./m.xlsx')
dt=dt.fillna('na')
li = []
for i in list(dt['target_groups']):
    j =i.split('|')
    li.append(j[0])
print(li)
dt['target_groups'] = li

dt['scraped_text'] = dt["scraped_text"].str.cat(dt["Website"],sep="")


#all = sum(li,[])
#print(len(set(all)))


rare_words = ['apr', 'two', 'first', 'thu', 'fri', 'mon', 'tue', 'wed', 'sat', 'sun', 'month', 'day', 'year',
              'thursday', 'january', 'february', 'march', 'april', 'june', 'july', 'august', 'september', 'october',
              'november', 'december', 'friday', 'saturday', 'sunday', 'thursday', 'monday', 'tuesday', 'wednesday',
              'date', 'week', 'daily', 'feb', 'september', "morning", "evening", "years", "weeks", "till", "ago",
              'will', 'werent', 'whom', 'three', 'first', 'twice']


def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize_stemming(text):
    pos = pos_tag([text])
    return WordNetLemmatizer().lemmatize(text, pos=get_simple_pos(pos[0][1]))


def preprocess(text):
    result = []
    stop_free = " ".join([lemmatize_stemming(token) for token in simple_preprocess(text) if
                          token not in STOPWORDS and len(
                              token) > 2 and token not in rare_words])
    return stop_free


dt['clean_data'] = dt.scraped_text.apply(preprocess)
print(dt)

"""from sklearn.preprocessing import MultiLabelBinarizer

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(dt['target_groups'])

# transform target variable
y = multilabel_binarizer.transform(dt['target_groups'])
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
xtrain, xval, ytrain, yval = iterative_train_test_split(dt['clean_data'], y, test_size=0.3)
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)


from sklearn.linear_model import LogisticRegression

# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier

# Performance metric
from sklearn.metrics import f1_score
lr = LogisticRegression()
clf = OneVsRestClassifier(lr)

# fit model on train data
clf.fit(xtrain_tfidf, ytrain)

# make predictions for validation set
y_pred = clf.predict(xval_tfidf)
print(multilabel_binarizer.inverse_transform(y_pred))
print(classification_report(yval,y_pred))
print(hamming_loss(yval,y_pred))


# predict probabilities
y_pred_prob = clf.predict_proba(xval_tfidf)

t = 0.1 # threshold value
y_pred_new = (y_pred_prob >= t).astype(int)

# evaluate performance
print(classification_report(yval,y_pred_new))
print(multilabel_binarizer.inverse_transform(y_pred_new))
print(hamming_loss(yval,y_pred_new))



"""

def vectorization(text):
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(text).toarray()
    return x


tf = vectorization(dt['clean_data'])
# ds=ds.drop(['clean_data'], axis=1)

print(dt.target_groups.unique())

label = LabelEncoder()
integer_encoded = label.fit_transform(dt["target_groups"].astype(str)).reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(tf, np.ravel(integer_encoded), test_size=0.30, random_state=11, stratify=integer_encoded)
smt = RandomOverSampler(sampling_strategy='minority')
sm = SMOTE()

unique, counts = np.unique(y_train, return_counts= True)
print((unique,counts))
class_weigh = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
e1 = np.unique(y_train)
e2 =class_weigh
res = {e1[i]: e2[i] for i in range(len(e1))}

x_train, y_train = smt.fit_resample(x_train, y_train)
x_train, y_train = sm.fit_resample(x_train,y_train)

print(np.bincount(y_train))



param_grid = {'C': [0.1],
              'gamma': [0.1],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(x_train, y_train)
# print best parameter after tuning
print(grid.best_params_)
print(grid.best_estimator_)

"""from sklearn.preprocessing import MultiLabelBinarizer

multilabel_binarizer = MultiLabelBinarizer()

y = multilabel_binarizer.fit_transform(dt['label'])
print(y)


vectorizer = TfidfVectorizer(max_df=0.1, max_features=20000)

xtrain, xtest, ytrain, ytest = train_test_split(
    dt['new_text'], y, random_state=9, test_size=0.2,shuffle=True
)
xt = vectorizer.fit_transform(xtrain)
xv = vectorizer.transform(xtest)


print(xt.shape)
print(xv.shape)

clf = OneVsRestClassifier(MultinomialNB())
#for label in labels:
clf.fit(xt, ytrain)
prediction = clf.predict(xv)
print(prediction)

y_pred_prob = clf.predict_proba(xv)

t = 0.2 # threshold value
y_pred_new = (y_pred_prob >= t).astype(int)

# evaluate performance
print(classification_report(ytest,y_pred_new))
print(multilabel_binarizer.inverse_transform(y_pred_new))
print(hamming_loss(ytest,y_pred_new))
print('test accuracy is {}'.format(accuracy_score(ytest, prediction)))
print('classification report {}'.format(classification_report(ytest,prediction)))
print(multilabel_binarizer.inverse_transform(prediction))"""


"""
def vectorization(text):
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(text).toarray()
    return x

tf = vectorization(dt['clean_data'])

label = LabelEncoder()
integer_encoded = label.fit_transform(dt["Class"][:100].astype(str)).reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(tf, np.ravel(integer_encoded), test_size=0.20, random_state=0, stratify= integer_encoded)

unique, counts = np.unique(y_train, return_counts= True)
print((unique,counts))"""




final = SVC(class_weight=res)
final.fit(x_train,y_train)
train_score = final.score(x_train, y_train)
print('Train score =', train_score)
prediction = final.predict(x_test)
print("#####Predictions: ", label.inverse_transform(prediction))

print('from here', classification_report(y_test, prediction))

print(y_test.shape)
print(y_test)

p=np.array(prediction)
print(p)
print(confusion_matrix(y_test, prediction))
y_test = y_test.reshape(1,-1)
test_score = final.score(y_test,p)
print('Test score = ', test_score)

#test_score = final.score(yee.reshape(-1,1), prediction.reshape(-1,1))
#print('Test Score =', test_score)"""

