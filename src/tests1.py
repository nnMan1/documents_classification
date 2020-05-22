import nltk
import numpy as np
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import joblib
import copy
import data_loader
from tokenizer import Tokenizer

#stop_words = set(stopwords.words('english'))

def check_novelty(cls_model, cls_vectorizer, cls_transformer, cls_databunch, docs_new, classes):
    clf = joblib.load(cls_model)
    vectorizer = joblib.load(cls_vectorizer)
    transformer = joblib.load(cls_transformer)
    vectorizer1 = copy.deepcopy(vectorizer)

    X_new_counts1 = vectorizer.transform(docs_new)
    X_new_counts2 = vectorizer1.fit_transform(docs_new)

    print(X_new_counts1.sum()/X_new_counts2.sum())

    X_new_tfidf = transformer.transform(X_new_counts1)

    decision_function = clf.decision_function(X_new_tfidf)

    print(decision_function)

    decision_value = (decision_function+0.5)*(1-(1-X_new_counts1.sum()/X_new_counts2.sum())/2)-0.5
    #ukoliko ima dosta rijeci koje se ne pojavljuju u dosadasnjem recniku, 
    #ali se pojavljuju u novom dokumentu
    #treba im dati odredjenu tezinu - tezina ce biti proporcionalna ukupnom broju 
    #u novom dokumentu rijeci i broju novih rijeci

    print(decision_value)

    for i in range(len(decision_value)):
        if(decision_value[i] < 0):
            print('File "{}" is novelty, {}'.format(i, classes[i]))
        
    
def find_outliers(train_data):

    vectorizer = CountVectorizer(decode_error = 'ignore')
    
    #Svectorizer = CountVectorizer(decode_error = 'ignore', ngram_range=(2, 2), analyzer='word')
    #vectorizer = CountVectorizer(decode_error = 'ignore', ngram_range=(9, 9), analyzer='char')
    #vectorizer = CountVectorizer(decode_error = 'ignore', ngram_range=(5, 5), analyzer='char_wb')

    train_counts = vectorizer.fit_transform(train_data)
    #features = vectorizer.get_feature_names()

    tfidf_transformer = TfidfTransformer()
    train_tfidf = tfidf_transformer.fit_transform(train_counts)

    print("Data ready for clustering")
    print('--------------------------------------------')

    clf = IsolationForest(n_estimators=200, max_samples=50, contamination=0.07, bootstrap=False, n_jobs=-1, random_state=42)
                            #broj drveta, kolicina zagadjenosti podataka (pomocu ovoga kontrolisemo vjerovaatnocu da se fajl proglasi za outlayer [0-1])
    clf.fit(train_tfidf)

    print('Training finished')
    print('--------------------------------------------')
    print("Finding outliers")
    print('--------------------------------------------')

    y_pred = clf.predict(train_tfidf)

    #print(train_data.filenames)

    print('Outliers are:')

    #for i in range(len(y_pred)):
        #if y_pred[i]==-1:
            #print(i)
    
    #print(clf.decision_function(train_tfidf)) #ako je decision_function<0 onda kazemo da je outlayer
    #print(y_pred)
    
    #joblib.dump(clf, './sgd_classifier.pkl', compress=9)
    #joblib.dump(tfidf_transformer, './sgd_transformer.pkl', compress=9)
    #joblib.dump(vectorizer, './sdg_vectorizer.pkl', compress=9) 
    #joblib.dump(train_data, './sgd_traindata.pkl', compress=9) 

    return y_pred

data = data_loader.load_documents()
tmp = find_outliers([d.text for d in data['train'][:1000]])

for i in range(len(tmp)):
    if tmp[i]==-1 and (data['train'][i].category in [['earn'], ['acq'], ['money-fx'], ['trade']]):
        print(i, data['train'][i].category)

#check_novelty('./sgd_classifier.pkl', './sdg_vectorizer.pkl', './sgd_transformer.pkl', './sgd_traindata.pkl', data['test'], data['test_classes'])
