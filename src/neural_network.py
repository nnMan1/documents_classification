import data_loader
from distance import Distance
import joblib
from scipy import stats
from sklearn import metrics
from sklearn.ensemble import IsolationForest
import random
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std
import numpy as np
from sklearn import svm
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer

class AnomalyDetector:
    def __init__(self, n_components=1, trashold=1):
        self.trashold = trashold
        self.model = GaussianMixture(covariance_type='diag', n_components=n_components)

    def __find_optimal_center(self, data):
        tmp=self.model.score_samples(data)
        tmp = [int(t) for t in tmp]
        tmp = [len([i for i in range(len(tmp)) if tmp[i]>t]) for t in range(max(tmp))]
        for i in range(len(tmp)):
            if tmp[len(tmp)-i-1]>0.1*len(data):
                self.max_score = len(tmp)-i
                return i

        self.max_score = len(tmp)
        return len(tmp)

    def fit(self,data_from_model, test_data=None):
        #uzima niz stringova
        if test_data is None:
            test_data = data[int(len(data_from_model)*2/3):]
            data_from_model = data[:int(len(data_from_model)*2/3)]
        
        self.model.fit(data_from_model)

        max_score = self.__find_optimal_center(test_data) #podaci naosnovu kojih cemo da odredimo centar
        return self

    def predict(self, data):
        score_samples = self.model.score_samples(data)

        ans = [-1]*len(data)
        for i in range(len(score_samples)):
            if score_samples[i]<self.max_score-self.trashold:
                ans[i] = 1
        
        return ans
    

def find_novelties(data_from_model, new_data, n_components = 1, trashold_similarity=0.17, trashold_offset=1):

    data_from_model = random.sample(data_from_model, min(len(data_from_model), 1200))

    test_data = data_from_model[int(len(data_from_model)*2/3):]
    data_from_model = data_from_model[:int(len(data_from_model)*2/3)]

    distance = Distance(data_from_model, trashold_similarity)
    data_from_model = distance.calculate_distance_matrix_optimized()
    test_data = distance.calculate_distance_for_optimized(test_data)

    anomalies_detector = AnomalyDetector(n_components=n_components, trashold=trashold_offset)
    anomalies_detector.fit(data_from_model, test_data=test_data)


    data_to_classify = new_data
    data_to_classify = distance.calculate_distance_for_optimized(data_to_classify)

    return anomalies_detector.predict(data_to_classify)

data = data_loader.load_documents(['earn','acq','money-fx','grain','crude'])

ans= find_novelties([d.text for d in data['train']], [d.text for d in data['test']])

for i in range(len(ans)):
    if ans[i]==-1:
        print(i, data['test'][i].category)

br = 0
for i in range(len(ans)):
    if ans[i]==-1 and data['test'][i].category[0] in ['earn','acq','money-fx','grain','crude']:
        br += 1

print("Correct data marked as novelty", br)

br = 0
for i in range(len(ans)):
    if ans[i]==1 and data['test'][i].category[0] not in ['earn','acq','money-fx','grain','crude']:
        br += 1

print("Novelties marked as normal data", br)

print("Total", len(ans))






