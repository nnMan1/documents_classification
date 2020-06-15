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

def filer(data_from_model, data_to_classify, category=None):

    test_data = data_from_model[int(len(data_from_model)*2/3):]
    data_from_model = data_from_model[:int(len(data_from_model)*2/3)]

    distance = Distance(data_from_model, 0.15)
    data_from_model = distance.calculate_distance_matrix_optimized()
    data_to_classify = distance.calculate_distance_for_optimized(data_to_classify)
    test_data = distance.calculate_distance_for_optimized(test_data)

    model = GaussianMixture(covariance_type='diag')
    #model = BayesianGaussianMixture()
    model.fit(data_from_model)

    max_score = max(model.score_samples(test_data))

    score_samples = model.score_samples(data_to_classify)

    ans = [-1]*len(data_to_classify)
    for i in range(len(score_samples)):
        if score_samples[i]<max_score-1:
            ans[i] = 1
    
    return ans

def find_novelties(train_data, new_data, correct_categories):

    included = [-1]*len(new_data)
    
    for correct_category in correct_categories:
        data_from_model = [d.text for d in train_data if d.category[0] == correct_category]

        data_from_model = random.sample(data_from_model, min(len(data_from_model), 600))

        data_to_classify = [new_data[i].text for i in range(len(new_data)) if included[i]==-1]

        mapping = [i for i in range(len(new_data)) if included[i]==-1]

        ans = filer(data_from_model, data_to_classify)

        for i in range(len(ans)):
            if ans[i]==1:
                included[mapping[i]]=1
            
    return included

data = data_loader.load_documents(['earn','acq','money-fx','grain','crude'])

ans= find_novelties([d for d in data['train']], data['test'], ['earn','acq','money-fx','grain','crude'])

for i in range(len(ans)):
    if ans[i]==-1:
        print(i, data['test'][i].category)

<<<<<<< HEAD
br = 0
for i in range(len(ans)):
    if ans[i]==-1 and data['test'][i].category[0] in ['earn','acq','money-fx','grain','crude']:
        br += 1

print("In worng", br)

br = 0
for i in range(len(ans)):
    if ans[i]==1 and data['test'][i].category[0] not in ['earn','acq','money-fx','grain','crude']:
        br += 1

print("Out worng", br)

print("Total", len(ans))


=======
>>>>>>> 4889dcef82d5cc64d443db05fe2227fc45fd8614




