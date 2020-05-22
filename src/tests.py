import data_loader
from clasifier import Classifier
import preload
from distance import Distance
import joblib
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.naive_bayes import MultinomialNB

def find_novelties(classifier, distance_matrix):
    return classifier.predict(distance_matrix)
    
def find_outliers(distance_matrix):    
    classifier = LocalOutlierFactor(n_neighbors=15, algorithm='brute', metric='precomputed', novelty=True, contamination=0.19, n_jobs=-1)
    classifier.fit(distance_matrix)
    return classifier.predict(distance_matrix), classifier

data = data_loader.load_documents()
distance_matrix = Distance([d.text  for d in data['train'][:2000]], min_similarity=0.13)
ans, classifier = find_outliers(distance_matrix.distances)

#distance_matrix = Distance([d.text  for d in data['train'][:1000]].append(data['test'][11]), min_similarity=0.13).distances

print("Outliers are")

for i in range(len(ans)):
   #if ans[i]==-1 and (data['train'][i].category in [['earn'], ['acq'], ['money-fx'], ['trade']]):
   if ans[i]==-1:
        print(i, data['train'][i].category)

distance_matrix = distance_matrix.calcilate_distance_for([d.text  for d in data['test'][:500]])

print("Novelties are")
ans = find_novelties(classifier, distance_matrix)

for i in range(len(ans)):
   #if ans[i]==-1 and (data['train'][i].category in [['earn'], ['acq'], ['money-fx'], ['trade']]):
   if ans[i]==-1:
        print(i, data['test'][i].category)
