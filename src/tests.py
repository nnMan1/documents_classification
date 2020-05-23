import data_loader
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

#distance = Distance([d.text  for d in data['train'][:2000]], min_similarity=0.13)
distance = Distance([d.text  for d in data['train']], min_similarity=0.13)

distance_matrix = distance.calculate_distance_matrix_opptimized()
ans, classifier = find_outliers(distance_matrix)

print("Outliers are")

for i in range(len(ans)):
   #if ans[i]==-1 and (data['train'][i].category in [['earn'], ['acq'], ['money-fx'], ['trade']]):
   if ans[i]==-1:
        print(i, data['test'][i].name, data['train'][i].category)

distance_matrix = distance.calculate_distance_for_optimized([d.text  for d in data['test'][:500]])

print("Novelties are")
ans = find_novelties(classifier, distance_matrix)

for i in range(len(ans)):
   #if ans[i]==-1 and (data['train'][i].category in [['earn'], ['acq'], ['money-fx'], ['trade']]):
   if ans[i]==-1:
        print(i, data['test'][i].name, data['test'][i].category)
