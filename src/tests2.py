import data_loader
import preload
from distance import Distance
import joblib
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import OPTICS
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier 
from sklearn.cluster import SpectralClustering
from sklearn.manifold import MDS
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import RandomForestClassifier

def find_novelties(classifier, distance_matrix):
    return classifier.predict(distance_matrix)
    
def find_outliers(distance_matrix):    
    
    classifier = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=0.8)
    clustering = classifier.fit_predict(distance_matrix)
    return clustering


data = data_loader.load_documents(['earn','acq','money-fx','grain','crude'])

random_sample = random.sample(data['train'], 1300)

distance = Distance([d.text  for d in random_sample], min_similarity=0.15)
categories = [d.category for d in random_sample]
#distance = Distance([d.text  for d in data['train']], min_similarity=0.13)
distance_matrix = distance.calculate_distance_matrix_opptimized()
#model = DBSCAN(eps=0.03, min_samples=3, metric='precomputed', algorithm='brute').fit(distance.calculate_distance_matrix_opptimized())

#model = AffinityPropagation(affinity='precomputed', )
#model = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=0.9)
#model = KNeighborsClassifier(n_neighbors=5, metric='precomputed')

#-----------------------------------------------------------------------------------------
model = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=0)
model.fit(distance.calculate_distance_matrix_opptimized(), categories)
#-----------------------------------------------------------------------------------------

#model = svm.SVC(gamma=0.01, C=100.)
#model.fit(distance_matrix, categories)

#-----------------------------------------------------------------------------------------

#model = SGDClassifier(max_iter=1000, loss='log').fit(distance_matrix, categories)

#model = FeatureAgglomeration(n_clusters=5)
#model.fit(distance_matrix)
#ans = model.transform(distance_matrix)

ans = model.predict(distance_matrix)

#distance_matrix = distance.calculate_distance_for_optimized([d.text  for d in data['train']])

#print("Novelties are")
#ans = find_novelties(classifier, distance_matrix)

br = 0

for i in range(len(ans)):
    print(i, ans[i], random_sample[i].category)
    if ans[i] != random_sample[i].category[0]:
        br=br+1

print(metrics.accuracy_score([d.category[0] for d in random_sample], ans))

print(br)

new_data = distance.calculate_distance_for_optimized([d.text for d in data['test'][:500]])

#ans = model.predict(new_data)

br = 0

for i in range(len(ans)):
    #print(i, ans[i], data['test'][i].category)
    if (data['test'][i].category[0] in ['earn','acq','money-fx','grain','crude']) and ans[i] != data['test'][i].category[0]:
        print(i, ans[i], data['test'][i].category)
        br+=1

print(br)

