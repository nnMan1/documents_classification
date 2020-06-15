import data_loader
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
from sklearn.neural_network import MLPClassifier

def find_novelties(classifier, distance_matrix):
    return classifier.predict(distance_matrix)
    
def find_outliers(distance_matrix):    
    
    classifier = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=0.8)
    clustering = classifier.fit_predict(distance_matrix)
    return clustering


data = data_loader.load_documents(['earn'])
#data = data_loader.load_documents()

random_sample = random.sample(data['train'], 2000)

train_sample = random_sample[:800]
test_sample = random_sample[1200:1999]

distance = Distance([d.text  for d in train_sample], min_similarity=0.15)

categories = [d.category for d in train_sample]
#distance = Distance([d.text  for d in data['train']], min_similarity=0.13)
distance_matrix = distance.calculate_distance_matrix_optimized()
#print(find_outliers(distance_matrix))
#model = DBSCAN(eps=0.03, min_samples=3, metric='precomputed', algorithm='brute').fit(distance.calculate_distance_matrix_opptimized())

#model = AffinityPropagation(affinity='precomputed', )
#model = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=0.9)dis

#----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------

#model = RandomForestClassifier()
#model.fit(distance_matrix, categories)
#ans = model.predict(distance_matrix)

#-----------------------------------------------------------------------------------------

#model = svm.SVC(decision_function_shape='ovo', probability=True)
#model = svm.SVC(gamma=0.01, C=150., decision_function_shape='ovo', probability=True)
#model.fit(distance_matrix, categories)
#ans = model.predict(distance.calculate_distance_for_optimized([d.text for d in test_sample]))

#------------------------------------------------------------------------------------------

model = svm.OneClassSVM(gamma='auto')
model.fit(distance_matrix)

#ans=model.predict(distance.calculate_distance_for_optimized([d.text for d in test_sample]))



#-----------------------------------------------------------------------------------------

#model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#model.fit(distance_matrix, categories)
#ans = model.predict(distance_matrix)

#-----------------------------------------------------------------------------------------
#model = FeatureAgglomeration(n_clusters=5)
#model.fit(distance_matrix)
#ans = model.transform(distance_matrix)


#distance_matrix = distance.calculate_distance_for_optimized([d.text  for d in data['train']])

#print("Novelties are")
#ans = find_novelties(classifier, distance_matrix)

#br = 0

#for i in range(len(ans)):
    #print(i, ans[i], random_sample[i].category)
    #if ans[i] != random_sample[i].category[0]:
        #br=br+1

#print(br)

#print(metrics.accuracy_score(ans,[d.category for d in test_sample] ))

new_data = distance.calculate_distance_for_optimized([d.text for d in data['test']])

ans = model.predict(new_data)

for i in range(len(ans)):
    #if data['test'][i].category[0] in ['earn','acq','money-fx','grain','crude']:
    if ans[i]==1:
        print(i, ans[i], data['test'][i].category)

br = 0

#print(br)

ans = model.predict_proba(new_data)

for i in range(len(ans)):
    #if data['test'][i].category[0] == 'acq':
        print(max(ans[i]),  data['test'][i].category)

print('Novelties are:')


for i in range(len(ans)):
   if max(ans[i])<0.80:
       print(i, data['test'][i].name, data['test'][i].category)

print()



