import data_loader
from clasifier import Classifier
import preload
from distance import Distance
import joblib
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.naive_bayes import MultinomialNB

#stop_words = set(stopwords.words('english'))

#def check_novelty(model, data):
   
    
def find_outliers(distance_matrix):   

    classification = classifier.get_classification()

    classifier = LocalOutlierFactor(n_neighbors=8, algorithm='brute', metric='precomputed', contamination=0.07, n_jobs=-1)
    ans = classifier.fit_predict(distance_matrix)
   
    return [i for i in range(len(ans)) if ans[i]==-1]

data = data_loader.load_documents()

distance_matrix = Distance([d.text  for d in data['train'][:1000]], min_similarity=0.13).distances
tmp = find_outliers(distance_matrix)
for i in range(len(tmp)):
        print(i, data['train'][i].category)
#distance_matrix = joblib.load('./sgd_data005.pkl').distances

#classifier = AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='precomputed')
#classifier = DBSCAN(eps=0.75, min_samples=find_outliers(distance_matrix.distances)8, metric='precomputed', algorithm='auto', leaf_size=30, n_jobs=-1)

#classes = classifier.fit_predict(distance_matrix.distances)

#find_outliers(distance_matrix.distances)

#tmp = find_outliers(distance_matrix)



#for outlier in outliers:
   # print(outlier, data['train_classes'][outlier])


#check_novelty('./sgd_classifier.pkl', './sdg_vectorizer.pkl', './sgd_transformer.pkl', './sgd_traindata.pkl', '../novelties/data/normal_data')


#print(ans)

#ans = classifier.predict(Distance(data['test']).distances)

#for i in range(len(ans)):
    #if ans[i]==-1:
        #print(i, classifier.negative_outlier_factor_[i], data['test_classes'][i])
#print(ans)