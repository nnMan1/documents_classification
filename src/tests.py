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
    classifier = Classifier()
    classifier.fit(distance_matrix)

    number_of_elements_in_classes = classifier.get_number_of_elements_in_classes()
    classification = classifier.get_classification()

    for i in range(len(classification)):
        print(i, classification[i])

    arr = []

    for i in range(len(classification)):
        if(number_of_elements_in_classes[classification[i]]<5):
            arr.append(i)

    return arr

data = data_loader.load_documents()
distance_matrix = Distance(data['test']).distances
#distance_matrix = joblib.load('./sgd_data0.1.pkl').distances

#classifier = AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='precomputed')
#classifier = DBSCAN(eps=0.75, min_samples=find_outliers(distance_matrix.distances)8, metric='precomputed', algorithm='auto', leaf_size=30, n_jobs=-1)

#classes = classifier.fit_predict(distance_matrix.distances)

#find_outliers(distance_matrix.distances)

outliers = find_outliers(distance_matrix)

#for outlier in outliers:
   # print(outlier, data['train_classes'][outlier])


#check_novelty('./sgd_classifier.pkl', './sdg_vectorizer.pkl', './sgd_transformer.pkl', './sgd_traindata.pkl', '../novelties/data/normal_data')


#classifier = LocalOutlierFactor(n_neighbors=2, algorithm='brute', metric='precomputed', novelty=True, contamination=0.05, n_jobs=-1)
#classifier.fit(distance_matrix.distances)
#for i in range(len(ans)):
    #if ans[i]==-1:
        #print(i, classifier.negative_outlier_factor_[i], data['train_classes'][i])
#print(ans)

#ans = classifier.predict(Distance(data['test']).distances)

#for i in range(len(ans)):
    #if ans[i]==-1:
        #print(i, classifier.negative_outlier_factor_[i], data['test_classes'][i])
#print(ans)