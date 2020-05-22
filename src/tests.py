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
    classifier = LocalOutlierFactor(n_neighbors=8, algorithm='brute', metric='precomputed', contamination=0.07, n_jobs=-1)
    ans = classifier.fit_predict(distance_matrix)
    return ans

data = data_loader.load_documents()
distance_matrix = Distance([d.text  for d in data['train'][:1000]], min_similarity=0.13).distances
ans = find_outliers(distance_matrix)

for i in range(len(ans)):
   if ans[i]==-1:
        print(i, data['train'][i].category)
#print(ans)

#ans = classifier.predict(Distance(data['test']).distances)

#for i in range(len(ans)):
    #if ans[i]==-1:
        #print(i, classifier.negative_outlier_factor_[i], data['test_classes'][i])
#print(ans)