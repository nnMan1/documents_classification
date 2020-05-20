from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from distance import Distance
import joblib
import data_loader

distance = joblib.load('./sgd_data.pkl')
data = data_loader.load_documents()

data_similarity = distance.distances

model = AgglomerativeClustering(affinity='precomputed', n_clusters=None, linkage='average', distance_threshold=0.9).fit(data_similarity)
#model = DBSCAN(metric='precomputed', min_samples=3, eps=0.73).fit(data_similarity)


def print_classes(model):
    for i in range(len(model.labels_)):
            print("{} {}".format(i, model.labels_[i]))

def print_statistic(model):
    for j in range(-1,200):
        br=0
        for i in range(len(model.labels_)):
            if(model.labels_[i]==j):
                br=br+1
        print(j, br)

print_classes(model)
#print_statistic(model)

        