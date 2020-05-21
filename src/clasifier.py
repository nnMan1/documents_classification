import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import joblib
import data_loader
from distance import Distance


class Classifier:

    def __init__(self, classifier = AgglomerativeClustering(affinity='precomputed', n_clusters=None, linkage='average', distance_threshold=0.90)):
        self.classifier = classifier
    
    def fit(self, data):
        self.data = data
        self.classifier.fit(data)
    
    def get_classification(self):
        return self.classifier.labels_
    
    def get_number_of_elements_in_classes(self):
        arr = []

        j=0

        while(True):
            br=0
            for i in range(len(self.classifier.labels_)):
                if(self.classifier.labels_[i]==j):
                    br=br+1
            
            if(br==0):
                break
            else:
                arr.append(br)
            
            j=j+1

        return arr

#print_statistic(model)

if __name__ == "__main__":
    data = data_loader.load_documents()
    distance_matrix = joblib.load('./sgd_data.pkl')

    classifier = Classifier()
    classifier.fit(distance_matrix)

    
        