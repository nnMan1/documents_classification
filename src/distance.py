import numpy as np
from  tokenizer import Tokenizer 
import data_loader
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from scipy import sparse
import joblib
from data_object import DataObject

class Distance:

    length_based_filtering_applied = False

    def __init__(self, data, min_similarity = 0.2):

        self.min_similarity = min_similarity

        self.distances = np.ones((len(data), len(data)))
        vectorizer = CountVectorizer(decode_error = 'ignore')

        tmp = vectorizer.fit_transform(data)
        self.shingles = [set(sparse.nonzero()[1]) for sparse in tmp]

        universal_set = vectorizer.get_feature_names()
        self.universal_set = { universal_set[i] : i  for i in range(0, len(universal_set) ) }
        
        self.indexes = range(len(self.shingles)) #svi elementi su poredjani kako su usli
        self.__apply_length_based_filtering()
        self.__calculate_distance_matrix()
    
    def __apply_length_based_filtering(self):
        
        lengths = []

        for i in range(len(self.shingles)):
            lengths.append(len(self.shingles[i]))
        
        self.indexes = sorted(self.indexes, key= lengths.__getitem__)
        
    def __jacard_distance(self, a, b):
        intersection = set(a).intersection(set(b))
        ans = len(intersection) / (len(a) + len(b) - len(intersection))
        return ans

    def __calculate_distance_matrix(self):
         for i in range(len(self.shingles)):
            self.distances[i,i] = 0
            for j in range(i,len(self.shingles)):
                #similaruity = self.__jacard_distance(self.indexes[i], self.indexes[j])
                similaruity = self.__jacard_distance(self.shingles[i],self.shingles[j])               
                if(similaruity >= self.min_similarity):
                    self.distances[i, j] = 0
                    self.distances[j, i] = 0

    def calcilate_distance_for(self, data):
        vectorizer = CountVectorizer(decode_error = 'ignore')
        tmp = vectorizer.fit_transform(data)

        universal_set = vectorizer.get_feature_names()
        cnt = len(self.universal_set)

        for key in universal_set:
            if not key in self.universal_set.keys():
                self.universal_set[key] = cnt
                cnt = cnt+1
            
        shingles = []

        for shingle in tmp:
            shingle = shingle.nonzero()[1]
            shingles.append(set([self.universal_set[universal_set[i]] for i in shingle]))
        
        ans = np.ones((len(shingles), len(self.shingles)))

        for i in range(len(self.shingles)):
            for j in range(len(shingles)):
                similaruity = self.__jacard_distance(self.shingles[i],shingles[j])  
                if(similaruity >= self.min_similarity):
                    ans[j, i] = 0
        
        return ans
                
                


if __name__=='__main__':


    distance = Distance(['marko kraljevic', 'musa kesedzija', 'marko kraljevic i musa kesedzija'])
    print(distance.distances)

    print(distance.calcilate_distance_for(['marko markovic']))



    #data = data_loader.load_documents()['train']
    #distance = Distance([d.text  for d in data[:500]])
    #joblib.dump(distance, './sgd_data005.pkl', compress=9)
    #distance = joblib.load('./sgd_data.pkl')
    #joblib.dump(distance, './sgd_data0.1.pkl', compress=9)

    #print(len(distance.indexes))
    #id = distance.indexes[7300]
    #print(distance.shingles[id])
    #print(distance.distances)
    #print(data['train'][id])
    #for i in range(len(distance.distances)):
        #for j in range(len(distance.distances[i])):
            #if distance.distances[i,j]!=1:
                #print(i, j)

  