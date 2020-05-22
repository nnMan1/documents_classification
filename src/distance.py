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
        self.distances = np.ones((len(data), len(data)))
        vectorizer = CountVectorizer(decode_error = 'ignore')
        tmp = vectorizer.fit_transform(data)
        self.universal_set = vectorizer.get_feature_names()
        self.shingles = [set(sparse.nonzero()[1]) for sparse in tmp]
        self.indexes = range(len(self.shingles)) #svi elementi su poredjani kako su usli
        self.__apply_length_based_filtering()
        self.__calculate_distance_matrix(min_similarity)
    
    def __apply_length_based_filtering(self):
        
        lengths = []

        for i in range(len(self.shingles)):
            lengths.append(len(self.shingles[i]))
        
        self.indexes = sorted(self.indexes, key= lengths.__getitem__)
        
    def __jacard_distance(self, i, j):
        a = self.shingles[i]
        b = self.shingles[j]
        intersection = set(a).intersection(set(b))
        ans = len(intersection) / (len(a) + len(b) - len(intersection))
        return ans

    def __check_prefixes(self, i, j, min_similarity):
        a = self.shingles[i]
        b = self.shingles[j]

        l_a = len(a)
        l_b = len(b)

        i=0
        j=0
        similar = 0
        distinct = 0

        while(i<l_a and j<l_b and (similar + l_a - i)/(l_b+distinct) >= min_similarity):
            while(j<l_b and b[j]<a[i]):
                j+=1

            if(j==l_b):
                break
            
            if(b[j]==a[i]):
                similar+=1
            else:
                distinct+=1

            i+=1

        if(j==l_b or i == l_a):
            return similar / (l_a + l_b - similar)

        return (similar + l_a - i)/(l_b+distinct)

    def __calculate_distance_matrix(self, min_similarity = 0.05):
         for i in range(len(self.shingles)):
            self.distances[i,i] = 0
            for j in range(i,len(self.shingles)):
                #similaruity = self.__jacard_distance(self.indexes[i], self.indexes[j])
                similaruity = self.__jacard_distance(i,j)               
                if(similaruity >= min_similarity):
                    self.distances[i, j] = 0
                    self.distances[j, i] = 0


if __name__=='__main__':
    data = data_loader.load_documents()['train']
    distance = Distance([d.text  for d in data[:500]])
    #joblib.dump(distance, './sgd_data005.pkl', compress=9)
    #distance = joblib.load('./sgd_data.pkl')
    #joblib.dump(distance, './sgd_data0.1.pkl', compress=9)

    #print(len(distance.indexes))
    #id = distance.indexes[7300]
    #print(distance.shingles[id])
    #print(distance.distances)
    #print(data['train'][id])
    for i in range(len(distance.distances)):
        for j in range(len(distance.distances[i])):
            if distance.distances[i,j]!=1:
                print(i, j)

  