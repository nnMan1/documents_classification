import numpy as np
from  tokenizer import Tokenizer 
import data_loader
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from scipy import sparse
import joblib

class Distance:

    length_based_filtering_applied = False

    def __init__(self, data):
        self.distances = np.ones((len(data), len(data)))
        tokenizer = Tokenizer(data, CountVectorizer(decode_error = 'ignore',stop_words = stopwords.words("english").append(['0','1','2','3','4','5','6','7','8','9'])))
        self.shingles = tokenizer.tokenized['shingling']
        self.universal_set = tokenizer.tokenized['universal_set']
        self.indexes = range(len(self.shingles)) #svi elementi su poredjani kako su usli
        self.__apply_length_based_filtering()
        self.__calculate_distance_matrix(min_similarity=0.08)
    
    def __apply_length_based_filtering(self):
        
        lengths = []

        for i in range(len(self.shingles)):
            lengths.append(len(self.shingles[i]))
        
        self.indexes = sorted(self.indexes, key= lengths.__getitem__)
        
    def __jacard_distance(self, i, j):
        a = self.shingles[i]
        b = self.shingles[j]
        intersectionSize = len(set(a).intersection(set(b)))
        unionSize = len(set(a).union(set(b)))
        ans = intersectionSize / unionSize
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

    def __calculate_distance_matrix(self, min_similarity = 0.2):
         for i in range(len(self.indexes)):
            self.distances[i,i] = 0
            j=i+1
            while(j<len(self.indexes) and len(self.shingles[self.indexes[i]])/len(self.shingles[self.indexes[j]])>=min_similarity):
                #similaruity = self.__jacard_distance(self.indexes[i], self.indexes[j])
                similaruity = self.__check_prefixes(self.indexes[i], self.indexes[j], min_similarity)
               
                if(similaruity >= min_similarity):
                    self.distances[self.indexes[i], self.indexes[j]] = 1 - similaruity
                    self.distances[self.indexes[j], self.indexes[i]] = 1 - similaruity
                    #potrebno nam je rastojanje a ne slicnost
                j+=1


if __name__=='__main__':
    data = data_loader.load_documents()
    distance = Distance(data['train'])
    #joblib.dump(distance, './sgd_data.pkl', compress=9)
    distance = joblib.load('./sgd_data.pkl')
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

  