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
        self.shingles = [sorted(sparse.nonzero()[1]) for sparse in tmp]

        universal_set = vectorizer.get_feature_names()
        self.universal_set = { universal_set[i] : i  for i in range(0, len(universal_set) ) }

        self.__calculate_token_frequency()

        for i in range(len(self.shingles)):
            self.shingles[i] = sorted(self.shingles[i], key= self.token_frequency.__getitem__)
        
        self.__hash_data()

    def __calculate_token_frequency(self):

        token_frequency= [0]*len(self.universal_set)
        
        for shingle in self.shingles:
            for token in shingle:
                token_frequency[token]+=1
        
        self.token_frequency = token_frequency
            
    def __hash_data(self):
        hash_bucets = {}

        for k in range(len(self.shingles)):
            shingle = self.shingles[k]
            l = int((1-self.min_similarity)*len(shingle)+1)
            for i in range(min(len(shingle),l)):
                if not shingle[i] in hash_bucets.keys():
                    hash_bucets[shingle[i]]={}
                
                if not i in hash_bucets[shingle[i]].keys():
                    hash_bucets[shingle[i]][i]={}

                if not (len(shingle)-i-1) in hash_bucets[shingle[i]][i].keys():
                    hash_bucets[shingle[i]][i][len(shingle)-i-1] = set()
                
                hash_bucets[shingle[i]][i][len(shingle)-i-1].add(k)

        self.hash_bucets = hash_bucets

    def __find_possible_documents(self, a, i, p):
        possible = set()
        l=p+i+1

        if not a in self.hash_bucets.keys():
            return possible
        
        j_values = [j for j in self.hash_bucets[a].keys() if j<=(p+1)/self.min_similarity - l]

        for j in j_values:
            q_values = [q for q in self.hash_bucets[a][j].keys() if (min(q,p)+1)/(max(p,q)+max(i,j)+1) >= self.min_similarity]
            for q in q_values:
                possible = possible.union(self.hash_bucets[a][j][q])
        
        return possible

    def __jacard_distance(self, a, b):
        intersection = set(a).intersection(set(b))
        ans = len(intersection) / (len(a) + len(b) - len(intersection))
        return ans

    def calculate_distance_matrix_optimized(self):

        for i in range(len(self.shingles)):
            a = self.shingles[i]
            possible = set()
            for k in range(min(len(a), int((1-self.min_similarity)*len(a)+1))):
                possible = possible.union(self.__find_possible_documents(a[k], k, len(a)-k-1))
            
            for j in possible:
                b = self.shingles[j]
                similarity = self.__jacard_distance(a,b)
                if similarity >= self.min_similarity:
                    self.distances[i,j]=0
        
        return self.distances

    def calculate_distance_for_optimized(self, data):
        vectorizer = CountVectorizer(decode_error = 'ignore')
        tmp = vectorizer.fit_transform(data)

        universal_set = vectorizer.get_feature_names()
        cnt = len(self.universal_set)

        for key in universal_set:
            if not key in self.universal_set.keys():
                self.universal_set[key] = cnt
                cnt = cnt+1
                self.token_frequency.append(1)
            
        shingles = []

        for shingle in tmp:
            shingle = shingle.nonzero()[1]
            shingle = [self.universal_set[universal_set[i]] for i in shingle]
            shingles.append(sorted(shingle, key= self.token_frequency.__getitem__))
        
        ans = np.ones((len(shingles), len(self.shingles)))

        for i in range(len(shingles)):
            a = shingles[i]
            possible = set()
            for k in range(min(len(a), int((1-self.min_similarity)*len(a)+1))):
                possible = possible.union(self.__find_possible_documents(a[k], k, len(a)-k-1))
            
            for j in possible:
                b = self.shingles[j]
                similarity = self.__jacard_distance(a,b)
                if similarity >= self.min_similarity:
                    ans[i, j] = 0        
        
        return ans
      
    def calculate_distance_matrix(self):
        for i in range(len(self.shingles)):
            self.distances[i,i] = 0
            for j in range(i,len(self.shingles)):
                similaruity = self.__jacard_distance(self.shingles[i],self.shingles[j])               
                if(similaruity >= self.min_similarity):
                    self.distances[i, j] = 0
                    self.distances[j, i] = 0
        
        return self.distances

    def calculate_distance_for(self, data):
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


    distance = Distance(['marko kraljevic', 'musa kesedzija', 'marko kraljevic i musa kesedzija', 'marko'])
    print(distance.distances)

    print(distance.calculate_distance_for_optimized(['marko markovic']))
    print(distance.calculate_distance_for(['marko markovic']))



    #data = data_loader.load_documents()['train']
    #distance = Distance([d.text  for d in data])
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

  