from sklearn.metrics import jaccard_score
from collections import defaultdict 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import data_loader
import scipy.sparse as sp

class Tokenizer:
    
    def __init__(self):
        self._stopwords_set = set(stopwords.words())
        return

    def _find_tokens(self, documents):

        dict = defaultdict()
        cnt = 0

        for text in documents:
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = text.lower()
            words = text.split()
            words.append('')

            for i in range(len(words)-2):
                if words[i] in self._stopwords_set and (not " ".join([words[i+1], words[i+2]]) in dict.keys()):
                    dict[" ".join([words[i+1], words[i+2]])] = cnt
                    cnt += 1

        self.tokens = dict
                    
    def fit_transform(self, documents):

        self._find_tokens(documents)

        ans = sp.lil_matrix((len(documents), len(self.tokens.keys())))

        for i in range(len(documents)):
            text = documents[i]
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = text.lower()
            words = text.split()
            words.append('')

            for j in range(len(words)-2):
                if words[j] in self._stopwords_set:
                    ans[i,self.tokens[" ".join([words[j+1], words[j+2]])]] += 1
            
        return ans

    def get_feature_names(self):
        tmp = [None]*len(self.tokens.keys())

        for key in self.tokens:
            tmp[self.tokens[key]] = key 

        return tmp
        
    def hash_documents(self, documents, k=2):
        arr = [None] * len(documents)
        cnt = 0

        for i in range(len(documents)):
            text = documents[i]
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = text.lower()
            words = text.split()
            words.append(['']*k)

            positons = set()

            for j in range(len(words)-k):
                if words[j] in self._stopwords_set:
                    positons.add(j+1)
                    positons.add(j+2)

            positons = [words[v] for v in positons];
            arr[i] = " ".join(positons)

        
        return arr


if __name__ == '__main__':
    data = data_loader.load_documents()
    
    tokenizer = Tokenizer()
    tmp = tokenizer.hash_documents([d.text for d in data['train']])

    print(tmp[0])
    print(data['train'][0].text)

    #tmp = tokenizer.fit_transform([d.text for d in data['train']])
    #universal_set = tokenizer.feature_names

    #tmp = tmp[0]

    #for i in tmp[1].nonzero()[1]:
        #print(universal_set[i])


    #vectorizer = CountVectorizer(decode_error = 'ignore', ngram_range=(2, 2), analyzer='word')
    #vectorizer = CountVectorizer(decode_error = 'ignore', ngram_range=(9, 9), analyzer='char')
    #vectorizer = CountVectorizer(decode_error = 'ignore', ngram_range=(5, 5), analyzer='char_wb')
    
    #print(tokenizer.get_tokenized_representation(1))
    #tmp = tokenizer.get_tokenized_representation(66)

    #for i in tmp:
        #print(tokenizer.get_token_with_index(i))