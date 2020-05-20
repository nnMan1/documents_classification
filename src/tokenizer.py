from sklearn.metrics import jaccard_score
import numpy as np
import data_loader
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

class Tokenizer:
    
    def __init__(self, data, vectorizer=CountVectorizer(decode_error = 'ignore')):
        self.data = data
        self.tokenized=self.tokenize(data, vectorizer)

    def tokenize(self, documents, vectorizer):
        vectorized = vectorizer.fit_transform(documents)
        shingles = []

        for doc in vectorized:
            shingles.append(sorted(doc.nonzero()[1]))

        return { "shingling": shingles, "universal_set": vectorizer.get_feature_names()}

    def get_tokenized_representation(self, index):
        #TODO: rijesiti problem sortiranja svaki put
        return self.tokenized['shingling'][index]

    def get_token_with_index(self, index):
        return self.tokenized['universal_set'][index]

if __name__ == '__main__':
    tokenizer = Tokenizer(data_loader.load_documents()['train'])
    
    #vectorizer = CountVectorizer(decode_error = 'ignore', ngram_range=(2, 2), analyzer='word')
    #vectorizer = CountVectorizer(decode_error = 'ignore', ngram_range=(9, 9), analyzer='char')
    #vectorizer = CountVectorizer(decode_error = 'ignore', ngram_range=(5, 5), analyzer='char_wb')
    
    print(tokenizer.get_tokenized_representation(1))
    tmp = tokenizer.get_tokenized_representation(66)

    for i in tmp:
        print(tokenizer.get_token_with_index(i))