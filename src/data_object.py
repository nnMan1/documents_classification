import re
import string
import nltk
import numpy
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
nltk.download('punkt')

class DataObject:

    def __init__(self, text, name, category=None):
        self.text = text
        self.category = category
        self.name = name

    def prepare_text(self):
        tokens = word_tokenize(self.text)
        #words = re.split(r'\W+', self.text)
        print(tokens[:100])

    def get_sentences_for_hashing(self, positions):
        sentences = sent_tokenize(self.text)

        hash = [sentences[i] for i in positions if i < len(sentences)]
        
        return " ".join(hash)


#if __name__=="__main__":
