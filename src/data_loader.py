import nltk
from nltk.corpus import reuters
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from data_object import DataObject

n_classes = 90
labels = reuters.categories()

def load_documents(categories=[]):

    if categories == []:
        categories = reuters.categories()

    if categories == []:
        categories = labels

    documents = reuters.fileids()
    test = [d for d in documents if d.startswith('test/') and len(reuters.categories(d)) == 1 ]
    train = [d for d in documents if d.startswith('training/') and len(reuters.categories(d)) == 1 and reuters.categories(d)[0] in categories]

    docs = {}
    docs['train'] = [DataObject(reuters.raw(doc_id), doc_id ,reuters.categories(doc_id)) for doc_id in train]
    docs['test'] = [DataObject(reuters.raw(doc_id), doc_id, reuters.categories(doc_id)) for doc_id in test]
    
    return docs

if __name__ == '__main__':
    data = load_documents(['earn', 'acq'])

    print(data['train'][0].text)

    #print(tokenize(data['train'])['shingling'][0])
   #print(data['train'][4])
   # print("len(data['x_train'])={}".format(len(data['train'])))
    #print("len(data['x_test'])={}".format(len(data['test'])))