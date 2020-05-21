import nltk
from nltk.corpus import reuters
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

n_classes = 90
labels = reuters.categories()

def load_documents():
    """
    Load the Reuters dataset.
    Returns
    -------
    data : dict
        with keys 'train', 'test'
    """
    
    documents = reuters.fileids()
    test = [d for d in documents if d.startswith('test/') and len(reuters.categories(d)) == 1 ]
    train = [d for d in documents if d.startswith('training/') and len(reuters.categories(d)) == 1]

    #for i in range(len(test)):
       # print("{} {}".format(i, reuters.categories(test[i])))

    docs = {}
    docs['train'] = [reuters.raw(doc_id) for doc_id in train]
    docs['test'] = [reuters.raw(doc_id) for doc_id in test]
    docs['train_names'] = train
    docs['test_names'] = test
    docs['train_classes'] = [reuters.categories(d) for d in documents if d.startswith('training/') and len(reuters.categories(d)) == 1]
    docs['test_classes'] = [reuters.categories(d) for d in documents if d.startswith('test/') and len(reuters.categories(d)) == 1]
    
    return docs

if __name__ == '__main__':
    print(reuters.raw("training/10112"))
    data = load_documents()
    #print(tokenize(data['train'])['shingling'][0])
   #print(data['train'][4])
    print("len(data['x_train'])={}".format(len(data['train'])))
    print("len(data['x_test'])={}".format(len(data['test'])))