import sys

sys.path.append('/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/NCE/')
sys.path.append('/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/Classifier/')

from keras.models import model_from_json
import gzip
import numpy
import cPickle
import load_data

from NCE import *

arg = load_data.arg_passing(sys.argv)

try:
    dataset = '../Dataset/Data/' + arg['-data'] + '.pkl.gz'
    saving = 'distance_feature/' + arg['-saving'] + '.pkl.gz'
except:
    emb_dim = 100
    data_pretrain = 'lstm2v_apache_dim' + str(emb_dim)
    dataset = '/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/Dataset/Data/CB.pkl.gz'
    saving = '/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/NCE/distance_feature/test.pkl.gz'
    max_len = 100
    vocab_size = 5000

    model_path = '/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/NCE/models/' + data_pretrain + '.json'
    param_path = '/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/NCE/bestmodels/' + data_pretrain + '.hdf5'


#### cosine distance_feature ####

train_t, train_d, train_y, valid_t, valid_d, valid_y, test_t, test_d, test_y = load_data.load_all(dataset)


# def listtostring(word_id):
#     str_id = []
#     for i in range(len(word_id)):
#         str_id.append(' '.join(map(str, word_id[i])))
#     return str_id
#
#
# def init_list_of_objects(size):
#     list_of_objects = list()
#     for i in range(0, size):
#         list_of_objects.append(list())  # different object reference each time
#     return list_of_objects
#
# def fillempty(text):
#     for i in range(len(text)):
#         if text[i] == '':
#             text[i] = text[i-1]
#     return text
#
#
# train_x = numpy.array(train_t) + numpy.array(train_d)
# valid_x = numpy.array(valid_t) + numpy.array(valid_d)
# test_x = numpy.array(test_t) + numpy.array(test_d)
#
# numberOfComponent = train_y.shape[1]
#
# Document = init_list_of_objects(numberOfComponent + len(train_x) + len(valid_x) + len(test_x))
#
# for i in range(numberOfComponent):
#     for j in range(len(train_x)):
#         if train_y[j, i] == 1:
#             Document[i] = Document[i] + train_x[j]
#
# for i in range(len(train_x)):
#     Document[numberOfComponent + i] = train_x[i]
#
# for i in range(len(valid_x)):
#     Document[numberOfComponent + len(train_x) + i] = valid_x[i]
#
# for i in range(len(test_x)):
#     Document[numberOfComponent + len(train_x) + len(valid_x) + i] = test_x[i]
#
# Document = listtostring(Document)
# Document = fillempty(Document)
#
# print 'build tfidf matrix...'
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# tfidf_vectorizer = TfidfVectorizer(stop_words=None, lowercase=False)
# tfidf_matrix = tfidf_vectorizer.fit_transform(Document)
#
# print 'compare cosine similairty...'
# from sklearn.metrics.pairwise import cosine_similarity
#
# train_cos = cosine_similarity(tfidf_matrix[:numberOfComponent - 1], tfidf_matrix[numberOfComponent:len(train_x)-1])
# train_cos = train_cos.transpose()
#
# similarity = cosine_similarity(tfidf_matrix[0:numberOfComponent - 1], tfidf_matrix[numberOfComponent:])
# similarity = similarity.transpose()

train_x = numpy.array(train_t) + numpy.array(train_d)
valid_x = numpy.array(valid_t) + numpy.array(valid_d)
test_x = numpy.array(test_t) + numpy.array(test_d)


def listtostring(word_id):
    str_id = []
    for i in range(len(word_id)):
        str_id.append(' '.join(map(str, word_id[i])))
    return str_id


def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0, size):
        list_of_objects.append(list())  # different object reference each time
    return list_of_objects

numberOfComponent = train_y.shape[1]
Document = init_list_of_objects(numberOfComponent + 1)  # add one for issue

#build corpus for each component
for i in range(numberOfComponent):
    for j in range(len(train_x)):
        if train_y[j, i] == 1:
            Document[i] = Document[i] + train_x[j]

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def extractcosinefeature(Document, text):

    cosine_feature = numpy.zeros([len(text), numberOfComponent])
    corpus = listtostring(Document)

    tfidf_vectorizer = TfidfVectorizer(stop_words=None, lowercase=False)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    for i in range(len(text)):
        x = list()
        x.append(list())
        x[0] = text[i]
        x = listtostring(x)
        issue_matrix = tfidf_vectorizer.transform(x)

        for j in range(numberOfComponent):
            similarity = cosine_similarity(tfidf_matrix[j], issue_matrix)  # the last array is a test issue = [numberOfComponent]
            cosine_feature[i, j] = similarity
    return cosine_feature

train_cos = extractcosinefeature(Document, train_x)
valid_cos = extractcosinefeature(Document, valid_x)
test_cos = extractcosinefeature(Document, test_x)

#########################
f = gzip.open(saving, 'wb')
cPickle.dump((train_cos, train_y, valid_cos, valid_y, test_cos, test_y), f)
f.close()
