#
# dataset = '/home/mc650/Dropbox/PythonWorkspace/PredictingComponent-multilabel/Dataset/Data/HBASE.pkl.gz'
# sys.path.append('/home/mc650/Dropbox/PythonWorkspace/PredictingComponent-multilabel/Classifier')
#
#
# sys.path.append('/home/mc650/Dropbox/PythonWorkspace/PredictingComponent-multilabel/Classifier')
# sys.path.append('/home/mc650/Dropbox/PythonWorkspace/PredictingComponent-multilabel/Classifier/')
#
#
# trian_t = numpy.array(train_t)
# trian_d = numpy.array(train_d)
#
# train_x = train_t + train_d
#




import numpy

import sys

sys.path.append('/home/mc650/Dropbox/PythonWorkspace/PredictingComponent-multilabel/Classifier/')
import prepare_data

args = prepare_data.arg_passing_any(sys.argv)
################################# LOAD DATA #######################################################


try:
    dataset = '../Dataset/Data/' + args['-data'] + '.pkl.gz'
    saving = args['-saving']
except:
    dataset = '../Dataset/Data/' + 'HBASE' + '.pkl.gz'
    dataset = '/home/mc650/Dropbox/PythonWorkspace/PredictingComponent-multilabel/Dataset/Data/HBASE.pkl.gz'
    dataset = '/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/DataSet/Data/BAM.pkl.gz'
    saving = 'test_tfidf_cos'

train_t, train_d, train_y, valid_t, valid_d, valid_y, test_t, test_d, test_y = prepare_data.load(dataset)


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


print 'convert word id to text...'

# train_x = numpy.array(train_t) + numpy.array(train_d)
# test_x = numpy.array(test_t) + numpy.array(test_d)

train_x = numpy.array(train_d)
test_x = numpy.array(test_d)

# print 'build collection of documents of each component...'
# numberOfComponent = train_y.shape[1]
# Document = init_list_of_objects(numberOfComponent + len(test_x))
#
# for i in range(numberOfComponent):
#     for j in range(len(train_x)):
#         if train_y[j, i] == 1:
#             Document[i] = Document[i] + train_x[j]
#
# print 'build collection of documents of test set...'
# for i in range(len(test_x)):
#     Document[numberOfComponent + i] = test_x[i]
#
# Document = listtostring(Document)
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
# similarity = cosine_similarity(tfidf_matrix[0:numberOfComponent - 1], tfidf_matrix[numberOfComponent:])
# similarity = similarity.transpose()
#
#
# numpy.savetxt('log/output/' + saving + "_actual.csv", test_y, delimiter=",")
# numpy.savetxt('log/output/' + saving + "_estimate.csv", similarity, delimiter=",")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print 'build collection of documents of each component...'
numberOfComponent = train_y.shape[1]
Document = init_list_of_objects(numberOfComponent + 1)  # add one for issue

for i in range(numberOfComponent):
    for j in range(len(train_x)):
        if train_y[j, i] == 1:
            Document[i] = Document[i] + train_x[j]

predict = numpy.zeros([len(test_x), numberOfComponent])

print 'compare cosine similairty...'
for i in range(len(test_x)):
    Document[len(Document) - 1] = test_x[i]
    corpus = listtostring(Document)

    tfidf_vectorizer = TfidfVectorizer(stop_words=None, lowercase=False)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    for j in range(numberOfComponent):
        similarity = cosine_similarity(tfidf_matrix[j], tfidf_matrix[numberOfComponent])  # the last array is a test issue = [numberOfComponent]
        predict[i, j] = similarity

numpy.savetxt('log/output/' + saving + "_actual.csv", test_y, delimiter=",")
numpy.savetxt('log/output/' + saving + "_estimate.csv", predict, delimiter=",")
