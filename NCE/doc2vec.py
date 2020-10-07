import sys

sys.path.append('/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/NCE/')
sys.path.append('/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/NCE/doc2vec_feature')
sys.path.append('/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/Classifier/')

import gzip
import cPickle
import load_data
from collections import namedtuple
from gensim.models import Doc2Vec

from NCE import *

arg = load_data.arg_passing(sys.argv)

try:
    dataset = arg['-data']
    dim = int(arg['-dim'])
    saving = arg['-saving']

    dataset_path = '../Dataset/Data/' + dataset + '.pkl.gz'

except:
    saving = 'doc2vec_test'
    dataset_path = '../Dataset/Data/' + 'HBASE' + '.pkl.gz'
    dataset_path = '/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/Dataset/Data/' + 'HBASE' + '.pkl.gz'
    dim = 100

train_t, train_d, train_y, valid_t, valid_d, valid_y, test_t, test_d, test_y = load_data.load_all(dataset_path)

train_x = []
valid_x = []
test_x = []

analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')


def trainDoc2vec(documents):
    taggeddocuments = []
    for i, text in enumerate(documents):
        # words = str(text)
        words = ",".join(str(c) for c in text)
        words = words.split(',')
        # print words
        tags = [i]
        taggeddocuments.append(analyzedDocument(words, tags))
    model = Doc2Vec(taggeddocuments, size=dim, window=10, min_count=1, workers=4)
    return model


def Doc2vecModeltoFeat(model):
    feat = []
    for i in range(len(model.docvecs)):
        feat.append(model.docvecs[i])
    return feat


trainVec = trainDoc2vec(train_t)
validVec = trainDoc2vec(valid_t)
testVec = trainDoc2vec(test_t)

trainFeat = Doc2vecModeltoFeat(trainVec)
validFeat = Doc2vecModeltoFeat(validVec)
testFeat = Doc2vecModeltoFeat(testVec)

train_x = numpy.array(trainFeat)
valid_x = numpy.array(validFeat)
test_x = numpy.array(testFeat)

print len(train_x)

print ('ntrain: %d, n_valid: %d, n_test: %d' % (len(train_y), len(valid_y), len(test_y)))

#########################

f = gzip.open('doc2vec_feature/' + saving + '.pkl.gz', 'wb')
cPickle.dump((train_x, train_y, valid_x, valid_y, test_x, test_y), f)
f.close()
