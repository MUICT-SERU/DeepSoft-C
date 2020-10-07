import gensim
import numpy
from gensim import corpora
import sys
sys.path.append('/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/NCE/')
sys.path.append('/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/Classifier/')
sys.path.append('../NCE')

import prepare_data
import sys
import scipy.stats
args = prepare_data.arg_passing_any(sys.argv)

import gzip
import cPickle

################################# LOAD DATA #######################################################


try:
    dataset = '../Dataset/Data/' + args['-data'] + '.pkl.gz'
    saving = args['-saving']
except:
    doc_rep = 'lda'
    dataset = '../Dataset/Data/' + 'HBASE' + '.pkl.gz'
    saving = 'test_lda-kl'
    dataset = '/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/DataSet/Data/BAM.pkl.gz'

# parameters
num_topics = 20
passes = 20

train_t, train_d, train_y, valid_t, valid_d, valid_y, test_t, test_d, test_y = prepare_data.load(dataset)

print 'Building LDA model...'


def listtostring(word_id):
    str_id = []
    for i in range(len(word_id)):
        str_id.append(map(str, word_id[i]))
    return str_id


train_t = listtostring(train_t)
train_d = listtostring(train_d)
valid_t = listtostring(valid_t)
valid_d = listtostring(valid_d)
test_t = listtostring(test_t)
test_d = listtostring(test_d)

str_compile = train_t + train_d + valid_t + valid_d + test_t + test_d

dictionary = corpora.Dictionary(str_compile)
print dictionary
corpus = [dictionary.doc2bow(text) for text in str_compile]

# numtopic needs to be config

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, update_every=0,
                                           passes=passes)


# bow = ldamodel.id2word.doc2bow(train_d[0]+train_t[0])
# doc_topics, word_topics, phi_values = ldamodel.get_document_topics(bow, per_word_topics=True)

def doctopics_to_features(doc_topics, num_topics):
    features = numpy.zeros(num_topics)
    for i in range(len(doc_topics)):
        features[doc_topics[i][0]] = doc_topics[i][1]
    return features


def zero_to_small(features):
    features[features == 0] = 0.00001
    return features


def one_to_zero(features):
    features[features == 100] = 0
    return features

def ldafeature(ldamodel, train_t, train_d, num_topics):
    ldafeatures = numpy.zeros(shape=(len(train_t), 20))
    for i in range(len(train_t)):
        bow = ldamodel.id2word.doc2bow(train_d[i] + train_t[i])
        doc_topics, word_topics, phi_values = ldamodel.get_document_topics(bow, per_word_topics=True)
        features = doctopics_to_features(doc_topics, num_topics)
        features = zero_to_small(features)
        ldafeatures[i] = features #numpy.vstack((ldafeatures, features))  # ldafeatures.append()
    return ldafeatures


train_x = ldafeature(ldamodel, train_t, train_d, num_topics)
valid_x = ldafeature(ldamodel, valid_t, valid_d, num_topics)
test_x = ldafeature(ldamodel, test_t, test_d, num_topics)


# scipy.stats.entropy(train_x[0],train_x[7])
###################################### BUILD MODEL##################################################
print 'Building KL model...'
#test_y is actual
def KL_BR(train_x, train_y, test_x):
    predict = numpy.zeros(shape=(len(test_x),numpy.shape(train_y)[1]))
    # for i in range(numpy.shape(predict)[1]): # binary components
    #     print 'component: %d ' %(i)
    for j in range(numpy.shape(predict)[0]): # loop to all test issues
        KL = numpy.zeros(len(train_x))
        for k in range(len(KL)): # compare test to all train to find nearest issue in train
            KL[k] = scipy.stats.entropy(train_x[k], test_x[j])
        nearest_index = numpy.argmin(KL)
        for i in range(numpy.shape(predict)[1]):
            predict[j][i] = train_y[nearest_index][i]
            # print 'nearest_index: %d , component: %d' % (nearest_index, train_y[nearest_index][i])
    return predict

def KL_centroid(train_x, train_y, test_x):
    predict = numpy.zeros(shape=(len(test_x), numpy.shape(train_y)[1]))
    #find component's centroid
    train_x = numpy.array(train_x)
    train_y = numpy.array(train_y)
    test_x = numpy.array(test_x)

    centroid = numpy.zeros(shape=(numpy.shape(train_y)[1], num_topics))
    for i in range(len(centroid)):
        centroid[i,:] = numpy.mean(train_x[train_y[:,i]>0], axis=0)

    centroid = zero_to_small(numpy.nan_to_num(centroid))

    #use KL to compare issues with each centroid
    for i in range(len(predict)):
        for j in range(len(predict[0])):
            predict[i,j] = scipy.stats.entropy(centroid[j], test_x[i])

    return one_to_zero(100-predict)  # change to negative, more value more chance

test_y_predict = KL_centroid(train_x, train_y, test_x)

numpy.savetxt('log/output/' + saving + "_actual.csv", test_y, delimiter=",")
numpy.savetxt('log/output/' + saving + "_estimate.csv", test_y_predict, delimiter=",")

