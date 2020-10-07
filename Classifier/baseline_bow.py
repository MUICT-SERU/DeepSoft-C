import numpy
import prepare_data
import sys

sys.path.append('../NCE')
#import load_data

args = prepare_data.arg_passing(sys.argv)
numpy.random.seed(args['-seed'])

from keras.optimizers import *
from keras.objectives import *
from create_model import *

from collections import namedtuple
import gzip

import cPickle
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence

################################# LOAD DATA #######################################################


try:
    doc_rep = args['-mode']
    dataset = '../Dataset/Data/' + args['-data'] + '.pkl.gz'
    nnet_model = args['-nnetM']
    saving = args['-saving']
    hid_dim = args['-dim']
    vocab_size = args['-vocab']
except:
    doc_rep = 'doc2vec'
    dataset = '../Dataset/Data/' + 'HBASE' + '.pkl.gz'
    saving = args['-saving']
    nnet_model = 'dense'
    vocab_size = 100

# if 'hid' in args['-reg']: dropout_hid = True
# else:
dropout_hid = False

train_t, train_d, train_y, valid_t, valid_d, valid_y, test_t, test_d, test_y = prepare_data.load(dataset)
train_x = []
valid_x = []
test_x = []

if doc_rep == 'bow':
    train_x = prepare_data.prepare_BoW(train_t, train_d, vocab_size)
    valid_x = prepare_data.prepare_BoW(valid_t, valid_d, vocab_size)
    test_x = prepare_data.prepare_BoW(test_t, test_d, vocab_size)

    print len(train_x)
elif doc_rep == 'doc2vec':
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
        model = Doc2Vec(taggeddocuments, size=vocab_size, window=10, min_count=1, workers=4)
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

print train_y.dtype
n_classes = train_y.shape[-1]
loss = multi_label_loss

print train_x.shape, train_y.shape
###################################### BUILD MODEL##################################################
print 'Building model...'

# n_classes, vocab_size, inp_len, emb_dim,
# seq_model='lstm', nnet_model='highway', pool_mode='mean',
# dropout_inp=False, dropout_hid=True

model = create_multi_label_BoW(n_classes=n_classes, vocab_size=vocab_size, hid_dim=10, nnet_model=nnet_model,
                               dropout=dropout_hid)

# model = create_fixed(n_classes=n_classes, inp_len=train_t.shape[1], emb_dim=hid_dim,
#                      seq_model=seq_model, nnet_model=nnet_model, pool_mode=pool,
#                      dropout_inp=dropout_inp, dropout_hid=dropout_hid)

model.summary()
opt = RMSprop(lr=0.01)
model.compile(optimizer=opt, loss=loss)

#train_y = numpy.expand_dims(train_y, -1)

fParams = 'bestmodels/' + saving + '.hdf5'
fResult = saving + '.txt'

if n_classes == -1:
    type = 'linear'
elif n_classes == 1:
    type = 'binary'
else:
    type = 'multi'

saveResult = SaveResult([valid_x, valid_y, test_x, test_y],
                        metric_type=type, fileResult=fResult, fileParams=fParams)

callbacks = [saveResult, NanStopping()]
his = model.fit(train_x, train_y,
                validation_data=(valid_x, valid_y),
                nb_epoch=6, batch_size=10, callbacks=callbacks)
