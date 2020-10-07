import numpy
import prepare_data
import sys

sys.path.append('/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/NCE/')
sys.path.append('/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/Classifier/')
sys.path.append('../NCE')
# import load_data

args = prepare_data.arg_passing(sys.argv)
numpy.random.seed(args['-seed'])

from keras.optimizers import *
from keras.objectives import *
from create_model import *

from collections import namedtuple
import gzip
import cPickle

################################# LOAD DATA #######################################################

print 'load parameters...'
try:
    feature = args['-feature']
    print feature
    dataset = args['-data']
    nnet_model = args['-nnetM']
    saving = args['-saving']
    if 'lstm' in feature:
        pretrain = args['-pretrain']
        dim = str(args['-dim'])
        print dim
    if 'doc2vec' in feature:
        dim = str(args['-dim'])
        print dim
    regs = args['-reg']
    node_size = int(args['-node_size'])
except:
    feature = 'distance'
    pretrain = 'apache'
    dataset = 'CB'
    nnet_model = 'dense'
    saving = 'test_lstm_distance_NN'
    dim = '50'
    regs = ['inphid']
    node_size = 7

# test environment
sys.path.append('/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/NCE/')
sys.path.append('/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/Classifier/')
dataset_lstm = '/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/NCE/lstm2v_feature/lstm2v_CB_apache_dim50.pkl.gz'
dataset_distance = '/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/NCE/distance_feature/tfidf_cosine_CB.pkl.gz'

if feature == 'lstm_distance':
    print 'load lstm feature...'
    dataset_lstm = '../NCE/lstm2v_feature/lstm2v_' + dataset + '_' + pretrain + '_dim' + dim + '.pkl.gz'
    train_lstm, train_y, valid_lstm, valid_y, test_lstm, test_y = prepare_data.load_lstm2v_features(dataset_lstm)

    print 'load distance feature...'
    dataset_distance = '../NCE/distance_feature/tfidf_cosine_' + dataset + '.pkl.gz'
    train_distance, train_y_distance, valid_distance, valid_y_distance, test_distance, test_y_distance = prepare_data.load_distance_features(
        dataset_distance)

    print 'concatenate lstm and distance features...'
    train_x = numpy.concatenate([train_lstm, train_distance], axis=1)
    valid_x = numpy.concatenate([valid_lstm, valid_distance], axis=1)
    test_x = numpy.concatenate([test_lstm, test_distance], axis=1)

elif feature == 'doc2vec_distance':
    print 'load doc2vec feature...'
    dataset_d2v = '../NCE/doc2vec_feature/doc2vec_' + dataset + '_dim' + dim + '.pkl.gz'
    train_d2v, train_y, valid_d2v, valid_y, test_d2v, test_y = prepare_data.load_doc2vec_features(dataset_d2v)

    print 'load distance feature...'
    dataset_distance = '../NCE/distance_feature/tfidf_cosine_' + dataset + '.pkl.gz'
    train_distance, train_y_distance, valid_distance, valid_y_distance, test_distance, test_y_distance = prepare_data.load_distance_features(
        dataset_distance)

    print 'concatenate lstm and distance features...'
    train_x = numpy.concatenate([train_d2v, train_distance], axis=1)
    valid_x = numpy.concatenate([valid_d2v, valid_distance], axis=1)
    test_x = numpy.concatenate([test_d2v, test_distance], axis=1)

elif feature == 'distance':
    print 'load distance feature...'
    dataset_distance = '../NCE/distance_feature/tfidf_cosine_' + dataset + '.pkl.gz'
    train_x, train_y, valid_x, valid_y, test_x, test_y = prepare_data.load_distance_features(
        dataset_distance)

elif feature == 'lstm':
    print 'load lstm feature...'
    dataset_lstm = '../NCE/lstm2v_feature/lstm2v_' + dataset + '_' + pretrain + '_dim' + dim + '.pkl.gz'
    train_x, train_y, valid_x, valid_y, test_x, test_y = prepare_data.load_lstm2v_features(dataset_lstm)
else:
    print 'check a feature type input'

# set dropout
if 'hid' in regs:
    dropout_hid = True
    print 'Dropout layer: True'
else:
    dropout_hid = False

print '---training set---'
print train_x.shape
print train_y.shape

print '---valid set---'
print valid_x.shape
print valid_y.shape

print '---testset---'
print test_x.shape
print test_y.shape
print '-------------'

n_classes = train_y.shape[-1]
n_features = train_x.shape[-1]
# set loss
loss = multi_label_loss

###################################### BUILD MODEL##################################################
print 'Building model...'

# n_classes, vocab_size, inp_len, emb_dim,
# seq_model='lstm', nnet_model='highway', pool_mode='mean',
# dropout_inp=False, dropout_hid=True

model = create_dense_multi_label(n_features=n_features, n_classes=n_classes,
                                 hidden_node_size=int(n_features / node_size), nnet_model=nnet_model,
                                 dropout=dropout_hid)

model.summary()
opt = RMSprop(lr=0.01)
model.compile(optimizer=opt, loss=loss)

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
                nb_epoch=20, batch_size=50, callbacks=callbacks)
