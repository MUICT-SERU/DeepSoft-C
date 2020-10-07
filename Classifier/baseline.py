import numpy
import gzip
import cPickle
import sys
import random

try:
    baseline = sys.argv[1]
    data = sys.argv[2]
except:
    baseline = 'frequency'
    data = 'HBASE'


def load(path):
    f = gzip.open(path, 'rb')

    train_t, train_d, train_y, \
    valid_t, valid_d, valid_y, \
    test_t, test_d, test_y = cPickle.load(f)

    return train_t, train_d, train_y, valid_t, valid_d, valid_y, test_t, test_d, test_y


# load file
dataset = '../Dataset/Data/' + data + '.pkl.gz'
train_t, train_d, train_y, valid_t, valid_d, valid_y, test_t, test_d, test_y = load(dataset)
print ('ntrain: %d, n_valid: %d, n_test: %d' % (len(train_y), len(valid_y), len(test_y)))
# test_y is ground truth

train_y = numpy.array(train_y)
valid_y = numpy.array(valid_y)

y_pred = []

if baseline == 'frequency':
    frequency = numpy.sum(numpy.vstack((train_y, valid_y)), axis=0)
    for i in range(len(test_y)):
        y_pred.append((frequency))
        # print y_pred
    # y_pred = numpy.fliplr(y_pred)
elif baseline == 'median':
    print 'median'

elif baseline == 'random':
    y_pred_base = numpy.ones(len(test_y))
    for i in range(len(y_pred_base)):
        y_pred_base[i] = random.choice(all)

numpy.savetxt('log/output/' + data + "_frequency_actual.csv", test_y, delimiter=",")
numpy.savetxt('log/output/' + data + "_frequency_estimate.csv", y_pred, delimiter=",")
