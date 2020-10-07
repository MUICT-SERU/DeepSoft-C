#
# Divide the dataset into 3 parts: training, validation and testing.
# Each line in output contains 3 numbers indicate the set (training, validation or testing) that the datapoint belongs.
# Morakot Note: divide the dataset into 3 parts based on creation date (the issues must be sorted ACS based on their creation date)
# 60%, 20%, 20%, for training, validation and test set
import numpy
import pandas
import sys

# datasetDict = {
#     'HBASE': 'apache_ph2'
#     , 'HIVE': 'apache_ph2'
#     , 'CASSANDRA': 'apache_ph2'
#     , 'INFRA': 'apache_ph2'
#     , 'CB': 'apache_ph2'
#     , 'HADOOP': 'apache_ph2'
#     , 'DS': 'duraspace_ph2'
#     , 'FCREPO': 'duraspace_ph2'
#     , 'ISLANDORA': 'duraspace_ph2'
#     , 'JRA': 'jira'
#     , 'CONF': 'jira'
#     , 'BAM': 'jira'
#     , 'JSW': 'jira'
#     , 'MDL': 'moodle_ph2'
#     , 'SPR': 'spring_ph2'
# }


trainingSize = 60
validationSize = 20
testSize = 20

# for project, repo in datasetDict.items():
try:
    project = sys.argv[1]
    repo = sys.argv[2]
except:
    print 'No sys.argv'
    project = 'HBASE'
    repo = 'apache_ph2'

print project + ' ' + repo

data_path = 'Data/' + project + '.csv'
data = pandas.read_csv(data_path).values
print 'No. of issue: ' + str(len(data))

labels = data[:, 4].astype('int64')
if trainingSize + validationSize + testSize == 100:
    numData = len(labels)
    numTrain = (trainingSize * numData) / 100
    numValidation = (validationSize * numData) / 100
    numTest = (testSize * numData) / 100

    print "Total data: %s" % numData
    print "Training size: %s, validation size: %s, testing size: %s" % (numTrain, numValidation, numTest)
    print "Total: %s" % (numTrain + numValidation + numTest)

    divided_set = numpy.zeros((len(labels), 3)).astype('int64')

    divided_set[0:numTrain - 1, 0] = 1
    divided_set[numTrain - 1:numTrain + numValidation - 1, 1] = 1
    divided_set[numTrain + numValidation - 1:numData, 2] = 1

    f = open('Data/' + project + '_3sets.txt', 'w')
    f.write('train\tvalid\ttest')
    for s in divided_set:
        f.write('\n%d\t%d\t%d' % (s[0], s[1], s[2]))
    f.close()
else:
    print 'check size'
