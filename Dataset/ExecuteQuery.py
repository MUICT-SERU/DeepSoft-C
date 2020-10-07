import MySQLdb
from decimal import Decimal

import numpy
import gzip
import cPickle
import sys
import random

datasetDict = {
    'HBASE': 'apache_ph2'
    , 'HIVE': 'apache_ph2'
    , 'CASSANDRA': 'apache_ph2'
    , 'INFRA': 'apache_ph2'
    , 'CB': 'apache_ph2'
    , 'HADOOP': 'apache_ph2'
    , 'DS': 'duraspace_ph2'
    , 'FCREPO': 'duraspace_ph2'
    , 'ISLANDORA': 'duraspace_ph2'
    , 'JRA': 'jira'
    , 'CONF': 'jira'
    , 'BAM': 'jira'
    , 'JSW': 'jira'
    , 'MDL': 'moodle_ph2'
    , 'SPR': 'spring_ph2'
    }


def load(path):
    f = gzip.open(path, 'rb')

    train_t, train_d, train_y, \
    valid_t, valid_d, valid_y, \
    test_t, test_d, test_y = cPickle.load(f)

    return train_t, train_d, train_y, valid_t, valid_d, valid_y, test_t, test_d, test_y

# dataset = '/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/Dataset/Data/HADOOP.pkl.gz'
#
# for data, _ in datasetDict.items():
#     dataset = '../Dataset/Data/' + data + '.pkl.gz'
#     train_t, train_d, train_y, valid_t, valid_d, valid_y, test_t, test_d, test_y = load(dataset)
#     print ('ntrain: %d, n_valid: %d, n_test: %d' % (len(train_y), len(valid_y), len(test_y)))
#     # test_y is ground truth
#
#     train_y = numpy.array(train_y)
#     valid_y = numpy.array(valid_y)
#     test_y = numpy.array(test_y)
#
#     train_y_frequency = numpy.sum(numpy.vstack((train_y, valid_y)), axis=0)
#     # valid_y_frequency = numpy.sum(numpy.vstack((valid_y)), axis=0)
#     test_y_frequency = numpy.sum(numpy.vstack((test_y)), axis=0)
#
#     all = numpy.vstack((train_y_frequency, test_y_frequency))
#
#     numpy.savetxt('DataAnalysis/' + data + "_frequency.csv", all, delimiter=",")
#
#     print data
#     print train_y_frequency.argsort()[-5:]
#     top = train_y_frequency.argsort()[-5:]
#     for i in range(len(top)):
#         print chr(top[i]+24)

connection = MySQLdb.connect(host='10.8.240.180', user='ReshDBAdmin', passwd='4688021')
cursor = connection.cursor()

# def countComponentLabel():
#
#     for project, repo in sorted(datasetDict.items()):
#         query = 'Update ' + repo + '.component_label Set label_mark = 1'
#         cursor.execute(query)
#         connection.commit()
#
#     for project, repo in sorted(datasetDict.items()):
#         query = 'Update ' + repo + '.component_label Set label_mark = 0 where issuekey LIKE \'' \
#                 + project + '-%\' and component in (SELECT component from (SELECT component, count(*) as num from ' \
#                 + repo + '.component_label WHERE issuekey LIKE \'' \
#                 + project + '-%\' group by component)T where num < 2);'
#         print query
#         cursor.execute(query)
#         connection.commit()
#
# countComponentLabel()

# def query():
#     for project, repo in sorted(datasetDict.items()):
#         query = "SELECT count(DISTINCT issueKey) as result from " + repo + ".component_label where issuekey like '" + project + "-%' and label_mark = 1"
#         print query
#         cursor.execute(query)
#         result = cursor.fetchall()
#         print project + '\t' + repo + '\t' + str(result[0][0])
#
# query()

# def issueswithcomponent():
#     for project, repo in sorted(datasetDict.items()):
#         query = "SELECT count(DISTINCT issueKey) as result from " + repo + ".component_label where issuekey like '" + project + "-%'"
#         # print query
#         cursor.execute(query)
#         result = cursor.fetchall()
#         print project + '\t' + str(result[0][0])
#
# issueswithcomponent()

# def nocomponent():
#     for project, repo in sorted(datasetDict.items()):
#         query = "SELECT count(DISTINCT component) as result from " + repo + ".component_label where issuekey like '" + project + "-%'"
#         # print query
#         cursor.execute(query)
#         result = cursor.fetchall()
#         print project + '\t' + str(result[0][0])
#
# nocomponent()

# def finalIssues():
#     for project, repo in sorted(datasetDict.items()):
#         query = "SELECT count(DISTINCT issuekey) as result from " + repo + ".component_label where issuekey like '" + project + "-%'  and label_mark = 1"
#         # print query
#         cursor.execute(query)
#         result = cursor.fetchall()
#         print project + '\t' + str(result[0][0])
#
# finalIssues()

def finalcomponent():
    for project, repo in sorted(datasetDict.items()):
        query = "SELECT count(DISTINCT component) as result from " + repo + ".component_label where issuekey like '" + project + "-%'  and label_mark = 1"
        # print query
        cursor.execute(query)
        result = cursor.fetchall()
        print project + '\t' + str(result[0][0])

finalcomponent()


# def investigateComponentInText():
#     for project, repo in sorted(datasetDict.items()):
#         query = "SELECT distinct component from " + repo + ".component_label where issuekey like '" + project + "-%';"
#         NoofMatchIssue = 0
#         # print query
#         cursor.execute(query)
#         componentList = cursor.fetchall()
#         for component in componentList:
#             # print component[0]
#             query = "SELECT count(*) as result from " + repo + ".component_issue where (title like '% " + component[0] + " %' or description like '% " + component[0] + " %') and component = '" + component[0] + "' and issuekey like '" + project + "-%';"
#             cursor.execute(query)
#             result = cursor.fetchone()
#             NoofMatchIssue = NoofMatchIssue + result[0]
#         print project + '\t' + repo + '\t' + str(NoofMatchIssue)
#     connection.close()

# investigateComponentInText()
