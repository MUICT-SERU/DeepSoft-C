import os
from datetime import datetime

# This run_script runs LSTM training. select mode to run

# mode option
#   pretrain-lstm:
#       use pretrain data to pretrain lstm. this mode is for end-to-end approach (LD-RNN). pretrain-lstm uses repolist.
#       output: 'bestmodels/' + saving + '.hdf5', 'log/' + saving + '.txt', 'models/' + saving + '.json'
#   lstm2vec:
#       after pretrain embeded matrix, lstm2vec mode can extract features from lstm for passing to any classifers.
#       output: 'lstm2v_feature/' + saving + '.pkl.gz'      output is a distance_feature vector of an issues using LSTM
# modeList = ['pretrain-lstm', 'lstm2vec']
modeList = ['pretrain-lstm']
modeList = ['lstm2vec']
# modeList = ['cosine_feature']
modeList = ['doc2vec']

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

dataset = {
    'HBASE': 'apache'
    , 'HIVE': 'apache'
    , 'CASSANDRA': 'apache'
    , 'INFRA': 'apache'
    , 'CB': 'apache'
    , 'HADOOP': 'apache'
    , 'DS': 'duraspace'
    , 'FCREPO': 'duraspace'
    , 'ISLANDORA': 'duraspace'
    , 'JRA': 'jira'
    , 'CONF': 'jira'
    , 'BAM': 'jira'
    , 'JSW': 'jira'
    , 'MDL': 'moodle'
    , 'SPR': 'spring'
    }

repoList = ['apache', 'duraspace', 'jira', 'moodle', 'spring']
# repoList = ['spring']

dims = ['10', '50', '100', '150', '200', '250', '300', '400', '500']
# dims = ['300', '400', '500']
dims = ['300']

vocab = '5000'
maxlen = '100'

# flag = 'THEANO_FLAGS=''mode=FAST_RUN,device=gpu,floatX=float32'' '
flag = ''

start_time_all = datetime.now()
for mode in modeList:
    if mode == 'pretrain-lstm':
        for dim in dims:
            for repo in repoList:
                start_time = datetime.now()
                print 'Start at: {}'.format(start_time)
                command = flag + 'python lstm_pretrain.py -data ' + repo + ' -saving lstm2v_' + repo + '_dim' + dim + ' -vocab ' + vocab + ' -dim ' + dim + ' -len ' + maxlen
                print command
                os.system(command)
                # end_time = datetime.now()
                #
                # print 'Start at:\t{}'.format(start_time)
                # print 'End at:\t{}'.format(end_time)
                # print('Duration:\t{}'.format(end_time - start_time))

    elif mode == 'lstm2vec':
        for project, repo in dataset.items():
            for dim in dims:
                command = 'python lstm2vec.py -dataPre ' + repo + ' -data ' + project + ' -vocab ' + vocab + ' -dim ' + dim + ' -len ' + maxlen + ' -saving lstm2v_' + project + '_' + repo + '_dim' + dim
                print command
                os.system(command)

    elif mode == 'cosine_feature':
        for project, repo in dataset.items():
            command = 'python cosine.py -data ' + project + ' -saving tfidf_cosine_' + project
            print command
            os.system(command)
    elif mode == 'doc2vec':
        dims = ['10', '50', '100', '200']
        for project, repo in dataset.items():
            for dim in dims:
                command = 'python doc2vec.py -data ' + project + ' -dim ' + dim + ' -saving doc2vec_' + project + '_dim' + dim
                print command
                os.system(command)

end_time_all = datetime.now()
print '##############################'
print 'Start at:\t{}'.format(start_time_all)
print 'End at:\t{}'.format(end_time_all)
print 'Duration all:\t{}'.format(end_time_all - start_time_all)
