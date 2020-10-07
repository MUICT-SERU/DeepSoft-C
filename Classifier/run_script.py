import os

# This run_script runs classifiers. select mode to run

# mode option
#   lstm-highway:
#       this is end-to-end approach using trained embedding matrix to initiated LSTM and uses Highway as a classifier.
#       output: The result is in /classification/log
#               <project name>_lstm_highway_dim<number of dimensions>_reginphid_prefixed_lm_poolmean.txt
#               Model is in models/HBASE_lstm_highway_dim10_reginphid_prefixed_lm_poolmean.json
#
#   lstm-NeuralNet:
#       this is end-to-end approach using trained embedding matrix to initiated LSTM and uses tradition Neural Net. as a classifier.
#       output: The result is in /classification/log
#               <project name>_lstm_NN_dim<number of dimensions>_reginphid_prefixed_lm_poolmean.txt
#               Model is in models/HBASE_lstm_highway_dim10_reginphid_prefixed_lm_poolmean.json
#
#   lstm-rf:
#       this mode runs RF using the distance_feature vectors extracted from trained LSTM
#       output: The result is in /classification/log
#               RF_lstm2v_<project name>_dim<number of dimensions>.txt
#
#   BoW-NeuralNet:
#       this is baseline for document representation using BoW and neural network
#       output: The result is in /classification/log
#               <project name>_bow_NN.txt
#               HBASE_bow_NN.txt
#   frequency:
#       this is a baseline using freguency of the components


# modeList = ['lstm-NeuralNet-e2e']
# modeList = ['lstm-NeuralNet', 'frequency']
# modeList = ['BoW-NeuralNet']
# modeList = ['doc2vec-NeuralNet']
# modeList = ['lstm-NeuralNet']
# modeList = ['LDA-KL']
# modeList = ['lstm2v-NeuralNet']
# modeList = ['doc2vec-NeuralNet']
modeList = ['frequency']
# modeList = ['lstm-NeuralNet']
# modeList = ['tfidf_cos']
# modeList = ['lstm2v_sim']
# modeList = ['lstm-distnace-NeuralNet']
# modeList = ['distance-NeuralNet']
# modeList = ['doc2vec-distnace-NeuralNet']
# modeList = ['lstm-NeuralNet']
# modeList = ['lstm-distnace-BR']

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

dims = ['10', '50', '100', '150', '200']
# dims = ['50']

startK = '5'
stopK = '20'
stepK = '5'

# flag = 'THEANO_FLAGS=''mode=FAST_RUN,device=gpu,floatX=float32'' '
flag = ''

note = ''  # nospace

for mode in modeList:
    # if mode == 'lstm-highway':
    #     nnet_models = ['highway']  # ['dense', 'highway']
    #     seq_models = ['lstm']  # ['gru', 'lstm', 'rnn']
    #     regs = ['inphid']  # ['x', 'inp', 'hid', 'inphid'] # 'x' means no dropout
    #     pretrains = ['fixed_lm']  # ['x', 'finetune', 'fixed'] should use finetune_lm or fixed_lm for using lstm
    #     # 'x' means no pretraining,
    #     # 'finetune' means embedding matrix is initialized by pretrained parameters
    #     # 'fixed' means using pretrained embedding matrix as input features
    #     # add '_lm' if using 'lstm' for pretraining, default: 'bilinear' for pretraining
    #     pools = ['mean']  # ['mean', 'max', 'last']
    #     maxlen = '100'
    #     for project, repo in dataset.items():
    #         for nnet in nnet_models:
    #             for seq in seq_models:
    #                 for dim in dims:
    #                     for reg in regs:
    #                         for pretrain in pretrains:
    #                             for pool in pools:
    #                                 cmd = 'THEANO_FLAGS=''mode=FAST_RUN,device=gpu,floatX=float32'' python training.py -data ' + project + ' -dataPre ' + repo + \
    #                                       ' -nnetM ' + nnet + ' -seqM ' + seq + ' -dim ' + dim + \
    #                                       ' -reg ' + reg + ' -pretrain ' + pretrain + ' -pool ' + pool + ' -len ' + maxlen
    #                                 cmd += ' -saving ' + project + '_' + seq + '_' + nnet + '_dim' + dim + \
    #                                        '_reg' + reg + '_pre' + pretrain + '_pool' + pool
    #                                 print cmd
    #                                 os.system(cmd)
    if mode == 'lstm-NeuralNet-e2e':
        nnet_models = ['dense']  # dense=traditional NN, hdl = 0
        seq_models = ['lstm']  # ['gru', 'lstm', 'rnn']
        regs = ['inphid']  # ['x', 'inp', 'hid', 'inphid'] # 'x' means no dropout
        pretrains = ['fixed']  # ['x', 'finetune', 'fixed'] should use finetune_lm or fixed_lm for using lstm
        # 'x' means no pretraining,
        # 'finetune' means embedding matrix is initialized by pretrained parameters
        # 'fixed' means using pretrained embedding matrix as input features
        # add '_lm' if using 'lstm' for pretraining, default: 'bilinear' for pretraining
        pools = ['mean']  # ['mean', 'max', 'last']
        maxlen = '100'
        for project, repo in dataset.items():
            for nnet in nnet_models:
                for seq in seq_models:
                    for dim in dims:
                        for reg in regs:
                            for pretrain in pretrains:
                                for pool in pools:
                                    cmd = flag + 'python training.py -data ' + project + ' -dataPre ' + repo + \
                                          ' -nnetM ' + nnet + ' -seqM ' + seq + ' -dim ' + dim + \
                                          ' -reg ' + reg + ' -pretrain ' + pretrain + ' -pool ' + pool + ' -len ' + maxlen
                                    cmd += ' -saving ' + project + '_' + seq + '_' + 'NeuralNet' + '_dim' + dim + \
                                           '_reg' + reg + '_pre' + pretrain + '_pool' + pool
                                    print cmd
                                    os.system(cmd)

                                    print 'compute performance@topK'
                                    cmd = 'python topKmetric.py -startK ' + startK + ' -stepK ' + stepK + ' -stopK ' + \
                                          stopK + ' -project ' + project + ' -fileName ' + project + '_' + seq + '_' + 'NeuralNet' + '_dim' + dim + \
                                          '_reg' + reg + '_pre' + pretrain + '_pool' + pool + ' -note ' + note
                                    os.system(cmd)
    # elif mode == 'lstm2v-NeuralNet':
    #     nnet_models = ['dense']  # dense=traditional NN, hdl = 0
    #     seq_models = ['lstm']  # ['gru', 'lstm', 'rnn']
    #     regs = ['x']  # ['x', 'inp', 'hid', 'inphid'] # 'x' means no dropout
    #     pretrains = ['x']  # ['x', 'finetune', 'fixed'] should use finetune_lm or fixed_lm for using lstm
    #     # 'x' means no pretraining,
    #     # 'finetune' means embedding matrix is initialized by pretrained parameters
    #     # 'fixed' means using pretrained embedding matrix as input features
    #     # add '_lm' if using 'lstm' for pretraining, default: 'bilinear' for pretraining
    #     pools = ['mean']  # ['mean', 'max', 'last']
    #     maxlen = '100'
    #     for project, repo in dataset.items():
    #         for nnet in nnet_models:
    #             for seq in seq_models:
    #                 for dim in dims:
    #                     for reg in regs:
    #                         for pretrain in pretrains:
    #                             for pool in pools:
    #                                 cmd = 'python training.py -data ' + project + ' -dataPre ' + repo + \
    #                                       ' -nnetM ' + nnet + ' -seqM ' + seq + ' -dim ' + dim + \
    #                                       ' -reg ' + reg + ' -pretrain ' + pretrain + ' -pool ' + pool + ' -len ' + maxlen
    #                                 cmd += ' -saving ' + project + '_' + seq + '_' + 'NeuralNet' + '_dim' + dim + \
    #                                        '_reg' + reg + '_pre' + pretrain + '_pool' + pool
    #                                 print cmd
    #                                 os.system(cmd)
    #
    #                                 print 'compute performance@topK'
    #                                 cmd = 'python topKmetric.py -startK ' + startK + ' -stepK ' + stepK + ' -stopK ' + \
    #                                       stopK + ' -fileName ' + project + '_' + seq + '_' + 'NeuralNet' + '_dim' + dim + \
    #                                       '_reg' + reg + '_pre' + pretrain + '_pool' + pool
    #                                 os.system(cmd)
    elif mode == 'BoW-NeuralNet':
        for project, repo in dataset.items():
            nnet_models = 'dense'
            maxlen = '100'
            cmd = 'python baseline_bow.py -data ' + project + ' -nnetM ' + nnet_models + ' -vocab ' + maxlen + ' -mode bow' + ' -saving ' + project + '_BoW_NN'
            print cmd
            os.system(cmd)

            print 'compute performance@topK'
            cmd = 'python topKmetric.py -startK ' + startK + ' -stepK ' + stepK + ' -stopK ' + \
                  stopK + ' -project ' + project + ' -fileName ' + project + '_BoW_NN'
            os.system(cmd)

    elif mode == 'doc2vec-NeuralNet':
        for project, repo in dataset.items():
            print project
            nnet_models = 'dense'
            maxlen = '100'
            cmd = 'python baseline_bow.py -data ' + project + ' -nnetM ' + nnet_models + ' -vocab ' + maxlen + ' -mode doc2vec' + ' -saving ' + project + '_doc2vec_NN'
            print cmd
            os.system(cmd)

            print 'compute performance@topK'
            cmd = 'python topKmetric.py -startK ' + startK + ' -stepK ' + stepK + ' -stopK ' + \
                  stopK + ' -project ' + project + ' -fileName ' + project + '_doc2vec_NN'
            os.system(cmd)
    elif mode == 'LDA-KL':
        for project, repo in dataset.items():
            note = '2_fix'
            print project
            cmd = 'python lda-kl.py -data ' + project + ' -saving ' + project + '_lda-kl'
            print cmd
            os.system(cmd)

            print 'compute performance@topK'
            cmd = 'python topKmetric.py -startK ' + startK + ' -stepK ' + stepK + ' -stopK ' + \
                  stopK + ' -project ' + project + ' -fileName ' + project + '_lda-kl' + ' -note ' + note
            os.system(cmd)
    elif mode == 'tfidf_cos':
        note = '3rd'
        for project, repo in dataset.items():
            print project
            cmd = 'python tfidf_cos.py -data ' + project + ' -saving ' + project + '_tfidf_cos'
            print cmd
            os.system(cmd)

            print 'compute performance@topK'
            cmd = 'python topKmetric.py -startK ' + startK + ' -stepK ' + stepK + ' -stopK ' + \
                  stopK + ' -project ' + project + ' -fileName ' + project + '_tfidf_cos' + ' -note ' + note
            os.system(cmd)

    elif mode == 'lstm-rf':
        classifier = 'rf'
        pretrain = 'lstm2v'
        for project, repo in dataset.items():
            x = pretrain + '_' + project
            for dim in dims:
                cmd = 'python ' + classifier + '.py -data ' + project + '_' + repo + ' -dim ' + dim + ' -pretrain ' + \
                      pretrain + ' -saving ' + classifier + '_' + x + '_dim' + dim
                print cmd
                os.system(cmd)
    elif mode == 'frequency':
        for project, repo in dataset.items():
            cmd = 'python baseline.py frequency ' + project
            print cmd
            os.system(cmd)

            print 'compute performance@topK'
            cmd = 'python topKmetric.py -startK ' + startK + ' -stepK ' + stepK + ' -stopK ' + \
                  stopK + ' -project ' + project + ' -fileName ' + project + '_frequency'
            os.system(cmd)
    elif mode == 'lstm2v_sim':
        for project, repo in dataset.items():
            for dim in dims:
                cmd = 'python lstm2v_sim.py -data ' + project + ' -repo ' + repo + ' -dim ' + dim + \
                      ' -saving ' + project + '_dim' + dim + '_lstm2v_cos'
                print cmd
                os.system(cmd)

                print 'compute performance@topK'
                cmd = 'python topKmetric.py -startK ' + startK + ' -stepK ' + stepK + ' -stopK ' + \
                      stopK + ' -project ' + project + ' -fileName ' + project + '_dim' + dim + '_lstm2v_cos' + ' -note ' + note
                os.system(cmd)
    elif mode == 'lstm-distnace-NeuralNet':
        note = '5_lr=0.03,ep=300,bs=100,nodropout'
        dims = ['10', '50', '100', '150', '200', '300']

        feature = 'lstm_distance'
        node_sizes = ['1', '2', '3', '4', '5', '6', '7', '8']

        nnet_model = 'dense'
        reg = 'no'
        for project, repo in dataset.items():
            for dim in dims:
                for node_size in node_sizes:
                    cmd = flag + 'python anyfeature_NN_multilabel.py -data ' + project + ' -pretrain ' + repo + \
                          ' -feature ' + feature + ' -nnetM ' + nnet_model + ' -dim ' + dim + \
                          ' -reg ' + reg + ' -node_size ' + node_size

                    cmd += ' -saving ' + project + '-pretrain_' + repo + '_lstm-distance_' + \
                           nnet_model + '_dim_' + dim + \
                           '_reg' + reg + '_node_size_' + node_size

                    print cmd
                    os.system(cmd)

                    print 'compute performance@topK'
                    cmd = 'python topKmetric.py -startK ' + startK + ' -stepK ' + stepK + ' -stopK ' + \
                          stopK + ' -project ' + project + ' -fileName ' + project + '-pretrain_' + repo + '_lstm-distance_' + \
                          nnet_model + '_dim_' + dim + \
                          '_reg' + reg + '_node_size_' + node_size + ' -note ' + note
                    os.system(cmd)

    elif mode == 'doc2vec-distnace-NeuralNet':
        note = '1_'  # <<< change when run
        dims = ['50', '100', '200']
        feature = 'doc2vec_distance'
        node_sizes = ['2', '3']
        nnet_model = 'dense'
        reg = 'inphid'
        for project, repo in dataset.items():
            for dim in dims:
                for node_size in node_sizes:
                    cmd = flag + 'python anyfeature_NN_multilabel.py -data ' + project + \
                          ' -feature ' + feature + ' -nnetM ' + nnet_model + ' -dim ' + dim + \
                          ' -reg ' + reg + ' -node_size ' + node_size

                    cmd += ' -saving ' + project + '_doc2vec-distance_' + \
                           nnet_model + '_dim_' + dim + \
                           '_reg' + reg + '_node_size_' + node_size

                    print cmd
                    os.system(cmd)

                    print 'compute performance@topK'
                    cmd = 'python topKmetric.py -startK ' + startK + ' -stepK ' + stepK + ' -stopK ' + \
                          stopK + ' -project ' + project + ' -fileName ' + project + '_doc2vec-distance_' + \
                          nnet_model + '_dim_' + dim + \
                          '_reg' + reg + '_node_size_' + node_size + ' -note ' + note
                    os.system(cmd)

    elif mode == 'distance-NeuralNet':
        feature = 'distance'
        node_sizes = ['2','3','4']
        nnet_model = 'dense'
        reg = 'inphid'
        note = '1st'
        for project, repo in dataset.items():
            for node_size in node_sizes:
                cmd = flag + 'python anyfeature_NN_multilabel.py -data ' + project + ' -pretrain ' + repo + \
                      ' -feature ' + feature + ' -nnetM ' + nnet_model + \
                      ' -reg ' + reg + ' -node_size ' + node_size

                cmd += ' -saving ' + project + '_distance_' + \
                       nnet_model + \
                       '_reg' + reg + '_node_size_' + node_size

                print cmd
                os.system(cmd)

                print 'compute performance@topK'
                cmd = 'python topKmetric.py -startK ' + startK + ' -stepK ' + stepK + ' -stopK ' + \
                      stopK + ' -project ' + project + ' -fileName ' + project + '_distance_' + \
                      nnet_model + \
                      '_reg' + reg + '_node_size_' + node_size + ' -note ' + note
                print cmd
                os.system(cmd)

    elif mode == 'lstm-NeuralNet':
        dims = ['10', '50', '100', '200']
        feature = 'lstm'
        node_sizes = ['2', '3']
        nnet_model = 'dense'
        reg = 'inphid'
        note = '1st'
        for project, repo in dataset.items():
            for node_size in node_sizes:
                cmd = flag + 'python anyfeature_NN_multilabel.py -data ' + project + ' -pretrain ' + repo + \
                      ' -feature ' + feature + ' -nnetM ' + nnet_model + \
                      ' -reg ' + reg + ' -node_size ' + node_size

                cmd += ' -saving ' + project + '_' + feature + '_' + \
                       nnet_model + \
                       '_reg' + reg + '_node_size_' + node_size

                print cmd
                os.system(cmd)

                print 'compute performance@topK'
                cmd = 'python topKmetric.py -startK ' + startK + ' -stepK ' + stepK + ' -stopK ' + \
                      stopK + ' -project ' + project + ' -fileName ' + project + '_' + feature + '_' + \
                      nnet_model + \
                      '_reg' + reg + '_node_size_' + node_size + ' -note ' + note
                print cmd
                os.system(cmd)

    elif mode == 'lstm-distnace-BR':
        feature = 'lstm_distance'
        dims = ['10', '50', '100', '200']
        classifiers = ['rf', 'svm']
        note = 'no'
        for dim in dims:
            for project, repo in dataset.items():
                for classifier in classifiers:
                    cmd = flag + 'python classifier_br.py -data ' + project + ' -pretrain ' + repo + \
                          ' -feature ' + feature + ' -dim ' + dim + \
                          ' -classifier ' + classifier

                    cmd += ' -saving ' + project + '_' + repo + \
                           '_' + feature + '_dim' + dim + \
                           '_' + classifier

                    print cmd
                    os.system(cmd)

                    print 'compute performance@topK'
                    cmd = 'python topKmetric.py -startK ' + startK + ' -stepK ' + stepK + ' -stopK ' + \
                          stopK + ' -project ' + project + ' -fileName ' + project + '_' + repo + \
                          '_' + feature + '_dim' + dim + \
                          '_' + classifier + ' -note ' + note
                    print cmd
                    os.system(cmd)
