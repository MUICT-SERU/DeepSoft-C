import prepare_data
import sys
import numpy
import MySQLdb
import topKmetric

def main():
    args = prepare_data.arg_passing(sys.argv)

    try:
        project = args['-project']
        fileName = args['-fileName']
        startK = int(args['-startK'])
        stepK = int(args['-stepK'])
        stopK = int(args['-stopK'])
        note = args['-note']
        measure = args['-measure']
    except:
        print 'No args'
        print 'Example: python topKmetric.py -project HBASE -startK 1 -stepK 1 -stopK 30 ' \
              '-fileName MDL_moodle_lstm_distance_dim10_svm -measure recall-at-k -note test'
        project = 'HBASE'
        fileName = 'MDL_moodle_lstm_distance_dim10_svm'
        startK = 1
        stepK = 1
        stopK = 30
        measure = 'recall-at-k'
        note = 'check overfit on trainingset'

    actualFile = 'log/output/' + fileName + '_actual.csv'
    estimateFile = 'log/output/' + fileName + '_estimate.csv'

    # actualFile = 'log/output/' + fileName + '_actual_train.csv'
    # estimateFile = 'log/output/' + fileName + '_estimate_train.csv'

    # actualFile = 'log/output/' + 'HBASE_lstm_NeuralNet_dim200_regx_prefixed_lm_poolmean' + '_actual.csv'
    # estimateFile = 'log/output/' + 'HBASE_lstm_NeuralNet_dim200_regx_prefixed_lm_poolmean' + '_estimate.csv'
    #
    # startK = 1
    # stopK = 50
    # stepK = 5

    # python topKmetric.py -startK 1 -stepK 5 -stopK 50 -fileName HBASE_lstm_NeuralNet_dim200_regx_prefixed_lm_poolmean

    actual = numpy.genfromtxt(actualFile, delimiter=',')
    estimate = numpy.genfromtxt(estimateFile, delimiter=',')

    # print len(actual)
    # print len(estimate)

    outputFileName = 'log/output/perf_K_' + fileName + '_' + str(startK) + '_' + str(stepK) + '_' + str(stopK) + '_' + measure

    if measure == 'recall-at-k':
        recall_k = topKmetric.recall(actual, estimate, startK, stopK, stepK)
        numpy.savetxt(outputFileName + ".csv", recall_k, delimiter=",", fmt='%1.4f')
        with open('log/output/perf_BR_SVM_' + str(startK) + '_' + str(stepK) + '_' + str(stopK) + '.csv',
                  'a') as myoutput:
            myoutput.write(fileName + "," + ",".join(map(str, recall_k)) + '\n')
    if measure == 'precision-at-k':
        prec_k = topKmetric.precision(actual, estimate, startK, stopK, stepK)
        numpy.savetxt(outputFileName + ".csv", prec_k, delimiter=",", fmt='%1.4f')
        with open('log/output/perf_BR_SVM_' + str(startK) + '_' + str(stepK) + '_' + str(stopK) + '.csv',
                  'a') as myoutput:
            myoutput.write(fileName + "," + ",".join(map(str, prec_k)) + '\n')
    if measure == 'map':
        map_score = topKmetric.mean_avg_prec(actual, estimate)
    if measure == 'mrr':
        mrr_score = topKmetric.mrr(actual, estimate)



    # connection = MySQLdb.connect(host='10.8.240.180', user='ReshDBAdmin', passwd='4688021')
    # cursor = connection.cursor()
    # print("%s, %s, %f, %f, %f, %f, %s" % (project, fileName, Recall_K[0], Recall_K[1], Recall_K[2], Recall_K[3], note))
    # cursor.execute('''INSERT into experiment.predict_component_result (project,method,recall5, recall10, recall15, recall20,note) VALUES
    #                 (%s, %s, %s, %s, %s, %s, %s)''',
    #                 (project, fileName, Recall_K[0], Recall_K[1], Recall_K[2], Recall_K[3], note,))
    # connection.commit()
    # connection.close()


main()