import prepare_data
import sys
import numpy
import topKmetric
import MySQLdb

def main():
    # read name from DB
    connection = MySQLdb.connect(host='127.0.0.1', user='ReshDBAdmin', passwd='4688021')
    cursor = connection.cursor()

    #check query with DB  project, method
    query = 'SELECT id,project,method FROM experiment.predict_component_result;'

    cursor.execute(query)
    fileNames = numpy.array(cursor.fetchall())

    for i in range(len(fileNames)):

        try:
            actualFile = 'log/output/' + fileNames[i,2] + '_actual.csv'
            estimateFile = 'log/output/' + fileNames[i,2] + '_estimate.csv'

            actual = numpy.genfromtxt(actualFile, delimiter=',')
            estimate = numpy.genfromtxt(estimateFile, delimiter=',')

            startK = 1
            stopK = 5
            stepK = 1

            recall_k = topKmetric.recall(actual, estimate, startK, stopK, stepK)
            precision_k = topKmetric.precision(actual, estimate, startK, stopK, stepK)
            map_score = topKmetric.mean_avg_prec(actual, estimate)
            mrr_score = topKmetric.mrr(actual, estimate)

            cursor.execute('''update experiment.predict_component_result set map = %s, mrr = %s , recall_1 = %s, recall_2 = %s, recall_3 = %s, recall_4 = %s, recall_5 = %s,
                                               precision_1 = %s, precision_2 = %s, precision_3 = %s, precision_4 = %s, precision_5 = %s where id = %s''',
                           (map_score, mrr_score, recall_k[0],recall_k[1],recall_k[2],recall_k[3],recall_k[4],precision_k[0],precision_k[1],precision_k[2],precision_k[3],precision_k[4],fileNames[i,0],))

            connection.commit()

        except Exception, e: print str(e)

        # cursor.execute('''INSERT into experiment.predict_component_result (project,method,recall5, recall10, recall15, recall20,note) VALUES
        #             (%s, %s, %s, %s, %s, %s, %s)''',
        #             (project, fileName, Recall_K[0], Recall_K[1], Recall_K[2], Recall_K[3], note,))
        # connection.commit()

main()


