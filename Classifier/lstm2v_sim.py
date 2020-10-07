#
# dataset = '/home/mc650/Dropbox/PythonWorkspace/PredictingComponent-multilabel/Dataset/Data/HBASE.pkl.gz'
# sys.path.append('/home/mc650/Dropbox/PythonWorkspace/PredictingComponent-multilabel/Classifier')
#
#
# sys.path.append('/home/mc650/Dropbox/PythonWorkspace/PredictingComponent-multilabel/Classifier')
# sys.path.append('/home/mc650/Dropbox/PythonWorkspace/PredictingComponent-multilabel/Classifier/')
#
#
# trian_t = numpy.array(train_t)
# trian_d = numpy.array(train_d)
#
# train_x = train_t + train_d
#




import numpy as np
import sys
# sys.path.append('/home/mc650/Dropbox/PythonWorkspace/PredictingComponent-multilabel/Classifier/')
# sys.path.append('/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/Classifier/')
import prepare_data

args = prepare_data.arg_passing_any(sys.argv)
################################# LOAD DATA #######################################################

# load LSTM features
try:
    dataset = '../NCE/lstm2v_feature/lstm2v_' + args['-data'] + '_' + args['-repo'] + '_dim' + args['-dim'] + '.pkl.gz'
    saving = args['-saving']
except:
    dataset = '../NCE/lstm2v_feature/lstm2v_JRA_jira_dim200.pkl.gz'
    dataset = '/home/mc650/Dropbox/PythonWorkspace/PredictingComponent-multilabel//NCE/lstm2v_feature/lstm2v_JRA_jira_dim200.pkl.gz'
    dataset = '/Users/Morakot/Dropbox/PythonWorkspace/PredictingComponent-multilabel/NCE/lstm2v_feature/lstm2v_JRA_jira_dim200.pkl.gz'
    saving = 'test_lstm2v_sim'

train_x, train_y, valid_x, valid_y, test_x, test_y = prepare_data.load_lstm2v_features(dataset)

# stack train and valid
train_x = np.vstack((train_x, valid_x))
train_y = np.vstack((train_y, valid_y))

# create np array for centroid of components
component_centroid = np.zeros([train_y.shape[1], train_x.shape[1]])

# mean of lstm of issues assigned the component is centrod
for i in range(len(component_centroid)):
    component_centroid[i] = np.mean(train_x[np.where(train_y[:, i] > 0)], axis=0)

# replace nan with 0
component_centroid = np.nan_to_num(component_centroid)

# Scaling features to a range [0,1]
from sklearn import preprocessing
# min_max_scaler = preprocessing.MinMaxScaler()
# component_centroid = min_max_scaler.fit_transform(component_centroid)
# test_x = min_max_scaler.transform(test_x)

normalizer = preprocessing.Normalizer().fit(component_centroid)
component_centroid = normalizer.transform(component_centroid)
test_x = normalizer.transform(test_x)


from sklearn.metrics.pairwise import cosine_similarity
predict = cosine_similarity(test_x, component_centroid)

np.savetxt('log/output/' + saving + "_actual.csv", test_y, delimiter=",")
np.savetxt('log/output/' + saving + "_estimate.csv", predict, delimiter=",")
