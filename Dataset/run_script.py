import os

# This run_script runs DatasettoCSV, DivideData, Preprocess_pretraindata

datasetDict = {
    # 'HBASE': 'apache_ph2'
    # , 'HIVE': 'apache_ph2'
    # , 'CASSANDRA': 'apache_ph2'
    # , 'INFRA': 'apache_ph2'
    # , 'CB': 'apache_ph2'
    # , 'HADOOP': 'apache_ph2'
    # , 'DS': 'duraspace_ph2'
    # , 'FCREPO': 'duraspace_ph2'
    # , 'ISLANDORA': 'duraspace_ph2'
     'JRA': 'jira'
#     , 'CONF': 'jira'
#     , 'BAM': 'jira'
#     , 'JSW': 'jira'
#     , 'MDL': 'moodle_ph2'
#     , 'SPR': 'spring_ph2'
    }

dataset = {
    # 'HBASE': 'apache'
    # , 'HIVE': 'apache'
    # , 'CASSANDRA': 'apache'
    # , 'INFRA': 'apache'
    # , 'CB': 'apache'
    # , 'HADOOP': 'apache'
    # , 'DS': 'duraspace'
    # , 'FCREPO': 'duraspace'
    # , 'ISLANDORA': 'duraspace'
     'JRA': 'jira'
    # , 'CONF': 'jira'
    # , 'BAM': 'jira'
    # , 'JSW': 'jira'
    # , 'MDL': 'moodle'
    # , 'SPR': 'spring'
}

# repoList = ['apache', 'jira', 'moodle']

# run DatatoCSV.py
for project, repo in datasetDict.items():
    print project + ' ' + repo
    cmd = 'python DatatoCSV.py ' + project + ' ' + repo
    print cmd
    os.system(cmd)

# # run DivideData.py
for project, repo in datasetDict.items():
    print project + ' ' + repo
    cmd = 'python DivideData.py ' + project + ' ' + repo
    print cmd
    os.system(cmd)

##run Preprocess_pretraindata.py
# for repo in repoList:
#     print repo
    # cmd = 'python Preprocess_pretraindata.py ' + repo
    # print cmd
    # os.system(cmd)

# # run Preprocess_labeleddata.py
for project, repo in dataset.items():
    print project + ' ' + repo
    cmd = 'python Preprocess_labeleddata.py ' + project + ' ' + repo
    print cmd
    os.system(cmd)
