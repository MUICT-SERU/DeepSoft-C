import gzip
import sys
import cPickle
import load_raw_text
import Preprocess_pretraindata


def main():
    # load training data:
    project = 'HBASE'
    repo = 'apache'
    try:
        project = sys.argv[1]
        repo = sys.argv[2]
    except:
        print 'No sys.argv'

    data_path = 'Data/' + project + '.csv'
    print data_path
    title, description, labels = load_raw_text.load(data_path)

    print len(title)
    print len(description)
    print len(labels)
    print '----'
    f = open('Data/' + project + '_3sets.txt', 'r')
    train_ids, valid_ids, test_ids = [], [], []
    count = -2
    for line in f:
        if count == -2:
            count += 1
            continue

        count += 1
        ls = line.split()
        if ls[0] == '1':
            train_ids.append(count)
        if ls[1] == '1':
            valid_ids.append(count)
        if ls[2] == '1':
            test_ids.append(count)

    print 'ntrain, nvalid, ntest: ', len(train_ids), len(valid_ids), len(test_ids)

    train_title, train_description, train_labels = title[train_ids], description[train_ids], labels[train_ids]
    valid_title, valid_description, valid_labels = title[valid_ids], description[valid_ids], labels[valid_ids]
    test_title, test_description, test_labels = title[test_ids], description[test_ids], labels[test_ids]

    print str(len(train_title)) + ' ' + str(len(train_description)) + ' ' + str(len(train_labels))
    print str(len(valid_title)) + ' ' + str(len(valid_description)) + ' ' + str(len(valid_labels))
    print str(len(test_title)) + ' ' + str(len(test_description)) + ' ' + str(len(test_labels))
    print '---'

    f_dict = gzip.open('Data/' + repo + '.dict.pkl.gz', 'rb')
    dictionary = cPickle.load(f_dict)
    train_t, train_d = Preprocess_pretraindata.grab_data(train_title, train_description, dictionary)
    valid_t, valid_d = Preprocess_pretraindata.grab_data(valid_title, valid_description, dictionary)
    test_t, test_d = Preprocess_pretraindata.grab_data(test_title, test_description, dictionary)

    # print str(len(train_t)) + ' ' + str(len(train_d)) + ' ' + str(len(train_labels))
    # print str(len(valid_t)) + ' ' + str(len(valid_d)) + ' ' + str(len(valid_labels))
    # print str(len(test_t)) + ' ' + str(len(test_d)) + ' ' + str(len(test_labels))
    # print '---'

    f = gzip.open('Data/' + project + '.pkl.gz', 'wb')

    # print train_labels.dtype
    # train_labels = train_labels.astype(int)
    # valid_labels = valid_labels.astype(int)
    # test_labels = test_labels.astype(int)
    # print train_labels.dtype
    # print type(train_d)
    print train_labels
    cPickle.dump((train_t, train_d, train_labels,
                  valid_t, valid_d, valid_labels,
                  test_t, test_d, test_labels), f, -1)
    f.close()


if __name__ == '__main__':
    main()
