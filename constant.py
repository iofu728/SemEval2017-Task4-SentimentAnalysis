'''
@Author: gunjianpan
@Date:   2019-04-12 15:14:02
@Last Modified by:   gunjianpan
@Last Modified time: 2019-04-13 16:51:24
'''

import os

data_dir = 'data/'
pickle_dir = 'pickle/'
embedding_dir = 'embeddings/'
result_dir = 'result/'

embedding_dim = 300
num_class = 3
filter_sizes = [2, 3, 4]
num_filter = 3


basic_data_dir = '{}Subtask_A/'.format(data_dir)
train_data_dir = '{}downloaded/'.format(basic_data_dir)
test_data_dir = '{}test/'.format(basic_data_dir)

train_data_list = [train_data_dir +
                   ii for ii in os.listdir(train_data_dir) if '.txt' in ii]
temp_test_path = [ii for ii in os.listdir(test_data_dir) if '.txt' in ii][0]
test_data_path = test_data_dir + temp_test_path

label2id = {
    'positive': 0,
    'neutral': 1,
    'negative': 2,
}

id2label = {
    0: 'positive',
    1: 'neutral',
    2: 'negative',
}
