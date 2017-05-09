import numpy as np
import os

LABEL_SIZE = 1
IMAGE_SIZE = 32
TRAIN_NUM = 10000
TRAIN_NUMS = 50000
CHANNEL_NUM = 3

data_dir = '/home/ghz/eddy/cifar10/cifar-10-batches-bin'

def extract_data(filenames):
    label_data = None
    image_data = None
    for f in filenames:
        f1 = open(f,'r')
        buf = f1.read(TRAIN_NUM*(LABEL_SIZE + IMAGE_SIZE*IMAGE_SIZE*CHANNEL_NUM))
        data = np.frombuffer(buf ,dtype=np.uint8)
        data = data.reshape(TRAIN_NUM , LABEL_SIZE + IMAGE_SIZE*IMAGE_SIZE*CHANNEL_NUM)
        data = np.hsplit(data , [LABEL_SIZE])

        label_data_one = data[0]
        image_data_one = data[1].reshape(TRAIN_NUM , IMAGE_SIZE , IMAGE_SIZE , CHANNEL_NUM)

        if label_data is None:
            label_data = label_data_one
            image_data = image_data_one
        else:
            label_data = np.concatenate((label_data, label_data_one))
            image_data = np.concatenate((image_data, image_data_one))

    return image_data , label_data

def extract_train_data(data_dir):
    filenames = [os.path.join(data_dir , 'data_batch_%d.bin' % i) for i in range(1,6)]
    return extract_data(filenames)

def extract_test_data(data_dir):
    filenames = [os.path.join(data_dir , 'test_batch.bin')]
    return extract_data(filenames)

def dense_to_one_hot(l_data , classes_num):
    l_data = l_data.reshape(-1)
    train_num_l = l_data.shape[0]

    temp_arr = np.arange(train_num_l)*classes_num
    aim_arr = np.zeros([train_num_l , classes_num])
    aim_arr.flat[l_data + temp_arr] = 1

    return aim_arr

class cifar10_dataset(object):
    def __init__(self):
        self.image_train_data , self.label_train_data = extract_train_data(data_dir)
        self.image_test_data ,self.label_test_data =extract_test_data(data_dir)

        self.label_train_data = dense_to_one_hot(self.label_train_data , 10)
        self.label_test_data = dense_to_one_hot(self.label_test_data , 10)

    def next_batch_data(self,batch_size):
        start = 0
        start += batch_size
        end = start + batch_size
        return self.image_train_data[start:end] , self.label_train_data[start:end]
    def test_data(self):
        return  self.image_test_data , self.label_test_data
