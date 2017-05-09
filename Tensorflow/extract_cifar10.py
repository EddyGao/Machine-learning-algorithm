# -*- coding:utf-8 -*-  

import gzip
import numpy as np
import os

import tensorflow as tf

LABEL_SIZE = 1  
IMAGE_SIZE = 32  
NUM_CHANNELS = 3  
PIXEL_DEPTH = 255  
NUM_CLASSES = 10  
  
TRAIN_NUM = 10000  
TRAIN_NUMS = 50000  
TEST_NUM = 10000  
 
#接着我们定义提取数据的函数  
def extract_data(filenames):  
    #验证文件是否存在  
    for f in filenames:  
        if not tf.gfile.Exists(f):  
            raise ValueError('Failed to find file: ' + f)  
    #读取数据  
    labels = None  
    images = None  
  
    for f in filenames:  
        bytestream=open(f,'rb')   
        #读取数据，首先将数据集中的数据读取进来作为buf  
        buf = bytestream.read(TRAIN_NUM * (IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS+LABEL_SIZE))  
        #把数据流转化为np的数组,为什么要转化为np数组呢，因为array数组只支持一维操作，为了满足我们的操作需求，我们利用np.frombuffer()将buf转化为numpy数组现在data的shape为（30730000，），3073是3*1024+1得到的，3个channel（r，g，b），每个channel有1024=32*32个信息，再加上 1 个label  
  
        data = np.frombuffer(buf, dtype=np.uint8)  
  
        #改变数据格式,将shape从原来的（30730000，）——>为（10000，3073）  
        data = data.reshape(TRAIN_NUM,LABEL_SIZE+IMAGE_SIZE* IMAGE_SIZE* NUM_CHANNELS)  
  
        #分割数组,分割数组，np.hsplit是在水平方向上，将数组分解为label_size的一部分和剩余部分两个数组，在这里label_size=1，也就是把标签label给作为一个数组单独切分出来如果你对np.split还不太了解，可以自行查阅一下，此时label_images的shape应该是这样的[array([.......]) , array([.......................])]     
        labels_images = np.hsplit(data, [LABEL_SIZE])  
  
        label = labels_images[0].reshape(TRAIN_NUM)#此时labels_images[0]就是我们上面切分数组得到的第一个数组，在这里就是label数组，这时的shape为array([[3] , [6] , [4] , ....... ,[7]])，我们把它reshape（）一下变为了array([3 , 6 , ........ ,7])    
  
        image = labels_images[1].reshape(TRAIN_NUM,IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)#此时labels_image[1]就是我们上面切分数组的剩余部分，也就是图片部分我们把它reshape（）为（10000，32，32，3）   
  
        if labels == None:  
            labels = label  
            images = image  
        else:  
            #合并数组，不能用加法  
            labels = np.concatenate((labels,label))  
            images = np.concatenate((images,image))  
  
    images = (images - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH  
  
    return labels,images  
  
#定义提取训练数据函数  
def extract_train_data(files_dir):  
    #获得训练数据  
    filenames = [os.path.join(files_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]  
    return extract_data(filenames)  

#定义提取测试数据函数      
def extract_test_data(files_dir):  
    #获得测试数据  
    filenames = [os.path.join(files_dir, 'test_batch.bin'),]  
    return extract_data(filenames)  
     
#把稠密数据label[1,5...]变为[[0,1,0,0...],[...]...]  
def dense_to_one_hot(labels_dense, num_classes):  
    #数据数量,np.shape[0]返回行数，对于一维数据返回的是元素个数,如果读取了5个文件的所有训练数据，那么现在的num_labels的值应该是50000  
    num_labels = labels_dense.shape[0]  
  
    #生成[0,1,2...]*10,[0,10,20...],之所以这样子是每隔10个数定义一次值，比如第0个，第11个，第22个......的值都赋值为1  
    index_offset = np.arange(num_labels) * num_classes  
  
    #初始化np的二维数组,一个全0，shape为(50000,10)的数组  
    labels_one_hot = np.zeros((num_labels, num_classes))  
  
    #相对应位置赋值变为[[0,1,0,0...],[...]...],np.flat将labels_one_hot砸平为1维  
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1  
  
    return labels_one_hot  

#定义cifar10数据集类  
class Cifar10DataSet(object):  
    """docstring for Cifar10DataSet"""  
    def __init__(self,data_dir):  
        super(Cifar10DataSet, self).__init__()  
        self.train_labels,self.train_images = extract_train_data(os.path.join(data_dir,'cifar10/cifar-10-batches-bin'))  
        self.test_labels,self.test_images = extract_test_data(os.path.join(data_dir,'cifar10/cifar-10-batches-bin'))  
          
        print self.train_labels.size  
  
        self.train_labels = dense_to_one_hot(self.train_labels,NUM_CLASSES)  
        self.test_labels = dense_to_one_hot(self.test_labels,NUM_CLASSES)  
  
        #epoch完成次数  
        self.epochs_completed = 0  
        #当前批次在epoch中进行的进度  
        self.index_in_epoch = 0  
  
  
    def next_train_batch(self,batch_size):  
        #起始位置  
        start = self.index_in_epoch  
        self.index_in_epoch += batch_size  
        #print "self.index_in_epoch: ",self.index_in_epoch  
        #完成了一次epoch  
        if self.index_in_epoch > TRAIN_NUMS:  
            #epoch完成次数加1,50000张全部训练完一次，那么没有数据用了怎么办，采取的办法就是将原来的数据集打乱顺序再用  
            self.epochs_completed += 1  
            #print "self.epochs_completed: ",self.epochs_completed  
            #打乱数据顺序，随机性  
            perm = np.arange(TRAIN_NUMS)  
            np.random.shuffle(perm)  
            self.train_images = self.train_images[perm]  
            self.train_labels = self.train_labels[perm]  
            start = 0  
            self.index_in_epoch = batch_size  
            #条件不成立会报错  
            assert batch_size <= TRAIN_NUMS  
  
        end = self.index_in_epoch  
        #print "start,end: ",start,end  
  
        return self.train_images[start:end], self.train_labels[start:end]  
  
    def test_data(self):  
        return self.test_images,self.test_labels  

def main():  
    cc = Cifar10DataSet('../data/')  
    cc.next_train_batch(100)  
  
if __name__ == '__main__':  
    main()  
