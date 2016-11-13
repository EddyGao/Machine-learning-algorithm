# eddy
eddy_file
卷积神经网络：cnn

计算机视觉处理的飞跃提升，在图像和语音识别方面表现出了强大的优势，学习卷积神经网络之前，我们已经假设你对神经网络已经有了初步的了解，如果没有的话，可以去看看tensorflow第一篇视频教程哦~
卷积神经网络包含输入层、隐藏层和输出层，隐藏层又包含卷积层和pooling层，图像输入到卷积神经网络后通过卷积来不断的提取特征，每提取一个特征就会增加一个feturemap，所以会看到视频教程中的立方体不断的增加厚度，那么为什么厚度增加了但是却越来越瘦了呢，哈哈这就是pooling层的作用喽，pooling层也就是下采样，通常采用的是max_pooling和mean_pooling，因为参数太多喽，所以通过pooling来稀疏参数，使我们的网络不至于太复杂，好啦，既然你对卷积神经网络已经有了大概的了解，我们将通过代码来实现一个基于MNIST数据集的简单卷积神经网络。

基于上一次卷积神经网络的介绍，我们在代码中实现一个基于MNIST数据集的例子
首先我们导入
'''
import tensorflow as tf
'''
采用的数据集依然是tensorflow里面的mnist数据集，我们需要先导入它
'''
from tensorflow.examples.tutorials.mnist import input_data
'''
本次课程代码用到的数据集就是来自于它
'''
mnist=input_data.read_data_sets('MNIST_data',one_hot=true)
'''
接着呢，我们定义Weights变量，输入shape，返回变量的参数。其中我们使用tf.truncted_normal产生随机变量来进行初始化。

'''
def weight_variable(shape):
	inital=tf.truncted_normal(shape,stddev=0.1)
	return tf.Variable(initial)
'''
同样的定义biase变量，输入shape ,返回变量的一些参数。其中我们使用tf.constant常量函数来进行初始化。
'''
def bias_variable(shape):
	initial=tf.constant(0.1,shape=shape)
	return tf.Variable(initial)
'''
定义卷积，tf.nn.conv2d函数是tensoflow里面的二维的卷积函数，x是图片的所有参数，W是此卷积层的权重，然后定义步长strides=[1,1,1,1]值，strides[0]和strides[3]的两个1是默认值，中间两个1代表padding时在x方向运动一步，y方向运动一步，padding采用的方式是SAME。
'''
def conv2d(x,W)
	#strides[1,x_movement,y_movement,1]
	#Must have strides[0]=strides[3]=1
	return tf.nn.conv2d(x,W,strides=[1,1,1,1]，padding='SAME')
'''
接着定义池化pooling，为了得到更多的图片信息，padding时我们选的是一次一步，也就是strides[1]=strides[2]=1，这样得到的图片尺寸没有变化，而我们希望压缩一下图片也就是参数能少一些从而减小系统的复杂度，因此我们采用pooling来稀疏化参数，也就是卷积神经网络中所谓的下采样层。pooling 有两种，一种是最大值池化，一种是平均值池化，本例采用的是最大值池化tf.max_pool()。池化的核函数大小为2x2，因此ksize=[1,2,2,1]，步长为2，因此strides=[1,2,2,1]
'''
def max_poo_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1])
	
'''
好啦，如果你对本节课内容已经了解，下一次课我们将构建卷积神经网络的架构~

本次课我们利用上节课定义好的函数来构建我们的网络
首先呢，我们定义一下输入的placeholder
'''
xs=tf.placeholder(tf.float32,[None,784])#28x28
ys=tf.placeholder(tf.float32,[None,10])
'''
我们还定义了dropout的placeholder，它是解决过拟合的有效手段
'''
keep_prob=tf.placeholder(tf.float32)
'''
接着呢，我们需要处理我们的xs，把xs的形状变成[-1,28,28,1]，-1代表先不考虑输入的图片例子多少这个维度，后面的1是channel的数量，因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3。
'''
x_image=tf.reshape(xs,[-1,28,28,1])
'''
接着我们定义第一层卷积,先定义本层的Weights,本层我们的卷积核patch的大小是5x5，因为黑白图片channel是1所以输入是1，输出是32个featuremap
'''
W_conv1=weight_variable([5,5,1,32])
'''
接着定义bias，它的大小是32个长度，因此我们传入它的shape为[32]
'''
b_conv1=bias_variable([32])
'''
定义好了Weights和bias，我们就可以定义卷积神经网络的第一个卷积层h_conv1=conv2d(x_image,W_conv1)+b_conv1,同时我们对h_conv1进行非线性处理，也就是激活函数来处理喽，这里我们用的是tf.nn.relu（修正线性单元）来处理，要注意的是，因为采用了SAME的padding方式，输出图片的大小没有变化依然是28x28，只是厚度变厚了，因此现在的输出大小就变成了28x28x32
'''
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
'''
最后我们再进行pooling的处理就ok啦，经过pooling的处理，输出大小就变为了14x14x32
'''
h_pool=max_pool_2x2(h_conv1)
'''
接着呢，同样的形式我们定义第二层卷积，本层我们的输入就是上一层的输出，本层我们的卷积核patch的大小是5x5，有32个featuremap所以输入就是32，输出呢我们定为64
'''
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
'''
接着我们就可以定义卷积神经网络的第二个卷积层，这时的输出的大小就是14x14x64
'''
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
'''
最后也是一个pooling处理，输出大小为7x7x64
'''
h_pool2=max_pool_2x2(h_conv2)
'''

好的，接下来我们定义我们的function layer,此时weight_variable的shape输入就是第二个卷积层的输出7*7*64，输出我们继续扩大，定为1024
'''
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
'''
接着我们通过tf.reshape()将h_pool2的输出值从一个三维的变为一维的数据,-1表示先不考虑输入图片例子维度
'''
#[n_samples,7,7,64]->>[n_samples,7*7*64]
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
'''
然后将h_pool2_flat与本层的W_fc1相乘（注意这个时候不是卷积了）
'''
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
'''
如果我们考虑过拟合问题，可以加一个dropout的处理
'''
h_fc1_drop=tf.nn.dropout(h_fc1,keep_drop)
'''
接下来我们就可以进行最后一层的构建了，好激动啊
输入是1024，最后的输出是10个(因为mnist数据集就是0-9十个类哈哈，所以当然是10个输出喽)，prediction就是我们最后的预测值
'''
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
'''
然后呢我们用softmax分类器（多分类，输出是各个类的概率）,对我们的输出进行分类
'''
prediction=tf.nn.softmax(tf.matmul(h_fc1_dropt,W_fc2),b_fc2)
'''
接着呢我们利用交叉熵损失函数来定义我们的cost function
'''
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
'''

我们用tf.train.AdamOptimizer()作为我们的优化器进行优化，使我们的cross_entropy最小
'''
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
'''
接着呢就是和之前视频讲的一样喽
定义回话
'''
sess=tf.Session()
'''
初始化变量
'''
sess.run(tf.initialize_all_variables())
'''
好啦接着就是训练数据啦，我们假定训练1000步，每50步输出一下准确率，注意sess.run()时记得要用feed_dict给我们的众多placeholder喂数据哦哈哈
以上呢就是本次教程的例子代码
