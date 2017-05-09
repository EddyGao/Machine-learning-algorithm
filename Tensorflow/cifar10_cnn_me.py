import tensorflow as tf
import cifar10_input_data

def compute_accuracy(test_xs,test_ys):
    y_pre = sess.run(prediction,feed_dict={xs:test_xs,ys:test_ys})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(test_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:test_xs,ys:test_ys})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.5)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

#define placeholder
xs = tf.placeholder(tf.float32, [None,32,32,3])
ys = tf.placeholder(tf.float32, [None,10])

#conv1 layer
W_conv1 = weight_variable([5,5,3,64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(xs,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#conv2 layer
W_conv2 = weight_variable([5,5,64,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#loacl3 layer full connected
W_lc3 = weight_variable([8*8*64,384])
b_lc3 = bias_variable([384])
h_pool2_flat = tf.reshape(h_pool2,[-1,8*8*64])
h_lc3 = tf.nn.relu(tf.matmul(h_pool2_flat,W_lc3)+b_lc3)

#local4 layer
W_lc4 = weight_variable([384,192])
b_lc4 = bias_variable([192])
h_lc4 = tf.nn.relu(tf.matmul(h_lc3,W_lc4)+b_lc4)

#sotfmax layer
W_softmax = weight_variable([192,10])
b_softmax = bias_variable([10])

prediction = tf.nn.softmax(tf.matmul(h_lc4,W_softmax)+b_softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#important step
sess = tf.Session()
sess.run(tf.initialize_all_variables())

#get cifar10 data_Set
cifar10_data_set = cifar10_input_data.cifar10_dataset()
test_images , test_labels = cifar10_data_set.test_data()

for step in range(1000):
    train_images, train_labels = cifar10_data_set.next_batch_data(128)
    sess.run(train_step,feed_dict={xs:train_images,ys:train_labels})
    if step%50 == 0:
        print (compute_accuracy(test_images,test_labels))
