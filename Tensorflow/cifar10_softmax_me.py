import tensorflow as tf
import cifar10_input_data

xs = tf.placeholder(tf.float32 , [None,3072])
W = tf.Variable(tf.zeros([3072 , 10]))
b = tf.Variable(tf.zeros([10])+0.01)
y = tf.nn.softmax(tf.matmul(xs , W) + b)
ys = tf.placeholder(tf.float32 , [None,10])

cross_entropy = -tf.reduce_sum(ys*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

cifar10_data = cifar10_input_data.cifar10_dataset()

correct_prediction = tf.equal(tf.argmax(y,1) , tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction , 'float'))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(2000):
    train_image_data , train_label_data = cifar10_data.next_batch_data(128)
    train_image_data = train_image_data.reshape([-1,3072])

    test_imgae_data , test_label_data =cifar10_data.test_data()
    test_imgae_data = test_imgae_data.reshape([-1,3072])

    sess.run(train_step,feed_dict={xs:train_image_data , ys:train_label_data})
    if i%50 == 0:
        print sess.run(accuracy,feed_dict={xs:test_imgae_data , ys:test_label_data})
