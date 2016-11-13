>本次课我们利用上节课定义好的函数来构建我们的网络

>首先呢，我们定义一下输入的`placeholder`
>
<pre><code> xs=tf.placeholder(tf.float32,[None,784])
 ys=tf.placeholder(tf.float32,[None,10])</code></pre>
>我们还定义了`dropout`的`placeholder`，它是解决过拟合的有效手段
<pre><code> keep_prob=tf.placeholder(tf.float32)</code></pre>



>接着呢，我们需要处理我们的`xs`，把`x`s的形状变成`[-1,28,28,1]`，-1代表先不考虑输入的图片例子多少这个维度，后面的1是channel的数量，因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3。
<pre><code>x_image=tf.reshape(xs,[-1,28,28,1])</code></pre>


>接着我们定义第一层卷积,先定义本层的`Weights`,本层我们的卷积核patch的大小是5x5，因为黑白图片channel是1所以输入是1，输出是32个featuremap
<pre><code>W_conv1=weight_variable([5,5,1,32])</code></pre>

> 接着定义`bias`，它的大小是32个长度，因此我们传入它的`shape`为`[32]`
<pre><code>b_conv1=bias_variable([32])</code></pre>

>  定义好了`Weights`和`bias`，我们就可以定义卷积神经网络的第一个卷积层`h_conv1=conv2d(x_image,W_conv1)+b_conv1`,同时我们对`h_conv1`进行非线性处理，也就是激活函数来处理喽，这里我们用的是`tf.nn.relu`（修正线性单元）来处理，要注意的是，因为采用了`SAME`的padding方式，输出图片的大小没有变化依然是28x28，只是厚度变厚了，因此现在的输出大小就变成了28x28x32

<pre><code>h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)</code></pre>

>最后我们再进行`pooling`的处理就ok啦，经过`pooling`的处理，输出大小就变为了14x14x32
<pre><code>h_pool=max_pool_2x2(h_conv1)</code></pre>

> 接着呢，同样的形式我们定义第二层卷积，本层我们的输入就是上一层的输出，本层我们的卷积核patch的大小是5x5，有32个featuremap所以输入就是32，输出呢我们定为64
<pre><code>W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])</code></pre>
 
>  接着我们就可以定义卷积神经网络的第二个卷积层，这时的输出的大小就是14x14x64 
<pre><code>h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)</code></pre>

>最后也是一个pooling处理，输出大小为7x7x64

<pre><code>h_pool2=max_pool_2x2(h_conv2)</code></pre>


>好的，接下来我们定义我们的function layer,此时`weight_variable`的`shape`输入就是第二个卷积层的输出7764，输出我们继续扩大，定为1024
<pre><code>W_fc1=weight_variable([7764,1024]) b_fc1=bias_variable([1024])</code></pre>

>接着我们通过`tf.reshape()`将`h_pool2`的输出值从一个三维的变为一维的数据,-1表示先不考虑输入图片例子维度
<pre><code>#[n_samples,7,7,64]->>[n_samples,7764]
h_pool2_flat=tf.reshape(h_pool2,[-1,7764]) </code></pre>

>然后将`h_pool2_flat`与本层的`W_fc1`相乘（注意这个时候不是卷积了）
<pre><code>h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)</code></pre>


>如果我们考虑过拟合问题，可以加一个dropout的处理
<pre><code>h_fc1_drop=tf.nn.dropout(h_fc1,keep_drop)</code></pre>

>接下来我们就可以进行最后一层的构建了，好激动啊 输入是1024，最后的输出是10个(因为mnist数据集就是0-9十个类哈哈，所以当然是10个输出喽)，prediction就是我们最后的预测值
<pre><code>W_fc2=weight_variable([1024,10]) b_fc2=bias_variable([10])</code></pre>

>然后呢我们用softmax分类器（多分类，输出是各个类的概率）,对我们的输出进行分类
<pre><code>prediction=tf.nn.softmax(tf.matmul(h_fc1_dropt,W_fc2),b_fc2)</code></pre>

>接着呢我们利用交叉熵损失函数来定义我们的cost function
<pre><code>cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
reduction_indices=[1]))</code></pre>
 
>我们用`tf.train.AdamOptimizer()`作为我们的优化器进行优化，使我们的`cross_entropy`最小
<pre><code>train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)</code></pre>

>接着呢就是和之前视频讲的一样喽 定义回话 
<pre><code>sess=tf.Session()</code></pre>


>初始化变量
<pre><code>sess.run(tf.initialize_all_variables())</code></pre>


>好啦接着就是训练数据啦，我们假定训练`1000`步，每`50`步输出一下准确率，注意`sess.run()`时记得要用`feed_dict`给我们的众多placeholder喂数据哦哈哈 以上呢就是本次教程的例子代码
