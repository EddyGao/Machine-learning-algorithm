# eddy
eddy_file

Tensorflow 是非常重视结构的, 我们得建立好了神经网络的结构, 才能将数字放进去,
运行这个结构.

这个例子简单的阐述了 tensorflow 当中如何用代码来运行我们搭建的结构.

首先, 我们这次需要加载 tensorflow 和 numpy 两个模块, 并且使用 numpy
来创建我们的数据.

{% highlight python %}
import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3
{% endhighlight %}

接着, 我们用 `tf.Variable` 来创建描述 `y` 的参数. 我们可以把 `y_data = x_data*0.1 + 0.3`
想象成 `y=Weights * x + biases`, 然后神经网络也就是学着把 Weights 变成 0.1, biases 变成 0.3.

{% highlight python %}
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases
{% endhighlight %}

接着就是计算 `y` 和 `y_data` 的误差:

{% highlight python %}
loss = tf.reduce_mean(tf.square(y-y_data))
{% endhighlight %}

反向传递误差的工作就教给`optimizer`了, 我们使用的误差专递方法是梯度下降法: `Gradient Descent`
让后我们使用 `optimizer` 来进行参数的更新.

{% highlight python %}
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
{% endhighlight %}

到目前为止, 我们只是建立了神经网络的结构, 还没有使用这个结构. 在使用这个结构之前,
我们必须先初始化所有之前定义的`Variable`, 所以这一步是很重要的!

{% highlight python %}
init = tf.initialize_all_variables()
{% endhighlight %}

接着,我们再创建会话 `Session`. 我们会在下一节中详细讲解 Session.
我们用 `Session` 来执行 `init` 初始化步骤. 并且,
用 `Session` 来 `run` 每一次 training 的数据. 逐步提升神经网络的预测准确性.

{% highlight python %}
sess = tf.Session()
sess.run(init)          # Very important

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
{% endhighlight %}
