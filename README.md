 

# 生成对抗网络

在这本笔记本中，我们将建立一个针对MNIST数据集训练的生成对抗网络（GAN）。从此，我们将能够生成新的手写数字！

来自Ian Goodfellow和Yoshua Bengio实验室的其他人员在2014年首次报道了GAN（https://arxiv.org/abs/1406.2661 ）（https://arxiv.org/abs/1406.2661 ）。此后，GAN已经爆炸式爆发。以下是几个例子：

* [Pix2Pix]（https://affinelayer.com/pixsrv/ ）
* [CycleGAN]（https://github.com/junyanz/CycleGAN ）
* [整个列表]（https://github.com/wiseodd/generative-models ）

GANs背后的想法是，您有两个网络，一个生成器$ G $和一个鉴别器$ D $，彼此竞争。发生器使假数据传递给鉴别器。鉴别器还可以看到真实的数据，并预测其接收到的数据是真实还是假的。训练生成器来愚弄鉴别器，它希望输出尽可能靠近实际数据的数据。并且对该鉴别器进行训练以确定哪些数据是真实的，哪个是假的。最终发生的是，生成器学习使得不能从实际数据到鉴别器的数据。

![GAN diagram](assets/gan_diagram.png)

GAN的一般结构如上图所示，使用MNIST图像作为数据。潜在的样本是发生器用来构成假图像的随机矢量。随着生成器通过训练学习，它计算出如何将这些随机向量映射到可以欺骗鉴别器的可识别图像。

鉴别器的输出是S形函数，其中0表示假图像，1表示真实图像。如果您只对生成新图像感兴趣，则可以在训练后丢弃鉴别器。 

## 模型输入

首先我们需要为我们的图形创建输入。 我们需要两个输入，一个用于鉴别器，一个用于发生器。 这里我们将调用鉴别器输入`inputs_real`和生成器输入`inputs_z`。 我们将为每个网络分配适当的大小。

## 生成器网络

![GAN Network](assets/gan_network.png)

在这里我们将构建生成器网络。为了使这个网络成为通用函数逼近器，我们至少需要一个隐藏层。我们应该使用一个泄漏的ReLU来让梯度顺畅地向下流过层。一个泄漏的ReLU就像一个正常的ReLU，除了负输入值有一个小的非零输出。

#### 可变范围
这里我们需要使用`tf.variable_scope`有两个原因。首先，我们要确保所有的变量名都以`generator`开头。同样，我们将在鉴别器变量前面加上“discriminator”。稍后，当我们对不同的网络进行培训时，这将有所帮助。

我们可以使用`tf.name_scope`来设置名称，但是我们也想用不同的输入重新使用这些网络。对于生成器，我们要训练它，但是在训练和训练之后也可以从这个_sample中获取。鉴别器将需要在假和真实输入图像之间共享变量。所以我们可以使用 `tf.variable_scope`

要使用`tf.variable_scope`，你可以使用`with`语句：
```python
with tf.variable_scope（'scope_name'，reuse = False）：
    # code here
```

以下是[TensorFlow文档]（https://www.tensorflow.org/programmers_guide/variable_scope#the_problem ）中的更多内容，以便使用`tf.variable_scope`进一步了解。

#### Leaky ReLU
TensorFlow不提供泄露ReLUs的操作，所以我们需要做一个。为此，您可以从线性完全连接的层获取输出，并将其传递到`tf.maximum`。通常，参数“alpha”设置负值的输出值。所以，负输入（`x`）值的输出是`alpha * x`，而正的`x`的输出是`x`：
$$
f(x) = max(\alpha * x, x)
$$

####  Tanh输出
发现生成器的发电机输出功率为$tanh$。这意味着我们必须将MNIST图像重新缩放到-1和1之间，而不是0和1。


## Discriminator

鉴别器网络与发生器网络几乎完全相同，只是我们使用S形输出层。


## 鉴别器和发生器损失

现在我们需要计算损失，这有点棘手。对于鉴别器，总损耗是实际和假图像的损耗之和，`d_loss = d_loss_real + d_loss_fake`。 Sigmoid交叉熵的损失，我们可以用`tf.nn.sigmoid_cross_entropy_with_logits`获得。我们还将在“tf.reduce_mean”中包装，以获得批次中所有图像的平均值。所以损失会看起来像

```python
tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
```

对于真实的图像逻辑，我们将使用我们从上面的单元格中的鉴别器获得的`d_logits_real`。对于标签，我们希望它们是所有标签，因为这些都是真实的图像。为了帮助鉴别器更好地泛化，标签从1.0减少到0.9，例如使用参数`smooth`。这被称为标签平滑，通常与分类器一起使用以提高性能。在TensorFlow中，它看起来像`labels = tf.ones_like（tensor）*（1 - smooth）`

假数据的鉴别器丢失是类似的。逻辑是`d_logits_fake`，我们从发送器输出传递给鉴别器。这些假的对象与所有零的标签一起使用。请记住，我们希望鉴别器为真实图像输出1，为假图像输出0，所以我们需要设置损耗来反映出来。

最后，发电机损耗正在使用`d_logits_fake`，即伪图像逻辑。但是，现在的标签都是这些标签。发生器试图愚弄鉴别器，所以它想要鉴别器输出假图像。



## Optimizers

我们要分别更新生成器和鉴别器变量。因此，我们需要为每个部分获取变量，并为这两个部分构建优化器。要获得所有可修改的变量，我们使用`tf.trainable_variables（）`。这将创建我们在图中定义的所有变量的列表。

对于生成器优化器，我们只想要生成变量。我们过去的自己很好，并使用可变范围来使用`generator'启动我们所有的生成器变量名。所以，我们只需要从`tf.trainable_variables（）'中遍历列表，并保留以`generator`开头的变量。每个变量对象都有一个属性`name`，它将变量的名称保存为一个字符串（例如`var.name =='weights_0``）。

我们可以做一些类似于鉴别器的事情。鉴别器中的所有变量都以`discriminator'开始。

然后，在优化器中，我们将变量列表传递给`minimize`方法的`var_list`关键字参数。这告诉优化器仅更新列出的变量。像`tf.train.AdamOptimizer().minimize(loss, var_list=var_list)`一样，只会训练`var_list`中的变量。



