首先介绍几个小概念：batch, iterations, epochs

**batch**

深度学习的优化算法，说白了就是梯度下降。每次的参数更新有两种方式。

第一种，遍历全部数据集算一次损失函数，然后算函数对各个参数的梯度，更新梯度。这种方法每更新一次参数都要把数据集里的所有样本都看一遍，计算量开销大，计算速度慢，不支持在线学习，这称为Batch gradient descent，批梯度下降。

另一种，每看一个数据就算一下损失函数，然后求梯度更新参数，这个称为随机梯度下降，stochastic gradient descent。这个方法速度比较快，但是收敛性能不太好，可能在最优点附近晃来晃去，hit不到最优点。两次参数的更新也有可能互相抵消掉，造成目标函数震荡的比较剧烈。

为了克服两种方法的缺点，现在一般采用的是一种折中手段，mini-batch gradient decent，小批的梯度下降，这种方法把数据分为若干个批，按批来更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性。另一方面因为批的样本数与整个数据集相比小了很多，计算量也不是很大。

基本上现在的梯度下降都是基于mini-batch的，所以深度学习框架的函数中经常会出现batch_size，就是指这个。 

关于如何将训练样本转换从batch_size的格式可以参考[训练样本的batch_size数据的准备](https://sthsf.github.io/wiki/deep%20learning/Tensorflow%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/Tensorflow%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86---%E8%AE%AD%E7%BB%83%E6%A0%B7%E6%9C%AC%E7%9A%84batch_size%E6%95%B0%E6%8D%AE%E7%9A%84%E5%87%86%E5%A4%87.html)。

**iterations**

iterations（迭代）：每一次迭代都是一次权重更新，每一次权重更新需要batch_size个数据进行Forward运算得到损失函数，再BP算法更新参数。1个iteration等于使用batchsize个样本训练一次。

**epochs**

epochs被定义为向前和向后传播中所有批次的单次训练迭代。这意味着1个周期是整个输入数据的单次向前和向后传递。简单说，epochs指的就是训练过程中数据将被“轮”多少次，就这样。

举个例子

训练集有1000个样本，batchsize=10，那么： 

训练完整个样本集需要： 

100次iteration，1次epoch。

具体的计算公式为： 

one epoch = numbers of iterations = N = 训练样本的数量/batch_size



首先，为什么需要有 Batch_Size 这个参数？

Batch 的选择，首先决定的是下降的方向。如果数据集比较小，完全可以采用全数据集 （ Full Batch Learning ）的形式，这样做至少有 2 个好处：其一，由全数据集确定的方向能够更好地代表样本总体，从而更准确地朝向极值所在的方向。其二，由于不同权重的梯度值差别巨大，因此选取一个全局的学习率很困难。 Full Batch Learning 可以使用Rprop只基于梯度符号并且针对性单独更新各权值。

对于更大的数据集，以上 2 个好处又变成了 2 个坏处：其一，随着数据集的海量增长和内存限制，一次性载入所有的数据进来变得越来越不可行。其二，以 Rprop 的方式迭代，会由于各个 Batch 之间的采样差异性，各次梯度修正值相互抵消，无法修正。这才有了后来RMSProp 的妥协方案。

既然 Full Batch Learning 并不适用大数据集，那么走向另一个极端怎么样？

所谓另一个极端，就是每次只训练一个样本，即 Batch_Size = 1。这就是在线学习（Online Learning）。线性神经元在均方误差代价函数的错误面是一个抛物面，横截面是椭圆。对于多层神经元、非线性网络，在局部依然近似是抛物面。使用在线学习，每次修正方向以各自样本的梯度方向修正，横冲直撞各自为政，难以达到收敛。如图所示：

可不可以选择一个适中的 Batch_Size 值呢？

当然可以，这就是批梯度下降法（Mini-batches Learning）。因为如果数据集足够充分，那么用一半（甚至少得多）的数据训练算出来的梯度与用全部数据训练出来的梯度是几乎一样的。

在合理范围内，增大 Batch_Size 有何好处？

- 内存利用率提高了，大矩阵乘法的并行化效率提高。
- 跑完一次 epoch（全数据集）所需的迭代次数减少，对于相同数据量的处理速度进一步加快。
- 在一定范围内，一般来说 Batch_Size 越大，其确定的下降方向越准，引起训练震荡越小。

盲目增大 Batch_Size 有何坏处？

- 内存利用率提高了，但是内存容量可能撑不住了。
- 跑完一次 epoch（全数据集）所需的迭代次数减少，要想达到相同的精度，其所花费的时间大大增加了，从而对参数的修正也就显得更加缓慢。
- Batch_Size 增大到一定程度，其确定的下降方向已经基本不再变化。





**时间序列的数据的batch_size的数据准备**

对于时间序列的数据集，模型的输入格式为[batch_size, seq_length, input_dim], 其中，batch_size表示一个batch中的样本的个数，seq_length表示序列的长度，input_dim表示输入样本的维度。

那实际工程下如何取准备这些数据呢，我们假设样本训练集[x1,x2,x3,...,xdatalength]的长度为data_length

**法一**

第一种就是先按照seq_length这个窗口进行截取，然后按照bacth_size个数据向后依次截取，则总的迭代次数iterations = (data_length - seq_length) // batch_size, 则一个batch中的第一行数据可以表示为[x1,x2,...,xseqlength],第二行的数据可以表示为[xseqlength+1,xseqlength+2,...,xseqlength+xseqlength+1], 最后一行数据可以表示为[xbatchsize]

程序模拟

假设序列为:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

```python
import numpy as np

batch_size = 4
seq_length = 3
raw_data = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

def get_batch(raw_data, batch_size, seq_length):
    data = np.array(raw_data)
    data_length = data.shape[0]
    num_steps = data_length - seq_length + 1
    iterations = num_steps // batch_size
    xdata=[]
    ydata=[]
    for i in range(num_steps-1):
        xdata.append(data[i:i+seq_length])
        ydata.append(data[i+1:i+1+seq_length])

    for batch in range(iterations):
        x = np.array(xdata)[batch * batch_size: batch * batch_size + batch_size, :]
        y = np.array(xdata)[batch * batch_size + 1: batch * batch_size + 1 + batch_size, :]
        yield x, y
```

输出的训练集数据的格式为：

```python
x1: [[1 2 3]
     [2 3 4]
     [3 4 5]
     [4 5 6]]
y1: [[2 3 4]
     [3 4 5]
     [4 5 6]
     [5 6 7]]
x2: [[ 5  6  7]
     [ 6  7  8]
     [ 7  8  9]
     [ 8  9 10]]
y2: [[ 6  7  8]
     [ 7  8  9]
     [ 8  9 10]
     [ 9 10 11]]
x3: [[ 9 10 11]
     [10 11 12]
     [11 12 13]
     [12 13 14]]
y3: [[10 11 12]
     [11 12 13]
     [12 13 14]
     [13 14 15]]
x4: [[13 14 15]
     [14 15 16]
     [15 16 17]
     [16 17 18]]
y4: [[14 15 16]
     [15 16 17]
     [16 17 18]
     [17 18 19]]
```

**法二**

第二种方法以bacth_size和seq_length为基础一个batch中应该包含的数据个数为batch_size * seq_length个数据，那么iterations= data_length//(batch_size * seq_length).

\- step1、利用numpy中的矩阵技巧，先将序列reshpe成[batch_size, seq_length* iterations]的形状，

\- step2、然后利用for循环将reshape后的数据截取成若干个batch。

程序模拟

```python
import numpy as np

batch_size = 4
seq_length = 3
raw_data = [1,2,3,4,5,6,7,8,9,10,11,12,13,
            14,15,16,17,18,19,20, 21, 22, 
            23, 24, 25, 26, 27, 28, 29, 30, 
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

def get_batch(raw_data, batch_size, seq_length):
    data = np.array(raw_data)
    data_length = data.shape[0]
    iterations = (data_length - 1) // (batch_size * seq_length)
    round_data_len = iterations * batch_size * seq_length
    xdata = data[:round_data_len].reshape(batch_size, iterations*seq_length)
    ydata = data[1:round_data_len+1].reshape(batch_size, iterations*seq_length)

    for i in range(iterations):
        x = xdata[:, i*seq_length:(i+1)*seq_length]
        y = ydata[:, i*seq_length:(i+1)*seq_length]
        yield x, y
```

step1 产生的结果为：

```python
x：
[[ 1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18]
 [19 20 21 22 23 24 25 26 27]
 [28 29 30 31 32 33 34 35 36]]
对应的标签y为：
[[ 2  3  4  5  6  7  8  9 10]
 [11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28]
 [29 30 31 32 33 34 35 36 37]]

step2 生成的结果为：
x1: [[ 1  2  3]
     [10 11 12]
     [19 20 21]
     [28 29 30]]
y1: [[ 2  3  4]
     [11 12 13]
     [20 21 22]
     [29 30 31]]
x2: [[ 4  5  6]
     [13 14 15]
     [22 23 24]
     [31 32 33]]
y2: [[ 5  6  7]
     [14 15 16]
     [23 24 25]
     [32 33 34]]
x3: [[ 7  8  9]
     [16 17 18]
     [25 26 27]
     [34 35 36]]
y3: [[ 8  9 10]
     [17 18 19]
     [26 27 28]
     [35 36 37]]
```

