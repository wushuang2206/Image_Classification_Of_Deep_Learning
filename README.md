# 深度学习、图像处理领域必学经典算法：LeNet、AlexNET、GoogLeNet、VGG、ResNet算法的实现

## 实验结果

先上结果：所有结果均在[Kaggle](https://www.kaggle.com/)平台上运行获得的，还有一个免费试用GPU的平台是[Google colab](https://colab.research.google.com/)。强烈推荐，笔记本配置不够的同学使用这两个平台进行学习，Kaggle每周固定时长的，一般为36到40多小时。colab检测到你经常在用GPU的话会不给你分配GPU，他会首先将GPU分配给用的少的账户。

* 各个模型的准确率 acc：  
  
  * LeNet（cifar10）：0.6413（500次迭代，模型好像还没收敛）
  
  * AlexNet：0.6594（289次迭代）
  
  * GoogLeNet：0.7617（127次迭代）  
  
  * VGG16：0.8172（116次迭代）  
  
  * VGG19：0.8203（72次迭代）  
  
  * Res18：0.7141（44次迭代）  
  
  * Res34：0.7336（169次迭代）  
  
  * Res50：0.7093（115次迭代）  
  
  * Res101：0.6656（100次迭代，acc还呈上升趋势）  
  
  * Res152：0.6500（72次迭代）  

* 其中，ResNet系列普遍都收敛的比较快，几乎都在10次迭代就达到acc：0.5；30~40次迭代达到acc：0.6，后面就出现了波动现象，而且波动范围比较大，有时甚至会出现acc下降0.3左右。个人观点是ResNet算法的梯度是比较大的，模型调整得比较快，但同时有个问题是调整过快容易在最优值附近震荡。还有的话，我做的任务是比较简单的（10分类任务），用ResNet系列的模型就有种大材小用的感觉，就是分类任务很简单，复杂的模型用在简单的问题上效果并不好。后续我也做了实验，使用残差结构搭建10层、14层、16层、26层的模型分别得到的acc（迭代均少于100次，大概在50到80次之间就可得到最优值）为0.7094，0.7383，0.7406，0.7289。从数据上看，可知在一个比较浅层的网络（如14、16层），它的效果已经是可以达到0.74左右了，而后面再加深模型层数，像Res34模型一直在最优值左右震荡，准确率上升缓慢直到169次才达到最好的结果。而在后面Res50、Res101、Res152可以看到训练速度依旧是挺快的，但是准确率下降得明显，从0.74一直下降到0.65，就是模型复杂学到的东西超过这个10分类任务需要的，致使分类错误率上升了。

* 同时，比较惊讶的是所有结果中，VGG网络的效果是最好的，在ImageNet的比赛中GoogLeNet和ResNet的效果都是比VGG要好的。这个我个人猜测是我使用的训练数据量不够，还有实验过程中比没有对数据进行任何数据增强的操作，还有一个原因可能是我将图片resize成了112$\times$112（问就是我内存不够，为了训练数据集尽可能地大，只能将尺寸降下来），而原论文算法中用的都是224$\times$224的。

## 数据集

* ImageNet小部分数据集（17各类别）：images_set文件夹为ImageNet2012数据集的一小部分，共有17类图片数据，每类图片有1000到1500张不等。对应为文件夹下每一个子文件夹为1类图片数据，子文件夹名字为类名，后面的数字为图片的数量。
  
  * ImageNet更多数据推荐去[官网](https://image-net.org/request)进行下载
  
  * 这里有一个专门针对ImageNet的[GitHub开源爬虫项目](https://github.com/mf1024/ImageNet-Datasets-Downloader)，也可以获得ImageNet数据
  
  * [Kaggle](https://www.kaggle.com/datasets/lijiyu/imagenet)上也有ImageNet的一些小数据集（7G），够我们做小项目用了

* cifar10数据集：这里原本是不用cifar数据的，但是主要是LeNet网络结构太简单了，模型的表达能力有限，在ImageNet数据上几乎不起作用，看个数据就知道了:用ImageNet数据集10个·类别训练LeNet网络（1000次迭代）训练和预测的acc依旧只有0.1左右。开始训练时也是0.1。

# Quick Start Examples

* 首先进入到项目所在的目录，检查环境和对相关的包是否满足版本要求
  
  可以直接使用`pip install -r requirements.txt`进行检查，不符合的话践行安装，符合的他会告诉你该包已经满足相应版本。

* 运行代码
  
  * 在cmd下运行
  
  ```python
  python run.py --model Res34 --optimizer adam --dataset images_set --project ./run --batch_size 128 --epochs 128 
  # model 可选Lenet5、Alexnet8、GoogLeNet、VGG16、VGG19、Res18、Res34、Res50、Res101、Res152
  # optimizer 优化器，可选adam、sgd、nadam、adagrad等等
  # dataset 可选images_set、cifar10
  # project 选择要将模型的checkpoints文件weights文件保存的目录
  # batch_size int 每个批次的大小
  # epochs int 迭代次数
  ```
  
  * IDLE中运行，如pycharm
    
    修改run.py中parse里的参数即可，各参数可选与cmd下运行一致。

## 相关论文参考文献

*     在最后，给大家附上相关论文的链接，其中LeNet有点久远了，感兴趣的可以自行到网上找。

* [**AlexNet论文链接：ImageNet Classification with Deep Convolutional Neural Networks**](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

* [**Dropout论文链接：Srivastava, Nitish, et al. "Dropout: a simple way to prevent neural networks from overfitting." Journal of Machine Learning Research 15.1 (2014): 1929-1958.**](https://dl.acm.org/doi/abs/10.5555/2627435.2670313)

* [**BN论文链接：Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015)**](https://arxiv.org/abs/1502.03167)

* **[GoogLeNet论文链接：Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.](https://arxiv.org/abs/1409.4842)**

* **[VGG论文链接：Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).](https://arxiv.org/abs/1409.1556)**

* [**ResNet论文链接：He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.**](https://arxiv.org/abs/1512.03385)

* [**YOLOv5项目链接：ultralytics/yolov5: YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite (github.com)**](https://github.com/ultralytics/yolov5)
