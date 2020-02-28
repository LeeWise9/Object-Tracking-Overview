# 目标追踪<br>
这是一篇关于目标追踪的综述，主要总结现有的常用数据集、目标追踪算法、研究热点等。<br>

## 目标跟踪：<br>
任务背景：目标跟踪通常指单目标跟踪。跟踪目标由第一帧给定，可由人工标注或检测算法获取。跟踪算法再在后续帧紧跟此目标。<br>

<p align="center">
	<img src="https://pic3.zhimg.com/80/v2-518e71f77d3bf360a2728f32e63cdc6f_720w.jpg" alt="Sample"  width="500">
</p>

技术特点：第一帧的BBox由检测算法提供或由人工标注<br>
技术难点（吴毅）：<br>
* 外观变形，光照变化，快速运动和运动模糊，背景相似干扰等<br>
<p align="center">
	<img src="https://pic4.zhimg.com/80/v2-1169ca84d569b5f8aff728d0de563869_720w.jpg" alt="Sample"  width="500">
</p>

* 平面外旋转，平面内旋转，尺度变化，遮挡和出视野等情况等<br>
<p align="center">
	<img src="https://pic3.zhimg.com/80/v2-3db98542589ec7abf17d52c20bcbdf12_720w.jpg" alt="Sample"  width="500">
</p>


## 常用数据集：<br>
OTB50（OTB-2013），OTB100（OTB-2015）<br>
官方测试代码与序列：[Visual Tracker Benchmark](http://cvlab.hanyang.ac.kr/tracker_benchmark/)<br>

VOT竞赛数据库：VOT2015，VOT2016<br>
VOT挑战赛平台与数据：[VOT Challenge | Challenges](http://votchallenge.net/challenges.html)<br>

### OTB和VOT区别：<br>
* 评价指标不同；<br>
* 图像素质不同：OTB含25%灰度序列，VOT全彩色序列，VOT序列分辨率更高；<br>
* 初始方法不同：OTB有随机帧、随机初始化方法；VOT是首帧初始化，每次跟踪失败5帧之后重新初始化；<br>
* VOT强调跟踪和检测应该兼并，跟踪过程中会多次初始化tracker；<br>
* VOT数据库每年更新，包括重新标注、改变评价指标等。<br>


## 2012年及之前相关工作：<br>
29个顶尖的tracker在OTB100数据集上的表现：<br>
按性能排序：Struck>SCM>ASLA；按速度排序：CSK(362fps)>CT(64fps)>TLD(28)。<br>

<p align="center">
	<img src="https://pic3.zhimg.com/80/v2-63db35d3d2f57965cea3c7226b759e60_720w.jpg" alt="Sample"  width="500">
</p>

<p align="center">
	<img src="https://pic3.zhimg.com/80/v2-92fe48e735d4978c81073808a4ae1585_720w.jpg" alt="Sample"  width="500">
</p>


## 2013-2017 SOTA：<br>
Struck，KCF，CN，DSST，SAMF，LCT，HCF，SRDCF<br>





## 方法分类：<br>
* 生成（generative）模型方法。典型算法[ASMS](https://github.com/vojirt/asms)（125fps）。<br>
* 判别（discriminative）模型方法。典型算法Struck（20fps）和TLD（28fps）。<br>
* 深度学习（Deep ConvNet based）类方法。典型算法：[MDNet](http://cvlab.postech.ac.kr/research/mdnet/)，[TCNN](http://www.votchallenge.net/vot2016/download/44_TCNN.zip)，[SiamFC](http://www.robots.ox.ac.uk/~luca/siamese-fc.html)，[SiamFC-R](http://www.iqiyi.com/w_19ruirwrel.html#vfrm=8-8-0-1)。<br>
* 相关滤波（correlation filter）类方法。典型算法：CSK（362fps），KCF（172fps），DCF（292fps），CN（152fps）。<br>


### 生成模型法<br>
生成模型法用当前帧目标特征建模，在下一帧寻找与模型最相似的区域。主要方法有卡尔曼滤波，粒子滤波，mean-shift等。<br>
生成模型的一个简单例子：从当前帧得知目标区域 80% 是红色、20% 是绿色，则在下一帧搜索寻找最符合这个颜色比例的区域。<br>
ASMS 与 DAT 都是仅颜色特征的算法而且速度很快，分别是 VOT2015 的第 20名 和 14 名，在 VOT2016 分别是 32 名和 31 名(中等水平)。<br>
ASMS 是 VOT2015 官方推荐的实时算法，平均帧率125FPS，在经典 mean-shift 框架下加入了尺度估计、经典颜色直方图特征，加入了两个先验(尺度不剧变+可能偏最大)作为正则项，和反向尺度一致性检查。


### 判别模型法<br>
判别模型法以目标区域为正样本，背景区域为负样本，使用机器学习方法训练分类器，下一帧用训练好的分类器寻找最优区域。<br>
OTB50 中大部分方法都是这一类。<br>
<p align="center">
	<img src="https://pic4.zhimg.com/80/v2-d2c2473036eda3641b1b689496b79609_720w.jpg" alt="Sample"  width="500">
</p>

分类器采用机器学习，训练中用到了背景信息，这样分类器就能专注区分前景和背景，所以判别类方法普遍都比生成类好。

比如，训练时 tracker 得知目标 80% 是红色，20% 是绿色，且背景中有橘红色，这样的分类器获得了更多信息，效果也相对更好。

Tracking-by-Detection 和检测算法非常相似。跟踪中为了尺度自适应也需要多尺度遍历搜索，区别仅在于跟踪算法对特征和在线机器学习的速度要求更高，检测范围和尺度更小。

大多数情况检测识别算法复杂度比较高，这时候用复杂度较低的跟踪算法更合适，只需在跟踪失败 (drift) 或一定间隔以后再次检测初始化 tracker 即可。毕竟 FPS 是追踪类算法最重要的指标之一。

Struck 和 TLD 都能实时跟踪，Struck 是 2012 年之前最好的方法，TLD是经典 long-term 的代表。


### 深度学习方法<br>
深度学习端到端的优势在目标跟踪方向体现不明显，还没和相关滤波类方法拉开差距，普遍面临着速度慢的问题。

另一个需要注意的问题是目标跟踪的数据库都没有严格的训练集和测试集，需要注意训练集与测试集有没有相似序列。直到 VOT2017 官方才指明要限制训练集，不能用相似序列训练模型。

该方法领域值得关注的研究包括但不限于：
* [Winsty](http://www.winsty.net/) 的系列研究；<br>
* VOT2015 的冠军 [MDNet](http://cvlab.postech.ac.kr/research/mdnet/)；<br>
* VOT2016 的冠军 [TCNN](http://www.votchallenge.net/vot2016/download/44_TCNN.zip)；<br>
* VOT2016 成绩优异的基于 ResNet 的 [SiamFC-R](http://www.iqiyi.com/w_19ruirwrel.html#vfrm=8-8-0-1)；<br>
* 速度突出的 的[SiamFC](http://www.robots.ox.ac.uk/~luca/siamese-fc.html)（80FPS）；<br>
* 速度更快的 GOTURN（100FPS），牺牲性能换取速度。<br>
（这些方法可以在王强维护的 [benchmark_results](https://github.com/foolwood/benchmark_results) 中找到）


### 相关滤波方法<br>
相关滤波类方法 correlation filter 简称 CF，或 discriminative correlation filter 简称 DCF。

相关滤波法的发展过程是速度与精度权衡的过程：从 MOSSE(615FPS) 到 CSK(362FPS) 再到 KCF(172FPS)，DCF(292FPS)，CN(152FPS)，CN2(202FPS)，速度越来越慢，效果越来越好，且始终保持在高速水平。

* MOSSE 是单通道灰度特征的相关滤波，因使用单通道图片，计算速度极快。<br>
* CSK 和 KCF 是牛津大学 [Henriques J F](http://www.robots.ox.ac.uk/~joao/index.html#) 的先后两篇研究成果，对后续研究产生了深远影响。CSK 在 MOSSE 的基础上扩展了密集采样和 kernel-trick ；KCF 在 CSK 的基础上扩展了多通道梯度的 HOG 特征。<br>
* 林雪平大学 Martin Danelljan 用多通道颜色特征 Color Names (CN) 扩展 CSK 得到了不错的效果，算法简称 [CN](http://www.cvl.isy.liu.se/research/objrec/visualtracking/colvistrack/index.html)。

HOG 是梯度特征，CN 是颜色特征，两者互补常搭配使用。

为解决尺度变化导致的跟踪目标丢失，2014 年前后有学者继续改进，添加了尺度自适应方法。<br>
* 浙大 Yang Li 的工作 [SAMF](https://github.com/ihpdep/samf) ，在 KCF 的基础上用了 HOG+CN 特征，使用平移滤波器在多尺度缩放的图像块上进行目标检测，取响应最大的平移位置及所在尺度。<br>
* Martin Danelljan 的 [DSST](http://www.cvl.isy.liu.se/research/objrec/visualtracking/scalvistrack/index.html)，使用了 HOG 特征，同时使用了平移滤波和尺度滤波。后续还研究出了加速版本 fDSST。<br>

上述两者有如下区别：<br>
* SAMF 有 7 个尺度，DSST 有 33 个尺度；<br>
* SAMF 同时优化平移和尺度，DSST 分步优化：先检测最佳平移再检测最佳尺度；<br>
* SAMF 只需一个滤波器，每个尺度检测提取一次特征和 FFT，在图像较大时计算量比 DSST 高；<br>
* DSST 分步优化可采用不同的方法和特征，需要额外训练一个滤波器，每帧尺度检测需采样 33 个图像块并分别计算特征、加窗、FFT 等，尺度滤波器比平移滤波器慢很多。<br>

为改善对快速变形和快速运动目标的追踪效果，2015 年前后有学者继续改进，着重解决边界效应问题。























## 参考文献：<br>
* 作者：YaqiLYU，链接：https://www.zhihu.com/question/26493945/answer/156025576
* Wu Y, Lim J, Yang M H. Online object tracking: A benchmark [C]// CVPR, 2013.<br>
* Wu Y, Lim J, Yang M H. Object tracking benchmark [J]. TPAMI, 2015.<br>
* Yilmaz A, Javed O, Shah M. Object tracking: A survey [J]. CSUR, 2006.<br>
* Kristan M, Pflugfelder R, Leonardis A, et al. The visual object tracking vot2013 challenge results [C]// ICCV, 2013.<br>
* Kristan M, Pflugfelder R, Leonardis A, et al. The Visual Object Tracking VOT2014 Challenge Results [C]// ECCV, 2014.<br>
* Kristan M, Matas J, Leonardis A, et al. The visual object tracking vot2015 challenge results [C]// ICCV, 2015.<br>
* Kristan M, Ales L, Jiri M, et al. The Visual Object Tracking VOT2016 Challenge Results [C]// ECCV, 2016.<br>
* Vojir T, Noskova J, Matas J. Robust scale-adaptive mean-shift for tracking [J]. Pattern Recognition Letters, 2014.<br>
* Hare S, Golodetz S, Saffari A, et al. Struck: Structured output tracking with kernels [J]. IEEE TPAMI, 2016. <br>
* Kalal Z, Mikolajczyk K, Matas J. Tracking-learning-detection [J]. IEEE TPAMI, 2012.<br>
* Nam H, Han B. Learning multi-domain convolutional neural networks for visual tracking [C]// CVPR, 2016.<br>
* Nam H, Baek M, Han B. Modeling and propagating cnns in a tree structure for visual tracking. arXiv preprint arXiv:1608.07242, 2016.<br>
* Bertinetto L, Valmadre J, Henriques J F, et al. Fully-convolutional siamese networks for object tracking [C]// ECCV, 2016.<br>
* Held D, Thrun S, Savarese S. Learning to track at 100 fps with deep regression networks [C]// ECCV, 2016.<br>
* Bolme D S, Beveridge J R, Draper B A, et al. Visual object tracking using adaptive correlation filters [C]// CVPR, 2010.<br>
* Henriques J F, Caseiro R, Martins P, et al. Exploiting the circulant structure of tracking-by- detection with kernels [C]// ECCV, 2012.<br>
* Henriques J F, Rui C, Martins P, et al. High-Speed Tracking with Kernelized Correlation Filters [J]. IEEE TPAMI, 2015.<br>
* Danelljan M, Shahbaz Khan F, Felsberg M, et al. Adaptive color attributes for real-time visual tracking [C]// CVPR, 2014.<br>
* Li Y, Zhu J. A scale adaptive kernel correlation filter tracker with feature integration [C]// ECCV, 2014.<br>
* Danelljan M, Häger G, Khan F, et al. Accurate scale estimation for robust visual tracking [C]// BMVC, 2014.<br>
* Danelljan M, Hager G, Khan F S, et al. Discriminative Scale Space Tracking [J]. IEEE TPAMI, 2017.<br>





