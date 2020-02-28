# 目标跟踪<br>
这是一篇关于目标跟踪的综述，主要总结现有的常用数据集、目标跟踪算法、资源、研究热点等。<br>

## 目标跟踪<br>
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


## 常用数据集<br>
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


## 方法分类<br>
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
* 速度突出的 的 [SiamFC](http://www.robots.ox.ac.uk/~luca/siamese-fc.html)（80FPS）；<br>
* 速度更快的 GOTURN（100FPS），牺牲性能换取速度。<br>
（这些方法可以在王强维护的 [benchmark_results](https://github.com/foolwood/benchmark_results) 中找到）


### 相关滤波方法<br>
相关滤波类方法 correlation filter 简称 CF，或 discriminative correlation filter 简称 DCF，该方法相关研究对目标跟踪类方法产生了深远影响。

相关滤波法的发展过程是速度与精度权衡的过程：从 MOSSE(615FPS) 到 CSK(362FPS) 再到 KCF(172FPS)，DCF(292FPS)，CN(152FPS)，CN2(202FPS)，速度越来越慢，效果越来越好，且始终保持在高速水平。

<br>

## 发展梳理<br>
下面按时间顺序，以相关滤波为重点，梳理目标跟踪近几年的发展脉络。<br>

### 2012年及之前的工作：<br>
29个顶尖的tracker在OTB100数据集上的表现：<br>
按性能排序：Struck>SCM>ASLA；按速度排序：CSK(362fps)>CT(64fps)>TLD(28)。<br>

<p align="center">
	<img src="https://pic3.zhimg.com/80/v2-63db35d3d2f57965cea3c7226b759e60_720w.jpg" alt="Sample"  width="500">
</p>

<p align="center">
	<img src="https://pic3.zhimg.com/80/v2-92fe48e735d4978c81073808a4ae1585_720w.jpg" alt="Sample"  width="500">
</p>

<br>

### 2013-2016 相关滤波：<br>
这段时期是相关滤波方法快速发展的时期。<br>
* MOSSE ：单通道灰度特征的相关滤波，因使用单通道图片，计算速度极快。<br>
* CSK 和 KCF 是牛津大学 [Henriques J F](http://www.robots.ox.ac.uk/~joao/index.html#) 的先后两篇研究成果，对后续研究产生了深远影响。CSK 在 MOSSE 的基础上扩展了密集采样和 kernel-trick ；KCF 在 CSK 的基础上扩展了多通道梯度的 HOG 特征。<br>
* 林雪平大学 Martin Danelljan 用多通道颜色特征 Color Names (CN) 扩展 CSK 得到了不错的效果，算法简称 [CN](http://www.cvl.isy.liu.se/research/objrec/visualtracking/colvistrack/index.html)。

HOG 是梯度特征，CN 是颜色特征，两者常搭配使用。

<br>

### 2014 - 尺度自适应<br>
为解决尺度变化导致的跟踪目标丢失，2014 年前后有学者继续改进，添加了尺度自适应方法。
<p align="center">
	<img src="https://pic1.zhimg.com/80/v2-ceafcb41ac2fca6a3b001bd5c240c93e_720w.jpg" alt="Sample"  width="500">
</p>

* 浙江大学 Yang Li 的工作 [SAMF](https://github.com/ihpdep/samf) ，在 KCF 的基础上用了 HOG+CN 特征，使用平移滤波器在多尺度缩放的图像块上进行目标检测，取响应最大的平移位置及所在尺度。<br>
* Martin Danelljan 的 [DSST](http://www.cvl.isy.liu.se/research/objrec/visualtracking/scalvistrack/index.html)，使用了 HOG 特征，同时使用了平移滤波和尺度滤波。后续还研究出了加速版本 fDSST。<br>

上述两者有如下区别：<br>
* SAMF 有 7 个尺度，DSST 有 33 个尺度；<br>
* SAMF 同时优化平移和尺度，DSST 分步优化：先检测最佳平移再检测最佳尺度；<br>
* SAMF 只需一个滤波器，每个尺度检测提取一次特征和 FFT，在图像较大时计算量比 DSST 高；<br>
* DSST 分步优化可采用不同的方法和特征，需要额外训练一个滤波器，每帧尺度检测需采样 33 个图像块并分别计算特征、加窗、FFT 等，尺度滤波器比平移滤波器慢很多。<br>

<br>

### 2015 - 边界效应<br>
为改善对快速变形和快速运动目标的追踪效果，2015 年前后有学者继续改进，着重解决边界效应(Boundary Effets)问题。<br>
<p align="center">
	<img src="https://pic4.zhimg.com/80/v2-56155346ce01fb7037856683cd68a286_720w.jpg" alt="Sample"  width="500">
</p>

* Martin Danelljan 的 [SRDCF](http://www.cvl.isy.liu.se/research/objrec/visualtracking/regvistrack/index.html)。忽略了所有移位样本的边界部分像素，限制让边界附近滤波器系数接近 0。速度 167FPS，性能不如 KCF。<br>
<p align="center">
	<img src="https://pic2.zhimg.com/80/v2-c5bb2010d16e93c6b3661dc54e06684b_720w.jpg" alt="Sample"  width="500">
</p>

* Hamed Kiani 的 MOSSE 改进算法，基于灰度特征的 [CFLM](http://www.hamedkiani.com/cfwlb.html) 和基于 HOG 特征的 [BACF](http://www.hamedkiani.com/bacf.html)，采用较大尺寸检测图像块和较小尺寸滤波器来提高真实样本比例，采用 ADMM 迭代优化。BACF 性能超过 SRDCF，速度 35FPS。<br>
<p align="center">
	<img src="https://pic4.zhimg.com/80/v2-8b5a1516ecc6c2bf4782b99ab031373a_720w.jpg" alt="Sample"  width="500">
</p>

两个解决方案都用更大的检测及更新图像块，训练作用域比较小的相关滤波器。但是 SRDCF 的滤波器系数从中心到边缘平滑过渡到 0，而 CFLM 直接用 0 填充滤波器边缘。<br>

<br>

### 2015-2017 卷积特征<br>
[Martin Danelljan](http://www.cvl.isy.liu.se/research/objrec/visualtracking/) 结合深度特征和相关滤波方法取得了很好的效果。<br>

* DSST 是 VOT2014 第一名，开创了平移滤波结合尺度滤波的方式。<br>
* SRDCF 是 VOT2015 的第四名，优化目标增加了空间约束项。<br>
* SRDCFdecon 在 SRDCF 的基础上，改进了样本和学习率问题。<br>
* DeepSRDCF 是 VOT2015 第二名，将 HOG 特征替换为 CNN 卷积特征（基于 VGG），效果有了极大提升。论文测试了不同卷积层在目标跟踪任务中的表现，第 1 层表现最好，第 2 和第 5 次之。<br>
<p align="center">
	<img src="https://pic2.zhimg.com/80/v2-15eaa1e7a50c7ad671fb84a42c7bfc20_720w.jpg" alt="Sample"  width="400">
</p>

* Chao Ma 的 HCF，结合多层卷积特征，用了 VGG19 的 Conv5-4, Conv4-4 和 Conv3-4 的激活值作为特征，在VOT2016排在28名。<br>
* C-COT 是 VOT2016 第一名，将 DeepSRDCF 的单层卷积的深度特征扩展为多成卷积的深度特征（VGG第 1 和 5 层）。<br>
* Martin Danelljan 在 2017CVPR 的 ECO 是 C-COT 的加速版，从模型大小、样本集大小和更新策略三方面加速，CPU上速度 60FPS。<br>

<br>

### 2016-2017 颜色统计特征<br>
2016 年，深度学习方法发挥优势，纯 CNN 方法与结合了深度特征的 CF 方法成绩排列靠前。<br>
VOT2016竞赛主办方公开了[部分 tracker 代码和主页](http://votchallenge.net/vot2016/trackers.html)。<br>

<p align="center">
	<img src="https://pic2.zhimg.com/80/v2-26092e9dec4292c77d652b9738a89bf5_720w.jpg" alt="Sample"  width="500">
</p>

* C-COT 排第一，是结合了多层深度特征的相关滤波；<br>
* TCNN 是纯 CNN 方法，VOT2016 的冠军；<br>
* 纯颜色方法 [DAT](http://lrs.icg.tugraz.at/members/possegger#dat) 和 ASMS 都在中等水平；<br>
* Luca Bertinetto 的 SiamFC 和 Staple 都表现不错。<br>

HOG 对快速变形和快速运动效果不好，但对运动模糊及光照变化等情况鲁棒；颜色统计特征对变形、快速运动不敏感，但对光照变化和背景相似颜色效果不佳。这两类方法可以互补，即 DSST 和 DAT 可以互补结合。<br>

<p align="center">
	<img src="https://pic3.zhimg.com/80/v2-6953c7c282e662af9c37c8fe5462c477_720w.jpg" alt="Sample"  width="500">
</p>

* [Staple](http://www.robots.ox.ac.uk/~luca/staple.html) 是模板特征方法 DSST 和统计特征方法 DAT 的结合，速度高达 80FPS。<br>
* 17CVPR 的 CSR-DCF，结合了相关滤波和颜色概率的方法，提出了空域可靠性和通道可靠性，性能直逼 C-COT，速度 13FPS。<br>
<p align="center">
	<img src="https://pic3.zhimg.com/80/v2-3a4be80f75f32314ca049d3e490d69b3_720w.jpg" alt="Sample"  width="500">
</p>

<br>

## 最新进展<br>
目标跟踪近几年发展迅速，融合了相关滤波（Correlation Filter）和卷积神经网络（CNN）的跟踪方法已经占据了目标跟踪的大半江山。如下图给出的是 2014-2017 年以来表现排名靠前的一些跟踪方法。<br>
按趋势来看，纯粹的深度神经网络方法、结合深度神经网络的相关滤波方法或将成为未来发展的主要走向。<br>
<p align="center">
	<img src="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWFnZXMyMDE4LmNuYmxvZ3MuY29tL2Jsb2cvMTA1MzY2MS8yMDE4MDMvMTA1MzY2MS0yMDE4MDMwNjA5NTMzMjk3Mi00MjM4ODMzNDkucG5n?x-oss-process=image/format,png" alt="Sample"  width="800">
</p>

<br>
<br>
<br>

## 参考文献：<br>
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
* Danelljan M, Hager G, Shahbaz Khan F, et al. Learning spatially regularized correlation filters for visual tracking [C]// ICCV. 2015.<br>
* Kiani Galoogahi H, Sim T, Lucey S. Correlation filters with limited boundaries [C]// CVPR, 2015.<br>
* Kiani Galoogahi H, Fagg A, Lucey S. Learning Background-Aware Correlation Filters for Visual Tracking [C]// ICCV, 2017.<br>
* Possegger H, Mauthner T, Bischof H. In defense of color-based model-free tracking [C]// CVPR, 2015.<br>
* Bertinetto L, Valmadre J, Golodetz S, et al. Staple: Complementary Learners for Real-Time Tracking [C]// CVPR, 2016.<br>
* Lukežič A, Vojíř T, Čehovin L, et al. Discriminative Correlation Filter with Channel and Spatial Reliability [C]// CVPR, 2017.<br>
* Ma C, Huang J B, Yang X, et al. Hierarchical convolutional features for visual tracking [C]// ICCV, 2015.<br>














