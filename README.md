<!--
 * @Author: gaorui
 * @email: 15735170462@163.com
 * @Date: 2020-12-09 18:00:38
 * @LastEditTime: 2020-12-09 18:49:09
 * @Description: 
-->
# QL-IQA
Learning Distance Distribution from Quality Levels for Blind Image Quality Assessment
### 一、准备工作：
从网上下载VGG16参数模型，因用到Siamese network，两个网络分支都必须是预训练好的，需转移下载好的VGG16模型参数到另一个网络分支（model/caffemodel.py)

### 二、预处理所有失真图像
对所有失真图像进行三通道的局部对比度归一化处理（Training data preparation scripts/lcn.m）

### 三、以图像对作为输入的网络训练，以学习质量等级之间的距离分布
1. 将所有失真图像汇合到一个文件夹下，并创建存储质量分数的文件dmos_all.mat
2. 统计失真图像，方便之后操作。（Training data preparation scripts/prep_training_notcrop.m）
3. 将所有图像与其质量分数对应在一个txt文件中，并分出训练集和测试集（Training data preparation scripts/prepare_scores_224.m）
4. 去除重复行，保证训练集和测试集之间没有交集(script.py)
5. 聚类。将训练集中所有图像的质量分数分成5类(Training data preparation scripts/cc.m)
6. 将分类结果对应到训练集中，给属于每一类的图像打标签1-5(script.py)
7. 对图像进行配对，相同类的图像对标签为0，内部差距为1的图像对标签为1，差距为2的图像对标签为2(script.py)
8. 将打标签之后的图像对文件转换为lmdb格式(data/convertSiac.cpp)
9. Loss function(model/loss/pair/...)
10. 开始训练(net/pair/train.sh)

### 四、以单幅图像作为输入的网络训练，以学习质量分数
1. 将带有质量分数标签的训练集单幅图像转换为lmdb格式(data/create_db.sh)
2. fusion layer的代码（medel/layer/fusion/together.py）
3. Loss function（model/layer/loss/single/...）
4. 开始训练(net/pair/single2.sh)

### 五、测试
1. 测试网络在测试集上评估图像质量分数的性能(net/test.sh)
2. 计算LCC、SROCC值(compute.m)

# 注意事项
1. 每条步骤后面括号中的内容为实现该步骤的程序文件；
2. 仅给出TID2013的net文件，LIVE数据库网络参数可根据LIVE数据量大小进行适当调整，不影响最终结果；
3. script.py是一个帮助性脚本文件，可能在复现代码过程中有一些数据处理会遇到问题，运行该脚本文件下的小程序或许可以解燃眉之急。譬如“去除重复行“，“给图像打标签”等等，该文件中的每个小程序功能在旁边都有注解，可直接运行通过；
4. 一些代码文件在运行时需要事先放在caffe下合适的位置，譬如loss layer和fusion layer以及生成lmdb数据格式的代码；
5. 运行网络时记得修改路径。