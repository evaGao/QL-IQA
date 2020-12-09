#!/usr/bin/env sh


./build/tools/caffe train --solver=/media/s408/zz/gr/NR-IQA-CNN-master1/TID2013/single/solver_vgg.prototxt --weights=/media/s408/zz/gr/NR-IQA-CNN-master1/TID2013/single/model/_iter_8000.caffemodel --gpu=0 2>&1   | tee /media/s408/zz/gr/NR-IQA-CNN-master1/TID2013/single/log/tid_single2.log

#--weights=/home/xiaogao/下载/NR-IQA-CNN-master1/train/model_iter_35000.caffemodel
#--weights=/home/xiaogao/下载/NR-IQA-CNN-master1/again/pair/models/_iter_3200.caffemodel
#--weights=/media/s408/zz/gr/NR-IQA-CNN-master1/again_end/pair/models_iter_17920.caffemodel
#--weights=/media/s408/zz/gr/RankIQA-master/models/VGG_ILSVRC_16_layers.caffemodel
