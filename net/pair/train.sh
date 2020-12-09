#!/usr/bin/env sh


./build/tools/caffe train --solver=/media/s408/zz/gr/NR-IQA-CNN-master1/LIVE_lcn/pair/net/solver_vgg.prototxt --weights /media/s408/zz/gr/NR-IQA-CNN-master1/models/IQA_CNN/VGG_ILSVRC_16_layers.caffemodel,/media/s408/zz/gr/NR-IQA-CNN-master1/models/IQA_CNN/VGG_ILSVRC_16_layers_p.caffemodel --gpu=0  2>&1   | tee /media/s408/zz/gr/NR-IQA-CNN-master1/LIVE_lcn/pair/log/train_vgg_livelcnp.log
#--weights /home/xiaogao/下载/RankIQA-master/models/VGG_ILSVRC_16_layers.caffemodel  /media/s408/zz/gr/NR-IQA-CNN-master1/models/IQA_CNN  ,/media/s408/zz/gr/NR-IQA-CNN-master1/models/IQA_CNN/VGG_ILSVRC_16_layers_p.caffemodel
