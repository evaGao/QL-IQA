#!/bin/bash

OUTDIR=outputs_vgg


   rm output_.txt
   rm scores_.txt
  
   ./build/tools/caffe test --weights=/media/s408/zz/gr/NR-IQA-CNN-master1/LIVE_lcn/single/model/_iter_8000.caffemodel --model=/media/s408/zz/gr/NR-IQA-CNN-master1/LIVE_lcn/single/net/singledouble.prototxt --iterations=201 --gpu=all
   cp output_.txt $OUTDIR/output1.txt
   cp scores_.txt $OUTDIR/scores1.txt

chmod -R a+rw $OUTDIR
