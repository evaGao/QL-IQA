import sys
caffe_root = '/media/xiaogao/zz/gr/NR-IQA-CNN-master1/'
sys.path.insert(0, caffe_root + 'python')
import caffe

model_old = "/media/xiaogao/zz/gr/NR-IQA-CNN-master1/models/IQA_CNN/VGG_ILSVRC_16_layers.caffemodel"
modelconfig_old = "/media/xiaogao/zz/gr/NR-IQA-CNN-master1/models/IQA_CNN/vgg.prototxt"
modelconfig_new = "/media/xiaogao/zz/gr/NR-IQA-CNN-master1/models/IQA_CNN/vgg1.prototxt"

net_old = caffe.Net(modelconfig_old, model_old, caffe.TEST)
net_new = caffe.Net(modelconfig_new, model_old, caffe.TEST)

# rewrite proto
file = open(modelconfig_old, 'r')
file_new = open(modelconfig_new, 'w')
for line in file.readlines():
    if line.find("name") != -1:
        indx = line.find('"\n')
        name = line[0:indx]+'_p"\n'
        file_new.write(name)
    else:
        file_new.write(line)

file_new.close()

#rewrite model
for layer_name, param in net_old.params.iteritems():
    #indx = layer_name[1:].find('"')+1
    new_name = layer_name+'_p'
    n = len(param)
    for i in range(n):
        net_new.params[new_name][i].data[...] = param[i].data[...]

net_new.save("/media/xiaogao/zz/gr/NR-IQA-CNN-master1/models/IQA_CNN/VGG_ILSVRC_16_layers_p.caffemodel")

for layer_name, param in net_old.params.iteritems():
    print(layer_name)
for layer_name, param in net_new.params.iteritems():
    print(layer_name)

print "Done"
