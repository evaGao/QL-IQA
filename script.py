#去除重复行
'''import shutil
readPath='C:/Users/Administrator/Desktop/mappingsa/scores_val_.txt'
writePath='C:/Users/Administrator/Desktop/mappingsa/dual.txt'
lines_seen=set()
outfiile=open(writePath,'a+',encoding='utf-8')
f=open(readPath,'r',encoding='utf-8')
for line in f:
    if line not in lines_seen:
        outfiile.write(line)
        lines_seen.add(line)
outfiile.close()
f.close()'''
#读取前几个特定内容
'''readPath='C:/Users/Administrator/Desktop/mappingsa/scores_val_.txt'
writePath='C:/Users/Administrator/Desktop/mappingsa/dual.txt'
with open(readPath, 'r') as fd:
    for line in fd:
        pos=line.find('/')
       # print(line[:pos])
        with open(writePath, 'a') as ff:
            ff.write(line[:pos]+'\n')'''

#在一个文件中去除另一个文件中的内容
'''readPath1='C:/Users/Administrator/Desktop/mappings/scores_train_.txt'
readPath2='C:/Users/Administrator/Desktop/mappingsa/end.txt'
writePath='C:/Users/Administrator/Desktop/mappings/end4.txt'
with open(readPath1, 'r') as f1:
    for line1 in f1:
        pos=line1.find('/')
        with open(readPath2, 'r') as f2:
            for line2 in f2:
             #   print(line2)
                if line1[:pos]==line2[:line2.find('\n')]:
                   # print(line2)
                    break
            if(line1[:pos]!=line2[:line2.find('\n')]):
              #  print(line1[:pos])
                with open(writePath, 'a') as ff:
                    ff.write(line1+'\r')'''



'''readPath1='C:/Users/Administrator/Desktop/mappings/scores_train_.txt'
readPath2='C:/Users/Administrator/Desktop/mappingsa/end.txt'
writePath='C:/Users/Administrator/Desktop/mappings/end3.txt'
with open(readPath1, 'r') as f1:
    for line1 in f1:
        pos=line1.find('/')
        with open(readPath2, 'r') as f2:
            for line2 in f2:
             #   print(line2)
                if line1[:pos]==line2[:pos]:
                    #print(line2)
                    break'''

import shutil
list=[]
readPath1='/home/xiaogao/下载/NR-IQA-CNN-master1/LIVE/aa/all/map224_notcrop/scores_train_2.txt'
readPath2='/home/xiaogao/下载/NR-IQA-CNN-master1/LIVE/aa/all/map224_notcrop/scores_val_1.txt'
writePath='/home/xiaogao/下载/NR-IQA-CNN-master1/LIVE/aa/all/map224_notcrop/scores.txt'
with open(readPath2, 'r') as f2:
    for line in f2:
        list.append(line)
with open(readPath1, 'r') as f1:
    for line1 in f1:
        if line1 not in list:
            with open(writePath, 'a') as ff:
                    ff.write(line1)


#将两个文件的行任意合并
'''
list=[]
readPath1='/home/xiaogao/下载/NR-IQA-CNN-master1/LIVE/aa/tt.txt'
readPath2='/home/xiaogao/下载/NR-IQA-CNN-master1/LIVE/aa/tt1.txt'
writePath='/home/xiaogao/下载/NR-IQA-CNN-master1/LIVE/aa/tte.txt'
with open(readPath1, 'r') as f1:
    for line1 in f1:
	list.append(line1)
        with open(readPath2, 'r') as f2:
            for line2 in f2:
                if (line2 not in list)&(line1!=line2):
                    line3=line1[:line1.find('\n\r')]+'\t'+line2
                    with open(writePath, 'a') as ff:
                        ff.write(line3)'''


#打乱文件内部的行顺序Linux命令：shuffle C:/Users/Administrator/Desktop/mappings/C.txt -o C:/Users/Administrator/Desktop/mappings/D.txt
        

#将文件的每行输出为列表
with open('/home/xiaogao/下载/NR-IQA-CNN-master1/LIVE/bb/f5.txt','r')as f:
    list1=[int(line.rstrip('\n'))for line in f]
    print(list1)


#摘选文件中特定的几行内容
lnum=0
with open('/home/xiaogao/下载/NR-IQA-CNN-master1/LIVE/aa/all/mappings/lable5.txt','a')as f1:
    with open('/home/xiaogao/下载/NR-IQA-CNN-master1/LIVE/aa/all/mappings/scores_train_.txt','r')as f2:
        for line in f2:
            lnum+=1
            if lnum in  list1:
                f1.write(line)


#给每个类打1-5的标签
readPath='/home/xiaogao/下载/NR-IQA-CNN-master1/LIVE/114/label5'
writePath='/home/xiaogao/下载/NR-IQA-CNN-master1/LIVE/114/5'
with open(readPath, 'r') as f1:
    for line1 in f1:
        pos=line1.find(' ')
        with open(writePath, 'a') as ff:
                    ff.write(line1[:pos+1]+str(5)+'\n')


#给类与类配对，并打0-2的标签
list=[]
readPath1='/home/xiaogao/下载/NR-IQA-CNN-master1/LIVE/114/5'
readPath2='/home/xiaogao/下载/NR-IQA-CNN-master1/LIVE/114/3'
writePath='/home/xiaogao/下载/NR-IQA-CNN-master1/LIVE/114/53'
with open(readPath1, 'r') as f1:
    for line1 in f1:
	list.append(line1)
        with open(readPath2, 'r') as f2:
            for line2 in f2:
                if (line2 not in list):
                    pos1=line1.find(' ')+1
                    pos2=line2.find(' ')+1
                    line3=line1[:pos1]+line2[:pos2]+str(abs(int(line1[pos1])-int(line2[pos2])))+'\n'
                    with open(writePath, 'a') as ff:
                        ff.write(line3)

#将所有文件按行合并
ex="/home/xiaogao/下载/NR-IQA-CNN-master1/LIVE/114/"
flist=[ex+str(11),ex+str(22),ex+str(33),ex+str(44),ex+str(55),ex+str(32),ex+str(43),ex+str(54),ex+str(31),ex+str(42),ex+str(53)]
writePath='/home/xiaogao/下载/NR-IQA-CNN-master1/LIVE/114/all'
for fr in flist:
    with open(fr,'r') as f1:
        for line in f1:
             with open(writePath, 'a') as ff:
                    ff.write(line)
#将文件按照某列的值排序
with open('/home/xiaogao/下载/NR-IQA-CNN-master1/LIVE/114/label44','a')as f1:
    f1.write(''.join(sorted(open('/home/xiaogao/下载/NR-IQA-CNN-master1/LIVE/114/label4'),key=lambda s: s.split()[1],reverse=1)))

with open('/home/xiaogao/下载/NR-IQA-CNN-master1/TID2013/114/scores1','a')as f1:
    f1.write(''.join(sorted(open('/home/xiaogao/下载/NR-IQA-CNN-master1/TID2013/114/scores'),key=lambda l: l.split()[0])))

#替换文件夹下图片的名字
import os
import shutil
import os.path
path='/home/xiaogao/下载/NR-IQA-CNN-master1/TID2013/distorted_images/'
copydir='/home/xiaogao/下载/NR-IQA-CNN-master1/TID2013/distorted_after/'
f=os.listdir(path)
n=0
for oldname in f:
    old=path+oldname
    new=copydir+'img'+str(5*24*(int(oldname[1:3])-1)+5*(int(oldname[4:6])-1)+int(oldname[-5]))+'.bmp'
    #print(old)
    shutil.copy(old,new)

#产生类似于live数据集下的info_all.txt文件
readPath='/home/xiaogao/下载/NR-IQA-CNN-master1/TID2013/mos_with_names.txt'
writePath='/home/xiaogao/下载/NR-IQA-CNN-master1/TID2013/info_all.txt'
i=1
with open(readPath, 'r') as fd:
    for line in fd:
        pos1=line.find(' ')+2
        pos2=line.find('_')
       # print(line[:pos])
        with open(writePath, 'a') as ff:
            ff.write('I'+line[pos1:pos2]+'.BMP'+' '+'img'+str(i)+'.bmp'+'\n')
        i=i+1

#统计文件行数
readPath='/home/xiaogao/下载/NR-IQA-CNN-master1/TID2013/114/pair.txt'
num=0
with open(readPath, 'r') as f1:
    for line1 in f1:
        num=num+1
print(num)
