#include <algorithm>
#include <vector>
 
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/change_loss_layer.hpp"

#include "caffe/util/math_functions.hpp"
 
namespace caffe {
 
template <typename Dtype>
void ChangeLossLayer<Dtype>::LayerSetUp(//装载该层数据
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1); //bottom[0]为第一个特征
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);//bottom[1]为第二个特征
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);//bottom[2]为标签
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
}
 
template <typename Dtype>
void ChangeLossLayer<Dtype>::Forward_cpu(//前向传播过程
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a 第一个特征
      bottom[1]->cpu_data(),  // b 第二个特征
      diff_.mutable_cpu_data());  // a_i-b_i 两个特征的对应元素相减
  const int channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.change_loss_param().margin(); //读取prototxt中设置好的margin, 一般为1
  Dtype add_front = this->layer_param_.change_loss_param().add_front(); 
  Dtype add_after = this->layer_param_.change_loss_param().add_after();
  Dtype range = this->layer_param_.change_loss_param().range();
  Dtype loss(0.0);
  Dtype dist(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    dist_sq_.mutable_cpu_data()[i] = *(diff_.cpu_data() + (i*channels));
    if (static_cast<int>(bottom[2]->cpu_data()[i])==0) {  // 如果sim标签为0
      dist=pow(std::max<Dtype>(-dist_sq_.cpu_data()[i],Dtype(0.0))-std::max<Dtype>(dist_sq_.cpu_data()[i]-margin,Dtype(0.0)),2)+pow(std::max<Dtype>(-bottom[0]->cpu_data()[i],Dtype(0.0))-std::max<Dtype>(bottom[0]->cpu_data()[i]-range,Dtype(0.0)),2);//损失为特征的距离,这里表示为dist
      loss+=dist;
    } else if(static_cast<int>(bottom[2]->cpu_data()[i])==1){  // sim标签为1
      dist=pow(std::max<Dtype>(-dist_sq_.cpu_data()[i],Dtype(0.0))-std::max<Dtype>(dist_sq_.cpu_data()[i]-2*margin,Dtype(0.0)),2)+pow(std::max<Dtype>(-bottom[0]->cpu_data()[i],Dtype(0.0))-std::max<Dtype>(bottom[0]->cpu_data()[i]-range,Dtype(0.0)),2);
      loss+=dist;
      } else if(static_cast<int>(bottom[2]->cpu_data()[i])==2){ // sim标签为2
        dist = pow(std::max<Dtype>(add_front-dist_sq_.cpu_data()[i],Dtype(0.0))-std::max<Dtype>(dist_sq_.cpu_data()[i]-add_after,Dtype(0.0)),2)+pow(std::max<Dtype>(-bottom[0]->cpu_data()[i],Dtype(0.0))-std::max<Dtype>(bottom[0]->cpu_data()[i]-range,Dtype(0.0)),2);
        loss += dist;
      }
   }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

//反向传播
template <typename Dtype>
void ChangeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype margin = this->layer_param_.change_loss_param().margin();
  Dtype add_front = this->layer_param_.change_loss_param().add_front(); 
  Dtype add_after = this->layer_param_.change_loss_param().add_after();
  Dtype range = this->layer_param_.change_loss_param().range();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[i]->num());
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype dist=*(diff_.cpu_data() + (j*channels));
	Dtype mdist(0.0);
	Dtype mdistp(0.0);
	mdist=margin-dist;
	mdistp=2*margin-dist;
	Dtype* bout = bottom[i]->mutable_cpu_diff();
        if (static_cast<int>(bottom[2]->cpu_data()[j])==0) { 
		if(bottom[0]->cpu_data()[j]<Dtype(0.0)){ 
          		if(dist<Dtype(0.0)){
			  	caffe_set(channels,alpha*dist, bout + (j*channels));
		  	}else if(mdist>=Dtype(0.0)){
			  	caffe_set(channels,Dtype(0.0), bout + (j*channels));
		  	}else if(mdist<Dtype(0.0)){
			  	caffe_set(channels,-alpha*mdist, bout + (j*channels));
		  	}
			if(i==0)
				caffe_set(channels,*(bout + (j*channels))+bottom[0]->cpu_data()[j], bout + (j*channels));
		}
		else if(bottom[0]->cpu_data()[j]-range<=Dtype(0.0)){
			if(dist<Dtype(0.0)){
			  	caffe_set(channels,alpha*dist, bout + (j*channels));
		  	}else if(mdist>=Dtype(0.0)){
			  	caffe_set(channels,Dtype(0.0), bout + (j*channels));
		  	}else if(mdist<Dtype(0.0)){
			  	caffe_set(channels,-alpha*mdist, bout + (j*channels));
		  	}
		}	
		else if(bottom[0]->cpu_data()[j]-range>Dtype(0.0)){
			if(dist<Dtype(0.0)){
			  	caffe_set(channels,alpha*dist, bout + (j*channels));
		  	}else if(mdist>=Dtype(0.0)){
			  	caffe_set(channels,Dtype(0.0), bout + (j*channels));
		  	}else if(mdist<Dtype(0.0)){
			  	caffe_set(channels,-alpha*mdist, bout + (j*channels));
		  	}
			if(i==0)
				caffe_set(channels,*(bout + (j*channels))+bottom[0]->cpu_data()[j]-range, bout + (j*channels));
		}		
        }else if(static_cast<int>(bottom[2]->cpu_data()[j])==1) { 
		 if(bottom[0]->cpu_data()[j]<Dtype(0.0)){
		 	if(dist<Dtype(0.0)){
				caffe_set(channels,alpha*dist, bout + (j*channels));
		  	}else if(mdistp>=Dtype(0.0)){
			  	caffe_set(channels,Dtype(0.0), bout + (j*channels));
		  	}else if(mdistp<Dtype(0.0)){
			  	caffe_set(channels,-alpha*mdistp, bout + (j*channels));
		  	}
			if(i==0)
				caffe_set(channels,*(bout + (j*channels))+bottom[0]->cpu_data()[j], bout + (j*channels));
		}
		else if(bottom[0]->cpu_data()[j]-range<=Dtype(0.0)){
			if(dist<Dtype(0.0)){
				caffe_set(channels,alpha*dist, bout + (j*channels));
		  	}else if(mdistp>=Dtype(0.0)){
			  	caffe_set(channels,Dtype(0.0), bout + (j*channels));
		  	}else if(mdistp<Dtype(0.0)){
			  	caffe_set(channels,-alpha*mdistp, bout + (j*channels));
		  	}
		}	
		else if(bottom[0]->cpu_data()[j]-range>Dtype(0.0)){
			if(dist<Dtype(0.0)){
				caffe_set(channels,alpha*dist, bout + (j*channels));
		  	}else if(mdistp>=Dtype(0.0)){
			  	caffe_set(channels,Dtype(0.0), bout + (j*channels));
		  	}else if(mdistp<Dtype(0.0)){
			  	caffe_set(channels,-alpha*mdistp, bout + (j*channels));
		  	}
			if(i==0)
				caffe_set(channels,*(bout + (j*channels))+bottom[0]->cpu_data()[j]-range, bout + (j*channels));
		}		
        }else if(static_cast<int>(bottom[2]->cpu_data()[j])==2) {
		if(bottom[0]->cpu_data()[j]<Dtype(0.0)){  
			if(dist<add_front){
				caffe_set(channels,-alpha*(add_front-dist), bout + (j*channels));
		  	}else if(dist<=add_after){
			  	caffe_set(channels,Dtype(0.0), bout + (j*channels));
		  	}else if(dist>add_after){
			  	caffe_set(channels,-alpha*(add_after-dist), bout + (j*channels));
		  	}
			if(i==0)
				caffe_set(channels,*(bout + (j*channels))+bottom[0]->cpu_data()[j], bout + (j*channels));
		}else if(bottom[0]->cpu_data()[j]-range<=Dtype(0.0)){  
			if(dist<add_front){
				caffe_set(channels,-alpha*(add_front-dist), bout + (j*channels));
		  	}else if(dist<=add_after){
			  	caffe_set(channels,Dtype(0.0), bout + (j*channels));
		  	}else if(dist>add_after){
			  	caffe_set(channels,-alpha*(add_after-dist), bout + (j*channels));
		  	}
		}else if(bottom[0]->cpu_data()[j]-range>Dtype(0.0)){  
			if(dist<add_front){
				caffe_set(channels,-alpha*(add_front-dist), bout + (j*channels));
		  	}else if(dist<=add_after){
			  	caffe_set(channels,Dtype(0.0), bout + (j*channels));
		  	}else if(dist>add_after){
			  	caffe_set(channels,-alpha*(add_after-dist), bout + (j*channels));
		  	}
			if(i==0)
				caffe_set(channels,*(bout + (j*channels))+bottom[0]->cpu_data()[j]-range, bout + (j*channels));
		}
        }
      }
    }
  }
  /*Dtype* bout1 = bottom[0]->mutable_cpu_diff();
  Dtype* bout2 = bottom[1]->mutable_cpu_diff();
  int channels = bottom[0]->channels();
  for(int i=0;i<bottom[0]->num();++i){
	*(bout1+(i*channels))=*(bout1+(i*channels))+*(bout2+(i*channels));
	*(bout2+(i*channels))=*(bout1+(i*channels));
  }*/
}
#ifdef CPU_ONLY
STUB_GPU(ChangeLossLayer);
#endif
 
INSTANTIATE_CLASS(ChangeLossLayer);
REGISTER_LAYER_CLASS(ChangeLoss);
 
}  // namespace caffe
