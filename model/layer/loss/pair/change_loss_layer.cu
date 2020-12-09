#include <algorithm>
#include <vector>
 
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/change_loss_layer.hpp"

#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ChangeLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // a
      bottom[1]->gpu_data(),  // b
      diff_.mutable_gpu_data());  // a_i-b_i
  caffe_gpu_powx(
      count,
      diff_.mutable_gpu_data(),  // a_i-b_i
      Dtype(1.0),
      diff_sq_.mutable_gpu_data());  // (a_i-b_i)^1
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      diff_sq_.gpu_data(),  // (a_i-b_i)^1
      summer_vec_.gpu_data(),
      Dtype(0.0),
      dist_sq_.mutable_gpu_data());  // \Sum (a_i-b_i)^2
  Dtype margin = this->layer_param_.change_loss_param().margin();
  Dtype add_front = this->layer_param_.change_loss_param().add_front(); 
  Dtype add_after = this->layer_param_.change_loss_param().add_after();
  Dtype range = this->layer_param_.change_loss_param().range();
  Dtype loss(0.0);
  Dtype dist(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    if (static_cast<int>(bottom[2]->cpu_data()[i])==0) {  // similar pairs
      dist=pow(std::max<Dtype>(-dist_sq_.cpu_data()[i],Dtype(0.0))-std::max<Dtype>(dist_sq_.cpu_data()[i]-margin,Dtype(0.0)),2)+pow(std::max<Dtype>(-bottom[0]->cpu_data()[i],Dtype(0.0))-std::max<Dtype>(bottom[0]->cpu_data()[i]-range,Dtype(0.0)),2);//损失为特征的距离,这里表示为dist
      loss=loss+dist;
    } else if(static_cast<int>(bottom[2]->cpu_data()[i])==1){  // sim标签为0
      dist=pow(std::max<Dtype>(-dist_sq_.cpu_data()[i],Dtype(0.0))-std::max<Dtype>(dist_sq_.cpu_data()[i]-2*margin,Dtype(0.0)),2)+pow(std::max<Dtype>(-bottom[0]->cpu_data()[i],Dtype(0.0))-std::max<Dtype>(bottom[0]->cpu_data()[i]-range,Dtype(0.0)),2);
      loss=loss+dist;
     } else if(static_cast<int>(bottom[2]->cpu_data()[i])==2){
        dist = pow(std::max<Dtype>(add_front-dist_sq_.cpu_data()[i],Dtype(0.0))-std::max<Dtype>(dist_sq_.cpu_data()[i]-add_after,Dtype(0.0)),2)+pow(std::max<Dtype>(-bottom[0]->cpu_data()[i],Dtype(0.0))-std::max<Dtype>(bottom[0]->cpu_data()[i]-range,Dtype(0.0)),2);
        loss =loss+dist;
      }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void CLLBackward(const int count, const int channels,
    const Dtype margin, const Dtype alpha, const Dtype add_after, const Dtype add_front, const Dtype range,
    const Dtype* y, const Dtype* diff, const Dtype* dist_sq,
    Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(i, count) {
    int n = i / channels;  // the num index, to access y and dist_sq
    Dtype dist=diff[i];
    Dtype mdist(0.0);
    Dtype mdistp(0.0);
    mdist=margin-dist;
    mdistp=2*margin-dist;
    if (static_cast<int>(y[n])==0) {  // similar pairs      		  
	if(dist<0.0){
		bottom_diff[i]=alpha*dist;
	}else if(mdist>=0.0){
		bottom_diff[i]=0;
	}else if(mdist<0.0){
			  bottom_diff[i]=-alpha*mdist;
	}
    } else if(static_cast<int>(y[n])==1) {  
		if(dist<0.0){
			  bottom_diff[i]=alpha*dist;
		  }else if(mdistp>=0){
			  bottom_diff[i]=0;
		  }else if(mdistp<0.0){
			  bottom_diff[i]=-alpha*mdistp;
		  }
        }else if(static_cast<int>(y[n])==2) {  
			if(dist<add_front){
			  bottom_diff[i]=-alpha*(add_front-dist);
		  }else if(dist<=add_after){
			  bottom_diff[i]=0;
		  }else if(dist>add_after){
			  bottom_diff[i]=-alpha*(add_after-dist);
		  }
        }
  }
}
template <typename Dtype>
__global__ void CLLBackward_one(const int count, const int channels,
    const Dtype margin, const Dtype alpha, const Dtype add_after, const Dtype add_front, const Dtype range,
    const Dtype* y, const Dtype* z, const Dtype* diff, const Dtype* dist_sq,
    Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(i, count) {
    int n = i / channels;  // the num index, to access y and dist_sq
    Dtype dist=diff[i];
    Dtype mdist(0.0);
    Dtype mdistp(0.0);
    mdist=margin-dist;
    mdistp=2*margin-dist;
    if (static_cast<int>(y[n])==0) {  // similar pairs      		  
	if(dist<0.0){
		bottom_diff[i]=alpha*dist;
	}else if(mdist>=0.0){
		bottom_diff[i]=0;
	}else if(mdist<0.0){
			  bottom_diff[i]=-alpha*mdist;
	}
    } else if(static_cast<int>(y[n])==1) {  
		if(dist<0.0){
			  bottom_diff[i]=alpha*dist;
		  }else if(mdistp>=0){
			  bottom_diff[i]=0;
		  }else if(mdistp<0.0){
			  bottom_diff[i]=-alpha*mdistp;
		  }
        }else if(static_cast<int>(y[n])==2) {  
			if(dist<add_front){
			  bottom_diff[i]=-alpha*(add_front-dist);
		  }else if(dist<=add_after){
			  bottom_diff[i]=0;
		  }else if(dist>add_after){
			  bottom_diff[i]=-alpha*(add_after-dist);
		  }
        }
    if(z[n]<0.0){
		bottom_diff[i]=bottom_diff[i]+z[n];
	}else if(z[n]-range<=0.0){
		bottom_diff[i]=bottom_diff[i];
	}else if(z[n]-range>0.0){
		bottom_diff[i]=bottom_diff[i]+z[n]-range;
	}
		
  }
}

template <typename Dtype>
void ChangeLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
 // for (int i = 0; i < 2; ++i) {
    if (propagate_down[0]) {
      const int count = bottom[0]->count();
      const int channels = bottom[0]->channels();
      Dtype margin = this->layer_param_.change_loss_param().margin();
      Dtype add_front = this->layer_param_.change_loss_param().add_front(); 
      Dtype add_after = this->layer_param_.change_loss_param().add_after();
      Dtype range = this->layer_param_.change_loss_param().range();
      const Dtype sign = (0 == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[0]->num());
      // NOLINT_NEXT_LINE(whitespace/operators)
      CLLBackward_one<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, alpha,add_after,add_front,range,
          bottom[2]->gpu_data(),  // pair similarity 0 or 1
	  bottom[0]->gpu_data(),
          diff_.gpu_data(),  // the cached eltwise difference between a and b
          dist_sq_.gpu_data(),  // the cached square distance between a and b
          bottom[0]->mutable_gpu_diff());
      CUDA_POST_KERNEL_CHECK;
    }else if(propagate_down[1]){
      const int count = bottom[0]->count();
      const int channels = bottom[0]->channels();
      Dtype margin = this->layer_param_.change_loss_param().margin();
      Dtype add_front = this->layer_param_.change_loss_param().add_front(); 
      Dtype add_after = this->layer_param_.change_loss_param().add_after();
      Dtype range = this->layer_param_.change_loss_param().range();
      const Dtype sign = (1 == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[0]->num());
      // NOLINT_NEXT_LINE(whitespace/operators)
      CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, alpha,add_after,add_front,range,
          bottom[2]->gpu_data(),  // pair similarity 0 or 1
          diff_.gpu_data(),  // the cached eltwise difference between a and b
          dist_sq_.gpu_data(),  // the cached square distance between a and b
          bottom[1]->mutable_gpu_diff());
      CUDA_POST_KERNEL_CHECK;
    }
 // }
  /*Dtype* bout1 = bottom[0]->mutable_gpu_diff();
  Dtype* bout2 = bottom[1]->mutable_gpu_diff();
  for(int i=0;i<bottom[0]->num();++i){
	bout1[i]=bout1[i]+bout2[i];
	bout2[i]=bout1[i];
  }*/
  //CUDA_POST_KERNEL_CHECK;
  
}
INSTANTIATE_LAYER_GPU_FUNCS(ChangeLossLayer);

}  // namespace caffe
