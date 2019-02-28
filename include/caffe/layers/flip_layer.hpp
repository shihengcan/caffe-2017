#ifndef CAFFE_FLIP_LAYER_HPP_
#define CAFFE_FLIP_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include <vector>
namespace caffe {
template<typename Dtype>
class FlipLayer : public Layer<Dtype>{
public:
  explicit FlipLayer(const LayerParameter& param);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top);
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "Flip"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  int axis;
};

}
#endif
