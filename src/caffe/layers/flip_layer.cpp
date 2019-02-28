#include "caffe/layers/flip_layer.hpp"
namespace caffe {

template <typename Dtype>
FlipLayer<Dtype>::FlipLayer(const LayerParameter& param) : Layer<Dtype>(param){
	FlipParameter flip_param = param.flip_param();
	axis = flip_param.axis();
}

template<typename Dtype>
void FlipLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
	top[0]->ReshapeLike(*bottom[0]);
}

template<typename Dtype>
void FlipLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	     const vector<Blob<Dtype>*>& top){

}
template<typename Dtype>
void FlipLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	     const vector<Blob<Dtype>*>& top){
	Dtype* top_data = top[0]->mutable_cpu_data();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	vector<int> axises;
	for(int i = 0; i < 4; ++i){
		if(i == axis)
			axises.push_back(bottom[0]->shape(i)-1);
		else
			axises.push_back(0);
	}
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	for(int n = 0; n < num; ++n){
		for(int c = 0; c < channels; ++c){
			for(int h = 0; h < height; ++h){
				for(int w = 0; w < width; ++w){
					int n_temp = (n - axises[0]);
					int c_temp = (c - axises[1]);
					int h_temp = (h - axises[2]);
					int w_temp = (w - axises[3]);
					n_temp = n_temp > 0 ? n_temp:-n_temp;
					c_temp = c_temp > 0 ? c_temp:-c_temp;
					h_temp = h_temp > 0 ? h_temp:-h_temp;
					w_temp = w_temp > 0 ? w_temp:-w_temp;
					top_data[((n*channels+c)*height+h)*width+w] = bottom_data[((n_temp*channels+c_temp)*height+h_temp)*width+w_temp];
				}
			}
		}
	}
}

template<typename Dtype>
void FlipLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		const Dtype* top_data = top[0]->cpu_diff();
		Dtype* bottom_data = bottom[0]->mutable_cpu_diff();
		vector<int> axises;
		for(int i = 0; i < 4; ++i){
			if(i == axis)
				axises.push_back(bottom[0]->shape(i)-1);
			else
				axises.push_back(0);
		}
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		for(int n = 0; n < num; ++n){
			for(int c = 0; c < channels; ++c){
				for(int h = 0; h < height; ++h){
					for(int w = 0; w < width; ++w){
						int n_temp = (n - axises[0]);
						int c_temp = (c - axises[1]);
						int h_temp = (h - axises[2]);
						int w_temp = (w - axises[3]);
						n_temp = n_temp > 0 ? n_temp:-n_temp;
						c_temp = c_temp > 0 ? c_temp:-c_temp;
						h_temp = h_temp > 0 ? h_temp:-h_temp;
						w_temp = w_temp > 0 ? w_temp:-w_temp;
						bottom_data[((n*channels+c)*height+h)*width+w] = top_data[((n_temp*channels+c_temp)*height+h_temp)*width+w_temp];
					}
				}
			}
		}
}

INSTANTIATE_CLASS(FlipLayer);
REGISTER_LAYER_CLASS(Flip);
}
