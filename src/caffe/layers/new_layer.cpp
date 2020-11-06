#include <vector>
#include "caffe/layers/new_layer.hpp"
 
namespace caffe {
 
template <typename Dtype>
void NewLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int count = bottom[0]->count();
	for (int i = 0; i < count; ++i) {
			top_data[i] = bottom_data[i];
	}
}
 
template <typename Dtype>
void NewLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const int count = bottom[0]->count();
		for (int i = 0; i < count; ++i) {
			bottom_diff[i] = top_diff[i];
		}
	}
}
 
#ifdef CPU_ONLY
	STUB_GPU(NewLayer);
#endif
 
INSTANTIATE_CLASS(NewLayer);  //类名，注：这个类名与prototxt文件中的层名不需一致
REGISTER_LAYER_CLASS(New); // 对应层的类型
 
}  // namespace caffe