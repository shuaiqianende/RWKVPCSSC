#include <torch/extension.h>


void cuda_forward_local(int B, int T, int C, float *w, float *u, float *k, float *v, float *y);
void cuda_backward_local(int B, int T, int C, float *w, float *u, float *k, float *v, float *gy, float *gw, float *gu, float *gk, float *gv);

void forward_local(int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y) {
    cuda_forward_local(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>());
}
void backward_local(int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &gy, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &gk, torch::Tensor &gv) {
    cuda_backward_local(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), gy.data_ptr<float>(), gw.data_ptr<float>(), gu.data_ptr<float>(), gk.data_ptr<float>(), gv.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, x) {
    x.def("forward", &forward_local, "wkv_local forward");
    x.def("backward", &backward_local, "wkv_local backward");
}

TORCH_LIBRARY(wkv_local, x) {
    x.def("forward", forward_local);
    x.def("backward", backward_local);
}
