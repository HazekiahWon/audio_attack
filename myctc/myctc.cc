#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <vector>
#include <algorithm>
//#include <math>

using namespace tensorflow;

REGISTER_OP("Myctc")
        .Input("phrase: int32") // should be 1dim
        .Input("logits: float32") // should be 2dim and probas
        .Output("myloss: float32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c){
            c->set_output(0, c->Scalar()); // the 0-th output is a scalar
            return Status::OK();
        });

class MyctcOp : public OpKernel {
public:
    explicit MyctcOp(OpKernelConstruction *context) : OpKernel(context) {}
    int max(int a, int b) {
        return a>b?a:b;
    }
    int min(int a, int b) {
        return a>b?b:a;
    }
    void Compute(OpKernelContext *context) override {
        // grab the inputs
        const Tensor &input1 = context->input(0);
        const Tensor &input2 = context->input(1);
        auto phrase = input1.vec<int32>();
        auto logits = input2.matrix<float>(); // suppose it is frame,n_vocab

        // create output
        Tensor *output_tensor = nullptr;
        TensorShape tmp_shape; // default 0 dims
        OP_REQUIRES_OK(context, context->allocate_output(0, tmp_shape, &output_tensor));

        auto output_flat = output_tensor->flat<float>(); // only one element

        int blank_ix = input2.shape().dim_size(1)-1; // shape[-1]-1
        int n_vocab{blank_ix+1};
        // params is exactly the transpose of logits-input2
        int seqLen = input1.shape().dim_size(0); // shape[0]

        int L = 2*seqLen+1;
        int T = input2.shape().dim_size(0);
        // make an array of LxT
        std::vector<std::vector<float>> alphas(L);
        for (auto vec: alphas)
            vec.resize(T); // default init for each ele

        // omit checking probas
        float c{0}, llForward{0}, myloss{0};
        c += alphas[0][0] = logits(0, blank_ix);
        c += alphas[1][0] = logits(0, phrase(0));
        alphas[0][0] /= c;
        alphas[1][0] /= c;

        llForward += log(c);

        for (int t=1; t<T; ++t) {
            auto start = max(0, L-2*(T-t));
            auto end = min(2*t+2, L);

            c = 0;
            for (int s=start; s<L; ++s) {
                int l = (s-1) / 2; // casting to make it floor
                if (s % 2==0) {
                    if (s==0) {
                        alphas[s][t] = alphas[s][t-1]*logits(t, blank_ix);
                    } else {
                        alphas[s][t] = (alphas[s][t - 1] + alphas[s - 1][t - 1]) * logits(t, blank_ix);
                    }
                } else if ((s==1) or (phrase(l)==phrase(l-1))) {
                    alphas[s][t] = (alphas[s][t - 1] + alphas[s - 1][t - 1]) * logits(phrase(l), t);
                } else {
                    alphas[s][t] = (alphas[s][t - 1] + alphas[s - 1][t - 1] + alphas[s - 2][t - 1])*logits(phrase(l), t);
                }

                c += alphas[s][t];
            }

            for (int s=start; s<L; ++s) {
                alphas[s][t] /= c;
            }

            llForward += log(c);

            for (int tmp=0; tmp<L; ++tmp) {
//                auto t_vec = logits(t);
                std::vector<float> vec(n_vocab);
                for (int i = 0; i < n_vocab; ++i)
                    vec[i] = logits(t, i);
                auto target = vec[tmp];
                float v{0};
                for (auto e: vec)
                    v += max(0, e - target);
                myloss += alphas[tmp][t] * v;
            }
        }

        output_flat(0) = myloss;

    }
};

REGISTER_KERNEL_BUILDER(Name("Myctc").Device(DEVICE_CPU), MyctcOp);

