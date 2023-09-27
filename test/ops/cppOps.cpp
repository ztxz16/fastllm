#include "fastllm.h"

void callBaseOp(int optype=0){
    fastllm::Data inputs = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2}, {1, 5});
    fastllm::Data outputs = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2}, {3, 4});

    switch (optype)
    {
    case 0:
        fastllm::AddTo(inputs, outputs, 1);
        break;
    case 1:
        fastllm::Cat(inputs, inputs, 0, outputs);
        break;
    case 2:
        fastllm::Mul(inputs, 2, outputs);
        break;
    case 3:
        fastllm::Permute(inputs, {1, 0}, outputs);
        break;
    case 4:
        fastllm::Split(inputs, 0, 0, 1, outputs);
        break;
    case 5:
        fastllm::Permute(inputs, {1, 0}, outputs);
        fastllm::MatMul(inputs, outputs, outputs);
        break;
    default:
        break;
    } 
    outputs.Print();
}

void callNormOp(int normType=0){
    fastllm::Data inputs = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2}, {1, 5}); 
    fastllm::Data weights = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2}, {1, 2});
    fastllm::Data gamma = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2}, {1, 1});
    fastllm::Data beta = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2}, {0, 0});
    fastllm::Data outputs;

    switch (normType)
    {
    case 0:
        fastllm::LayerNorm(inputs, gamma, beta, -1, outputs);
        break;
    case 1:
        fastllm::RMSNorm(inputs, weights, 1e-5, outputs);
        break;
    default:
        break;
    }
    outputs.Print();
}
    

void callLinearOp(){
    fastllm::Data inputs = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2}, {1, 2}); 
    fastllm::Data weights = fastllm::Data(fastllm::DataType::FLOAT32, {3, 2}, {3, 4, 5, 5, 6, 7});
    fastllm::Data bias = fastllm::Data(fastllm::DataType::FLOAT32, {1, 3}, {0, 1, 1});
    fastllm::Data outputs;
    fastllm::Linear(inputs, weights, bias, outputs);
    outputs.Print();
}

void callActivationOp(int activateType=0){
    fastllm::Data inputs = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2}, {1, 5});
    fastllm::Data outputs;
    switch (activateType)
    {
    case 0:
        fastllm::Silu(inputs, outputs);
        break;
    case 1:
        fastllm::Softmax(inputs, outputs, -1);
        break;
    case 2:
        fastllm::GeluNew(inputs, outputs);
        break;
    case 3:
        fastllm::Swiglu(inputs, outputs);
        break;
    default:
        break;
    }
    outputs.Print();
}

void callAttentionOp(int group=1, int attentionType=0){
    const fastllm::Data q = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2, 3}, {1, 2, 3, 4, 5, 6});
    const fastllm::Data k = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2, 3}, {5, 6, 7, 8, 9, 10});
    const fastllm::Data v = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2, 3}, {1, 1, 1, 2, 1, 3});
    const fastllm::Data mask = fastllm::Data();
    int dims = q.dims.back();
    float scale = 1/sqrt(dims);
    fastllm::Data output;

    fastllm::Attention(q, k, v, mask, output, group, scale, attentionType);
}

void testBase(){
    printf("testing BaseOp...\n");
    for (int i=0;i<6;i++){
        callBaseOp(i);
    }
    printf("test BaseOp finished!\n");
}

void testActivation(){
    printf("testing ActivationOp...\n");
    for (int i=0;i<4;i++){
        callActivationOp(i);
    }
    printf("test ActivationOp finished!\n");
}

void testAttention(){
    printf("testing AttentionOp...\n");
    callAttentionOp();
    printf("test AttentionOp finished!\n");
}

void testLinaer(){
    printf("testing LinearOp...\n");
    callLinearOp();
    printf("test LinearOp finished!\n");
}

void testNorm(){
    printf("testing NormOp...\n");
    for (int i=0;i<2;i++){
        callNormOp(i);
    }
    printf("test NormOp finished!\n");
}

void testAll(){
    testBase();
    testActivation();
    testAttention();
    testNorm();
    testLinaer();
}


int main(){
    testAll();
}