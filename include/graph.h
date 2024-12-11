//
// Created by huangyuyang on 6/24/24.
//

#ifndef FASTLLM_GRAPH_H
#define FASTLLM_GRAPH_H

#include "fastllm.h"
#include "executor.h"

namespace fastllm {
    // 计算图基本信息
    struct ComputeGraphInfo {
        std::string type; // {none, llm, embedding}

        std::vector <std::string> inputs;
        std::vector <std::string> outputs;
    };

    // 节点信息
    struct ComputeGraphNode {
        std::string name;

        ComputeGraphNode () {}

        ComputeGraphNode (const std::string &name) :
            name(name) {}
    };

    // 算子信息（计算图的边）
    struct ComputeGraphOp {
        std::string type; 
        std::map <std::string, std::string> datas;
        std::map <std::string, float> floatParams;
        std::map <std::string, int> intParams;

        ComputeGraphOp (const std::string &type, 
                        const std::map <std::string, std::string> &datas,
                        const std::map <std::string, float> &floatParams,
                        const std::map <std::string, int> &intParams) :
                    type(type), datas(datas), floatParams(floatParams), intParams(intParams) {}
    };

    // 计算图
    struct ComputeGraph {
        ComputeGraphInfo info;
        std::vector <ComputeGraphNode> nodes;
        std::vector <ComputeGraphOp> ops;

        void Clear();

        void Update();

        void Add(ComputeGraphNode &input, float v, ComputeGraphNode &output); // output = input + v
        void AddTo(ComputeGraphNode &input0, ComputeGraphNode &input1, float alpha = 1.0); // input0 += input1 * alpha
        void Cat(ComputeGraphNode &input0, ComputeGraphNode &input1, int axis, ComputeGraphNode &output);
        void DataTypeAs(ComputeGraphNode &input, ComputeGraphNode &input1); // 将input的dataType设成和input1一样
        void Embedding(ComputeGraphNode &input, ComputeGraphNode &weight, ComputeGraphNode &output);
        void ExpandHead(ComputeGraphNode &input, int headDim);
        void FusedAttention(ComputeGraphNode &q, ComputeGraphNode &k, ComputeGraphNode &v, 
                            ComputeGraphNode &curk, ComputeGraphNode &curv, 
                            ComputeGraphNode &original, ComputeGraphNode &mask, ComputeGraphNode &output, 
                            ComputeGraphNode &seqLens,
                            float scale, int maskType, int unitLen); // 融合的attention
        void Gelu(ComputeGraphNode &input, ComputeGraphNode &output);
        void Linear(ComputeGraphNode &input, ComputeGraphNode &weight, ComputeGraphNode &bias, ComputeGraphNode &output);
        void LlamaRotatePosition2D(ComputeGraphNode &input, ComputeGraphNode &positionIds, ComputeGraphNode &sinData, ComputeGraphNode &cosData, int rotaryDim); // 2D position for llama
        void Mul(ComputeGraphNode &input, float v, ComputeGraphNode &output); // output = input * v
        void MulTo(ComputeGraphNode &input0, ComputeGraphNode &input1); // input0 *= input1
        void Repeat(ComputeGraphNode &input, int axis, int repeatTimes, ComputeGraphNode &output);
        void RMSNorm(ComputeGraphNode &input, ComputeGraphNode &weight, float eps, ComputeGraphNode &output);
        void Silu(ComputeGraphNode &input, ComputeGraphNode &output);
        void Split(ComputeGraphNode &input, int axis, int start, int end, ComputeGraphNode &output);
        void SplitLastTokenStates(ComputeGraphNode &input, ComputeGraphNode &seqLens, ComputeGraphNode &output);
        void Swiglu(ComputeGraphNode &input, ComputeGraphNode &output);

        // 以下op用于调试
        void Exit(); // 退出
        void Print(ComputeGraphNode &input); // 打印
    };

    // 优化计算图
    void OptimizeComputeGraph(ComputeGraph &graph, WeightMap &weight);

    // 执行计算图
    void RunComputeGraph (const ComputeGraph &graph, 
                            const std::map <std::string, int> &deviceMap,
                            const std::map <std::string, Data*> &inputs,
                            const std::map <std::string, Data*> &weights,
                            const std::map <std::string, Data*> &outputs, 
                            std::vector <std::vector <Data*> > &pastKeys, 
                            std::vector <std::vector <Data*> > &pastValues,
                            std::vector <Data*> &masks);
}

#endif //FASTLLM_GRAPH_H
