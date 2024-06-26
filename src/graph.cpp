#include "graph.h"
#include "executor.h"

namespace fastllm {
    void RunComputeGraph (const ComputeGraph &graph, 
                            const std::map <std::string, int> &deviceMap,
                            std::map <std::string, Data*> inputs,
                            std::map <std::string, Data*> weights,
                            std::map <std::string, Data*> outputs) {
        Executor excutor;
        std::map <std::string, Data*> tempDatas;
        std::map <std::string, Data*> allDatas;

        for (auto &it : inputs) {
            allDatas[it.first] = it.second;
        }
        for (auto &it : weights) {
            allDatas[it.first] = it.second;
        }
        for (auto &it : outputs) {
            allDatas[it.first] = it.second;
        }
        for (auto &node : graph.nodes) {
            if (allDatas.find(node.name) == allDatas.end()) {
                allDatas[node.name] = new Data();
            }
        }
        Data emptyData;

        for (int i = 0; i < graph.ops.size(); i++) {
            auto &op = graph.ops[i];
            // 一些没实现的算子
            if (op.type == "Exit") {
                exit(0);
            } else if (op.type == "Print") {
                auto data = allDatas[op.datas.find("input")->second];
                data->ToDevice(DataDevice::CPU);
                data->Print();
            } else if (op.type == "ExpandHeads") {
                auto data = allDatas[op.datas.find("input")->second];
                int headDim = op.intParams.find("headDim")->second;
                std::vector <int> dims = data->dims;
                dims.pop_back();
                dims.push_back(-1);
                dims.push_back(headDim);
                data->Reshape(dims);
            } else if (op.type == "FusedAttention") {
                {
                    std::vector <int> axis = {0, 2, 1, 3};
                    Data axisData = Data(DataType::INT32PARAM, {(int)axis.size()});
                    axisData.Allocate();
                    for (int i = 0; i < axisData.Count(0); i++) {
                        ((int32_t*)axisData.cpuData)[i] = axis[i];
                    }
                    std::vector <std::string> qkvs = {"q", "curk", "curv"};
                    for (int i = 0; i < qkvs.size(); i++) {
                        auto data = allDatas[op.datas.find(qkvs[i])->second];
                        excutor.Run("PermuteSelf", {
                            {"input", data}, {"axis", &axisData}
                        }, {}, {});
                        data->Reshape({-1, data->dims[2], data->dims[3]});
                    }
                }

                int unitLen = op.intParams.find("unitLen")->second;
                for (int i = 0; i < 2; i++) {                    
                    auto cache = allDatas[op.datas.find(i == 0 ? "k" : "v")->second];
                    auto cur = allDatas[op.datas.find(i == 0 ? "curk" : "curv")->second];

                    while ((cache->dims.size() == 0 && (cache->expansionDims.size() == 0 || cur->dims[1] > cache->expansionDims[1]))
                        || (cache->dims.size() > 0 && cache->dims[1] + cur->dims[1] > cache->expansionDims[1])) {
                        std::vector <int> newDims;
                        if (cache->Count(0) == 0 || cache->dims.size() == 0) {
                            newDims = std::vector <int> {cur->dims[0], ((cur->dims[1] - 1) / unitLen + 1) * unitLen, cur->dims[2]};
                        } else {
                            newDims = cache->dims;
                            newDims[1] += ((cur->dims[1] - 1) / unitLen + 1) * unitLen;
                        }
                        cache->Expansion(newDims);
                    }
                    excutor.Run("CatDirect", {
                            {"input0", cache}, {"input1", cur}
                    }, {}, {{"axis", 1}});
                }

                DataDict dataDict;
                for (auto &it : op.datas) {
                    dataDict[it.first] = allDatas[it.second];
                }
                excutor.Run("Attention", dataDict, op.floatParams, op.intParams);

                {
                    auto output = allDatas[op.datas.find("output")->second];
                    auto original = allDatas[op.datas.find("original")->second];
                    int bsz = original->dims[0], seqlen = original->dims[1];
                    std::vector <int> axis = {1, 0, 2};
                    Data axisData = Data(DataType::INT32PARAM, {(int)axis.size()});
                    axisData.Allocate();
                    for (int i = 0; i < axisData.Count(0); i++) {
                        ((int32_t*)axisData.cpuData)[i] = axis[i];
                    }
                    excutor.Run("PermuteSelf", {
                            {"input", output}, {"axis", &axisData}
                    }, {}, {});
                    output->Reshape({seqlen, bsz, -1});
                    excutor.Run("PermuteSelf", {
                            {"input", output}, {"axis", &axisData}
                    }, {}, {});
                }
            } else if (op.type == "SplitLastTokenStates") {
                auto input = allDatas[op.datas.find("input")->second];
                auto output = allDatas[op.datas.find("output")->second];
                int len = input->dims[1];
                if (len == 1) {
                    output->Resize(input->dims);
                    output->FakeFrom(*input, 0);
                } else {
                    excutor.Run("Split", {
                        {"input", input}, {"output", output}
                    }, {}, {{"axis", 1}, {"start", len - 1}, {"end", len}});                
                }
            } else {
                DataDict dataDict;
                for (auto &it : op.datas) {
                    if (allDatas.find(it.second) == allDatas.end()) {
                        dataDict[it.first] = &emptyData;
                    } else {
                        dataDict[it.first] = allDatas[it.second];
                    }
                }
                excutor.Run(op.type, dataDict, op.floatParams, op.intParams);
            }
        }

        for (auto it : tempDatas) {
            delete it.second;
        }
    }

    void ComputeGraph::Clear() {
        this->nodes.clear();
        this->ops.clear();
    }

    void ComputeGraph::Update() {
        this->nodes.clear();
        std::set <std::string> nodeNames;
        for (auto &op : this->ops) {
            for (auto &data : op.datas) {
                nodeNames.insert(data.second);
            }
        }
        for (auto &name : nodeNames) {
            this->nodes.push_back(ComputeGraphNode(name));
        }
    }

    void ComputeGraph::Print(ComputeGraphNode &input) {
        this->ops.push_back (
            ComputeGraphOp("Print", 
                {{"input", input.name}}, 
                {}, {})
        );
    }

    void ComputeGraph::Exit() {
        this->ops.push_back (ComputeGraphOp("Exit", {}, {}, {}));
    }

    void ComputeGraph::Embedding(ComputeGraphNode &input, ComputeGraphNode &weight, ComputeGraphNode &output) {
        this->ops.push_back (
            ComputeGraphOp("Embedding", 
                {{"input", input.name}, {"weight", weight.name}, {"output", output.name}}, 
                {}, {})
        );
    }

    void ComputeGraph::RMSNorm(ComputeGraphNode &input, ComputeGraphNode &weight, float eps, ComputeGraphNode &output) {
        this->ops.push_back (
            ComputeGraphOp("RMSNorm", 
                {{"input", input.name}, {"weight", weight.name}, {"output", output.name}}, 
                {{"eps", eps}}, {})
        );
    }

    void ComputeGraph::Linear(ComputeGraphNode &input, ComputeGraphNode &weight, ComputeGraphNode &bias, ComputeGraphNode &output) {
        this->ops.push_back (
            ComputeGraphOp("Linear", 
                {{"input", input.name}, {"weight", weight.name}, {"bias", bias.name}, {"output", output.name}}, 
                {}, {})
        );
    }

    void ComputeGraph::ExpandHead(ComputeGraphNode &input, int headDim) {
        this->ops.push_back (
            ComputeGraphOp("ExpandHeads", 
                {{"input", input.name}}, 
                {}, {{"headDim", headDim}})
        );
    }

    void ComputeGraph::AddTo(ComputeGraphNode &input0, ComputeGraphNode &input1, float alpha) {
        this->ops.push_back (
            ComputeGraphOp("AddTo", 
                {{"input0", input0.name}, {"input1", input1.name}}, 
                {{"alpha", alpha}}, {})
        );
    }

    void ComputeGraph::MulTo(ComputeGraphNode &input0, ComputeGraphNode &input1) {
        this->ops.push_back (
            ComputeGraphOp("MulTo", 
                {{"input0", input0.name}, {"input1", input1.name}}, 
                {}, {})
        );
    }

    void ComputeGraph::Silu(ComputeGraphNode &input, ComputeGraphNode &output) {
        this->ops.push_back (
            ComputeGraphOp("Silu", 
                {{"input", "w1"}, {"output", "w1"}}, 
                {}, {})
        );
    }

    void ComputeGraph::LlamaRotatePosition2D(ComputeGraphNode &input, ComputeGraphNode &positionIds, 
        ComputeGraphNode &sinData, ComputeGraphNode &cosData, int rotaryDim) {
        this->ops.push_back (
            ComputeGraphOp("LlamaRotatePosition2D", 
                {{"input", input.name}, {"positionIds", positionIds.name}, {"sin", sinData.name}, {"cos", cosData.name}}, 
                {}, {{"rotaryDim", rotaryDim}})
        );
    }

    void ComputeGraph::FusedAttention(ComputeGraphNode &q, ComputeGraphNode &k, ComputeGraphNode &v, 
        ComputeGraphNode &curk, ComputeGraphNode &curv, 
        ComputeGraphNode &original, ComputeGraphNode &mask, ComputeGraphNode &output, 
        float scale, int maskType, int unitLen) {
        this->ops.push_back(
            ComputeGraphOp("FusedAttention", 
                {{"q", q.name}, {"k", k.name}, {"v", v.name},
                {"curk", curk.name}, {"curv", curv.name},
                {"original", original.name},
                {"mask", mask.name}, {"output", output.name}}, 
                {{"scale", scale}}, {{"maskType", maskType}, {"unitLen", unitLen}})
        );
    }

    void ComputeGraph::Split(ComputeGraphNode &input, int axis, int start, int end, ComputeGraphNode &output) {
        this->ops.push_back (
            ComputeGraphOp("Split", 
                {{"input", input.name}, {"output", output.name}}, 
                {}, {{"axis", axis}, {"start", start}, {"end", end}})
        );
    }

    void ComputeGraph::SplitLastTokenStates(ComputeGraphNode &input, ComputeGraphNode &output) {
        this->ops.push_back (
            ComputeGraphOp("SplitLastTokenStates", 
                {{"input", input.name}, {"output", output.name}}, 
                {}, {})
        );
    }
}