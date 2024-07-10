#include "utils.h"
#include "graph.h"
#include "executor.h"

namespace fastllm {
    void OptimizeComputeGraph(ComputeGraph &graph, WeightMap &weight) {
        auto ops = graph.ops;
        graph.ops.clear();
        for (int i = 0; i < ops.size(); i++) {
            auto &op = ops[i];
            if (op.type == "Linear") {
                int j = i;
                while (j + 1 < ops.size() && ops[j + 1].type == "Linear" && ops[j + 1].datas["input"] == ops[i].datas["input"]) {
                    j++;
                }

                bool canmerge = true;
                int hasBiasMask = 0;
                for (int l = i; l <= j; l++) {
                    if (weight.weight.find(ops[l].datas["bias"]) != weight.weight.end()) {
                        hasBiasMask |= 2;
                    } else {
                        hasBiasMask |= 1;
                    }
                }
                if (hasBiasMask == 3) {
                    canmerge = false;
                }
                if (j > i && canmerge) {
                    auto &firstWeight = weight[ops[i].datas["weight"]];                    
                    std::string mergeWeightName = "", mergeBiasName = "", outputName = "";
                    std::vector <int> lens;
                    int lensSum = 0;
                    for (int l = i; l <= j; l++) {
                        mergeWeightName += ops[l].datas["weight"];
                        mergeBiasName += ops[l].datas["bias"];
                        outputName += ops[l].datas["output"] + "___merge___";
                        lens.push_back(weight[ops[l].datas["weight"]].dims[0]);
                        lensSum += lens.back();
                    }

                    if (weight.weight.find(ops[i].datas["bias"]) != weight.weight.end()) {
                        weight.weight[mergeBiasName] = Data(weight[ops[i].datas["bias"]].dataType, {lensSum});
                        Data &merge = weight.weight[mergeBiasName];
                        weight.weight[mergeBiasName].name = mergeBiasName;
                        merge.Allocate();
                        long long offset = 0;
                        for (int l = i; l <= j; l++) {
                            Data &curbias = weight[ops[l].datas["bias"]];
                            memcpy(merge.cpuData + offset, curbias.cpuData, curbias.GetBytes());
                            offset += curbias.GetBytes();
                        }
                    } else {
                        if (mergeBiasName != "") {
                            weight.weight[mergeBiasName] = Data();
                        } 
                    }

                    weight.weight[mergeWeightName] = Data(firstWeight.dataType, {lensSum, firstWeight.dims[1]});
                    Data &merge = weight.weight[mergeWeightName];
                    weight.weight[mergeWeightName].name = mergeWeightName;
                    merge.Allocate();
                    merge.group = firstWeight.group;
                    merge.groupCnt = firstWeight.groupCnt;
                    merge.perChannelAxis = firstWeight.perChannelAxis;
                    long long offset = 0;
                    for (int l = i; l <= j; l++) {
                        Data &curWeight = weight[ops[l].datas["weight"]];
                        memcpy(merge.cpuData + offset, curWeight.cpuData, curWeight.GetBytes());
                        merge.perChannelsConfigs = AppendVector(merge.perChannelsConfigs, curWeight.perChannelsConfigs);
                        merge.zeros = AppendVector(merge.zeros, curWeight.zeros);
                        merge.scales = AppendVector(merge.scales, curWeight.scales);
                        merge.mins = AppendVector(merge.mins, curWeight.mins);
                        offset += curWeight.GetBytes();
                        weight.weight.erase(ops[l].datas["weight"]);
                    }
                    ComputeGraphNode input(op.datas["input"]), weight(mergeWeightName), bias(mergeBiasName), mid(outputName);
                    graph.Linear(input, weight, bias, mid);
                    offset = 0;
                    for (int l = i; l <= j; l++) {
                        ComputeGraphNode output(ops[l].datas["output"]);
                        graph.Split(mid, -1, offset, offset + lens[l - i], output);
                        offset += lens[l - i];
                    }
                    i = j;
                    continue;
                }
            }

            graph.ops.push_back(op);
        }
    }

    void RunComputeGraph (const ComputeGraph &graph, 
                            const std::map <std::string, int> &deviceMap,
                            std::map <std::string, Data*> inputs,
                            std::map <std::string, Data*> weights,
                            std::map <std::string, Data*> outputs) {
        Executor &excutor = *((Executor*)GetExecutor());
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
            } else if (op.type == "DataTypeAs") {
                auto input = allDatas[op.datas.find("input")->second];
                DataType dataType = allDatas[op.datas.find("input1")->second]->dataType;
                if (input->dataType != dataType) {
                    if (dataType == DataType::FLOAT32) {
                        excutor.Run("ToFloat32", {
                                {"input", input}
                        }, {}, {});
                    } else if (dataType == DataType::FLOAT16) {
                        excutor.Run("ToFloat16", {
                                {"input", input}
                        }, {}, {});
                    } 
                }
            } else if (op.type == "ExpandHeads") {
                auto data = allDatas[op.datas.find("input")->second];
                int headDim = op.intParams.find("headDim")->second;
                std::vector <int> dims = data->dims;
                dims.pop_back();
                dims.push_back(-1);
                dims.push_back(headDim);
                data->Reshape(dims);
            } else if (op.type == "FusedAttention") {
                std::vector <int> seqLens;
                {
                    auto data = allDatas[op.datas.find("seqLens")->second];
                    for (int i = 0; i < data->Count(0); i++) {
                        seqLens.push_back(((int*)data->cpuData)[i]);
                    }
                }

                if (seqLens.size() == 1) {
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
                } else {
                    int batch = seqLens.size(), total = 0;
                    bool all1 = true;
                    for (int i = 0; i < seqLens.size(); i++) {
                        if (seqLens[i] != 1) {
                            all1 = false;
                            break;
                        }
                    }
                    
                    if (all1) {
                        std::vector <Data> curQs, curKs, curVs, curOutputs;
                        curQs.resize(batch);
                        curKs.resize(batch);
                        curVs.resize(batch);
                        curOutputs.resize(batch);
                        auto &q = *allDatas[op.datas.find("q")->second];
                        auto &k = *allDatas[op.datas.find("curk")->second];
                        auto &v = *allDatas[op.datas.find("curv")->second];

                        q.Reshape({-1, q.dims[2], q.dims[3]});
                        k.Reshape({-1, k.dims[2], k.dims[3]});
                        v.Reshape({-1, v.dims[2], v.dims[3]});
                        int embed_dim = q.dims[1] * v.dims[2];

                        std::vector <int> qdims = {q.dims[1], 1, q.dims[2]};
                        std::vector <uint64_t> qstrides = {(uint64_t)q.dims[2], (uint64_t)q.dims[2], 1};
                        std::vector <int> kdims = {k.dims[1], 1, k.dims[2]};
                        std::vector <uint64_t> kstrides = {(uint64_t)k.dims[2], (uint64_t)k.dims[2], 1};
                        std::vector <int> vdims = {v.dims[1], 1, v.dims[2]};
                        std::vector <uint64_t> vstrides = {(uint64_t)v.dims[2], (uint64_t)v.dims[2], 1};
                        for (int b = 0; b < batch; b++) {
                            curQs[b].dims = qdims;
                            curQs[b].strides = qstrides;
                            curQs[b].FakeFrom(q, b * q.strides[0] * q.unitSize);
                            curKs[b].dims = kdims;
                            curKs[b].strides = kstrides;
                            curKs[b].FakeFrom(k, b * k.strides[0] * k.unitSize);
                            curVs[b].dims = vdims;
                            curVs[b].strides = vstrides;
                            curVs[b].FakeFrom(v, b * v.strides[0] * v.unitSize);
                        }
                        total = batch;

                        int unitLen = op.intParams.find("unitLen")->second;
                        for (int i = 0; i < 2; i++) {
                            std::vector <Data*> caches, curs;
                            for (int b = 0; b < batch; b++) {                    
                                auto cache = allDatas[op.datas.find(i == 0 ? "k" : "v")->second + "_" + std::to_string(b)];
                                auto cur = i == 0 ? &curKs[b] : &curVs[b];
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
                                caches.push_back(cache);
                                curs.push_back(cur);                    
                            }
                            CatDirectBatch(caches, curs, 1);
                        }

                        auto &attenOutput = *allDatas[op.datas.find("output")->second];
                        attenOutput.dataType = q.dataType;
                        attenOutput.ToDevice(q.dataDevice);
                        attenOutput.Resize({1, batch, embed_dim});
                        attenOutput.Allocate();
                        std::vector <Data> curContextLayer;
                        std::vector <Data*> qs, keys, values, masks, contexts;
                        curContextLayer.resize(batch);
                        qs.resize(batch);
                        keys.resize(batch);
                        values.resize(batch);
                        masks.resize(batch);
                        contexts.resize(batch);

                        for (int b = 0; b < batch; b++) {
                            std::string sb = "_" + std::to_string(b);
                            qs[b] = (&curQs[b]);
                            keys[b] = allDatas[op.datas.find("k")->second + sb];
                            values[b] = allDatas[op.datas.find("v")->second + sb];
                            masks[b] = allDatas[op.datas.find("mask")->second + sb];
                            curContextLayer[b].FakeFrom(attenOutput, b * embed_dim * attenOutput.unitSize);
                            contexts[b] = (&curContextLayer[b]);
                        }
                        AttentionBatch(qs, keys, values, masks, contexts, qs[0]->dims[0] / values[0]->dims[0], op.floatParams.find("scale")->second, 1);
                    } else {
                        std::vector <Data> curQs, curKs, curVs, curOutputs;
                        curQs.resize(batch);
                        curKs.resize(batch);
                        curVs.resize(batch);
                        curOutputs.resize(batch);
                        for (int b = 0; b < batch; b++) {
                            excutor.Run("Split", {
                                    {"input", allDatas[op.datas.find("q")->second]}, {"output", &curQs[b]}
                            }, {}, {{"axis", 1}, {"start", total}, {"end", total + seqLens[b]}});
                            excutor.Run("Split", {
                                    {"input", allDatas[op.datas.find("curk")->second]}, {"output", &curKs[b]}
                            }, {}, {{"axis", 1}, {"start", total}, {"end", total + seqLens[b]}});
                            excutor.Run("Split", {
                                    {"input", allDatas[op.datas.find("curv")->second]}, {"output", &curVs[b]}
                            }, {}, {{"axis", 1}, {"start", total}, {"end", total + seqLens[b]}});
                            total += seqLens[b];
                        }
                        std::vector <int> axis = {0, 2, 1, 3};
                        Data axisData = Data(DataType::INT32PARAM, {(int)axis.size()});
                        axisData.Allocate();
                        for (int i = 0; i < axisData.Count(0); i++) {
                            ((int32_t*)axisData.cpuData)[i] = axis[i];
                        }
                        for (int b = 0; b < batch; b++) {
                            excutor.Run("PermuteSelf", {
                                {"input", (Data*)&curQs[b]}, {"axis", &axisData}
                            }, {}, {});
                            curQs[b].Reshape({-1, curQs[b].dims[2], curQs[b].dims[3]});

                            excutor.Run("PermuteSelf", {
                                {"input", (Data*)&curKs[b]}, {"axis", &axisData}
                            }, {}, {});
                            curKs[b].Reshape({-1, curKs[b].dims[2], curKs[b].dims[3]});

                            excutor.Run("PermuteSelf", {
                                {"input", (Data*)&curVs[b]}, {"axis", &axisData}
                            }, {}, {});
                            curVs[b].Reshape({-1, curVs[b].dims[2], curVs[b].dims[3]});
                        }

                        int unitLen = op.intParams.find("unitLen")->second;
                        for (int b = 0; b < batch; b++) {
                            for (int i = 0; i < 2; i++) {                    
                                auto cache = allDatas[op.datas.find(i == 0 ? "k" : "v")->second + "_" + std::to_string(b)];
                                auto cur = i == 0 ? &curKs[b] : &curVs[b];
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
                        }

                        for (int b = 0; b < batch; b++) {
                            std::string sb = "_" + std::to_string(b);
                            Data *k = allDatas[op.datas.find("k")->second + sb];
                            Data *v = allDatas[op.datas.find("v")->second + sb];
                            Data *mask = allDatas[op.datas.find("mask")->second + sb];
                            excutor.Run("Attention", {
                                    {"q", (Data*)&curQs[b]}, {"k", k}, {"v", v},
                                    {"mask", mask}, {"output", (Data*)&curOutputs[b]}
                            }, {{"scale", op.floatParams.find("scale")->second}}, 
                            {{"maskType", 0}});
                        }

                        for (int b = 0; b < batch; b++) {
                            std::vector <int> axis = {1, 0, 2};
                            Data axisData = Data(DataType::INT32PARAM, {(int)axis.size()});
                            axisData.Allocate();
                            for (int i = 0; i < axisData.Count(0); i++) {
                                ((int32_t*)axisData.cpuData)[i] = axis[i];
                            }
                            Data *output = (Data*)&curOutputs[b];
                            excutor.Run("PermuteSelf", {
                                    {"input", output}, {"axis", &axisData}
                            }, {}, {});
                            output->Reshape({seqLens[b], 1, -1});
                            excutor.Run("PermuteSelf", {
                                    {"input", output}, {"axis", &axisData}
                            }, {}, {});
                        }

                        auto lastOutput = allDatas[op.datas.find("output")->second];
                        for (int b = 0; b < batch; b++) {
                            Data *output = (Data*)&curOutputs[b];
                            if (b == 0) {
                                lastOutput->dataType = output->dataType;
                                std::vector <int> dims = output->dims;
                                dims[1] = 0;
                                lastOutput->Resize(dims);
                                dims[1] = total;
                                lastOutput->Expansion(dims);
                            }
                            excutor.Run("CatDirect", {
                                    {"input0", lastOutput}, {"input1", output}
                            }, {}, {{"axis", 1}});                        
                        }
                    }
                }
            } else if (op.type == "SplitLastTokenStates") {
                std::vector <int> seqLens;
                {
                    auto data = allDatas[op.datas.find("seqLens")->second];
                    for (int i = 0; i < data->Count(0); i++) {
                        seqLens.push_back(((int*)data->cpuData)[i]);
                    }
                }
                auto input = allDatas[op.datas.find("input")->second];
                auto output = allDatas[op.datas.find("output")->second];
                int len = input->dims[1];
                if (len == 1) {
                    output->Resize(input->dims);
                    output->FakeFrom(*input, 0);
                } else if (input->dims[0] == 1 && seqLens.size() > 1) {
                    auto lastOutput = allDatas[op.datas.find("output")->second];
                    int total = 0;
                    for (int b = 0; b < seqLens.size(); b++) {
                        Data output;
                        excutor.Run("Split", {
                            {"input", input}, {"output", (Data*)&output}
                        }, {}, {{"axis", 1}, {"start", total + seqLens[b] - 1}, {"end", total + seqLens[b]}});
                        if (b == 0) {
                            lastOutput->dataType = output.dataType;
                            std::vector <int> dims = output.dims;
                            dims[1] = 0;
                            lastOutput->Resize(dims);
                            dims[1] = seqLens.size();
                            lastOutput->Expansion(dims);
                        }
                        excutor.Run("CatDirect", {
                                {"input0", lastOutput}, {"input1", (Data*)&output}
                        }, {}, {{"axis", 1}});                    
                        total += seqLens[b];    
                    }
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

    void ComputeGraph::DataTypeAs(ComputeGraphNode &input, ComputeGraphNode &input1) {
        this->ops.push_back (
            ComputeGraphOp("DataTypeAs", 
                {{"input", input.name}, {"input1", input1.name}}, 
                {}, {})
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
                {{"input", input.name}, {"output", output.name}}, 
                {}, {})
        );
    }

    void ComputeGraph::Swiglu(ComputeGraphNode &input, ComputeGraphNode &output) {
        this->ops.push_back (
            ComputeGraphOp("Swiglu", 
                {{"input", input.name}, {"output", output.name}}, 
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
        ComputeGraphNode &seqLens,
        float scale, int maskType, int unitLen) {
        this->ops.push_back(
            ComputeGraphOp("FusedAttention", 
                {{"q", q.name}, {"k", k.name}, {"v", v.name},
                {"curk", curk.name}, {"curv", curv.name},
                {"original", original.name},
                {"mask", mask.name}, {"output", output.name}, {"seqLens", seqLens.name}}, 
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

    void ComputeGraph::SplitLastTokenStates(ComputeGraphNode &input, ComputeGraphNode &seqLens, ComputeGraphNode &output) {
        this->ops.push_back (
            ComputeGraphOp("SplitLastTokenStates", 
                {{"input", input.name}, {"output", output.name}, {"seqLens", seqLens.name}}, 
                {}, {})
        );
    }
}