#include "graphllm.h"

namespace fastllm {
    class FastllmJsonModelConfig : GraphLLMModelConfig {
    public:
        json11::Json config;

        void Init(const std::string &configString) {
            std::string error;
            config = json11::Json::parse(configString, error);
        }

        void InitParams(GraphLLMModel *model) {
        }

        std::map <std::string, std::vector <std::pair <std::string, DataType> > >
                GetTensorMap(GraphLLMModel *model, const std::vector <std::string> &tensorNames) {
            std::string embeddingName = "";
            std::map <std::string, std::vector <std::pair <std::string, DataType> > > ret;
            for (auto &op : config.array_items()) {
                std::string type = op["type"].string_value();
                std::map <std::string, std::string> weights;
                for (auto &it : op["nodes"].object_items()) {
                    auto &key = it.first;
                    auto &node = it.second;
                    if (node["type"].string_value() == "weight") {
                        weights[key] = node["name"].string_value();
                    }
                }
                if (type == "Embedding") {
                    embeddingName = weights["weight"];
                    ret[embeddingName].push_back(std::make_pair(embeddingName, DataType::DATA_AUTO_EMBEDDING));
                } else if (type == "Linear") {
                    auto linearName = weights["weight"];
                    if (std::find(tensorNames.begin(), tensorNames.end(), linearName) != tensorNames.end()) {
                        ret[linearName].push_back(std::make_pair(linearName, DataType::DATA_AUTO_LINEAR));
                    } else {
                        ret[embeddingName].push_back(std::make_pair(linearName, DataType::DATA_AUTO_LINEAR));
                    }
                }
            }
            for (auto &name : tensorNames) {
                if (ret[name].size() == 0) {
                    ret[name].push_back(std::make_pair(name, DataType::DATA_AUTO_NONE));
                }
            }
            return ret;
        }

        void BuildGraph(GraphLLMModel *model) {
            auto &graph = *(model->GetGraph());
            std::map <std::string, ComputeGraphNode> wNodes;
            for (auto &it : model->weight.weight) {
                wNodes[it.first] = ComputeGraphNode(it.first);
            }

            for (auto &op : config.array_items()) {
                std::string type = op["type"].string_value();
                std::map <std::string, std::string> datas;
                std::map <std::string, float> floatParams;
                std::map <std::string, int> intParams;

                for (auto &it : op["nodes"].object_items()) {
                    auto &key = it.first;
                    auto &node = it.second;
                    std::string type = node["type"].string_value();
                    std::string name = node["name"].string_value();
                    if (type == "data") {
                        datas[key] = name;
                    } else if (type == "weight") {
                        datas[key] = wNodes[name].name;
                    } else if (type == "config.float") {
                        floatParams[key] = atof(model->weight.dicts[name].c_str());
                    } else if (type == "config.int") {
                        intParams[key] = atoi(model->weight.dicts[name].c_str());
                    } else if (type == "constant.float") {
                        floatParams[key] = node["value"].number_value();
                    } else if (type == "constant.int") {
                        intParams[key] = node["value"].int_value();
                    }
                }

                graph.ops.push_back (ComputeGraphOp(type, datas, floatParams, intParams));
            }
        
            OptimizeComputeGraph(graph, model->weight);
            graph.Update();
        }
    };
    REGISTERGRAPHMODELCONFIG(fastllmJson, FastllmJsonModelConfig)
}