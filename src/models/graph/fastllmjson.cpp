#include "graphllm.h"

namespace fastllm {
    class FastllmJsonModelConfig : GraphLLMModelConfig {
    public:
        json11::Json json, graphJson, configJson, tokenizerConfigJson, generationConfigJson;

        void Init(const std::string &configString) {
            std::string error;
            json = json11::Json::parse(configString, error);
            graphJson = json["graph"];
            configJson = json["config"];
            tokenizerConfigJson = json["tokenizer_config"];
            generationConfigJson = json["generation_config"];
        }

        void InitParams(GraphLLMModel *model) {
            if (configJson["max_positions"].is_number()) {
                model->max_positions = configJson["max_positions"].int_value();
            }
            if (configJson["rope_base"].is_number()) {
                model->rope_base = configJson["rope_base"].number_value();
            }
            if (configJson["rope_factor"].is_number()) {
                model->rope_factor = configJson["rope_factor"].number_value();
            }

            if (configJson["pre_prompt"].is_string()) {
                model->pre_prompt = configJson["pre_prompt"].string_value();
            }
            if (configJson["user_role"].is_string()) {
                model->user_role = configJson["user_role"].string_value();
            }
            if (configJson["bot_role"].is_string()) {
                model->bot_role = configJson["bot_role"].string_value();
            }
            if (configJson["history_sep"].is_string()) {
                model->history_sep = configJson["history_sep"].string_value();
            }
        }

        std::map <std::string, std::vector <std::pair <std::string, DataType> > >
                GetTensorMap(GraphLLMModel *model, const std::vector <std::string> &tensorNames) {
            std::string embeddingName = "";
            std::map <std::string, std::vector <std::pair <std::string, DataType> > > ret;
            for (auto &op : graphJson.array_items()) {
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

            for (auto &op : graphJson.array_items()) {
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