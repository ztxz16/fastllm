//
// Created by huangyuyang on 5/11/23.
//

#ifndef TEST_FASTLLM_H
#define TEST_FASTLLM_H

#define _USE_MATH_DEFINES
#include <vector>
#include <cstdint>
#include <string>
#include <map>
#include <set>
#include <queue>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <functional>
#include <memory>
#include <locale>
#include <codecvt>
#include "devices/cpu/alivethreadpool.h"
#include "json11.hpp"

#ifdef USE_SENTENCEPIECE
#include <sentencepiece_processor.h>
#endif

namespace fastllm {
    void SetDeviceMap(const std::map <std::string, int> &deviceMap);
    void SetMoeDeviceMap(const std::map <std::string, int> &moeDeviceMap);

    std::map <std::string, int> GetDeviceMap();
    std::map <std::string, int> GetMoeDeviceMap();

    void PrintInstructionInfo();
    void SetThreads(int t);
    void SetLowMemMode(bool m);
    void SetKVCacheInCPU(bool kvCacheInCPU);
    void SetHistoryCacheInCPU(bool v);
    bool GetLowMemMode();
    void SetCudaEmbedding(bool v);
    bool GetCudaEmbedding();
    int GetThreads();
    bool GetKVCacheInCPU();
    bool GetHistoryCacheInCPU();
    AliveThreadPool *GetAlivePool();

    struct GenerationConfig {
        int output_token_limit = -1; // 最多输出多少, <= 0代表无限制
        int output_token_least = 0; // 最低输出的多少
        int input_token_length = 0;
        int last_n = 64; // 末尾last_n个token计入重复惩罚
        float repeat_penalty = 1.0f; // 重复惩罚系数，1.0代表不惩罚
        int top_k = 1; // top_k采样
        float top_p = 1.0; // top_p采样
        float temperature = 1.0; // 温度参数，一般在0.1 ~ 1.0之间，设大这个参数可以带来结果的多样性
        bool output_logits = false; // 是否返回logits
        bool enable_hash_id = false; // 给会话添加hash id
        bool add_special_tokens = true; // prompt添加special tokens（chatglm模型生效）
        std::multiset <int> stop_token_ids;

        bool IsSimpleGreedy() const {
            if (fabs(repeat_penalty - 1) > 1e-8) {
                return false;
            }
            if (top_k > 1) {
                return false;
            }
            return true;
        }
    };

    struct LastTokensUnit {
        int tot = 0;
        std::multiset <int> tokenSet;
        std::queue <int> tokenQueue;

        LastTokensUnit () {}

        LastTokensUnit (int tot) {
            Init(tot);
        }

        void Init(int tot) {
            this->tot = tot;
            tokenSet.clear();
            while (tokenQueue.size() > 0) {
                tokenQueue.pop();
            }
        }

        void Push(int id) {
            if (tokenQueue.size() == tot) {
                tokenSet.erase(tokenSet.find(tokenQueue.front()));
                tokenQueue.pop();
            }
            tokenQueue.push(id);
            tokenSet.insert(id);
        }
    };

    struct LastTokensManager {
        std::vector <LastTokensUnit> units;

        LastTokensManager () {}

        LastTokensManager (int batch, int lastN) {
            units.resize(batch);
            for (int i = 0; i < batch; i++) {
                units[i].Init(lastN);
            }
        }
    };

    struct LowBitConfig {
        int bit;
        float min, max;
        uint8_t zeroPoint;
        float scale;
        int type; // 0: 有zero点 1: 不需要zero点

        LowBitConfig(float min, float max, int bit, int type) {
            this->min = min;
            this->max = max;
            this->bit = bit;
            this->type = type;
            Reset();
        }

        LowBitConfig () {

        }

        void Reset() {
            /*if (type == 1) {
                this->scale = (max - min) / 15.0;
                return;
            }*/
            /*if (type == 1) {
                this->scale = std::max(fabs(max), fabs(min)) / 7.0;
                this->min = this->scale * (-7.0);
                return;
            }*/
            min = std::min(min, 0.f);
            max = std::max(max, 0.f);

            const float qmin = 0;
            const float qmax = (1 << bit) - 1;
            scale = (max - min) / (qmax - qmin);
            const float initial_zero_point = qmin - min / scale;
            zeroPoint = 0;
            if (initial_zero_point < qmin) {
                zeroPoint = qmin;
            } else if (initial_zero_point > qmax) {
                zeroPoint = qmax;
            } else {
                zeroPoint = static_cast<uint8_t>(std::round(initial_zero_point));
            }

            if (type == 1) {
                this->min = -this->scale * zeroPoint;
                return;
            }
        }

        uint8_t quantization(const float &realNumber) const {
            if (type == 0) {
                return (uint8_t) (std::min((double) ((1 << bit) - 1),
                                           (double) std::max(realNumber / scale + zeroPoint + 0.5, 0.0)));
            } else {
                return (uint8_t) (std::max(0.f, std::min(15.f, (realNumber - min) / scale + 0.5f)));
            }
        }

        float invQuantization(const uint8_t &qNumber) const {
            if (type == 0) {
                return (scale * ((float) qNumber - (float) zeroPoint));
            } else {
                return min + scale * qNumber;
            }
        }
    };

    enum DataType {
        FLOAT32 = 0, BFLOAT16 = 1, INT16 = 2, INT8 = 3, INT4 = 4, INT2 = 5, BIT = 6, FLOAT16 = 7,
        INT4_NOZERO = 8, // 不用zeroPoint的int4, floatValue = min + uint4Value * scale
        INT4_GROUP = 9, // 不用zeroPoint的int4, floatValue = min + uint4Value * scale, 且使用分组量化
        FP8_E4M3 = 10,
        INT2_GROUP = 11, // 不用zeroPoint的int2, floatValue = min + uint4Value * scale, 且使用分组量化
        BASE3_GROUP = 12, // 三元量化，-1 0 1
        INT32PARAM = 100, // int32的参数，这种类型的数据永远存在CPU上
        DATA_AUTO_NONE = 99999, DATA_AUTO_LINEAR, DATA_AUTO_EMBEDDING, DATA_AUTO_CONV
    };

    static std::map <DataType, int> DataTypeBits = {
        {DataType::FLOAT32, 32}, {DataType::BFLOAT16, 16}, {DataType::INT16, 16}, 
        {DataType::INT8, 8}, {DataType::INT4, 4}, {DataType::INT2, 2}, {DataType::BIT, 1}, 
        {DataType::FLOAT16, 16}, {DataType::INT4_NOZERO, 4}, {DataType::INT4_GROUP, 4},
        {DataType::FP8_E4M3, 8}, {DataType::INT2_GROUP, 2}, {DataType::BASE3_GROUP, 2}
    };

    enum DataDevice {
        CPU = 0, CUDA = 1
    };

    enum WeightType {
        NONE = 0, LINEAR = 1, EMBEDDING = 2, CONV2D = 3, AUTO = 99999
    };

    struct FileMmap {
    public:
        FileMmap(const std::string &path);
        ~FileMmap();

        char *data;
        size_t size;
    };

    struct ModelLoader {
        ModelLoader(const char *buffer, size_t size) : data(buffer), size(size), ptr(buffer) {}

        int64_t tell() const { return ptr - data; }

        void seek(int64_t offset, int whence);

        template <typename T>
        T read_basic() {
            T obj = *(T *)ptr;
            ptr += sizeof(T);
            return obj;
        }

        std::string ReadString();
        int ReadInt();
        float ReadFloat();
        uint8_t* ReadBytes(uint64_t bytes);

        const char *const data;
        size_t size;
        const char *ptr;
    };

    class Data {
    public:
        bool isFake = false; // 没有创建空间，指向别的data（无需销毁）

        long long cacheUid = 0; // 用来标注Cache id
        bool isKVCache = false; // 是否是KV Cache TODO: 做一些KVCache的管理

        bool lockInCPU = false; // 如果lock在CPU上，那么不允许移动到其余设备
        WeightType weightType = WeightType::NONE; // 权重类型，NONE代表非权重（或未知权重）

        DataType dataType = DataType::FLOAT32; // 数据类型
        int unitSize, unitSizeDiv = 1; // 单个元素的字节数 = unitSIze / unitSizeDiv

        std::vector <int> dims; // 数据形状
        std::vector <uint64_t> strides; // 跨度

        uint64_t expansionSize = 0; // 扩容后的尺寸
        uint64_t expansionBytes = 0; // 扩容后的字节数
        std::vector <int> expansionDims; // 预扩容的形状
        uint8_t *cpuData = nullptr; // 数据指针

	    void *cudaData = nullptr;
        std::vector <void*> extraCudaData;
        std::vector <void*> extraCudaHalfData;

        void *deviceData = nullptr;
        std::vector <void*> extraDeviceData;

        DataDevice dataDevice = DataDevice::CPU;
        std::vector <int> dataDeviceIds;

        // 以下参数用于量化，对FLOAT数据不适用
        int perChannelAxis = -1; // 沿哪个轴分通道量化，-1代表没有分通道
        int group = -1, groupCnt = -1; // 分组量化，group代表组数，groupCnt代表每组有多少个元素，-1代表不使用分组量化

        // FP8的分组量化， [blockK, blockM]的小矩阵为一组
        int blockK = -1, blockM = -1;

        // 以下为每个通道/分组的量化参数
        // 1. 若不使用分通道量化，那么总组数 = 1
        // 2. 若使用分通道量化，那么总组数 = 通道数
        // 3. 若使用分组量化，那么总组数 = 通道数 * 组数(group)
        std::vector <LowBitConfig> perChannelsConfigs; // perChannelsConfigs[i]代表第i个通道的min, max; 如果没有分通道，perChannelsConfigs[0]代表全局min, max
        std::vector <float> scales, mins;
        std::vector <int> zeros;
        std::vector <int> weightSum; // 作为权重时，有时候需要存一些和加速计算

        std::vector <uint16_t> halfScales; // 某些量化方式使用float16的scales

        std::string name; // weightName
        std::string fileName;
        long long filePos;
        std::shared_ptr<FileMmap> mapFile;

        bool directMemory = false; // 直接分配/释放Memory，不经过缓存

        bool multiDeviceData = false;
        std::map <int, Data*> multiDeviceDatas;
        
        Data () {};

        Data (DataType type);

        Data (DataType type, const std::vector <int> &dims); // 构造函数

        Data (DataType type, const std::vector <int> &dims, DataDevice device, void *ptr); // 构造函数，使用已有数据地址的Fake data

        // 构造函数，创建好之后从data复制数据
        // data中是原始数据，如果type不是float那么需要量化
        Data (DataType type, const std::vector <int> &dims, const std::vector <float> &data);

        ~Data(); // 析构函数

        Data (const Data &ori); // 深拷贝

        void CreateFromOriData(WeightType weightType, DataType oriDataType, uint8_t *oriData, float *oriMins, float *oriScales, 
                int groupCnt = -1, int blockK = -1, int blockM = -1); // 从oriData中创建

        void CopyFrom(const Data &ori); // 复制

        void FakeFrom(const Data &ori, size_t offset); // 将data指针指向ori的data + offset，delete时不销毁

        uint64_t GetBytes() const; // 获取总字节数

        void Allocate(); // 分配内存

        void Allocate(float v); // 分配内存并初始化

        void Expansion(const std::vector <int> &dims); // 预扩容到相应尺寸

        void MallocSpace(uint64_t size); // 在设备上分配

        void FreeSpace(); // 回收设备上的内存

        void UpdateUnitSize(); // 更新unitSize

        void Resize(const std::vector <int> &dims); // 更改尺寸

        void Reshape(const std::vector <int> &dims); // 更改尺寸,但不修改数据

        uint64_t Count(int i) const; // dims[i] * strides[i]

        void PrintShape() const; // 输出形状

        std::vector<int> Shape() const; 

        void Print() const; // 输出

        void CalcWeightSum(); // 计算WeightSum

        void ToDevice(DataDevice device); // 移动到指定device

        void ToDevice(DataDevice device, const std::vector <int> &deviceIds); // 移动到指定device

        void ToDevice(void *device);

        void SetMapFile(std::shared_ptr<FileMmap> file) {
        	mapFile = file;
        }

        void SetKVCache();

        // 计算形成Fastllm格式需要多少Bytes
        uint64_t GetFastllmFormateBytes();

        // 导出成Fastllm格式
        void ExportFastllmFormat(uint8_t *bytes);

        // 从Fastllm格式中创建
        void CreateFromFastllmFormat(uint8_t *datas, uint64_t len);
    };

    struct PartitionLinkNode {
        std::pair <int, int> *cur = nullptr;
        PartitionLinkNode *next = nullptr;
        PartitionLinkNode *prev = nullptr;
        int id = -1;

        PartitionLinkNode *Skip(int t) {
            PartitionLinkNode *ret = this;
            while (t--) {
                if (ret != nullptr) {
                    ret = ret->next;
                }
            }
            return ret;
        }
    };

    struct Tokenizer {
        enum TokenizerType {
            BPE = 0,
            NORMAL = 1,
            QWEN = 2,
            GLM = 3,
            BERT = 4
        };

        struct TrieNode {
            int tokenId;
            float score;
            std::map <int, TrieNode*> next;
            TrieNode();
        };
        struct Symbol {
            TrieNode *node;
            char *s;
            int pos, len;
            int prev, next;
            int fixId;

            Symbol (Tokenizer::TrieNode *node,
                    char *s, int pos, int len,
                    int prev, int next, int fixId) {
                this->node = node;
                this->s = s;
                this->pos = pos;
                this->len = len;
                this->prev = prev;
                this->next = next;
                this->fixId = fixId;
            }
        };
        struct SymbolPairs {
            float score;
            int l, r, size;

            SymbolPairs(float score, int l, int r, int size) {
                this->score = score;
                this->l = l;
                this->r = r;
                this->size = size;
            }
        };

        friend bool operator < (const SymbolPairs &a, const SymbolPairs &b) {
            return a.score < b.score || (a.score == b.score && a.l > b.l);
        }

        json11::Json tokenizerConfig;
        std::string chatTemplate = "";

        TrieNode *root;

        TrieNode *specialRoot = nullptr;

        TokenizerType type = TokenizerType::BPE;

        bool addDummyPrefix = true;   // 是否在首位添加空格
        bool removeExtraWhitespaces = true;   // 是否将多个空格合并为一个
        bool byteAsChar = false;  // 是否将byte变为展示字符

        std::unordered_map <int, std::string> tokenToStringDict;
        std::unordered_map <int, float> tokenToScoreDict;
        std::unordered_map <std::string, int> stringToTokenDict;
        std::vector <std::string> specialTokens;

        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        std::unordered_map <wchar_t, wchar_t> byteCharDict;
        std::unordered_map <wchar_t, wchar_t> charByteDict;
#ifdef USE_SENTENCEPIECE
        std::unique_ptr<sentencepiece::SentencePieceProcessor> spProcessor;
#endif

        Tokenizer ();

        ~Tokenizer();

        void Clear(); // 清空分词器

        void TryMergePairs(std::vector<Symbol> &symbols, int l, int r, std::priority_queue <SymbolPairs> &q); // 插入备选symbol

        int GetRank(std::vector <Symbol> &symbols, PartitionLinkNode *cur, int skip);

        int GetRank(std::vector<Symbol> &symbols,  std::vector<std::pair<int, int>> &partitions, int idx, int skip);

        void Insert(const std::string &s, int tokenId, float score = 1.0f); // 插入一个token

        void SetSpecialTokens(const std::map <std::string, int> &specialTokens); // 设置需要优先处理的特殊token

        void SetTokenizerConfig(const json11::Json &config);

        std::string Normalize(const std::string &ori); // 字符规范化

        Data Encode(const std::string &s); // 编码

        std::string Decode(const Data &data); // 解码

        std::string DecodeTokens(const std::vector <int> &tokens); // 解码

        int GetTokenId(const std::string &s); // 获取s对应的tokenid

        std::string GetToken(int id); // 获取id对应的token
    };

    std::string GetModelTypeFromFile(const std::string &fileName);

    struct WeightMap {
        int versionId = 2;

        Tokenizer tokenizer;

        std::map <std::string, std::string> dicts;

        std::unordered_map <std::string, Data> weight;

        std::map <std::string, std::map <std::string, std::string>> peftDict;

        std::set <std::string> embeddingNames;

        std::set <std::string> linearNames;

        void LoadFromFile(const std::string &fileName); // 从文件读取

        void SaveLowBitModel(const std::string &fileName, int bit); // 存储成量化模型, bit = 0代表直接存

        void AddTokenizerWord(const std::string &key, int value, float score); // 增加一个词

        void AddDict(const std::string &key, const std::string &value); // 插入一个词条

        void AddAdapterDict(const std::string &name, const std::string &key, const std::string &value);

        void AddEmptyWeight(const std::string &key, const std::vector<int> &dims, fastllm::DataType dataType);

        void AddWeight(const std::string &key, const std::vector <int> &dims,
                       DataType dataType, WeightType weightType, DataType oriDataType, uint8_t *oriData,
                       int groupCnt = -1); // 插入一个权重

        void ReleaseWeight(); // 释放所有权重占用的空间

        void AddQLinearWeight(const std::string &key, const std::vector <int> &dims,
                              int bit, float *scales, uint8_t *oriData); // 插入一个Qlinear层的权重，量化规则为float value = scales * oriData

        WeightType GetWeightType(const std::string &key); // 获取某个权重的类型（若未判断出来，则为None)

        Data &operator [] (const std::string &key);
    };

    void *GetExecutor();

    void ClearProfiler();

    void PrintProfiler();

    void ApplyDeviceMap(const std::map <std::string, int> &deviceMap, int current, int total); // 执行到了current, 一共total，使用deviceMap切换设备

    int LLMSamplingOnly(Data &logits, int outerOffset, const GenerationConfig &config);

    int LLMSampling(Data &logits, int outerOffset,
                    const GenerationConfig &config, const LastTokensUnit &tokens); // 对logits里[outerOffset * vocabSize, (outerOffset + 1) * vocabSize]做Sampling

    void ToDataType(const Data &input, DataType dataType);
    void ToDataType(const Data &input, Data &output, DataType dataType);

    void CopyKVCache(Data &oldCache, Data &newCache, int oldBsStart, int newBsStart, int bs, int offset);

    bool CanRunMergeMOE(const Data &input, std::vector <Data*> &biass);
    void MergeMOE(const Data &input, const Data &logits, Data &gateBias, std::vector <Data*> &weights, std::vector <Data*> &biass, 
                Data &w1, Data &w2, Data &w3,
                float routeScale, float sharedScale, int topk, bool needNorm, Data &output);
    
    void MergeMLA(Data &qNope, Data &qPe, Data &kvCache, Data &peCache, const Data &mask, Data &output, float softmaxScale);

    void Attention(const Data &q, const Data &k, const Data &v, const Data &mask, Data &output,
                   int group, float scale, int attentionType);

    void AttentionBatch(std::vector <Data*> &q, std::vector <Data*> &k, std::vector <Data*> &v,
                        std::vector <Data*> &mask, std::vector <Data*> &output,
                        int group, float scale, int attentionType);

    void Conv2D(const Data &input, Data &weight, Data &bias, int inputChannels, int outputChannels, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, Data &output);

    void Embedding(const Data &input, Data &weight, Data &output);

    void RMSNorm(const Data &input, const Data &weight, float eps, Data &output);

    void LayerNorm(Data &input, Data &gamma, Data &beta, int axis, Data &output);

    void Linear(Data &input, Data &weight, const Data &bias, Data &output);

    enum LinearExType {
        ExTypeNone = 0,
        ExSwiglu = 1,
        ExGelu = 2,
        ExSilu = 3
    };
    
    bool CanRunLinearEx(LinearExType exType);

    bool CanRunMergeAttention();
    
    void MergeAttention(Data &input, Data &weight0, Data &bias0, Data &weight1, Data &bias1, 
                        Data &qkv, Data &q, Data &k, Data &v,
                        int qNum, int kvNum, int headDim, int rotDim, float attentionScale,
                        const Data &positionIds, Data &sinData, Data &cosData,
                        std::vector <Data*> &keys, std::vector <Data*> &values, std::vector <Data*> &masks, 
                        Data &output);

    bool CanRunMLP();

    void MLP(Data &input, Data &weight0, const Data &bias0, Data &weight1, const Data &bias1, Data &output); // mlp

    void LinearEx(Data &input, Data &weight, const Data &bias, Data &output,
                    LinearExType exType); // 扩展Linear，可以接后续操作

    void Split(const Data &input, int axis, int start, int end, Data &output);

    void Repeat(const Data &input, int axis, int repeatTimes, Data &output);

    void Cat(const Data &input0, const Data &input1, int axis, Data &output);

	void CatDirect(Data &input0, const Data &input1, int axis); // 直接把input1的数据拷贝到input0后面（需要input0提前扩容了足够的空间）

    void MatMul(const Data &input0, const Data &input1, Data &output, float alpha = 1.0, int group = 1);

    void MatMulTransB(const Data &input0, const Data &input1, Data &output, float alpha = 1.0, int group = 1);

    void Softmax(const Data &input, Data &output, int axis);

    void Silu(const fastllm::Data &input, fastllm::Data &output);

    void TanH(const Data &input, Data &output);

    void Relu(const Data &input, Data &output);

    void Sigmoid(const Data &input, Data &output);

    void Normalize(const Data &input, Data &output, int axis);

    void Gelu(const Data &input, Data &output);
    
    void GeluNew(const Data &input, Data &output);

    void Swiglu(const fastllm::Data &input, fastllm::Data &output);

    void Mul(const Data &input, float v, Data &output);

    void MulTo(Data &input0, const Data &input1); // input0 *= input1

    void AddTo(Data &input0, const Data &input1, float alpha = 1.0); // input0 += input1 * alpha

    void AttentionMask(Data &input, const Data &mask, float maskValue); // 把input里对应位置mask中为1的部分变成maskValue

    void AttentionExtendedMask(Data &input, const Data &mask); // bert中的extended mask

    void AlibiMask(Data &input, const Data &mask, float maskValue); // alibi mask

    void Permute(const Data &input, const std::vector<int> &axis, Data &output); // 转置

    void PermuteSelf(const Data &input, const std::vector<int> &axis); // 转置

    void TopK(const Data &input, Data &output, int topK); // 求topk

    void RotatePosition2D(Data &input, const Data &positionIds, Data &sinData, Data &cosData, int rotaryDim); // 2D position

    void NearlyRotatePosition2D(Data &input, const Data &positionIds, Data &sinData, Data &cosData, int rotaryDim); // 2D position, 相邻的元素旋转

    void LlamaRotatePosition2D(Data &input, const Data &positionIds, Data &sinData, Data &cosData, int rotaryDim); // 2D position for llama

    void RepeatPenalty(Data &input, const Data &penalty, const Data &penaltyScale); // 重复惩罚

    void ApplyLognAttn(Data &input, const Data &lognAttn, const Data &positionIds);

    void MulBatch(std::vector <Data*> &input, float v, std::vector <Data*> &output);

    void SplitBatch(const Data &input, int axis, int part, std::vector <Data*> &outputs); // 将input沿着axis轴切开，每份axis上的尺寸为1，放到outputs里

    void CatBatch(std::vector <Data*> &input, int axis, Data &outputs); // 将input沿着axis轴合起来，每份axis上的尺寸为1，放到output里

    void MatMulBatch(std::vector <Data*> &input0, std::vector <Data*> &input1, std::vector <Data*> &output, float alpha = 1.0);

    void MatMulTransBBatch(std::vector <Data*> &input0, std::vector <Data*> &input1, std::vector <Data*> &output, float alpha = 1.0);

    void SoftmaxBatch(std::vector <Data*> &input, std::vector <Data*> &output, int axis);

    void CatDirectBatch(std::vector <Data*> &input0, std::vector <Data*> &input1, int axis);

    void AppendKVCacheBatch(std::vector <Data*> &cache, const Data &input);

    void LoraLayer(Data &input, Data &weight, Data &loraA, Data &loraB, const Data &bias, Data &output, 
                   std::map <std::string, std::string> loraConfig);

    void IA3Layer(Data &input, Data &weight, Data &ia3_l, Data &bias, Data &output,
                  std::map <std::string, std::string> ia3Config);
}

#endif //TEST_FASTLLM_H
