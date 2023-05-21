//
// Created by huangyuyang on 5/11/23.
//

#ifndef TEST_FASTLLM_H
#define TEST_FASTLLM_H

#include <vector>
#include <cstdint>
#include <string>
#include <map>
#include <unordered_map>
#include <cmath>
#include <algorithm>

namespace fastllm {
    void SetThreads(int t);

    struct LowBitConfig {
        int bit;
        float min, max;
        uint8_t zeroPoint;
        float scale;

        LowBitConfig(float min, float max, int bit) {
            this->min = min;
            this->max = max;
            this->bit = bit;
            Reset();
        }

        LowBitConfig () {

        }

        void Reset() {
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
        }

        uint8_t quantization(const float &realNumber) const {
            return (uint8_t) (std::min((double)((1 << bit) - 1), std::max(realNumber / scale + zeroPoint + 0.5, 0.0)));
        }

        float invQuantization(const uint8_t &qNumber) const {
            return (scale * ((float) qNumber - (float) zeroPoint));
        }
    };

    enum DataType {
        FLOAT32 = 0, BFLOAT16 = 1, INT16 = 2, INT8 = 3, INT4 = 4, INT2 = 5, BIT = 6
    };

    enum WeightType {
        NONE = 0, LINEAR = 1, EMBEDDING = 2
    };

    struct Data {
        WeightType weightType = WeightType::NONE; // 权重类型，NONE代表非权重（或未知权重）

        DataType dataType = DataType::FLOAT32; // 数据类型
        int unitSize, unitSizeDiv = 1; // 单个元素的字节数 = unitSIze / unitSizeDiv

        std::vector <int> dims; // 数据形状
        std::vector <uint64_t> strides; // 跨度

        uint64_t expansionSize = 0; // 扩容后的尺寸
        uint8_t *cpuData = nullptr; // 数据指针

	    void *cudaData = nullptr;

        // 这两个参数用于量化，对FLOAT数据不适用
        int perChannelAxis = -1; // 沿哪个轴分通道量化，-1代表没有分通道
        std::vector <LowBitConfig> perChannelsConfigs; // perChannelsConfigs[i]代表第i个通道的min, max; 如果没有分通道，perChannelsConfigs[0]代表全局min, max
        std::vector <int> weightSum; // 作为权重时，有时候需要存一些和加速计算

        Data () {};

        Data (DataType type);

        Data (DataType type, const std::vector <int> &dims); // 构造函数

        // 构造函数，创建好之后从data复制数据
        // data中是原始数据，如果type不是float那么需要量化
        Data (DataType type, const std::vector <int> &dims, const std::vector <float> &data);

        ~Data(); // 析构函数

        Data (const Data &ori); // 深拷贝

        void CopyFrom(const Data &ori); // 复制

        uint64_t GetBytes() const; // 获取总字节数

        void Allocate(); // 分配内存

        void Allocate(float v); // 分配内存并初始化

        void Expansion(uint64_t size); // 将尺寸扩容为size * unitSize，并保留之前的数据；之后分配内存时如果未达到扩容的尺寸，则直接复用

        void UpdateUnitSize(); // 更新unitSize

        void Resize(const std::vector <int> &dims); // 更改尺寸

        void Reshape(const std::vector <int> &dims); // 更改尺寸,但不修改数据

        uint64_t Count(int i) const; // dims[i] * strides[i]

        void Print() const; // 输出

        void Permute(const std::vector <int> &axis); // 转置

        void CalcWeightSum(); // 计算WeightSum
    };

    struct Tokenizer {
        struct TrieNode {
            int tokenId;
            std::map <int, TrieNode*> next;
            TrieNode();
        };
        TrieNode *root;

        std::unordered_map <int, std::string> tokenToStringDict;

        Tokenizer ();

        ~Tokenizer();

        void Clear(); // 清空分词器

        void Insert(const std::string &s, int tokenId); // 插入一个token

        Data Encode(const std::string &s); // 编码

        std::string Decode(const Data &data); // 解码
    };

    struct WeightMap {
        int versionId;

        Tokenizer tokenizer;

        std::map <std::string, Data> weight;

        void LoadFromFile(const std::string &fileName); // 从文件读取

        void SaveLowBitModel(const std::string &fileName, int bit); // 存储成量化模型

        Data &operator [] (const std::string &key);
    };

    void Embedding(const Data &input, Data &weight, Data &output);

    void LayerNorm(const Data &input, const Data &gamma, const Data &beta, int axis, Data &output);

    void Linear(const Data &input, Data &weight, const Data &bias, Data &output);

    void Split(const Data &input, int axis, int start, int end, Data &output);

    void Cat(const Data &input0, const Data &input1, int axis, Data &output);

    void CatDirectAxis0(Data &input0, const Data &input1); // 直接把input1的数据拷贝到input0后面（axis = 0的Cat操作，需要input0提前扩容了足够的空间）

    void MatMulTransB(const Data &input0, const Data &input1, Data &output, float alpha = 1.0);

    void Softmax(const Data &input, Data &output, int axis);

    void GeluNew(const Data &input, Data &output);

    void Mul(const Data &input, float v, Data &output);

    void AddTo(Data &input0, const Data &input1); // input0 += input1

    void AddTo(Data &input0, const Data &input1, float alpha); // input0 += input1 * alpha

    void Permute(const Data &input, const std::vector<int> &axis, Data &output); // 转置
}

#endif //TEST_FASTLLM_H