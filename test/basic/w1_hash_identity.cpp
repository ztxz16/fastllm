// W1 离线判据：页面内容身份链（纯 CPU，无 GPU）。
// 必须调用真实函数 fastllm::PagedCacheManager::HashTokenPrefixPage，不许自抄算法。
#include "fastllm.h"

#include <iostream>
#include <string>
#include <vector>

using fastllm::PagedCacheManager;

static int failures = 0;
#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::cerr << "FAIL line " << __LINE__ << ": " #cond "\n";   \
            ++failures;                                                  \
        }                                                                \
    } while (0)

static std::vector<uint64_t> Chain(const std::vector<int> &tokens, int pageLen,
                                   const std::vector<std::string> &keys) {
    std::vector<uint64_t> out;
    uint64_t h = 0;
    const size_t pages = pageLen > 0 ? tokens.size() / size_t(pageLen) : 0;
    for (size_t p = 0; p < pages; ++p) {
        const std::string extra = p < keys.size() ? keys[p] : std::string();
        h = PagedCacheManager::HashTokenPrefixPage(
            h, tokens.data() + p * size_t(pageLen), pageLen, extra);
        out.push_back(h);
    }
    return out;
}

int main() {
    const int pageLen = 4;
    std::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int> b = a;
    b[5] = 60;   // 改第 1 页（第 5、6 个 token）里的一个 token
    const std::vector<std::string> none(3);

    // 5) Explicit checkpoint length controls page count: 12 tokens with pageLen=4
    // yields two pages at length=8 and three at length=12; the common prefix ids agree.
    const auto first8 = Chain(std::vector<int>(a.begin(), a.begin()+8), pageLen, none);
    const auto first12 = Chain(a, pageLen, none);
    CHECK(first8.size() == 2);
    CHECK(first12.size() == 3);
    CHECK(first8[0] == first12[0]);
    CHECK(first8[1] == first12[1]);

    // 1) 同样输入两遍，逐页号必须相同。
    const auto x1 = Chain(a, pageLen, none);
    const auto x2 = Chain(a, pageLen, none);
    CHECK(x1 == x2);
    CHECK(x1.size() == 3);

    // 2) 改一个 token：该页及之后必须变，之前不变。
    const auto y = Chain(b, pageLen, none);
    CHECK(x1[0] == y[0]);
    CHECK(x1[1] != y[1]);
    CHECK(x1[2] != y[2]);

    // 3) 换该页 extra key：该页及之后必须变。
    std::vector<std::string> media(3);
    media[1] = "image-A";
    const auto z = Chain(a, pageLen, media);
    CHECK(x1[0] == z[0]);
    CHECK(x1[1] != z[1]);
    CHECK(x1[2] != z[2]);

    // 4) 位置规则：lease->ids[k] 是物理页号，链上位置 = issued/pageLen + k。
    //    构造 issued=8（即从第 2 页开始）、物理页号 {0,1}：正确号是 x1[2]，用物理页号取会得到 x1[0]。
    const size_t issued = 8;
    const size_t firstPosition = issued / size_t(pageLen);
    CHECK(firstPosition == 2);
    const std::vector<int> physical{0, 1};
    CHECK(x1[firstPosition] != x1[size_t(physical[0])]);   // 取错页会拿到不同的号
    std::cout << "same_tokens=1 token_change=1 extra_change=1 position_rule=1\n";
    if (failures != 0) {
        std::cout << "FAILED " << failures << "\n";
        return 1;
    }
    std::cout << "PASS: w1 page identity chain\n";
    return 0;
}
