#include "models/lingbot_world/runtime.h"
#include <stdexcept>

namespace fastllm { namespace lingbot {
Tensor TextEncoder::encode(const std::vector<int> &tokens) {
    if(tokens.empty() || tokens.size()>512) throw std::invalid_argument("LingBot text must contain 1..512 tokens");
    auto padded=tokens; padded.resize(512,0);
    auto x=rt_.embedding(weights_.get("token_embedding.weight"),padded);
    // Load only the current block so text encoding leaves room for world caches.
    weights_.clear();
    for(int i=0;i<24;++i) {
        auto p="blocks."+std::to_string(i)+".";
        auto y=rt_.t5Norm(x,weights_.get(p+"norm1.weight"));
        auto q=weights_.linear(y,p+"attn.q"),k=weights_.linear(y,p+"attn.k"),v=weights_.linear(y,p+"attn.v");
        y=rt_.relativeAttention(q,k,v,weights_.get(p+"pos_embedding.embedding.weight"),tokens.size());
        y=weights_.linear(y,p+"attn.o");
        x=rt_.add(x,y);
        y=rt_.t5Norm(x,weights_.get(p+"norm2.weight"));
        auto gate=weights_.linear(y,p+"ffn.gate.0");
        gate=rt_.t5Gelu(gate);
        y=weights_.linear(y,p+"ffn.fc1");
        y=rt_.multiply(gate,y);
        y=weights_.linear(y,p+"ffn.fc2");
        x=rt_.add(x,y);
        weights_.clear();
    }
    x=rt_.t5Norm(x,weights_.get("norm.weight"));
    return x.rowSlice(0,tokens.size());
}
}} // namespace fastllm::lingbot
