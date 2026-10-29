#include "models/hy_nomas/runtime.h"
#include "json11.hpp"
#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace fastllm { namespace nomas {
namespace {
#include "nfc_tables.inc"
struct UnicodeRange { uint32_t first,last; uint8_t mask; };
// Reuse fastllm's existing Unicode tables, rather than adding a regex library.
constexpr UnicodeRange categories[]={
#include "../../unicode-categories.inc"
};
unsigned category(uint32_t c) {
    auto it=std::lower_bound(std::begin(categories),std::end(categories),c,
        [](const UnicodeRange &r,uint32_t v) { return r.last<v; });
    return it!=std::end(categories) && it->first<=c?it->mask:0;
}
bool space(uint32_t c) {
    return (c>=9 && c<=13) || c==32 || c==0x85 || c==0xa0 || c==0x1680 ||
        (c>=0x2000 && c<=0x200a) || c==0x2028 || c==0x2029 || c==0x202f || c==0x205f || c==0x3000;
}
std::string utf8(uint32_t c) {
    if(c<128) return std::string(1,char(c));
    if(c<2048) return std::string{char(0xc0|(c>>6)),char(0x80|(c&63))};
    if(c<65536) return std::string{char(0xe0|(c>>12)),char(0x80|((c>>6)&63)),char(0x80|(c&63))};
    return std::string{char(0xf0|(c>>18)),char(0x80|((c>>12)&63)),char(0x80|((c>>6)&63)),char(0x80|(c&63))};
}
struct Char { uint32_t c; size_t offset; unsigned cat; };
std::vector<Char> decode(const std::string &text) {
    std::vector<Char> chars;
    for(size_t i=0;i<text.size();) {
        size_t start=i; uint32_t c=uint8_t(text[i++]);
        int extra=c<0x80?0:(c>=0xc2 && c<=0xdf)?1:(c>=0xe0 && c<=0xef)?2:(c>=0xf0 && c<=0xf4)?3:-1;
        if(extra<0 || i+extra>text.size()) throw std::invalid_argument("invalid UTF-8 prompt");
        uint32_t min=extra==1?0x80:extra==2?0x800:extra==3?0x10000:0;
        if(extra) c&=(1u<<(6-extra))-1;
        while(extra--) { uint8_t next=text[i++]; if((next&0xc0)!=0x80) throw std::invalid_argument("invalid UTF-8 continuation"); c=(c<<6)|(next&63); }
        if(c<min || c>0x10ffff || (c>=0xd800 && c<=0xdfff)) throw std::invalid_argument("invalid UTF-8 codepoint");
        chars.push_back({c,start,category(c)});
    }
    return chars;
}
unsigned combining(uint32_t c) {
    auto it=std::lower_bound(std::begin(nfcCCC),std::end(nfcCCC),c,[](const auto &r,uint32_t v) { return r[0]<v; });
    return it!=std::end(nfcCCC) && (*it)[0]==c?(*it)[1]:0;
}
void decompose(uint32_t c,std::vector<uint32_t> &out) {
    if(c>=0xac00 && c<0xd7a4) {
        uint32_t index=c-0xac00; out.push_back(0x1100+index/588); out.push_back(0x1161+index%588/28);
        if(index%28) out.push_back(0x11a7+index%28); return;
    }
    auto it=std::lower_bound(std::begin(nfcDecomposition),std::end(nfcDecomposition),c,[](const auto &r,uint32_t v) { return r[0]<v; });
    if(it!=std::end(nfcDecomposition) && (*it)[0]==c) { decompose((*it)[1],out); if((*it)[2]) decompose((*it)[2],out); }
    else out.push_back(c);
}
uint32_t compose(uint32_t a,uint32_t b) {
    if(a>=0x1100 && a<0x1113 && b>=0x1161 && b<0x1176) return 0xac00+(a-0x1100)*588+(b-0x1161)*28;
    if(a>=0xac00 && a<0xd7a4 && (a-0xac00)%28==0 && b>0x11a7 && b<0x11c3) return a+b-0x11a7;
    auto pair=std::make_pair(a,b);
    auto it=std::lower_bound(std::begin(nfcComposition),std::end(nfcComposition),pair,[](const auto &r,const auto &v) { return std::make_pair(r[0],r[1])<v; });
    return it!=std::end(nfcComposition) && (*it)[0]==a && (*it)[1]==b?(*it)[2]:0;
}
std::string nfc(const std::string &text) {
    std::vector<uint32_t> decomposed;
    for(auto &c:decode(text)) decompose(c.c,decomposed);
    for(size_t i=1;i<decomposed.size();++i) {
        size_t j=i; unsigned c=combining(decomposed[j]);
        while(c && j>0 && combining(decomposed[j-1])>c) { std::swap(decomposed[j-1],decomposed[j]); --j; }
    }
    std::vector<uint32_t> result; size_t starter=0; unsigned last=0;
    for(uint32_t c:decomposed) {
        unsigned cc=combining(c); uint32_t combined=result.empty()?0:compose(result[starter],c);
        if(combined && (last==0 || last<cc)) result[starter]=combined;
        else { if(cc==0) starter=result.size(); result.push_back(c); last=cc; }
    }
    std::string out; for(uint32_t c:result) out+=utf8(c); return out;
}
std::vector<std::string> preTokens(const std::string &text) {
    auto chars=decode(text);
    size_t count=chars.size(); chars.push_back({0,text.size(),0}); std::vector<std::string> out;
    auto letter=[&](size_t i) { return i<count && (chars[i].cat&1); };
    auto number=[&](size_t i) { return i<count && (chars[i].cat&4); };
    auto newline=[&](size_t i) { return chars[i].c==10 || chars[i].c==13; };
    auto punctuation=[&](size_t i) { return i<count && !space(chars[i].c) && !(chars[i].cat&5); };
    for(size_t i=0;i<count;) {
        size_t end=i;
        if(chars[i].c=='\'') {
            for(const std::string suffix:{"s","t","re","ve","m","ll","d"}) {
                bool match=i+1+suffix.size()<=count;
                for(size_t j=0;match && j<suffix.size();++j) {
                    uint32_t c=chars[i+1+j].c; if(c>='A' && c<='Z') c+=32;
                    match=c==uint32_t(suffix[j]);
                }
                if(match) { end=i+1+suffix.size(); break; }
            }
        }
        if(end==i) {
            size_t j=i;
            if(!newline(i) && !letter(i) && !number(i) && letter(i+1)) ++j;
            if(letter(j)) { while(letter(j)) ++j; end=j; }
        }
        if(end==i && number(i)) end=i+1;
        if(end==i) {
            size_t j=i; if(chars[j].c==32 && punctuation(j+1)) ++j;
            if(punctuation(j)) { while(punctuation(j)) ++j; while(j<count && newline(j)) ++j; end=j; }
        }
        if(end==i && space(chars[i].c)) {
            size_t j=i,lastNewline=i;
            while(j<count && space(chars[j].c)) { if(newline(j)) lastNewline=j+1; ++j; }
            if(lastNewline>i) end=lastNewline;
            else if(j==count) end=j;
            else end=j-i>1?j-1:j;
        }
        if(end==i) end=i+1;
        out.push_back(text.substr(chars[i].offset,chars[end].offset-chars[i].offset)); i=end;
    }
    return out;
}
}

QwenTokenizer::QwenTokenizer(const std::string &directory) {
    std::ifstream f(directory+"/tokenizer.json"); if(!f) throw std::runtime_error("missing Qwen tokenizer.json");
    std::string data((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>()),err;
    auto root=json11::Json::parse(data,err);
    if(!err.empty() || root["model"]["type"].string_value()!="BPE") throw std::runtime_error("unsupported Qwen tokenizer");
    for(auto &kv:root["model"]["vocab"].object_items()) vocab_[kv.first]=kv.second.int_value();
    int rank=0;
    for(auto &m:root["model"]["merges"].array_items()) {
        std::string a,b;
        if(m.is_string()) { auto s=m.string_value(); auto split=s.find(' '); if(split==std::string::npos) throw std::runtime_error("invalid BPE merge"); a=s.substr(0,split); b=s.substr(split+1); }
        else { a=m[0].string_value(); b=m[1].string_value(); }
        ranks_[a+'\0'+b]=rank++;
    }
    for(auto &s:root["added_tokens"].array_items()) special_.emplace_back(s["content"].string_value(),s["id"].int_value());
    std::sort(special_.begin(),special_.end(),[](const auto &a,const auto &b) { return a.first.size()>b.first.size(); });
    int next=256;
    for(int i=0;i<256;++i) bytes_[i]=utf8((i>=33 && i<=126) || (i>=161 && i<=172) || i>=174?i:next++);
}
void QwenTokenizer::plain(const std::string &text,std::vector<int> &out) const {
    for(auto &part:preTokens(nfc(text))) {
        std::vector<std::string> pieces; for(uint8_t c:part) pieces.push_back(bytes_[c]);
        while(pieces.size()>1) {
            int best=std::numeric_limits<int>::max(); size_t index=0;
            for(size_t i=0;i+1<pieces.size();++i) { auto it=ranks_.find(pieces[i]+'\0'+pieces[i+1]); if(it!=ranks_.end() && it->second<best) { best=it->second; index=i; } }
            if(best==std::numeric_limits<int>::max()) break;
            pieces[index]+=pieces[index+1]; pieces.erase(pieces.begin()+index+1);
        }
        for(auto &p:pieces) { auto it=vocab_.find(p); if(it==vocab_.end()) throw std::runtime_error("missing BPE token"); out.push_back(it->second); }
    }
}
std::vector<int> QwenTokenizer::encode(const std::string &text) const {
    if(text.size()>65536) throw std::invalid_argument("text exceeds 64 KiB");
    std::vector<int> out; size_t start=0;
    while(start<text.size()) {
        size_t at=text.size(); const std::pair<std::string,int> *found=nullptr;
        for(auto &token:special_) {
            auto pos=text.find(token.first,start); if(pos<at) { at=pos; found=&token; }
        }
        plain(text.substr(start,at-start),out);
        if(!found) break;
        out.push_back(found->second); start=at+found->first.size();
    }
    return out;
}
std::vector<int> QwenTokenizer::videoPrompt(const std::string &prompt,int &crop) const {
    // The spaces are part of the released Python template, not formatting.
    const std::string system="You are a helpful assistant. Describe the video by detailing the following aspects: "
        "        1. The main content and theme of the video. "
        "        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
        "        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
        "        4. background environment, light, style and atmosphere. "
        "        5. camera angles, movements, and transitions used in the video.";
    std::string prefix="<|im_start|>system\n"+system+"<|im_end|>\n<|im_start|>user\n";
    auto ids=encode(prefix+prompt+"<|im_end|>\n<|im_start|>assistant\n");
    auto marker=encode("<|im_start|>user\n"); crop=0;
    auto at=std::search(ids.begin(),ids.end(),marker.begin(),marker.end());
    if(at==ids.end()) throw std::runtime_error("Qwen user template marker missing");
    crop=int(at-ids.begin()+marker.size());
    if(ids.size()>size_t(crop+1280)) ids.resize(crop+1280);
    return ids;
}

}} // namespace fastllm::nomas
