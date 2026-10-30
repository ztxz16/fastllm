#include "models/lingbot_world/runtime.h"
#include "json11.hpp"
#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace fastllm { namespace lingbot {
namespace {
const std::unordered_map<std::string,std::string> entities={
#include "html_entities.inc"
};
std::string utf8(uint32_t c) {
    if(c<128) return std::string(1,char(c));
    if(c<2048) return {char(0xc0|(c>>6)),char(0x80|(c&63))};
    if(c<65536) return {char(0xe0|(c>>12)),char(0x80|((c>>6)&63)),char(0x80|(c&63))};
    return {char(0xf0|(c>>18)),char(0x80|((c>>12)&63)),char(0x80|((c>>6)&63)),char(0x80|(c&63))};
}
std::string unescape(const std::string &text) {
    std::string out;
    for(size_t i=0;i<text.size();) {
        if(text[i]!='&') { out+=text[i++]; continue; }
        bool found=false;
        if(i+2<text.size() && text[i+1]=='#') {
            size_t j=i+2; int base=10;
            if(text[j]=='x' || text[j]=='X') { base=16; ++j; }
            size_t start=j; uint32_t value=0;
            while(j<text.size()) {
                unsigned char c=text[j]; int digit=c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:c>='A'&&c<='F'?c-'A'+10:99;
                if(digit>=base) break;
                value=std::min<uint32_t>(0x110000,value*base+digit); ++j;
            }
            if(j>start) {
                if(j<text.size() && text[j]==';') ++j;
                if(!value || value>0x10ffff || (value>=0xd800 && value<=0xdfff)) value=0xfffd;
                out+=utf8(value); i=j; found=true;
            }
        } else {
            for(size_t length=std::min<size_t>(32,text.size()-i-1);length;--length) {
                auto it=entities.find(text.substr(i+1,length));
                if(it!=entities.end()) { out+=it->second; i+=length+1; found=true; break; }
            }
        }
        if(!found) out+=text[i++];
    }
    return out;
}
bool whitespace(uint32_t c) {
    return (c>=9&&c<=13)||(c>=0x1c&&c<=0x20)||c==0x85||c==0xa0||c==0x1680||
        (c>=0x2000&&c<=0x200a)||c==0x2028||c==0x2029||c==0x202f||c==0x205f||c==0x3000;
}
std::string clean(const std::string &source) {
    // Reuse the existing validated UTF-8/NFC implementation. Typical prompts
    // also receive the official whitespace, quote and HTML entity cleanup.
    auto text=nomas::normalizeNFC(unescape(unescape(source)));
    std::string out; bool pending=false;
    for(size_t i=0;i<text.size();) {
        uint32_t c=uint8_t(text[i++]); int extra=c<128?0:c<224?1:c<240?2:3;
        if(extra) c&=(1u<<(6-extra))-1;
        while(extra--) c=(c<<6)|(uint8_t(text[i++])&63);
        if(whitespace(c)) { if(!out.empty()) pending=true; continue; }
        if(c<32 || (c>=0x7f && c<0xa0) || c==0xfeff) continue;
        if(pending) { out+=' '; pending=false; }
        if(c>=0xff01 && c<=0xff5e) c-=0xfee0;
        if(c==0x2018 || c==0x2019 || c==0x201b) c='\'';
        if(c==0x201c || c==0x201d || c==0x201f) c='"';
        if(c>=0xfb00 && c<=0xfb06) {
            const char *ligatures[]={"ff","fi","fl","ffi","ffl","st","st"}; out+=ligatures[c-0xfb00];
        } else out+=utf8(c);
    }
    return out;
}
}
Tokenizer::Tokenizer(const std::string &directory) {
    std::ifstream file(directory+"/tokenizer.json");
    if(!file) throw std::runtime_error("missing UMT5 tokenizer.json");
    std::string data((std::istreambuf_iterator<char>(file)),{}),error;
    auto root=json11::Json::parse(data,error);
    if(!error.empty() || root["model"]["type"].string_value()!="Unigram" || root["model"]["byte_fallback"].bool_value())
        throw std::runtime_error("unsupported UMT5 tokenizer format");
    unknown_=root["model"]["unk_id"].int_value();
    float minimum=0; int index=0;
    for(const auto &entry:root["model"]["vocab"].array_items()) {
        auto piece=entry[0].string_value(); float score=entry[1].number_value();
        vocab_.emplace(piece,std::make_pair(index++,score));
        minimum=std::min(minimum,score); maxPiece_=std::max(maxPiece_,piece.size());
    }
    if(vocab_.empty()) throw std::runtime_error("empty UMT5 vocabulary");
    unknownScore_=minimum-10.f;
    for(const auto &entry:root["added_tokens"].array_items()) {
        auto piece=entry["content"].string_value();
        if(!piece.empty()) special_.emplace_back(piece,entry["id"].int_value());
    }
    std::sort(special_.begin(),special_.end(),[](const auto &a,const auto &b) { return a.first.size()>b.first.size(); });
}
std::vector<int> Tokenizer::word(const std::string &text) const {
    std::vector<float> scores(text.size()+1,-std::numeric_limits<float>::infinity());
    std::vector<int> from(text.size()+1,-1),ids(text.size()+1,unknown_);
    scores[0]=0;
    for(size_t i=0;i<text.size();) {
        unsigned char c=text[i]; size_t length=c<128?1:c<224?2:c<240?3:4;
        bool single=false;
        for(size_t j=i+length;j<=text.size() && j-i<=maxPiece_;++j) {
            if(j<text.size() && (uint8_t(text[j])&0xc0)==0x80) continue;
            auto it=vocab_.find(text.substr(i,j-i)); if(it==vocab_.end()) continue;
            single|=j==i+length;
            float value=scores[i]+it->second.second;
            if(value>scores[j]) { scores[j]=value; from[j]=i; ids[j]=it->second.first; }
        }
        if(!single && scores[i]+unknownScore_>scores[i+length]) {
            scores[i+length]=scores[i]+unknownScore_; from[i+length]=i; ids[i+length]=unknown_;
        }
        i+=length;
    }
    std::vector<int> out;
    for(int at=int(text.size());at>0;at=from[at]) {
        if(from[at]<0) throw std::runtime_error("unreachable Unigram token boundary");
        if(ids[at]!=unknown_ || out.empty() || out.back()!=unknown_) out.push_back(ids[at]);
    }
    std::reverse(out.begin(),out.end()); return out;
}
std::vector<int> Tokenizer::encode(const std::string &source) const {
    if(source.size()>65536) throw std::invalid_argument("LingBot prompt exceeds 65536 bytes");
    auto text=clean(source); std::vector<int> out;
    auto plain=[&](const std::string &part) {
        size_t start=0;
        while(start<part.size()) {
            while(start<part.size() && part[start]==' ') ++start;
            if(start==part.size()) break;
            size_t end=part.find(' ',start); if(end==std::string::npos) end=part.size();
            auto ids=word("▁"+part.substr(start,end-start)); out.insert(out.end(),ids.begin(),ids.end()); start=end;
        }
    };
    size_t start=0;
    for(size_t i=0;i<text.size();) {
        const std::pair<std::string,int> *match=nullptr;
        for(const auto &item:special_) if(text.compare(i,item.first.size(),item.first)==0) { match=&item; break; }
        if(match) { plain(text.substr(start,i-start)); out.push_back(match->second); i+=match->first.size(); start=i; }
        else ++i;
    }
    plain(text.substr(start));
    if(out.size()>511) out.resize(511);
    out.push_back(1); return out;
}
}} // namespace fastllm::lingbot
