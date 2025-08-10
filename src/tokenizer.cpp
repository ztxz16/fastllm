//
// Created by tylunasli on 8/8/25.
//

#include "utils.h"

#include "fastllm.h"

#include <cstring>
#include <cmath>
#include <cfloat>
#include <thread>
#include <algorithm>

#ifdef PY_API
#include <pybind11/embed.h>
namespace py = pybind11;
#endif
namespace fastllm {

    Tokenizer::TrieNode::TrieNode() {
        this->tokenId = -999999;
        this->score = 0.0f;
    }

    Tokenizer::Tokenizer() {
        root = new TrieNode();
        int n = 0;
        wchar_t special_token = L'\x0';
        for (; special_token < L'!'; special_token++, n++) {
            byteCharDict[L'\x100' + n] = special_token;
            charByteDict[special_token] = L'\x100' + n;
        }
        for (special_token = L'\x7F'; special_token < L'\xA1'; special_token++, n++) {
            byteCharDict[L'\x100' + n] = special_token;
            charByteDict[special_token] = L'\x100' + n;
        }
        byteCharDict[L'\x100' + n++] = L'\xAD';
        charByteDict[L'\xAD'] = L'\x100' + (n - 1);
    }

    Tokenizer::~Tokenizer() {
        Clear();
        delete root;
    }

    void Tokenizer::Clear() {
        std::vector <TrieNode*> q;
        q.push_back(root);
        for (int i = 0; i < q.size(); i++) {
            TrieNode *now = q[i];
            for (auto it : now->next) {
                q.push_back(it.second);
            }
        }
        if (specialRoot != nullptr) {
            q.push_back(specialRoot);
            for (int i = q.size() - 1; i < q.size(); i++) {
                TrieNode *now = q[i];
                for (auto it : now->next) {
                    q.push_back(it.second);
                }
            }
        }
        for (TrieNode * node : q)
            delete node;
        q.clear();
        root = new TrieNode();
        specialRoot = nullptr;
        tokenToStringDict.clear();
        tokenToScoreDict.clear();
        stringToTokenDict.clear();
    }

    void Tokenizer::Insert(const std::string &s, int tokenId, float score) {
        TrieNode *now = this->root;
        for (int i = 0; i < s.size(); i++) {
            if (now->next.find(s[i]) == now->next.end()) {
                now->next[s[i]] = new TrieNode();
            }
            now = now->next[s[i]];
        }
        now->tokenId = tokenId;
        now->score = score;
        tokenToStringDict[tokenId] = s;
        tokenToScoreDict[tokenId] = score;
        stringToTokenDict[s] = tokenId;
    }

    void Tokenizer::SetSpecialTokens(const std::map<std::string, int>& specialTokenMap) {
        if (specialRoot == nullptr)
            specialRoot = new TrieNode();
        for (auto &it : specialTokenMap) {
            TrieNode *now = this->specialRoot;
            std::string normalized = Normalize(it.first, false);
            for (int i = 0; i < normalized.size(); i++) {
                if (now->next.find(normalized[i]) == now->next.end()) {
                    now->next[normalized[i]] = new TrieNode();
                }
                now = now->next[normalized[i]];
            }
            now->tokenId = it.second;
            now->score = 0.0f;
            tokenToStringDict[it.second] = it.first;
            stringToTokenDict[it.first] = it.second;
            specialTokens.push_back(it.first);
        }
    }

    void Tokenizer::SetTokenizerConfig(const json11::Json &config) {
        this->tokenizerConfig = config;
        if (config["chat_template"].is_string()) {
            this->chatTemplate = config["chat_template"].string_value();
        }
    }

    void Tokenizer::TryMergePairs(std::vector<Symbol> &symbols, int l, int r, std::priority_queue <SymbolPairs> &q) {
        if (l == -1 || r == -1 || symbols[l].len == 0 || symbols[r].len == 0) {
            return;
        }
        auto now = symbols[l].node;
        char *s = symbols[r].s;
        int pos = symbols[r].pos, len = symbols[r].len;
        for (int i = pos; i < pos + len; i++) {
            if (now->next.find(s[i]) != now->next.end()) {
                now = now->next[s[i]];
            } else {
                return;
            }
        }
        if (now->tokenId == -999999) {
            return;
        }
        q.push(SymbolPairs(now->score, l, r, symbols[l].len + symbols[r].len));
    }

    int Tokenizer::GetRank(std::vector <Symbol> &symbols, PartitionLinkNode *cur, int skip) {
        auto nxt = cur->Skip(skip + 2);
        if (nxt == nullptr) {
            return std::numeric_limits<int>::max();
        }
        auto s = symbols[0].s + symbols[0].pos;
        std::string key(s + cur->cur->first, s + nxt->cur->first);
        if (stringToTokenDict.find(key) != stringToTokenDict.end()) {
            return stringToTokenDict[key];
        }
        return std::numeric_limits<int>::max();
    }

    int Tokenizer::GetRank(std::vector<Symbol> &symbols,  std::vector<std::pair<int, int>> &partitions, int idx, int skip) {
        if (idx + skip + 2 >= partitions.size()) {
            return std::numeric_limits<int>::max();
        }
        auto s = symbols[0].s + symbols[0].pos;
        std::string key(s + partitions[idx].first, s + partitions[idx + skip + 2].first);
        if (stringToTokenDict.find(key) != stringToTokenDict.end()) {
            return stringToTokenDict[key];
        }
        return std::numeric_limits<int>::max();
    }

    std::string Tokenizer::Normalize(const std::string &ori, const bool addDummyPrefix) {
        if (this->byteAsChar) {
            std::wstring ws(ori.size(), L' ');
            for (int i=0; i < ori.length(); i++) {
                wchar_t wi = static_cast<wchar_t>(static_cast<unsigned char>(ori[i]));
                if (charByteDict.find(wi) != charByteDict.end()) {
                    wi = charByteDict[wi];
                }
                ws[i] = wi;
            }
            return converter.to_bytes(ws);
        }
        if (blankRepeatCount > 1) {
            std::string blankReplaced(ori);
            for (int c = blankRepeatCount; c > 1; c--) {
                std::string blank("<|blank_" + std::to_string(c) + "|>");
                size_t pos = 0;
                while ((pos = blankReplaced.find(std::string(c, ' '), pos)) != std::string::npos) {
                    blankReplaced.replace(pos, c, blank);
                    pos += blank.length();
                }
            }
        }
        std::string blank = "";
        blank += 226, blank += 150, blank += 129;
        std::string s = (addDummyPrefix && this->addDummyPrefix) ? blank : "";
        if (15 < ori.size() && ori.substr(0, 15) == "<FLM_FIX_TOKEN_") {
            s = "";
        }
        for (int i = 0; i < ori.size(); i++) {
            if (ori[i] == ' ') {
                if (!(this->removeExtraWhitespaces && i > 0 && ori[i - 1] == ' ')) {
                    s += blank;
                }
            } else if (!this->tokenizerConfig["replacements"].is_null()) {
                json11::Json replacement = tokenizerConfig["replacements"][ori.substr(i, 1)];
                if (replacement.is_string())
                    s += replacement.string_value();
            } else {
                s += ori[i];
            }
        }
        return s;
    }

    bool isDigitOrChar(char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    }

    std::vector<float> Tokenizer::UnigramEncode(const std::string &s) {
        // SymbolPairs.l 表示上一个位置
        // SymbolPairs.r 表示选择上一个位置的第几个TrieNode
        std::vector<TrieNode *> specialIds;
        std::vector<std::vector<TrieNode *>> lattice(s.size() + 1, std::vector<TrieNode *>());
        std::vector<std::vector<SymbolPairs>> latticeScores(s.size() + 1, std::vector<SymbolPairs>());
        for (int i = 0; i < s.size(); i++) {
            if (i + 3 < s.size() && s[i] == '<' && s[i + 1] == 'F' && s[i + 2] == 'L' && s[i + 3] == 'M') {
                if (i + 15 < s.size() && s.substr(i, 15) == "<FLM_FIX_TOKEN_") {
                    int start = i;
                    i += 15;
                    TrieNode *fixNode = new TrieNode();
                    fixNode->tokenId = 0;
                    fixNode->score = -0.1;
                    while (s[i] >= '0' && s[i] <= '9') {
                        fixNode->tokenId = fixNode->tokenId * 10 + s[i] - '0';
                        i++;
                    }
                    specialIds.push_back(fixNode);
                    lattice[start].push_back(fixNode);
                    latticeScores[start].push_back(SymbolPairs(0.F, i + 1, 0, i - start));
                    continue;
                }
            }
            if (this->specialRoot != nullptr) {
                TrieNode *now = this->specialRoot;
                int next = i;
                for (; next < s.size(); next++) {
                    if (now->next.find(s[next]) == now->next.end())
                        break;
                    now = now->next[s[next]];
                }
                if (now->tokenId != -999999 && next > i) {
                    lattice[i].push_back(now);
                    latticeScores[i].push_back(SymbolPairs(now->score, next, -1, next - i));
                    i = next - 1;
                    continue;
                }
            }

            TrieNode *now = this->root;
            for (int j = i; j < s.size(); j++) {
                if (now->next.find(s[j]) != now->next.end()) {
                    now = now->next[s[j]];
                    if (now->tokenId != -999999) {
                        lattice[i].push_back(now);
                        latticeScores[i].push_back(SymbolPairs(now->score, j + 1, -1, j - i + 1));
                    }
                } else {
                    break;
                }
            }
            if (latticeScores[i].empty()) {
                // 未识别的字符
                uint8_t c = (uint8_t) (s[i]);
                std::string now = "<0x00>";
                now[3] = (c / 16 > 9 ? ('A' + c / 16 - 10) : ('0' + c / 16));
                now[4] = (c % 16 > 9 ? ('A' + c % 16 - 10) : ('0' + c % 16));
                if (stringToTokenDict.find(now) != stringToTokenDict.end()) {
                    TrieNode *byte = new TrieNode();
                    byte->tokenId = stringToTokenDict[now];
                    byte->score = FLT_MAX - 10.0f;
                    specialIds.push_back(byte);
                    lattice[i].push_back(byte);
                    latticeScores[i].push_back(SymbolPairs(0.F, i + 1, -1, 1));
                }
            }
        }
        TrieNode *empty = new TrieNode();
        specialIds.push_back(empty);
        lattice[s.size()].push_back(empty);
        latticeScores[s.size()].push_back(SymbolPairs(0.F, s.size(), -1, 0));
        // viterbi 求解
        for (int i = 0; i < s.size(); i++) {
            for (int j = 0; j < latticeScores[i].size(); j++) {
                int jNext = i + latticeScores[i][j].size;
                for (int k = 0; k < latticeScores[jNext].size(); k++) {
                    float newScore = latticeScores[i][j].score + lattice[jNext][k]->score;
                    if (latticeScores[jNext][k].r == -1 || latticeScores[jNext][k].score < newScore) {
                        latticeScores[jNext][k].l = i;
                        latticeScores[jNext][k].r = j;
                        latticeScores[jNext][k].score = newScore;
                    }
                }
            }
        }
        std::vector<float> v;
        int pos = s.size();
        int row = latticeScores[s.size()][0].l, column = latticeScores[s.size()][0].r;
        while (column != -1) {
            SymbolPairs& node = latticeScores[row][column];
            v.push_back(lattice[row][column]->tokenId);
            row = node.l;
            column = node.r;
        }
        std::reverse(v.begin(), v.end());
        for (TrieNode * node : specialIds)
            delete node;
        return v;
    }

    std::vector<float> Tokenizer::BytePairEncode(const std::string &s) {
            std::vector<Symbol> symbols;
            for (int i = 0; i < s.size(); i++) {
                if (i + 3 < s.size() && s[i] == '<' && s[i + 1] == 'F' && s[i + 2] == 'L' && s[i + 3] == 'M') {
                    if (i + 15 < s.size() && s.substr(i, 15) == "<FLM_FIX_TOKEN_") {
                        i += 15;
                        int now = 0;
                        while (s[i] >= '0' && s[i] <= '9') {
                            now = now * 10 + s[i] - '0';
                            i++;
                        }
                        symbols.push_back(Symbol(nullptr, (char *) s.data(), i, 0, (int) symbols.size() - 1,
                                                 (int) symbols.size() + 1, now));
                        continue;
                    }
                }

                if (this->specialRoot != nullptr) {
                    TrieNode *now = this->specialRoot;
                    int next = i;
                    for (; next < s.size(); next++) {
                        if (now->next.find(s[next]) == now->next.end())
                            break;
                        now = now->next[s[next]];
                    }
                    if (now->tokenId != -999999 && next > i) {
                        symbols.push_back(Symbol(nullptr, (char *)s.data(), i, 0, (int) symbols.size() - 1,
                                          (int) symbols.size() + 1, now->tokenId));
                        i = next - 1;
                        continue;
                    }
                }

                int tokenId = -999999, pos = i - 1;
                TrieNode *now = this->root;
                for (int j = i; j < s.size(); j++) {
                    if (now->next.find(s[j]) != now->next.end()) {
                        now = now->next[s[j]];
                        if (now->tokenId != -999999) {
                            tokenId = now->tokenId;
                            pos = j;
                            break;
                        }
                    } else {
                        break;
                    }
                }
                if (pos >= i) {
                    symbols.push_back(Symbol(now, (char *) s.data(), i, pos - i + 1, (int) symbols.size() - 1,
                                             (int) symbols.size() + 1, -999999));
                    i = pos;
                } else {
                    symbols.push_back(Symbol(nullptr, (char *) s.data(), i, 0, (int) symbols.size() - 1,
                                             (int) symbols.size() + 1, -999999));
                }
            }
            symbols.back().next = -1;

            std::priority_queue<SymbolPairs> workQueue;
            for (int i = 1; i < symbols.size(); i++) {
                TryMergePairs(symbols, i - 1, i, workQueue);
            }

            while (!workQueue.empty()) {
                auto top = workQueue.top();
                workQueue.pop();
                if (symbols[top.l].len == 0 || symbols[top.r].len == 0 ||
                    symbols[top.l].len + symbols[top.r].len != top.size) {
                    continue;
                }

                for (int i = symbols[top.r].pos; i < symbols[top.r].pos + symbols[top.r].len; i++) {
                    symbols[top.l].node = symbols[top.l].node->next[symbols[top.r].s[i]];
                }
                symbols[top.l].len += symbols[top.r].len;
                symbols[top.r].len = 0;
                symbols[top.l].next = symbols[top.r].next;
                if (symbols[top.r].next >= 0) {
                    symbols[symbols[top.r].next].prev = top.l;
                }

                TryMergePairs(symbols, symbols[top.l].prev, top.l, workQueue);
                TryMergePairs(symbols, top.l, symbols[top.l].next, workQueue);
            }

            std::vector<float> v;
            for (int i = 0; i < symbols.size(); i++) {
                if (symbols[i].len > 0) {
                    v.push_back(symbols[i].node->tokenId);
                } else if (symbols[i].node == nullptr) {
                    if (symbols[i].fixId != -999999) {
                        v.push_back(symbols[i].fixId);
                    } else {
                        // 未识别的字符
                        uint8_t c = (uint8_t) (symbols[i].s[symbols[i].pos]);
                        std::string now = "<0x00>";
                        now[3] = (c / 16 > 9 ? ('A' + c / 16 - 10) : ('0' + c / 16));
                        now[4] = (c % 16 > 9 ? ('A' + c % 16 - 10) : ('0' + c % 16));
                        if (stringToTokenDict.find(now) != stringToTokenDict.end()) {
                            v.push_back(stringToTokenDict[now]);
                        }
                    }
                }
            }
            return v;
    }

    Data Tokenizer::Encode(const std::string &ori) {
        if (this->type == TokenizerType::GLM) {
            const std::map<std::string, int> glmSpecialTokens = {{"[MASK]", 50003}, {"<|startofpiece|>", 50006}, {"<|endofpiece|>", 50007}, {"[sMASK]", 50008}, {"[gMASK]", 50009}};
            if (this->specialTokens.empty())
                SetSpecialTokens(glmSpecialTokens);
        }
        if (this->type == TokenizerType::BPE || this->type == TokenizerType::GLM) {
#ifdef USE_SENTENCEPIECE
            std::string &s = const_cast<std::string &>(ori);
            std::vector<float> v;
            int findPos = 0;
            while (findPos < s.length()) {
                int nextSpecialToken = -1;
                int nextSpecialTokenPos = -1;
                int nextSpecialTokenLen = -1;
                for (auto token : this->specialTokens) {
                    int ind = s.find(token, findPos);
                    if (ind >= 0 && (nextSpecialTokenPos < 0 || ind < nextSpecialTokenPos)) {
                        nextSpecialTokenPos = ind;
                        nextSpecialToken = stringToTokenDict[token];
                        nextSpecialTokenLen = token.length();
                    }
                }
                std::string subStr;
                if (nextSpecialTokenPos < 0) {
                    subStr = s.substr(findPos);
                    findPos = s.length();
                } else {
                    subStr = s.substr(findPos, nextSpecialTokenPos - findPos);
                    findPos = nextSpecialTokenPos + nextSpecialTokenLen;
                }
                if (subStr.length() > 0) {
                    if (spProcessor != nullptr) {
                        std::vector<int> ids;
                        spProcessor->Encode(subStr, &ids);
                        for (int id : ids) {
                            v.push_back(id);
                        }
                    } else {
                        std::string s = Normalize(subStr);
                        std::vector<float> &&subTokenIds = BytePairEncode(s);
                        v.insert(v.end(), subTokenIds.begin(), subTokenIds.end());
                    }
                }
                if (nextSpecialTokenPos >= 0) {
                    v.push_back(nextSpecialToken);
                }
            }
#else
            std::string s = Normalize(ori);
            std::vector<float> &&v = BytePairEncode(s);
#endif
            return Data (DataType::FLOAT32, {1, (int)v.size()}, v);
        } else if (this->type == TokenizerType::QWEN) {
            std::map<std::string, int> specialTokens = {{"<|im_start|>", 151644}, {"<|im_end|>", 151645}, {"<|endoftext|>", 151643}};
            for (int i = 0; i < ori.size(); i++) {
                if (i + 3 < ori.size() && ori[i] == '<' && ori[i + 1] == 'F' && ori[i + 2] == 'L' && ori[i + 3] == 'M') {
                    if (i + 15 < ori.size() && ori.substr(i, 15) == "<FLM_FIX_TOKEN_") {
                        i += 15;
                        int now = 0;
                        while (ori[i] >= '0' && ori[i] <= '9') {
                            now = now * 10 + ori[i] - '0';
                            i++;
                        }
                        specialTokens["<FLM_FIX_TOKEN_" + std::to_string(now) + ">"] = now;
                        continue;
                    }
                }
            }
            
            // comment these special tokens for now
            // for (int i = 0; i < 205; i++) {
            //     specialTokens.insert("<|extra_" + std::to_string(i) + "|>");
            // }

            std::vector<std::pair<int, int>> sep;
            for (auto &token : specialTokens) {
                int pos = 0;
                while ((pos = ori.find(token.first, pos)) != std::string::npos) {
                    sep.push_back({pos, token.first.size()});
                    pos += token.first.size();
                }
            }
            sep.push_back({ori.size(), 1}); // use this to tokenize the last few words
            std::sort(sep.begin(), sep.end(), std::greater<std::pair<int, int>>());

            std::vector<Symbol> symbols;
            std::vector<float> v;

            for (int i = 0; i <= ori.size(); i++) {
                if (i == sep.back().first) {
                    if (!symbols.empty()) {
                        symbols.back().next = -1;
                        std::string cur = ori.substr(i - symbols.size(), symbols.size());
                        std::vector<std::pair<int, int>> partitions(symbols.size() + 1);
                        std::vector <PartitionLinkNode> nodes(symbols.size() + 1);
                        for (int j = 0; j <= (int) symbols.size(); j++) {
                            partitions[j] = std::make_pair(j, std::numeric_limits<int>::max());
                        }
                        for (int j = 0; j <= (int) symbols.size(); j++) {
                            nodes[j].cur = &partitions[j];
                            if (j > 0) {
                                nodes[j].prev = &nodes[j - 1];
                            }
                            if (j + 1 < nodes.size()) {
                                nodes[j].next = &nodes[j + 1];
                            }
                            nodes[j].id = j;
                        }
                        for (int j = 0; j < partitions.size() - 2; j++) {
                            partitions[j].second = GetRank(symbols, partitions, j, 0);
                        }
                        std::set <std::pair <int, int> > pq;
                        for (int j = 0; j < nodes.size(); j++) {
                            pq.insert(std::make_pair(nodes[j].cur->second, j));
                        }
                        int del = 0;
                        while (partitions.size() - del > 1) {
                            int min_rank = pq.begin()->first;
                            auto sel = &nodes[pq.begin()->second];

                            if (min_rank != std::numeric_limits<int>::max()) {
                                pq.erase(std::make_pair(sel->cur->second, sel->id));
                                sel->cur->second = GetRank(symbols, sel, 1);
                                pq.insert(std::make_pair(sel->cur->second, sel->id));
                                if (sel->prev != nullptr) {
                                    pq.erase(std::make_pair(sel->prev->cur->second, sel->prev->id));
                                    sel->prev->cur->second = GetRank(symbols, sel->prev, 1);
                                    pq.insert(std::make_pair(sel->prev->cur->second, sel->prev->id));
                                }
                                pq.erase(std::make_pair(sel->next->cur->second, sel->next->id));
                                sel->next = sel->next->next;
                                sel->next->prev = sel;
                                del++;
                            } else {
                                break;
                            }
                        }
                        auto it = &nodes[0];
                        while (it != nullptr && it->next != nullptr) {
                            std::string key = cur.substr(it->cur->first, it->next->cur->first - it->cur->first);
                            v.push_back((float) stringToTokenDict[key]);
                            it = it->next;
                        }
                        symbols.clear();
                    }

                    std::string special = ori.substr(sep.back().first, sep.back().second);
                    if (specialTokens.find(special) != specialTokens.end()) {
                        v.push_back(specialTokens[special]);
                    }

                    i += sep.back().second - 1;
                    sep.pop_back();

                    continue;
                }

                int tokenId = -999999, pos = i - 1;
                TrieNode *now = this->root;
                for (int j = i; j < ori.size(); j++) {
                    if (now->next.find(ori[j]) != now->next.end()) {
                        now = now->next[ori[j]];
                        if (now->tokenId != -999999) {
                            tokenId = now->tokenId;
                            pos = j;
                            break;
                        }
                    } else {
                        break;
                    }
                }
                if (pos >= i) {
                    symbols.push_back(Symbol(now, (char *) ori.data(), i, pos - i + 1, (int) symbols.size() - 1,
                                             (int) symbols.size() + 1, -999999));
                    i = pos;
                } else {
                    symbols.push_back(Symbol(nullptr, (char *) ori.data(), i, 0, (int) symbols.size() - 1,
                                             (int) symbols.size() + 1, -999999));
                }
            }

            return Data (DataType::FLOAT32, {1, (int)v.size()}, v);
        } else if (this->type == TokenizerType::BERT) {
            std::vector <float> v;
            for (int i = 0; i < ori.size(); i++) {
                int tokenId = -999999, pos = i - 1;
                TrieNode *now = this->root;

                if (i > 0 && isDigitOrChar(ori[i - 1]) && isDigitOrChar(ori[i])) {
                    now = now->next['#']->next['#'];
                }
                for (int j = i; j < ori.size(); j++) {
                    if (now->next.find(ori[j]) != now->next.end()) {
                        now = now->next[ori[j]];
                        if (now->tokenId != -999999) {
                            tokenId = now->tokenId;
                            pos = j;
                        }
                    } else {
                        break;
                    }
                }
                if (pos >= i) {
                    i = pos;
                    v.push_back(tokenId);
                }
            }

            return Data (DataType::FLOAT32, {1, (int)v.size()}, v);
        } else {
            std::vector <float> v;
            for (int i = 0; i < ori.size(); i++) {
                int tokenId = -999999, pos = i - 1;
                TrieNode *now = this->root;
                for (int j = i; j < ori.size(); j++) {
                    if (now->next.find(ori[j]) != now->next.end()) {
                        now = now->next[ori[j]];
                        if (now->tokenId != -999999) {
                            tokenId = now->tokenId;
                            pos = j;
                        }
                    } else {
                        break;
                    }
                }
                if (pos >= i) {
                    i = pos;
                    v.push_back(tokenId);
                }
            }

            return Data (DataType::FLOAT32, {1, (int)v.size()}, v);
        }
    }

    std::string Tokenizer::DecodeTokens(const std::vector<int> &tokens) {
        std::string ret = "";
        for (int i = 0; i < tokens.size(); i++) {
            std::string s = tokenToStringDict[tokens[i]];
            if (s.size() == 6 && s.substr(0, 3) == "<0x" && s.back() == '>') {
                int c = 0;
                for (int i = 3; i < 5; i++) {
                    c *= 16;
                    if (s[i] >= '0' && s[i] <= '9') {
                        c += (s[i] - '0');
                    } else {
                        c += (s[i] - 'A' + 10);
                    }
                }

                s = " ";
                s[0] = c;
            }
            if (s == "<n>") {
                ret += "\n";
            } else if (s == "<|tab|>") {
                ret += "\t";
            } else {
                ret += s;
            }
        }

        std::string blank = "";
        blank += 226, blank += 150, blank += 129;
        while (true) {
            std::string::size_type pos(0);
            if ((pos = ret.find(blank)) != std::string::npos)
                ret.replace(pos, blank.length(), " ");
            else break;
        }
        if (this->byteAsChar) {
            std::wstring wret = converter.from_bytes(ret);
            std::string decoded(wret.size(), ' ');
            for (int i=0; i < wret.length(); i++) {
                if (byteCharDict.find(wret[i]) != byteCharDict.end()) {
                    wret[i] = byteCharDict[wret[i]];
                }
                decoded[i] = static_cast<char>(wret[i]);
            }
            ret = decoded;
        }
        int pos = ret.find("<|blank_");
        if (pos != -1) {
            int space_num = atoi(ret.substr(8, ret.size() - 10).c_str());
            return std::string(space_num, ' ');
        }

        return ret;
    }

    std::string Tokenizer::Decode(const Data &data) {
        std::vector <int> tokens;
        for (int i = 0; i < data.Count(0); i++) {
            tokens.push_back((int) ((float *) data.cpuData)[i]);
        }
#ifdef USE_SENTENCEPIECE
        if (spProcessor != nullptr) {
            std::string result;
            spProcessor->Decode(tokens, &result);
            return result;
        }
#endif
        return DecodeTokens(tokens);
    }

    int Tokenizer::GetTokenId(const std::string &s) {
        AssertInFastLLM(stringToTokenDict.find(s) != stringToTokenDict.end(), 
                        "Tokenizer.GetTokenId error: can't find token \"" + s + "\"");
        return stringToTokenDict[s];
    }

    std::string Tokenizer::GetToken(int id) {
        AssertInFastLLM(tokenToStringDict.find(id) != tokenToStringDict.end(), 
                        "Tokenizer.GetToken error: can't find tokenid \"" + std::to_string(id) + "\"");
        return this->DecodeTokens(std::vector <int> {id}).c_str();
    }

}
