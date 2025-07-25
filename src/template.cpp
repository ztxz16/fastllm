//
// Created by huangyuyang on 5/29/24.
//

#include "template.h"

namespace fastllm {
    bool JinjaVar::BoolValue() const {
        if (this->type == JinjaInt) {
            return (this->intValue != 0);
        } else if (this->stringValue == "false") {
            return false;
        } else if (this->type == JinjaString) {
            return !this->stringValue.empty();
        } else if (this->type == JinjaArray) {
            return !this->arrayValue.empty();
        } else if (this->type == JinjaNone) {
            return false;
        }
        ErrorInFastLLM("Jinja error: " + this->Dump() + " is not bool.");
        return false;
    }

    JinjaVar& JinjaVar::operator[] (const JinjaVar &b) {
        if (this->type == JinjaArray) {
            AssertInFastLLM(b.type == JinjaInt, "Jinja Error: subscript for array should be integer.");
            long long value = b.intValue;
            if (value < 0)
                value = b.intValue + this->arrayValue.size();
            AssertInFastLLM(value < this->arrayValue.size(), "Jinja error: subscript out of range.");
            return this->arrayValue[value];
        } else if (this->type == JinjaDict) {
            return this->dictValue[b.DirectValue()];
        } else {
            ErrorInFastLLM("Jinja Error: unable to use subscript.");
            return this->arrayValue[0];
        }
    }

    std::string JinjaVar::DirectValue() const {
        if (this->type == JinjaInt) {
            return std::to_string(intValue);
        } else if (this->type == JinjaFloat) {
            return std::to_string(floatValue);
        } else {
            return stringValue;
        } 
    }

    std::string JinjaVar::Dump() const {
        std::string ret = "";
        if (this->type == JinjaNone) {
            if (stringValue == "") {
                return "null";
            } else {
                return stringValue;
            }
        } else if (this->type == JinjaInt) {
            return std::to_string(intValue);
        } else if (this->type == JinjaFloat) {
            return std::to_string(floatValue);
        } else if (this->type == JinjaString) {
            return "\"" + stringValue + "\"";
        } else if (this->type == JinjaArray) {
            for (int i = 0; i < arrayValue.size(); i++) {
                ret += std::string((i == 0) ? "[ " : ", ") + arrayValue[i].Dump();
            }
            ret += " ]";
        } else if (this->type == JinjaDict) {
            bool first = true;
            for (auto &it : dictValue) {
                ret += std::string(first ? "{ " : ", ") + "\"" + it.first + "\": " + it.second.Dump();
                first = false;
            }
            ret += " }";
        }
        return ret;
    }

    bool JinjaBlock::IsWhite(char c) {
        return (c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == 0);
    }

    bool JinjaBlock::IsDigit(char c) {
        return (c >= '0' && c <= '9'); 
    }
        
    bool JinjaBlock::IsAlpha(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '$' || c == '_';
    }

    int JinjaBlock::FindNextChar(int pos, int end, int target) {
        if (target == ' ') {
            while (!IsWhite(value[pos]) && pos < end) {
                pos++;
            }
        } else if (target == -1) {
            while ((IsDigit(value[pos]) || IsAlpha(value[pos])) && pos < end) {
                pos++;
            }
        }
        return pos;
    }

    JinjaBlock::JinjaBlock(const std::string &value) {
        this->value = value;
        int len = value.size();

        if (value.size() < 4) {
            return;
        }

        if (value[0] == '{') {
            if (value[1] == '{' || value[1] == '%') {
                AssertInFastLLM(value[len - 1] == '}' && value[len - 2] == (value[1] == '%' ? '%' : '}'), 
                                "Jinja block error: " + value);
                int st = 2, end = len - 2;
                if (value[2] == '-')
                    st = 3;
                if (value[len - 3] == '-')
                    end = len - 3;
                while (st < end) {
                    char now = value[st];
                    if (IsWhite(now)) {
                        st++;
                        continue;
                    } else if (now == '>') {
                        if (st + 1 < end && value[st + 1] == '=') {
                            tokens.push_back(JinjaToken(JinjaToken::JinjaTokenMoreEqual));
                            st += 2;    
                        } else {
                            tokens.push_back(JinjaToken(JinjaToken::JinjaTokenMore));
                            st++;
                        }
                    } else if (now == '<') {
                        if (st + 1 < end && value[st + 1] == '=') {
                            tokens.push_back(JinjaToken(JinjaToken::JinjaTokenLessEqual));
                            st += 2;    
                        } else {
                            tokens.push_back(JinjaToken(JinjaToken::JinjaTokenLess));
                            st++;
                        }
                    } else if (now == '=') {
                        if (st + 1 < end && value[st + 1] == '=') {
                            tokens.push_back(JinjaToken(JinjaToken::JinjaTokenEqual));
                            st += 2;    
                        } else {
                            tokens.push_back(JinjaToken(JinjaToken::JinjaTokenAssign));
                            st++;
                        }
                    } else if (now == '!') {
                        if (st + 1 < end && value[st + 1] == '=') {
                            tokens.push_back(JinjaToken(JinjaToken::JinjaTokenNotEqual));
                            st += 2;    
                        } else {
                            tokens.push_back(JinjaToken(JinjaToken::JinjaTokenNot));
                            st++;
                        }
                    } else if (now == '\'' || now == '\"') {
                        bool ok = false;
                        std::string cur = "";
                        for (int j = st + 1; j < end; j++) {
                            if (value[j] == now) {
                                tokens.push_back(JinjaToken(JinjaToken::JinjaTokenSTRING, cur));
                                st = j + 1;
                                ok = true;
                                break;
                            }
                            if (value[j] == '\\') {
                                AssertInFastLLM(j + 1 < end, "Jinja error: parse string failed: " + value.substr(st, std::min(10, (int)value.size() - st)));
                                cur += escapeChars[value[++j]];
                            } else {
                                cur += value[j];
                            }
                        }
                        AssertInFastLLM(ok, "Jinja error: parse string failed: " + value.substr(st, std::min(10, (int)value.size() - st)));
                    } else if (singleCharTokens.find(now) != singleCharTokens.end()) {
                        if (singleCharTokens[now] == JinjaToken::JinjaTokenLSB && !tokens.empty()) {
                            if (tokens.back().type == JinjaToken::JinjaTokenID)
                                tokens.back().type = JinjaToken::JinjaTokenFUNC;
                        }
                        tokens.push_back(JinjaToken(singleCharTokens[now]));
                        st++;
                        continue;
                    } else if (IsAlpha(now)) {
                        int j = FindNextChar(st + 1, end, -1);
                        std::string id = value.substr(st, j - st);
                        if (keyWords.find(id) != keyWords.end()) {
                            tokens.push_back(JinjaToken(keyWords[id], keyWords[id] == JinjaToken::JinjaTokenBOOL ? id : ""));
                        } else {
                            tokens.push_back(JinjaToken(JinjaToken::JinjaTokenID, id));
                        }
                        st = j;
                    } else if (IsDigit(now)) {
                        int j = FindNextChar(st + 1, end, -1);
                        bool flag = 1;
                        for (int k = st + 1; k < j; k++) {
                            if (!IsDigit(value[k])) {
                                flag = 0;
                            }
                        }

                        if (flag) {
                            tokens.push_back(JinjaToken(JinjaToken::JinjaToken::JinjaTokenNUM, value.substr(st, j - st)));
                        } else {
                            ErrorInFastLLM("Jinja error: unsupport number: " + value.substr(st, j - st));
                        }

                        st = j;
                    } else {
                        ErrorInFastLLM("Jinja parse failed: " + value);
                    }
                }

                AssertInFastLLM(tokens.size() > 0, "Jinja parse failed: " + value);
                if (value[1] == '{') {
                    type = JinjaBlockType::JinjaBlockVar;
                } else {
                    if (tokens[0].type == JinjaToken::JinjaTokenFor) {
                        type = JinjaBlockType::JinjaBlockFor;
                    } else if (tokens[0].type == JinjaToken::JinjaTokenEndFor) {
                        type = JinjaBlockType::JinjaBlockEndFor;
                    } else if (tokens[0].type == JinjaToken::JinjaTokenSet) {
                        type = JinjaBlockType::JinjaBlockSet;
                    } else if (tokens[0].type == JinjaToken::JinjaTokenIf) {
                        type = JinjaBlockType::JinjaBlockIf;
                    } else if (tokens[0].type == JinjaToken::JinjaTokenElseIf) {
                        type = JinjaBlockType::JinjaBlockElseIf;
                    } else if (tokens[0].type == JinjaToken::JinjaTokenElse) {
                        type = JinjaBlockType::JinjaBlockElse;
                    } else if (tokens[0].type == JinjaToken::JinjaTokenEndif) {
                        type = JinjaBlockType::JinjaBlockEndIf;
                    } else {
                        ErrorInFastLLM("Jinja parse failed (Unknown block type): " + value);
                    }
                }
            }
        }        
    }

    JinjaVar JinjaBinaryOp(const JinjaVar &a, const JinjaVar &b, JinjaToken::JinjaToKenType op) {
        if (op == JinjaToken::JinjaTokenAnd) {
            return a.BoolValue() && b.BoolValue();
        } else if (op == JinjaToken::JinjaTokenOr) {
            return a.BoolValue() || b.BoolValue();
        } else if (op == JinjaToken::JinjaTokenAdd) {
            if (a.type == JinjaVar::JinjaString && b.type == JinjaVar::JinjaString) {
                return a.stringValue + b.stringValue;
            } else if (a.type == JinjaVar::JinjaInt && b.type == JinjaVar::JinjaInt) {
                return a.intValue + b.intValue;
            }
        } else if (op == JinjaToken::JinjaTokenSub) {
            if (a.type == JinjaVar::JinjaFloat && b.type == JinjaVar::JinjaFloat) {
                return a.floatValue - b.floatValue;
            } else if (a.type == JinjaVar::JinjaInt && b.type == JinjaVar::JinjaInt) {
                return a.intValue - b.intValue;
            }
        } else if (op == JinjaToken::JinjaTokenMod) {
            if (a.type == JinjaVar::JinjaInt && b.type == JinjaVar::JinjaInt) {
                return a.intValue % b.intValue;
            }
        } else if (op == JinjaToken::JinjaTokenIn) {
            if (b.type == JinjaVar::JinjaDict) {
                return b.dictValue.find(a.stringValue) != b.dictValue.end();
            } else if (a.type == JinjaVar::JinjaString && b.type == JinjaVar::JinjaString) {
                return b.stringValue.find(a.stringValue) != std::string::npos;
            } else if (b.type == JinjaVar::JinjaNone) {
                return a.type == JinjaVar::JinjaNone;
            }
        } else if (op == JinjaToken::JinjaTokenEqual) {
            if (b.type == JinjaVar::JinjaNone) {
                if (b.stringValue == "defined")
                    return (a.type != JinjaVar::JinjaNone);
                else if (b.stringValue == "string")
                    return (a.type == JinjaVar::JinjaString);
            }
            if (a.type != b.type) {
                return false;
            }
            if (a.type == JinjaVar::JinjaNone && b.stringValue == "none") {
                return true;
            }
            if (a.type == JinjaVar::JinjaString) {
                return a.stringValue == b.stringValue;
            }
            if (a.type == JinjaVar::JinjaInt) {
                return a.intValue == b.intValue;
            }
            if (a.type == JinjaVar::JinjaFloat) {
                return a.floatValue == b.floatValue;
            }
            if (a.type == JinjaVar::JinjaNone) {
                return a.stringValue == b.stringValue;
            }
        } else if (op == JinjaToken::JinjaTokenLess) {
            if (a.type == JinjaVar::JinjaInt && b.type == JinjaVar::JinjaInt) {
                return a.intValue < b.intValue;
            } else if (a.type == JinjaVar::JinjaFloat && b.type == JinjaVar::JinjaFloat) {
                return a.floatValue < b.floatValue;
            }
        } else if (op == JinjaToken::JinjaTokenLessEqual) {
            if (a.type == JinjaVar::JinjaInt && b.type == JinjaVar::JinjaInt) {
                return a.intValue <= b.intValue;
            } else if (a.type == JinjaVar::JinjaFloat && b.type == JinjaVar::JinjaFloat) {
                return a.floatValue <= b.floatValue;
            }
        } else if (op == JinjaToken::JinjaTokenMore) {
            return JinjaBinaryOp(b, a, JinjaToken::JinjaTokenLess);
        } else if (op == JinjaToken::JinjaTokenMoreEqual) {
            return JinjaBinaryOp(b, a, JinjaToken::JinjaTokenLessEqual);
        } else if (op == JinjaToken::JinjaTokenNotEqual) {
            JinjaVar temp = JinjaBinaryOp(a, b, JinjaToken::JinjaTokenEqual);
            return (int)(!temp.BoolValue());
        }

        ErrorInFastLLM("Unsupport op: op = " + std::to_string(op) + " a = " + a.Dump() + " b = " + b.Dump());
        return JinjaVar();
    }

    JinjaVar JinjaTrim(const JinjaVar &a) {
        AssertInFastLLM(a.type == JinjaVar::JinjaString, "Jinja error: trim only takes effect on strings");
        std::string s = a.stringValue;
        s.erase(0, s.find_first_not_of(" \n\r\t"));
        s.erase(s.find_last_not_of(" \n\r\t") + 1);
        return s;
    }

    int GetOpLevel(JinjaToken::JinjaToKenType type) {
        if (type == JinjaToken::JinjaTokenNamespace) {
            return -3;
        } else if (type == JinjaToken::JinjaTokenAnd || type == JinjaToken::JinjaTokenOr) {
            return -2;
        } else if (type == JinjaToken::JinjaTokenNot) {
            return -1;
        } else if (type == JinjaToken::JinjaTokenEqual || type == JinjaToken::JinjaTokenNotEqual || type == JinjaToken::JinjaTokenIn
                || type == JinjaToken::JinjaTokenLess || type == JinjaToken::JinjaTokenMore) {
            return 0;
        } else if (type == JinjaToken::JinjaTokenAdd || type == JinjaToken::JinjaTokenSub) {
            return 1;
        } else if (type == JinjaToken::JinjaTokenMul || type == JinjaToken::JinjaTokenDiv || type == JinjaToken::JinjaTokenMod) {
            return 2;
        } else if (type == JinjaToken::JinjaTokenFilter || type == JinjaToken::JinjaTokenFUNC) {
            return 3;
        } else if (type == JinjaToken::JinjaTokenDOT) {
            return 4;
        } else if (type == JinjaToken::JinjaTokenSlice) {
            return 0;
        } else if (type == JinjaToken::JinjaTokenLSB || type == JinjaToken::JinjaTokenLMB) {
            return -5;
        } else {
            ErrorInFastLLM("Jinja error: unsupport op: " + std::to_string(type));
            return -1;
        }
    }

    static std::map<std::string, JinjaFunction> functionMap;

    static std::map<std::string, int> functionArgCount;

    static void initFunctionMap() {
        functionMap["trim"] = JinjaTrim;
        functionMap["split"] = [](const JinjaVar &a) {
            JinjaVar string = a.arrayValue[0];
            JinjaVar delimiter = a.arrayValue[1];
            std::vector<JinjaVar> tokens;
            size_t start = 0;
            size_t end = string.stringValue.find(delimiter.stringValue);
            size_t delimiter_length = delimiter.stringValue.size();
        
            while (end != std::string::npos) {
                tokens.push_back(JinjaVar(string.stringValue.substr(start, end - start)));
                start = end + delimiter_length;
                end = string.stringValue.find(delimiter.stringValue, start);
            }
            tokens.push_back(JinjaVar(string.stringValue.substr(start)));
            return JinjaVar(tokens);
        };
        functionArgCount["trim"] = 1;
        functionArgCount["split"] = 2;
        functionMap["length"] = [](const JinjaVar &element) {
            if (element.type == JinjaVar::JinjaVarType::JinjaString)
                return JinjaVar((long long) element.stringValue.length());
            else if (element.type == JinjaVar::JinjaVarType::JinjaArray)
                return JinjaVar((long long) element.arrayValue.size());
            else if (element.type == JinjaVar::JinjaVarType::JinjaDict)
                return JinjaVar((long long) element.dictValue.size());
            return element;
        };
        functionArgCount["length"] = 2;
        functionMap["startswith"] = [](const JinjaVar &a) {
            std::string string = a.arrayValue[0].stringValue;
            std::string prefix = a.arrayValue[1].stringValue;
            if (prefix.size() > string.size()) return JinjaVar(0);
            JinjaVar temp(string.compare(0, prefix.size(), prefix));
            return JinjaVar((int)(!temp.BoolValue()));
        };
        functionArgCount["startswith"] = 2;
        functionMap["endswith"] = [](const JinjaVar &a) {
            std::string string = a.arrayValue[0].stringValue;
            std::string suffix = a.arrayValue[1].stringValue;
            if (suffix.size() > string.size()) return JinjaVar(0);
            JinjaVar temp(string.compare(string.size() - suffix.size(), suffix.size(), suffix));
            return JinjaVar((int)(!temp.BoolValue()));
        };
        functionArgCount["endswith"] = 2;
        functionMap["lstrip"] = [](const JinjaVar &a) {
            std::string string = a.arrayValue[0].stringValue;
            std::string delimiter = " \t\n\r\f\v";
            if (a.arrayValue.size() > 1)
                delimiter = a.arrayValue[1].stringValue;
            std::string::size_type pos = string.find_first_not_of(delimiter);
            return JinjaVar(string.erase(0, string.find_first_not_of(delimiter)));
        };
        functionArgCount["lstrip"] = 2;
        functionMap["rstrip"] = [](const JinjaVar &a) {
            std::string string = a.arrayValue[0].stringValue;
            std::string delimiter = " \t\n\r\f\v";
            if (a.arrayValue.size() > 1)
                delimiter = a.arrayValue[1].stringValue;
            return JinjaVar(string.erase(string.find_last_not_of(delimiter) + 1));
        };
        functionArgCount["rstrip"] = 2;
        functionMap["strip"] = [](const JinjaVar &a) {
            std::string string = a.arrayValue[0].stringValue;
            std::string delimiter = " \t\n\r\f\v";
            if (a.arrayValue.size() > 1)
                delimiter = a.arrayValue[1].stringValue;
            string.erase(0, string.find_first_not_of(delimiter));
            string.erase(string.find_last_not_of(delimiter) + 1);
            return JinjaVar(string);
        };
        functionArgCount["strip"] = 2;
    }

    JinjaTemplate::JinjaTemplate (const std::string &temp) {
        this->temp = temp;
        // 词法解析
        int pos = 0;
        bool trimNext = false;
        for (int i = 0; i < temp.size(); i++) {
            if (temp[i] == '{' && i + 1 < temp.size() && (temp[i + 1] == '{' || temp[i + 1] == '%') ) {
                size_t curEnd = temp[i + 1] == '%' ? temp.find("%}", i + 2) : temp.find("}}", i + 2);
                AssertInFastLLM(curEnd != -1, 
                                "Can't find blockend: " + temp.substr(i, std::min(10, (int)temp.size() - i)));
                std::string part = temp.substr(pos, i - pos);
                if (temp[i + 2] == '-')
                    part.erase(0, part.find_first_not_of(" \n\r\t"));
                if (trimNext)
                    part.erase(part.find_last_not_of(" \n\r\t") + 1);
                if (!part.empty())
                    blocks.push_back(JinjaBlock(part));
                // 处理切片语法糖
                part = temp.substr(i, curEnd + 2 - i);
                std::string::size_type slicepos = part.find("[::");
                if (slicepos != std::string::npos)
                    part.replace(slicepos, 3, "[0:0:");
                slicepos = part.find("[:");
                if (slicepos != std::string::npos)
                    part.replace(slicepos, 2, "[0:");
                slicepos = part.find("::]");
                if (slicepos != std::string::npos)
                    part.replace(slicepos, 3, ":0:1]");
                slicepos = part.find(":]");
                if (slicepos != std::string::npos)
                    part.replace(slicepos, 2, ":0]");
                slicepos = part.find(":-");
                if (slicepos != std::string::npos)
                    part.replace(slicepos, 2, ":0-");
                slicepos = part.find("is not");
                if (slicepos != std::string::npos)
                    part.replace(slicepos, 6, "!=");
                blocks.push_back(JinjaBlock(part));
                trimNext = (temp[curEnd - 1] == '-');
                pos = curEnd + 2;
                i = curEnd + 1;
            }
        }
        blocks.push_back(temp.substr(pos));
        if (functionMap.empty())
            initFunctionMap();
    }

    JinjaVar JinjaTemplate::ComputeExpression(JinjaVar &local, std::vector <JinjaToken> tokens, int st, int end, JinjaVar *setValue) {
        std::vector <JinjaToken> suffixExp; // 后缀表达式
        std::vector <JinjaToken> ops; // 符号栈

        // 1 中缀表达式转后缀表达式
        for (int i = st; i < end; i++) {
            if (tokens[i].type == JinjaToken::JinjaTokenID || 
                tokens[i].type == JinjaToken::JinjaTokenBOOL ||
                tokens[i].type == JinjaToken::JinjaTokenNUM ||
                tokens[i].type == JinjaToken::JinjaTokenSTRING) {
                suffixExp.push_back(tokens[i]);
            } else if (tokens[i].type == JinjaToken::JinjaTokenLSB || tokens[i].type == JinjaToken::JinjaTokenLMB) {
                ops.push_back(tokens[i]);
            } else if (tokens[i].type == JinjaToken::JinjaTokenRSB) {
                while (ops.size() > 0 && ops.back().type != JinjaToken::JinjaTokenLSB) {
                    suffixExp.push_back(ops.back());
                    ops.pop_back();
                }
                AssertInFastLLM(ops.size() > 0 && ops.back().type == JinjaToken::JinjaTokenLSB, "Error: barckets doesn't match.");
                ops.pop_back();
                if (!ops.empty() && ops.back().type == JinjaToken::JinjaTokenFUNC) {
                    suffixExp.push_back(ops.back());
                    ops.pop_back();
                }
            } else if (tokens[i].type == JinjaToken::JinjaTokenNamespace) {
                // 目前仅支持 "变量 = 表达式" 格式
                int index = tokens[i + 1].type == JinjaToken::JinjaTokenLSB ? 1 : 0;
                AssertInFastLLM(
                    tokens.size() - i >= 3 &&
                    tokens[i + index + 1].type == JinjaToken::JinjaTokenID &&
                    tokens[i + index + 2].type == JinjaToken::JinjaTokenAssign,
                    "Jinja error: only support format \"(var = expression)\"."
                );
                ops.push_back(tokens[i]);
            } else if (tokens[i].type == JinjaToken::JinjaTokenRMB) {
                while (ops.size() > 0 && ops.back().type != JinjaToken::JinjaTokenLMB) {
                    suffixExp.push_back(ops.back());
                    ops.pop_back();
                }
                AssertInFastLLM(ops.size() > 0 && ops.back().type == JinjaToken::JinjaTokenLMB, "Error: barckets doesn't match.");
                if (suffixExp.back().type != JinjaToken::JinjaTokenSlice)
                    suffixExp.push_back(tokens[i]);
                ops.pop_back();
            } else if (tokens[i].type == JinjaToken::JinjaTokenSlice) {
                if (!ops.empty() && ops.back().type == JinjaToken::JinjaTokenSlice)
                    ops.pop_back();
                ops.push_back(tokens[i]);
            } else if (tokens[i].type == JinjaToken::JinjaTokenFUNC) {
                if (!ops.empty() && ops.back().type == JinjaToken::JinjaTokenDOT)
                    ops.pop_back();
                while (ops.size() > 0 && GetOpLevel(ops.back().type) > GetOpLevel(tokens[i].type)) {
                    suffixExp.push_back(ops.back());
                    ops.pop_back();
                }
                ops.push_back(tokens[i]);
            } else if (tokens[i].type == JinjaToken::JinjaTokenDOT ||
                        tokens[i].type == JinjaToken::JinjaTokenAdd ||
                        tokens[i].type == JinjaToken::JinjaTokenSub ||
                        tokens[i].type == JinjaToken::JinjaTokenMul ||
                        tokens[i].type == JinjaToken::JinjaTokenDiv ||
                        tokens[i].type == JinjaToken::JinjaTokenMod ||
                        tokens[i].type == JinjaToken::JinjaTokenEqual ||
                        tokens[i].type == JinjaToken::JinjaTokenNotEqual ||
                        tokens[i].type == JinjaToken::JinjaTokenLess ||
                        tokens[i].type == JinjaToken::JinjaTokenMore ||
                        tokens[i].type == JinjaToken::JinjaTokenSlice ||
                        tokens[i].type == JinjaToken::JinjaTokenIn ||
                        tokens[i].type == JinjaToken::JinjaTokenAnd ||
                        tokens[i].type == JinjaToken::JinjaTokenOr ||
                        tokens[i].type == JinjaToken::JinjaTokenNot ||
                        tokens[i].type == JinjaToken::JinjaTokenFilter) {
                while (ops.size() > 0 && GetOpLevel(ops.back().type) > GetOpLevel(tokens[i].type)) {
                    suffixExp.push_back(ops.back());
                    ops.pop_back();
                }
                ops.push_back(tokens[i]);
            }
        }
        while (ops.size() > 0) {
            suffixExp.push_back(ops.back());
            ops.pop_back();
        }

        // 2. 后缀表达式求值
        std::vector <JinjaVar> vars;
        for (auto &it : suffixExp) {
            if (it.type == JinjaToken::JinjaTokenID) {
                vars.push_back(JinjaVar(JinjaVar::JinjaNone, it.value));
            } else if (it.type == JinjaToken::JinjaTokenBOOL) {
                vars.push_back(JinjaVar((int) (it.value == "false" ? 0 : 1)));
            } else if (it.type == JinjaToken::JinjaTokenSTRING) {
                vars.push_back(JinjaVar(it.value));
            } else if (it.type == JinjaToken::JinjaTokenNUM) {
                vars.push_back(JinjaVar(atoll(it.value.c_str())));
            } else if (it.type == JinjaToken::JinjaTokenDOT) {
                AssertInFastLLM(vars.size() > 1, "Jinja Error: expression error.");
                JinjaVar a = vars[vars.size() - 2], b = vars.back();
                if (a.type == JinjaVar::JinjaNone) {
                    if (setValue != nullptr)
                        local[a][b] = *setValue;
                    a = local[a];
                } else if (setValue != nullptr) {
                    a[b] = *setValue;
                }
                vars.pop_back();
                vars.pop_back();
                vars.push_back(a[b]);
            } else if (it.type == JinjaToken::JinjaTokenRMB) {
                AssertInFastLLM(vars.size() > 1, "Jinja Error: expression error.");
                JinjaVar a = vars[vars.size() - 2], b = vars.back();
                if (b.type == JinjaVar::JinjaNone) {
                    b = local[b];
                }
                if (a.type == JinjaVar::JinjaNone) {
                    if (setValue != nullptr)
                        local[a][b] = *setValue;
                    a = local[a];
                } else if (setValue != nullptr) {
                    a[b] = *setValue;
                }
                vars.pop_back();
                vars.pop_back();
                vars.push_back(a[b]);
            } else if (it.type == JinjaToken::JinjaTokenNamespace) {
                AssertInFastLLM(vars.size() > 1, "Jinja Error: expression error.");
                JinjaVar last = vars.back();
                int shift = (last.type == JinjaVar::JinjaDict) ? 1 : 0 ;
                JinjaVar a = vars[vars.size() - 2 - shift], b = vars[vars.size() - 1 - shift];
                if (b.type == JinjaVar::JinjaNone) {
                    b = local[b];
                }
                vars.pop_back();
                vars.pop_back();
                if (last.type == JinjaVar::JinjaDict) {
                    vars.pop_back();
                    last.dictValue[a.stringValue] = b;
                    vars.push_back(last);
                } else {
                    vars.push_back(JinjaVar({{a.stringValue, b}}));
                }
            } else if (it.type == JinjaToken::JinjaTokenFUNC) {
                if (functionMap.find(it.value) != functionMap.end()) {
                    int argCount = functionArgCount[it.value];
                    AssertInFastLLM(vars.size() >= argCount, "Jinja Error: function expression error.");
                    JinjaVar args;
                    for (int k=0; k<argCount; k++) {
                        JinjaVar a = vars.back();
                        if (a.type == JinjaVar::JinjaNone) {
                            a = local[a];
                        }
                        args.arrayValue.insert(args.arrayValue.begin(), a);
                        vars.pop_back();
                    }
                    vars.push_back(functionMap[it.value](args));
                } else {
                    ErrorInFastLLM("Jinja Error: unsupport function " + it.value);
                }
            } else if (it.type == JinjaToken::JinjaTokenFilter) {
                AssertInFastLLM(vars.size() > 1, "Jinja Error: expression error.");
                JinjaVar a = vars[vars.size() - 2], b = vars.back();
                if (a.type == JinjaVar::JinjaNone) {
                    a = local[a];
                }
                vars.pop_back();
                vars.pop_back();
                if (functionMap.find(b.stringValue) != functionMap.end()) {
                    vars.push_back(functionMap[b.stringValue](a));
                } else {
                    ErrorInFastLLM("Jinja Error: unsupport filter " + b.stringValue);
                }
            } else if (it.type == JinjaToken::JinjaTokenNot) {
                AssertInFastLLM(vars.size() >= 1, "Jinja Error: expression error.");
                JinjaVar a = vars.back();
                vars.pop_back();
                if (a.type == JinjaVar::JinjaNone && a.stringValue == "defined") {
                    vars.push_back(JinjaVar(JinjaVar::JinjaNone, "none"));
                } else if (a.type == JinjaVar::JinjaNone && a.stringValue == "none") {
                    vars.push_back(JinjaVar(JinjaVar::JinjaNone, "defined"));
                } else {
                    if (a.type == JinjaVar::JinjaNone) {
                        a = local[a];
                    }
                    vars.push_back(a.type == JinjaVar::JinjaNone ? JinjaVar(1) : JinjaVar(!a.BoolValue()));
                }
            } else if (it.type == JinjaToken::JinjaTokenSub) {
                AssertInFastLLM(vars.size() > 0, "Jinja Error: expression '-' error.");
                JinjaVar a = vars.back();
                if (a.type == JinjaVar::JinjaNone)
                    a = local[a];
                AssertInFastLLM(a.type == JinjaVar::JinjaInt || a.type == JinjaVar::JinjaFloat, "Jinja Error: expression '-' error.");
                if (vars.size() > 1) {
                    JinjaVar b = vars[vars.size() - 2];
                    if (b.type == JinjaVar::JinjaNone)
                        b = local[b];
                    if (b.type == JinjaVar::JinjaInt || b.type == JinjaVar::JinjaFloat) {
                        vars.pop_back();
                        vars.pop_back();
                        vars.push_back(JinjaBinaryOp(b, a, it.type));
                        continue;
                    }
                }
                vars.pop_back();
                if (a.type == JinjaVar::JinjaInt)
                    vars.push_back(JinjaVar(-a.intValue));
                else
                    vars.push_back(JinjaVar(-a.floatValue));
            } else if (it.type == JinjaToken::JinjaTokenAdd ||
                        it.type == JinjaToken::JinjaTokenMul ||
                        it.type == JinjaToken::JinjaTokenDiv ||
                        it.type == JinjaToken::JinjaTokenMod ||
                        it.type == JinjaToken::JinjaTokenAssign ||
                        it.type == JinjaToken::JinjaTokenEqual ||
                        it.type == JinjaToken::JinjaTokenNotEqual ||
                        it.type == JinjaToken::JinjaTokenLess ||
                        it.type == JinjaToken::JinjaTokenMore ||
                        it.type == JinjaToken::JinjaTokenIn ||
                        it.type == JinjaToken::JinjaTokenAnd ||
                        it.type == JinjaToken::JinjaTokenOr) {
                AssertInFastLLM(vars.size() > 1, "Jinja Error: expression error.");
                JinjaVar a = vars[vars.size() - 2], b = vars.back();
                if (a.type == JinjaVar::JinjaNone && it.type != JinjaToken::JinjaTokenIn) {
                    a = local[a];
                }
                if (b.type == JinjaVar::JinjaNone && b.stringValue != "defined" && b.stringValue != "none" && b.stringValue != "string") {
                    b = local[b];
                }
                vars.pop_back();
                vars.pop_back();
                vars.push_back(JinjaBinaryOp(a, b, it.type));
            } else if (it.type == JinjaToken::JinjaTokenSlice) {
                AssertInFastLLM(vars.size() >= 3, "Jinja Error: slice expression error.");
                JinjaVar a = vars[vars.size() - 3], b = vars[vars.size() - 2], e = vars.back(), s = JinjaVar(1);
                if (a.type == JinjaVar::JinjaInt) {
                    a = vars[vars.size() - 4], b = vars[vars.size() - 3], e = vars[vars.size() - 2], s = vars.back();
                    vars.pop_back();
                }
                if (a.type == JinjaVar::JinjaNone)
                    a = local[a];
                AssertInFastLLM(a.type == JinjaVar::JinjaArray && b.type == JinjaVar::JinjaInt && e.type == JinjaVar::JinjaInt
                    && s.type == JinjaVar::JinjaInt,"Jinja Error: slice expression error.");
                vars.pop_back();
                vars.pop_back();
                vars.pop_back();
                if (e.intValue <= 0)
                    e.intValue += a.arrayValue.size();
                std::vector<JinjaVar> subArray;
                if (s.intValue == 1) {
                    subArray = std::vector<JinjaVar>(a.arrayValue.begin() + b.intValue, a.arrayValue.begin() + e.intValue);
                } else {
                    if (s.intValue < 0) {
                        for (size_t i = e.intValue; i > b.intValue; i += s.intValue)
                            subArray.push_back(a.arrayValue[i-1]);
                    } else {
                        for (size_t i = b.intValue; i < e.intValue; i += s.intValue)
                            subArray.push_back(a.arrayValue[i]);
                    }
                }
                vars.push_back(JinjaVar(subArray));
            }
        }

        AssertInFastLLM(vars.size() == 1, "Jinja Error: expression error.");
        if (vars[0].type == JinjaVar::JinjaNone) {
            vars[0] = local[vars[0]];
        }
        return vars[0];
    }

    void JinjaTemplate::Parse(int st, int end, JinjaVar &var, std::string &ret) {
        for (int i = st; i < end; i++) {
            JinjaBlock &curBlock = blocks[i];
            if (curBlock.type == JinjaBlock::JinjaBlockType::JinjaBlockOriginal) {
                ret += curBlock.value;
            } else if (curBlock.type == JinjaBlock::JinjaBlockType::JinjaBlockVar) {
                ret += ComputeExpression(var, curBlock.tokens, 0, curBlock.tokens.size()).DirectValue();
            } else if (curBlock.type == JinjaBlock::JinjaBlockFor) {
                int cnt = 0;
                int endPos = -1;
                for (int j = i + 1; j < end; j++) {
                    if (blocks[j].type == JinjaBlock::JinjaBlockType::JinjaBlockFor) {
                        cnt++;
                    } else if (blocks[j].type == JinjaBlock::JinjaBlockType::JinjaBlockEndFor) {
                        if ((cnt--) == 0) {
                            endPos = j;
                            break;
                        }
                    }
                }
                AssertInFastLLM(endPos != -1, "Jinja error: no endfor block for " + curBlock.value);
                // 目前仅支持 "for 变量 in 表达式" 格式
                AssertInFastLLM(
                    curBlock.tokens.size() >= 4 &&
                    curBlock.tokens[1].type == JinjaToken::JinjaTokenID &&
                    curBlock.tokens[2].type == JinjaToken::JinjaTokenIn,
                    "Jinja error: only support format \"for var in expression\"."
                );

                std::string iterId = curBlock.tokens[1].value;
                JinjaVar exp = ComputeExpression(var, curBlock.tokens, 3, curBlock.tokens.size());
                JinjaVar original = var[iterId];
                var["loop"] = {{"index", 1}, {"index0", 0}, {"first", 1}, {"last", 0}};
                if (exp.type == JinjaVar::JinjaArray) {
                    for (auto &it : exp.arrayValue) {
                        var[iterId] = it;
                        Parse(i + 1, endPos, var, ret);
                        var["loop"]["index"].intValue++;
                        var["loop"]["index0"].intValue++;
                        var["loop"]["first"].intValue = 0;
                        var["loop"]["last"].intValue = (var["loop"]["index"].intValue == exp.arrayValue.size());
                    }
                } else if (exp.type == JinjaVar::JinjaDict) {
                    for (auto &it : exp.dictValue) {
                        var[iterId] = it.second;
                        Parse(i + 1, endPos, var, ret);
                        var["loop"]["index"].intValue++;
                        var["loop"]["index0"].intValue++;
                        var["loop"]["first"].intValue = 0;
                        var["loop"]["last"].intValue = (var["loop"]["index"].intValue == exp.arrayValue.size());
                    }
                } else {
                    ErrorInFastLLM(exp.Dump() + " is not iterable");
                }
                var[iterId] = original;
                i = endPos;
            } else if (curBlock.type == JinjaBlock::JinjaBlockIf || curBlock.type == JinjaBlock::JinjaBlockType::JinjaBlockElseIf) {
                int cnt = 0;
                int elsePos = -1;
                int endPos = -1;
                for (int j = i + 1; j <= end; j++) {
                    if (blocks[j].type == JinjaBlock::JinjaBlockType::JinjaBlockIf) {
                        cnt++;
                    } else if (blocks[j].type == JinjaBlock::JinjaBlockType::JinjaBlockElse) {
                        if (cnt == 0 && elsePos == -1) {
                            elsePos = j;
                        }
                    } else if (blocks[j].type == JinjaBlock::JinjaBlockType::JinjaBlockElseIf) {
                        if (cnt == 0 && elsePos == -1) {
                            elsePos = j;
                        }
                    } else if (blocks[j].type == JinjaBlock::JinjaBlockType::JinjaBlockEndIf) {
                        if ((cnt--) == 0) {
                            endPos = j;
                            break;
                        }
                    }
                }
                AssertInFastLLM(endPos != -1, "Jinja error: no endif block for " + curBlock.value);
                // 目前仅支持 "if 表达式" 格式
                AssertInFastLLM(
                    curBlock.tokens.size() >= 2,
                    "Jinja error: only support format \"if expression\"."
                );

                JinjaVar exp = ComputeExpression(var, curBlock.tokens, 1, curBlock.tokens.size());
                if (exp.BoolValue()) {
                    if (elsePos != -1) {
                        Parse(i + 1, elsePos, var, ret);
                    } else {
                        Parse(i + 1, endPos, var, ret);
                    }
                } else {
                    if (elsePos != -1) {
                        int nextPos = (blocks[elsePos].type == JinjaBlock::JinjaBlockType::JinjaBlockElse) ? elsePos + 1 : elsePos;
                        Parse(nextPos, endPos, var, ret);
                    }
                }
                if (blocks[endPos].type == JinjaBlock::JinjaBlockType::JinjaBlockElseIf)
                    i = endPos - 1;
                else
                    i = endPos;
            } else if (curBlock.type == JinjaBlock::JinjaBlockSet) {
                // 目前仅支持 "set 变量 = 表达式" 格式
                if (curBlock.tokens.size() >= 4 &&
                        curBlock.tokens[1].type == JinjaToken::JinjaTokenID &&
                        curBlock.tokens[2].type == JinjaToken::JinjaTokenAssign) {
                    std::string iterId = curBlock.tokens[1].value;
                    var[iterId] = ComputeExpression(var, curBlock.tokens, 3, curBlock.tokens.size());
                } else {
                    int assignPos = 0;
                    for (; curBlock.tokens[assignPos].type != JinjaToken::JinjaTokenAssign && assignPos < curBlock.tokens.size(); assignPos++);
                    AssertInFastLLM(assignPos > 0 && assignPos < curBlock.tokens.size() - 1,
                        "Jinja error: only support format \"set var = expression\".");
                    JinjaVar value = ComputeExpression(var, curBlock.tokens, assignPos + 1, curBlock.tokens.size());
                    ComputeExpression(var, curBlock.tokens, 1, assignPos, &value);
                }
            } else {
                ErrorInFastLLM("Jinja usupport block: " + curBlock.value);
            }
        }
    }

    std::string JinjaTemplate::Apply(const JinjaVar &var) {
        std::string ret = "";
        JinjaVar localVar = var;
        Parse(0, blocks.size(), localVar, ret);
        return ret;
    }
}
