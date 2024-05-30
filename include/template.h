//
// Created by huangyuyang on 5/29/24.
//

#ifndef FASTLLM_TEMPLATE_H
#define FASTLLM_TEMPLATE_H

#include "utils.h"

namespace fastllm {
    struct JinjaVar {
        enum JinjaVarType {
            JinjaNone = 0, JinjaInt = 1, JinjaFloat = 2, JinjaString = 3,
            JinjaArray = 100, JinjaDict = 101
        };

        JinjaVarType type = JinjaNone;
        long long intValue;
        float floatValue;
        std::string stringValue;
        std::vector <JinjaVar> arrayValue;
        std::map <std::string, JinjaVar> dictValue;

        JinjaVar () {}
        JinjaVar (int intValue) : type(JinjaInt), intValue(intValue) {}
        JinjaVar (long long intValue) : type(JinjaInt), intValue(intValue) {}
        JinjaVar (float floatValue) : type(JinjaFloat), floatValue(floatValue) {}
        JinjaVar (const char *stringValue) : type(JinjaString), stringValue(stringValue) {}
        JinjaVar (const std::string &stringValue) : type(JinjaString), stringValue(stringValue) {}
        JinjaVar (const std::vector <JinjaVar> &vars) : type(JinjaArray), arrayValue(vars) {}
        JinjaVar (std::initializer_list <std::pair <std::string, JinjaVar> > dict) : type(JinjaDict) {
            for (auto &it : dict) {
                this->dictValue[it.first] = it.second;
            }
        }

        JinjaVar (JinjaVarType type, const std::string &value) : type(type), stringValue(value) {}

        bool BoolValue() const;

        JinjaVar &operator[] (const JinjaVar &b);

        std::string DirectValue() const;

        std::string Dump() const;
    };
    using JinjaArray = std::vector <JinjaVar>;

    // 词法分析后的Token
    struct JinjaToken {
        enum JinjaToKenType {
            JinjaTokenID = 0, JinjaTokenNUM, JinjaTokenSTRING, JinjaTokenDOT, 
            JinjaTokenLMB, JinjaTokenRMB, JinjaTokenLSB, JinjaTokenRSB,
            JinjaTokenSet, JinjaTokenFor, JinjaTokenEndFor, JinjaTokenIf, JinjaTokenElse, JinjaTokenEndif,
            JinjaTokenIn,
            JinjaTokenAssign, JinjaTokenNotEqual, JinjaTokenEqual, JinjaTokenAdd, JinjaTokenSub, JinjaTokenMul, JinjaTokenDiv,
            JinjaTokenNot, JinjaTokenAnd, JinjaTokenOr,
            JinjaTokenFliter
        };

        JinjaToKenType type;
        std::string value;

        JinjaToken (JinjaToKenType type, const std::string &value = "") : type(type), value(value) {}
    };

    static std::map <char, JinjaToken::JinjaToKenType> singleCharTokens = {
            {'(', JinjaToken::JinjaToKenType::JinjaTokenLSB},
            {')', JinjaToken::JinjaToKenType::JinjaTokenRSB},
            {'[', JinjaToken::JinjaToKenType::JinjaTokenLMB},
            {']', JinjaToken::JinjaToKenType::JinjaTokenRMB},
            {'.', JinjaToken::JinjaToKenType::JinjaTokenDOT},
            {'+', JinjaToken::JinjaToKenType::JinjaTokenAdd},
            {'-', JinjaToken::JinjaToKenType::JinjaTokenSub},
            {'*', JinjaToken::JinjaToKenType::JinjaTokenMul},
            {'/', JinjaToken::JinjaToKenType::JinjaTokenDiv},
            {'|', JinjaToken::JinjaToKenType::JinjaTokenFliter}
    };

    static std::map <char, char> escapeChars = {
            {'n', '\n'}, {'r', '\r'}, {'t', '\t'}, {'b', '\b'}, {'f', '\f'}, {'v', '\v'}, {'\\', '\\'},
            {'\'', '\''}, {'\"', '\"'}, {'0', '\0'}
    };

    static std::map <std::string, JinjaToken::JinjaToKenType> keyWords = {
            {"for", JinjaToken::JinjaToKenType::JinjaTokenFor},
            {"endfor", JinjaToken::JinjaToKenType::JinjaTokenEndFor},
            {"if", JinjaToken::JinjaToKenType::JinjaTokenIf},
            {"else", JinjaToken::JinjaToKenType::JinjaTokenElse},
            {"endif", JinjaToken::JinjaToKenType::JinjaTokenEndif},
            {"set", JinjaToken::JinjaToKenType::JinjaTokenSet},
            {"in", JinjaToken::JinjaToKenType::JinjaTokenIn},
            {"and", JinjaToken::JinjaToKenType::JinjaTokenAnd},
            {"or", JinjaToken::JinjaToKenType::JinjaTokenOr},
    };

    // 一个Jinja块
    struct JinjaBlock {
        enum JinjaBlockType {
            JinjaBlockOriginal = 0, JinjaBlockEmpty, JinjaBlockVar, JinjaBlockFor, 
            JinjaBlockEndFor, JinjaBlockIf, JinjaBlockElse, JinjaBlockEndIf,
            JinjaBlockSet
        };

        JinjaBlockType type = JinjaBlockType::JinjaBlockOriginal;
        std::string value;
        std::vector <JinjaToken> tokens;

        bool IsWhite(char c);

        bool IsDigit(char c);

        bool IsAlpha(char c);

        int FindNextChar(int pos, int end, int target);

        JinjaBlock () {}

        JinjaBlock(const std::string &value);
    };

    JinjaVar JinjaBinaryOp(const JinjaVar &a, const JinjaVar &b, JinjaToken::JinjaToKenType op);

    JinjaVar JinjaTrim(const JinjaVar &a);

    int GetOpLevel(JinjaToken::JinjaToKenType type);

    // Jinja模板
    struct JinjaTemplate {
        std::string temp;
        std::vector <JinjaBlock> blocks;

        JinjaTemplate () {}

        JinjaTemplate (const std::string &temp);

        JinjaVar ComputeExpression(JinjaVar &local, std::vector <JinjaToken> tokens, int st, int end);

        void Parse(int st, int end, JinjaVar &var, std::string &ret);

        std::string Apply(const JinjaVar &var);
    };
}

#endif //FASTLLM_TEMPLATE_H