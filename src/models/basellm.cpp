//
// Created by huangyuyang on 6/25/23.
//

#include "basellm.h"

namespace fastllm {
    int ResponseContextDict::CreateHandle() {
        locker.lock();
        int newId = 0;
        while (dicts.find(newId) != dicts.end()) {
            newId++;
        }
        dicts[newId] = new ResponseContext();
        locker.unlock();
        return newId;
    }

    ResponseContext *ResponseContextDict::GetHandle(int handleId) {
        locker.lock();
        ResponseContext *ret = dicts.find(handleId) != dicts.end() ? dicts[handleId] : nullptr;
        locker.unlock();
        return ret;
    }

    void ResponseContextDict::RemoveHandle(int handleId) {
        locker.lock();
        if (dicts.find(handleId) != dicts.end()) {
            delete dicts[handleId];
            dicts.erase(handleId);
        }
        locker.unlock();
    }

    void ResponseContext::Init(int blocks) {
        pastKeyValues.clear();
        for (int i = 0; i < blocks; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }
        intParams.clear();
        currentTokens.clear();
        preTokens = 0;
    }
}