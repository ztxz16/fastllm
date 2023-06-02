//
// Created by huangyuyang on 6/2/23.
//
#pragma once

#ifndef FASTLLM_UTILS_H
#define FASTLLM_UTILS_H

#include <map>
#include <chrono>
#include <string>
#include <cstdio>

namespace fastllm {
    static double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds> (time2 - time1);
        return double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
    };

    struct TimeRecord {
        std::map<std::string, float> v;
        std::chrono::system_clock::time_point t;

        void Clear() {
            v.clear();
        }

        void Record() {
            t = std::chrono::system_clock::now();
        }

        void Record(const std::string &key) {
            auto now = std::chrono::system_clock::now();
            v[key] += GetSpan(t, now);
            t = now;
        }

        void Print() {
            float s = 0;
            for (auto &it: v) {
                printf("%s: %f s.\n", it.first.c_str(), it.second);
                s += it.second;
            }
            printf("Total: %f s.\n", s);
        }
    };
}

#endif //FASTLLM_UTILS_H
