//
// Created by huangyuyang on 11/4/24.
//

#ifndef ALIVETHREAD_H
#define ALIVETHREAD_H

#include <thread>
#include <vector>
#include <unistd.h>
#include <cstring>

namespace fastllm {
    static void barrier() {
#ifdef __aarch64__
        asm volatile("dmb ish");
#else
        __asm__ __volatile__("": : :"memory");
#endif
    }
    struct MultiThreadBaseOp {
        virtual void Run() = 0;
    };

    struct AliveThreadTask {
        int signal;
        MultiThreadBaseOp *op;

        AliveThreadTask () {
            signal = 0;
            op = nullptr;
        }
    };

    struct AliveThreadLoop {
        int id;
        AliveThreadTask realTask;
        volatile AliveThreadTask *task;

        AliveThreadLoop(int id)  {
            this->id = id;
            this->task = &this->realTask;
        }

        void operator()() {
            auto lastRunTime = std::chrono::system_clock::now();
            while (true) {
                barrier();
                if (task->signal == 1) {
                    task->op->Run();
                    task->signal = 0;
                    barrier();
                    lastRunTime = std::chrono::system_clock::now();
                }

                auto duration = std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::system_clock::now() - lastRunTime);
                double gap = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
                if (gap > 3) {
                    sleep(0);
                }
            }
        }

        void PushOp(MultiThreadBaseOp *op) {
            this->task->op = op;
            barrier();
            this->task->signal = 1;
            barrier();
        }

        void Wait() {
            while (true) {
                int a = task->signal;
                if (a == 0) {
                    break;
                }
            }
        }
    };

    struct AliveThreadPool {
        std::vector <AliveThreadLoop*> loops;
        std::vector <std::thread*> threads;
        
        AliveThreadPool (int threadNum) {
            for (int i = 0; i < threadNum; i++) {
                this->loops.push_back(new AliveThreadLoop(i));
                this->threads.push_back(new std::thread(*(this->loops[i])));
            }
        }

        void PushOp(int tid, MultiThreadBaseOp *op) {
            this->loops[tid]->PushOp(op);
        }

        void Wait(int tid) {
            this->loops[tid]->Wait();
        }

        void Shutdown() {
            /// TODO: shutdown
        }
    };

    struct MultiThreadMemcpyOp : MultiThreadBaseOp {
        uint8_t *input, *output;
        int len;

        MultiThreadMemcpyOp (uint8_t *output, uint8_t *input, int len) : input(input), output(output), len(len) {}

        void Run() {
            memcpy(output, input, len);
        }
    };

    static void RunMultiThreadMemcpy(uint8_t *output, uint8_t *input, int len, AliveThreadPool *pool) {
        if (len < 256 * 1024) {
            memcpy(output, input, len);
            return;
        }
        int threadNum = pool->threads.size();
        int per = len / pool->threads.size();
        int cur = 0;
        std::vector<fastllm::MultiThreadMemcpyOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new MultiThreadMemcpyOp(output + cur, input + cur, end - cur));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    // [n, m, k] -> [m, n, k], 以k个元素为单位做转置
    struct MultiThreadTransposeByLineOp : MultiThreadBaseOp {
        uint8_t *input, *output;
        int n, m, k, st, end;

        MultiThreadTransposeByLineOp (uint8_t *output, uint8_t *input, int n, int m, int k, int st, int end) : 
            input(input), output(output), n(n), m(m), k(k), st(st), end(end) {}

        void Run() {
            for (int x = st; x < end; x++) {
                int i = x / m, j = x % m;
                memcpy(output + (j * n + i) * k, input + (i * m + j) * k, k);
            }
        }
    };

    // [n, m, k] -> [m, n, k], 以k个元素为单位做转置
    static void RunMultiThreadTransposeByLine(uint8_t *output, uint8_t *input, int n, int m, int k, AliveThreadPool *pool) {
        /*if (len < 256 * 1024) {
            memcpy(output, input, len);
            return;
        }*/
        int threadNum = pool->threads.size();
        int len = n * m;
        int per = len / pool->threads.size();
        int cur = 0;
        std::vector<fastllm::MultiThreadTransposeByLineOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new MultiThreadTransposeByLineOp(output, input, n, m, k, cur, end));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }
}

#endif 
