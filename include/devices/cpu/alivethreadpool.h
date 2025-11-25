//
// Created by huangyuyang on 11/4/24.
//

#ifndef ALIVETHREAD_H
#define ALIVETHREAD_H

#include <thread>
#include <vector>
#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#else
#include <unistd.h>
#endif
#include <cstring>

namespace fastllm {
    static void barrier() {
#ifdef __aarch64__
        asm volatile("dmb ish");
#elif defined(_WIN32) || defined(_WIN64)
        MemoryBarrier();
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
            int cnt = 0;
            auto lastRunTime = std::chrono::system_clock::now();
            while (true) {
                barrier();
                if (task->signal == 1) {
                    task->op->Run();
                    task->signal = 0;
                    barrier();
                    lastRunTime = std::chrono::system_clock::now();
                }

                cnt = (cnt + 1) & ((1 << 16) - 1);
                if (cnt == 0) {
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::system_clock::now() - lastRunTime);
                    double gap = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
                    if (gap > 3) {
                        std::this_thread::sleep_for(std::chrono::microseconds(2000));
                    }
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

        bool TryWait() {
            int a = task->signal;
            return a == 0;
        }
    };

    struct AliveThreadPool {
        std::pair <int, int> curActivateThreadInterval; // 设定当前激活 [curActivateThreadInterval.first, curActivateThreadInterval.second) 的线程  

        std::vector <AliveThreadLoop*> loops;
        std::vector <std::thread*> threads;
        
        AliveThreadPool (int threadNum) {
            for (int i = 0; i < threadNum; i++) {
                this->loops.push_back(new AliveThreadLoop(i));
                this->threads.push_back(new std::thread(*(this->loops[i])));
            }
            curActivateThreadInterval = std::make_pair(0, threadNum);
        }

        void PushOp(int tid, MultiThreadBaseOp *op) {
            this->loops[tid]->PushOp(op);
        }

        void Wait(int tid) {
            this->loops[tid]->Wait();
        }

        bool TryWait(int tid) {
            return this->loops[tid]->TryWait();
        }

        void Shutdown() {
            /// TODO: shutdown
        }

        void ResizeThreads(int threadNum) {
            for (int i = this->threads.size(); i < threadNum; i++) {
                this->loops.push_back(new AliveThreadLoop(i));
                this->threads.push_back(new std::thread(*(this->loops[i])));
            }
            curActivateThreadInterval = std::make_pair(0, threadNum);
        }
    };

    struct MultiThreadMultiOps : MultiThreadBaseOp {
        std::vector <MultiThreadBaseOp*> ops;

        void Run() {
            for (int i = 0; i < ops.size(); i++) {
                ops[i]->Run();
            }
        }

        ~MultiThreadMultiOps() {
            for (int i = 0; i < ops.size(); i++) {
                delete[] ops[i];
            }
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

    static void RunMultiThreadMemcpy(uint8_t *output, uint8_t *input, int len, AliveThreadPool *pool, bool force = false) {
        if (!force && len < 256 * 1024) {
            memcpy(output, input, len);
            return;
        }
        int threadNum = pool->threads.size();
        threadNum = std::min(8, threadNum);

        int per = len / threadNum;
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

    struct MultiThreadMemcpyMultiLinesTask {
        uint8_t *output, *input;
        size_t len;

        MultiThreadMemcpyMultiLinesTask () {}

        MultiThreadMemcpyMultiLinesTask (uint8_t *output, uint8_t *input, size_t len) :
            output(output), input(input), len(len) {}
    };

    struct MultiThreadMemcpySingleMultiLinesOp : MultiThreadBaseOp {
        uint8_t *input, *output;
        uint64_t rows, lens, inputStride, outputStride;

        MultiThreadMemcpySingleMultiLinesOp (uint8_t *input, uint8_t *output, 
            uint64_t rows, uint64_t lens, uint64_t inputStride, uint64_t outputStride) : 
            input(input), output(output), rows(rows), lens(lens), inputStride(inputStride), outputStride(outputStride) {}

        void Run() {
            for (int i = 0; i < rows; i++) {
                memcpy(output + i * outputStride, input + i * inputStride, lens);
            }            
        }
    };

    struct MultiThreadMemcpyMultiLinesOp : MultiThreadBaseOp {
        MultiThreadMemcpyMultiLinesTask *tasks;
        int st, end;

        MultiThreadMemcpyMultiLinesOp (MultiThreadMemcpyMultiLinesTask *tasks, int st, int end) : 
            tasks(tasks), st(st), end(end) {}

        void Run() {
            for (int i = st; i < end; i++) {
                memcpy(tasks[i].output, tasks[i].input, tasks[i].len);
            }            
        }
    };

    static void RunMultiThreadMemcpyMultiLines(std::vector <MultiThreadMemcpyMultiLinesTask> &tasks, AliveThreadPool *pool) {
        int threadNum = pool->threads.size();
        int n = tasks.size();
        int per = n / pool->threads.size();
        int cur = 0;
        std::vector<fastllm::MultiThreadMemcpyMultiLinesOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? n : cur + per + (cur + per * (threadNum - i) < n));
            ops.push_back(new MultiThreadMemcpyMultiLinesOp(
                tasks.data(), cur, end));
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

    struct MultiThreadReduceOp : MultiThreadBaseOp {
        int inputLen;
        float **inputs, *values;
        float *output, *lastOutput;
        int st, end;

        MultiThreadReduceOp (int inputLen, float **inputs, float *values, float *output, float *lastOutput, int st, int end) : 
            inputLen(inputLen), inputs(inputs), values(values), output(output), lastOutput(lastOutput), st(st), end(end) {}

        void Run() {
            for (int i = 0; i < inputLen; i++) {
                float value = values[i];
                for (int j = st; j < end; j++) {
                    output[j] += value * inputs[i][j];
                }
            }

            if (lastOutput != nullptr) {
                memcpy(lastOutput + st, output + st, (end - st) * sizeof(float));
            }
        }
    };

    static void RunMultiThreadReduce(int inputLen, float **inputs, float *values, 
                                float *output, float *lastOutput, int dim, AliveThreadPool *pool) {
        int threadNum = pool->threads.size();
        threadNum = std::min(threadNum, 8);

        int per = dim / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadReduceOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? dim : cur + per + (cur + per * (threadNum - i) < dim));
            ops.push_back(new MultiThreadReduceOp(inputLen, inputs, values, output, lastOutput, cur, end));
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

    struct MultiThreadMoeReduceOp : MultiThreadBaseOp {
        std::vector <std::pair <int, float> > *task;
        std::vector <float> *tempResult;
        float *curOutput; 
        int dim, st, end;

        MultiThreadMoeReduceOp (std::vector <std::pair <int, float> > *task, 
                                std::vector <float> *tempResult,
                                float *curOutput, 
                                int dim, int st, int end) : 
            task(task), tempResult(tempResult), curOutput(curOutput), dim(dim), st(st), end(end) {}

        void Run() {
            for (int i = st; i < end; i++) {
                float value = (*task)[i].second;
                float *lastResult = tempResult->data() + (*task)[i].first * dim;
                float *curResult = curOutput + i * dim;
                for (int j = 0; j < dim; j++) {
                    lastResult[j] += value * curResult[j];
                }
            }   
        }
    };

    static void RunMultiThreadMoeReduce(std::vector <std::pair <int, float> > *task, 
                                        std::vector <float> *tempResult, float *curOutput, int dim, AliveThreadPool *pool) {
        int threadNum = pool->threads.size();
        threadNum = std::min(threadNum, 8);

        int n = task->size();
        int per = n / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadMoeReduceOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? n : cur + per + (cur + per * (threadNum - i) < n));
            ops.push_back(new MultiThreadMoeReduceOp(task, tempResult, curOutput, dim, cur, end));
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
