//
// Created by huangyuyang on 11/4/24.
//

#ifndef ALIVETHREAD_H
#define ALIVETHREAD_H

#include <thread>
#include <vector>
#include <unistd.h>

namespace fastllm {
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
                asm volatile("dmb ish");
                if (task->signal == 1) {
                    task->op->Run();
                    task->signal = 0;
                    asm volatile("dmb ish");
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
            asm volatile("dmb ish");
            this->task->signal = 1;
            asm volatile("dmb ish");
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
    };
}

#endif 
