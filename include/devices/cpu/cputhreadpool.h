//
// Created by huangyuyang on 7/5/23.
//

#ifndef FASTLLCPUTHREADPOOL_H
#define FASTLLCPUTHREADPOOL_H

#include <mutex>
#include <queue>
#include <functional>
#include <future>
#include <thread>
#include <utility>
#include <vector>

namespace fastllm {
    template <typename T>
    class TaskQueue {
    private:
        std::queue <T> q;
        std::mutex locker;
    public:
        TaskQueue() {}

        ~TaskQueue() {}

        bool Empty() {
            std::unique_lock<std::mutex> lock(locker);
            return q.empty();
        }

        int Size() {
            std::unique_lock<std::mutex> lock(locker);
            return q.size();
        }

        void Push(T &t) {
            std::unique_lock<std::mutex> lock(locker);
            q.emplace(t);
        }

        bool Pop(T &t) {
            std::unique_lock<std::mutex> lock(locker);
            if (q.empty()) {
                return false;
            }
            t = std::move(q.front());
            q.pop();
            return true;
        }
    };

    class ThreadPool {
    private:
        class ThreadWorker
        {
        private:
            int id;
            ThreadPool *pool;
        public:
            ThreadWorker(ThreadPool *pool, const int id) : pool(pool), id(id) {}

            void operator()() {
                std::function<void()> func;
                bool dequeued;

                while (!pool->shutdown) {
                    {
                        std::unique_lock<std::mutex> lock(pool->locker);
                        if (pool->queue.Empty()) {
                            pool->cv.wait(lock);
                        }

                        dequeued = pool->queue.Pop(func);
                    }
                    if (dequeued) {
                        func();
                    }
                }
            }
        };

        bool shutdown = false;
        TaskQueue<std::function<void()>> queue;
        std::vector<std::thread> threads;
        std::mutex locker;
        std::condition_variable cv;
    public:
        ThreadPool(const int t = 4) : threads(std::vector<std::thread>(t)) {
            for (int i = 0; i < threads.size(); ++i) {
                threads[i] = std::thread(ThreadWorker(this, i));
            }
        }
        void Shutdown() {
            shutdown = true;
            cv.notify_all();
            for (int i = 0; i < threads.size(); ++i) {
                if (threads[i].joinable()) {
                    threads[i].join();
                }
            }
        }

        template<typename F, typename... Args>
        auto Submit(F &&f, Args &&...args) -> std::future<decltype(f(args...))> {
            std::function<decltype(f(args...))()> func = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
            auto task_ptr = std::make_shared<std::packaged_task<decltype(f(args...))()>>(func);
            std::function<void()> warpper_func = [task_ptr]() {
                (*task_ptr)();
            };
            queue.Push(warpper_func);
            cv.notify_one();
            return task_ptr->get_future();
        }
    };
}

#endif //FASTLLCPUTHREADPOOL_H
