//
// Created by huangyuyang on 6/13/23.
//

#ifndef FASTLLM_EXECUTOR_H
#define FASTLLM_EXECUTOR_H

#include "device.h"

namespace fastllm {
    class Executor {
    private:
        std::vector <BaseDevice*> devices;
        std::map <std::string, float> profiler;

    public:
        Executor (); // 创建默认的Executor

        ~Executor(); // 析构

        void ClearDevices(); // 清空 devices

        void AddDevice(BaseDevice *device); // 增加一个device

        void SetFirstDevice(const std::string &device); // 设定优先的device

        std::string GetFirstDeviceType(); // 获取优先device的type

        std::vector <int> GetDeviceIds(const std::string &device); // 获取指定device的deviceIds

        bool HasDevice(const std::string &deviceType); // 判断devices中是否包含指定类型的device

        bool CanRunOnFirstDevice(const std::string &opType, const fastllm::DataDict &datas, const fastllm::FloatDict &floatParams,
                       const fastllm::IntDict &intParams);
                       
        // 运行一个op
        void Run(const std::string &opType, const fastllm::DataDict &datas, const fastllm::FloatDict &floatParams,
                 const fastllm::IntDict &intParams);

        // 强制在指定 deviceType 的设备上运行一个 op (例如 "cpu")。
        // 如果存在多个匹配的设备，按 devices 列表中的注册顺序选择第一个能跑该 op 的设备。
        void RunOnDevice(const std::string &deviceType,
                         const std::string &opType,
                         const fastllm::DataDict &datas,
                         const fastllm::FloatDict &floatParams,
                         const fastllm::IntDict &intParams);

        void ClearProfiler();

        void AddProfiler(const std::string &opType, float spend);

        void PrintProfiler();

        std::string firstDevice;
    };
}

#endif //FASTLLM_EXECUTOR_H
