# ROCm wheel 构建镜像

编译、架构选择、镜像选项和 wheel 安装说明统一维护在 `master` 分支：

- [ROCm 编译与 wheel 打包](https://github.com/ztxz16/fastllm/blob/master/docs/rocm.md)

两个 Dockerfile 共用 `install-sdk.sh`，构建 context 均为 `whl_docker_rocm/`。
在仓库根目录运行 `bash make_whl_rocm_docker.sh --help` 可查看入口参数。
