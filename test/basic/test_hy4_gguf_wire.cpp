#include "gguf.h"
#include "../../src/models/hy4_flash_ops.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
template <typename T> void Write(std::ostream &out, T value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}
void WriteString(std::ostream &out, const std::string &s) {
    Write<uint64_t>(out, s.size());
    out.write(s.data(), s.size());
}
void WriteTensorFile(const std::filesystem::path &file, const std::string &arch, const std::string &name,
                     int columns, uint32_t wireType, const std::vector<uint8_t> &bytes) {
    std::ofstream out(file, std::ios::binary);
    out.write("GGUF", 4);
    Write<uint32_t>(out, 3);
    Write<uint64_t>(out, 1);
    Write<uint64_t>(out, 1);
    WriteString(out, "general.architecture");
    Write<uint32_t>(out, GGUF_TYPE_STRING);
    WriteString(out, arch);
    WriteString(out, name);
    Write<uint32_t>(out, 2);
    Write<uint64_t>(out, columns);
    Write<uint64_t>(out, 1);
    Write<uint32_t>(out, wireType);
    Write<uint64_t>(out, 0);
    while (static_cast<uint64_t>(out.tellp()) % fastllm::GGUF_DEFAULT_ALIGNMENT)
        out.put(0);
    out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}
void CheckWireType(const std::filesystem::path &dir, bool scalar) {
    const std::string arch = scalar ? "hy_v4_flash" : "deepseek_v41";
    const std::string name = scalar ? "per_layer_token_embd.weight" : "output.weight";
    const auto file = dir / (arch + ".gguf");
    const int columns = scalar ? 96 : QK_MXFP8;
    std::vector<uint8_t> bytes(scalar ? columns : sizeof(block_mxfp8), 0x38);
    if (!scalar)
        bytes.front() = 128; // E8M0 scale 2 times E4M3 1.
    WriteTensorFile(file, arch, name, columns, 43, bytes);
    std::vector<fastllm::ReadGGUFTask> tasks;
    fastllm::AppendGGUFTasks(arch, file.string(), tasks);
    if (tasks.size() != 1 || tasks[0].tensor.type != (scalar ? GGML_TYPE_I8 : GGML_TYPE_MXFP8) ||
        ggml_nbytes(&tasks[0].tensor) != bytes.size())
        throw std::runtime_error("GGUF type 43 layout");
    if (scalar) {
        bytes[0] = 0xb8;
        bytes[1] = 0x01;
        bytes[2] = 0x7e;
        bytes[3] = 0xff;
        uint16_t row[96];
        fastllm::ReadEngramRow(bytes.data(), true, 0, row);
        if (row[0] != 0xbf80 || row[1] != 0x3b00 || row[2] != 0x43e0 || row[3] != 0x7fc0 || row[4] != 0x3f80)
            throw std::runtime_error("HY4 E4M3 scalar values");
    } else {
        float result[QK_MXFP8];
        ggml_type_to_float(tasks[0].tensor.type)(bytes.data(), result, QK_MXFP8);
        for (float value : result)
            if (value != 2.f)
                throw std::runtime_error("MXFP8 block scale");
    }
}
void CheckUnsupportedType(const std::filesystem::path &dir, const std::string &name, uint32_t type) {
    const auto file = dir / "unsupported.gguf";
    WriteTensorFile(file, "hy_v4_flash", name, 32, type, {});
    std::vector<fastllm::ReadGGUFTask> tasks;
    try {
        fastllm::AppendGGUFTasks("hy_v4_flash", file.string(), tasks);
    } catch (const std::runtime_error &error) {
        const std::string message = error.what();
        if (message.find(std::to_string(type)) != std::string::npos && message.find(name) != std::string::npos)
            return;
        throw;
    }
    throw std::runtime_error("Unsupported GGUF tensor type accepted");
}
} // namespace

int main() {
    const auto dir =
        std::filesystem::temp_directory_path() /
        ("fastllm-hy4-wire-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (!std::filesystem::create_directory(dir))
        return 1;
    int status = 0;
    try {
        CheckWireType(dir, false);
        CheckWireType(dir, true);
        CheckUnsupportedType(dir, "output.weight", 32767);
        CheckUnsupportedType(dir, "output.weight", 43);
        CheckUnsupportedType(dir, "per_layer_token_embd.weight", GGML_TYPE_Q8_0);
        std::cout << "PASS: HY4 scalar E4M3, block-scaled MXFP8 and unsupported tensor types\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    std::filesystem::remove_all(dir);
    return status;
}
