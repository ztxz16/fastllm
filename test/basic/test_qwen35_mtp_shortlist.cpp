#include "models/qwen3_5_mtp_shortlist.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <set>

using fastllm::mtp_shortlist::Read;
using fastllm::mtp_shortlist::Project;
using fastllm::mtp_shortlist::Shard;
static void Check(bool ok) { if (!ok) throw std::runtime_error("shortlist regression failed"); }
int main(int argc, char **argv) {
    const auto dir = std::filesystem::temp_directory_path() /
        ("fastllm-mtp-shortlist-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        Check(std::filesystem::create_directory(dir));
        const auto file = (dir / "ids.txt").string();
        std::vector<int> ids;
        for (const char *invalid : {"", "1 1", "-1", "512", "1x", "999999999999999999999999"}) {
            { std::ofstream out(file); out << invalid; }
            Check(!Read(file.c_str(), 512, ids) && ids.empty());
        }
        { std::ofstream out(file); out << "260 7 259 3"; }
        Check(Read(file.c_str(), 512, ids) && ids == std::vector<int>({3, 7, 259, 260}));
        Shard a, b, empty;
        Check(Project(ids, {{0,256}}, 256, 512, a));
        Check(Project(ids, {{256,512}}, 256, 512, b));
        Check(a.logicalSize == 2 && b.logicalSize == 2 && a.rows.size() == 128 && b.rows.size() == 128);
        Check(a.rows[0] == 3 && a.rows[1] == 7 && a.rows.back() == 7);
        Check(b.rows[0] == 3 && b.rows[1] == 4 && b.tokenIds.back() == 260);
        // Noncontiguous ranges in reverse order must map to physical source rows.
        Check(Project(ids, {{256,384},{0,128}}, 256, 512, a));
        Check(a.rows[0] == 131 && a.rows[1] == 135 && a.rows[2] == 3 && a.rows[3] == 4);
        Check(a.tokenIds[0] == 3 && a.tokenIds[2] == 259);
        Check(Project(ids, {{128,256}}, 128, 512, empty) && empty.rows.empty());
        Check(Project(ids, {}, 0, 512, empty) && empty.rows.empty());
        Check(!Project(ids, {{0,256},{128,384}}, 512, 512, empty));
        Check(!Project(ids, {{0,256}}, 255, 512, empty));
        Check(!Project({3,3}, {{0,256}}, 256, 512, empty));
        Check(!Project({7,3}, {{0,256}}, 256, 512, empty));
        // Padding introduces no candidates; tied logits still choose the lowest ID.
        Check(Project({3,259}, {{0,512}}, 512, 512, a));
        std::set<int> candidates(a.tokenIds.begin(), a.tokenIds.end());
        Check(candidates == std::set<int>({3,259}) && a.tokenIds.front() == 3);
        // DFlash merges compact proxy IDs before restoring global IDs. The
        // sorted maps preserve tie order across contiguous shards, and the
        // unpadded logical vector rejects duplicate alignment rows.
        using fastllm::mtp_shortlist::MapProxyId;
        const std::vector<int> left{3,7,127}, right{259,380,500};
        Check(MapProxyId(0, 0, left) == 3 && MapProxyId(2, 0, left) == 127);
        Check(MapProxyId(256, 256, right) == 259 && MapProxyId(258, 256, right) == 500);
        Check(MapProxyId(3, 0, left) == -1 && MapProxyId(255, 256, right) == -1);
        Check(MapProxyId(-1, 0, left) == -1 && MapProxyId(259, 256, right) == -1);
        Check(MapProxyId(2, 0, left) < MapProxyId(256, 256, right));
        if (argc > 1) {
            Check(Read(argv[1], 248320, ids) && ids.size() == 131072);
            Check(Project(ids, {{0,124160}}, 124160, 248320, a));
            Check(Project(ids, {{124160,248320}}, 124160, 248320, b));
            Check(a.logicalSize == 89286 && b.logicalSize == 41786);
            Check(a.rows.size() == 89344 && b.rows.size() == 41856);
            std::set<int> combined(a.tokenIds.begin(), a.tokenIds.end());
            combined.insert(b.tokenIds.begin(), b.tokenIds.end());
            Check(combined == std::set<int>(ids.begin(), ids.end()));
        }
        std::filesystem::remove_all(dir);
        std::cout << "PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::filesystem::remove_all(dir);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
