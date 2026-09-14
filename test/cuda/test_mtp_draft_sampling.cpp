#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool b, const char *m) { if (!b) throw std::runtime_error(m); }
template<class T> void array(const std::vector<T> &v) {
    std::cout << '[';
    for (size_t i=0; i<v.size(); ++i) std::cout << (i ? "," : "") << v[i];
    std::cout << ']';
}
struct Case { const char *name; std::vector<float> p, q; int draftK; float temperature=1, topP=1; int verifyK=-1; };

// ---- 过滤语义 CPU 模型（与 flashinfer kernel 判据逐条对齐，供结构性断言使用）----
// 严格不等前缀语义：以"严格大于"计数/求和，同值（tie）整组同判定：
//   JOINT: 保留 y <=> #{z: p_z > p_y} < K 且 Σ_{z: p_z > p_y} p_z < P
//          （"先 top-p 再 top-k"与之等价，因为两个条件都是原始概率的降序前缀约束）
//   CHAIN: 先取 top-K（p >= 第 K 大值，tie 整组），重归一后再在该分布上做 top-p
// topP >= 1 对应 TopPRenormProb 的 fast-path（整行保留后按行和归一），不产生截断。
static std::vector<double> toDist(const std::vector<float> &p, float temperature) {
    std::vector<double> d(p.size(), 0.0);
    double s = 0;
    for (size_t j = 0; j < p.size(); ++j) {
        d[j] = std::pow((double)p[j], 1.0 / (double)temperature);
        s += d[j];
    }
    if (s > 0) {
        for (double &x : d) x /= s;
    }
    return d;
}
// cntGt[y] = #{z: d_z > d_y}，sumGt[y] = Σ_{z: d_z > d_y} d_z
static void strictGreaterStats(const std::vector<double> &d, std::vector<int> &cntGt,
                               std::vector<double> &sumGt) {
    cntGt.assign(d.size(), 0);
    sumGt.assign(d.size(), 0.0);
    for (size_t y = 0; y < d.size(); ++y) {
        int c = 0;
        double s = 0;
        for (size_t z = 0; z < d.size(); ++z) {
            if (d[z] > d[y]) { ++c; s += d[z]; }
        }
        cntGt[y] = c;
        sumGt[y] = s;
    }
}
static std::vector<char> jointKeep(const std::vector<double> &d, int K, double P) {
    std::vector<int> cntGt;
    std::vector<double> sumGt;
    strictGreaterStats(d, cntGt, sumGt);
    std::vector<char> keep(d.size(), 0);
    for (size_t y = 0; y < d.size(); ++y) {
        keep[y] = (cntGt[y] < K && (P >= 1.0 || sumGt[y] < P)) ? 1 : 0;
    }
    return keep;
}
static std::vector<char> chainKeep(const std::vector<double> &d, int K, double P) {
    std::vector<double> sorted = d;
    std::sort(sorted.rbegin(), sorted.rend());
    std::vector<char> topK(d.size(), 0);
    if (K >= (int)d.size()) {
        for (size_t y = 0; y < d.size(); ++y) topK[y] = (d[y] > 0) ? 1 : 0;
    } else {
        double pivot = sorted[K - 1];
        for (size_t y = 0; y < d.size(); ++y) topK[y] = (d[y] > 0 && d[y] >= pivot) ? 1 : 0;
    }
    double s = 0;
    for (size_t y = 0; y < d.size(); ++y) {
        if (topK[y]) s += d[y];
    }
    std::vector<double> q(d.size(), 0.0);
    if (s > 0) {
        for (size_t y = 0; y < d.size(); ++y) {
            if (topK[y]) q[y] = d[y] / s;
        }
    }
    std::vector<int> cntGt;
    std::vector<double> sumGt;
    strictGreaterStats(q, cntGt, sumGt);
    std::vector<char> keep(d.size(), 0);
    for (size_t y = 0; y < d.size(); ++y) {
        keep[y] = (topK[y] && (P >= 1.0 || sumGt[y] < P)) ? 1 : 0;
    }
    return keep;
}
static std::vector<double> filteredDist(const std::vector<double> &d, const std::vector<char> &keep) {
    std::vector<double> out(d.size(), 0.0);
    double s = 0;
    for (size_t y = 0; y < d.size(); ++y) {
        if (keep[y]) { out[y] = d[y]; s += d[y]; }
    }
    if (s > 0) {
        for (double &x : out) x /= s;
    }
    return out;
}

int main() {
    check(cudaSetDevice(0));
    constexpr int vocab=128, samples=6000;
    std::vector<Case> cases = {
        {"equal_80_20", {0.8f,0.2f}, {0.8f,0.2f}, 2},
        {"equal_60_30_10", {0.6f,0.3f,0.1f}, {0.6f,0.3f,0.1f}, 3},
        {"different_p_q", {0.2f,0.5f,0.3f}, {0.6f,0.3f,0.1f}, 3},
        {"draft_support_subset", {0.2f,0.5f,0.3f}, {0.6f,0.3f,0.1f}, 2},
        {"pivot_ties_K20", std::vector<float>(vocab,1.0f/vocab), std::vector<float>(vocab,1.0f/vocab), 20},
        {"temperature_06", {0.2f,0.5f,0.3f}, {0.6f,0.3f,0.1f}, 3, 0.6f, 1.0f},
        {"nucleus_075", {0.2f,0.5f,0.3f}, {0.6f,0.3f,0.1f}, 3, 1.0f, 0.75f},
        {"nucleus_095", {0.9f,0.07f,0.03f}, {0.7f,0.28f,0.02f}, 3, 1.0f, 0.95f},
        {"candidate_cap_K64", std::vector<float>(vocab,1.0f/vocab), std::vector<float>(vocab,1.0f/vocab), 64},
    };
    // 可区分 JOINT/CHAIN 的 case：128 维几何偏斜分布（r=0.9），topK=20 < 128，
    // topP=0.95 的 nucleus 边界（~第 29 位）落在 top-K 之外：
    //   JOINT 保留 top-20（|R_joint|=20）；CHAIN 先取 top-20 再在重归一化后做 nucleus，
    //   保留 top-18（|R_chain|=18），JOINT-only 尾部 = {18,19}（按 JOINT 重归一后的草稿律
    //   尾部质量 3.25%，N=6000 约 195 条，真卡实测 204 条），
    //   TV≈0.03；真卡实测：验证侧退化为 CHAIN 时 max_z≈10.2 > 7（阈值）被捕获。
    {
        std::vector<float> p(vocab);
        double ps=0;
        for (int j=0;j<vocab;++j) { p[j]=(float)std::pow(0.9,j); ps+=p[j]; }
        for (int j=0;j<vocab;++j) p[j]/=(float)ps;
        cases.push_back({"skewed_128_k20_p95", p, p, 20, 1.0f, 0.95f, 20});
    }
    bool all_ok=true;
    for (const auto &c:cases) {
        std::vector<float> draftLogits(vocab,-1000.0f), targetLogits(samples*2*vocab,-1000.0f);
        for (int j=0;j<(int)c.q.size();++j) draftLogits[j]=std::log(c.q[j]);
        for (int i=0;i<samples*2;++i)
            for (int j=0;j<(int)c.p.size();++j) targetLogits[i*vocab+j]=std::log(c.p[j]);
        float *dDraft, *dTarget;
        check(cudaMalloc(&dDraft,vocab*sizeof(float)));
        check(cudaMalloc(&dTarget,targetLogits.size()*sizeof(float)));
        check(cudaMemcpy(dDraft,draftLogits.data(),vocab*sizeof(float),cudaMemcpyHostToDevice));
        check(cudaMemcpy(dTarget,targetLogits.data(),targetLogits.size()*sizeof(float),cudaMemcpyHostToDevice));
        std::vector<int> drafts(samples), ids(samples*c.draftK), ordinary(samples*2,-1);
        std::vector<float> probabilities(samples*c.draftK), temperatures(samples*2,c.temperature), topP(samples*2,c.topP);
        std::vector<int> topK(samples*2,c.verifyK>0?c.verifyK:(int)c.p.size());
        require(FastllmCudaTopKTopPSampling(dTarget,temperatures.data(),topK.data(),topP.data(),ordinary.data(),samples*2,vocab),"ordinary sampling failed");
        std::vector<int> ordinaryCounts(vocab,0), draftCounts(vocab,0), verifiedCounts(vocab,0);
        for (int t:ordinary) { require(t>=0&&t<vocab,"ordinary invalid token"); ++ordinaryCounts[t]; }
        std::vector<double> expectedP(vocab,0), expectedDraft(vocab,0), draftVariance(vocab,0);
        double pSum=0;
        for (int j=0;j<(int)c.p.size();++j) { expectedP[j]=std::pow(c.p[j],1.0/c.temperature);pSum+=expectedP[j]; }
        for (double &p:expectedP) p/=pSum;
        std::vector<double> sortedP=expectedP;std::sort(sortedP.rbegin(),sortedP.rend());
        double cumulative=0, threshold=0;
        for(double p:sortedP) { cumulative+=p;threshold=p;if(cumulative>=c.topP)break; }
        pSum=0;for(double &p:expectedP) { if(p<threshold)p=0;pSum+=p; }
        // JOINT top-k（仅 verifyK<支撑集大小时生效）：保留集内第 K 大值 pivot，
        // 保留 p>=pivot（tie 组整组，与 RadixTopKRenormProbMultiCTA 的 >= pivot 一致）
        int verifyK = c.verifyK>0 ? c.verifyK : (int)c.p.size();
        if (verifyK < (int)c.p.size()) {
            std::vector<double> surv;
            for (double p : expectedP) if (p > 0) surv.push_back(p);
            std::sort(surv.rbegin(),surv.rend());
            if ((int)surv.size() > verifyK) {
                double pivot = surv[verifyK-1];
                pSum=0;
                for (double &p : expectedP) { if (p < pivot) p = 0; pSum += p; }
            }
        }
        for(double &p:expectedP)p/=pSum;
        // 过滤语义 CPU 模型（与上面 expectedP 同源，见文件头 helper）：
        //   verifyJoint / verifyChain = 验证侧目标分布的 JOINT / CHAIN 保留集
        //   draftJoint               = 草稿侧 q 的 JOINT 保留集（草稿采样走 JOINT 顺序）
        std::vector<double> verifyDist = toDist(c.p, c.temperature);
        std::vector<char> verifyJoint = jointKeep(verifyDist, verifyK, c.topP);
        std::vector<char> verifyChain = chainKeep(verifyDist, verifyK, c.topP);
        std::vector<double> verifyJointProb = filteredDist(verifyDist, verifyJoint);
        std::vector<double> draftDist = toDist(c.q, c.temperature);
        std::vector<char> draftJoint = jointKeep(draftDist, c.draftK, c.topP);
        std::vector<double> draftJointProb = filteredDist(draftDist, draftJoint);
        int draftKeepSize = 0, verifyTailSize = 0;
        for (size_t y = 0; y < draftJoint.size(); ++y) if (draftJoint[y]) ++draftKeepSize;
        if (verifyK < (int)c.p.size() && c.topP < 1.0f) {
            for (size_t y = 0; y < verifyJoint.size(); ++y) {
                if (verifyJoint[y] && !verifyChain[y]) ++verifyTailSize;
            }
        }
        // 自校验：当 top-k 与 top-p 同时生效时，JOINT 与 CHAIN 保留集必须不同且
        // JOINT-only 尾部（R_joint \ R_chain）非空，否则该 case 无法捕获
        // "验证侧 JOINT 分支退化为 CHAIN"的回归（尾部为空时二元断言恒真 = 空转）
        if (verifyK < (int)c.p.size() && c.topP < 1.0f) {
            require(verifyJoint != verifyChain && verifyTailSize > 0,
                "case cannot distinguish joint from chain filter order (empty JOINT-only tail)");
        }
        double maxSumError=0;
        for (int i=0;i<samples;++i) {
            int count=0;
            require(FastllmCudaMtpDraftSpecSampling(dDraft,c.temperature,c.draftK,c.topP,0x123456789abcdef0ULL+i,
                &drafts[i],ids.data()+i*c.draftK,probabilities.data()+i*c.draftK,&count,vocab),"draft sampling failed");
            require(count>0&&count<=c.draftK&&drafts[i]>=0&&drafts[i]<vocab,"invalid draft");
            double sum=0; bool inSupport=false;
            std::vector<bool> seen(vocab,false);
            for (int j=0;j<count;++j) {
                int id=ids[i*c.draftK+j]; float probability=probabilities[i*c.draftK+j];
                require(id>=0&&id<vocab&&!seen[id]&&probability>0,"invalid candidate set");
                require(id<(int)draftJoint.size()&&draftJoint[id],
                    "draft candidate is outside the JOINT keep set");
                seen[id]=true; sum+=probability;
                expectedDraft[id]+=probability;draftVariance[id]+=probability*(1.0-probability);
                inSupport |= id==drafts[i];
            }
            require(inSupport,"draft is outside reported q support");
            // 草稿侧 JOINT 覆盖断言：|R_joint| <= draftK 时候选集必须恰好覆盖 R_joint。
            // master 的链式顺序会把 R_joint \ R_chain 的尾部 token（本 case = {18,19}）
            // 从候选集中剔除（候选数只剩 |R_chain|），而"概率和 == 1"这类自洽检查
            // 抓不到该回归——只有覆盖断言能抓到。
            if (draftKeepSize <= c.draftK) {
                for (int y=0;y<(int)draftJoint.size();++y) {
                    require(!draftJoint[y]||(bool)seen[y],
                        "draft candidates do not cover the JOINT keep set (chain-order regression?)");
                }
            }
            maxSumError=std::max(maxSumError,std::abs(sum-1));
            ++draftCounts[drafts[i]];
        }
        std::vector<int> output(samples*2,-1), accepted(samples,-1);
        // MTP 分布化草稿路径：jointFilterOrder=true（与上游 PR #733 引入的
        // FastllmCudaMtpDraftSpecSampling + FastllmCudaDFlashRejectionSampling
        // 签名同步）。草稿侧采样用 JOINT 顺序，验证侧目标分布必须用 JOINT 顺序
        // 才能保证"草稿律 == 上报 q"的无损前提。
        require(FastllmCudaDFlashRejectionSampling(dTarget,temperatures.data(),topK.data(),topP.data(),
            drafts.data(),ids.data(),probabilities.data(),output.data(),accepted.data(),samples,1,c.draftK,vocab,
            /* jointFilterOrder */ true),"verification failed");
        // 第二组：链式过滤（DFlash 既有语义，未被本 commit 改动）回归。
        std::vector<int> output_chain(samples*2,-1), accepted_chain(samples,-1);
        require(FastllmCudaDFlashRejectionSampling(dTarget,temperatures.data(),topK.data(),topP.data(),
            drafts.data(),ids.data(),probabilities.data(),output_chain.data(),accepted_chain.data(),samples,1,c.draftK,vocab,
            /* jointFilterOrder */ false),"chain-filter verification failed");
        int acceptedTotal=0;
        for (int i=0;i<samples;++i) {
            require(accepted[i]>=0&&accepted[i]<=1,"invalid acceptance count");
            require(output[2*i]>=0&&output[2*i]<vocab,"invalid verified token");
            ++verifiedCounts[output[2*i]];
            if(accepted[i]) { require(output[2*i]==drafts[i],"accepted token differs from draft"); ++acceptedTotal; }
        }
        // JOINT-only 尾部（R_joint \ R_chain）二元接收率测试：
        // 草稿来自 JOINT 过滤，因此 draft 会落在 R_joint \ R_chain 的尾部
        // （skewed_128_k20_p95：R_joint = {0..19}、R_chain = {0..17}，尾部 = {18,19}，
        //  按 JOINT 重归一后的草稿律，尾部质量 3.25%，N=6000 约 195 条，真卡实测 204 条）。
        //  该尾部在两个过滤顺序下的目标分布不同：
        //   - jointFilterOrder=true （JOINT）：p_J(d) > 0，接收概率 = min(1, p_J(d)/q(d))；
        //     草稿侧与验证侧同分布时 p_J(d) == q(d) → 恒收（结构性硬约束）；
        //   - jointFilterOrder=false（CHAIN）：p_C(d) = 0 → 恒拒（结构性硬约束）。
        // 判据必须是「在 R_joint 内但不在 R_chain 内」——不能写成「在 top-K 内但在
        // 原始 nucleus 外」：后者在 JOINT 下同样被拒（JOINT 保留集本身含 nucleus 条件），
        // 且对 nucleus 边界落在 top-K 之外的分布恒为空集，断言会退化成恒真（空转）。
        // 仅对 verifyK<支撑集 && topP<1 的 case 启用（否则 JOINT == CHAIN）。
        int joBndAccept = 0, joBndTotal = 0, chBndAccept = 0;
        double joBndExpected = 0.0, joBndVariance = 0.0;
        bool tailAlwaysAccepted = true;
        if (verifyTailSize > 0) {
            for (int i = 0; i < samples; ++i) {
                int d = drafts[i];
                if (d < 0 || d >= (int)verifyJoint.size()) continue;
                if (!verifyJoint[d] || verifyChain[d]) continue;
                ++joBndTotal;
                if (accepted[i]) ++joBndAccept;
                require(accepted_chain[i] == 0,
                    "joint-only-tail draft was accepted under CHAIN filter (expected 0); "
                    "this signals jointFilterOrder=false is treating tail as in-support");
                if (accepted_chain[i]) ++chBndAccept;
                // Leviathan 接收概率 min(1, p_J(d)/q(d))；两侧同分布时恒为 1
                double r = (draftJointProb[d] > 0) ?
                    std::min(1.0, verifyJointProb[d]/draftJointProb[d]) : 0.0;
                joBndExpected += r; joBndVariance += r*(1.0-r);
                if (r < 1.0 - 1e-12) tailAlwaysAccepted = false;
            }
            // 硬断言 1：CHAIN 必须把 JOINT-only 尾部的所有 draft 拒掉（一个都不收）
            require(chBndAccept == 0,
                "CHAIN filter accepted a JOINT-only-tail draft; "
                "jointFilterOrder=false path is inconsistent with nucleus boundary");
            if (tailAlwaysAccepted) {
                // 硬断言 2：草稿与目标在同一 JOINT 保留集上同分布 → JOINT 必须全收
                require(joBndAccept == joBndTotal,
                    "JOINT filter rejected a JOINT-only-tail draft; "
                    "jointFilterOrder=true path is wrong (should accept all in-keep-set drafts)");
            } else {
                // 异分布时按 Leviathan 期望值 min(1, p_J/q) 做 3σ 下界检验
                require(joBndAccept >= joBndExpected - 3.0*std::sqrt(joBndVariance) - 3.0,
                    "JOINT acceptance count is far below the Leviathan expectation");
            }
        }
        double maxZ=0, ordinaryMaxZ=0, draftMaxZ=0;
        bool ok=maxSumError<1e-5;
        for(int j=0;j<vocab;++j) {
            double p=expectedP[j];
            if(draftVariance[j]>0) draftMaxZ=std::max(draftMaxZ,std::abs(draftCounts[j]-expectedDraft[j])/std::sqrt(draftVariance[j]));
            else ok &= std::abs(draftCounts[j]-expectedDraft[j])<1e-4;
            double sd=std::sqrt(samples*p*(1-p));
            double ordinarySd=std::sqrt(2*samples*p*(1-p));
            if(sd>0) maxZ=std::max(maxZ,std::abs(verifiedCounts[j]-samples*p)/sd);
            else ok &= verifiedCounts[j]==int(samples*p);
            if(ordinarySd>0) ordinaryMaxZ=std::max(ordinaryMaxZ,std::abs(ordinaryCounts[j]-2*samples*p)/ordinarySd);
            else ok &= ordinaryCounts[j]==int(2*samples*p);
        }
        ok &= maxZ<7 && ordinaryMaxZ<7 && draftMaxZ<7;
        all_ok &= ok;
        std::cout << "{\"case\":\"" << c.name << "\",\"samples\":" << samples
                  << ",\"draft_k\":" << c.draftK << ",\"accepted\":" << acceptedTotal
                  << ",\"max_q_sum_error\":" << maxSumError << ",\"max_z\":" << maxZ
                  << ",\"draft_max_z\":" << draftMaxZ << ",\"ordinary_max_z\":" << ordinaryMaxZ
                  << ",\"draft_keep_size\":" << draftKeepSize
                  << ",\"verify_joint_only_tail\":" << verifyTailSize
                  << ",\"joint_tail_drafts\":" << joBndTotal
                  << ",\"joint_tail_accepted_JOINT\":" << joBndAccept
                  << ",\"joint_tail_accepted_CHAIN\":" << chBndAccept
                  << ",\"pass\":" << (ok?"true":"false")
                  << ",\"expected\":"; array(expectedP);
        std::cout << ",\"draft_counts\":"; array(draftCounts);
        std::cout << ",\"verified_counts\":"; array(verifiedCounts);
        std::cout << ",\"ordinary_counts\":"; array(ordinaryCounts);
        std::cout << "}\n" << std::flush;
        check(cudaFree(dDraft)); check(cudaFree(dTarget));
    }
    return all_ok?0:2;
}
