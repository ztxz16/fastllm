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
    //   JOINT 保留 top-20；CHAIN 先取 top-20 再在重归一化后做 nucleus，
    //   会切掉 top-20 的尾部（~第 18-19 位），TV≈0.03，Z 检验可捕获。
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
        // 自校验：当 top-k 与 top-p 同时生效时，JOINT 与 CHAIN 保留集必须不同，
        // 否则该 case 无法捕获"验证侧 JOINT 分支退化为 CHAIN"的回归
        if (verifyK < (int)c.p.size() && c.topP < 1.0f) {
            auto jointSet = [&c]() {
                std::vector<int> s(c.p.size(),0);
                for (size_t y=0;y<c.p.size();++y) {
                    double cnt=0,sm=0;
                    for (size_t z=0;z<c.p.size();++z) if (c.p[z]>c.p[y]) {cnt++;sm+=c.p[z];}
                    if (cnt < c.verifyK && sm < c.topP) s[y]=1;
                }
                return s;
            };
            auto chainSet = [&c]() {
                std::vector<double> sorted(c.p.begin(),c.p.end());
                std::sort(sorted.rbegin(),sorted.rend());
                double pivot = sorted[c.verifyK-1];
                std::vector<double> q(c.p.size());
                double s=0;
                for (size_t j=0;j<q.size();++j) { if (c.p[j]>=pivot) q[j]=c.p[j]; s+=q[j]; }
                for (auto &x:q) x/=s;
                std::vector<int> r(q.size(),0);
                for (size_t y=0;y<q.size();++y) {
                    double sm=0;
                    for (size_t z=0;z<q.size();++z) if (q[z]>q[y]) sm+=q[z];
                    if (sm < c.topP) r[y]=1;
                }
                return r;
            };
            require(jointSet() != chainSet(),
                "case cannot distinguish joint from chain filter order");
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
                seen[id]=true; sum+=probability;
                expectedDraft[id]+=probability;draftVariance[id]+=probability*(1.0-probability);
                inSupport |= id==drafts[i];
            }
            require(inSupport,"draft is outside reported q support");
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
        // JOINT/CHAIN 二元接收率测试：草稿来自 JOINT 过滤（保留 top-K=20），
        // 因此 draft ∈ {0..19}。当 JOINT+CHAIN 退化时，CHAIN 在 top-K 内
        // 重归一化后做 nucleus，会切掉 JOINT-only 尾部（典型 skewed_128_k20_p95
        // 的 {18,19}）。此时：
        //   - jointFilterOrder=true  下这些 token 的目标分布 p_J(d) > 0，
        //     u·q(d) < p_J(d) ≈ 1（p_J 与 q 来自同一过滤），接收概率 ≈ 1；
        //   - jointFilterOrder=false 下这些 token 的目标分布 p_C(d) = 0，
        //     u·q(d) < 0 恒假，接收概率 = 0。
        // 所以「draft ∈ JOINT-only 尾部」这一组在两种过滤下的接收数差异
        // 接近 ±N，是稳定可分的二元信号（Fisher/Binomial N≈100 即 p<1e-30），
        // 比分布 Z-test 灵敏几个数量级——后者要 N≈30K。
        // 仅对 verifyK<支撑集 && topP<1 的 case 启用（否则 JOINT=CHAIN）。
        int joBndAccept = 0, joBndTotal = 0, chBndAccept = 0;
        if (verifyK < (int)c.p.size() && c.topP < 1.0f) {
            // JOINT 保留集 = {rank<verifyK} ∩ {cumsum<topP}；其外的尾部 = JOINT-only
            auto isJointTail = [&](int t) {
                if (t < 0 || t >= (int)c.p.size()) return false;
                // 排名 t 的"更大概率"个数 = rank
                int rank = 0;
                for (int z = 0; z < (int)c.p.size(); ++z)
                    if (c.p[z] > c.p[t]) ++rank;
                if (rank >= verifyK) return false;  // 已在 top-K 之外，永远拒
                double sm = 0.0;
                for (int z = 0; z < (int)c.p.size(); ++z)
                    if (c.p[z] > c.p[t]) sm += c.p[z];
                return sm >= c.topP;  // 在 top-K 内但在 nucleus 外 → JOINT 收、CHAIN 不收
            };
            // 链式组已在 line 142-145 调用过 accepted_chain/output_chain；这里直接用。
            for (int i = 0; i < samples; ++i) {
                if (!isJointTail(drafts[i])) continue;
                ++joBndTotal;
                if (accepted[i]) ++joBndAccept;
                require(accepted_chain[i] == 0,
                    "joint-tail draft was accepted under CHAIN filter (expected 0); "
                    "this signals jointFilterOrder=false is treating tail as in-support");
                if (accepted_chain[i]) ++chBndAccept;
            }
            // 二元信号断言：CHAIN 必须把 JOINT-only 尾部的所有 draft 拒掉
            // （一个都不收是 CHAIN 路径的硬约束；JOINT 全收是它的硬约束）。
            // 这两道断言都来自过滤集合的定义，是结构性保证而非统计估计。
            require(chBndAccept == 0,
                "CHAIN filter accepted a JOINT-only-tail draft; "
                "jointFilterOrder=false path is inconsistent with nucleus boundary");
            require(joBndAccept == joBndTotal,
                "JOINT filter rejected a JOINT-only-tail draft; "
                "jointFilterOrder=true path is wrong (should accept all in-keep-set drafts)");
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
