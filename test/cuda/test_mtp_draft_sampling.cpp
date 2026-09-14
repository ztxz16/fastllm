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
struct Case { const char *name; std::vector<float> p, q; int draftK; float temperature=1, topP=1; };
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
        std::vector<int> topK(samples*2,c.p.size());
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
        for(double &p:expectedP)p/=pSum;
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
        require(FastllmCudaDFlashRejectionSampling(dTarget,temperatures.data(),topK.data(),topP.data(),
            drafts.data(),ids.data(),probabilities.data(),output.data(),accepted.data(),samples,1,c.draftK,vocab),"verification failed");
        int acceptedTotal=0;
        for (int i=0;i<samples;++i) {
            require(accepted[i]>=0&&accepted[i]<=1,"invalid acceptance count");
            require(output[2*i]>=0&&output[2*i]<vocab,"invalid verified token");
            ++verifiedCounts[output[2*i]];
            if(accepted[i]) { require(output[2*i]==drafts[i],"accepted token differs from draft"); ++acceptedTotal; }
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
                  << ",\"draft_max_z\":" << draftMaxZ << ",\"ordinary_max_z\":" << ordinaryMaxZ << ",\"pass\":" << (ok?"true":"false")
                  << ",\"expected\":"; array(expectedP);
        std::cout << ",\"draft_counts\":"; array(draftCounts);
        std::cout << ",\"verified_counts\":"; array(verifiedCounts);
        std::cout << ",\"ordinary_counts\":"; array(ordinaryCounts);
        std::cout << "}\n" << std::flush;
        check(cudaFree(dDraft)); check(cudaFree(dTarget));
    }
    return all_ok?0:2;
}
