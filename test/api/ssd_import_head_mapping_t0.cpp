// Independent global-head oracle for SSD import assembly.
// Production baseline may contain the historical count-only fast path; compare
// candidate bytes with head identities rather than merely with that baseline.
#define main LegacyAssemblyMain
#include "assembly_t0.cpp"
#undef main

static std::vector<std::string> HeadOracle(const TestCase &tc) {
    std::vector<std::string> outputs;
    for (const auto &group : tc.bundles) {
        using Key = std::tuple<int,int,int>;
        std::map<Key,std::string> heads;
        int tokens = -1, dim = -1;
        size_t unit = 0;
        for (const auto &bundle : group.second) {
            const auto &bytes = tc.bytes.at(size_t(bundle["chunks"]["index"].int_value()));
            for (const auto &entry : bundle["segments"].array_items()) {
                const int count = entry["tokens"].int_value();
                const int nheads = entry["heads"].int_value();
                const int edim = entry["dim"].int_value();
                const size_t eunit = entry["dtype"].int_value() == 10 ? 1 : 2;
                if (tokens < 0) { tokens = count; dim = edim; unit = eunit; }
                Check(count == tokens && edim == dim && eunit == unit,"oracle_layout");
                std::vector<std::pair<int,int>> ranges;
                for (const auto &r : entry["ranges"].array_items())
                    ranges.emplace_back(r[0].int_value(),r[1].int_value());
                if (ranges.empty()) ranges.emplace_back(0,nheads);
                int local = 0;
                const size_t row = size_t(dim)*unit, offset = size_t(entry["offset"].number_value());
                for (auto range : ranges) for (int global = range.first; global < range.second; ++global, ++local) {
                    std::string column(size_t(tokens)*row,'\0');
                    for (int token = 0; token < tokens; ++token) {
                        const size_t source = offset+(size_t(token)*nheads+local)*row;
                        Check(source <= bytes.size() && row <= bytes.size()-source,"oracle_bounds");
                        column.replace(size_t(token)*row,row,bytes.data()+source,row);
                    }
                    const Key key(entry["layer"].int_value(),entry["side"].int_value(),global);
                    const auto found = heads.find(key);
                    if (found == heads.end()) heads.emplace(key,std::move(column));
                    else Check(found->second == column,"oracle_replicated_conflict");
                }
                Check(local == nheads,"oracle_range_heads");
            }
        }
        for (const auto &target : tc.pools) {
            const auto &pool = *target.second;
            const int nheads = pool.pool->dims[2];
            const size_t row = size_t(dim)*unit;
            std::string out(size_t(tokens)*nheads*row,'\0');
            int local = 0;
            for (auto range : pool.ranges) for (int global = range.first; global < range.second; ++global,++local) {
                const auto found = heads.find({std::get<0>(target.first),std::get<1>(target.first),global});
                Check(found != heads.end(),"oracle_missing_head");
                for (int token = 0; token < tokens; ++token)
                    out.replace((size_t(token)*nheads+local)*row,row,found->second.data()+size_t(token)*row,row);
            }
            Check(local == nheads,"oracle_target_heads");
            outputs.push_back(std::move(out));
        }
    }
    return outputs;
}

static void Verify(TestCase &tc,const char *name,const char *expected = "") {
    const auto candidate = RunCandidate(tc,true);
    Check(candidate.error == expected,(std::string("candidate_error:")+name+":"+candidate.error).c_str());
    if (!*expected) Check(candidate.outputs == HeadOracle(tc),(std::string("oracle_output:")+name).c_str());
    std::printf("control=%s result=PASS error=%s memcpy_bytes=%llu compare_bytes=%llu\n",
        name,candidate.error.c_str(),(unsigned long long)candidate.memcpy_bytes,(unsigned long long)candidate.compare_bytes);
}

static TestCase DistinctHeads(bool replicate,bool conflict,bool fp16,bool emptyRanges = false) {
    TestCase tc;tc.managers.reserve(2);tc.length=8;
    const int tokens=8,dim=16;const size_t unit=fp16?2:1, bytes=size_t(tokens)*dim*unit;
    std::string data(bytes*2,'\0');
    for (size_t i=0;i<bytes;++i) {
        data[i]=char((i*7+13)%251);
        data[bytes+i]=replicate?data[i]:char((i*11+93)%251);
    }
    if (conflict) data.back() ^= 1;
    tc.bytes.push_back(std::move(data));
    Json::array segments;
    for (int rank=0;rank<2;++rank) {
        const int global=replicate?0:rank;
        segments.emplace_back(Json::object{{"begin",0},{"tokens",tokens},{"heads",1},{"dim",dim},
            {"dtype",fp16?3:10},{"bytes",double(bytes)},{"offset",double(rank*bytes)},{"layer",3},
            {"side",0},{"rank",rank},{"ranges",emptyRanges?Json::array{}:Json::array{Json::array{global,global+1}}}});
        AddPool(tc,3,0,rank,1,dim,{{global,global+1}});
        tc.managers.back().unitSize=unit;
    }
    tc.bundles[0]={Json::object{{"bytes",double(bytes*2)},{"chunks",Json::object{{"index",0}}},{"segments",segments}}};
    return tc;
}

int main(int argc,char **argv) {
    try {
        Check(argc == 2,"usage: head_mapping_t0 <metadata.json>");
        auto real=FromReal(ReadJson(argv[1]),1);
        Verify(real,"real_metadata_distinct_shard_bytes");
        const auto prior=RunBase(real,true);
        Check(prior.error.empty(),"baseline_real_unexpected_error");
        const bool priorWrong=prior.outputs != HeadOracle(real);
        std::printf("baseline_real_mapping_mismatch=%d baseline_memcpy_bytes=%llu\n",
            priorWrong,(unsigned long long)prior.memcpy_bytes);
        const char *errors[]={"","ssd_kv_replicated_conflict","ssd_kv_range_heads","ssd_kv_ranges",
            "ssd_bundle_segment","ssd_target_range_heads","ssd_target_range_bounds",
            "ssd_restore_cancelled_or_timeout","ssd_bundle_coverage","ssd_kv_missing_range",
            "","ssd_kv_dtype","ssd_bundle_segment",""};
        for(int i=0;i<14;++i) {auto tc=Synthetic(i);Verify(tc,("legacy_"+std::to_string(i)).c_str(),errors[i]);}
        for(bool fp16:{false,true}) {
            auto distinct=DistinctHeads(false,false,fp16);
            Verify(distinct,fp16?"distinct_fp16":"distinct_fp8");
            if (!fp16) Check(RunBase(distinct,true).outputs != HeadOracle(distinct),
                            "baseline_distinct_bug_not_reproduced");
            auto replica=DistinctHeads(true,false,fp16);
            Verify(replica,fp16?"replica_equal_fp16":"replica_equal_fp8");
            Check(RunCandidate(replica,false).compare_bytes>0,"replica_was_not_compared");
            auto bad=DistinctHeads(true,true,fp16);
            Verify(bad,fp16?"replica_conflict_fp16":"replica_conflict_fp8","ssd_kv_replicated_conflict");
        }
        auto noRanges=DistinctHeads(true,false,false,true);Verify(noRanges,"empty_ranges_default");
        auto missing=DistinctHeads(false,false,false);
        missing.pools.begin()->second->ranges={{2,3}};
        Verify(missing,"same_size_disjoint_target","ssd_kv_missing_range");
        std::puts("PASS independent global-head mapping and existing assembly controls");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr,"FAIL head mapping: %s\n",e.what());return 1;
    }
}
