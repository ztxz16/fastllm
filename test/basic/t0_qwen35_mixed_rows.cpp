// W4-BMIX T0: CPU-only row plan, budget, offset, and negative-control oracle.
#include "models/qwen35_mixed_rows.h"
#include <iostream>
#include <string>
#include <vector>

using namespace fastllm::qwen35_mixed_rows;
static int cases = 0, failures = 0;
static void Check(const char *name, bool actual, bool expected = true) {
    ++cases;
    if (actual != expected) ++failures;
    std::cout << "{\"kind\":\"case\",\"name\":\"" << name
              << "\",\"expected\":" << (expected ? "true" : "false")
              << ",\"actual\":" << (actual ? "true" : "false") << "}\n";
}

int main() {
    std::string error; Plan plan;
    Check("positive_decode_first", BuildPlan({1, 3, 9}, 2048, 2080, 4096, plan));
    Check("positive_order", plan.seqLens == std::vector<int>({1, 3, 9, 2048}) &&
          plan.isPrefill == std::vector<unsigned char>({0, 0, 0, 1}));
    Check("positive_offsets", plan.tokenOffsets ==
          std::vector<std::size_t>({0, 1, 4, 13, 2061}));
    Check("positive_budget", plan.decodeTokens == 13 && plan.prefillTokens == 2048);
    Check("positive_commit_rows", ValidateCommittedRows(plan, {0, 0, 0, 0}, &error));

    Plan before = plan;
    Check("empty_decode_rejected", BuildPlan({}, 2048, 2080, 4096, plan, &error), false);
    Check("invalid_keeps_plan", plan.seqLens == before.seqLens &&
          plan.tokenOffsets == before.tokenOffsets);
    Check("prefill_one_rejected", BuildPlan({1}, 1, 2048, 4096, plan, &error), false);
    Check("decode_zero_rejected", BuildPlan({0}, 2048, 2048, 4096, plan, &error), false);
    Check("decode_ten_rejected", BuildPlan({10}, 2048, 2048, 4096, plan, &error), false);
    Check("prefill_cap_rejected", BuildPlan({1}, 4097, 8192, 4096, plan, &error), false);
    Check("budget_rejected", BuildPlan({1, 9}, 2048, 2056, 4096, plan, &error), false);
    Check("returned_row_count_rejected", ValidateCommittedRows(before, {0}, &error), false);

    int mutation = 0;
    if (BuildPlan({1, 3}, 2048, 2052, 4096, plan, &error)) ++mutation;
    Check("valid_allows_mutation", mutation == 1);
    Plan rejected = plan;
    if (BuildPlan({1, 10}, 2048, 4096, 4096, rejected, &error)) ++mutation;
    Check("invalid_does_not_mutate_marker", mutation == 1);
    std::cout << "{\"kind\":\"summary\",\"cases\":" << cases
              << ",\"failures\":" << failures << "}\n";
    return failures == 0 ? 0 : 1;
}
