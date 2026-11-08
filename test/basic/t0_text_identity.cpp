// T0 offline check (CPU only, no GPU): export and import must pick the same page-key input for
// page_identity. Uses the real fastllm::PagedCacheManager::HashTokenPrefixPage and the real inline
// ResponseContext::PrefixCachePageKeys(); the two gate expressions below restate
// qwen3_5_persistent.inc (export ~2233/3053-3056, import ~4210-4212) and must be kept in sync with it.
#include "fastllm.h"
#include "models/basellm.h"

#include <iostream>
#include <string>
#include <vector>

using fastllm::PagedCacheManager;
using fastllm::ResponseContext;

static int failures = 0;
#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::cerr << "FAIL line " << __LINE__ << ": " #cond "\n";   \
            ++failures;                                                  \
        }                                                                \
    } while (0)

// Same chain rule as ComputePageIdentities (qwen3_5_persistent.inc ~150-165).
static std::vector<uint64_t> Identities(const std::vector<int> &tokens, int pageLen,
                                        const std::vector<std::string> *keys) {
    std::vector<uint64_t> out;
    uint64_t chain = 0;
    const size_t pages = tokens.size() / size_t(pageLen);
    for (size_t p = 0; p < pages; ++p) {
        const std::string extra = keys != nullptr ? (*keys)[p] : std::string();
        chain = PagedCacheManager::HashTokenPrefixPage(chain, tokens.data() + p * size_t(pageLen), pageLen, extra);
        out.push_back(chain);
    }
    return out;
}

// Export rule: job.runtimeUsesPageKeys = ctx->PrefixCachePageKeys() != nullptr.
static const std::vector<std::string> *ExportKeys(const ResponseContext &ctx) {
    return ctx.PrefixCachePageKeys();
}
// Current import rule (before fix C): use the vector whenever it is non-empty.
static const std::vector<std::string> *OldImportKeys(const ResponseContext &ctx) {
    return !ctx.prefixCachePageKeys.empty() ? &ctx.prefixCachePageKeys : nullptr;
}
// Fixed import rule (fix C): same accessor as export.
static const std::vector<std::string> *NewImportKeys(const ResponseContext &ctx) {
    return ctx.PrefixCachePageKeys();
}

static void Report(const char *name, const std::vector<std::string> *exp, const std::vector<std::string> *imp,
                   const std::vector<uint64_t> &a, const std::vector<uint64_t> &b) {
    std::cout << name << ": export_keys=" << (exp ? "vector" : "null") << " import_keys=" << (imp ? "vector" : "null")
              << " pages=" << a.size() << " first_export=" << std::hex << (a.empty() ? 0 : a[0])
              << " first_import=" << (b.empty() ? 0 : b[0]) << std::dec << " equal=" << (a == b) << "\n";
}

int main() {
    const int pageLen = fastllm::GetPageLen();
    std::vector<int> tokens(size_t(pageLen) * 3);
    for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = int(1000 + i);
    const std::vector<std::string> keys{"text-key-0", "text-key-1", "text-key-2"};
    std::cout << "pageLen=" << pageLen << "\n";

    // Case 1: text-only request with non-empty text fallback keys (qwen3_5_persistent.inc ~1633-1645).
    ResponseContext text;
    text.prefixCachePageKeys = keys;
    const auto expText = Identities(tokens, pageLen, ExportKeys(text));
    const auto oldText = Identities(tokens, pageLen, OldImportKeys(text));
    const auto newText = Identities(tokens, pageLen, NewImportKeys(text));
    Report("text_old", ExportKeys(text), OldImportKeys(text), expText, oldText);
    Report("text_new", ExportKeys(text), NewImportKeys(text), expText, newText);
    const bool bugReproduced = expText != oldText;
    CHECK(newText == expText);

    // Case 2: multimodal control: identity spans present -> accessor returns the vector on both sides.
    ResponseContext mm;
    mm.prefixCachePageKeys = keys;
    mm.multimodalPrefixSpans.emplace_back();
    CHECK(mm.PrefixCachePageKeys() != nullptr);
    const auto expMm = Identities(tokens, pageLen, ExportKeys(mm));
    const auto oldMm = Identities(tokens, pageLen, OldImportKeys(mm));
    const auto newMm = Identities(tokens, pageLen, NewImportKeys(mm));
    Report("mm_old", ExportKeys(mm), OldImportKeys(mm), expMm, oldMm);
    Report("mm_new", ExportKeys(mm), NewImportKeys(mm), expMm, newMm);
    CHECK(oldMm == expMm);
    CHECK(newMm == expMm);

    // Case 3: keys matter (sanity): hashing with vs without keys must differ.
    CHECK(Identities(tokens, pageLen, &keys) != Identities(tokens, pageLen, nullptr));

    std::cout << "bug_reproduced_with_old_import_rule=" << bugReproduced << "\n";
    if (failures) { std::cout << "FAIL: t0 text identity (" << failures << ")\n"; return 1; }
    std::cout << "PASS: t0 text identity (new rule matches export for text and multimodal)\n";
    return 0;
}
