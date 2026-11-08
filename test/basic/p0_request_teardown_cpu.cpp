#include "models/basellm.h"
#include "utils/prefix_cache_logging.h"
#include <iostream>
#include <stdexcept>

using namespace fastllm;
static void Need(bool ok, const char *name) {
    std::cout << "{\"kind\":\"teardown_case\",\"name\":\"" << name
        << "\",\"pass\":" << (ok ? "true" : "false") << "}\n";
    if (!ok) throw std::runtime_error(name);
}
class TeardownModel : public basellm {
public:
    using basellm::ShutdownRuntime;
    int removed = 0;
    void OnResponseContextRemoved(ResponseContext *) override { ++removed; }
    std::string MakeInput(const std::string &, int, const std::string &) override { return {}; }
    std::string MakeHistory(const std::string &, int, const std::string &, const std::string &) override { return {}; }
};
int main() {
    ResponseContextDict dictionary;
    const int handle = dictionary.CreateHandle();
    ResponseContext *context = dictionary.GetHandle(handle);
    const auto ledger = context->resourceLedger;
    const uint64_t oldGeneration = context->nativeRequestGeneration;
    Need(ledger && oldGeneration != 0, "real_create_captures_ledger_and_generation");
    RegisterFastllmPublicRequestId(handle, "cpu-teardown-old");
    auto reader = RequestResourceHold::Reader(ledger, 701);
    auto pending = RequestResourceHold::Pending(ledger, oldGeneration);
    dictionary.RemoveHandle(handle);
    auto state = ledger->Snapshot();
    Need(!dictionary.GetHandle(handle) && state.transientOwners == 2 &&
         state.temporaryReaders == 1 && state.pendingOperations == 1,
         "delete_does_not_forge_zero_for_external_holders");
    dictionary.RemoveHandle(handle);
    Need(ledger->Snapshot().transientOwners == 2,
         "duplicate_remove_does_not_double_release_context");
    reader.Reset(); pending.Reset();
    Need(ledger->Snapshot().transientOwners == 0 &&
         !ledger->Snapshot().consistencyError, "external_holders_release_after_real_delete");

    const int reused = dictionary.CreateHandle();
    context = dictionary.GetHandle(reused);
    const auto nextLedger = context->resourceLedger;
    const auto nextGeneration = context->nativeRequestGeneration;
    Need(reused == handle && nextGeneration > oldGeneration,
         "reused_native_handle_has_new_generation");
    RegisterFastllmPublicRequestId(reused, "cpu-teardown-new");
    EndFastllmNativeRequest(reused, oldGeneration);
    Need(ReadRequestStatusGeneration(reused) == nextGeneration &&
         ReadRequestStatusIdentity(reused).publicId == "cpu-teardown-new",
         "stale_close_cannot_close_reused_generation");
    dictionary.RemoveHandle(reused);
    Need(nextLedger->Snapshot().transientOwners == 0 &&
         nextLedger->Snapshot().liveGenerations.empty() &&
         !nextLedger->Snapshot().consistencyError,
         "real_post_member_destructor_zero");
    {
        TeardownModel model;
        const int shutdownHandle = model.responseContextDict.CreateHandle();
        const auto shutdownLedger = model.responseContextDict.GetHandle(shutdownHandle)->resourceLedger;
        model.ShutdownRuntime();
        Need(model.removed == 1 && model.responseContextDict.dicts.empty() &&
             shutdownLedger->Snapshot().transientOwners == 0,
             "shutdown_routes_through_native_terminal_path");
        model.ShutdownRuntime();
        Need(model.removed == 1, "shutdown_is_idempotent");
    }
    {
        TeardownModel model;
        const int handle = model.responseContextDict.CreateHandle();
        ResponseContext *context = model.responseContextDict.GetHandle(handle);
        const auto publicationLedger = context->resourceLedger;
        PagedPublishJob publication;
        publication.requestPending = RequestResourceHold::Pending(
            publicationLedger, context->nativeRequestGeneration);
        model.EnqueuePagedPublish(std::move(publication));
        model.FlushPagedPublish();
        Need(publicationLedger->Snapshot().pendingOperations == 0 &&
             publicationLedger->Snapshot().transientOwners == 1,
             "real_publication_terminal_precedes_flush_return");
        model.RemoveResponseContext(handle);
        Need(publicationLedger->Snapshot().transientOwners == 0,
             "real_publication_owner_does_not_survive_context_terminal");
        model.StopPagedPublish();
    }
    std::cout << "{\"kind\":\"summary\",\"scope\":\"actual_ResponseContextDict_CPU_only\",\"runtime\":false}\n";
    return 0;
}
