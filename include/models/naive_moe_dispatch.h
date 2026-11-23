#pragma once
// Pure admission policy, shared by the loader and the prepared model. A
// rejected specialization always retains the ordinary TP weight layout.
namespace fastllm {
inline bool NaiveMoeGeometrySupported(int hidden, int experts, int selected, int layers,
                                     bool normalize, float scale) {
    return hidden == 4096 && experts == 256 && selected == 8 &&
        layers > 0 && layers <= 48 && normalize && scale == 1.0f;
}
enum class NaiveMoeBackend { LegacyTP, MegaEP, FlexTP, FlexEP, LegacyFlexTP };
inline NaiveMoeBackend SelectNaiveMoeBackend(int ep, bool sharedFlex, bool oldFlex,
        bool megaAvailable, bool flexAvailable, bool geometry, bool format, bool enabled) {
    if (!enabled || !geometry || !format) return NaiveMoeBackend::LegacyTP;
    if (ep > 1) {
        if (!megaAvailable) return NaiveMoeBackend::LegacyTP;
        return ep == 8 && sharedFlex && flexAvailable ? NaiveMoeBackend::FlexEP : NaiveMoeBackend::MegaEP;
    }
    if (!flexAvailable) return NaiveMoeBackend::LegacyTP;
    if (sharedFlex) return NaiveMoeBackend::FlexTP;
    return oldFlex ? NaiveMoeBackend::LegacyFlexTP : NaiveMoeBackend::LegacyTP;
}
inline bool NaiveMoeUsesEP(NaiveMoeBackend b) {
    return b == NaiveMoeBackend::MegaEP || b == NaiveMoeBackend::FlexEP;
}
inline const char *NaiveMoeBackendName(NaiveMoeBackend b) {
    switch (b) {
        case NaiveMoeBackend::MegaEP: return "mega-ep";
        case NaiveMoeBackend::FlexEP: return "flex-ep/shared-mega-weights";
        case NaiveMoeBackend::FlexTP: return "flex-tp/owned-packed";
        case NaiveMoeBackend::LegacyFlexTP: return "legacy-flex-tp/owned-packed";
        default: return "legacy-tp/original-weights";
    }
}
}
