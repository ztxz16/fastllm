"""Curated ModelScope catalog shown by the launcher download page.

The records are deliberately data-only.  The browser can present the same
catalog as cards and as a compact select, while the TUI keeps its smaller
legacy choice list.
"""


MODELSCOPE_DOWNLOAD_GROUPS = (
    {
        "id": "qwen38-27b",
        "label": "Qwen3.8 · 27B",
        "description": "稠密主模型，覆盖原始、FP8、NVFP4 与 DFlash2 草稿权重。",
        "accent": "green",
        "models": (
            {
                "preset": "qwen38-27b-bf16",
                "id": "Qwen/Qwen3.8-27B",
                "label": "Qwen3.8-27B · BF16",
                "meta": "原始质量",
                "include": "",
            },
            {
                "preset": "qwen38-27b-fp8",
                "id": "Qwen/Qwen3.8-27B-FP8",
                "label": "Qwen3.8-27B · FP8",
                "meta": "约 31 GB",
                "include": "",
            },
            {
                "preset": "qwen38-27b-nvfp4",
                "id": "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4",
                "label": "Qwen3.8-27B · QUASAR NVFP4",
                "meta": "显存友好",
                "include": "",
            },
            {
                "preset": "qwen38-27b-dflash2",
                "id": "z-lab/Qwen3.8-27B-DFlash2",
                "label": "Qwen3.8-27B · DFlash2",
                "meta": "推测解码草稿",
                "include": "",
            },
        ),
    },
    {
        "id": "qwen38-flash-gsq",
        "label": "Qwen3.8-Flash-Next · GSQ-RCO",
        "description": "选择一个量化子目录；下载器只拉选中的分片与共享 mmproj。",
        "accent": "violet",
        "models": (
            {
                "preset": "flash-gsq-q2-0",
                "id": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
                "label": "GSQ-RCO · Q2_0",
                "meta": "约 66.4 GB · 速度优先",
                "include": "Q2_0/*,mmproj-*.gguf",
            },
            {
                "preset": "flash-gsq-iq2-xs",
                "id": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
                "label": "GSQ-RCO · IQ2_XS",
                "meta": "约 68.0 GB · 体积优先",
                "include": "IQ2_XS/*,mmproj-*.gguf",
            },
            {
                "preset": "flash-gsq-iq3-xxs",
                "id": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
                "label": "GSQ-RCO · IQ3_XXS",
                "meta": "约 75.8 GB · 平衡",
                "include": "IQ3_XXS/*,mmproj-*.gguf",
            },
            {
                "preset": "flash-gsq-iq3-s",
                "id": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
                "label": "GSQ-RCO · IQ3_S",
                "meta": "约 83.6 GB · 质量优先",
                "include": "IQ3_S/*,mmproj-*.gguf",
            },
            {
                "preset": "flash-gsq-coder-iq1-m",
                "id": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF",
                "label": "GSQ-RCO Coder · IQ1_M",
                "meta": "代码场景",
                "include": "IQ1_M/*,mmproj-*.gguf",
            },
        ),
    },
    {
        "id": "deepseek-v41",
        "label": "DeepSeek · V4.1 Flash",
        "description": "大 MoE 混合推理模型，建议使用自动规划器部署。",
        "accent": "orange",
        "models": (
            {
                "preset": "deepseek-v41-flash-1017",
                "id": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
                "label": "DeepSeek V4.1 Flash · 1017",
                "meta": "约 510 GB · 混合推理",
                "include": "",
            },
        ),
    },
    {
        "id": "quick-check",
        "label": "快速验证",
        "description": "小模型和自定义入口，用于先验证环境、权限与下载链路。",
        "accent": "slate",
        "models": (
            {
                "preset": "qwen3-0.6b",
                "id": "Qwen/Qwen3-0.6B",
                "label": "Qwen3 · 0.6B",
                "meta": "快速验证",
                "include": "",
            },
        ),
    },
)
