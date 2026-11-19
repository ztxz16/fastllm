"""MoE 批次不变候选的启动证据检查；不把配置启用当作 GPU 验收通过。"""

BACKEND = 'cutlass_sm70_moe_experts_v1'
CONFIG = [1, 64, 128, 32, 32, 64, 32, 8, 8, 4, 2, 1]


def validate_ready(ready, requested):
    """/** 校验原生加载回执及所有 worker 配置；缺失或静默回退均拒绝启动。 */"""
    if type(requested) is not bool:
        raise ValueError('batch_invariant must be boolean')
    mode = ready.get('batch_invariant')
    if mode is None and not requested:
        return  # 关闭模式保持与旧二进制的兼容。
    if not isinstance(mode, dict) or mode.get('requested') is not requested or mode.get('effective') is not requested:
        raise RuntimeError('batch-invariant requested/effective mode mismatch')
    expected = CONFIG if requested else []
    if (mode.get('backend') != (BACKEND if requested else 'legacy') or
            mode.get('config') != expected or mode.get('scope') != 'moe_experts'):
        raise RuntimeError('batch-invariant backend/configuration mismatch')
    workers = ready.get('workers', [])
    if not workers or any(w.get('batch_invariant') is not requested or
                          w.get('expert_gemm_config') != expected for w in workers):
        raise RuntimeError('batch-invariant worker configuration mismatch')
