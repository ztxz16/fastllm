"""校验原生执行证据；开启 Graph 后不接受旧产物、静默回退或资源泄漏。"""


def validate_graph(states, world, enabled, *, clean=True):
    if not isinstance(states, list) or len(states) != world:
        raise RuntimeError('missing CUDA Graph diagnostics; rebuild native binaries')
    counters = ('captures', 'replays', 'warmups', 'eager_decodes')
    for state in states:
        if (not isinstance(state, dict) or state.get('enabled') is not enabled or
                type(state.get('active')) is not bool or
                any(type(state.get(key)) is not int or state[key] < 0
                    for key in (*counters, 'workspace_bytes'))):
            raise RuntimeError('invalid CUDA Graph diagnostics')
        if (state['captures'] != state['warmups'] or state['replays'] < state['captures'] or
                state['active'] != bool(state['workspace_bytes']) or
                state['workspace_bytes'] > 256 * 1024 * 1024 or
                (state['active'] and not state['captures']) or
                (clean and (state['active'] or state['workspace_bytes'])) or
                (enabled and state['eager_decodes']) or
                (not enabled and any(state[key] for key in ('captures', 'replays', 'warmups', 'active')))):
            raise RuntimeError('CUDA Graph fallback, inconsistent counters or leaked workspace')
        if any(state[key] != states[0][key] for key in (*counters, 'active')):
            raise RuntimeError('inconsistent CUDA Graph ranks')
    return states


def validate_allreduce(states, world, enabled, *, previous=None, require_progress=False):
    """验证 GPU 实际规约计数和组级 IPC；模型卸载后计数仍须单调，关闭组才释放。"""
    if not isinstance(states, list) or len(states) != world:
        raise RuntimeError('missing AllReduce diagnostics; rebuild native binaries')
    observed = []
    for state in states:
        ar = state.get('allreduce') if isinstance(state, dict) else None
        if (not isinstance(ar, dict) or ar.get('enabled') is not enabled or
                ar.get('active') is not enabled or
                any(type(ar.get(k)) is not int or ar[k] < 0
                    for k in ('custom_calls', 'nccl_calls', 'workspace_bytes')) or
                not 0 < ar['workspace_bytes'] < 3 * 1024 * 1024 or
                (enabled and (world != 2 or ar['workspace_bytes'] != 2 * 1024 * 1024 + 4)) or
                (not enabled and ar['custom_calls'])):
            raise RuntimeError('invalid AllReduce diagnostics or backend fallback')
        if observed and ar != observed[0]:
            raise RuntimeError('inconsistent AllReduce ranks')
        observed.append(ar)
    if previous is not None:
        old = validate_allreduce(previous, world, enabled)[0]
        if any(observed[0][k] < old[k] for k in ('custom_calls', 'nccl_calls')):
            raise RuntimeError('AllReduce counters regressed')
        if require_progress and observed[0]['custom_calls'] <= old['custom_calls']:
            raise RuntimeError('custom AllReduce was not executed')
    elif require_progress and not observed[0]['custom_calls']:
        raise RuntimeError('custom AllReduce was not executed')
    return observed
