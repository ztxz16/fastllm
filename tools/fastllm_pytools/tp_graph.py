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
