"""校验原生执行证据；开启 Graph 后不接受旧产物、静默回退或资源泄漏。"""


from .qwen3_limits import MAX_PROMPT_TOKENS


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


def validate_prefix_directory(prefix, *, required=False):
    """/** 校验原生快照目录的版本、边界与唯一性；旧产物仅在未要求缓存路由时兼容。 */"""
    if not required and 'directory_version' not in prefix and 'snapshots' not in prefix:
        return None
    snapshots = prefix.get('snapshots')
    if (type(prefix.get('directory_version')) is not int or prefix['directory_version'] not in (1, 2) or
            not isinstance(snapshots, list) or len(snapshots) != prefix['entries']):
        raise RuntimeError('missing/invalid prefix directory; rebuild native generate and worker binaries')
    paged = prefix['directory_version'] == 2
    page = prefix.get('page_len', 0)
    if paged and (type(page) is not int or page <= 0):
        raise RuntimeError('invalid paged prefix page size')
    seen = set()
    for entry in snapshots:
        if not isinstance(entry, dict) or set(entry) != {'position_begin', 'token_ids'}:
            raise RuntimeError('invalid prefix directory entry')
        position, ids = entry['position_begin'], entry['token_ids']
        if (type(position) is not int or not 0 <= position <= 1048576 or not isinstance(ids, list) or
                not 1 <= len(ids) <= MAX_PROMPT_TOKENS or position + len(ids) - 1 > 1048576 or
                any(type(token) is not int or not 0 <= token <= 4294967295 for token in ids)):
            raise RuntimeError('invalid prefix directory tokens/position')
        if paged and len(ids) % page:
            raise RuntimeError('paged prefix contains a partial page')
        key = (position, tuple(ids))
        if key in seen:
            raise RuntimeError('duplicate prefix directory entry')
        seen.add(key)
    return snapshots


def validate_prefix(states, world, enabled, *, previous=None, reset=False, cleared=False, prompt_tokens=None):
    """检查 worker 实际复用长度、资源上限和跨 rank 一致性；卸载必须清零。"""
    counters = ('lookups', 'hits', 'reused_tokens', 'evictions', 'clears')
    integers = ('entries', 'device_bytes', 'last_reused_tokens', 'last_computed_tokens', *counters)
    if not isinstance(states, list) or len(states) != world:
        raise RuntimeError('missing prefix cache diagnostics; rebuild native binaries')
    observed = []
    for state in states:
        p = state.get('prefix_cache') if isinstance(state, dict) else None
        if (not isinstance(p, dict) or p.get('enabled') is not enabled or
                any(type(p.get(k)) is not int or p[k] < 0 for k in integers)):
            raise RuntimeError('invalid prefix cache diagnostics')
        if observed and any(p.get(k) != observed[0].get(k) for k in ('directory_version', 'page_len', 'max_pages')):
            raise RuntimeError('inconsistent prefix directory version/page configuration')
        paged = p.get('directory_version') == 2
        if paged:
            resources = ('page_len', 'max_pages', 'pool_bytes', 'active_pages', 'cached_pages')
            if (p.get('cache_layout') != 'paged' or any(type(p.get(k)) is not int or p[k] < 0 for k in resources) or
                    not p['page_len'] or not p['max_pages'] or p['device_bytes'] > p['pool_bytes'] or
                    p['last_reused_tokens'] % p['page_len'] or
                    (reset and any(p[k] for k in ('pool_bytes', 'active_pages', 'cached_pages'))) or
                    (cleared and p['cached_pages']) or
                    (p['lookups'] and p.get('attention_backend') not in ('native_paged', 'flashinfer_paged'))):
                raise RuntimeError('invalid FT paged prefix cache resources/backend')
        if ((not paged and (p['entries'] > 8 or p['device_bytes'] > 64 * 1024 * 1024 or
                bool(p['entries']) != bool(p['device_bytes']))) or p['hits'] > p['lookups'] or
                p['reused_tokens'] < p['hits'] or p['last_reused_tokens'] >= MAX_PROMPT_TOKENS or
                p['last_computed_tokens'] > MAX_PROMPT_TOKENS or p['last_reused_tokens'] + p['last_computed_tokens'] > MAX_PROMPT_TOKENS or
                (p['lookups'] and not p['last_computed_tokens']) or
                ((reset or not enabled) and any(p[k] for k in integers)) or
                (cleared and (p['entries'] or p['device_bytes']))):
            raise RuntimeError('prefix cache fallback, invalid counters or leaked snapshots')
        if observed and any(p[k] != observed[0][k] for k in integers if k not in (('device_bytes', 'evictions', 'clears') if paged else ('device_bytes',))):
            raise RuntimeError('inconsistent prefix cache ranks')
        if observed and not paged and p['evictions'] != observed[0]['evictions']:
            raise RuntimeError('inconsistent prefix cache ranks')
        directory = validate_prefix_directory(p)
        if observed and directory != observed[0].get('snapshots'):
            raise RuntimeError('inconsistent prefix directory ranks')
        observed.append(p)
    if previous is not None and not reset:
        prior = validate_prefix(previous, world, enabled)
        for current, earlier in zip(observed, prior):
            if any(current[k] < earlier[k] for k in counters):
                raise RuntimeError('prefix cache counters regressed')
            if cleared and current['clears'] != earlier['clears'] + int(enabled):
                raise RuntimeError('prefix cache clear counter mismatch')
        old = prior[0]
        p = observed[0]
        if 'snapshots' in old:
            validate_prefix_directory(p, required=True)
        if any(p[k] < old[k] for k in counters):
            raise RuntimeError('prefix cache counters regressed')
        if cleared:
            if (p['clears'] != old['clears'] + int(enabled) or
                    any(p[k] != old[k] for k in integers if k not in ('entries', 'device_bytes', 'clears'))):
                raise RuntimeError('prefix cache clear changed execution state')
        if enabled and prompt_tokens is not None:
            reused = p['last_reused_tokens']
            if (p['lookups'] != old['lookups'] + 1 or
                    p['last_computed_tokens'] + reused != prompt_tokens or
                    p['reused_tokens'] != old['reused_tokens'] + reused or
                    p['hits'] != old['hits'] + int(reused > 0)):
                raise RuntimeError('prefix cache did not account for this request')
    return observed
