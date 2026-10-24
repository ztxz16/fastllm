"""Measured MoE curves and bounded, byte-aware route replay for offline tuning.

Predictions rank experiments, never authorize a launch. Cache replay approximates
the native admission/maintenance pipeline; its error is calibrated against fresh
whole-model timings and reported explicitly, rather than advertised as a speed.
"""
from collections import Counter, defaultdict
import heapq
import json
import math
from pathlib import Path
import statistics

GiB = 2**30
MiB = 2**20
COMPONENTS = {'disk_load', 'cpu_experts', 'numa_gpu'}


def group(event, backend=None, rows=None):
    rows = event['rows'] if rows is None else rows
    return (backend or event['backend'], event['phase'], tuple(event['format']),
            tuple(event['weight_shape']), max(0, math.ceil(math.log2(max(1, rows)))))


def features(histogram, sizes, gpu_bytes=0, read_bytes=0):
    active = sum(sizes[e] for e in histogram)
    routed = sum(sizes[e]*count for e, count in histogram.items())
    return [1., routed/GiB, max(0, active-gpu_bytes)/GiB, read_bytes/GiB]


def _validate_routes(record):
    ids, sizes = record['ids'], record['bytes']
    if (not sizes or any(not isinstance(n, (int, float)) or not math.isfinite(n) or n <= 0 for n in sizes)
            or not isinstance(record['rows'], int) or record['rows'] <= 0
            or not isinstance(record['topk'], int) or record['topk'] <= 0
            or len(ids) != record['rows']*record['topk']
            or any(not isinstance(i, int) or not 0 <= i < len(sizes) for i in ids)
            or not isinstance(record['layer'], int)
            or len(record['format']) != 4 or len(record['weight_shape']) != 2
            or any(not isinstance(n, int) for n in record['format']+record['weight_shape'])):
        raise ValueError('invalid expert route record')


def capture(path, request):
    """Read an explicitly completed native capture; retain no prompt text."""
    path = Path(path)
    if path.stat().st_size > 65*MiB:
        raise ValueError('autotune capture exceeds size bound')
    data = [json.loads(line) for line in path.read_text().splitlines()]
    if not data or data[-1] != {'kind': 'complete', 'schema': 1}:
        raise ValueError('incomplete autotune capture')
    cutoff = request['start_seconds'] + request['ttft_seconds']
    end = request['start_seconds'] + request['score_seconds'] + 1
    events, samples, components = [], [], []
    prefill_rows = Counter()
    other = {'prefill': 0., 'decode': 0.}
    for record in data[:-1]:
        if not isinstance(record, dict):
            raise ValueError('invalid calibration record')
        ms = record.get('ms', 0)
        if not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms < 0:
            raise ValueError('invalid calibration duration')
        if record.get('kind') != 'op':
            if record.get('kind') in COMPONENTS and ms > 0 and record.get('units', 0) > 0:
                components.append(record)
            continue
        if not request['start_seconds'] <= record.get('start', 0) <= end:
            continue
        phase = 'prefill' if record['start'] < cutoff else 'decode'
        if record.get('op') != 'MergeMOE':
            other[phase] += ms
            continue
        required = ('ids', 'rows', 'topk', 'bytes', 'format', 'weight_shape', 'layer')
        if any(k not in record for k in required):
            continue
        _validate_routes(record)
        ids, sizes = record['ids'], record['bytes']
        # Token polling can lag a first decode kernel by a fraction of a ms.
        # Once the complete input has been seen, subsequent calls are decode.
        # Some native GPU prefill paths bypass MergeMOE, so a *short* captured
        # prefill must not cause all later decode calls to be relabeled prefill.
        if request.get('input_tokens'):
            if prefill_rows[record['layer']] >= request['input_tokens']:
                phase = 'decode'
            if phase == 'prefill':
                prefill_rows[record['layer']] += record['rows']
        record['phase'] = phase
        events.append(record)
        histogram = Counter(ids)
        total = len(ids)
        # Native disk counters count routes. Experts in one typed layer have
        # equal payload sizes; use the route fraction for mixed records too.
        hit = min(1., record.get('gpu_hits', 0)/max(1, total))
        active = sum(sizes[e] for e in histogram)
        samples.append({'group': group(record), 'x': features(histogram, sizes, active*hit,
                         record.get('read_bytes', 0)), 'ms': ms})
    if not events:
        raise ValueError('capture contains no usable expert routes')
    return {'schema': 1, 'events': events, 'samples': samples, 'components': components,
            'other_ms': other, 'request': request,
            'prefill_rows_by_layer': dict(prefill_rows),
            'limitations': ['sampled routing workload', 'approximate cache admission',
                           'operator boundaries synchronized only during calibration']}


class HeatTier:
    """Byte capacity, decaying frequency, scan resistance, and bounded heaps.

    Disk's saturating heat/admission margin follows DiskExpertCache. NUMA cache
    uses a separate pool and frequency prior; native slot partition/async
    admission differences are covered by the prediction's uncertainty.
    """
    def __init__(self, capacity, heat, margin=0):
        self.capacity = max(0, int(capacity))
        self.heat = heat
        self.margin = margin
        self.entries = {}
        self.heap = []
        self.used = 0
        self.epoch = -1

    def _entry(self, key):
        score, epoch, last, visits = self.heat.get(key, (0, 0, 0, 0))
        return (score >> min(8, max(0, self.epoch-epoch)), last, key)

    def touch(self, key):
        if key in self.entries:
            heapq.heappush(self.heap, self._entry(key))

    def weakest(self, epoch):
        if epoch != self.epoch or len(self.heap) > 3*len(self.entries)+64:
            self.epoch = epoch
            self.heap = [self._entry(key) for key in self.entries]
            heapq.heapify(self.heap)
        while self.heap:
            old = self.heap[0]
            if old[2] in self.entries and old == self._entry(old[2]):
                return old[2]
            heapq.heappop(self.heap)
        return None

    def remove(self, key):
        self.used -= self.entries.pop(key, 0)

    def admit(self, key, size, epoch, preload=False):
        if key in self.entries or size > self.capacity:
            return []
        victims = []
        def reject():
            for previous, n in victims:
                self.entries[previous] = n
                self.used += n
                self.touch(previous)
            return None
        while self.used+size > self.capacity:
            victim = self.weakest(epoch)
            if victim is None:
                return reject()
            if not preload:
                visits = self.heat.get(key, (0, 0, 0, 0))[3]
                old_visits = self.heat.get(victim, (0, 0, 0, 0))[3]
                if ((visits < 2 and old_visits) or
                        self._entry(key)[0] <= self._entry(victim)[0]+self.margin):
                    return reject()
            # A byte-sized record can evict several smaller records. No future
            # route is observed here, and admission cannot improve this lookup.
            victims.append((victim, self.entries[victim]))
            self.remove(victim)
        self.entries[key] = size
        self.used += size
        self.touch(key)
        return victims


def fit(samples):
    """Small nonnegative least-squares fit; no numpy/runtime dependency."""
    if not samples:
        return [0., 0., 0., 0.]
    xs = [s['x'] for s in samples]
    ys = [s['ms'] for s in samples]
    # Normalize columns before coordinate descent, retaining their physical
    # units in the returned coefficients. A tiny ridge stabilizes cold-only data.
    scales = [max(max(x[j] for x in xs), 1e-9) for j in range(4)]
    xx = [[x[j]/scales[j] for j in range(4)] for x in xs]
    coefficients = [0.]*4
    residual = ys[:]
    for _ in range(32):
        for j in range(4):
            old = coefficients[j]
            norm = sum(x[j]*x[j] for x in xx) + 1e-6
            value = max(0., old+sum(x[j]*r for x, r in zip(xx, residual))/norm)
            coefficients[j] = value
            for i, x in enumerate(xx):
                residual[i] += (old-value)*x[j]
    return [value/scale for value, scale in zip(coefficients, scales)]


def distance(a, b):
    x, y = a['settings'], b['settings']
    ac = {str(k): v for k,v in (a['host_gpu_cache'] or {}).items()}
    bc = {str(k): v for k,v in (b['host_gpu_cache'] or {}).items()}
    return (abs(a['numa_layers']-b['numa_layers'])/4 +
            abs(math.log2(max(1, x['chunked_prefill_size'])/max(1, y['chunked_prefill_size']))) +
            abs(x['moe_cuda_cache']-y['moe_cuda_cache'])/(4*GiB) +
            sum(abs(ac.get(d, 0)-bc.get(d, 0))/GiB for d in set(ac) | set(bc)))


class CostModel:
    def __init__(self, layers, mapping):
        self.layers = layers
        self.mapping = mapping
        self.capture = None
        self.observations = []
        self.curves = {}
        self.samples = defaultdict(list)
        self.projections = {}
        self.events_by_chunk = {}
        self.component_samples = []
        self.disk_ms_per_byte = 0.
        self.layouts = {}
        self.raw_cache = {}

    def add(self, trial):
        calibration = trial.get('measurement', {}).get('calibration')
        if not calibration:
            if trial.get('measurement', {}).get('samples'):
                self.observations.append(trial)
            return
        path = Path(calibration['path'])
        if path.stat().st_size > 65*MiB:
            raise ValueError('oversized calibration profile')
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or value.get('schema') != 1 or not value.get('events'):
            raise ValueError('invalid calibration profile')
        # Validate before mutating the predictor: one malformed saved event must
        # not leave a partially installed layout or break later valid trials.
        for event in value['events']:
            _validate_routes(event)
            if event['phase'] not in ('prefill', 'decode') or not isinstance(event['backend'], str):
                raise ValueError('invalid calibrated phase/backend')
        for sample in value['samples']:
            if (len(sample['x']) != 4 or not math.isfinite(sample['ms']) or sample['ms'] <= 0 or
                    any(not math.isfinite(x) or x < 0 for x in sample['x'])):
                raise ValueError('invalid calibrated curve sample')
            key = sample['group']
            if (len(key) != 5 or not isinstance(key[0], str) or key[1] not in ('prefill', 'decode')
                    or len(key[2]) != 4 or len(key[3]) != 2
                    or any(not isinstance(n, int) for n in [*key[2], *key[3], key[4]])):
                raise ValueError('invalid calibrated curve key')
        components = [c for c in value.get('components', []) if c['kind'] in COMPONENTS]
        for c in components:
            if (any(not math.isfinite(c[k]) or c[k] <= 0 for k in ('units', 'ms', 'time'))
                    or any(not isinstance(n, int) for n in c.get('key', []))
                    or (c['kind'] == 'numa_gpu' and not isinstance(c.get('device'), int))):
                raise ValueError('invalid calibrated component')
        if self.capture is None:
            self.capture = value
        changed = False
        for event in value['events']:
            key = (event['backend'], event['layer'])
            layout = {k:event[k] for k in ('format','weight_shape','bytes')}
            if self.layouts.get(key) != layout:
                changed = True
                self.layouts[key] = layout
        if changed:
            self.projections.clear()
        for sample in value['samples']:
            key = sample['group']
            key = (key[0], key[1], tuple(key[2]), tuple(key[3]), key[4])
            self.samples[key].append(sample)
        for key, samples in self.samples.items():
            # Uniformly retain coverage across the capture, not just early layers.
            if len(samples) > 256:
                self.samples[key] = samples[::math.ceil(len(samples)/256)]
            self.curves[key] = fit(self.samples[key])
        self.component_samples = (self.component_samples + components)[-8192:]
        # Observed busy-time union accounts for parallel loader lanes without
        # assuming a fixed worker count or summing overlapping I/O durations.
        loads = [c for c in components if c['kind'] == 'disk_load']
        if loads:
            end = busy = 0.
            for start, stop in sorted((c['time']-c['ms']/1000, c['time']) for c in loads):
                busy += max(0., stop-max(start,end))
                end = max(end,stop)
            self.disk_ms_per_byte = busy*1000/sum(c['units'] for c in loads)
        self.observations.append(trial)
        self.raw_cache.clear()

    def component_estimate(self, key, x):
        """Missing whole-call curves use measured concurrent lanes, not bandwidth constants.

        GPU pipeline samples include H2D/compute overlap and contention with the
        CPU. Balance the remaining experts by those measured service rates; SSD
        work is another completion bound, not an unconditional serial addition.
        Whole-call curves replace this approximation as soon as they are sampled.
        """
        matching = [e for e in self.layouts.values()
                    if tuple(e['format']) == key[2] and tuple(e['weight_shape']) == key[3]]
        if not matching:
            return None
        size = statistics.median(e['bytes'][0] for e in matching)
        cpu, gpu = [], defaultdict(list)
        for sample in self.component_samples:
            k = sample.get('key', [])
            if (sample['kind'] == 'cpu_experts' and len(k) == 7 and tuple(k[:4]) == key[2]
                    and (k[5],k[4]) == key[3]):
                if abs(math.ceil(math.log2(max(1,k[-1])))-key[4]) <= 1:
                    cpu.append(sample['ms']/sample['units'])
            if sample['kind'] == 'numa_gpu' and len(k) >= 11:
                if ((k[7],k[9],k[8],k[10]) == key[2] and (k[3],k[2]) == key[3]
                        and (len(k)<12 or abs(math.ceil(math.log2(max(1,k[-1])))-key[4]) <= 1)):
                    gpu[sample['device']].append(sample['ms']/sample['units'])
        active = x[2]*GiB/max(1,size)
        routes = x[1]*GiB/max(1,size)
        rates = sum(1/statistics.median(values) for values in gpu.values())
        if cpu and active:
            rates += 1/(statistics.median(cpu)*max(1,routes/active))
        if not rates:
            return None
        compute = active/rates
        io = x[3]*GiB*self.disk_ms_per_byte
        return max(compute, io)

    def _events(self, chunk):
        if chunk in self.events_by_chunk:
            return self.events_by_chunk[chunk]
        prefill, decode = defaultdict(list), []
        templates = {}
        for e in self.capture['events']:
            layer = e['layer']
            if not 0 <= layer < self.layers:
                continue  # Draft/auxiliary calls remain in the measured residual.
            templates[layer] = e
            if e['phase'] == 'prefill':
                prefill[layer].extend(e['ids'])
            else:
                decode.append((e, Counter(e['ids'])))
        events = []
        length = max((len(ids)//templates[layer]['topk'] for layer, ids in prefill.items()), default=0)
        for offset in range(0, length, chunk):
            for layer in sorted(prefill):
                e = dict(templates[layer], phase='prefill')
                ids = prefill[layer][offset*e['topk']:(offset+chunk)*e['topk']]
                if not ids:
                    continue
                e['rows'] = len(ids)//e['topk']
                events.append((e, Counter(ids)))
        events += decode
        self.events_by_chunk[chunk] = events
        return events

    def project(self, candidate):
        """Replay actual accesses for each budget; cache the inexpensive features."""
        if candidate['id'] in self.projections:
            return self.projections[candidate['id']]
        from .hybrid_auto import layer_device
        config = candidate['settings']
        heat = {}
        cpu_capacity = config['moe_cpu_cache']
        ram = HeatTier(cpu_capacity-(64*MiB if cpu_capacity >= GiB else 0), heat)
        disk_gpu = {int(d): HeatTier(config['moe_cuda_cache'], heat, 16) for d in self.mapping}
        # Pure CPU/NUMA plans use the scalar cache budget. Mixed plans pass an
        # explicit per-device NUMA pool, independent of the disk scalar pool.
        host_cache = candidate['host_gpu_cache']
        if host_cache is None:
            host_cache = {d: config['moe_cuda_cache'] if config['moe_device'] != 'disk' else 0
                          for d in self.mapping}
        host_gpu = {int(d): HeatTier(host_cache.get(d, host_cache.get(str(d), 0)), heat, 1)
                    for d in self.mapping}
        first_disk = candidate['gpu_layers']+candidate['numa_layers']
        sizes = {e['layer']: e['bytes'] for e in self.capture['events'] if 0 <= e['layer'] < self.layers}
        # PreloadDiskMoeCache spreads each expert column across disk layers.
        # Respect payload capacity/reserve; native alignment remains approximate.
        disk_layers = sorted(layer for layer in sizes if layer >= first_disk)
        for expert in range(max((len(sizes[layer]) for layer in disk_layers), default=0)):
            for layer in disk_layers:
                if expert < len(sizes[layer]):
                    size = sizes[layer][expert]
                    if ram.used+size <= ram.capacity:
                        ram.admit((layer, expert), size, 0, preload=True)
        sums = defaultdict(lambda: [0., 0., 0., 0.])
        traffic = {'read_bytes': 0, 'gpu_hit_bytes': 0, 'active_bytes': 0}
        clock = 0
        events = self._events(config['chunked_prefill_size'])
        # Prime with the captured prefill once; measured requests follow native
        # warmup and earlier requests. Cold-start traffic remains a separate fact.
        for prime in (True, False):
            for event, histogram in events:
                if prime and event['phase'] != 'prefill':
                    continue
                layer = event['layer']
                backend = ('cuda' if layer < candidate['gpu_layers'] else
                           'numa' if layer < first_disk else 'disk')
                # NUMA registration may repack GGUF into an R4 layout. Match
                # the measured execution format for this layer/backend instead
                # of treating file layout and CPU layout as identical kernels.
                event = dict(event, **self.layouts.get((backend,layer), {}))
                device = layer_device(layer, self.layers, self.mapping)
                tier = host_gpu[device] if backend == 'numa' else disk_gpu[device]
                gpu_bytes = read_bytes = 0
                for expert, routes in sorted(histogram.items()):
                    key, size = (layer, expert), event['bytes'][expert]
                    clock += 1
                    epoch = clock//4096
                    old, old_epoch, _, visits = heat.get(key, (0, epoch, 0, 0))
                    heat[key] = (min(255, (old >> min(8, epoch-old_epoch))+
                                 max(1, min(8, math.ceil(8*routes/event['rows'])))), epoch, clock, min(2, visits+1))
                    ram.touch(key); tier.touch(key)
                    if backend == 'cuda' or key in tier.entries:
                        gpu_bytes += size
                        continue
                    if backend == 'disk' and key not in ram.entries:
                        read_bytes += size
                    victims = tier.admit(key, size, epoch)
                    if key in tier.entries:
                        if backend == 'disk':
                            ram.remove(key)
                            for victim, n in victims or []:
                                ram.admit(victim, n, epoch)
                    elif backend == 'disk':
                        ram.admit(key, size, epoch)
                if not prime:
                    x = features(histogram, event['bytes'], gpu_bytes, read_bytes)
                    key = group(event, backend)
                    sums[key] = [a+b for a, b in zip(sums[key], x)]
                    traffic['read_bytes'] += read_bytes
                    traffic['gpu_hit_bytes'] += gpu_bytes
                    traffic['active_bytes'] += sum(event['bytes'][e] for e in histogram)
        result = dict(sums), traffic
        self.projections[candidate['id']] = result
        return result

    def raw(self, candidate):
        if candidate['id'] in self.raw_cache:
            return self.raw_cache[candidate['id']]
        sums, traffic = self.project(candidate)
        totals = {'prefill': 0., 'decode': 0.}
        missing = total_calls = 0
        for key, x in sums.items():
            total_calls += x[0]
            coef = self.curves.get(key)
            if coef is None:
                missing += x[0]
                component = self.component_estimate(key, x)
                if component is not None:
                    totals[key[1]] += component
                    continue
                choices = [k for k in self.curves if k[1:4] == key[1:4]]
                if not choices:
                    choices = [k for k in self.curves if k[1] == key[1]]
                if not choices:
                    continue
                near = min(choices, key=lambda k: (k[0] != key[0], k[2:4] != key[2:4], abs(k[4]-key[4])))
                coef = self.curves[near]
            value = sum(a*b for a, b in zip(x, coef))
            totals[key[1]] += max(0., value)
        for phase in totals:
            totals[phase] += self.capture['other_ms'][phase]
        covered = len({e['layer'] for e in self.capture['events'] if 0 <= e['layer'] < self.layers})
        input_rows = self.capture.get('request',{}).get('input_tokens',0)
        coverage = self.capture.get('prefill_rows_by_layer',{})
        incomplete = (sum(max(0,1-coverage.get(str(l),coverage.get(l,0))/input_rows)
                          for l in range(self.layers))/self.layers) if input_rows and coverage else 0
        result = totals, max(missing/max(1,total_calls), 1-covered/self.layers, incomplete), traffic
        self.raw_cache[candidate['id']] = result
        return result

    def predict(self, candidate):
        if not self.capture or not self.observations:
            return {'seconds': candidate['estimated_seconds'], 'uncertainty': 1., 'source': 'uncalibrated_prior'}
        raw, missing, traffic = self.raw(candidate)
        neighbors = sorted(self.observations, key=lambda t: distance(candidate, t))[:3]
        corrected = {'prefill': [], 'decode': []}
        for observed in neighbors:
            reference, _, _ = self.raw(observed)
            samples = observed['measurement']['samples']
            weight = 1/(1+distance(candidate, observed))**2
            measured = {'prefill': statistics.mean(s['ttft_seconds'] for s in samples),
                        'decode': statistics.mean(s['decode_seconds_per_token'] for s in samples)*
                                  (512-statistics.mean(s.get('first_batch_tokens', 1) for s in samples))}
            for phase in corrected:
                corrected[phase].append((measured[phase]*max(.05, raw[phase]/max(1e-6, reference[phase])), weight))
        phase_seconds = {p: sum(v*w for v,w in values)/sum(w for _,w in values)
                         for p, values in corrected.items()}
        spread = max((max(s['score_seconds'] for s in t['measurement']['samples'])-
                      min(s['score_seconds'] for s in t['measurement']['samples']))/
                     t['measurement']['score_seconds'] for t in neighbors)
        errors = [abs(t['measurement']['score_seconds']/t['prediction']['seconds']-1)
                  for t in self.observations if t.get('prediction', {}).get('source') == 'measured_curves']
        uncertainty = min(1., .1+missing*.5+distance(candidate, neighbors[0])*.025+
                          max(spread, statistics.median(errors) if errors else 0))
        return {'seconds': sum(phase_seconds.values()), 'phases': phase_seconds,
                'uncertainty': uncertainty, 'missing_curve_fraction': missing,
                'source': 'measured_curves', 'traffic': traffic}


def select_next(pool, trials, model, history):
    tried = {t['id'] for t in trials}
    remaining = [p for p in pool if p['id'] not in tried]
    ranked = []
    for item in pool:
        prediction = model.predict(item)
        # Historical speed is only an ordering hint; a new run is mandatory.
        if prediction['source'] == 'uncalibrated_prior' and item['history_key'] in history:
            prediction['seconds'] = history[item['history_key']]
        ranked.append({'id': item['id'], **prediction})
    by_id = {p['id']: p for p in ranked}
    if not remaining:
        return None, ranked
    if not trials:
        chosen = pool[0]
    elif not any(t.get('numa_layers', 0) and t.get('measurement', {}).get('calibration') for t in trials):
        mixed = [p for p in remaining if p['numa_layers']]
        # Cover the other end of the memory trade-off, rather than discarding
        # it on an uncalibrated CPU/disk bandwidth guess.
        chosen = min(mixed, key=lambda p: (-p['numa_layers'],
                     abs(math.log2(max(1,p['settings']['chunked_prefill_size'])/2048)),
                     sum((p['host_gpu_cache'] or {}).values()))) if mixed else remaining[0]
    else:
        chosen = min(remaining, key=lambda p: by_id[p['id']]['seconds']*
                     (1-.5*by_id[p['id']]['uncertainty']))
        # Alternate exploration with a neighboring cut around the measured best.
        successful = [t for t in trials if 'measurement' in t]
        if successful and len(trials)%2 == 0:
            best = min(successful, key=lambda t: t['measurement']['score_seconds'])
            nearby = [p for p in remaining if abs(p['numa_layers']-best['numa_layers']) in (1,2)
                      and p['settings']['chunked_prefill_size'] == best['settings']['chunked_prefill_size']
                      and p['host_gpu_cache'] == best['host_gpu_cache']]
            if nearby:
                chosen = min(nearby, key=lambda p: by_id[p['id']]['seconds'])
    return chosen, ranked
