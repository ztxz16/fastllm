from collections import Counter
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from fastllm_pytools import autotune_cost as cost

MiB = 2**20


def candidate(numa=0, ram=0, gpu=0, host=0, chunk=8):
    return {'id': str((numa,ram,gpu,host,chunk)), 'history_key': str((numa,ram,gpu,host,chunk)),
            'gpu_layers': 0, 'numa_layers': numa, 'host_gpu_cache': {0:host,1:host},
            'settings': {'moe_device': 'disk', 'moe_device_layers': 4-numa,
                         'moe_cpu_cache': ram, 'moe_cuda_cache': gpu, 'chunked_prefill_size': chunk},
            'estimated_seconds': 100.}


def fixture():
    events = []
    for phase in ('prefill','decode'):
        for step in range(1 if phase == 'prefill' else 4):
            for layer in range(4):
                ids = [0,1]*4 if phase == 'prefill' else [step%2]
                events.append({'kind':'op','op':'MergeMOE','layer':layer,'backend':'disk',
                               'phase':phase,'rows':len(ids),'topk':1,'ids':ids,
                               'bytes':[MiB,MiB],'format':[1,2,1,2],'weight_shape':[64,64],
                               'start':10+len(events)*.01,'ms':2.,'read_bytes':len(set(ids))*MiB})
    samples = [{'group':cost.group(e),'x':cost.features(Counter(e['ids']),e['bytes'],0,e['read_bytes']),
                'ms':e['ms']} for e in events]
    return {'schema':1,'events':events,'samples':samples,'components':[],
            'other_ms':{'prefill':10.,'decode':10.}}


class CaptureTest(unittest.TestCase):
    def test_partial_native_prefill_does_not_relabel_later_decode(self):
        events = [dict(fixture()['events'][0],rows=1,ids=[0],start=10.),
                  dict(fixture()['events'][4],start=11.)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'trace.jsonl'
            p.write_text('\n'.join(json.dumps(e) for e in events+[{'kind':'complete','schema':1}]))
            result = cost.capture(p,{'start_seconds':10.,'ttft_seconds':.5,'score_seconds':2.,'input_tokens':8})
            self.assertEqual([e['phase'] for e in result['events']],['prefill','decode'])
            self.assertEqual(result['prefill_rows_by_layer'][0],1)

    def test_completed_capture_preserves_route_order_and_phase(self):
        events = fixture()['events']
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'trace.jsonl'
            path.write_text('\n'.join(json.dumps(e) for e in events+[{'kind':'complete','schema':1}]))
            request = {'start_seconds':10.,'ttft_seconds':.04,'score_seconds':1.}
            result = cost.capture(path, request)
            self.assertEqual(result['events'][0]['ids'], [0,1]*4)
            self.assertEqual(result['events'][3]['phase'],'prefill')
            self.assertEqual(result['events'][4]['phase'],'decode')
            path.write_text(json.dumps(events[0]))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                cost.capture(path,request)

    def test_malformed_routes_and_nonfinite_durations_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'trace.jsonl'
            for change in ({'ids':[999]*8}, {'ms':float('nan')}, {'bytes':[0,MiB]}):
                event = dict(fixture()['events'][0],**change)
                p.write_text(json.dumps(event)+'\n'+json.dumps({'kind':'complete','schema':1}))
                with self.assertRaises(ValueError):
                    cost.capture(p,{'start_seconds':10.,'ttft_seconds':.1,'score_seconds':1.})


class ReplayTest(unittest.TestCase):
    def model(self):
        m = cost.CostModel(4,{0:1,1:1})
        m.capture = fixture()
        return m

    def test_byte_capacity_and_failed_multi_victim_admission_are_atomic(self):
        heat = {'a':(1,0,1,2),'b':(20,0,2,2),'c':(5,0,3,2)}
        tier = cost.HeatTier(10,heat)
        tier.admit('a',4,0,preload=True);tier.admit('b',6,0,preload=True)
        self.assertIsNone(tier.admit('c',8,0))
        self.assertEqual(tier.entries,{'a':4,'b':6})
        self.assertEqual(tier.used,10)
        self.assertEqual(tier.admit('too_large',11,0),[])

    def test_cpu_cache_changes_reads_without_future_hits(self):
        m = self.model()
        _, cold = m.project(candidate())
        _, cached = m.project(candidate(ram=8*MiB))
        self.assertGreater(cold['read_bytes'],0)
        self.assertEqual(cached['read_bytes'],0)
        self.assertEqual(cold['active_bytes'],cached['active_bytes'])

    def test_disk_and_numa_gpu_caches_are_independent(self):
        m = self.model()
        _, no_host = m.project(candidate(numa=2,gpu=MiB))
        _, host = m.project(candidate(numa=2,gpu=MiB,host=4*MiB))
        self.assertEqual(no_host['read_bytes'],host['read_bytes'])
        self.assertGreater(host['gpu_hit_bytes'],no_host['gpu_hit_bytes'])

    def test_pure_numa_scalar_cache_matches_explicit_per_device_cache(self):
        m = self.model()
        scalar = candidate(numa=4, gpu=4*MiB)
        scalar['settings']['moe_device'] = 'cpu'
        scalar['host_gpu_cache'] = None
        _, actual = m.project(scalar)
        _, explicit = m.project(candidate(numa=4, gpu=4*MiB, host=4*MiB))
        self.assertGreater(actual['gpu_hit_bytes'], 0)
        self.assertEqual(actual, explicit)

    def test_ram_preload_spreads_expert_columns_across_layers(self):
        m = self.model()
        # Only decode, so prefill priming cannot hide the preload layout.
        m.capture['events'] = [dict(e, phase='decode', rows=1, ids=[0]) for e in fixture()['events'][:4]]
        _, traffic = m.project(candidate(ram=4*MiB))
        self.assertEqual(traffic['read_bytes'], 0)

    def test_invalid_calibration_cannot_partially_update_predictor(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'calibration.json'
            value = fixture()
            value['events'][-1]['bytes'] = [0, MiB]
            p.write_text(json.dumps(value))
            m = cost.CostModel(4, {0:1,1:1})
            with self.assertRaisesRegex(ValueError, 'route'):
                m.add({'measurement':{'calibration':{'path':str(p)}}})
            self.assertIsNone(m.capture)
            self.assertEqual(m.layouts, {})
            self.assertEqual(m.curves, {})

    def test_chunk_replay_preserves_all_routes_and_last_short_chunk(self):
        m = self.model()
        events = m._events(3)
        first = [e for e,h in events if e['phase']=='prefill' and e['layer']==0]
        self.assertEqual([e['rows'] for e in first],[3,3,2])
        self.assertEqual(sum(sum(h.values()) for e,h in events if e['phase']=='prefill'),32)

    def test_calibration_uses_real_latency_and_reports_missing_backends(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'calibration.json';p.write_text(json.dumps(fixture()))
            t = candidate()
            t['measurement'] = {'calibration':{'path':str(p)},'score_seconds':10.,
                                'samples':[{'ttft_seconds':2.,'decode_seconds_per_token':8/511,
                                            'score_seconds':10.,'first_batch_tokens':1}]*2}
            m = self.model();m.add(t)
            prediction = m.predict(t)
            self.assertAlmostEqual(prediction['seconds'],10.)
            other = m.predict(candidate(numa=2))
            self.assertGreater(other['missing_curve_fraction'],0)
            self.assertGreater(other['uncertainty'],prediction['uncertainty'])
            self.assertEqual(other['source'],'measured_curves')

    def test_nonnegative_fit_recovers_a_measured_bandwidth_curve(self):
        samples = [{'x':[1.,x,0.,0.], 'ms':2.+3*x} for x in (.1,1,2,5,10)]
        c = cost.fit(samples)
        self.assertTrue(all(math.isfinite(n) and n>=0 for n in c))
        self.assertAlmostEqual(c[0]+c[1]*4,14.,delta=.05)

    def test_disk_service_rate_counts_observed_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            value=fixture()
            value['components']=[{'kind':'disk_load','units':100,'ms':2.,'time':10.},
                                 {'kind':'disk_load','units':100,'ms':2.,'time':10.001}]
            p=Path(d)/'calibration.json';p.write_text(json.dumps(value))
            m=self.model()
            m.add({'measurement':{'calibration':{'path':str(p)}}})
            self.assertAlmostEqual(m.disk_ms_per_byte,.015)

    def test_repacked_backend_uses_its_measured_format(self):
        m = self.model()
        changed = dict(m.capture['events'][0], backend='numa', format=[1,99,1,99])
        m.layouts[('numa',0)] = {k:changed[k] for k in ('format','weight_shape','bytes')}
        sums,_ = m.project(candidate(numa=1))
        keys = [k for k in sums if k[0]=='numa']
        self.assertTrue(keys)
        self.assertTrue(all(k[2]==(1,99,1,99) for k in keys))

    def test_new_whole_model_measurement_corrects_ranking_without_new_capture(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'calibration.json';p.write_text(json.dumps(fixture()))
            a=candidate()
            a['measurement']={'calibration':{'path':str(p)},'score_seconds':10.,'samples':[
                {'ttft_seconds':2.,'decode_seconds_per_token':8/511,'score_seconds':10.}]*2}
            m=self.model();m.add(a)
            b=candidate(numa=3)
            before=m.predict(b)['seconds']
            b['measurement']={'score_seconds':1.5,'samples':[{'ttft_seconds':.5,'decode_seconds_per_token':1/511,'score_seconds':1.5}]*2}
            m.add(b)
            self.assertLess(m.predict(b)['seconds'],before)


class SearchTest(unittest.TestCase):
    def test_explores_far_boundary_and_continues_beyond_three(self):
        pool = [candidate(numa=n) for n in range(4)]
        m = cost.CostModel(4,{0:1,1:1})
        selected,ranked = cost.select_next(pool,[],m,{})
        self.assertEqual(selected,pool[0]);self.assertEqual(len(ranked),4)
        pool[-1]['estimated_seconds']=10000
        selected,_ = cost.select_next(pool,[pool[0]],m,{})
        self.assertEqual(selected,pool[-1])
        selected,_ = cost.select_next(pool,pool[:3],m,{})
        self.assertEqual(selected,pool[-1])
        self.assertIsNone(cost.select_next(pool,pool,m,{})[0])

    def test_json_cache_device_keys_do_not_change_distance(self):
        a = candidate(host=MiB)
        b = json.loads(json.dumps(a))
        self.assertEqual(cost.distance(a,b),0)


if __name__ == '__main__':
    unittest.main()
