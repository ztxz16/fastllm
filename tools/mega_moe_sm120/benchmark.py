"""Build-only comparison of native EP graph recipes; never imported by inference."""
import statistics
import threading

import torch
from cuda.bindings import driver as cuda


def check(result):
    if int(result[0]):
        raise RuntimeError('EP benchmark CUDA call failed: ' + str(result[0]))
    return result[1:]


def benchmark(provider, capacity):
    """Time repeated child graphs, including communication, on all EP ranks.

    Balanced and skewed routes use deterministic nonzero inputs. The enclosing
    graph keeps Python launch gaps out of small-batch comparisons. Its outputs
    also validate that repeated graph replay advances the communication epochs.
    """
    ranks = len(provider.devices)
    barrier = threading.Barrier(ranks)
    repetitions = 8
    results = {}
    for pattern in ('balanced', 'skewed'):
        def run(rank):
            with torch.cuda.device(provider.devices[rank]):
                x, ids, scores, stream = provider.inputs[rank][capacity]
                generator = torch.Generator(device=x.device).manual_seed(1731 + rank)
                with torch.cuda.stream(stream):
                    x.copy_(torch.randn(x.shape, generator=generator, device=x.device) * .2)
                    logits = torch.rand((capacity, provider.experts), generator=generator, device=x.device)
                    if pattern == 'skewed':
                        logits[:max(1, capacity // 4), :provider.experts // ranks] += .25
                    ids.copy_(logits.topk(provider.topk, dim=1).indices)
                    scores.fill_(1.0 / provider.topk)
                stream.synchronize()
                child = cuda.CUgraph(provider.graphs[rank][0, capacity].raw_cuda_graph())
                graph, = check(cuda.cuGraphCreate(0))
                executable = None
                try:
                    previous = []
                    # Green-context graph launches cannot be stream-captured;
                    # compose explicit child nodes, as the native loader does.
                    for _ in range(repetitions):
                        node, = check(cuda.cuGraphAddChildGraphNode(graph, previous, len(previous), child))
                        previous = [node]
                    executable, = check(cuda.cuGraphInstantiate(graph, 0))
                    times = []
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    for sample in range(4):
                        barrier.wait(timeout=300)
                        with torch.cuda.stream(stream):
                            begin.record(stream)
                            check(cuda.cuGraphLaunch(executable, cuda.CUstream(stream.cuda_stream)))
                            end.record(stream)
                        end.synchronize()
                        if sample:
                            times.append(begin.elapsed_time(end) / repetitions)
                finally:
                    if executable is not None:check(cuda.cuGraphExecDestroy(executable))
                    check(cuda.cuGraphDestroy(graph))
                output = provider.workspaces[rank][capacity].output.float().cpu()
                if not torch.isfinite(output).all() or output.norm() == 0:
                    raise RuntimeError('Invalid EP benchmark output')
                return times, output

        def guarded(rank):
            try:
                return run(rank)
            except BaseException:
                barrier.abort()
                raise

        measurements = list(provider.pool.map(guarded, range(ranks)))
        # The slowest EP rank determines completion of a collective round.
        milliseconds = statistics.median(max(v[0][sample] for v in measurements) for sample in range(3))
        results[pattern] = dict(milliseconds=milliseconds, outputs=[v[1] for v in measurements])
    return results


def select_candidate(baseline, candidate, exact=False):
    """Select a faster recipe; data-movement changes must preserve exact output."""
    comparison = {}
    for pattern in baseline:
        errors = []
        bitwise_equal = True
        for expected, actual in zip(baseline[pattern]['outputs'], candidate[pattern]['outputs']):
            equal = torch.equal(expected, actual)
            bitwise_equal = bitwise_equal and equal
            # Exact data-movement candidates need no second scan for an L2 norm.
            errors.append(0.0 if equal else float((actual - expected).norm() / expected.norm().clamp_min(1e-12)))
        comparison[pattern] = dict(baseline_ms=baseline[pattern]['milliseconds'],
                                   candidate_ms=candidate[pattern]['milliseconds'],
                                   max_relative_l2=max(errors), bitwise_equal=bitwise_equal)
    valid = all(v['bitwise_equal'] if exact else v['max_relative_l2'] <= .005 for v in comparison.values())
    no_regression = all(v['candidate_ms'] <= v['baseline_ms'] * 1.05 for v in comparison.values())
    gain = sum(v['candidate_ms'] for v in comparison.values()) <= .95 * sum(v['baseline_ms'] for v in comparison.values())
    return valid and no_regression and gain, comparison
