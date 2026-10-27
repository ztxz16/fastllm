"""Analyze saved generation text without loading FastLLM or a GPU."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def _common_prefix(sequence, a, b, limit):
    """Exact comparisons, with no probabilistic hash matching."""
    low, high = 0, limit
    while low < high:
        middle = (low + high + 1) // 2
        if sequence[a:a + middle] == sequence[b:b + middle]:
            low = middle
        else:
            high = middle - 1
    return low


def find_cycle(sequence, *, min_repetitions=3, min_span=256, max_period=16384):
    """Find a long exact cycle using repeated 32-unit anchors as candidates.

    Candidate search is bounded: up to 16 distinct occurrence patterns, the
    first/last 64 positions per pattern, and four nearby prior occurrences.
    A reported cycle is verified exactly; absence is not proof of no repetition.
    """
    size = len(sequence)
    if size < min_span:
        return None
    sequence = sequence if isinstance(sequence, str) else tuple(sequence)
    positions = collections.defaultdict(list)
    for i in range(size - 31):
        positions[sequence[i:i + 32]].append(i)
    patterns = sorted((p for p in positions.values() if len(p) >= min_repetitions),
                      key=len, reverse=True)
    reverse = sequence[::-1]
    seen_patterns, best = set(), None
    for offsets in patterns:
        pattern = tuple(p - offsets[0] for p in offsets)
        if pattern in seen_patterns:
            continue
        seen_patterns.add(pattern)
        if len(seen_patterns) > 16:
            break
        offsets = sorted(set(offsets[:64] + offsets[-64:]))
        for i, a in enumerate(offsets):
            for b in offsets[i + 1:i + 5]:
                period = b - a
                if period > max_period or period * min_repetitions > size:
                    continue
                right = _common_prefix(sequence, a, b, size - b)
                left = _common_prefix(reverse, size - a, size - b, a)
                start, end = a - left, b + right
                span = end - start
                if span < max(min_span, period * min_repetitions):
                    continue
                if best and (span, -period) <= (best['span'], -best['period']):
                    continue
                best = dict(start=start, end=end, span=span, period=period,
                            full_repetitions=span // period,
                            trailing_units=span % period)
                if span == size and period == 1:
                    return best
    return best


def analyze_text(text, tokenizer=None):
    sequence = text if tokenizer is None else tuple(tokenizer.encode(text, add_special_tokens=False).ids)
    unit = 'characters' if tokenizer is None else 'tokens'
    counts = {}
    for width in [16, 64]:
        grams = collections.Counter(sequence[i:i + width]
                                    for i in range(max(0, len(sequence) - width + 1)))
        counts[str(width)] = max(grams.values(), default=0)
    cycle = find_cycle(sequence, max_period=16384 if tokenizer is None else 4096)
    if cycle:
        begin, period = cycle['start'], cycle['period']
        block = sequence[begin:begin + period]
        cycle['excerpt'] = block[:256] if tokenizer is None else tokenizer.decode(list(block[:256]))
    return dict(unit=unit, length=len(sequence), chars=len(text),
                text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
                max_ngram_occurrences=counts, cycle=cycle)


def load_tokenizer(path):
    if path is None:
        return None
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ValueError('--tokenizer requires: python3 -m pip install tokenizers') from error
    path = Path(path)
    return Tokenizer.from_file(str(path / 'tokenizer.json' if path.is_dir() else path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path, help='Saved reasoning.txt or content.txt files')
    parser.add_argument('--tokenizer', type=Path, help='Optional local tokenizer.json or model directory')
    parser.add_argument('--output', type=Path, help='Write JSON; default is stdout')
    parser.add_argument('--fail-on-cycle', action='store_true')
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    rows = [dict(file=str(p), **analyze_text(p.read_text(encoding='utf-8'), tokenizer)) for p in args.files]
    output = json.dumps(rows, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(output, encoding='utf-8')
    else:
        print(output, end='')
    return 2 if args.fail_on_cycle and any(r['cycle'] for r in rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())
