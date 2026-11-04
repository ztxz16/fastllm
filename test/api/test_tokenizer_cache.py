"""Tokenizer cache races, without loading a native model or CUDA library."""
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import unittest


source = Path(__file__).resolve().parents[2] / 'tools/fastllm_pytools/llm.py'
tree = ast.parse(source.read_text())
definitions = [node for node in tree.body
               if isinstance(node, (ast.ClassDef, ast.FunctionDef))
               and node.name in ('TokenizerCache', 'encode_hf_prompt')]
scope = {'threading': threading}
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), scope)
TokenizerCache = scope['TokenizerCache']


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


class TokenizerCacheTest(unittest.TestCase):
    def test_publish_during_lookup_and_eviction(self):
        cache = TokenizerCache()
        for _ in range(100):
            cache.add(['prefix'], [list(b'prefix')])
        entered, release = threading.Event(), threading.Event()
        match = cache.prompt_can_match

        def pause(*args):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('Concurrent publication blocked')
            return match(*args)

        cache.prompt_can_match = pause
        with ThreadPoolExecutor(max_workers=2) as pool:
            result = pool.submit(cache.tokenize_with_cache, Tokenizer(), 'prefix suffix')
            try:
                self.assertTrue(entered.wait(5))
                pool.submit(cache.add, ['new'], [list(b'new')]).result(timeout=5)
            finally:
                release.set()
            self.assertEqual(result.result(timeout=5), list(b'prefix suffix'))

    def test_published_history_is_immutable(self):
        cache = TokenizerCache()
        prompt, tokens = ['prefix'], [[99]]
        cache.add(prompt, tokens)
        prompt[0] = 'changed'
        tokens[0].append(100)
        self.assertEqual(cache.tokenize_with_cache(Tokenizer(), 'prefix!'), [99, 33])

    def test_miss_and_longest_matching_prefix(self):
        cache = TokenizerCache()
        cache.add(['pre'], [[1]])
        cache.add(['pre', 'fix'], [[1], [2]])
        self.assertEqual(cache.tokenize_with_cache(Tokenizer(), 'prefix!'), [1, 2, 33])
        self.assertEqual(cache.tokenize_with_cache(Tokenizer(), 'other'), list(b'other'))


if __name__ == '__main__':
    unittest.main()
