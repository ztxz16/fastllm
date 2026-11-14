"""正式模型工厂的 Attention DP/EP 适配器。"""

from .parallel_config import build_parallel_plan, is_attention_ep
from .process_tp import ProcessTPModel
from .qwen_moe_process_session import BACKEND


class ProcessMoeModel(ProcessTPModel):
    """/** 保留正式文本/流式生命周期；所有请求共享一个 EP 组和独立 Attention KV。 */"""

    backend = BACKEND

    def __init__(self, args):
        from .util import load_model_args

        load_model_args(args)
        if not is_attention_ep(args):
            raise ValueError('ProcessMoeModel requires --attention-dp')
        self.plan = build_parallel_plan(args)
        self.concurrency = self.attention_dp_size = self.ep_size = self.plan['ep_size']
        self.routing = getattr(args, 'dp_routing', 'round_robin')
        devices = [rank['device']['logical_id'] for rank in self.plan['ranks']]
        super().__init__(args, _device_ids=devices)

    def response(self, query, history=None, max_length=32, top_k=0, top_p=1.0,
                 temperature=0.0, repeat_penalty=1.0, seed=0):
        """/** 文本锁仅保护分词与解码；生成等待不占锁，允许不同Attention rank并发推进。 */"""
        with self._text_lock:
            request = self._request(query, history, max_length, top_k, top_p, temperature, repeat_penalty, seed)
        report = self.generate(request)
        with self._text_lock:
            return self.hf_tokenizer.decode(report['token_ids'], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
