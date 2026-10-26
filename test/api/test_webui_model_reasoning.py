import io
import json
import unittest
from unittest.mock import patch
from fastllm_pytools.webui_server import OpenAIModelClient, WebUIRuntime


class ModelReasoningTest(unittest.TestCase):
    def test_model_alias_uses_api_metadata_and_sends_exact_effort(self):
        client = OpenAIModelClient('http://localhost:1/v1', 'custom-alias')
        for efforts, default in ((['none', 'low', 'high', 'max'], 'high'),
                                 (['none', 'low', 'medium', 'high', 'xhigh'], 'xhigh')):
            metadata = {'supported_reasoning_efforts': efforts, 'default_reasoning_effort': default}
            with patch('fastllm_pytools.webui_server.with_model_metadata', return_value={'modelName':client.model_name, 'modelMetadata': metadata}):
                config = client.reasoning_config()
                self.assertEqual({key:config[key] for key in metadata}, metadata)
            for effort in efforts:
                payload = client._payload([], 32, 1, 1, 1, 1, effort, False)
                self.assertEqual(payload['reasoning_effort'], effort)
                self.assertEqual(payload['chat_template_kwargs']['reasoning_effort'], effort)
                self.assertEqual(payload['chat_template_kwargs']['enable_thinking'], effort != 'none')
                self.assertEqual(WebUIRuntime._pi_thinking_level({'thinking_level':effort}),
                                 'off' if effort == 'none' else effort)
            self.assertEqual(client._thinking_args('default')['reasoning_effort'], default)
            self.assertEqual(client._thinking_args('unsupported')['reasoning_effort'], default)

    def test_no_metadata_does_not_invent_an_effort(self):
        client = OpenAIModelClient('http://localhost:1/v1', 'alias')
        self.assertEqual(client._thinking_args('default'), {})

    def test_discovery_without_model_name_and_failed_refresh_keep_options(self):
        client = OpenAIModelClient('http://localhost:1/v1', '')
        metadata = {'id':'alias', 'supported_reasoning_efforts':['none', 'high', 'max'],
                    'default_reasoning_effort':'high'}
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps({'data':[metadata]}).encode())):
            config = client.reasoning_config()
        self.assertEqual(client.model_name, 'alias')
        self.assertEqual(config['supported_reasoning_efforts'], ['none', 'high', 'max'])
        with patch('urllib.request.urlopen', side_effect=TimeoutError):
            self.assertEqual(client.reasoning_config(), config)
        self.assertEqual(client._thinking_args('max')['reasoning_effort'], 'max')

    def test_legacy_labels_preserve_model_specific_meanings(self):
        for name, efforts, default, middle, high in (
                ('Qwen3.8-27B', ['none','low','medium','xhigh'], 'xhigh', 'medium', 'xhigh'),
                ('Kimi-K3', ['none','low','high','max'], 'max', 'high', 'max'),
                ('GLM5', ['none','low','high','max'], 'max', 'high', 'max'),
                ('DeepSeek-V4.1', ['none','low','high','max'], 'high', 'high', 'high')):
            with self.subTest(model=name):
                client = OpenAIModelClient('http://localhost:1/v1', name)
                client.model_metadata = {'supported_reasoning_efforts':efforts,
                                         'default_reasoning_effort':default}
                self.assertEqual(client._thinking_args('中')['reasoning_effort'], middle)
                self.assertEqual(client._thinking_args('高')['reasoning_effort'], high)
                self.assertEqual(client._thinking_args('关闭')['reasoning_effort'], 'none')

    def test_canonical_value_without_metadata_is_not_remapped(self):
        client = OpenAIModelClient('http://localhost:1/v1', 'alias')
        for effort in ('none','max','xhigh'):
            self.assertEqual(client._thinking_args(effort)['reasoning_effort'], effort)
