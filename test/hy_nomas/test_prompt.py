"""PE transport and scene shaping tests; no model or GPU required."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import io

path = Path(__file__).resolve().parents[2] / "tools/fastllm_pytools/nomas_prompt.py"
spec = importlib.util.spec_from_file_location("nomas_prompt", path)
pe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pe)

SCENE = {"perspective_view": "third-person observer", "Style": "real-world footage",
         "subject_list": [{"subject_id": "<|Subject1|>", "display_name": "skier",
                           "subject_attribute": "One baby wearing two separate skis."}],
         "segments": [{"start_time": "00:00.000", "end_time": "00:08.000",
                       "visual_description_long": "A snowy ski resort.", "description_change": None,
                       "contained_actions": [
                           {"start_time": "00:00.000", "end_time": "00:02.000", "action_description": "Standing on skis."},
                           {"start_time": "00:02.000", "end_time": "00:08.000", "action_description": "Jumping."}]}]}


class PromptTests(unittest.TestCase):
    def test_shape_crops_future_actions_without_accelerating_them(self):
        result = json.loads(pe.shape_scene(SCENE, 20/24))
        segment = result['segments'][0]
        self.assertNotIn('start_time', segment)
        self.assertNotIn('description_change', segment)
        self.assertEqual(len(segment['contained_actions']), 1)
        self.assertEqual(segment['contained_actions'][0]['end_time'], '00:00.833')
        self.assertEqual(segment['contained_actions'][0]['action_description'], 'Standing on skis.')
        self.assertEqual(len(SCENE['segments'][0]['contained_actions']), 2)

    def test_invalid_scenes_fail_before_generation(self):
        for value in (None, {}, {'subject_list': []}):
            with self.assertRaises(ValueError): pe.shape_scene(value, 1)
        bad = copy.deepcopy(SCENE)
        bad['subject_list'].append(bad['subject_list'][0])
        with self.assertRaises(ValueError): pe.shape_scene(bad, 1)
        bad = copy.deepcopy(SCENE)
        bad['segments'][0]['contained_actions'][0]['end_time'] = 'invalid'
        with self.assertRaises(ValueError): pe.shape_scene(bad, 1)

    def test_http_stream_preserves_schema_and_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ('system_prompt', 'first_turn'):
                (Path(folder)/('online_user_video_rewriter_'+name+'.md')).write_text('test rules')
            client = pe.PromptEnhancer('http://127.0.0.1:8008/v1/chat/completions', 'Nomas-PE', folder)
            def response(finish='stop'):
                events = [{'choices': [{'delta': {'content': json.dumps(SCENE)}, 'finish_reason': None}]},
                          {'choices': [{'delta': {}, 'finish_reason': finish}]}]
                return io.BytesIO((''.join('data: '+json.dumps(e)+'\n\n' for e in events)+'data: [DONE]\n\n').encode())
            with patch.object(pe.urllib.request, 'urlopen', return_value=response()) as call:
                result = client.enhance('A baby on skis', seed=42, num_frames=20)
                sent = json.loads(call.call_args.args[0].data)
                self.assertFalse(sent['chat_template_kwargs']['enable_thinking'])
                self.assertTrue(sent['stream'])
                current = sent['messages'][-1]['content'].split('Current user input JSON:\n')[-1]
                self.assertEqual(json.loads(current)['subject_description'], 'A baby on skis')
                self.assertEqual(json.loads(current)['environment_description'], 'A baby on skis')
                self.assertEqual(result['source_prompt'], 'A baby on skis')
                self.assertEqual(result['attempts'], 1)
            with patch.object(pe.urllib.request, 'urlopen', return_value=response('length')):
                with self.assertRaises(ValueError): client.complete([], 42, lambda: False)
            with patch.object(pe.urllib.request, 'urlopen', return_value=response()):
                with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                    client.complete([], 42, lambda: True)


if __name__ == '__main__':
    unittest.main()
