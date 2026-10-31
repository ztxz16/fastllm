"""Compare native camera features with official relative poses and rays."""
import argparse
import ctypes
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from compare_vae import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--official', type=Path, required=True)
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('official_camera', args.official/'wan/utils/cam_utils.py')
    camera = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(camera)
    torch.backends.cuda.matmul.allow_tf32 = False
    lib = ctypes.CDLL(str(args.bridge))
    ptr = ctypes.POINTER(ctypes.c_float)
    lib.lingbot_test_camera.argtypes = [ptr]*3+[ctypes.c_int]*3
    lib.lingbot_test_poses.argtypes = [ptr]*3+[ctypes.c_int]
    lib.lingbot_test_error.restype = ctypes.c_char_p
    def pointer(value):
        return None if value is None else value.ctypes.data_as(ptr)
    def array(value):
        return value.detach().float().cpu().contiguous().numpy()
    results = []
    def record(name, value, expected):
        result = dict(name=name, **metrics(value, expected), different=int(np.count_nonzero(value != expected)))
        result['passed'] = result['finite'] and result['relative_l2'] < 2e-6
        results.append(result)
        print(name, result['relative_l2'], result['max_abs'], result['different'], flush=True)
    fixture = np.load(args.inputs)
    height, width = fixture['first_image'].shape[:2]
    intrinsics = fixture['intrinsics'].astype(np.float32)
    pose = torch.eye(4)
    for segment, action in enumerate(['stay', 'forward', 'look_left', 'right', 'look_up', 'backward', 'down', 'look_right', 'look_down', 'left', 'up']):
        old = pose.clone()
        delta = torch.eye(4)
        if action in ['forward', 'backward', 'left', 'right', 'up', 'down']:
            axis = dict(forward=2, backward=2, left=0, right=0, up=1, down=1)[action]
            delta[axis, 3] = -.1 if action in ['backward', 'left', 'up'] else .1
        elif action != 'stay':
            angle = np.deg2rad(-4 if action in ['look_left', 'look_down'] else 4)
            c, s = float(np.cos(angle)), float(np.sin(angle))
            delta[:3, :3] = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]] if action in ['look_left', 'look_right']
                else [[1, 0, 0], [0, c, -s], [0, s, c]])
        poses = []
        for frame in range(3):
            if segment or frame:
                pose = pose@delta
            poses.append(pose.clone())
        trajectory = torch.stack(poses)
        expected_pose = camera.compute_relative_poses(torch.stack(([old] if segment else [])+poses), framewise=True)
        if segment:
            expected_pose = expected_pose[1:]
        native_pose = np.empty((3, 4, 4), np.float32)
        if lib.lingbot_test_poses(pointer(array(trajectory)), pointer(array(old)) if segment else None, pointer(native_pose), 3):
            raise RuntimeError(lib.lingbot_test_error().decode())
        record(f'{action}/poses', native_pose, array(expected_pose))
        k = torch.from_numpy(intrinsics).repeat(3, 1).cuda()
        expected = camera.get_plucker_embeddings(expected_pose.cuda(), k, height, width)
        expected = expected.reshape(3, height//8, 8, width//8, 8, 6).permute(0, 1, 3, 5, 2, 4).reshape(3, height//8, width//8, 384).bfloat16()
        expected = array(expected)
        for label, poses_array in [('same_pose', array(expected_pose)), ('native_pose', native_pose)]:
            output = np.empty_like(expected)
            if lib.lingbot_test_camera(pointer(poses_array), pointer(intrinsics), pointer(output), 3, height, width):
                raise RuntimeError(lib.lingbot_test_error().decode())
            record(f'{action}/{label}', output, expected)
    passed = all(result['passed'] for result in results)
    args.output.write_text(json.dumps(dict(results=results, passed=passed), indent=2))
    if not passed:
        raise SystemExit('camera precision gate failed')


if __name__ == '__main__':
    main()
