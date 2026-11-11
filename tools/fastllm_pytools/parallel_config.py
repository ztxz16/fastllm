"""DP/TP 配置和 CUDA 设备分配；本模块不加载模型或启动推理进程。"""

import ctypes
import os
import re
import sys
import uuid


class _CUuuid(ctypes.Structure):
    _fields_ = [('bytes', ctypes.c_ubyte * 16)]


def cuda_inventory():
    """/**
     * 查询当前进程实际可见的 CUDA 设备，不创建或保留 CUDA context。
     * 只调用 Driver API 的初始化和设备查询接口，不导入推理库；任何查询
     * 失败均报错，不用物理卡数或环境变量长度替代实际可见卡数。
     * @return 包含驱动支持的 CUDA 版本及有序设备身份的字典。
     */"""
    if not sys.platform.startswith('linux'):
        raise ValueError('parallel GPU planning currently requires Linux + NVIDIA CUDA')
    try:
        driver = ctypes.CDLL('libcuda.so.1')
    except OSError as error:
        raise RuntimeError(f'cannot query CUDA devices: {error}') from error

    signatures = {
        'cuInit': [ctypes.c_uint],
        'cuDriverGetVersion': [ctypes.POINTER(ctypes.c_int)],
        'cuDeviceGetCount': [ctypes.POINTER(ctypes.c_int)],
        'cuDeviceGet': [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        'cuDeviceGetUuid_v2': [ctypes.POINTER(_CUuuid), ctypes.c_int],
        'cuDeviceGetName': [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int],
    }
    for name, arguments in signatures.items():
        try:
            function = getattr(driver, name)
        except AttributeError as error:
            raise RuntimeError(f'CUDA driver is missing {name}') from error
        function.argtypes, function.restype = arguments, ctypes.c_int

    def check(name, *arguments):
        status = getattr(driver, name)(*arguments)
        if status != 0:
            raise RuntimeError(f'{name} failed with CUDA error {status}')

    version = ctypes.c_int()
    check('cuDriverGetVersion', ctypes.byref(version))
    status = driver.cuInit(0)
    if status == 100:  # CUDA_ERROR_NO_DEVICE
        return {'driver_cuda_version': version.value, 'devices': []}
    if status != 0:
        raise RuntimeError(f'cuInit failed with CUDA error {status}')

    count = ctypes.c_int()
    check('cuDeviceGetCount', ctypes.byref(count))
    devices = []
    for ordinal in range(count.value):
        device, identity = ctypes.c_int(), _CUuuid()
        name = ctypes.create_string_buffer(256)
        check('cuDeviceGet', ctypes.byref(device), ordinal)
        check('cuDeviceGetUuid_v2', ctypes.byref(identity), device.value)
        check('cuDeviceGetName', name, len(name), device.value)
        devices.append({'logical_id': ordinal, 'uuid': str(uuid.UUID(bytes=bytes(identity.bytes))),
                        'name': name.value.decode('utf-8', errors='replace')})
    return {'driver_cuda_version': version.value, 'devices': devices}


def validate_parallel_options(args, *, planning=False):
    """/**
     * 校验 CLI 与配置文件共用的并行参数；旧 DP=1 执行不触发 GPU 查询。
     * @param args 已展开配置文件的参数对象。
     * @param planning 为 True 时也检查单副本的预览限制。
     * @return DP 副本数；非法类型、后端或新模式参数冲突抛出 ValueError。
     */"""
    dp = getattr(args, 'dp', 1)
    if type(dp) is not int or dp < 1:
        raise ValueError('--dp must be a positive integer (JSON booleans/floats are not integers)')
    preview = getattr(args, 'parallel_plan', False)
    if type(preview) is not bool:
        raise ValueError('--parallel-plan must be boolean')
    backend = getattr(args, 'tp_backend', 'thread')
    if backend not in ('thread', 'process'):
        raise ValueError('tp_backend must be thread or process')
    if dp == 1 and not preview and not planning:
        return dp

    command = getattr(args, 'command', None)
    if command not in (None, 'chat', 'run', 'serve', 'server'):
        raise ValueError('parallel planning currently supports chat/run/server/serve only')
    if dp > 1 and backend != 'process':
        raise ValueError('DP>1 requires --tp-backend process')
    if getattr(args, 'device', None) not in (None, '') or getattr(args, 'moe_device', '') not in (None, ''):
        raise ValueError('parallel planning cannot combine --device/--moe_device; select GPUs with CUDA_VISIBLE_DEVICES')
    for name in ('cuda_graph', 'custom_allreduce'):
        if type(getattr(args, name, False)) is not bool:
            raise ValueError(f'{name} must be boolean')
    if getattr(args, 'cuda_graph', False) and backend != 'process':
        raise ValueError('--cuda-graph requires --tp-backend process')
    if dp > 1:
        _dp_tp_size(args)
    return dp


def _dp_tp_size(args):
    """/** 解析多副本的每副本卡数；空值表示单卡，不接受旧设备列表语义。 */"""
    value = getattr(args, 'tp', '')
    if value == '' or value is None:
        return 1
    if type(value) is not int and not (isinstance(value, str) and re.fullmatch(r'[0-9]+', value.strip())):
        raise ValueError('DP>1 requires --tp to be a positive GPU count; select/order GPUs with CUDA_VISIBLE_DEVICES')
    count = int(value)
    if not 1 <= count <= 64:
        raise ValueError('DP>1 requires --tp in 1..64 GPUs per replica')
    return count


def _single_replica_ids(args, available):
    """/** 预览单副本的简单 CUDA 设备列表；不改变正常启动的旧 TP 解析路径。 */"""
    spec = str(getattr(args, 'tp', '') or '').strip()
    if getattr(args, 'tp_backend', 'thread') == 'process':
        from .process_tp import process_devices
        return [int(item) for item in process_devices(spec).split(',')]
    lower = spec.lower()
    if lower in ('', 'false', 'off', 'none', 'disable'):
        return [0]
    if lower in ('auto', 'true', 'on'):
        return list(range(available))
    if re.fullmatch(r'[0-9]+', spec):
        count = max(1, int(spec))
        if count > available:
            raise ValueError(f'not enough visible CUDA devices: need {count}, found {available}; preview does not clamp')
        return list(range(count))
    if re.fullmatch(r'[0-9]+(?:,[0-9]+)+', spec):
        return [int(item) for item in spec.split(',')]
    raise ValueError('--parallel-plan supports unweighted --tp counts, lists or auto; normal DP=1 execution is unchanged')


def build_parallel_plan(args):
    """/**
     * 按 CUDA 可见顺序生成副本设备分配，仅返回计划，不加载或执行模型。
     * @param args 已展开配置文件的参数；DP>1 时每副本卡数相同且设备互斥。
     * @return 包含 DP/TP 大小、逻辑编号、UUID、驱动信息的 JSON 可序列化字典。
     *         卡数不足、重复身份或不支持的配置直接报错，不静默裁剪或回退。
     */"""
    dp = validate_parallel_options(args, planning=True)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        entries = [item.strip() for item in visible.split(',')]
        if not all(entries) or '-1' in entries:
            raise ValueError('CUDA_VISIBLE_DEVICES has no usable devices or contains an empty/disabled entry')
        # CUDA 会在重复序号处截断枚举；提前拒绝，避免接受一个被静默截断的计划。
        canonical = [str(int(item)) if re.fullmatch(r'[0-9]+', item) else item for item in entries]
        if len(set(canonical)) != len(canonical):
            raise ValueError('CUDA_VISIBLE_DEVICES contains duplicate devices')

    inventory = cuda_inventory()
    devices = inventory['devices']
    available = len(devices)
    if not available:
        raise ValueError('no visible CUDA devices')
    if visible is not None and available != len(entries):
        raise ValueError(f'CUDA_VISIBLE_DEVICES lists {len(entries)} entries but CUDA exposes {available}; '
                         'check invalid/duplicate selectors or a visibility change after CUDA initialization')
    if len({device['uuid'] for device in devices}) != available:
        raise ValueError('duplicate CUDA device UUIDs; replicas must use distinct devices')
    if dp > 1:
        tp = _dp_tp_size(args)
        needed = dp * tp
        if needed > available:
            raise ValueError(f'not enough visible CUDA devices: DP={dp} * TP={tp} needs {needed}, found {available}')
        assignments = [list(range(rank * tp, (rank + 1) * tp)) for rank in range(dp)]
    else:
        selected = _single_replica_ids(args, available)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError('TP devices must be nonempty and distinct')
        if any(device < 0 or device >= available for device in selected):
            raise ValueError(f'TP device index exceeds {available} actual visible CUDA devices')
        tp, assignments = len(selected), [selected]
    if getattr(args, 'custom_allreduce', False) and (getattr(args, 'tp_backend', 'thread') != 'process' or tp != 2):
        raise ValueError('--custom-allreduce requires process TP=2 within each replica')

    return {'dp_size': dp, 'tp_size': tp, 'total_gpus': dp * tp, 'tp_backend': getattr(args, 'tp_backend', 'thread'),
            'preview_only': True, 'multi_replica_execution_available': False,
            'cuda_visible_devices': visible, 'driver_cuda_version': inventory['driver_cuda_version'],
            'visible_device_count': available,
            'replicas': [{'dp_rank': rank, 'devices': [dict(devices[index]) for index in ids]}
                         for rank, ids in enumerate(assignments)]}


def prepare_parallel_entry(args, *, allow_preview=False):
    """/**
     * 为正式 CLI、模型工厂及直接进程后端入口提供同一启动边界。
     * @param args 已展开配置的参数。
     * @param allow_preview 仅 CLI 允许返回预览；模型工厂不得把计划当模型返回。
     * @return 预览时返回分配字典；旧单副本执行返回 None。多副本执行尚未接入时
     *         完成配置与设备校验后明确报错，防止退回单副本执行。
     */"""
    dp = validate_parallel_options(args)
    if getattr(args, 'parallel_plan', False):
        if not allow_preview:
            raise ValueError('--parallel-plan is a CLI preview; use build_parallel_plan for a programmatic plan')
        return build_parallel_plan(args)
    if dp > 1:
        build_parallel_plan(args)
        raise NotImplementedError('DP>1 多副本执行尚未接入；使用 --parallel-plan 预览 GPU 分配')
    return None
