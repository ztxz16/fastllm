"""Configure weight residency in the native interactive world pipelines."""
import ctypes
import logging

WEIGHT_RESIDENCY_MODES = {'auto': -1, 'on': 1, 'off': 0}


class WeightResidency:
    def __init__(self, library, prefix, handle, mode):
        if mode not in WEIGHT_RESIDENCY_MODES:
            raise ValueError('weight residency must be auto, on or off')
        self.requested = mode
        self.handle = handle
        self.configure = getattr(library, prefix + '_pipeline_weight_residency', None)
        if self.configure is None:
            if mode == 'on':
                raise RuntimeError('Weight residency requires an updated native library')
            self.enabled = False
            return
        self.configure.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.configure.restype = ctypes.c_int
        self.error = getattr(library, prefix + '_error')
        self._apply(WEIGHT_RESIDENCY_MODES[mode])

    def _apply(self, mode):
        result = self.configure(self.handle, mode)
        if result < 0:
            raise RuntimeError(self.error().decode())
        self.enabled = bool(result)

    def status(self):
        return {'requested': self.requested, 'effective': 'on' if self.enabled else 'off'}

    def run(self, operation, *args):
        try:
            return operation(*args)
        except RuntimeError as error:
            # The native step commits world history and RNG only after success.
            # Retry only a GPU allocation failure and only in automatic mode.
            if self.requested != 'auto' or not self.enabled:
                raise
            message = str(error).strip().lower()
            gpu_allocation_failed = (
                message == 'out of memory' or ('cuda' in message and 'out of memory' in message)
                or 'cublas_status_alloc_failed' in message
                or message.endswith(('cublas error 3', 'cublas error: 3'))
            )
            if not gpu_allocation_failed:
                raise
        self._apply(0)
        logging.getLogger(__name__).warning('GPU allocation failed; retrying with weight residency off')
        return operation(*args)
