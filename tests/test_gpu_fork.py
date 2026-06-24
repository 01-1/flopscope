import numpy as np

import flopscope as flops
import flopscope.numpy as fnp
import flopscope._gpu as flops_gpu


def test_gpu_controls_are_exposed():
    assert "+gpu.np" in flops.__version__
    status = flops.gpu_status()
    assert status["min_flops"] == 5_000_000
    assert status["min_transfer_bytes"] == 0
    assert status["min_flops_per_byte"] == 0.05
    assert "matmul" in status["supported_functions"]


def test_gpu_opt_in_preserves_cpu_array_contract():
    previous = flops.gpu_enabled()
    flops.configure_gpu(True)
    try:
        with flops.BudgetContext(flop_budget=10_000) as budget:
            a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
            b = fnp.array([[5.0], [6.0]])
            out = fnp.matmul(a, b)

        assert isinstance(out, fnp.ndarray)
        np.testing.assert_allclose(np.asarray(out), np.array([[17.0], [39.0]]))
        assert budget.flops_used > 0
    finally:
        flops.configure_gpu(previous)


def test_gpu_runtime_failure_falls_back_to_cpu(monkeypatch):
    class FakeRuntime:
        @staticmethod
        def getDeviceCount():
            return 1

    class FakeStream:
        null = object()

    class FakeCuda:
        runtime = FakeRuntime()
        Stream = FakeStream()

    class FakePool:
        def free_all_blocks(self):
            pass

    class FakeCupy:
        cuda = FakeCuda()
        float32 = np.float32

        @staticmethod
        def empty(*args, **kwargs):
            raise RuntimeError("synthetic cuda allocation failure")

        @staticmethod
        def get_default_memory_pool():
            return FakePool()

        @staticmethod
        def get_default_pinned_memory_pool():
            return FakePool()

    monkeypatch.setattr(flops_gpu, "_cupy", FakeCupy)
    monkeypatch.setattr(flops_gpu, "_cupy_error", None)
    monkeypatch.setattr(flops_gpu, "_runtime_error", None)
    monkeypatch.setattr(flops_gpu, "_runtime_checked", False)

    previous = flops.gpu_enabled()
    flops.configure_gpu(True)
    try:
        used_gpu, result = flops_gpu.maybe_call_gpu(np.matmul, np.eye(2), np.eye(2))
        assert used_gpu is False
        assert result is None
        status = flops.gpu_status()
        assert status["available"] is False
        assert "synthetic cuda allocation failure" in status["runtime_error"]
    finally:
        flops.configure_gpu(previous)
