from geqtrain.train.components.distributed import DistributedManager


def test_non_ddp_explicit_cuda_device_sets_current_cuda_device(monkeypatch):
    calls = []
    monkeypatch.setattr("torch.cuda.set_device", lambda dev: calls.append(str(dev)))
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    manager = DistributedManager(config={"ddp": False, "device": "cuda:2"})

    assert str(manager.device) == "cuda:2"
    assert calls == ["cuda:2"]


def test_non_ddp_cpu_device_does_not_set_cuda_device(monkeypatch):
    calls = []
    monkeypatch.setattr("torch.cuda.set_device", lambda dev: calls.append(str(dev)))

    manager = DistributedManager(config={"ddp": False, "device": "cpu"})

    assert str(manager.device) == "cpu"
    assert calls == []
