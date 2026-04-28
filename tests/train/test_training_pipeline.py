import types
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

from geqtrain.train._key import TRAIN
from geqtrain.train.components.inference import prepare_chunked_input_data
from geqtrain.train.components.loop import TrainingLoop
from geqtrain.train.metrics import Metrics
from geqtrain.train.trainer import Trainer, check_for_config_updates
from geqtrain.utils.config import Config
from geqtrain.utils.torch_geometric import Data, Batch


class _DummyDist:
    is_distributed = False
    is_master = True
    device = torch.device("cpu")

    def sync_tensor(self, tensor):
        return tensor

    def sync_dict_of_tensors(self, data_dict, keys=None):
        return data_dict


class _DummyConfig(dict):
    def get(self, key, default=None):
        return super().get(key, default)

    def as_dict(self):
        return dict(self)


class _DummyLossStat:
    def __call__(self, loss, loss_contrib):
        return {"loss": loss.item()}

    def reset(self):
        return None

    def to(self, device):
        return self

    def current_result(self):
        return {"loss": 0.0}


class _DummyLoss:
    def __init__(self):
        self.loss_stat = _DummyLossStat()

    def __call__(self, pred, ref, **kwargs):
        loss = torch.mean((pred["y"] - ref["y"]) ** 2)
        return loss, {"loss": loss.detach()}

    def reset(self):
        return None

    def to(self, device):
        return self

    def current_result(self):
        return {"loss": 0.0}


class _DummyMetrics:
    def reset(self):
        return None

    def to(self, device):
        return self

    def __call__(self, pred, ref):
        return {"metric": torch.tensor(0.0, device=pred["y"].device)}

    def current_result(self, dist_manager=None):
        return {"metric": torch.tensor(0.0)}


class _DummySummary:
    def set_phase_results(self, phase, loss_results, metrics_results):
        self.results = (phase, loss_results, metrics_results)


class _StepCounter(torch.optim.SGD):
    def __init__(self, params, lr=0.1):
        super().__init__(params, lr=lr)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure=closure)


def _fake_run_inference(
    model,
    data,
    device,
    config,
    loss_fn=None,
    already_computed_nodes=None,
    is_train=False,
    current_epoch=0,
):
    x = data.to(device)
    out = {"y": model(x)}
    ref = {"y": torch.zeros_like(out["y"])}
    return out, ref, torch.empty(0, device=device), 0


def test_accumulation_flush_steps(monkeypatch):
    monkeypatch.setattr("geqtrain.train.components.loop.run_inference", _fake_run_inference)

    model = torch.nn.Linear(1, 1, bias=False)
    optim = _StepCounter(model.parameters(), lr=0.1)
    trainer = types.SimpleNamespace(
        model=model,
        optim=optim,
        loss=_DummyLoss(),
        metrics=_DummyMetrics(),
        ema=None,
        dist=_DummyDist(),
        lr_sched=None,
        warmup_sched=None,
        config=_DummyConfig({"accumulation_steps": 2, "chunking": False}),
        dl_train=[torch.ones(4, 1) for _ in range(3)],
        dl_val=[],
        metrics_metadata={},
        iepoch=0,
        _dispatch_callbacks=lambda *args, **kwargs: None,
    )

    loop = TrainingLoop(trainer)
    loop.run_phase(TRAIN, _DummySummary())

    assert optim.step_calls == 2


def test_steps_per_epoch_accounts_for_accumulation():
    trainer = Trainer.__new__(Trainer)
    trainer.dl_train = [None] * 5
    trainer.config = {"accumulation_steps": 2}

    assert trainer._get_steps_per_epoch() == 3


def test_prepare_chunked_input_empty_edges_returns_batch():
    edge_index = torch.empty((2, 0), dtype=torch.long)
    data = Data(pos=torch.zeros((2, 3)), edge_index=edge_index)
    batch = Batch.from_data_list([data])

    chunked, center_nodes = prepare_chunked_input_data(
        batch=batch,
        already_computed_nodes=None,
        batch_max_atoms=1000,
        chunk_ignore_keys=[],
    )

    assert chunked is batch
    assert center_nodes.numel() == 0


def _ddp_metrics_worker(rank, world_size, init_file, queue):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{init_file}",
    )

    metrics = Metrics(components=["y"])
    pred = {"y": torch.tensor([[rank + 1.0], [rank + 2.0]])}
    ref = {"y": torch.zeros_like(pred["y"])}
    metrics(pred, ref)

    dist_manager = types.SimpleNamespace(
        is_distributed=True,
        device=torch.device("cpu"),
        world_size=world_size,
        sync_tensor=lambda tensor: (dist.all_reduce(tensor, op=dist.ReduceOp.SUM), tensor)[1] / world_size,
    )
    result = metrics.current_result(dist_manager=dist_manager)
    value = next(iter(result.values())).item()
    queue.put(value)

    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
def test_ddp_metrics_are_aggregated(tmp_path):
    init_file = tmp_path / "dist_init"
    init_file.write_text("")
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()

    world_size = 2
    mp.spawn(
        _ddp_metrics_worker,
        args=(world_size, str(init_file), queue),
        nprocs=world_size,
        join=True,
    )

    results = [queue.get() for _ in range(world_size)]
    for value in results:
        assert value == pytest.approx(4.5)


def test_restart_sets_append(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    saved_config = Config.from_dict({"root": str(tmp_path), "run_name": "run"})
    torch.save({"config": saved_config}, run_dir / "trainer.pth")

    new_config = Config.from_dict({"root": str(tmp_path), "run_name": "run"})
    new_config.filepath = str(tmp_path / "config.yaml")

    trainer = Trainer.__new__(Trainer)
    trainer.dist = types.SimpleNamespace(
        is_master=True,
        is_distributed=False,
        broadcast_object=lambda obj, src=0: obj,
    )

    final_config = Trainer._resolve_config_and_restart_status(trainer, new_config)
    assert final_config.get("append") is True


def test_restart_config_comparison_ignores_missing_vs_empty_nested_fields(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    saved_config = Config.from_dict(
        {
            "root": str(tmp_path),
            "run_name": "run",
            "some_complex": {
                "name": "graph_input_attrs",
                "attributes": {},
                "eq_attributes": {},
                "nested": {"deep_empty": None},
            },
        }
    )
    torch.save({"config": saved_config}, run_dir / "trainer.pth")

    new_config = Config.from_dict(
        {
            "root": str(tmp_path),
            "run_name": "run",
            "some_complex": {
                "name": "graph_input_attrs",
            },
        }
    )
    new_config.filepath = str(tmp_path / "config.yaml")

    final_config = check_for_config_updates(new_config)
    assert final_config["some_complex"]["name"] == "graph_input_attrs"
