"""自定义模型的训练核心 —— 与界面完全解耦。

界面层(`ui_train.py`)只负责把工作区里的图片取出来、起一个后台线程、把
`progress` 回调的消息塞进队列; 真正的训练逻辑都在这里, 因此可以在没有 Tk
的环境里直接测试(见 `selftest.py`)。

`.pt` 文件是**自描述**的: 里面存了架构描述、标签表、输入模式与归一化统计量,
所以导入模型时不需要用户额外提供任何信息。
"""

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import arch as arch_mod
from dataset import PARTITION_TEST, PARTITION_TRAIN
from preprocess import DEFAULT_MEAN, DEFAULT_STD, INPUT_CENTER, INPUT_MODES, ink_to_tensor

FORMAT = "painter-pt"
VERSION = 1

# 归一化统计量的合理区间: 计算出来的值跑出这个范围说明数据形态不寻常
# (例如用户基于 center 裁剪后的小样本), 这时回退到 MNIST 官方统计量。
MEAN_RANGE = (0.02, 0.6)
STD_RANGE = (0.03, 0.8)


class TrainError(Exception):
    """训练无法进行或中途失败。"""


@dataclass
class Sample:
    ink: np.ndarray
    label: str


@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 32
    lr: float = 1e-3
    augment: bool = True
    seed: int = 0
    weight_decay: float = 1e-4
    input_mode: str = INPUT_CENTER


@dataclass
class TrainResult:
    ok: bool = False
    cancelled: bool = False
    accuracy: float | None = None       # 测试集 macro 平均通过率(没有测试集则 None)
    train_accuracy: float = 0.0
    epochs_done: int = 0
    seconds: float = 0.0
    mean: float = DEFAULT_MEAN
    std: float = DEFAULT_STD
    labels: list[str] = field(default_factory=list)
    state_dict: dict | None = None
    message: str = ""


# --------------------------------------------------------------------------- #
# 数据准备
# --------------------------------------------------------------------------- #
def load_samples(workspace, partition: str) -> list[Sample]:
    """把工作区某个分区里的全部图片取成 (墨迹, 标签) 列表。"""
    part = workspace.partition(partition)
    samples: list[Sample] = []
    for label in part.labels():
        dataset = part.dataset(label)
        for number in dataset.indices():
            samples.append(Sample(ink=dataset.ink(number), label=label))
    return samples


def compute_stats(samples: list[Sample], mode: str) -> tuple[float, float]:
    """按输入模式在**成对**的 28x28 图上算归一化统计量。"""
    if not samples:
        return DEFAULT_MEAN, DEFAULT_STD
    total = 0.0
    total_sq = 0.0
    count = 0
    for sample in samples:
        tensor = ink_to_tensor(sample.ink, mode)
        if tensor is None:
            continue
        values = tensor.numpy().ravel()
        total += float(values.sum())
        total_sq += float(np.square(values).sum())
        count += values.size
    if count == 0:
        return DEFAULT_MEAN, DEFAULT_STD
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    std = math.sqrt(variance)
    if not (MEAN_RANGE[0] <= mean <= MEAN_RANGE[1]):
        return DEFAULT_MEAN, DEFAULT_STD
    if not (STD_RANGE[0] <= std <= STD_RANGE[1]):
        return DEFAULT_MEAN, DEFAULT_STD
    return mean, std


def _build_tensors(samples: list[Sample], labels: list[str], mode: str,
                   mean: float, std: float) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """返回 (x, y, 每个样本在 batches 里的来源下标)。空画板会被丢弃。"""
    index_of = {label: i for i, label in enumerate(labels)}
    tensors: list[torch.Tensor] = []
    targets: list[int] = []
    kept: list[int] = []
    for position, sample in enumerate(samples):
        tensor = ink_to_tensor(sample.ink, mode, mean, std)
        if tensor is None:
            continue
        tensors.append(tensor[0])          # (1, 28, 28)
        targets.append(index_of[sample.label])
        kept.append(position)
    if not tensors:
        return (torch.zeros(0, 1, 28, 28), torch.zeros(0, dtype=torch.long), [])
    return torch.stack(tensors), torch.tensor(targets, dtype=torch.long), kept


def _augment(batch: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """随机旋转/平移/缩放。不依赖 torchvision。"""
    count = batch.shape[0]
    if count == 0:
        return batch
    from torch.nn.functional import affine_grid, grid_sample

    angle = (torch.rand(count, generator=generator) * 2 - 1) * (math.pi / 18)   # ±10°
    scale = 1.0 + (torch.rand(count, generator=generator) * 2 - 1) * 0.10       # ±10%
    tx = (torch.rand(count, generator=generator) * 2 - 1) * 0.10                # ±10%
    ty = (torch.rand(count, generator=generator) * 2 - 1) * 0.10

    cos = torch.cos(angle) * scale
    sin = torch.sin(angle) * scale
    theta = torch.zeros(count, 2, 3)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = ty

    grid = affine_grid(theta, batch.shape, align_corners=False)
    return grid_sample(batch, grid, mode="bilinear", padding_mode="zeros",
                       align_corners=False)


def suggest_epochs(samples: list[Sample]) -> int:
    """数据越多需要的 epoch 越少, 给界面一个合理默认值。"""
    count = len(samples)
    if count <= 20:
        return 60
    if count <= 60:
        return 30
    if count <= 200:
        return 20
    return 12


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def train_model(workspace, architecture: dict, config: TrainConfig,
                on_progress: Callable[[dict], None] | None = None,
                cancel: "Callable[[], bool] | None" = None) -> TrainResult:
    """在 CPU/GPU 上训练一个自定义模型。

    on_progress: 每步/每轮回调一个 dict(epoch, total_epochs, phase, accuracy…)。
    cancel:      返回 True 时尽快停止, 结果里 cancelled=True 且不返回权重。
    """
    progress = on_progress or (lambda _info: None)
    cancelled = cancel or (lambda: False)
    result = TrainResult()
    started = time.time()

    train_samples = load_samples(workspace, PARTITION_TRAIN)
    test_samples = load_samples(workspace, PARTITION_TEST)
    if not train_samples:
        raise TrainError("训练集中没有任何图片，请先在训练集中绘制样本。")

    labels = sorted(workspace.train_labels())
    if len(labels) < 2:
        raise TrainError(
            f"训练集中仅有 {len(labels)} 个数据集，至少需要 2 个才能进行分类训练。")
    counts = Counter(s.label for s in train_samples)
    empty = [label for label in labels if counts[label] == 0]
    if empty:
        raise TrainError(f"以下数据集没有任何图片：{' '.join(empty)}")

    mode = config.input_mode if config.input_mode in INPUT_MODES else INPUT_CENTER
    mean, std = compute_stats(train_samples, mode)
    result.mean, result.std, result.labels = mean, std, labels

    x_train, y_train, _kept = _build_tensors(train_samples, labels, mode, mean, std)
    x_test, y_test, _kept = _build_tensors(test_samples, labels, mode, mean, std)
    if x_train.shape[0] == 0:
        raise TrainError("训练集中的图片均为空白画板，没有可用样本。")
    if x_test.shape[0] == 0:
        x_test = None
        y_test = None

    torch.manual_seed(config.seed)
    architecture = {**architecture, "num_classes": len(labels)}
    model = arch_mod.build_model(architecture)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 类别不平衡 -> 按频率取倒数做权重, 并让采样器的各类别出现次数尽量均衡
    weights = torch.tensor([1.0 / counts[label] for label in labels], dtype=torch.float32)
    weights = weights * (len(weights) / weights.sum())
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)
    generator = torch.Generator().manual_seed(config.seed)

    sample_weights = torch.tensor([weights[int(t)] for t in y_train], dtype=torch.float32)
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights.double(), num_samples=int(x_train.shape[0]), replacement=True,
        generator=generator)

    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=min(config.batch_size, max(1, int(x_train.shape[0]))),
        sampler=sampler, drop_last=False)

    best_accuracy = -1.0
    best_state: dict | None = None
    result.epochs_done = 0
    total_steps = max(1, len(loader))

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for step, (batch_x, batch_y) in enumerate(loader, start=1):
            if cancelled():
                result.cancelled = True
                result.seconds = time.time() - started
                result.message = f"已在第 {epoch} 轮取消"
                progress({"phase": "cancelled", "epoch": epoch,
                          "total_epochs": config.epochs})
                return result

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if config.augment:
                batch_x = _augment(batch_x, generator)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

            running += float(loss.detach()) * batch_x.shape[0]
            seen += batch_x.shape[0]
            if step % 5 == 0 or step == total_steps:
                progress({"phase": "train", "epoch": epoch,
                          "total_epochs": config.epochs, "step": step,
                          "total_steps": total_steps,
                          "loss": running / max(1, seen),
                          "accuracy": None})

        model.eval()
        train_accuracy = _accuracy(model, x_train, y_train, device, cap=2000)
        if x_test is not None:
            accuracy = _accuracy(model, x_test, y_test, device)
            scheduler.step(accuracy)
        else:
            # 没有测试集: 用训练集准确率当监控指标(会有乐观偏差, 但总比没有好)
            accuracy = train_accuracy
            scheduler.step(accuracy)

        result.epochs_done = epoch
        result.train_accuracy = train_accuracy
        result.accuracy = accuracy
        progress({"phase": "epoch", "epoch": epoch, "total_epochs": config.epochs,
                  "loss": running / max(1, seen), "accuracy": accuracy,
                  "train_accuracy": train_accuracy,
                  "lr": optimizer.param_groups[0]["lr"]})

        if accuracy is not None and accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if cancelled():
        result.cancelled = True
        result.seconds = time.time() - started
        result.message = "已取消训练"
        return result

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model = install_state_dict(model, best_state, architecture=architecture)
    model.eval().cpu()

    result.ok = True
    result.state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    result.accuracy = best_accuracy if best_accuracy >= 0 else None
    result.seconds = time.time() - started
    result.message = (f"训练完成：共 {result.epochs_done} 轮，"
                      f"用时 {result.seconds:.1f} 秒")
    return result


def _accuracy(model, x: torch.Tensor, y: torch.Tensor, device, cap: int | None = None
              ) -> float:
    """macro 平均通过率: 每个类别各自算通过率再平均, 小样本类别不会被淹没。"""
    if x.shape[0] == 0:
        return 0.0
    if cap is not None and x.shape[0] > cap:
        x = x[:cap]
        y = y[:cap]
    model.eval()
    with torch.no_grad():
        prediction = model(x.to(device)).argmax(dim=1).cpu()
    total_hit = 0
    classes = 0
    scores = []
    for label in y.unique():
        mask = y == label
        hits = int((prediction[mask] == label).sum())
        count = int(mask.sum())
        scores.append(hits / count)
        total_hit += hits
        classes += count
    if classes == 0:
        return 0.0
    return float(sum(scores) / len(scores))


def predict_proba(model, ink: np.ndarray, mode: str, mean: float, std: float
                  ) -> np.ndarray | None:
    """单张墨迹 -> 各类概率。空画板返回 None。"""
    tensor = ink_to_tensor(ink, mode, mean, std)
    if tensor is None:
        return None
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        probs = torch.softmax(model(tensor.to(device)), dim=1)[0].cpu().numpy()
    return probs


# --------------------------------------------------------------------------- #
# .pt 存取(自描述)
# --------------------------------------------------------------------------- #
def install_state_dict(model: nn.Module, state: dict,
                       architecture: dict | None = None) -> nn.Module:
    """把权重装进模型并**校验真的装进去了**, 返回可用的模型。

    不依赖 `load_state_dict` 的返回值就完事: 这里直接逐个 `copy_`, 装完再和源
    张量比对, 不一致就按架构重建模型重来一次 —— 训练出来的模型必须能原样复现。
    """
    target = model.state_dict()
    missing = [key for key in state if key not in target]
    if missing:
        raise TrainError(
            f"权重与模型结构不匹配，多出以下参数：{', '.join(missing[:5])}")
    with torch.no_grad():
        for key, value in state.items():
            target[key].copy_(value)

    first = next(iter(state), None)
    if first is None or torch.equal(model.state_dict()[first], state[first]):
        return model

    # 极少见: copy_ 之后仍不一致。按架构重建一次再装。
    if architecture is None:
        raise TrainError("无法将权重正确装进模型")
    labels = int(architecture.get("num_classes") or state[first].shape[0])
    fresh = arch_mod.build_model({**architecture, "num_classes": labels})
    fresh_target = fresh.state_dict()
    if any(key not in fresh_target for key in state):
        raise TrainError("无法将权重正确装进模型")
    with torch.no_grad():
        for key, value in state.items():
            fresh_target[key].copy_(value)
    if not torch.equal(fresh.state_dict()[first], state[first]):
        raise TrainError("无法将权重正确装进模型，文件可能已损坏")
    return fresh


def make_checkpoint(architecture: dict, labels: list[str], state_dict: dict,
                    mean: float, std: float, name: str, accuracy: float | None,
                    epochs: int | None = None, extra: dict | None = None) -> dict:
    payload = {
        "format": FORMAT,
        "version": VERSION,
        "name": name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "arch": {**architecture, "num_classes": len(labels)},
        "labels": list(labels),
        "input_mode": architecture.get("input_mode", INPUT_CENTER),
        "mean": float(mean),
        "std": float(std),
        "accuracy": None if accuracy is None else float(accuracy),
        "epochs": epochs,
        "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
    }
    if extra:
        payload["extra"] = extra
    return payload


def save_checkpoint(path, payload: dict) -> None:
    torch.save(payload, path)


def default_arch_for(state_dict: dict, num_classes: int) -> dict:
    """给没有架构信息的旧权重(裸 state_dict)挑一个架构。

    现有 `mnist_cnn.pt` 就是这种: 10 类 -> 用默认架构。
    """
    return arch_mod.default_arch(num_classes)


def load_checkpoint(path) -> tuple[dict, nn.Module]:
    """读取 `.pt`。返回 (描述信息, 已 eval 的模型)。

    兼容两种内容:
      * 本程序保存的自描述 checkpoint(带 arch/labels/state_dict)
      * 裸 `OrderedDict` 权重(例如老的 mnist_cnn.pt) -> 按内置 MNIST 配置解释
    """
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:                     # noqa: BLE001 - 报错要原样带给用户
        raise TrainError(f"无法读取模型文件 {path}:\n{exc}") from exc
    path = Path(path)

    if isinstance(payload, dict) and payload.get("format") == FORMAT \
            and "state_dict" in payload:
        state = payload["state_dict"]
        labels = [str(x) for x in payload.get("labels") or []]
        architecture = arch_mod.normalize_arch(payload.get("arch"),
                                               num_classes=len(labels) or None)
        mean = float(payload.get("mean", DEFAULT_MEAN))
        std = float(payload.get("std", DEFAULT_STD))
        if std <= 0:
            std = DEFAULT_STD
        info = {
            "name": payload.get("name") or path.stem,
            "labels": labels,
            "arch": architecture,
            "input_mode": architecture.get("input_mode", INPUT_CENTER),
            "mean": mean,
            "std": std,
            "accuracy": payload.get("accuracy"),
            "trained_at": payload.get("trained_at"),
            "epochs": payload.get("epochs"),
            "legacy": False,
            "size_bytes": None,
        }
        model = arch_mod.build_model({**architecture, "num_classes": len(labels)})
        model = install_state_dict(model, state, architecture=architecture)
    else:
        # 裸 state_dict: 现有 mnist_cnn.pt 就是这种
        if not isinstance(payload, dict) or not payload:
            raise TrainError(f"{path} 里没有可用的权重")
        candidates: list[int] = []
        detected = _last_linear_out(payload)
        if detected:
            candidates.append(detected)
        if 10 not in candidates:
            candidates.append(10)
        model = None
        last_error = None
        for num_classes in candidates:
            candidate = arch_mod.build_model(
                arch_mod.default_arch(int(num_classes), INPUT_CENTER))
            try:
                model = install_state_dict(candidate, payload)
            except TrainError as exc:
                last_error = exc
                continue
            break
        if model is None:
            raise TrainError(
                "这是旧的裸权重文件，与内置 MNIST 结构不匹配，无法加载：\n"
                f"{last_error}")
        labels = [str(i) for i in range(candidates[0])]
        info = {
            "name": path.stem,
            "labels": labels,
            "arch": arch_mod.default_arch(len(labels), INPUT_CENTER),
            "input_mode": INPUT_CENTER,
            "mean": DEFAULT_MEAN,
            "std": DEFAULT_STD,
            "accuracy": None,
            "trained_at": None,
            "epochs": None,
            "legacy": True,
            "size_bytes": None,
        }

    if not info["labels"]:
        raise TrainError(f"{path} 中没有标签表，无法确定每个输出的含义。")
    model.eval()
    return info, model


def _last_linear_out(state: dict) -> int | None:
    """从 state_dict 里猜最后一个全连接层的输出维度。

    键名规则来自 `arch.build_model()`: 全连接层都在 classifier 命名空间下,
    键形如 `classifier.1.weight`, 所以取出编号最大的那个即可。
    """
    best_index = -1
    best_out = None
    for key, value in state.items():
        if not key.endswith(".weight") or getattr(value, "dim", lambda: 0)() != 2:
            continue
        stem = key[: -len(".weight")]
        head, _, tail = stem.rpartition(".")
        index = int(tail) if tail.isdigit() else 0
        if index >= best_index:
            best_index = index
            best_out = int(value.shape[0])
    return best_out
