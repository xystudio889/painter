"""可配置网络的架构描述 —— 卡片式"模型参数"界面的后端。

设计要点
--------
1. **默认即现有网络**: `default_arch()` 描述的卡片序列构建出的模型, 与
   `model.DigitCNN` 的 state_dict 键名/形状**完全一致**, 因此老的
   `mnist_cnn.pt` 可以直接加载进卡片式构建出的模型(这一点有自测保证)。

2. 输出层不建卡片: 由 `num_classes` 自动追加 `Linear(-> num_classes)`,
   训练用 `CrossEntropyLoss`。

3. 两种输入模式:
   - `center`: 裁剪 + 缩放 + 重心居中到 28x28, 与 MNIST 风格一致(默认)
   - `raw`   : 整幅画布缩放到 28x28, 适合自定义的 MLP 类结构

4. 形状校验: `validate_arch()` 逐卡前向推导张量形状, 报错带**卡片序号**,
   卡片 UI 就能直接把错误标在对应卡片下面。
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from preprocess import INPUT_CENTER, INPUT_MODES

TYPES = ("conv", "pool", "fc")
ACTIVATIONS = ("relu", "none")

KERNEL_CHOICES = (1, 3, 5)
PADDING_CHOICES = (0, 1, 2)
POOL_CHOICES = (2, 3, 4)

CHANNEL_RANGE = (1, 512)
UNITS_RANGE = (8, 4096)
DROPOUT_RANGE = (0.0, 0.9)
MASKS_RANGE = (0, 10)
LABELS_RANGE = (2, 1000)

INPUT_SIZE = 28


class ArchError(ValueError):
    """架构描述不合法, 无法构建模型。"""


@dataclass
class CardError:
    """一条校验错误。`index` 为 None 表示与具体卡片无关(全局错误)。"""

    index: int | None
    message: str

    def __str__(self) -> str:
        if self.index is None:
            return self.message
        return f"第 {self.index + 1} 张卡片：{self.message}"


def _as_int(value: Any, name: str, low: int, high: int, errors: list[CardError],
            index: int | None) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        errors.append(CardError(index, f"{name}必须为整数，实际收到 `{value!r}`"))
        return None
    if not (low <= number <= high):
        errors.append(CardError(
            index, f"{name}必须在 {low}~{high} 之间，实际收到 {number}"))
        return None
    return number


def _as_float(value: Any, name: str, low: float, high: float,
              errors: list[CardError], index: int | None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(CardError(index, f"{name}必须为数字，实际收到 `{value!r}`"))
        return None
    if not (low <= number <= high):
        errors.append(CardError(
            index, f"{name}必须在 {low}~{high} 之间，实际收到 {number}"))
        return None
    return number


def default_card() -> dict:
    return {"type": "conv", "channels": 32, "kernel": 3, "padding": 1,
            "activation": "relu"}


def new_card(card_type: str) -> dict:
    """按类型造一张默认参数的卡片。"""
    if card_type == "pool":
        return {"type": "pool", "window": 2}
    if card_type == "fc":
        return {"type": "fc", "units": 128, "activation": "relu", "dropout": 0.25}
    return default_card()


def default_arch(num_classes: int = 10, input_mode: str = INPUT_CENTER) -> dict:
    """等价于 `DigitCNN` 的默认架构(2 层卷积 + 2 层全连接)。"""
    return {
        "input_mode": input_mode,
        "input_shape": [1, INPUT_SIZE, INPUT_SIZE],
        "num_classes": int(num_classes),
        "source": "default",
        "cards": [
            {"type": "conv", "channels": 32, "kernel": 3, "padding": 1, "activation": "relu"},
            {"type": "pool", "window": 2},
            {"type": "conv", "channels": 64, "kernel": 3, "padding": 1, "activation": "relu"},
            {"type": "pool", "window": 2},
            {"type": "fc", "units": 128, "activation": "relu", "dropout": 0.25},
        ],
    }


def normalize_arch(arch: dict | None, num_classes: int | None = None) -> dict:
    """把外部读进来的架构描述补齐成完整形式(不改内容, 只填缺省)。"""
    if not isinstance(arch, dict):
        arch = {}
    cards = arch.get("cards")
    if not isinstance(cards, list) or not cards:
        base = default_arch(num_classes or arch.get("num_classes") or 10)
        if num_classes:
            base["num_classes"] = int(num_classes)
        return base
    out = {
        "input_mode": arch.get("input_mode") if arch.get("input_mode") in INPUT_MODES
        else INPUT_CENTER,
        "input_shape": list(arch.get("input_shape") or [1, INPUT_SIZE, INPUT_SIZE]),
        "num_classes": int(num_classes if num_classes is not None
                           else arch.get("num_classes") or 10),
        "source": arch.get("source") or "custom",
        "cards": [dict(card) if isinstance(card, dict) else {} for card in cards],
    }
    return out


def validate_arch(arch: dict) -> list[CardError]:
    """逐卡推导形状并返回所有错误(空列表 = 合法)。"""
    errors: list[CardError] = []
    if not isinstance(arch, dict):
        return [CardError(None, "架构描述不是字典")]

    num_classes = _as_int(arch.get("num_classes", 0), "类别数", *LABELS_RANGE,
                          errors, None)
    if arch.get("input_mode") not in INPUT_MODES:
        errors.append(CardError(None, f"输入模式必须为 {INPUT_MODES} 之一"))

    shape_raw = arch.get("input_shape") or [1, INPUT_SIZE, INPUT_SIZE]
    try:
        if len(shape_raw) != 3:
            raise ValueError
        c, h, w = (int(shape_raw[0]), int(shape_raw[1]), int(shape_raw[2]))
        if min(c, h, w) < 1:
            raise ValueError
    except (TypeError, ValueError):
        errors.append(CardError(None, "输入形状必须为 [通道, 高, 宽] 三个正整数"))
        return errors

    cards = arch.get("cards")
    if not isinstance(cards, list) or not cards:
        errors.append(CardError(None, "至少需要一张卡片"))
        return errors

    # 逐卡前向推导
    for index, card in enumerate(cards):
        if not isinstance(card, dict):
            errors.append(CardError(index, "卡片内容必须为字典"))
            continue
        card_type = card.get("type")
        if card_type not in TYPES:
            errors.append(CardError(
                index, f"类型必须为 {TYPES} 之一，实际收到 `{card_type!r}`"))
            continue
        if card_type == "conv":
            if c is None:
                errors.append(CardError(index, "前面已拍平为一维，此处不能再连接卷积层"))
                continue
            channels = _as_int(card.get("channels", 32), "通道数", *CHANNEL_RANGE,
                               errors, index)
            kernel = _as_int(card.get("kernel", 3), "卷积核",
                             KERNEL_CHOICES[0], KERNEL_CHOICES[-1], errors, index)
            if kernel is not None and kernel not in KERNEL_CHOICES:
                errors.append(CardError(index, f"卷积核必须为 {KERNEL_CHOICES} 之一"))
            padding = _as_int(card.get("padding", 1), "填充", 0, 2, errors, index)
            if card.get("activation", "relu") not in ACTIVATIONS:
                errors.append(CardError(index, f"激活函数必须为 {ACTIVATIONS} 之一"))
            if channels is None or kernel is None or padding is None:
                continue
            h = h + 2 * padding - kernel + 1
            w = w + 2 * padding - kernel + 1
            c = channels
        elif card_type == "pool":
            if c is None:
                errors.append(CardError(index, "前面已拍平为一维，此处不能再连接池化层"))
                continue
            window = _as_int(card.get("window", 2), "池化窗口", 2, 4, errors, index)
            if window is None:
                continue
            if window > min(h, w):
                errors.append(CardError(
                    index, f"池化窗口 {window} 大于特征图尺寸（{w}×{h}），"
                           "会得到空输出"))
                continue
            h = h // window
            w = w // window
            if min(h, w) < 1:
                errors.append(CardError(index, "池化后特征图尺寸变为 0，请减小池化窗口"))
                continue
        else:  # fc
            if c is not None:
                # 第一次遇到全连接: 先把前面拍平, 之后通道维度就没了
                c = None
                h = w = 1
            units = _as_int(card.get("units", 128), "隐藏单元", *UNITS_RANGE, errors, index)
            if card.get("activation", "relu") not in ACTIVATIONS:
                errors.append(CardError(index, f"激活函数必须为 {ACTIVATIONS} 之一"))
            if card.get("dropout") is not None:
                _as_float(card.get("dropout"), "Dropout", *DROPOUT_RANGE, errors, index)
            if units is None:
                continue

    return errors


def _activation(name: str) -> nn.Module:
    return nn.ReLU(inplace=True) if name == "relu" else nn.Identity()


def build_model(arch: dict) -> nn.Module:
    """按架构描述构建模型。不合法时抛 `ArchError`。"""
    errors = validate_arch(arch)
    if errors:
        raise ArchError("; ".join(str(e) for e in errors))

    arch = normalize_arch(arch)
    channels, height, width = (int(v) for v in arch["input_shape"])
    num_classes = int(arch["num_classes"])

    features: list[nn.Module] = []
    classifier: list[nn.Module] = []
    # 已经拍平过一次就不再重复 Flatten(键名索引要保持可预测)
    flattened = False

    for card in arch["cards"]:
        ctype = card["type"]
        if ctype == "conv":
            kernel = int(card.get("kernel", 3))
            padding = int(card.get("padding", 1))
            out_channels = int(card.get("channels", 32))
            # 刻意把 Conv 与激活**平铺**进 features, 而不是包一层 Sequential:
            # 这样默认架构生成的 state_dict 键名与手写版 DigitCNN 完全一致,
            # 老的 mnist_cnn.pt 可以直接装进来。
            features.append(nn.Conv2d(channels, out_channels,
                                      kernel_size=kernel, padding=padding))
            features.append(_activation(card.get("activation", "relu")))
            channels = out_channels
            height = height + 2 * padding - kernel + 1
            width = width + 2 * padding - kernel + 1
        elif ctype == "pool":
            window = int(card.get("window", 2))
            features.append(nn.MaxPool2d(window))
            height //= window
            width //= window
        else:  # fc
            if not flattened:
                classifier.append(nn.Flatten())
                flattened = True
                flat_in = channels * height * width
            else:
                # 第二张及以后的 FC 卡片接在上一张的输出上
                flat_in = _last_flat_out(classifier)
            units = int(card.get("units", 128))
            classifier.append(nn.Linear(flat_in, units))
            classifier.append(_activation(card.get("activation", "relu")))
            if card.get("dropout") is not None and float(card.get("dropout") or 0) > 0:
                classifier.append(nn.Dropout(float(card["dropout"])))

    if not flattened:
        # 只写了卷积/池化, 没写全连接: 自己补一次拍平
        classifier.append(nn.Flatten())
        flat_in = channels * height * width
    else:
        # 末尾的输出层接在最后一张 FC 卡片的输出上
        flat_in = _last_flat_out(classifier)

    classifier.append(nn.Linear(flat_in, num_classes))
    return ArchModel(nn.Sequential(*features), nn.Sequential(*classifier))


def _last_flat_out(classifier: list[nn.Module]) -> int:
    """找出 classifier 里最后一个全连接层的输出维度; 没有则说明还没拍平。"""
    for module in reversed(classifier):
        if isinstance(module, nn.Linear):
            return module.out_features
    raise ArchError("全连接层之前必须先有一层全连接或拍平操作")


class ArchModel(nn.Module):
    """卡片式构建出的网络, 与 `DigitCNN` 保持同样的平铺命名。

    用 `features` / `classifier` 两个 Sequential 组装, 这样默认架构生成的
    state_dict 键名与手写版 `DigitCNN` 完全一致(老的 mnist_cnn.pt 可以直接装)。
    """

    def __init__(self, features: nn.Sequential, classifier: nn.Sequential):
        super().__init__()
        self.features = features
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=1)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def describe_arch(arch: dict) -> str:
    """一行人类可读的架构摘要, 用于对话框顶部。"""
    try:
        model = build_model(arch)
    except ArchError as exc:
        return f"架构不合法：{exc}"
    cards = normalize_arch(arch)["cards"]
    text = " → ".join(_describe_card(card) for card in cards)
    return f"{text} → Linear({normalize_arch(arch)['num_classes']})    " \
           f"参数 {count_parameters(model):,}"


def _describe_card(card: dict) -> str:
    ctype = card.get("type")
    if ctype == "conv":
        return (f"Conv({card.get('channels')}, k{card.get('kernel')}"
                f", p{card.get('padding')})")
    if ctype == "pool":
        return f"Pool({card.get('window')})"
    if ctype == "fc":
        drop = card.get("dropout")
        extra = f", drop {drop}" if drop else ""
        return f"FC({card.get('units')}{extra})"
    return str(ctype)
