"""画板墨迹 -> 模型输入张量的预处理。

画板上的墨迹是一个 0~1 的 float32 矩阵(越大越黑), 训练和识别必须走同一套
预处理, 否则训练分布和推理分布不一致, 准确率会崩。

两种输入模式:
    center : 先按外接框裁剪, 再 LANCZOS 缩放并**按重心居中**放进 28x28。
             这是 MNIST 风格, 也是项目一贯的做法。
    raw    : 整幅画布(不裁剪)等比缩放到 28x28 后归一化。
             给自定义结构(例如直接吃整幅图的 MLP)用。
"""

import numpy as np
import torch
from PIL import Image

from mnist_data import MNIST_MEAN, MNIST_STD

# 下面尺寸是 96 DPI 下的"逻辑像素", 运行时按显示器 DPI 等比放大
BASE_DPI = 96.0
CANVAS_SIZE = 280          # 书写区边长
MARGIN = 2                 # 标准化时四周留白
BOX_SIDE = 28 - 2 * MARGIN  # 缩放后内容边长

# 归一化用的均值/标准差: 默认沿用 MNIST 官方统计量
DEFAULT_MEAN = MNIST_MEAN
DEFAULT_STD = MNIST_STD

INPUT_CENTER = "center"
INPUT_RAW = "raw"
INPUT_MODES = (INPUT_CENTER, INPUT_RAW)

# 判定"有墨迹"的阈值, 低于它认为画板是空的
INK_THRESHOLD = 0.05


def has_ink(arr: np.ndarray) -> bool:
    """画板上是否有内容。"""
    return float(arr.max()) >= INK_THRESHOLD


def center_ink(arr: np.ndarray) -> np.ndarray | None:
    """裁剪 + 缩放 + 重心居中, 返回 (28, 28) float32 或 None(空画板)。

    arr: (H, W) float32, 取值 0~1 表示墨迹浓度。
    """
    if not has_ink(arr):
        return None

    ys, xs = np.nonzero(arr > INK_THRESHOLD)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = arr[y0:y1, x0:x1]

    h, w = crop.shape
    scale = BOX_SIDE / max(h, w)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    # 用 LANCZOS 做抗锯齿缩放, 模拟 MNIST 的平滑笔画
    img = Image.fromarray((crop * 255).astype(np.uint8), mode="L")
    small = np.asarray(img.resize((nw, nh), Image.LANCZOS), dtype=np.float32) / 255.0

    # 按重心把数字摆到 28x28 正中央
    total = float(small.sum())
    if total <= 0:
        return None
    rows = np.arange(nh, dtype=np.float32)
    cols = np.arange(nw, dtype=np.float32)
    cy = float((small.sum(axis=1) * rows).sum() / total)
    cx = float((small.sum(axis=0) * cols).sum() / total)

    out = np.zeros((28, 28), dtype=np.float32)
    oy = int(round(13.5 - cy))
    ox = int(round(13.5 - cx))
    oy = max(0, min(28 - nh, oy))
    ox = max(0, min(28 - nw, ox))
    out[oy:oy + nh, ox:ox + nw] = small
    return out


def preprocess(arr: np.ndarray):
    """兼容旧接口: 等价于 center 模式。返回 (28, 28) float32 或 None。"""
    return center_ink(arr)


def raw_ink(arr: np.ndarray, size: int = 28) -> np.ndarray:
    """整幅画布缩放到 size x size, 不裁剪不居中。返回 float32 (size, size)。"""
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    small = np.asarray(img.resize((size, size), Image.LANCZOS), dtype=np.float32) / 255.0
    return small


def to_tensor(digit: np.ndarray, mean: float = DEFAULT_MEAN,
              std: float = DEFAULT_STD) -> torch.Tensor:
    """(28, 28) 灰度图 -> (1, 1, 28, 28) 归一化张量。"""
    x = torch.from_numpy(np.ascontiguousarray(digit)).float()
    x = (x - mean) / std
    return x.view(1, 1, 28, 28)


def ink_to_tensor(arr: np.ndarray, mode: str = INPUT_CENTER,
                  mean: float = DEFAULT_MEAN, std: float = DEFAULT_STD
                  ) -> torch.Tensor | None:
    """画板墨迹 -> 模型输入张量。空画板返回 None。

    mode: "center" 走 center_ink, "raw" 走 raw_ink。
    """
    if mode == INPUT_RAW:
        if not has_ink(arr):
            return None
        return to_tensor(raw_ink(arr), mean, std)
    digit = center_ink(arr)
    if digit is None:
        return None
    return to_tensor(digit, mean, std)


def ink_to_pil(arr: np.ndarray) -> Image.Image:
    """画板墨迹 -> 灰度 PIL 图(墨迹=深色), 用于显示与保存。"""
    return Image.fromarray(((1.0 - arr) * 255.0).astype(np.uint8), mode="L")
