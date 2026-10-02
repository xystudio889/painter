"""读取 MNIST 的 IDX 二进制文件。

MNIST 官方文件格式(大端序):
    前 4 字节  magic number, 低 8 位表示维度个数
    随后每个维度占 4 字节
    之后是原始像素 / 标签数据
"""

from pathlib import Path

import numpy as np

# MNIST 官方公布的全局均值与标准差(用于归一化)
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

DATA_DIR = Path(__file__).resolve().parent / "mnist"

FILES = {
    "train_images": "train-images.idx3-ubyte",
    "train_labels": "train-labels.idx1-ubyte",
    "test_images": "t10k-images.idx3-ubyte",
    "test_labels": "t10k-labels.idx1-ubyte",
}


def read_idx(path) -> np.ndarray:
    """读取一个 IDX 文件并返回 numpy 数组。"""
    with open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        ndim = magic & 0xFF
        dtype_code = (magic >> 8) & 0xFF
        if not 1 <= ndim <= 4 or dtype_code != 0x08:
            raise ValueError(
                f"{path} 不是合法的 uint8 IDX 文件 (magic=0x{magic:08x})"
            )
        dims = [int.from_bytes(f.read(4), "big") for _ in range(ndim)]
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != int(np.prod(dims)):
        raise ValueError(f"{path} 数据长度与文件头声明不一致")
    return data.reshape(dims)


def load_mnist(data_dir=DATA_DIR):
    """加载完整数据集。

    返回 (train_x, train_y, test_x, test_y)。
    train_x / test_x 形状为 (N, 28, 28),dtype=uint8;
    train_y / test_y 形状为 (N,),dtype=uint8。
    """
    data_dir = Path(data_dir)
    missing = [n for n in FILES.values() if not (data_dir / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"在 {data_dir} 中找不到以下文件: {', '.join(missing)}"
        )
    return (
        read_idx(data_dir / FILES["train_images"]),
        read_idx(data_dir / FILES["train_labels"]),
        read_idx(data_dir / FILES["test_images"]),
        read_idx(data_dir / FILES["test_labels"]),
    )


if __name__ == "__main__":
    tx, ty, ex, ey = load_mnist()
    print("训练集:", tx.shape, tx.dtype, ty.shape, ty.dtype)
    print("测试集:", ex.shape, ex.dtype, ey.shape, ey.dtype)
    print("像素范围:", tx.min(), tx.max())
    print("各类别样本数:", np.bincount(ty))
