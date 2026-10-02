"""内存中的"自定义训练模式"工作区。

三个概念:
    分区 (partition) —— 固定两个: train / test
    数据集 (dataset) —— 名字就是**单个字符的标签**, 例如 "a" / "5"
    图片 (sample)    —— 画板上画出来的图, 编号由程序按数字递增分配

图片数据本身以 `numpy.ndarray` (H, W) float32 0~1 的墨迹形式放在内存里。
磁盘上的形态只有 `.ptp` 包(用户主动保存时), 见 `ptp.py`。
"""

import numpy as np

PARTITION_TRAIN = "train"
PARTITION_TEST = "test"
PARTITIONS = (PARTITION_TRAIN, PARTITION_TEST)

PARTITION_NAMES = {
    PARTITION_TRAIN: "训练集",
    PARTITION_TEST: "测试集",
}


def is_valid_label(text: str) -> bool:
    """数据集名必须是**单个非空白字符** —— 它就是分类标签, 也是匹配度列表里显示的名字。"""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if len(stripped) != 1:
        return False
    return not stripped.isspace()


def validate_label(text: str) -> str:
    """校验并规范化数据集名, 不合法时抛 `ValueError` 并带上可读原因。"""
    if not isinstance(text, str):
        raise ValueError("数据集名称必须为字符串")
    stripped = text.strip()
    if stripped == "":
        raise ValueError("数据集名称不能为空")
    if len(stripped) > 1:
        raise ValueError(
            f"数据集名称必须为单个字符，「{text}」包含 {len(stripped)} 个字符")
    if stripped.isspace():
        raise ValueError("数据集名称不能为空白字符")
    return stripped


def next_number(existing) -> int:
    """从 0 开始找第一个没被占用的编号。"""
    used = {int(i) for i in existing}
    number = 0
    while number in used:
        number += 1
    return number


class DigitSet:
    """一个数据集: 单字标签 + 一批按编号索引的图片。"""

    def __init__(self, label: str):
        self.label = label
        self._samples: dict[int, np.ndarray] = {}

    def put(self, number: int, ink: np.ndarray) -> None:
        arr = np.asarray(ink, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"图片必须为二维矩阵，实际收到 {arr.shape}")
        self._samples[int(number)] = arr

    def ink(self, number: int) -> np.ndarray:
        return self._samples[int(number)]

    def has(self, number: int) -> bool:
        return int(number) in self._samples

    def remove(self, number: int) -> bool:
        return self._samples.pop(int(number), None) is not None

    def indices(self) -> list[int]:
        return sorted(self._samples)

    def numbers_sorted(self) -> list[int]:
        return sorted(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __contains__(self, number) -> bool:
        return int(number) in self._samples

    def next_number(self) -> int:
        """下一个可用编号。删除过图片后可能复用空出来的号, 这是刻意的。"""
        return next_number(self._samples)


class Partition:
    """一个分区(train / test), 里面装着若干数据集。"""

    def __init__(self, name: str):
        self.name = name
        self._datasets: dict[str, DigitSet] = {}

    # ----------------------------- 数据集 ----------------------------- #
    def dataset(self, label: str) -> DigitSet:
        """取数据集, 不存在就**创建**(惰性创建让调用处更简洁)。"""
        existing = self._datasets.get(label)
        if existing is None:
            existing = DigitSet(label)
            self._datasets[label] = existing
        return existing

    def find(self, label: str) -> DigitSet | None:
        return self._datasets.get(label)

    def create(self, label: str) -> DigitSet:
        if label in self._datasets:
            raise ValueError(f"数据集「{label}」已存在")
        return self.dataset(label)

    def remove_dataset(self, label: str) -> bool:
        return self._datasets.pop(label, None) is not None

    def put(self, label: str, number: int, ink: np.ndarray) -> None:
        """往某个数据集写一张图(数据集不存在会自动创建)。"""
        self.dataset(label).put(number, ink)

    def ink(self, label: str, number: int) -> np.ndarray:
        """读某个数据集里的一张图。"""
        return self.dataset(label).ink(number)

    def labels(self) -> list[str]:
        return sorted(self._datasets)

    def image_count(self) -> int:
        return sum(len(d) for d in self._datasets.values())

    def __contains__(self, label) -> bool:
        return label in self._datasets

    def __len__(self) -> int:
        return len(self._datasets)


class Workspace:
    """整个工作区: 两个分区 + 待保存标记 + 来源信息。"""

    def __init__(self):
        self._partitions = {name: Partition(name) for name in PARTITIONS}
        self.dirty = False
        self.path = None           # 最近一次保存/打开的 .ptp 路径
        self.modelinfo: dict | None = None   # 打开 .ptp 时读到的原始 modelinfo
        self.sample_mode = "center"          # 保存 .pnt 时记录的输入模式

    # ----------------------------- 分区 ----------------------------- #
    def partition(self, name: str) -> Partition:
        try:
            return self._partitions[name]
        except KeyError:
            raise KeyError(f"未知分区「{name}」，仅支持 {PARTITIONS}") from None

    def labels(self) -> list[str]:
        """两个分区里出现过的所有数据集名(排序后), 也就是类别标签表。"""
        names: set[str] = set()
        for partition in self._partitions.values():
            names.update(partition.labels())
        return sorted(names)

    def train_labels(self) -> list[str]:
        """只出现在训练集里的数据集名(训练真正要用的类别表)。"""
        return self._partitions[PARTITION_TRAIN].labels()

    def index_map(self) -> dict[str, dict[str, list[int]]]:
        """写进 modelinfo.json 的 partitions 字段。"""
        return {
            name: {label: self._partitions[name].dataset(label).indices()
                   for label in self._partitions[name].labels()}
            for name in PARTITIONS
        }

    def image_count(self, partition: str | None = None) -> int:
        if partition is not None:
            return self.partition(partition).image_count()
        return sum(p.image_count() for p in self._partitions.values())

    def is_empty(self) -> bool:
        return all(len(p) == 0 for p in self._partitions.values())

    # ----------------------------- 增删 ----------------------------- #
    def remove_dataset(self, label: str) -> bool:
        """从**两个分区**一起删除同名数据集(标签消失, 类别表随之变化)。"""
        removed = False
        for partition in self._partitions.values():
            removed = partition.remove_dataset(label) or removed
        return removed

    def add_sample(self, partition: str, label: str, ink: np.ndarray) -> int:
        """往指定数据集加一张图, 自动分配编号。返回编号。"""
        dataset = self.partition(partition).dataset(label)
        number = dataset.next_number()
        dataset.put(number, ink)
        self.dirty = True
        return number

    # ----------------------------- 标记 ----------------------------- #
    def mark_dirty(self) -> None:
        self.dirty = True

    def mark_clean(self, path=None) -> None:
        self.dirty = False
        if path is not None:
            self.path = path

    # ----------------------------- 摘要 ----------------------------- #
    def describe(self) -> str:
        parts = []
        for name in PARTITIONS:
            partition = self._partitions[name]
            parts.append(f"{PARTITION_NAMES[name]} {len(partition)} 类 / "
                         f"{partition.image_count()} 张")
        return "    ".join(parts)

    def train_class_counts(self) -> dict[str, int]:
        train = self._partitions[PARTITION_TRAIN]
        return {label: len(train.dataset(label)) for label in train.labels()}

    def empty_train_labels(self) -> list[str]:
        """训练集里一张图都没有的数据集名。"""
        return [label for label, count in self.train_class_counts().items() if count == 0]
