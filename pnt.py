"""自制图片格式 `.pnt` —— 保留画板的墨迹深浅。

为什么不用 PNG: 这个格式要存的是画板的原始灰度(**保存时刻的位图**),
不做 28x28 裁剪/居中; 训练和推理时才按 input_mode 走同一套预处理, 两者
绝不分叉。自制格式没有压缩/色彩空间歧义, 头部自描述宽高, 也便于向后兼容。

文件布局(小端):
    offset 0   : b"PNT1"              魔数
    offset 4   : uint16 version = 1
    offset 6   : uint32 width
    offset 10  : uint32 height
    offset 14  : uint8  input_mode    (0=center, 1=raw)
    offset 15  : 6 字节保留, 写入 0
    offset 21  : width*height 字节, uint8 灰度, 墨迹深浅 = round(ink*255)

读取端还会回退到同名 `.png`(见 `load_ink`), 保证 plan B 的素材仍能打开。
"""

import struct
from pathlib import Path

import numpy as np
from PIL import Image

MAGIC = b"PNT1"
VERSION = 1
HEADER_SIZE = 21
MODE_CENTER = 0
MODE_RAW = 1
_EXTENSION = ".pnt"
EXTENSION = _EXTENSION
# 枚举数据集目录时同时考虑的扩展名(按优先级)
KNOWN_EXTENSIONS = (".pnt", ".png")


def mode_to_code(mode: str) -> int:
    return MODE_RAW if mode == "raw" else MODE_CENTER


def code_to_mode(code: int) -> str:
    return "raw" if code == MODE_RAW else "center"


# 兼容旧名(内部使用)
_mode_to_code = mode_to_code
_code_to_mode = code_to_mode


class PntError(Exception):
    """`.pnt` 文件损坏或格式不符。"""


def encode_pnt(ink: np.ndarray, mode: str = "center") -> bytes:
    """把画板墨迹编码成 `.pnt` 字节流(写 zip 时不必先落盘)。"""
    arr = np.asarray(ink, dtype=np.float32)
    if arr.ndim != 2:
        raise PntError(f"墨迹必须为二维矩阵，实际收到 {arr.shape}")
    h, w = arr.shape
    if not (1 <= h <= 0xFFFF and 1 <= w <= 0xFFFF):
        raise PntError(f"画布尺寸超出该文件格式的表示范围：{w}×{h}")
    data = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    header = MAGIC + struct.pack("<HIIB", VERSION, w, h, mode_to_code(mode)) + b"\0" * 6
    return header + np.ascontiguousarray(data).tobytes()


def decode_pnt(data: bytes, name: str = "内存数据") -> tuple[np.ndarray, str]:
    """从字节流解析 `.pnt`, 返回 (ink, input_mode)。"""
    if len(data) < HEADER_SIZE:
        raise PntError(f"{name} 文件过小，不是合法的 .pnt 文件")
    if data[:4] != MAGIC:
        raise PntError(f"{name} 的魔数不是 PNT1，可能不是 .pnt 文件")
    version, w, h, mode_code = struct.unpack("<HIIB", data[4:15])
    if version > VERSION:
        raise PntError(f"{name} 的版本 {version} 高于本程序支持的 {VERSION}")
    expected = HEADER_SIZE + w * h
    if len(data) != expected:
        raise PntError(
            f"{name} 的数据长度不符：头部声明 {w}×{h}，应为 {expected} 字节，"
            f"实际为 {len(data)} 字节")
    raw = data[HEADER_SIZE:]
    ink = np.frombuffer(raw, dtype=np.uint8).reshape(h, w).astype(np.float32) / 255.0
    return ink, code_to_mode(mode_code)


def declared_size(data: bytes) -> tuple[int, int] | None:
    """只读头部, 返回 (宽, 高); 不是 `.pnt` 时返回 None。"""
    if len(data) < HEADER_SIZE or data[:4] != MAGIC:
        return None
    _version, w, h, _mode = struct.unpack("<HIIB", data[4:15])
    return w, h


def save_pnt(path, ink: np.ndarray, mode: str = "center") -> Path:
    """把画板墨迹写成 `.pnt`。返回实际写入的路径。

    ink: (H, W) float32, 0~1, 越大越黑。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(encode_pnt(ink, mode))
    return path


def _load_png_ink(path: Path) -> np.ndarray:
    """读取旧素材 PNG, 自动判断是"黑墨白底"还是"白墨黑底"。"""
    img = Image.open(path).convert("L")
    gray = np.asarray(img, dtype=np.float32) / 255.0
    if min(gray.shape) < 2:
        # 太小了, 退回按"深色是墨"处理
        return 1.0 - gray
    corners = np.concatenate([
        gray[:1, :].ravel(), gray[-1:, :].ravel(),
        gray[:, :1].ravel(), gray[:, -1:].ravel(),
    ])
    background = float(corners.mean())
    # 白底黑字(照片/截图)PIL 里字是暗的 -> 墨迹 = 1 - 灰度
    # 黑底白字(画板导出)字是亮的 -> 墨迹 = 灰度
    return (1.0 - gray) if background > 0.5 else gray


def load_ink(path, mode_out: list | None = None) -> np.ndarray:
    """读取 `.pnt`(或回退 `.png`), 返回 (H, W) float32 墨迹。

    mode_out: 传入一个 list 时, 会把文件里记录的 input_mode 追加进去。
    """
    path = Path(path)
    if path.suffix.lower() == _EXTENSION:
        ink, mode = read_pnt(path)
    else:
        ink, mode = _load_png_ink(path), "center"
    if mode_out is not None:
        mode_out.append(mode)
    return ink


def read_pnt(path) -> tuple[np.ndarray, str]:
    """读取 `.pnt` 文件, 返回 (ink, input_mode)。"""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PntError(f"无法读取 {path.name}: {exc}") from exc
    return decode_pnt(data, path.name)


def pnt_thumbnail(ink: np.ndarray, size: int) -> np.ndarray:
    """把墨迹缩成 size x size 灰度(0~255, 墨迹=深色), 用于列表缩略图。"""
    img = Image.fromarray(((1.0 - np.asarray(ink, dtype=np.float32)) * 255.0)
                          .astype(np.uint8), mode="L")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def list_dataset_files(directory) -> list[tuple[int, Path]]:
    """列出一个数据集目录下的样本, 返回按编号升序的 [(编号, 路径)]。

    同名时 `.pnt` 优先于 `.png`; 文件名不是纯数字的跳过。
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found: dict[int, Path] = {}
    for child in directory.iterdir():
        if not child.is_file():
            continue
        ext = child.suffix.lower()
        if ext not in KNOWN_EXTENSIONS:
            continue
        try:
            index = int(child.stem)
        except ValueError:
            continue
        if index < 0:
            continue
        previous = found.get(index)
        if previous is None or (previous.suffix.lower() == ".png" and ext == ".pnt"):
            found[index] = child
    return sorted(found.items())


def next_index(directory, existing) -> int:
    """给数据集目录挑一个没被占用的最小可用编号(从 0 开始)。"""
    used = {i for i, _ in list_dataset_files(directory)}
    used.update(int(i) for i in existing)
    index = 0
    while index in used:
        index += 1
    return index


def sample_path(directory, index: int) -> Path:
    return Path(directory) / f"{index}{_EXTENSION}"
