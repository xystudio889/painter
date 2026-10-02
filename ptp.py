"""`.ptp` 工程包(.ptp = painter 工程, 本质是一个 zip)。

结构:
    (root)/
    ├─ train/<数据集单字>/<编号>.pnt
    ├─ test/<数据集单字>/<编号>.pnt
    └─ modelinfo.json

模型权重**不打进包里**: 模型统一用 `.pt` 传递, `modelinfo.json` 里只记录
"当时用的是哪个模型 / 什么结构", 需要时由用户自行导入 `.pt`。
"""

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

import pnt
from dataset import PARTITIONS, Workspace, is_valid_label
from preprocess import DEFAULT_MEAN, DEFAULT_STD, INPUT_CENTER, INPUT_MODES

MODELINFO_NAME = "modelinfo.json"
FORMAT = "ptp"
VERSION = 1
EXTENSION = ".ptp"
# 单张图片在包内的大小上限, 超过按异常处理(防止 zipbomb)
MAX_IMAGE_BYTES = 4 * 1024 * 1024

DEFAULT_TEST_CONFIG = {
    "epochs": 12, "batch_size": 32, "lr": 0.001,
    "augment": True, "warm_start": True, "seed": 0,
}


class PtPError(Exception):
    """`.ptp` 文件损坏或结构不符。"""


class LoadReport:
    """一次导入的统计与问题记录。"""

    def __init__(self):
        self.datasets = 0
        self.images = 0
        self.skipped: list[str] = []      # 跳过/损坏的条目说明
        self.warnings: list[str] = []     # 不致命但需要提示的问题

    @property
    def summary(self) -> str:
        text = f"数据集 {self.datasets} 个，图片 {self.images} 张"
        if self.skipped:
            text += f"，已跳过 {len(self.skipped)} 个无法读取的文件"
        return text

    def detail(self, limit: int = 12) -> str:
        parts = []
        if self.warnings:
            parts.append("说明：\n" + "\n".join(f"  · {w}" for w in self.warnings[:limit]))
            if len(self.warnings) > limit:
                parts.append(f"  … 另有 {len(self.warnings) - limit} 条说明")
        if self.skipped:
            parts.append("已跳过：\n" + "\n".join(f"  · {s}" for s in self.skipped[:limit]))
            if len(self.skipped) > limit:
                parts.append(f"  … 另有 {len(self.skipped) - limit} 个文件被跳过")
        return "\n".join(parts)


def default_test_config() -> dict:
    return dict(DEFAULT_TEST_CONFIG)


def default_modelinfo(arch: dict | None = None, labels: list[str] | None = None,
                      model: dict | None = None) -> dict:
    """生成一份完整的 modelinfo(所有键都在)。"""
    return {
        "format": FORMAT,
        "version": VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "preprocess": {
            "input_mode": INPUT_CENTER,
            "mean": float(DEFAULT_MEAN),
            "std": float(DEFAULT_STD),
            "canvas": None,
        },
        "partitions": {name: {} for name in PARTITIONS},
        "class_labels": list(labels or []),
        "arch": arch,
        "model": model,
        "test_config": default_test_config(),
    }


def normalize_modelinfo(info: dict | None) -> dict:
    """补齐缺失字段, 让读取端对旧文件/手改文件保持宽容。"""
    merged = default_modelinfo()
    if isinstance(info, dict):
        merged.update(info)
    pre = merged.get("preprocess")
    if not isinstance(pre, dict):
        pre = {}
    mode = pre.get("input_mode")
    merged["preprocess"] = {
        "input_mode": mode if mode in INPUT_MODES else INPUT_CENTER,
        "mean": float(pre.get("mean", DEFAULT_MEAN)),
        "std": float(pre.get("std", DEFAULT_STD)),
        "canvas": pre.get("canvas"),
    }
    if not isinstance(merged.get("class_labels"), list):
        merged["class_labels"] = []
    parts = merged.get("partitions")
    merged["partitions"] = parts if isinstance(parts, dict) else {n: {} for n in PARTITIONS}
    tconf = merged.get("test_config")
    if not isinstance(tconf, dict):
        tconf = {}
    merged["test_config"] = {**default_test_config(), **tconf}
    merged["format"] = FORMAT
    merged["version"] = VERSION
    return merged


def build_modelinfo(workspace: Workspace, canvas: int | None = None,
                    arch: dict | None = None, model: dict | None = None,
                    base_info: dict | None = None) -> dict:
    """从当前工作区生成 modelinfo(保留原文件里的额外字段)。"""
    info = normalize_modelinfo(base_info)
    info["saved"] = datetime.now().isoformat(timespec="seconds")
    if info.get("preprocess", {}).get("canvas") is None and canvas is not None:
        info["preprocess"]["canvas"] = int(canvas)
    info["partitions"] = workspace.index_map()
    info["class_labels"] = workspace.labels()
    if arch is not None:
        info["arch"] = arch
    if model is not None:
        info["model"] = model
    return info


def save_ptp(path, workspace: Workspace, canvas: int | None = None,
             arch: dict | None = None, model: dict | None = None,
             base_info: dict | None = None) -> Path:
    """把工作区打包成 `.ptp`。返回实际写入的路径。"""
    path = Path(path)
    if path.suffix.lower() != EXTENSION:
        path = path.with_suffix(EXTENSION)

    info = build_modelinfo(workspace, canvas=canvas, arch=arch, model=model,
                           base_info=base_info)
    payload = json.dumps(info, ensure_ascii=False, indent=2).encode("utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MODELINFO_NAME, payload)
        for part_name in PARTITIONS:
            partition = workspace.partition(part_name)
            for label in sorted(partition.labels()):
                dataset = partition.dataset(label)
                for index in sorted(dataset.indices()):
                    arcname = f"{part_name}/{label}/{index}{pnt.EXTENSION}"
                    zf.writestr(arcname, pnt.encode_pnt(dataset.ink(index)))
    return path


def read_modelinfo(path) -> dict:
    """只读 `modelinfo.json`(不读图片), 用于导入前的预览。"""
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as zf:
            if MODELINFO_NAME not in zf.namelist():
                raise PtPError(f"压缩包内缺少 {MODELINFO_NAME}，不是合法的 .ptp 文件")
            raw = zf.read(MODELINFO_NAME)
    except zipfile.BadZipFile as exc:
        raise PtPError(f"{path.name} 不是合法的 zip 压缩包：{exc}") from exc
    except OSError as exc:
        raise PtPError(f"无法读取 {path.name}：{exc}") from exc
    try:
        info = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PtPError(f"{MODELINFO_NAME} 不是合法的 JSON：{exc}") from exc
    if not isinstance(info, dict):
        raise PtPError(f"{MODELINFO_NAME} 的顶层必须为 JSON 对象")
    return normalize_modelinfo(info)


def _decode_entry(name: str, data: bytes, report: LoadReport):
    """把 zip 里的一个条目解成 (ink, mode) 或 None。"""
    ext = Path(name).suffix.lower()
    if ext == pnt.EXTENSION:
        size = pnt.declared_size(data)
        if size is None:
            report.skipped.append(f"`{name}` 不是合法的 .pnt 文件（魔数或头部错误）")
            return None
        try:
            return pnt.decode_pnt(data, name)
        except pnt.PntError as exc:
            report.skipped.append(f"`{name}` 读取失败：{exc}")
            return None
    # 兼容 plan B 的 PNG 素材
    from PIL import Image
    try:
        gray = np.asarray(Image.open(io.BytesIO(data)).convert("L"),
                          dtype=np.float32) / 255.0
    except OSError as exc:
        report.skipped.append(f"`{name}` 不是合法的图片：{exc}")
        return None
    return 1.0 - gray, INPUT_CENTER


def load_ptp(path) -> tuple[Workspace, dict, LoadReport]:
    """读取 `.ptp`, 返回 (工作区, modelinfo, 报表)。

    损坏的**单个**图片会被跳过并记进报表, 而不是整包失败 —— 用户的画作比
    一次严格的校验更宝贵。结构性错误(不是 zip / 没有 modelinfo)才抛异常。
    """
    path = Path(path)
    info = read_modelinfo(path)
    report = LoadReport()
    workspace = Workspace()

    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise PtPError(f"{path.name} 不是合法的 zip 压缩包：{exc}") from exc

    bad_dirs_reported: set[str] = set()
    strict_mode = info["preprocess"]["input_mode"]

    with zf:
        for entry in zf.infolist():
            name = entry.filename
            if entry.is_dir() or name == MODELINFO_NAME:
                continue
            parts = name.replace("\\", "/").split("/")
            if len(parts) != 3:
                report.skipped.append(
                    f"`{name}` 的层级不符合 <分区>/<数据集>/<编号>.pnt")
                continue
            part_name, label_dir, filename = parts
            if part_name not in PARTITIONS:
                report.skipped.append(
                    f"`{name}` 的分区名 `{part_name}` 不是 train/test")
                continue
            if not is_valid_label(label_dir):
                if label_dir not in bad_dirs_reported:
                    bad_dirs_reported.add(label_dir)
                    report.skipped.append(
                        f"目录名 `{label_dir}` 不是单个有效字符，整个目录已跳过")
                continue
            stem, dot, ext = filename.rpartition(".")
            if not dot or ("." + ext.lower()) not in pnt.KNOWN_EXTENSIONS:
                report.skipped.append(f"`{name}` 的扩展名不受支持")
                continue
            try:
                index = int(stem)
            except ValueError:
                report.skipped.append(f"`{name}` 的文件名不是数字编号")
                continue
            if index < 0:
                report.skipped.append(f"`{name}` 的编号为负数")
                continue

            if entry.file_size > MAX_IMAGE_BYTES:
                report.skipped.append(f"`{name}` 超过 4 MB，疑似异常，已跳过")
                continue
            try:
                data = zf.read(name)
            except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
                report.skipped.append(f"`{name}` 解压失败：{exc}")
                continue

            decoded = _decode_entry(name, data, report)
            if decoded is None:
                continue
            ink, mode = decoded
            if mode not in INPUT_MODES:
                mode = strict_mode
            workspace.partition(part_name).put(label_dir, index, ink)

    # 声明过但包里没有的图片
    declared = info.get("partitions")
    if isinstance(declared, dict):
        for part_name, datasets in declared.items():
            if part_name not in PARTITIONS or not isinstance(datasets, dict):
                continue
            for label, indices in datasets.items():
                if not isinstance(indices, list) or not is_valid_label(str(label)):
                    continue
                present = workspace.partition(part_name).dataset(str(label)).indices()
                missing = [i for i in indices if isinstance(i, int) and i not in present]
                if missing:
                    report.warnings.append(
                        f"{part_name}/{label} 中声明的 {len(missing)} 张图片在包内不存在")

    for part_name in PARTITIONS:
        partition = workspace.partition(part_name)
        report.datasets += len(partition.labels())
        report.images += sum(len(partition.dataset(label).indices())
                             for label in partition.labels())
    workspace.dirty = False
    return workspace, info, report
