"""纯命令行自测 —— 不需要 Tk 窗口。

    python selftest.py

覆盖 .pnt / .ptp 格式往返、架构与默认网络的等价性、训练流程、.pt 自描述往返,
以及老 mnist_cnn.pt 的兼容加载。

临时文件写在**工作目录**下的 `.selftest_tmp/`, 跑完就删, 不会污染项目。
"""

import contextlib
import shutil
import sys
import traceback
import uuid
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# 中文输出在 cp936 控制台下会乱码, 统一成 UTF-8
for stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import arch as arch_mod          # noqa: E402
import pnt                       # noqa: E402
import ptp                       # noqa: E402
import trainer                   # noqa: E402
from dataset import (             # noqa: E402
    PARTITION_TEST, PARTITION_TRAIN, Workspace, is_valid_label, next_number,
    validate_label,
)
from model import DigitCNN       # noqa: E402
from preprocess import INPUT_RAW  # noqa: E402

TMP_ROOT = ROOT / ".selftest_tmp"

PASSED = 0
FAILED = 0


@contextlib.contextmanager
def workdir():
    """在**工作目录内**开一个临时目录。

    两个约束: 工作目录之外写入会被沙箱拒绝; 而 `tempfile.mkdtemp` 建出来的
    目录在本环境下同样写不进去, 所以这里手工 `mkdir` + 唯一名字。
    """
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TMP_ROOT / f"t{uuid.uuid4().hex[:10]}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        with contextlib.suppress(OSError):
            TMP_ROOT.rmdir()


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [OK]   {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name}" + (f"  <- {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def fake_ink(size: int = 60, seed: int = 0, kind: str = "rect") -> np.ndarray:
    """造一张假画板: 有柔化边缘的墨迹, 便于验证深浅信息是否被保留。"""
    rng = np.random.default_rng(seed)
    canvas = np.zeros((size, size), dtype=np.float32)
    y0, x0 = size // 5, size // 4
    y1, x1 = int(size * 0.85), int(size * 0.8)
    if kind == "rect":
        canvas[y0:y1, x0:x1] = 0.9
    elif kind == "diag":
        for i in range(size):
            canvas[min(size - 1, i), :] = 0.8
    else:
        canvas[y0:y1, x0:x1] = np.linspace(0.1, 1.0, x1 - x0, dtype=np.float32)
    # 加一点平滑边缘(模拟笔尖柔化)
    from PIL import Image
    img = Image.fromarray((canvas * 255).astype(np.uint8), mode="L")
    canvas = np.asarray(img.resize((size, size), Image.LANCZOS), dtype=np.float32) / 255.0
    canvas += rng.random((size, size), dtype=np.float32) * 0.02
    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
def test_labels() -> None:
    section("数据集名(单字标签)校验")
    check("单字字母合法", is_valid_label("a"))
    check("单个汉字合法", is_valid_label("猫"))
    check("单个数字合法", is_valid_label("7"))
    check("带空格包裹的单个字符合法", is_valid_label(" a "))
    check("多字符不合法", not is_valid_label("ab"))
    check("空串不合法", not is_valid_label(""))
    check("纯空白不合法", not is_valid_label("   "))
    try:
        validate_label("ab")
        check("validate_label 拒绝多字符", False)
    except ValueError:
        check("validate_label 拒绝多字符", True)
    check("编号从 0 开始", next_number(set()) == 0)
    check("跳过已占用编号", next_number({0, 1, 3}) == 2)


def test_pnt() -> None:
    section(".pnt 往返(保留墨迹深浅)")
    with workdir() as tmp:
        tmp = Path(tmp)
        ink = fake_ink(60, seed=1, kind="grad")
        target = tmp / "0.pnt"
        pnt.save_pnt(target, ink, "raw")
        check("文件已写出", target.exists())

        loaded, mode = pnt.read_pnt(target)
        check("尺寸一致", loaded.shape == ink.shape, f"{loaded.shape} vs {ink.shape}")
        check("输入模式保留", mode == "raw", mode)
        worst = float(np.abs(loaded - ink).max())
        check("量化误差 < 1/255", worst <= 1.0 / 255.0 + 1e-6, f"最大偏差 {worst:.6f}")
        check("不是纯二值(深浅信息保留)",
              len(np.unique(np.rint(loaded * 255).astype(int))) > 8,
              f"灰阶数 {len(np.unique(np.rint(loaded * 255).astype(int)))}")

        # 空图 / 单像素 / 非 28x28 尺寸
        for name, sample in (("空图", np.zeros((8, 8), np.float32)),
                             ("单像素", np.zeros((5, 5), np.float32)),
                             ("1x1", np.ones((1, 1), np.float32))):
            if name == "单像素":
                sample[2, 2] = 1.0
            path = tmp / f"{name}.pnt"
            pnt.save_pnt(path, sample)
            back, _ = pnt.read_pnt(path)
            check(f"{name} 往返一致", back.shape == sample.shape
                  and float(np.abs(back - sample).max()) < 1e-6)

        # 头部公开接口
        data = target.read_bytes()
        check("declared_size 解析正确", pnt.declared_size(data) == (60, 60))
        encoded = pnt.encode_pnt(ink)
        decoded, _ = pnt.decode_pnt(encoded, "内存")
        check("encode/decode 内存往返", np.allclose(decoded, loaded))

        # 损坏文件必须报错而不是静默
        bad = tmp / "bad.pnt"
        bad.write_bytes(b"NOPE" + b"\0" * 40)
        try:
            pnt.read_pnt(bad)
            check("坏魔数报错", False)
        except pnt.PntError:
            check("坏魔数报错", True)

        trunc = tmp / "trunc.pnt"
        trunc.write_bytes(encoded[: len(encoded) - 10])
        try:
            pnt.read_pnt(trunc)
            check("长度不符报错", False)
        except pnt.PntError:
            check("长度不符报错", True)

        # PNG 兼容退路
        from PIL import Image
        png = tmp / "legacy.png"
        Image.fromarray(((1.0 - ink) * 255).astype(np.uint8), mode="L").save(png)
        png_ink = pnt.load_ink(png)
        check("PNG 退路尺寸正确", png_ink.shape == ink.shape)
        check("PNG 退路深浅方向正确", float(png_ink.max()) > 0.5)


def build_workspace() -> Workspace:
    """造一个覆盖两个分区、两个数据集的工作区。"""
    ws = Workspace()
    ws.partition(PARTITION_TRAIN).put("a", 0, fake_ink(60, 1, "rect"))
    ws.partition(PARTITION_TRAIN).put("a", 1, fake_ink(60, 2, "rect"))
    ws.partition(PARTITION_TRAIN).put("b", 0, fake_ink(60, 3, "diag"))
    ws.partition(PARTITION_TEST).put("a", 0, fake_ink(60, 4, "rect"))
    ws.partition(PARTITION_TEST).put("b", 0, fake_ink(60, 5, "diag"))
    return ws


def test_ptp() -> None:
    section(".ptp 打包 / 解包")
    with workdir() as tmp:
        tmp = Path(tmp)
        ws = build_workspace()
        target = tmp / "proj" / "demo"       # 故意不给扩展名
        out = ptp.save_ptp(target, ws, canvas=280, arch=arch_mod.default_arch(2),
                           model={"name": "自制", "pt": None, "labels": ["a", "b"]})
        check("自动补上 .ptp 扩展名", out.name == "demo.ptp", out.name)

        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
        expected = {"modelinfo.json", "train/a/0.pnt", "train/a/1.pnt", "train/b/0.pnt",
                    "test/a/0.pnt", "test/b/0.pnt"}
        check("zip 结构与规定完全一致", names == expected,
              f"多: {sorted(names - expected)} 缺: {sorted(expected - names)}")

        info = ptp.read_modelinfo(out)
        check("modelinfo 记录分区索引", info["partitions"] == {
            "train": {"a": [0, 1], "b": [0]}, "test": {"a": [0], "b": [0]}},
            str(info["partitions"]))
        check("modelinfo 记录类别标签", info["class_labels"] == ["a", "b"])
        check("modelinfo 冗余保存了模型引用", info["model"]["name"] == "自制")

        restored, info2, report = ptp.load_ptp(out)
        check("解包无跳过", report.skipped == [], str(report.skipped))
        check("解包计数正确", (report.datasets, report.images) == (4, 5),
              f"数据集 {report.datasets} 张数 {report.images}")
        check("每个分区各自的数据集都回来了", restored.index_map() == ws.index_map(),
              str(restored.index_map()))
        ok = True
        for part in (PARTITION_TRAIN, PARTITION_TEST):
            for label in ws.partition(part).labels():
                src = ws.partition(part).dataset(label)
                dst = restored.partition(part).dataset(label)
                for number in src.indices():
                    if not np.allclose(src.ink(number), dst.ink(number), atol=1 / 255):
                        ok = False
        check("每张图的墨迹都还原(含深浅)", ok)
        check("解包后 dirty=False", restored.dirty is False)

        # 损坏包
        broken = tmp / "no_info.ptp"
        with zipfile.ZipFile(broken, "w") as zf:
            zf.writestr("train/a/0.pnt", pnt.encode_pnt(fake_ink(20)))
        try:
            ptp.load_ptp(broken)
            check("缺 modelinfo.json 报错", False)
        except ptp.PtPError:
            check("缺 modelinfo.json 报错", True)

        notzip = tmp / "notzip.ptp"
        notzip.write_bytes(b"this is not a zip")
        try:
            ptp.load_ptp(notzip)
            check("非 zip 报错", False)
        except ptp.PtPError:
            check("非 zip 报错", True)

        # 单个坏图片只跳过, 不整包失败
        partial = tmp / "partial.ptp"
        with zipfile.ZipFile(partial, "w") as zf:
            zf.writestr("modelinfo.json", b'{"format":"ptp","version":1}')
            zf.writestr("train/a/0.pnt", pnt.encode_pnt(fake_ink(20)))
            zf.writestr("train/a/1.pnt", b"broken-bytes")
            zf.writestr("train/zz/0.pnt", pnt.encode_pnt(fake_ink(20)))
            zf.writestr("elsewhere/a/0.pnt", pnt.encode_pnt(fake_ink(20)))
            zf.writestr("bogus/0.pnt", pnt.encode_pnt(fake_ink(20)))
        ws2, _, rep2 = ptp.load_ptp(partial)
        check("坏图片被跳过而不是整包失败",
              ws2.partition(PARTITION_TRAIN).dataset("a").indices() == [0])
        check("跳过项有记录", len(rep2.skipped) >= 3, str(rep2.skipped))
        check("分区名非法被记录", any("分区名" in s for s in rep2.skipped))


def test_arch() -> None:
    section("架构描述 <-> 模型")
    spec = arch_mod.default_arch(10)
    check("默认架构本身合法", arch_mod.validate_arch(spec) == [],
          str([str(e) for e in arch_mod.validate_arch(spec)]))
    model = arch_mod.build_model(spec)
    hand = DigitCNN()
    check("参数量一致 (421642)", arch_mod.count_parameters(model) == 421642,
          str(arch_mod.count_parameters(model)))
    check("与手写 DigitCNN 参数量相同",
          arch_mod.count_parameters(model) == arch_mod.count_parameters(hand))
    check("state_dict 键名完全一致", list(model.state_dict()) == list(hand.state_dict()),
          str(list(model.state_dict())))
    check("张量形状完全一致",
          all(model.state_dict()[k].shape == hand.state_dict()[k].shape
              for k in hand.state_dict()))
    try:
        model.load_state_dict(hand.state_dict())
        check("可以装入手写版的权重", True)
    except RuntimeError as exc:
        check("可以装入手写版的权重", False, str(exc))

    x = torch.randn(4, 1, 28, 28)
    check("默认架构输出 (N,10)", tuple(model(x).shape) == (4, 10))
    check("predict_proba 概率和为 1",
          bool(torch.allclose(model.predict_proba(x).sum(dim=1),
                              torch.ones(4), atol=1e-5)))

    # 合法但更复杂的结构
    big = {
        "input_mode": INPUT_RAW, "input_shape": [1, 28, 28], "num_classes": 3,
        "cards": [
            {"type": "conv", "channels": 8, "kernel": 3, "padding": 1, "activation": "relu"},
            {"type": "pool", "window": 2},
            {"type": "conv", "channels": 16, "kernel": 5, "padding": 2, "activation": "none"},
            {"type": "pool", "window": 2},
            {"type": "fc", "units": 64, "activation": "relu", "dropout": 0.2},
            {"type": "fc", "units": 32, "activation": "none", "dropout": 0.0},
        ],
    }
    check("多张 FC 卡片的结构合法", arch_mod.validate_arch(big) == [],
          str([str(e) for e in arch_mod.validate_arch(big)]))
    big_model = arch_mod.build_model(big)
    check("复杂结构输出类别数正确", tuple(big_model(x).shape) == (4, 3))

    # 纯 FC(没有卷积)也应该能跑
    mlp = {"input_mode": INPUT_RAW, "input_shape": [1, 28, 28], "num_classes": 2,
           "cards": [{"type": "fc", "units": 16, "activation": "relu", "dropout": 0.1}]}
    check("纯 FC 结构合法", arch_mod.validate_arch(mlp) == [])
    check("纯 FC 结构输出正确", tuple(arch_mod.build_model(mlp)(x).shape) == (4, 2))

    # 各种非法结构
    bad_cases = {
        "池化把图压成 0": {"num_classes": 2, "input_mode": "center",
                      "input_shape": [1, 28, 28],
                      "cards": [{"type": "pool", "window": 4}] * 3},
        "通道数为 0": {"num_classes": 2, "input_mode": "center",
                    "input_shape": [1, 28, 28],
                    "cards": [{"type": "conv", "channels": 0, "kernel": 3, "padding": 1}]},
        "未知卡片类型": {"num_classes": 2, "input_mode": "center",
                    "input_shape": [1, 28, 28], "cards": [{"type": "transformer"}]},
        "没有卡片": {"num_classes": 2, "input_mode": "center",
                   "input_shape": [1, 28, 28], "cards": []},
        "类别数太少": {"num_classes": 1, "input_mode": "center",
                   "input_shape": [1, 28, 28],
                   "cards": [{"type": "fc", "units": 8}]},
        "输入模式非法": {"num_classes": 2, "input_mode": "huge",
                    "input_shape": [1, 28, 28], "cards": [{"type": "fc", "units": 8}]},
        "卷积核非法": {"num_classes": 2, "input_mode": "center",
                   "input_shape": [1, 28, 28],
                   "cards": [{"type": "conv", "channels": 4, "kernel": 7, "padding": 0}]},
    }
    for name, spec_bad in bad_cases.items():
        errors = arch_mod.validate_arch(spec_bad)
        check(f"拒绝: {name}", len(errors) > 0)
        check(f"拒绝时可以说明原因: {name}", all(str(e) for e in errors))
    check("错误带卡片序号", any(e.index is not None
                          for e in arch_mod.validate_arch(bad_cases["通道数为 0"])))

    # normalize 行为
    check("空架构回落到默认",
          arch_mod.normalize_arch(None)["cards"] == arch_mod.default_arch()["cards"])
    check("normalize 保留已有卡片",
          arch_mod.normalize_arch(big)["cards"][0]["channels"] == 8)


def test_trainer() -> None:
    section("训练与预测")
    ws = Workspace()
    # 两类可分的合成数据: 竖条 vs 横条; 每类 6 张
    for index in range(6):
        vertical = np.zeros((60, 60), np.float32)
        vertical[8:52, 22:38] = 0.9
        vertical += np.random.default_rng(100 + index).random((60, 60),
                                                              dtype=np.float32) * 0.03
        ws.partition(PARTITION_TRAIN).put("v", index, np.clip(vertical, 0, 1))

        horizontal = np.zeros((60, 60), np.float32)
        horizontal[22:38, 8:52] = 0.9
        horizontal += np.random.default_rng(200 + index).random((60, 60),
                                                                dtype=np.float32) * 0.03
        ws.partition(PARTITION_TRAIN).put("h", index, np.clip(horizontal, 0, 1))
    ws.partition(PARTITION_TEST).put("v", 0, ws.partition(PARTITION_TRAIN).ink("v", 0))
    ws.partition(PARTITION_TEST).put("h", 0, ws.partition(PARTITION_TRAIN).ink("h", 0))

    spec = arch_mod.default_arch(2)
    spec["input_mode"] = INPUT_RAW          # 整幅图更利于合成数据
    config = trainer.TrainConfig(epochs=8, batch_size=8, lr=2e-3, augment=False, seed=0)
    seen: list[dict] = []
    result = trainer.train_model(ws, spec, config, on_progress=seen.append)
    check("训练成功", result.ok, result.message)
    check("回调有进度消息", len(seen) > 0)
    check("回调消息含 epoch 字段", all("epoch" in info for info in seen))
    check("返回了权重", result.state_dict is not None)
    check("标签表正确", result.labels == ["h", "v"], str(result.labels))
    check("记录了归一化统计量", 0 < result.mean < 1 and 0 < result.std < 2,
          f"mean={result.mean} std={result.std}")
    check("测试集通过率 = 1.0", result.accuracy == 1.0, str(result.accuracy))

    # 训练后的模型能正确区分两张图
    checkpoint = trainer.make_checkpoint(spec, result.labels, result.state_dict,
                                         result.mean, result.std, "合成模型",
                                         result.accuracy, result.epochs_done)
    with workdir() as tmp:
        path = Path(tmp) / "model.pt"
        trainer.save_checkpoint(path, checkpoint)
        check(".pt 已写出", path.exists() and path.stat().st_size > 0)
        info, model = trainer.load_checkpoint(path)
        check("自描述: 名称", info["name"] == "合成模型")
        check("自描述: 标签", info["labels"] == ["h", "v"])
        check("自描述: 架构", info["arch"]["cards"] == spec["cards"])
        check("自描述: 输入模式", info["input_mode"] == INPUT_RAW)
        check("自描述: 归一化参数", abs(info["mean"] - result.mean) < 1e-6)
        check("自描述: 通过率", abs((info["accuracy"] or 0) - result.accuracy) < 1e-6)
        check("不是 legacy", info["legacy"] is False)

        probs = trainer.predict_proba(model, ws.partition(PARTITION_TRAIN).ink("v", 0),
                                      info["input_mode"], info["mean"], info["std"])
        check("预测竖条 -> v", info["labels"][int(np.argmax(probs))] == "v",
              str(probs))
        probs = trainer.predict_proba(model, ws.partition(PARTITION_TRAIN).ink("h", 0),
                                      info["input_mode"], info["mean"], info["std"])
        check("预测横条 -> h", info["labels"][int(np.argmax(probs))] == "h",
              str(probs))
        check("空画板返回 None",
              trainer.predict_proba(model, np.zeros((60, 60), np.float32),
                                    info["input_mode"], info["mean"], info["std"]) is None)

        # 取消
        cancelled = trainer.train_model(ws, spec, trainer.TrainConfig(epochs=5),
                                        cancel=lambda: True)
        check("取消时不返回权重", cancelled.cancelled and cancelled.state_dict is None)
        check("取消时不报成功", not cancelled.ok)

    # 各种拒绝路径
    empty = Workspace()
    empty.partition(PARTITION_TRAIN).put("a", 0, fake_ink(30))
    try:
        trainer.train_model(empty, arch_mod.default_arch(2), trainer.TrainConfig(epochs=1))
        check("单类别时拒绝训练", False)
    except trainer.TrainError:
        check("单类别时拒绝训练", True)

    blank = Workspace()
    blank.partition(PARTITION_TRAIN).put("a", 0, np.zeros((30, 30), np.float32))
    blank.partition(PARTITION_TRAIN).put("b", 0, np.zeros((30, 30), np.float32))
    try:
        trainer.train_model(blank, arch_mod.default_arch(2), trainer.TrainConfig(epochs=1))
        check("全空白样本时拒绝训练", False)
    except trainer.TrainError:
        check("全空白样本时拒绝训练", True)


def test_real_mnist_model() -> None:
    section("老 mnist_cnn.pt 兼容加载")
    path = ROOT / "models" / "mnist_cnn.pt"       # 模型都集中在 models/ 下
    if not path.exists():
        path = ROOT / "mnist_cnn.pt"              # 兼容还摆在根目录的旧布局
    if not path.exists():
        print("  [skip] 没有 mnist_cnn.pt, 跳过")
        return
    try:
        info, model = trainer.load_checkpoint(path)
    except trainer.TrainError as exc:
        check("能加载裸 state_dict", False, str(exc))
        return
    check("能加载裸 state_dict", True)
    check("识别为 legacy", info["legacy"] is True)
    check("标签是 0-9", info["labels"] == [str(i) for i in range(10)], str(info["labels"]))
    check("沿用 MNIST 归一化",
          0.12 < info["mean"] < 0.14 and 0.30 < info["std"] < 0.32,
          f"{info['mean']}/{info['std']}")

    # 用真实 MNIST 数据抽 200 张测试, 准确率应该很高
    import mnist_data
    try:
        _tx, ty, ex, ey = mnist_data.load_mnist()
    except FileNotFoundError:
        print("  [skip] 没有 mnist 数据文件, 跳过准确率检查")
        return
    count = 200
    correct = 0
    for i in range(count):
        ink = ex[i].astype(np.float32) / 255.0
        probs = trainer.predict_proba(model, ink, info["input_mode"],
                                      info["mean"], info["std"])
        if probs is not None and int(np.argmax(probs)) == int(ey[i]):
            correct += 1
    accuracy = correct / count
    check("真 MNIST 子集准确率 > 90%", accuracy > 0.90, f"{accuracy:.1%}")


def test_image_e2e() -> None:
    section("真实 MNIST 图片 -> .pnt -> .ptp -> 训练 全链路")
    import mnist_data
    try:
        _tx, ty, ex, ey = mnist_data.load_mnist()
    except FileNotFoundError:
        print("  [skip] 没有 mnist 数据文件, 跳过")
        return

    ws = Workspace()
    # 挑两个数字, 各 10 张, 先放大到画板尺寸(280)的墨迹, 模拟用户画出来的
    for label in ("0", "1"):
        indices = [i for i in range(len(ey)) if int(ey[i]) == int(label)][:10]
        for position, i in enumerate(indices):
            small = ex[i].astype(np.float32) / 255.0
            upscaled = np.asarray(
                Image.fromarray((small * 255).astype(np.uint8), mode="L")
                .resize((168, 168), Image.LANCZOS), dtype=np.float32) / 255.0
            canvas = np.zeros((280, 280), np.float32)
            canvas[56:224, 56:224] = upscaled
            if position < 8:
                ws.partition(PARTITION_TRAIN).put(label, position, canvas)
            else:
                ws.partition(PARTITION_TEST).put(label, position - 8, canvas)

    with workdir() as tmp:
        tmp = Path(tmp)
        package = ptp.save_ptp(tmp / "e2e.ptp", ws, canvas=280,
                               arch=arch_mod.default_arch(2))
        restored, _info, report = ptp.load_ptp(package)
        check("全链路: 打包解包无跳过", report.skipped == [], str(report.skipped))
        check("全链路: 图片数一致",
              restored.image_count() == ws.image_count())

        config = trainer.TrainConfig(epochs=6, batch_size=8, lr=1e-3, augment=True, seed=0)
        result = trainer.train_model(restored, arch_mod.default_arch(2), config)
        check("全链路: 训练成功", result.ok, result.message)
        check("全链路: 测试集通过率 = 1.0", result.accuracy == 1.0, str(result.accuracy))

        model_path = tmp / "e2e.pt"
        trainer.save_checkpoint(model_path, trainer.make_checkpoint(
            arch_mod.default_arch(2), result.labels, result.state_dict,
            result.mean, result.std, "e2e", result.accuracy, result.epochs_done))
        info, model = trainer.load_checkpoint(model_path)
        correct = 0
        total = 0
        test_part = restored.partition(PARTITION_TEST)
        for label in test_part.labels():
            dataset = test_part.dataset(label)
            for number in dataset.indices():
                probs = trainer.predict_proba(model, dataset.ink(number),
                                              info["input_mode"], info["mean"], info["std"])
                total += 1
                if info["labels"][int(np.argmax(probs))] == label:
                    correct += 1
        check("全链路: 重新加载后预测全对", correct == total, f"{correct}/{total}")


def main() -> int:
    print(f"自测开始 (torch {torch.__version__}, python {sys.version.split()[0]})")
    suites = [test_labels, test_pnt, test_ptp, test_arch, test_trainer,
              test_real_mnist_model, test_image_e2e]
    crashed = 0
    for suite in suites:
        try:
            suite()
        except Exception:                        # noqa: BLE001 - 自测要看到全部细节
            crashed += 1
            print(f"  [CRASH] {suite.__name__} 抛出异常:")
            traceback.print_exc()
    print(f"\n{'=' * 52}")
    print(f"通过 {PASSED} 项, 失败 {FAILED} 项, 异常中断 {crashed} 个套件")
    if FAILED or crashed:
        print("结果: 未通过")
        return 1
    print("结果: 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
