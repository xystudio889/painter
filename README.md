# 手写数字识别 (MNIST + CNN + Tkinter)

用 PyTorch 在本地 MNIST 数据集上训练一个卷积神经网络,再由 Tkinter 图形界面
实时识别鼠标手写的数字。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `mnist_data.py` | 解析 MNIST 的 IDX 二进制文件,加载数据集 |
| `model.py` | `DigitCNN` 网络结构(2 层卷积 + 2 层全连接) |
| `train.py` | 训练脚本,保存最优模型为 `mnist_cnn.pt` |
| `app.py` | Tkinter 图形界面:书写、清空、实时识别、置信度展示 |
| `mnist/` | 原始数据集(4 个 `idx` 文件) |
| `mnist_cnn.pt` | 训练产物,模型权重 |
| `requirements.txt` | 依赖清单 |

## 环境要求

- Python 3.10+(开发环境为 3.14.6)
- `torch`(有 CUDA 会自动使用 GPU,没有也能用 CPU)
- `numpy`、`Pillow`
- `tkinter`(Windows 官方 Python 安装包自带,不是 pip 依赖)

安装依赖:

```
pip install -r requirements.txt
```

> 从 PyPI 默认安装的是 CPU 版 torch。要用 NVIDIA GPU 加速训练,请到
> <https://pytorch.org/get-started/locally/> 选择对应 CUDA 版本的安装命令。
> Linux 下若缺少 tkinter,执行 `sudo apt install python3-tk`。

## 使用步骤

1. 训练模型(已训练过可跳过):

   ```
   python train.py
   ```

   可选参数:`--epochs 5`、`--batch-size 128`、`--lr 1e-3`、`--cpu`、`--no-augment`。

   当前默认配置在测试集上的准确率约为 **99.2%**。

2. 启动界面:

   ```
   python app.py
   ```

3. 在左侧白色书写区按住鼠标左键写出 0~9 的数字,右侧会显示识别结果、
   置信度以及 10 个类别的概率分布。勾选"实时识别"后每落笔一次即刷新结果。

## 高 DPI 支持

界面在 4K / 高分屏以及 Windows 缩放(125% / 150% / 200% …)下会自动适配:

- **开启进程 DPI 感知**:在创建窗口前调用
  `SetProcessDpiAwarenessContext(PER_MONITOR_AWARE_V2)`,失败时依次回退到
  `SetProcessDpiAwareness(2)` 和 `SetProcessDPIAware()`。不开启的话 Windows 会把
  96 DPI 渲染的位图整体拉伸,界面发虚。
- **按 DPI 等比缩放**:所有尺寸常量都是 96 DPI 下的"逻辑像素",运行时经
  `self._px()` 换算成物理像素,覆盖书写区画布、画笔半径与边缘柔化、控件内边距、
  分隔线间距、概率条宽高。
- **字号同步**:调用 `tk scaling` 设为 `dpi / 72`,使点值字号(以及 ttk 默认字体)
  随 DPI 等比放大。
- **跨显示器自适应**:窗口被拖到不同缩放比的显示器时会重排界面,并把已有墨迹
  用 LANCZOS 重采样到新尺寸,不会清空已写内容。状态栏会显示当前
  `显示缩放: 100% (96 DPI)`。

预处理的裁剪、缩放、重心居中都是比例运算,因此画布尺寸变化不会影响识别精度
(已在 280 / 350 / 560 / 840 px 下验证)。

## 识别流程说明

```
鼠标笔画 → 280x280 灰度墨迹图 → 裁剪外接框 → LANCZOS 缩放至 20x20
        → 按重心居中放入 28x28 → (x-0.1307)/0.3081 归一化 → CNN → softmax
```

这一套预处理(尤其是"裁剪 + 按重心居中")是让手写数字能对齐 MNIST 训练分布、
从而获得高准确率的关键。

## 写作小技巧

- 数字写得大一些、笔画粗一些,并与 MNIST 风格一致(白底黑字自动反色)。
- 尽量写在画布中央,虽然预处理会自动居中,但不要超出画布边界。
