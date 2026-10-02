"""手写数字识别 —— Tkinter 图形界面。

运行:
    python app.py

用鼠标在左侧黑色书写区画出 0~9 的数字, 程序会实时给出识别结果与置信度。
需要先用 train.py 训练出 mnist_cnn.pt。

支持高 DPI: 启动时开启进程 DPI 感知, 并按显示器 DPI 等比缩放界面; 窗口被拖到
不同缩放比的显示器时会自动重排(Windows 10+ Per-Monitor V2)。
"""

import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import torch
from PIL import Image, ImageTk

from mnist_data import MNIST_MEAN, MNIST_STD
from model import DigitCNN

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "mnist_cnn.pt"

# 下面所有尺寸都是 96 DPI 下的"逻辑像素", 运行时按显示器 DPI 等比放大到物理像素
BASE_DPI = 96.0
CANVAS_SIZE = 280      # 书写区边长
PEN_MIN, PEN_MAX = 4, 48   # 笔尖粗细(直径)逻辑像素范围
PEN_DEFAULT = 24           # 默认笔尖直径
SOFT_RATIO = 0.22          # 笔尖边缘柔化宽度 / 笔尖半径
ERASER_GAIN = 1.6          # 橡皮相对画笔的放大倍数
MARGIN = 2             # 标准化时四周留白
BOX_SIDE = 28 - 2 * MARGIN  # 缩放后内容边长

BAR_W, BAR_H, BAR_GAP = 210, 16, 6   # 概率条宽度 / 高度 / 间距
BAR_LABEL_W, BAR_TEXT_W = 32, 44     # 左侧数字编号宽度 / 右侧百分比文字宽度


# --------------------------------------------------------------------------- #
# 高 DPI 支持
# --------------------------------------------------------------------------- #
def enable_dpi_awareness() -> None:
    """开启进程 DPI 感知。

    必须在创建 Tk 窗口之前调用。否则 Windows 会把 96 DPI 渲染的界面直接
    按比例拉伸位图, 导致文字和线条发虚(模糊)。
    """
    if sys.platform != "win32":
        return
    import ctypes

    try:
        # Windows 10 1703+: Per-Monitor V2, 窗口跨显示器时能拿到各自 DPI
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass

    try:
        # Windows 8.1+: 1=System, 2=Per-Monitor; 返回 0 表示成功(S_FALSE 也算可用)
        if ctypes.windll.shcore.SetProcessDpiAwareness(2) in (0, 1):
            return
    except (AttributeError, OSError):
        pass

    try:
        # Windows Vista / 7
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def get_window_dpi(window: tk.Misc) -> float:
    """获取窗口所在显示器的 DPI, 失败时退回 Tk 自己的估算值。"""
    if sys.platform == "win32":
        import ctypes
        try:
            hwnd = window.winfo_id()
            for handle in (hwnd, ctypes.windll.user32.GetParent(hwnd)):
                if not handle:
                    continue
                dpi = ctypes.windll.user32.GetDpiForWindow(handle)
                if dpi:
                    return float(dpi)
        except (AttributeError, OSError, tk.TclError):
            pass
    try:
        return float(window.winfo_fpixels("1i"))
    except tk.TclError:
        return BASE_DPI


# --------------------------------------------------------------------------- #
# 图像预处理: 把书写区图像转成符合 MNIST 风格 (28x28, 黑底白字, 居中) 的张量
# --------------------------------------------------------------------------- #
def preprocess(arr: np.ndarray):
    """arr: (H, W) float32, 取值 0~1 表示墨迹浓度。返回 (28, 28) float32 或 None。"""
    if arr.max() < 0.05:
        return None

    ys, xs = np.nonzero(arr > 0.05)
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


def to_tensor(digit: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(digit).float()
    x = (x - MNIST_MEAN) / MNIST_STD
    return x.view(1, 1, 28, 28)


# --------------------------------------------------------------------------- #
# 主界面
# --------------------------------------------------------------------------- #
class DigitApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("手写数字识别")
        root.iconbitmap(ROOT / 'icon.ico')
        root.resizable(False, False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        # ---- 高 DPI: 探测当前显示器 DPI, 并让 Tk 的点值字号同步放大 ----
        self.dpi = get_window_dpi(root)
        self.scale = max(self.dpi, BASE_DPI) / BASE_DPI
        root.tk.call("tk", "scaling", self.dpi / 72.0)

        self.canvas_size = self._px(CANVAS_SIZE)
        self.arr = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
        self.last_xy = None
        self.last_probs = None

        self._build_ui()
        self._apply_metrics()
        self._load_model()
        self._render()

        # 窗口被拖到不同 DPI 的显示器时(Win10+ Per-Monitor V2)重新排版
        root.bind("<Configure>", self._on_configure)

    # --------------------------- 高 DPI 缩放 --------------------------- #
    def _px(self, value: float) -> int:
        """逻辑像素 -> 物理像素。"""
        return max(1, int(round(value * self.scale)))

    def _apply_metrics(self):
        """按当前缩放比设置所有以像素为单位的尺寸。可重复调用(DPI 变化时重排)。"""
        # 笔尖粗细来自滑条(逻辑像素直径), 换算成物理像素半径
        radius = float(self.pen_size.get()) / 2.0
        if self._is_eraser():
            radius *= ERASER_GAIN
        self.brush_radius = max(1.5, radius * self.scale)
        self.brush_soft = max(0.8, radius * SOFT_RATIO * self.scale)

        self.main.configure(padding=self._px(10))
        self.left.configure(padding=self._px(8))
        self.tools.pack_configure(pady=(0, self._px(8)))
        self.pen_scale.configure(length=self._px(150))
        self.right.configure(padding=(self._px(16), 0, 0, 0))

        self.canvas.configure(width=self.canvas_size, height=self.canvas_size,
                              highlightthickness=self._px(2))
        self.btns.pack_configure(pady=self._px(10))
        self.btn_clear.pack_configure(padx=(0, self._px(6)))
        self.btn_recognize.pack_configure(padx=(0, self._px(6)))
        self.chk_live.pack_configure(padx=(0, self._px(6)))

        bar_w = self._px(BAR_LABEL_W + BAR_W + 4 + BAR_TEXT_W)
        bar_h = 10 * (self._px(BAR_H) + self._px(BAR_GAP)) + self._px(4)
        self.bar_canvas.configure(width=bar_w, height=bar_h)
        self.bar_canvas.pack_configure(pady=self._px(4))

        self.sep.pack_configure(pady=self._px(10))
        self.result_lbl.pack_configure(pady=(0, self._px(4)))
        self.status_lbl.grid_configure(pady=self._px(8))
        self._draw_bars(self.last_probs)

    def _on_configure(self, event):
        if event.widget is not self.root:
            return
        self.root.after_idle(self._sync_dpi)

    def _sync_dpi(self):
        """DPI 发生变化(换显示器 / 改系统缩放)时重采样画布并重排界面。"""
        dpi = get_window_dpi(self.root)
        if abs(dpi - self.dpi) < 1.0:
            return
        self.dpi = dpi
        self.scale = max(dpi, BASE_DPI) / BASE_DPI
        self.root.tk.call("tk", "scaling", dpi / 72.0)

        new_size = self._px(CANVAS_SIZE)
        if new_size != self.canvas_size:
            # 把已有墨迹重采样到新的画布尺寸, 避免清空用户已写的内容
            img = Image.fromarray((self.arr * 255).astype(np.uint8), mode="L")
            self.arr = np.asarray(img.resize((new_size, new_size), Image.LANCZOS),
                                  dtype=np.float32) / 255.0
            self.canvas_size = new_size

        self._apply_metrics()
        self._render()
        self._update_status()
        self.recognize()

    # --------------------------- 工具栏 --------------------------- #
    def _set_tool(self, cursor: str):
        self.canvas.configure(cursor=cursor)
        # 橡皮通常比画笔稍粗, 切换后按新倍数刷新笔尖尺寸
        self._apply_metrics()

    def _is_eraser(self) -> bool:
        return self.tool_var.get() == "eraser"

    def _on_pen_change(self, _value):
        self.pen_lbl.configure(text=str(int(round(self.pen_size.get()))))
        self._apply_metrics()

    # --------------------------- 界面构建 --------------------------- #
    def _build_ui(self):
        self.main = ttk.Frame(self.root, padding=10)
        self.main.grid(row=0, column=0)

        # ---- 左: 书写区 ----
        self.left = ttk.LabelFrame(self.main, text="书写区", padding=8)
        self.left.grid(row=0, column=0, sticky="n")

        # ---- 工具栏: 画笔 / 橡皮 / 粗细滑条 ----
        self.tools = ttk.Frame(self.left)
        self.tools.pack(fill="x", pady=(0, 8))

        self.tool_var = tk.StringVar(value="brush")
        self.btn_brush = ttk.Radiobutton(
            self.tools, text=" 画笔 ", value="brush", variable=self.tool_var,
            style="Toolbutton", command=lambda: self._set_tool("crosshair"),
        )
        self.btn_brush.pack(side="left")
        self.btn_eraser = ttk.Radiobutton(
            self.tools, text=" 橡皮 ", value="eraser", variable=self.tool_var,
            style="Toolbutton", command=lambda: self._set_tool("dotbox"),
        )
        self.btn_eraser.pack(side="left")

        ttk.Separator(self.tools, orient="vertical").pack(
            side="left", fill="y", padx=self._px(8))
        ttk.Label(self.tools, text="粗细", font=("Segoe UI", 9)).pack(side="left")

        self.pen_size = tk.DoubleVar(value=PEN_DEFAULT)
        self.pen_scale = ttk.Scale(
            self.tools, from_=PEN_MIN, to=PEN_MAX, orient="horizontal",
            length=150, variable=self.pen_size, command=self._on_pen_change,
        )
        self.pen_scale.pack(side="left", padx=self._px(4))
        self.pen_lbl = ttk.Label(self.tools, text=str(PEN_DEFAULT), width=3,
                                 anchor="center", font=("Consolas", 9))
        self.pen_lbl.pack(side="left")

        # ---- 书写区画布 ----
        self.canvas = tk.Canvas(
            self.left, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="white", highlightthickness=2, highlightbackground="#444",
            cursor="crosshair",
        )
        self.canvas.pack()
        self.image_id = self.canvas.create_image(0, 0, anchor="nw")

        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # ---- 右: 结果区 ----
        self.right = ttk.Frame(self.main, padding=(16, 0, 0, 0))
        self.right.grid(row=0, column=1, sticky="n")

        ttk.Label(self.right, text="识别结果", font=("Segoe UI", 11)).pack(anchor="w")
        self.result_var = tk.StringVar(value="—")
        self.result_lbl = ttk.Label(
            self.right, textvariable=self.result_var, font=("Consolas", 56, "bold"),
            foreground="#0a6ebd", width=3, anchor="center",
        )
        self.result_lbl.pack(pady=(0, 4))

        self.conf_var = tk.StringVar(value="置信度: —")
        ttk.Label(self.right, textvariable=self.conf_var,
                  font=("Segoe UI", 10)).pack(anchor="w")

        self.sep = ttk.Separator(self.right, orient="horizontal")
        self.sep.pack(fill="x", pady=10)

        ttk.Label(self.right, text="各数字概率", font=("Segoe UI", 10)).pack(anchor="w")
        self.bar_canvas = tk.Canvas(
            self.right,
            width=BAR_LABEL_W + BAR_W + 4 + BAR_TEXT_W,
            height=10 * (BAR_H + BAR_GAP) + 4,
            highlightthickness=0,
        )
        self.bar_canvas.pack(pady=(4, 10))

        # ---- 按钮 ----
        self.btns = ttk.Frame(self.left)
        self.btns.pack(fill="x", pady=(10, 0))
        self.btn_clear = ttk.Button(self.btns, text="清空", command=self.clear)
        self.btn_clear.pack(side="left")
        self.btn_recognize = ttk.Button(self.btns, text="识别", command=self.recognize)
        self.btn_recognize.pack(side="left", padx=6)

        self.live = tk.BooleanVar(value=True)
        self.chk_live = ttk.Checkbutton(self.btns, text="实时识别", variable=self.live)
        self.chk_live.pack(side="left", padx=6)

        self.model_info = "正在加载模型…"
        self.status_var = tk.StringVar(value=self.model_info)
        self.status_lbl = ttk.Label(self.main, textvariable=self.status_var,
                                    font=("Segoe UI", 9), foreground="#666")
        self.status_lbl.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Label(self.main,
                  text="提示: 尽量把数字写在中间、笔画写粗一些",
                  font=("Segoe UI", 9), foreground="#999").grid(
            row=2, column=0, columnspan=2, sticky="w")

    def _update_status(self):
        self.status_var.set(
            f"{self.model_info}    显示缩放: {self.scale * 100:.0f}% ({self.dpi:.0f} DPI)"
        )

    def _draw_bars(self, probs):
        c = self.bar_canvas
        c.delete("all")
        label_x = self._px(14)
        bar_x = self._px(BAR_LABEL_W)
        bar_w = self._px(BAR_W)
        bar_h = self._px(BAR_H)
        gap = self._px(BAR_GAP)
        top = self._px(2)
        text_x = bar_x + bar_w + self._px(4)
        for i in range(10):
            y = i * (bar_h + gap) + top
            c.create_text(label_x, y + bar_h / 2, text=str(i),
                          font=("Consolas", 10, "bold"))
            c.create_rectangle(bar_x, y, bar_x + bar_w, y + bar_h,
                               outline="#c8c8c8", fill="#f4f4f4")
            if probs is not None:
                p = float(probs[i])
                w = max(0.0, min(1.0, p)) * bar_w
                color = "#0a6ebd" if p >= 0.5 else "#7fb2dd"
                if w > 0:
                    c.create_rectangle(bar_x, y, bar_x + w, y + bar_h,
                                       outline="", fill=color)
                if p >= 0.01:
                    c.create_text(text_x, y + bar_h / 2,
                                  text=f"{p * 100:4.1f}%", anchor="w",
                                  font=("Consolas", 8))

    # --------------------------- 模型加载 --------------------------- #
    def _load_model(self):
        if not MODEL_PATH.exists():
            self.model_info = f"未找到模型文件 {MODEL_PATH.name}"
            self._update_status()
            messagebox.showwarning(
                "缺少模型",
                f"未找到模型文件:\n{MODEL_PATH}\n\n"
                "请先在终端运行以下命令训练模型:\n\n    python train.py",
            )
            return
        model = DigitCNN()
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(self.device).eval()
        self.model = model
        size_kb = MODEL_PATH.stat().st_size / 1024
        self.model_info = (
            f"模型: {MODEL_PATH.name} ({size_kb:.0f} KB)    "
            f"设备: {self.device.type.upper()}"
        )
        self._update_status()

    # --------------------------- 绘制逻辑 --------------------------- #
    def _stamp(self, x: float, y: float):
        r = self.brush_radius
        pad = r + self.brush_soft + 1
        size = self.canvas_size
        x0, x1 = max(0, int(x - pad)), min(size, int(x + pad) + 1)
        y0, y1 = max(0, int(y - pad)), min(size, int(y + pad) + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        val = np.clip((r - d) / self.brush_soft + 0.5, 0.0, 1.0).astype(np.float32)
        region = self.arr[y0:y1, x0:x1]
        if self._is_eraser():
            # 橡皮: 按覆盖率把墨迹淡出, 中心完全擦除、边缘平滑过渡
            self.arr[y0:y1, x0:x1] = np.minimum(region, 1.0 - val)
        else:
            self.arr[y0:y1, x0:x1] = np.maximum(region, val)

    def _draw_segment(self, x0, y0, x1, y1):
        dist = max(1.0, math.hypot(x1 - x0, y1 - y0))
        steps = int(dist) + 1
        for i in range(steps + 1):
            t = i / steps
            self._stamp(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)

    def _render(self):
        disp = ((1.0 - self.arr) * 255.0).astype(np.uint8)
        self._photo = ImageTk.PhotoImage(Image.fromarray(disp, mode="L"))
        self.canvas.itemconfigure(self.image_id, image=self._photo)

    def _on_press(self, event):
        self.last_xy = (event.x, event.y)
        self._stamp(event.x, event.y)
        self._render()

    def _on_drag(self, event):
        if self.last_xy is not None:
            self._draw_segment(self.last_xy[0], self.last_xy[1], event.x, event.y)
        self.last_xy = (event.x, event.y)
        self._render()
        if self.live.get():
            self.recognize()

    def _on_release(self, _event):
        self.last_xy = None
        if not self.live.get():
            self.recognize()

    # --------------------------- 识别 --------------------------- #
    def recognize(self):
        if self.model is None:
            return
        digit = preprocess(self.arr)
        if digit is None:
            self.last_probs = None
            self.result_var.set("—")
            self.conf_var.set("置信度: —")
            self._draw_bars(None)
            return

        x = to_tensor(digit).to(self.device)
        probs = self.model.predict_proba(x)[0].cpu().numpy()
        pred = int(probs.argmax())

        self.last_probs = probs
        self.result_var.set(str(pred))
        self.conf_var.set(f"置信度: {probs[pred] * 100:.1f}%")
        self._draw_bars(probs)

    def clear(self):
        self.arr[:] = 0.0
        self.last_xy = None
        self._render()
        self.recognize()


def main():
    # 必须在创建 Tk 窗口之前开启 DPI 感知, 否则界面会被系统拉伸变模糊
    enable_dpi_awareness()
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    DigitApp(root)
    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
