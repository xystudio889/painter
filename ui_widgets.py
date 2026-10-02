"""界面层共用控件与工具。

这里放的是纯 Tk 的东西: 可滚动容器、缩略图缓存、数据集配色、单字输入对话框、
对话框基类。业务逻辑都在 `dataset.py` / `arch.py` / `pnt.py` / `ptp.py` 里。
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from dpiutil import Metrics, center_on_parent, enable_dpi_awareness
from preprocess import ink_to_pil

# 进程 DPI 感知必须在创建任何 Tk 窗口之前开启, 这里在模块导入时就做掉,
# 保证所有对话框(ui_dataset/ui_arch/ui_train/ui_test)都受益。
enable_dpi_awareness()

THUMB_SIZE = 30            # 列表缩略图边长(逻辑像素)
THUMB_CACHE_LIMIT = 400    # 缩略图 LRU 上限

# 数据集配色: 按标签索引循环取用, 让很多数据集也能一眼区分
LABEL_COLORS = (
    "#0a6ebd", "#c0392b", "#1e8449", "#8e44ad", "#d68910", "#16a085",
    "#2c3e50", "#cb4335", "#117864", "#7d3c98", "#b9770e", "#1a5276",
)
COLOR_PASS = "#1e8449"
COLOR_FAIL = "#c0392b"
COLOR_MUTED = "#8a8a8a"
COLOR_EMPTY = "#cfcfcf"


def label_color(label: str) -> str:
    """按数据集名稳定地取一个颜色。"""
    if not label:
        return COLOR_MUTED
    index = sum((i + 1) * ord(ch) for i, ch in enumerate(label)) % len(LABEL_COLORS)
    return LABEL_COLORS[index]


def fit_size(width: int, height: int, box: int) -> tuple[int, int]:
    """等比缩放到不超过 box x box。"""
    if width <= 0 or height <= 0:
        return 1, 1
    scale = min(box / width, box / height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def thumbnail_image(ink, size: int) -> Image.Image:
    """把画板墨迹做成小缩略图。

    两个关键点(不做的话缩略图会看着"没显示"):
    1. **先把墨迹按最大值归一化** —— 画得轻的笔画原本只有 0.2 的灰度,
       缩到 26px 之后几乎全白, 看起来就像空的。
    2. 用 LANCZOS 缩小, 细笔画不会被锯齿吃掉。
    """
    array = np.asarray(ink, dtype=np.float32)
    peak = float(array.max()) if array.size else 0.0
    if peak > 0.02:
        array = array / peak                 # 归一化, 保留相对深浅
    image = Image.fromarray(np.clip((1.0 - array) * 255.0, 0, 255).astype(np.uint8),
                            mode="L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.LANCZOS)
    return image


class ScrollFrame(ttk.Frame):
    """一个竖向可滚动容器。

    用法:
        sf = ScrollFrame(parent, height=360)
        sf.pack(fill="both", expand=True)
        ttk.Label(sf.body, text="...").pack()      # 往 sf.body 里塞内容
    """

    def __init__(self, parent, width: int | None = None, height: int | None = None,
                 **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0,
                                width=width or 200, height=height or 200)
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scroll.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.body = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", self._on_body_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        # 滚轮: 焦点在容器内时也要能滚
        for widget in (self.canvas, self.body):
            widget.bind("<Enter>", lambda _e: self._bind_wheel())
            widget.bind("<Leave>", lambda _e: self._unbind_wheel())

    # --------------------------- 事件 --------------------------- #
    def _on_body_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        # 让内容宽度跟随容器, 文本才能正常换行/撑满
        self.canvas.itemconfigure(self._window, width=event.width)

    def _bind_wheel(self):
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _unbind_wheel(self):
        self.canvas.unbind_all("<MouseWheel>")

    def _on_wheel(self, event):
        if self._scrollable():
            self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _scrollable(self) -> bool:
        box = self.canvas.bbox("all")
        return bool(box) and box[3] > self.canvas.winfo_height()

    # --------------------------- 操作 --------------------------- #
    def clear(self):
        for child in self.body.winfo_children():
            child.destroy()
        self.canvas.yview_moveto(0)

    def scroll_to_top(self):
        self.canvas.yview_moveto(0)

    def scroll_to(self, fraction: float):
        self.canvas.yview_moveto(max(0.0, min(1.0, fraction)))


class ThumbCache:
    """缩略图缓存: 按 key 缓存 `PhotoImage`, 条目超过上限就丢掉最早的。"""

    def __init__(self, master: tk.Misc, limit: int = THUMB_CACHE_LIMIT):
        self.master = master
        self.limit = limit
        self._items: dict[tuple, tk.PhotoImage] = {}

    def get(self, key: tuple, ink, size: int, scale: float) -> tk.PhotoImage:
        cached = self._items.get(key)
        if cached is not None:
            return cached
        photo = self.make(ink, size, scale)
        self._items[key] = photo
        self._trim()
        return photo

    def refresh(self, key: tuple, ink, size: int, scale: float) -> tk.PhotoImage:
        """按新墨迹重做一张, 替换掉缓存里的旧图(画板实时更新缩略图用)。

        调用方要把 `get()` 拿到的旧图**先留在自己手里**, 等控件换上新的再放掉:
        直接丢会把 Tk 那边的图片对象删掉, 而控件(例如 Treeview 的某一行)还指着它。
        """
        photo = self.make(ink, size, scale)
        self._items[key] = photo
        self._trim()
        return photo

    def make(self, ink, size: int, scale: float) -> tk.PhotoImage:
        physical = max(8, int(round(size * scale)))
        image = thumbnail_image(ink, physical)
        # 必须用 PIL 的 PhotoImage: tkinter 自带的 PhotoImage 不认 `image=` 选项
        return ImageTk.PhotoImage(image, master=self.master)

    def drop(self, key: tuple) -> None:
        self._items.pop(key, None)

    def drop_prefix(self, partition: str, label: str) -> None:
        for key in [k for k in self._items if k[:2] == (partition, label)]:
            self._items.pop(key, None)

    def clear(self):
        self._items.clear()

    def _trim(self) -> None:
        while len(self._items) > self.limit:
            self._items.pop(next(iter(self._items)))

    def __len__(self) -> int:
        return len(self._items)


class Dialog(tk.Toplevel):
    """对话框基类: 置顶、模态、居中、应用 DPI 缩放。"""

    def __init__(self, parent, title: str, resizable: bool = True, modal: bool = True):
        super().__init__(parent)
        self.parent = parent
        self.title(title)
        self.resizable(resizable, resizable)
        self.transient(parent)
        self.metrics = Metrics(self)
        self.metrics.apply(self)
        self._last_dpi = self.metrics.dpi
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.bind("<Escape>", lambda _e: self.cancel())

    def center(self):
        self.update_idletasks()
        center_on_parent(self, self.parent)

    def show(self):
        """居中后进入模态循环, 返回 `self.result`。"""
        self.center()
        if self.winfo_exists():
            self.deiconify()
            self.lift()
            self.focus_force()
            self.grab_set()
            self.wait_window(self)
        return getattr(self, "result", None)

    def cancel(self):
        self.result = None
        self.destroy()

    def confirm(self):
        self.result = True
        self.destroy()


class SingleCharDialog(Dialog):
    """要求输入**单个字符**的小对话框(用于新建数据集)。"""

    def __init__(self, parent, title: str, prompt: str, taken: set[str],
                 message: str = ""):
        super().__init__(parent, title, resizable=False)
        self.taken = {t.strip() for t in taken}
        self.result: str | None = None
        px = self.metrics.px

        body = ttk.Frame(self, padding=px(14))
        body.pack(fill="both", expand=True)

        ttk.Label(body, text=prompt, font=("Segoe UI", 10)).pack(anchor="w")
        if message:
            ttk.Label(body, text=message, font=("Segoe UI", 9),
                      foreground="#666").pack(anchor="w", pady=(2, 6))

        self.entry = ttk.Entry(body, font=("Segoe UI", 14), width=6,
                               justify="center")
        self.entry.pack(pady=(8, 6))
        self.entry.bind("<KeyRelease>", lambda _e: self._validate())
        self.entry.bind("<Return>", lambda _e: self._ok())

        self.hint = ttk.Label(body, text="仅支持单个字符，例如 a / 5 / 猫",
                              font=("Segoe UI", 9), foreground=COLOR_MUTED)
        self.hint.pack(anchor="w")

        # 主操作(创建/确定)在右、取消在左 —— 和 Windows 习惯一致
        buttons = ttk.Frame(body)
        buttons.pack(fill="x", pady=(px(12), 0))
        ttk.Button(buttons, text="取消", command=self.cancel).pack(side="right")
        self.ok_btn = ttk.Button(buttons, text="创建", command=self._ok)
        self.ok_btn.pack(side="right", padx=(0, px(6)))

        self.after(50, self.entry.focus_set)

    def _validate(self) -> str | None:
        text = self.entry.get().strip()
        problem = None
        if not text:
            problem = "请输入一个字符"
        elif len(text) > 1:
            problem = f"仅支持单个字符，当前输入了 {len(text)} 个字符"
        elif text in self.taken:
            problem = f"当前分区中已存在数据集「{text}」"
        if problem:
            self.hint.configure(text=problem, foreground=COLOR_FAIL)
            self.ok_btn.state(["disabled"])
        else:
            self.hint.configure(text="可以创建", foreground=COLOR_PASS)
            self.ok_btn.state(["!disabled"])
        return text if not problem else None

    def _ok(self):
        text = self._validate()
        if text:
            self.result = text
            self.destroy()


def ask_single_char(parent, title: str, prompt: str, taken: set[str],
                    message: str = "") -> str | None:
    """弹出单字输入框, 返回合法字符或 None。"""
    return SingleCharDialog(parent, title, prompt, taken, message).show()


def ask_yes_no(parent, title: str, message: str) -> bool:
    from tkinter import messagebox
    return bool(messagebox.askyesno(title, message, parent=parent))


def warm_style(root: tk.Misc) -> None:
    """沿用主窗口的 ttk 主题(vista 在 Windows 上最好看)。"""
    try:
        ttk.Style(root).theme_use("vista")
    except tk.TclError:
        pass
