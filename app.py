"""手写字符识别 / 自定义训练 —— Tkinter 图形界面。

运行:
    python app.py

左侧画板区画图, 右侧显示识别结果与"匹配度"(当前图片对每个数据集的匹配程度,
数据集多了可以滚动)。侧边栏是自定义训练模式的三段式: 分区 / 数据集 / 按钮。

自定义训练模式里的三个概念:
    分区   —— 训练集(train) 和 测试集(test)
    数据集 —— 名字是**单个字符**, 它同时就是分类标签
    图片   —— 画板上画出来的图, 编号由程序从 0 起自动分配

模型统一用 `.pt` 文件传递, 工程用 `.ptp` 打包(zip: train/ + test/ + modelinfo.json)。

支持高 DPI: 启动时开启进程 DPI 感知, 并按显示器 DPI 等比缩放界面。
"""

import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk

import numpy as np
import torch
from PIL import Image, ImageTk

import arch as arch_mod
import preprocess as pp
import ptp
import trainer
from dataset import (
    PARTITION_NAMES, PARTITION_TEST, PARTITION_TRAIN, Workspace, is_valid_label,
)
from dpiutil import Metrics, enable_dpi_awareness
from paths import app_root
from preprocess import CANVAS_SIZE, ink_to_pil
from settings import Settings
from ui_dataset import DatasetSidebar
from ui_widgets import (
    COLOR_EMPTY, COLOR_FAIL, COLOR_MUTED, COLOR_PASS, ScrollFrame, label_color,
)

ROOT = app_root()
MODELS_DIR = ROOT / "models"                # 所有 .pt 模型都放这个目录
MODEL_PATH = MODELS_DIR / "mnist_cnn.pt"    # 默认模型
MNIST_LABELS = [str(i) for i in range(10)]
MNIST_MODEL_NAME = "mnist 数字识别"

# 逻辑像素尺寸(96 DPI 基准)
PEN_MIN, PEN_MAX = 4, 48   # 笔尖粗细(直径)范围
PEN_DEFAULT = 24           # 默认笔尖直径
SOFT_RATIO = 0.22          # 笔尖边缘柔化宽度 / 笔尖半径
ERASER_GAIN = 1.6          # 橡皮相对画笔的放大倍数

MATCH_ROW_H = 22           # 匹配度每一行的高度
MATCH_BAR_W = 120          # 匹配度进度条宽度
MATCH_BAR_H = 10           # 匹配度进度条高度

MODE_MAIN = "main"         # 主界面: 画 + 识别
MODE_CUSTOM = "custom"     # 自定义训练模式: 分区/数据集/训练/测试

LIVE_DELAY_MS = 30         # 拖笔时实时识别的最小间隔(合并密集的鼠标事件)


def preprocess(arr: np.ndarray):
    """兼容旧调用: 裁剪 + 缩放 + 重心居中。"""
    return pp.preprocess(arr)


class ModelEntry:
    """一个已加载/可加载的模型。"""

    def __init__(self, name: str, path, labels: list[str], architecture: dict,
                 input_mode: str, mean: float, std: float, accuracy=None,
                 legacy: bool = False):
        self.name = name
        self.path = Path(path) if path else None
        self.labels = list(labels)
        self.architecture = architecture
        self.input_mode = input_mode
        self.mean = mean
        self.std = std
        self.accuracy = accuracy
        self.legacy = legacy
        self.model = None            # 懒加载

    @property
    def display(self) -> str:
        return self.name

    def ensure_loaded(self):
        """按需真正把权重读进来。"""
        if self.model is None:
            if self.path is None:
                raise trainer.TrainError(f"模型「{self.name}」尚未关联 .pt 文件")
            info, model = trainer.load_checkpoint(self.path)
            self.model = model
            self.model.eval()
        return self.model


class DigitApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("手写字符识别")
        try:
            root.iconbitmap(ROOT / "icon.ico")
        except tk.TclError:
            pass
        root.resizable(False, False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --------------------------- 高 DPI --------------------------- #
        self.metrics = Metrics(root)
        self.metrics.apply(root)
        self.dpi = self.metrics.dpi
        self.scale = self.metrics.scale

        self.canvas_size = self._px(CANVAS_SIZE)
        self.arr = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
        self.last_xy = None
        # 画笔工具的状态: 切模式会重建画板控件, 但这些状态要留着
        self.tool_var = tk.StringVar(value="brush")
        self.pen_size = tk.DoubleVar(value=PEN_DEFAULT)
        # 识别模式: 勾上就每落笔实时识别, 取消则只认「识别」按钮
        self.live = tk.BooleanVar(value=True)

        # --------------------------- 数据状态 --------------------------- #
        self.workspace = Workspace()
        self.current_image: tuple[str, str, int] | None = None
        self.models: list[ModelEntry] = []
        self.model_index = 0
        self.architecture = arch_mod.default_arch(10)
        self.training = False
        self.mode = MODE_MAIN
        self.settings = Settings()
        # 匹配度列表的行控件: 标签 -> 那一行的控件引用, 刷新时只改不重建
        self._match_rows: dict[str, dict] = {}
        self._match_structure = None        # 当前列表长什么样, 变了才重建
        self._live_job = None               # 拖笔时那次"稍后识别"的定时器

        self._build_ui()
        self._apply_metrics()
        # 先量两页的自然尺寸, 再把窗口对齐到主界面; 模型信息可能有变化, 后面再量一次
        self._measure_pages()
        self._apply_page_size(MODE_MAIN)
        self._load_models()
        self._measure_pages()
        self._apply_page_size(self.mode)
        self._render()
        self._refresh_title()

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind("<Configure>", self._on_configure)

    # ======================================================================= #
    # 高 DPI
    # ======================================================================= #
    def _px(self, value: float) -> int:
        return self.metrics.px(value)

    def _px0(self, value: float) -> int:
        return self.metrics.px0(value)

    def _on_configure(self, event):
        if event.widget is not self.root:
            return
        self.root.after_idle(self._sync_dpi)

    def _sync_dpi(self):
        if not self.metrics.refresh(self.root):
            return
        self.dpi = self.metrics.dpi
        self.scale = self.metrics.scale

        new_size = self._px(CANVAS_SIZE)
        if new_size != self.canvas_size:
            img = Image.fromarray((self.arr * 255).astype(np.uint8), mode="L")
            self.arr = np.asarray(img.resize((new_size, new_size), Image.LANCZOS),
                                  dtype=np.float32) / 255.0
            self.canvas_size = new_size

        self.sidebar.thumbs.clear()
        self._apply_metrics()
        self._render()
        self.sidebar.refresh()
        self._refresh_match_list()
        self._update_status()

    def _apply_metrics(self):
        px = self.metrics.px
        radius = float(self.pen_size.get()) / 2.0
        if self._is_eraser():
            radius *= ERASER_GAIN
        self.brush_radius = max(1.5, radius * self.scale)
        self.brush_soft = max(0.8, radius * SOFT_RATIO * self.scale)

        self.page_main.configure(padding=px(10))
        self.left.configure(padding=px(8))
        self.tools.pack_configure(pady=(0, px(8)))
        self.pen_scale.configure(length=px(150))
        self.canvas.configure(width=self.canvas_size, height=self.canvas_size,
                              highlightthickness=px(2))
        self.btns.pack_configure(pady=(px(10), 0))
        self.btn_clear.pack_configure(padx=(0, px(6)))
        if self.mode == MODE_MAIN:
            # 识别模式底部: 清空 / 识别 / 实时识别
            self.btn_recognize.pack_configure(padx=(0, px(6)))
            self.chk_live.pack_configure(padx=(0, px(6)))
        else:
            # 自定义模式底部: 清空 / 删除 / 下一张
            self.btn_delete.pack_configure(padx=(0, px(6)))
            self.btn_next.pack_configure(padx=(0, px(6)))

        # 两页的排版都按当前 DPI 设一遍: 隐藏的那一页也要有正确的尺寸,
        # 否则切页/量尺寸时会用到旧的像素值。
        self.right.configure(padding=(px(16), 0, 0, 0))
        self.result_lbl.grid_configure(pady=(0, px(4)))
        self.match_label.grid_configure(pady=(px(6), px(4)))
        self.match_scroll.configure(width=px(MATCH_BAR_W + 150), height=px(300))
        self.custom_right.configure(padding=(px(12), 0, 0, 0))
        self.match_label_custom.grid_configure(pady=(px(8), px(4)))
        self.match_scroll_custom.configure(width=px(MATCH_BAR_W + 150),
                                           height=px(270))

        self._render()
        self._refresh_match_list()
        self._refresh_train_hint()

    # ======================================================================= #
    # 界面构建
    # ======================================================================= #
    def _build_ui(self):
        px = self.metrics.px
        # 两页共用的状态变量只在这里建一次。
        # 以前主界面和自定义模式各建一份 StringVar, 后建的那份会把 self.xxx_var 顶掉,
        # 先建的那份再没人引用, 被回收时连它的 Tcl 变量一起删掉 —— 控件的 textvariable
        # 就指向了一个不存在的变量, 所以再怎么 set() 界面也不会变。
        self.model_var = tk.StringVar(value="正在加载…")
        self.result_var = tk.StringVar(value="—")
        self.conf_var = tk.StringVar(value="置信度：—")

        self.shell = ttk.Frame(self.root, padding=px(10))
        self.shell.grid(row=0, column=0)
        self.shell.columnconfigure(0, weight=1)
        self.shell.rowconfigure(0, weight=1)

        # ------------------------------ 两页 ------------------------------ #
        # 两页放在同一个格子里, 各自按内容自然撑开; 高度对齐见 _equalize_page_heights
        self.page_main = ttk.Frame(self.shell)
        self.page_custom = ttk.Frame(self.shell)
        for page in (self.page_main, self.page_custom):
            page.grid(row=0, column=0, sticky="nsew")
        self.page_custom.grid_remove()      # 启动时显示主界面

        # 主界面: 画板 | 模型+识别结果+匹配度 + 进入自定义模式的入口
        self.page_main.columnconfigure(0, weight=1)
        self.page_main.columnconfigure(1, weight=1)
        self.page_main.rowconfigure(0, weight=1)
        self.left = self._build_canvas_area(self.page_main, MODE_MAIN)
        self.left.grid(row=0, column=0, sticky="ns")

        self.right = ttk.Frame(self.page_main, padding=(px(16), 0, 0, 0))
        self.right.grid(row=0, column=1, sticky="ns")
        # 一段一段往下排, 每段返回下一个空闲行号。
        # 不能像以前那样各段都从 row=0 开始: 几段控件会挤进同一个 grid 单元格,
        # 后建的那段整个盖在先建的上面(蓝色大字和模型选择框就是这样被盖掉的)。
        row = self._build_model_row(self.right, 0, compact=False)
        row = self._build_result_head(self.right, row)
        row = self._build_enter_block(self.right, row)
        self._build_match_panel(self.right, row, compact=False)

        # 自定义模式: 画板 | 侧边栏 | 训练/测试
        self._build_custom_page()

        # ------------------------------ 底部状态 ------------------------------ #
        self.model_info = "正在加载模型…"
        self.status_var = tk.StringVar(value=self.model_info)
        self._status_full = self.model_info        # 未截断的状态栏文字
        self.status_lbl = ttk.Label(self.shell, textvariable=self.status_var,
                                    font=("Segoe UI", 9), foreground="#666")
        self.status_lbl.grid(row=1, column=0, columnspan=2, sticky="w",
                             pady=(px(8), 0))

    def _measure_pages(self) -> None:
        """量出两页各自的自然尺寸(启动时做一次)。

        量的时候两页都得真的在 geometry manager 里, 否则量到的是 1;
        量完恢复成"只显示当前页"。
        """
        try:
            pages = (self.page_main, self.page_custom)
            for page in pages:
                page.grid_propagate(True)
            self.shell.update_idletasks()
            for page in pages:
                page.grid()
                page.update_idletasks()
                self.shell.update_idletasks()
            self.page_natural = {
                MODE_MAIN: (self.page_main.winfo_reqwidth(),
                            self.page_main.winfo_reqheight()),
                MODE_CUSTOM: (self.page_custom.winfo_reqwidth(),
                              self.page_custom.winfo_reqheight()),
            }
            for page in pages:
                if page is not self._page_of(self.mode):
                    page.grid_remove()
            self.shell.update_idletasks()
        except tk.TclError:
            self.page_natural = {}

    def _apply_page_size(self, mode: str | None = None) -> None:
        """让当前页用**两页共同的高度**(所以切页时只有宽度变, 高度不跳)。

        高度取两页自然高度的较大值; 宽度用当前页自己的, 这样窄页面(识别模式)
        不会被另一页撑出大片空白。

        这里顺手把 shell 的尺寸**钉死**(关掉 grid 的自动收缩): 否则底部状态栏
        那行文字一长 —— 比如点了「模型参数」的确定, 状态栏写上一串架构描述 ——
        它就会把整个窗口撑宽一圈。宽度只在切页 / 改 DPI 时重算, 平时子控件改不动窗口。
        """
        if mode is None:
            mode = self.mode
        natural = getattr(self, "page_natural", None)
        if not natural or mode not in natural:
            return
        try:
            # 高度取两页的较大值: 无论在哪一页, 窗口都一样高
            page_height = max(natural[MODE_MAIN][1], natural[MODE_CUSTOM][1])
            # 宽度按当前页自己的来: 窄页面不会被另一页撑出大片空白
            page_width = natural[mode][0]
            pad = self.metrics.px(10)
            page = self._page_of(mode)
            for other in (self.page_main, self.page_custom):
                other.grid_propagate(True)
            # 状态栏那一行也要占高度: grid 自动收缩关掉之后得自己算进去。
            # 宽度用**没截断**的完整文字量, 否则一旦截过, 后面就再也长不回来了。
            status_text = self._status_full or self.status_var.get()
            status_height = self.status_lbl.winfo_reqheight() + self.metrics.px(8)
            self._status_width = max(page_width, self._status_text_width(status_text))
            page.configure(width=page_width, height=page_height)
            page.grid_propagate(False)
            self.shell.configure(width=self._status_width + pad * 2,
                                 height=page_height + status_height + pad * 2)
            self.shell.grid_propagate(False)
            # 宽度可能刚变大, 把状态栏文字按新宽度重新放一遍(之前可能被截短了)
            self.status_var.set(self._fit_status_text(status_text))
            self.shell.update_idletasks()
            self.root.update_idletasks()
            self.root.geometry("")       # 让窗口按新内容收放
        except tk.TclError:
            pass

    def _page_of(self, mode: str) -> ttk.Frame:
        return self.page_custom if mode == MODE_CUSTOM else self.page_main

    # ------------------------------ 画板区 ------------------------------ #
    def _build_canvas_area(self, parent, mode: str) -> ttk.LabelFrame:
        """画板 + 画笔工具 + 底部按钮行。

        底部一行的内容随模式不同:
            识别模式   : 清空 / 识别 / 实时识别
            自定义模式 : 清空 / 删除 / 下一张
        Tk 不支持把控件在父容器之间搬家, 所以切换模式时**重建**这块;
        真正要留住的状态(墨迹 `self.arr`、画笔粗细、当前工具、实时识别开关)
        都在本对象上, 重建不会丢东西。
        """
        px = self.metrics.px
        # 换页时旧画板区会留在原地(只是被取消 grid), 必须先销毁, 否则会越堆越多
        old = getattr(self, "left", None)
        if old is not None and old.winfo_exists():
            old.destroy()
        # 底部按钮只属于当前模式, 先清掉引用, 免得 hasattr 判断出错
        self.btn_recognize = None
        self.btn_next = None
        self.btn_delete = None
        left = ttk.LabelFrame(parent, text="画板", padding=px(8))

        self.tools = ttk.Frame(left)
        self.tools.pack(fill="x", pady=(0, px(8)))
        ttk.Radiobutton(self.tools, text=" 画笔 ", value="brush",
                        variable=self.tool_var, style="Toolbutton",
                        command=lambda: self._set_tool("crosshair")).pack(side="left")
        ttk.Radiobutton(self.tools, text=" 橡皮 ", value="eraser",
                        variable=self.tool_var, style="Toolbutton",
                        command=lambda: self._set_tool("dotbox")).pack(side="left")
        ttk.Separator(self.tools, orient="vertical").pack(side="left", fill="y",
                                                          padx=px(8))
        ttk.Label(self.tools, text="粗细", font=("Segoe UI", 9)).pack(side="left")
        self.pen_scale = ttk.Scale(self.tools, from_=PEN_MIN, to=PEN_MAX,
                                   orient="horizontal", length=150,
                                   variable=self.pen_size,
                                   command=self._on_pen_change)
        self.pen_scale.pack(side="left", padx=px(4))
        self.pen_lbl = ttk.Label(self.tools, text=str(int(self.pen_size.get())),
                                 width=3, anchor="center", font=("Consolas", 9))
        self.pen_lbl.pack(side="left")

        self.canvas = tk.Canvas(left, width=self.canvas_size,
                                height=self.canvas_size, bg="white",
                                highlightthickness=px(2),
                                highlightbackground="#444", cursor="crosshair")
        self.canvas.pack()
        self.image_id = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        self.btns = ttk.Frame(left)
        self.btns.pack(fill="x", pady=(px(10), 0))
        self.btn_clear = ttk.Button(self.btns, text="清空", width=8,
                                    command=self.clear)
        self.btn_clear.pack(side="left")
        if mode == MODE_MAIN:
            self.btn_recognize = ttk.Button(self.btns, text="识别", width=8,
                                            command=self.recognize)
            self.btn_recognize.pack(side="left", padx=(0, px(6)))
            self.chk_live = ttk.Checkbutton(self.btns, text="实时识别",
                                            variable=self.live)
            self.chk_live.pack(side="left")
        else:
            # 先 pack 下一张, 再 pack 删除, 显示顺序才是「删除 下一张」
            self.btn_next = ttk.Button(self.btns, text="下一张", width=12,
                                       command=self.next_image)
            self.btn_next.pack(side="right")
            self.btn_delete = ttk.Button(self.btns, text="删除", width=8,
                                         command=self.delete_image)
            self.btn_delete.pack(side="right", padx=(0, px(6)))
        return left

    # ------------------------------ 模型行 ------------------------------ #
    def _build_model_row(self, parent, row: int, compact: bool = False) -> int:
        """模型选择(可切换)或只读的模型名(compact)。返回下一个空闲行号。

        统一用 grid 排版, 这样"模型"标签、选择框、下面那行说明自上而下对齐,
        不会出现控件左右错位。模型说明那一行两页各有一份(见 `_set_model_hint`)。
        """
        px = self.metrics.px
        parent.columnconfigure(0, weight=1)
        if compact:
            # 自定义模式只读显示, 压成一行, 省下来的高度留给匹配度列表
            self.model_line = ttk.Label(
                parent, textvariable=self.model_var, font=("Segoe UI", 10),
                foreground="#0a6ebd")
            self.model_line.grid(row=row, column=0, sticky="w")
            row += 1
        else:
            ttk.Label(parent, text="模型", font=("Segoe UI", 10)).grid(
                row=row, column=0, sticky="w")
            row += 1
            model_row = ttk.Frame(parent)
            model_row.grid(row=row, column=0, sticky="w", pady=(px(2), px(6)))
            self.model_combo = ttk.Combobox(model_row, textvariable=self.model_var,
                                            state="readonly", width=26,
                                            font=("Segoe UI", 9))
            self.model_combo.pack(side="left")
            self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)
            ttk.Button(model_row, text="导入", width=6,
                       command=self.import_model).pack(side="left", padx=(px(4), 0))
            row += 1
        row += 1
        return row

    # ------------------------------ 识别结果(蓝色大字) ------------------------------ #
    def _build_result_head(self, parent, row: int) -> int:
        """识别结果: 标题 + 蓝色大字 + 置信度。只有主界面有这一块。"""
        px = self.metrics.px
        ttk.Label(parent, text="识别结果", font=("Segoe UI", 11)).grid(
            row=row, column=0, sticky="w", pady=(px(10), 0))
        row += 1
        self.result_lbl = ttk.Label(parent, textvariable=self.result_var,
                                    font=("Consolas", 36, "bold"),
                                    foreground="#0a6ebd", width=4,
                                    anchor="center")
        self.result_lbl.grid(row=row, column=0, pady=(0, px(4)))
        row += 1
        self.conf_lbl = ttk.Label(parent, textvariable=self.conf_var,
                                  font=("Segoe UI", 10))
        self.conf_lbl.grid(row=row, column=0, sticky="w")
        return row + 1

    # ------------------------------ 进入自定义模式的入口 ------------------------------ #
    def _build_enter_block(self, parent, row: int) -> int:
        """分隔线 + 「自定义训练模式」按钮 + 一行说明。只有主界面有。"""
        px = self.metrics.px
        enter = ttk.Frame(parent)
        enter.grid(row=row, column=0, sticky="ew", pady=(px(10), 0))
        enter.columnconfigure(0, weight=1)
        ttk.Separator(enter, orient="horizontal").grid(row=0, column=0, sticky="ew",
                                                       pady=(0, px(8)))
        ttk.Button(enter, text="自定义训练模式", width=20,
                   command=lambda: self.set_mode(MODE_CUSTOM)).grid(
            row=1, column=0, sticky="w")
        ttk.Label(enter, text="自行绘制样本、搭建网络并训练模型",
                  font=("Segoe UI", 9), foreground=COLOR_MUTED).grid(
            row=2, column=0, sticky="w", pady=(px(2), 0))
        return row + 1

    # ------------------------------ 匹配度列表 ------------------------------ #
    def _build_match_panel(self, parent, row: int, compact: bool) -> int:
        """匹配度标题 + 可滚动列表。

        主界面和自定义模式各有一份: 两页都在, 更新走 `_match_scroll()` 选当前页,
        所以隐藏的那一页不会把可见的那一页顶掉。
        """
        px = self.metrics.px
        parent.columnconfigure(0, weight=1)
        if not compact:
            ttk.Separator(parent, orient="horizontal").grid(
                row=row, column=0, sticky="ew", pady=px(10))
            row += 1
        label = ttk.Label(parent, font=("Segoe UI", 10),
                          text="置信度")
        label.grid(row=row, column=0, sticky="w",
                   pady=(px(8), px(4)) if compact else (px(6), px(4)))
        row += 1
        scroll = ScrollFrame(parent, width=px(MATCH_BAR_W + 150),
                             height=px(270 if compact else 300))
        scroll.grid(row=row, column=0, sticky="nsew")
        parent.rowconfigure(row, weight=1)
        if compact:
            self.match_label_custom = label
            self.match_scroll_custom = scroll
        else:
            self.match_label = label
            self.match_scroll = scroll
        return row + 1

    def _match_scroll(self) -> ScrollFrame:
        """当前页面上的匹配度列表。"""
        return (self.match_scroll_custom if self.mode == MODE_CUSTOM
                else self.match_scroll)

    def _set_model_hint(self, text: str, error: bool = False) -> None:
        """模型下面那行说明。两页各有一份, 一起更新, 切页回来才是对的。"""
        color = COLOR_FAIL if error else COLOR_MUTED
        for label in (getattr(self, "model_hint", None),
                      getattr(self, "model_hint_custom", None)):
            if label is not None:
                label.configure(text=text, foreground=color)

    def _refresh_model_hint(self) -> None:
        """按当前模型刷新说明文字(模型类别列表)。"""
        entry = self.current_model
        if entry is None:
            self._set_model_hint("")
            return
        self._set_model_hint(
            f"{len(entry.labels)} 类：{' '.join(entry.labels[:12])}"
            + ("…" if len(entry.labels) > 12 else ""))

    # ------------------------------ 自定义模式页 ------------------------------ #
    def _build_custom_page(self):
        px = self.metrics.px
        self.page_custom.columnconfigure(0, weight=1)
        self.page_custom.columnconfigure(1, weight=1)
        self.page_custom.rowconfigure(0, weight=1)

        # 画板区会由 set_mode 填进 column=0

        # 中间: 自定义训练侧边栏(分区 / 数据集 / 按钮), 高度撑满整页
        sidebar_box = ttk.LabelFrame(self.page_custom, text="分区与数据集",
                                     padding=px(8))
        sidebar_box.grid(row=0, column=1, sticky="nsew", padx=(px(10), 0))
        self.sidebar_box = sidebar_box
        self.sidebar = DatasetSidebar(sidebar_box, self, self.metrics, self.workspace)
        self.sidebar.pack(fill="both", expand=True)

        # 右侧: 模型 + 训练/测试 + 匹配度
        # 注意: 千万不能再叫 self.right —— 那是主界面右栏, 会被整页盖掉
        self.custom_right = ttk.Frame(self.page_custom, padding=(px(12), 0, 0, 0))
        self.custom_right.grid(row=0, column=2, sticky="nsew")

        back = ttk.Frame(self.custom_right)
        back.grid(row=0, column=0, sticky="ew")
        ttk.Button(back, text="← 返回主界面", width=14,
                   command=lambda: self.set_mode(MODE_MAIN)).pack(anchor="w")

        model_box = ttk.Frame(self.custom_right)
        model_box.grid(row=1, column=0, sticky="ew", pady=(px(4), 0))
        self._build_model_row(model_box, 0, compact=True)

        train_box = ttk.LabelFrame(self.custom_right, text="训练与测试",
                                   padding=px(6))
        train_box.grid(row=2, column=0, sticky="ew", pady=(px(6), 0))
        self.btn_train = ttk.Button(train_box, text="训练模型", width=18,
                                    command=self.start_training)
        self.btn_train.pack(anchor="w")
        self.btn_test = ttk.Button(train_box, text="测试测试集", width=18,
                                   command=self.run_test)
        self.btn_test.pack(anchor="w", pady=(px(4), 0))
        self.btn_arch = ttk.Button(train_box, text="模型参数", width=18,
                                   command=self.open_arch_dialog)
        self.btn_arch.pack(anchor="w", pady=(px(4), 0))
        self.train_hint = ttk.Label(train_box, text="", font=("Segoe UI", 9),
                                    foreground=COLOR_MUTED, justify="left",
                                    wraplength=px(240))
        self.train_hint.pack(anchor="w", pady=(px(6), 0))

        # 匹配度区: 撑满剩余高度, 数据集多的时候在内部滚动
        self.custom_right.rowconfigure(3, weight=1)
        result_box = ttk.Frame(self.custom_right)
        result_box.grid(row=3, column=0, sticky="nsew", pady=(px(10), 0))
        result_box.rowconfigure(0, weight=1)
        result_box.columnconfigure(0, weight=1)
        self._build_match_panel(result_box, 0, compact=True)

    # ======================================================================= #
    # 模式切换: 主界面 <-> 自定义训练模式
    # ======================================================================= #
    def set_mode(self, mode: str):
        """切换界面。Tk 不能给控件换父容器, 所以画板区在这两页里各建一份。"""
        if mode not in (MODE_MAIN, MODE_CUSTOM) or mode == self.mode:
            return
        if self.training:
            messagebox.showinfo("正在训练", "训练结束之后再切换界面。")
            return
        self.mode = mode
        if mode == MODE_CUSTOM:
            self.page_main.grid_remove()
            self.page_custom.grid()
            host, column = self.page_custom, 0
            self.sidebar.partition_var.set(self.sidebar.partition)
            self.sidebar.refresh()
        else:
            self.page_custom.grid_remove()
            self.page_main.grid()
            host, column = self.page_main, 0
        self.left = self._build_canvas_area(host, mode)
        self.left.grid(row=0, column=column, sticky="ns")
        self._apply_metrics()
        # 先刷新状态栏再定窗口尺寸: 这一页的状态文字可能比上一页长, 量宽度要按它来
        self._update_status()
        self._apply_page_size(mode)           # 画板区重建后重新对齐两页尺寸
        self._render()
        self.recognize()
        self._refresh_title()

    def _fit_window(self):
        """切页之后把窗口收放到当前页面的尺寸(两页高度一致, 所以不会跳)。"""
        try:
            self.shell.update_idletasks()
            self.root.geometry("")
        except tk.TclError:
            pass

    def _refresh_train_hint(self):
        """自定义模式里提示当前数据够不够训练。"""
        if not hasattr(self, "train_hint"):
            return
        labels = self.workspace.train_labels()
        counts = self.workspace.train_class_counts()
        test_count = self.workspace.partition(PARTITION_TEST).image_count()
        if not labels:
            text = ("尚未创建数据集。\n"
                    "请先在侧边栏点击「新建数据集」，并为每个数据集绘制若干样本。\n"
                    "样本归入训练集还是测试集，由上方「分区」选择框决定。")
        elif len(labels) < 2:
            text = (f"训练集仅有 1 个数据集（{labels[0]}）。\n"
                    "至少需要 2 个数据集才能进行分类训练。")
        else:
            empty = [label for label, count in counts.items() if count == 0]
            if empty:
                text = ("以下数据集在训练集中没有任何样本：\n"
                        f"  {' '.join(empty)}\n请先补充样本。")
            else:
                total = sum(counts.values())
                text = (f"训练集 {len(labels)} 类 / {total} 张，"
                        f"测试集 {test_count} 张。\n当前数据已满足训练条件。")
                if test_count == 0:
                    text += "\n（测试集为空，训练时将使用训练集进行评估）"
        self.train_hint.configure(text=text)

    # ======================================================================= #
    # 模型
    # ======================================================================= #
    def _load_models(self):
        """加载 models/ 下的模型, 然后按 settings.json 恢复上次选中的模型。

        默认模型是 `models/mnist_cnn.pt`; 目录里别的 `.pt` 也一起登记进下拉框,
        所以自己训练出来的模型只要丢进 models/ 就能直接选, 不用每次「导入」。
        """
        if not MODEL_PATH.exists():
            self.model_info = f"未找到默认模型 {MODEL_PATH.name}"
            self._update_status()
            messagebox.showwarning(
                "缺少默认模型",
                f"未找到默认模型文件：\n{MODEL_PATH}\n\n"
                "可采取以下任一方式：\n"
                "  · 进入「自定义训练模式」绘制样本，训练生成 .pt 模型；\n"
                "  · 点击模型右侧的「导入」按钮选择已有的 .pt 文件"
                "（将复制到 models/ 目录）。")
        else:
            try:
                info, _model = trainer.load_checkpoint(MODEL_PATH)
                entry = ModelEntry(
                    name=MNIST_MODEL_NAME, path=MODEL_PATH, labels=info["labels"],
                    architecture=info["arch"], input_mode=info["input_mode"],
                    mean=info["mean"], std=info["std"],
                    accuracy=info["accuracy"], legacy=info["legacy"])
                self._register_model(entry, select=True)
                size_kb = MODEL_PATH.stat().st_size / 1024
                # 不要带"模型："前缀 —— 状态栏自己会写一个, 免得重复
                self.model_info = (f"{MODEL_PATH.name}（{size_kb:.0f} KB）    "
                                   f"设备：{self.device.type.upper()}")
            except trainer.TrainError as exc:
                self.model_info = "默认模型加载失败"
                messagebox.showerror("模型加载失败", str(exc))
        self._discover_models()
        self._restore_saved_model()
        self._refresh_models_combo()
        self._update_status()

    def _discover_models(self):
        """把 models/ 目录里其它的 .pt 也登记进模型列表。

        顺手把读出来的权重留在条目上, 免得选中时再读一遍。目录里不是本程序格式的
        文件(读不了)静静跳过, 不拿弹窗打扰用户。
        """
        if not MODELS_DIR.is_dir():
            return
        known = {entry.path for entry in self.models if entry.path is not None}
        for path in sorted(MODELS_DIR.glob("*.pt")):
            if path in known:
                continue
            try:
                info, model = trainer.load_checkpoint(path)
            except trainer.TrainError:
                continue
            entry = ModelEntry(
                name=info["name"] or path.stem, path=path, labels=info["labels"],
                architecture=info["arch"], input_mode=info["input_mode"],
                mean=info["mean"], std=info["std"], accuracy=info["accuracy"],
                legacy=info["legacy"])
            entry.model = model
            self._register_model(entry)

    def _restore_saved_model(self):
        """恢复 settings.json 里记住的模型。找不到就安静地留在默认模型上。"""
        saved = self.settings.model
        if not isinstance(saved, dict) or not saved.get("path"):
            return
        path = Path(saved["path"])
        for index, entry in enumerate(self.models):
            if entry.path is not None and entry.path == path:
                self.model_index = index     # 扫 models/ 时已经登记过, 直接用
                return
        if not path.exists():
            self.model_info = f"上次使用的模型已不存在，已改用默认模型：{path.name}"
            return
        try:
            info, model = trainer.load_checkpoint(path)
        except trainer.TrainError as exc:
            self.model_info = f"上次使用的模型加载失败，已改用默认模型：{exc}"
            return
        entry = ModelEntry(
            name=saved.get("name") or path.stem, path=path,
            labels=info["labels"], architecture=info["arch"],
            input_mode=info["input_mode"], mean=info["mean"], std=info["std"],
            accuracy=info["accuracy"], legacy=info["legacy"])
        entry.model = model                  # 权重已经在手上, 不用再读一遍
        self._register_model(entry, select=True)

    def _remember_current_model(self):
        entry = self.current_model
        if entry is None:
            return
        self.settings.remember_model(entry.name, entry.path)

    def _register_model(self, entry: ModelEntry, select: bool = False):
        """登记一个模型。路径相同的视为同一个模型, 直接替换。"""
        for index, existing in enumerate(self.models):
            if existing.path is not None and entry.path is not None \
                    and existing.path == entry.path:
                self.models[index] = entry
                if select:
                    self.model_index = index
                return
        self.models.append(entry)
        if select or len(self.models) == 1:
            self.model_index = len(self.models) - 1

    def _refresh_models_combo(self):
        values = [entry.display for entry in self.models]
        if hasattr(self, "model_combo"):
            # 主界面: 可以切换模型; 自定义模式里只显示当前模型名
            self.model_combo.configure(values=values)
        if not values:
            self.model_var.set("（无可用模型）")
            if hasattr(self, "model_combo"):
                self.model_combo.configure(state="disabled")
            self._set_model_hint("")
            return
        if hasattr(self, "model_combo"):
            self.model_combo.configure(state="readonly")
        self.model_index = max(0, min(self.model_index, len(values) - 1))
        self.model_var.set(values[self.model_index])
        self._refresh_model_hint()

    @property
    def current_model(self) -> ModelEntry | None:
        if not self.models:
            return None
        return self.models[self.model_index]

    def _on_model_selected(self, _event=None):
        index = self.model_combo.current()
        if index < 0 or index >= len(self.models):
            return
        self.model_index = index
        entry = self.models[index]
        try:
            entry.ensure_loaded()
        except trainer.TrainError as exc:
            messagebox.showerror("模型加载失败",
                                 f"无法加载模型「{entry.name}」：\n{exc}")
        self._refresh_models_combo()
        self._remember_current_model()      # 记住这次选择, 下次启动直接用它
        self.recognize()

    def import_model(self):
        """导入外部模型: 先确认读得动, 再复制进 `models/`。

        复制而不是原地引用, 是为了让 models/ 成为模型的唯一去处 —— 下次启动
        自动扫描就能看到它, 原文件挪走/删掉也不影响。
        """
        path = filedialog.askopenfilename(
            parent=self.root, title="导入模型（.pt）",
            filetypes=[("PyTorch 模型", "*.pt"), ("所有文件", "*.*")])
        if not path:
            return
        source = Path(path)
        try:
            info, model = trainer.load_checkpoint(source)
        except trainer.TrainError as exc:
            messagebox.showerror("导入失败", str(exc))
            return
        try:
            target = self._copy_into_models(source)
        except OSError as exc:
            messagebox.showerror("导入失败",
                                 f"无法将模型复制到 {MODELS_DIR}：\n{exc}")
            return
        if target is None:
            return                      # 用户在覆盖确认里选了「否」
        entry = ModelEntry(
            name=info["name"] or target.stem, path=target, labels=info["labels"],
            architecture=info["arch"], input_mode=info["input_mode"],
            mean=info["mean"], std=info["std"], accuracy=info["accuracy"],
            legacy=info["legacy"])
        entry.model = model             # 刚读出来的权重直接用, 不再读一遍
        self._register_model(entry, select=True)
        self._refresh_models_combo()
        self._remember_current_model()
        self._update_status()
        self.recognize()

    def _copy_into_models(self, source: Path) -> Path | None:
        """把外部 `.pt` 复制进 `models/`, 返回落地后的路径。

        已经在 models/ 里的文件原样返回(不用复制); 同名文件已存在时先问一句,
        免得悄悄盖掉训练好的模型。返回 None 表示用户放弃这次导入。
        """
        target = MODELS_DIR / source.name
        try:
            in_place = source.resolve() == target.resolve()
        except OSError:
            in_place = False
        if in_place:
            return source
        if target.exists() and not messagebox.askyesno(
                "覆盖同名模型",
                f"models 目录中已存在文件 {target.name}。\n\n"
                "是否用本次选择的文件覆盖它？"):
            return None
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target

    def _default_model_dir(self) -> str:
        """模型的默认存放目录(models/)。目录不存在就建出来 —— 否则文件对话框
        会忽略 initialdir。建不出来就退回程序目录。"""
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
        except OSError:
            return str(ROOT)
        return str(MODELS_DIR)

    # ======================================================================= #
    # 绘制
    # ======================================================================= #
    def _set_tool(self, cursor: str):
        self.canvas.configure(cursor=cursor)
        self._apply_metrics()

    def _is_eraser(self) -> bool:
        return self.tool_var.get() == "eraser"

    def _on_pen_change(self, _value):
        self.pen_lbl.configure(text=str(int(round(self.pen_size.get()))))
        self._apply_metrics()

    def _stamp(self, x: float, y: float):
        radius = self.brush_radius
        pad = radius + self.brush_soft + 1
        size = self.canvas_size
        x0, x1 = max(0, int(x - pad)), min(size, int(x + pad) + 1)
        y0, y1 = max(0, int(y - pad)), min(size, int(y + pad) + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        value = np.clip((radius - distance) / self.brush_soft + 0.5, 0.0,
                        1.0).astype(np.float32)
        region = self.arr[y0:y1, x0:x1]
        if self._is_eraser():
            self.arr[y0:y1, x0:x1] = np.minimum(region, 1.0 - value)
        else:
            self.arr[y0:y1, x0:x1] = np.maximum(region, value)

    def _draw_segment(self, x0, y0, x1, y1):
        import math
        distance = max(1.0, math.hypot(x1 - x0, y1 - y0))
        steps = int(distance) + 1
        for index in range(steps + 1):
            t = index / steps
            self._stamp(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)

    def _render(self):
        self._photo = ImageTk.PhotoImage(ink_to_pil(self.arr))
        self.canvas.itemconfigure(self.image_id, image=self._photo)
        self._refresh_thumbnail()

    def _refresh_thumbnail(self):
        """把画板上的当前图同步到侧边栏那一行的缩略图上。

        只改那一行(见 `DatasetSidebar.update_thumbnail`), 不重建列表, 所以边画
        边更新也不会闪、不用重开工程。识别模式没有"当前图片", 这里直接跳过。
        """
        if self.current_image is None:
            return
        partition, label, number = self.current_image
        self.sidebar.update_thumbnail(partition, label, number, self.arr)

    def _wants_live(self) -> bool:
        """是否边画边识别。自定义模式始终刷新, 识别模式看"实时识别"勾选框。"""
        return self.mode == MODE_CUSTOM or bool(self.live.get())

    def _live_now(self):
        """立刻识别一次(起笔 / 抬笔 / 清空这类"一次性"动作)。"""
        self._cancel_live()
        if self._wants_live():
            self.recognize()

    def _live_soon(self):
        """拖笔时的实时识别: 合并成每 LIVE_DELAY_MS 一次。

        鼠标划得快时产生的事件比我们能处理的还多, 每个事件都从头算一遍识别 + 重画
        列表, 事件队列只会越积越多, 最后画板和置信度一起卡住。隔一小段时间跑一次,
        中间的直接跳过 —— 反正下一次算的是最新墨迹, 结果不会旧。
        """
        if not self._wants_live() or self._live_job is not None:
            return
        self._live_job = self.root.after(LIVE_DELAY_MS, self._live_fire)

    def _live_fire(self):
        self._live_job = None
        self.recognize()

    def _cancel_live(self):
        if self._live_job is not None:
            self.root.after_cancel(self._live_job)
            self._live_job = None

    def _on_press(self, event):
        self.last_xy = (event.x, event.y)
        self._stamp(event.x, event.y)
        self._render()
        self._live_now()            # 起笔马上出结果, 不用等

    def _on_drag(self, event):
        if self.last_xy is not None:
            self._draw_segment(self.last_xy[0], self.last_xy[1], event.x, event.y)
        self.last_xy = (event.x, event.y)
        self._render()
        self._live_soon()

    def _on_release(self, _event):
        self.last_xy = None
        self._live_now()            # 抬笔补一次最终的: 拖笔期间可能被跳过了

    def clear(self):
        self.arr[:] = 0.0
        self.last_xy = None
        self._render()
        self.recognize()

    def get_ink(self) -> np.ndarray:
        """当前画板墨迹(深拷贝), 供保存/训练/入库使用。"""
        return self.arr.copy()

    def set_ink(self, ink: np.ndarray):
        """把一张图放到画板上(按画布尺寸重采样)。"""
        ink = np.asarray(ink, dtype=np.float32)
        if ink.shape != (self.canvas_size, self.canvas_size):
            img = Image.fromarray((ink * 255).astype(np.uint8), mode="L")
            image = img.resize((self.canvas_size, self.canvas_size), Image.LANCZOS)
            ink = np.asarray(image, dtype=np.float32) / 255.0
        self.arr = np.ascontiguousarray(ink)
        self.last_xy = None
        # 换图这一瞬间先断掉"当前图片"的关联: 马上要画的这次 _render 拿的是**新**
        # 墨迹, 不能拿去盖上一张图的缩略图。调用方紧接着会把正确的编号设回来。
        self.current_image = None
        self._render()

    # ======================================================================= #
    # 识别与匹配度
    # ======================================================================= #
    def recognize(self):
        """识别当前画板, 并刷新"匹配度"列表。"""
        entry = self.current_model
        probs = None
        if entry is not None:
            try:
                model = entry.ensure_loaded()
                probs = trainer.predict_proba(model, self.arr, entry.input_mode,
                                              entry.mean, entry.std)
            except trainer.TrainError as exc:
                self._set_model_hint(str(exc), error=True)
                probs = None

        if probs is None:
            self.result_var.set("—")
            self.conf_var.set("置信度：—")
            self._refresh_match_list(None)
            return
        self._refresh_model_hint()      # 上一次报的错现在没了, 说明文字要恢复
        best = int(np.argmax(probs))
        label = entry.labels[best] if best < len(entry.labels) else str(best)
        self.result_var.set(label)
        self.conf_var.set(f"置信度：{probs[best] * 100:.1f}%")
        self._refresh_match_list(probs)

    def _refresh_match_list(self, probs=None):
        """刷新"匹配度"列表。两页说的不是一回事, 但**都不排序**:

            识别模式   —— 模型对**每一个输出类**的置信度, 按模型标签表的顺序
            自定义模式 —— 画板上的图对**每一个数据集**的匹配度, 按数据集名的顺序

        不排序是有意的: 每行固定待在原地, 边写边识别时列表不会整块跳来跳去。
        识别模式也不能去翻工作区的数据集: 那里通常是空的(没建过数据集)。

        **行本身没变时只改数字和进度条**, 不销毁重建控件 —— 这个方法每落一笔都会
        被调一次, 重建 10 行就是 40 个控件, 鼠标划快时事件堆积, 这块根本来不及画出来。
        行只在切页 / 换模型 / 增删数据集时重建。
        """
        if not hasattr(self, "match_scroll"):
            return
        note, rows = (self._dataset_rows(probs) if self.mode == MODE_CUSTOM
                      else self._class_rows(probs))
        # 行的高矮胖瘦也跟着 DPI 走, 所以把条宽也算进"结构"里: 换显示器/缩放时重建
        structure = (self.mode, self.metrics.px(MATCH_BAR_W), note,
                     tuple((label, marked) for label, _value, marked in rows))
        if structure != self._match_structure:
            self._build_match_rows(note, rows, structure)
        self._paint_match_rows(rows)

    # ---------------------- 识别模式: 模型全部类别 ---------------------- #
    def _class_rows(self, probs):
        """识别模式要显示的内容: 模型全部输出类, 顺序就是模型标签表的顺序。

        返回 (提示文字 或 None, [(标签, 概率, 是不是自己数据集里的类)])。
        属于自己数据集的标签会多一个圆点。
        """
        entry = self.current_model
        if entry is None:
            return "当前无可用模型\n请点击模型右侧的「导入」按钮选择 .pt 文件", []
        if probs is None:
            return "请在画板上书写字符，此处将显示各类别的置信度", []
        mine = set(self.workspace.labels())
        values = [float(probs[i]) if i < len(probs) else 0.0
                  for i in range(len(entry.labels))]
        return None, [(label, value, label in mine)
                      for label, value in zip(entry.labels, values)]

    # ---------------------- 自定义模式: 各个数据集 ---------------------- #
    def _dataset_rows(self, probs):
        """自定义模式要显示的内容: 每个数据集一行, 顺序就是数据集名的顺序。

        当前模型里没有的标签概率记 None, 会显示成"模型未覆盖"。
        """
        labels = self.workspace.labels()
        if not labels:
            return "尚未创建数据集\n请先在侧边栏点击「新建数据集」", []
        entry = self.current_model
        if probs is None:
            return "请在画板上书写字符，此处将显示各数据集的匹配度", []
        index_of = {label: i for i, label in enumerate(entry.labels)}
        rows = []
        for label in labels:
            column = index_of.get(label)
            covered = column is not None and column < len(probs)
            rows.append((label, float(probs[column]) if covered else None, False))
        return None, rows

    # ------------------------------ 建行 / 刷行 ------------------------------ #
    def _build_match_rows(self, note, rows, structure) -> None:
        """**重建**列表(只在行本身变了的时候走这里)。"""
        scroll = self._match_scroll()
        scroll.clear()
        self._match_rows = {}
        self._match_structure = structure
        if note is not None:
            self._match_note(scroll, note)
            return
        for label, _value, marked in rows:
            self._match_rows[label] = self._make_match_row(scroll, label, marked)

    def _make_match_row(self, scroll, label: str, marked: bool) -> dict:
        """搭出一行的控件并记住它们, 之后只改不重建。"""
        px = self.metrics.px
        bar_w = px(MATCH_BAR_W)
        bar_h = px(MATCH_BAR_H)
        row = ttk.Frame(scroll.body)
        row.pack(fill="x", pady=px(1))

        # 固定宽度的圆点列: 有就画, 没有也占位, 保证所有行的标签左边缘对齐
        tk.Label(row, text="●" if marked else "", width=2,
                 font=("Segoe UI", 8), fg=label_color(label),
                 bg=self._bg(row)).pack(side="left")

        tk.Label(row, text=f" {label} ", font=("Consolas", 11, "bold"),
                 fg="white", bg=label_color(label), width=2).pack(side="left")

        canvas = tk.Canvas(row, width=bar_w, height=bar_h,
                           highlightthickness=0, bg=self._bg(row))
        canvas.pack(side="left", padx=(px(6), px(6)))
        canvas.create_rectangle(0, 0, bar_w, bar_h, outline=COLOR_EMPTY,
                                fill="#f4f4f4")
        # 进度条先按 0 宽建好, 刷新时只改坐标, 不再增删图元
        bar = canvas.create_rectangle(0, 0, 0, 0, outline="",
                                      fill=label_color(label))
        pct = ttk.Label(row, text="—", font=("Consolas", 9), width=7, anchor="e")
        pct.pack(side="left")
        return {"bar": canvas, "fill": bar, "bar_w": bar_w, "bar_h": bar_h,
                "pct": pct, "empty": None}

    def _paint_match_rows(self, rows) -> None:
        """只改数字 / 条长 / 高亮 —— 每落一笔都会走这里, 必须便宜。"""
        top = max((value for _label, value, _marked in rows if value is not None),
                  default=None)
        for label, value, _marked in rows:
            record = self._match_rows.get(label)
            if record is None:
                continue
            canvas = record["bar"]
            fill = record["fill"]
            if value is None:
                if record["empty"] is None:
                    record["empty"] = canvas.create_text(
                        record["bar_w"] // 2, record["bar_h"] // 2,
                        text="模型未覆盖", font=("Segoe UI", 7), fill=COLOR_MUTED)
                canvas.coords(fill, 0, 0, 0, 0)
                record["pct"].configure(text="—")
                continue
            if record["empty"] is not None:
                canvas.delete(record["empty"])
                record["empty"] = None
            canvas.coords(fill, 0, 0,
                          max(0.0, min(1.0, value)) * record["bar_w"],
                          record["bar_h"])
            canvas.itemconfigure(
                fill, fill=COLOR_PASS if top and value == top
                else label_color(label))
            record["pct"].configure(text=f"{value * 100:5.1f}%")

    def _match_note(self, scroll, text: str) -> None:
        """列表里的提示文字(没有模型 / 还没有数据集 / 还没下笔)。"""
        px = self.metrics.px
        ttk.Label(scroll.body, text=text, font=("Segoe UI", 9),
                  foreground=COLOR_MUTED, justify="left",
                  wraplength=px(MATCH_BAR_W + 130)).pack(anchor="w", pady=px(8))

    def _bg(self, widget) -> str:
        try:
            return widget.cget("bg") or "#f0f0f0"
        except tk.TclError:
            return "#f0f0f0"

    # ======================================================================= #
    # 侧边栏回调
    # ======================================================================= #
    def sidebar_partition_changed(self, partition: str):
        self._update_status()

    def sidebar_open_image(self, partition: str, label: str, number: int,
                           focus_canvas: bool = False):
        dataset = self.workspace.partition(partition).find(label)
        if dataset is None or not dataset.has(number):
            return
        self.set_ink(dataset.ink(number))
        self.current_image = (partition, label, number)
        self._update_status()
        self.recognize()
        if focus_canvas:
            self.canvas.focus_set()

    def sidebar_create_dataset(self, partition: str, label: str):
        if not is_valid_label(label):
            messagebox.showerror("数据集名称不合法", "数据集名称必须为单个字符。")
            return
        target = self.workspace.partition(partition)
        if label in target:
            # 同名数据集在**这个分区**里已经有了(另一个分区有不算): 说明一下就好,
            # 别让 Partition.create 抛出来把界面打断
            messagebox.showinfo(
                "数据集已存在",
                f"{PARTITION_NAMES[partition]}中已存在数据集「{label}」，"
                "无需重复创建。")
            return
        target.create(label)
        self.workspace.mark_dirty()
        number = self.workspace.add_sample(partition, label, np.zeros(
            (self.canvas_size, self.canvas_size), dtype=np.float32))
        self.sidebar.expand(label)          # 展开它, 新图这一行才看得见
        self.sidebar.refresh()
        self.sidebar.select(partition, label, number)
        self.set_ink(self.workspace.partition(partition).ink(label, number))
        self.current_image = (partition, label, number)
        self._refresh_title()
        self._update_status()
        self.recognize()

    def sidebar_delete_dataset(self, label: str):
        if self.current_image is not None and self.current_image[1] == label:
            self.current_image = None
        self.workspace.remove_dataset(label)
        self.workspace.mark_dirty()
        self.sidebar.forget(label)          # 清选中/展开态, 顺手扔掉缩略图
        self.sidebar.refresh()
        self.architecture["num_classes"] = max(2, len(self.workspace.labels()))
        self._refresh_title()
        self._update_status()
        self.recognize()

    def sidebar_open_arch(self):
        self.open_arch_dialog()

    def sidebar_run_test(self):
        self.run_test()

    def sidebar_save_project(self):
        self.save_project()

    def sidebar_open_project(self):
        self.open_project()

    # ======================================================================= #
    # 图片 / 数据集操作
    # ======================================================================= #
    def _stash_canvas(self) -> tuple[str, str, int] | None:
        """把画板当前内容写回工作区, 返回它对应的 (分区, 数据集, 编号)。

        改动图之前必须先调用它, 否则会丢笔。
        """
        if self.current_image is None:
            return None
        partition, label, number = self.current_image
        self.workspace.partition(partition).put(label, number, self.get_ink())
        return self.current_image

    def next_image(self):
        """在**当前打开图片所在的数据集**里新建一张空画板。"""
        partition, label, _number = self.sidebar.get_context()
        if label is None:
            labels = self.workspace.partition(partition).labels()
            if not labels:
                messagebox.showinfo(
                    "尚未创建数据集",
                    "请先在侧边栏点击「新建数据集」创建数据集。")
                return
            label = labels[0]
        self._stash_canvas()
        number = self.workspace.add_sample(
            partition, label, np.zeros((self.canvas_size, self.canvas_size),
                                       dtype=np.float32))
        self.sidebar.expand(label)          # 展开它, 新图这一行才看得见
        self.sidebar.refresh()
        self.sidebar.select(partition, label, number)
        self.set_ink(self.workspace.partition(partition).ink(label, number))
        self.current_image = (partition, label, number)
        self._refresh_title()
        self._update_status()
        self.recognize()
        self.canvas.focus_set()

    def delete_image(self):
        """删除**当前打开**的这张图。

        编号**不重排**: 删掉 3 号之后数据集里就是 0、1、2、4, 空出来的 3 号留给
        下次「下一张」补上(`DigitSet.next_number` 会找最小的空号)。
        """
        if self.current_image is None:
            messagebox.showinfo(
                "未打开任何图片",
                "请先在侧边栏中打开一张图片，再执行删除。")
            return
        partition, label, number = self.current_image
        dataset = self.workspace.partition(partition).find(label)
        if dataset is None or not dataset.has(number):
            # 图已经不在了(例如数据集刚被删掉), 清掉引用就行
            self.current_image = None
            return
        if not messagebox.askyesno(
                "删除图片",
                f"确定要删除{PARTITION_NAMES[partition]}中数据集「{label}」的 "
                f"{number} 号图片吗？\n\n"
                "图片编号不会重排；下次点击「下一张」时将复用该编号。"):
            return

        dataset.remove(number)
        self.workspace.mark_dirty()
        self.sidebar.drop_thumbnail(partition, label, number)
        self.sidebar.clear_selection()
        self.current_image = None
        self.arr[:] = 0.0               # 画板跟着清掉, 免得还以为是那张图
        self.sidebar.refresh()
        self._render()
        self._refresh_title()
        self._update_status()
        self.recognize()
        self.canvas.focus_set()

    # ======================================================================= #
    # 保存 / 打开工程
    # ======================================================================= #
    def _default_project_name(self) -> str:
        labels = self.workspace.labels()
        if labels:
            return "painter_" + "".join(labels[:6]) + ".ptp"
        return "painter.ptp"

    def _default_project_dir(self) -> str:
        """新工程默认存哪儿: 已经存过的就跟着旧文件走, 否则落在程序目录。

        程序目录是自己写得进去的地方(模型、settings.json 都在这儿), 比让
        文件对话框随便挑一个没权限的目录要靠谱。
        """
        if self.workspace.path is not None:
            return str(Path(self.workspace.path).parent)
        return str(ROOT)

    def _current_model_ref(self) -> dict | None:
        entry = self.current_model
        if entry is None:
            return None
        return {"name": entry.name,
                "pt": str(entry.path) if entry.path else None,
                "labels": entry.labels,
                "accuracy": entry.accuracy}

    def save_project(self, ask_path: bool | None = None) -> bool:
        """保存为 .ptp。返回是否真的保存了。

        路径是这么定的:
            · 新工程(还没存过)          -> 问一次存哪儿
            · 打开过 / 存过的工程        -> **直接覆盖**原来那个文件, 不再弹框
            · 那个位置写不进去(权限/占用/盘满) -> 说明原因, 再问一个新的路径
        `ask_path=True` 会强制弹框(相当于「另存为」), `False` 则只往原路径写。
        """
        if self.training:
            messagebox.showinfo("正在训练", "请等待训练结束后再保存工程。")
            return False
        self._stash_canvas()
        if ask_path is None:
            ask_path = self.workspace.path is None
        path = None if ask_path else self.workspace.path

        while True:
            if not path:
                path = filedialog.asksaveasfilename(
                    parent=self.root, title="保存工程（.ptp）",
                    defaultextension=".ptp",
                    filetypes=[("Painter 工程", "*.ptp"), ("所有文件", "*.*")],
                    initialfile=self._default_project_name(),
                    initialdir=self._default_project_dir())
                if not path:
                    return False        # 用户取消
            try:
                out = ptp.save_ptp(path, self.workspace, canvas=self.canvas_size,
                                   arch=self.architecture,
                                   model=self._current_model_ref(),
                                   base_info=self.workspace.modelinfo)
                break
            except OSError as exc:
                # 权限不足最常见(例如没权限写这个目录)。说清楚, 再给一次机会
                # 换个位置 —— 用户也可以直接取消文件对话框放弃保存。
                if not messagebox.askretrycancel(
                        "无法保存到该位置",
                        f"无法写入以下文件：\n{path}\n\n{exc}\n\n"
                        "请点击「重试」并选择其他保存位置。"):
                    return False
                path = None

        self.workspace.mark_clean(out)
        self.workspace.modelinfo = ptp.read_modelinfo(out)
        self._refresh_title()
        self._update_status()
        size_kb = out.stat().st_size / 1024
        self.model_info = f"已保存：{out.name}（{size_kb:.1f} KB）"
        self._update_status()
        return True

    def open_project(self):
        if not self._confirm_discard("打开工程"):
            return
        path = filedialog.askopenfilename(
            parent=self.root, title="打开工程（.ptp）",
            filetypes=[("Painter 工程", "*.ptp"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            workspace, info, report = ptp.load_ptp(path)
        except ptp.PtPError as exc:
            messagebox.showerror("打开失败", str(exc))
            return

        self.workspace = workspace
        self.workspace.path = Path(path)
        self.workspace.modelinfo = info
        self.sidebar.workspace = workspace
        self.sidebar.thumbs.clear()
        self.sidebar.clear_selection()
        self.sidebar.collapse()
        self.current_image = None
        self.arr[:] = 0.0

        if isinstance(info.get("arch"), dict) and info["arch"].get("cards"):
            self.architecture = arch_mod.normalize_arch(
                info["arch"], max(2, len(workspace.labels())))

        self.sidebar.set_partition(PARTITION_TRAIN)
        self.sidebar.refresh()
        self._render()
        self._refresh_title()
        self._update_status()
        self.recognize()

        message = f"已打开 {Path(path).name}\n\n{report.summary}"
        detail = report.detail()
        if detail:
            message += f"\n\n{detail}"
        model_ref = info.get("model")
        if isinstance(model_ref, dict) and model_ref.get("pt"):
            known = any(entry.path and str(entry.path) == model_ref["pt"]
                        for entry in self.models)
            if not known:
                message += (f"\n\n提示：该工程原先使用的模型为「{model_ref.get('name')}」，"
                            f"请通过「导入」按钮加载 {model_ref['pt']}。")
        (messagebox.showinfo if not report.skipped else messagebox.showwarning)(
            "打开工程", message)

    def _confirm_discard(self, action: str) -> bool:
        """有未保存改动时先问一句。返回是否可以继续。"""
        if not self.workspace.dirty:
            return True
        answer = messagebox.askyesnocancel(
            action,
            "当前工程存在未保存的改动。\n\n"
            "选择「是」将先保存工程；选择「否」将放弃改动并继续。")
        if answer is None:
            return False
        if answer:
            return self.save_project()
        return True

    # ======================================================================= #
    # 模型参数
    # ======================================================================= #
    def open_arch_dialog(self):
        from ui_arch import ArchDialog
        num_classes = max(2, len(self.workspace.labels()) or 2)
        architecture = self.architecture
        if architecture.get("num_classes") != num_classes:
            architecture = arch_mod.normalize_arch(architecture, num_classes)
        dialog = ArchDialog(self.root, architecture, num_classes,
                            metrics=self.metrics,
                            canvas_height=int(self.root.winfo_height()))
        result = dialog.show()
        if isinstance(result, dict):
            self.architecture = result
            self.workspace.mark_dirty()
            self.model_info = f"架构：{self._arch_summary(result)}"
            self._update_status()

    def _arch_summary(self, architecture: dict) -> str:
        """架构的**短**摘要, 专供状态栏用。

        `arch_mod.describe_arch()` 会把每张卡片都写出来(一百多个字符)。状态栏是
        单行文字, 而它的自然宽度会直接决定窗口宽度 —— 这么长的摘要会把自定义模式
        整个撑宽一圈。所以状态栏只报层数和参数量, 完整链路在「模型参数」对话框里
        本来就看得到。
        """
        cards = arch_mod.normalize_arch(architecture).get("cards", [])
        text = f"{len(cards)} 层"
        try:
            model = arch_mod.build_model(architecture)
        except arch_mod.ArchError:
            return text
        return f"{text} / {arch_mod.count_parameters(model):,} 参数"

    # ======================================================================= #
    # 训练
    # ======================================================================= #
    def start_training(self):
        if self.training:
            messagebox.showinfo("正在训练", "已有训练任务正在进行。")
            return
        train_labels = self.workspace.train_labels()
        counts = self.workspace.train_class_counts()
        if len(train_labels) < 2:
            messagebox.showinfo(
                "暂无法训练",
                "训练集中至少需要两个数据集（类别）。\n\n"
                "请先在侧边栏点击「新建数据集」创建数据集，"
                "并为每个数据集绘制若干样本。")
            return
        empty = [label for label, count in counts.items() if count == 0]
        if empty:
            messagebox.showinfo(
                "暂无法训练",
                "以下数据集在训练集中没有任何图片：\n\n"
                f"  {' '.join(empty)}\n\n请先补充样本，或删除这些数据集。")
            return
        if not self.architecture.get("cards"):
            messagebox.showinfo("暂无法训练",
                                "网络结构中没有任何层，请先在「模型参数」中添加。")
            return
        errors = arch_mod.validate_arch(
            {**self.architecture, "num_classes": len(train_labels)})
        if errors:
            messagebox.showerror(
                "网络结构不合法",
                "当前网络结构不合法，请先修正：\n\n"
                + "\n".join(f"  · {error}" for error in errors[:8]))
            return

        self._stash_canvas()
        self.sidebar.refresh()
        mode = self.architecture.get("input_mode", "center")
        suggested = f"painter_{''.join(train_labels[:6])}.pt"
        path = filedialog.asksaveasfilename(
            parent=self.root, title="保存训练好的模型（.pt）",
            defaultextension=".pt", initialfile=suggested,
            initialdir=self._default_model_dir(),
            filetypes=[("PyTorch 模型", "*.pt"), ("所有文件", "*.*")])
        if not path:
            return

        config = trainer.TrainConfig(
            epochs=trainer.suggest_epochs(trainer.load_samples(
                self.workspace, PARTITION_TRAIN)),
            batch_size=32, lr=1e-3, augment=True, seed=0, input_mode=mode)
        self._set_training(True)
        try:
            from ui_train import TrainDialog
            dialog = TrainDialog(self.root, self.workspace, self.architecture,
                                 config, metrics=self.metrics,
                                 default_name=Path(path).stem,
                                 default_path=str(Path(path).parent))
            dialog.attach_target(path)
            dialog.start()
            result = dialog.show()
        finally:
            self._set_training(False)

        if result and result.get("saved") and result.get("path"):
            info = result["info"]
            entry = ModelEntry(
                name=Path(result["path"]).stem, path=Path(result["path"]),
                labels=info["labels"], architecture=info["arch"],
                input_mode=info["input_mode"], mean=info["mean"],
                std=info["std"], accuracy=info.get("accuracy"))
            # 刚训练出来的权重就在内存里, 直接装进模型, 不用再从磁盘读一遍
            try:
                model = arch_mod.build_model(
                    {**info["arch"], "num_classes": len(info["labels"])})
                entry.model = trainer.install_state_dict(
                    model, info["state_dict"], architecture=info["arch"])
                entry.model.eval()
            except (trainer.TrainError, arch_mod.ArchError) as exc:
                # 内存里装不上就回退到从文件读, 保证模型一定可用
                entry.model = None
                self.model_info = f"训练结果已保存，正在重新读取模型：{exc}"
            self._register_model(entry, select=True)
            self._refresh_models_combo()
            self._remember_current_model()
            self.workspace.modelinfo = ptp.build_modelinfo(
                self.workspace, canvas=self.canvas_size, arch=self.architecture,
                model=self._current_model_ref(), base_info=self.workspace.modelinfo)
            self.workspace.mark_dirty()
            accuracy = info.get("accuracy")
            accuracy_text = "—" if accuracy is None else f"{accuracy * 100:.1f}%"
            self.model_info = (f"已训练并切换模型：{entry.name}    "
                               f"测试集通过率 {accuracy_text}")
            self._update_status()
            self.recognize()
        elif result is None:
            self.model_info = "训练已取消或未保存"
            self._update_status()

    def _set_training(self, busy: bool):
        self.training = busy
        state = ["disabled"] if busy else ["!disabled"]
        for button in (self.btn_next, self.btn_delete, self.btn_clear,
                       self.btn_train, self.btn_test, self.btn_arch):
            if button is not None:      # 另一页的那几个按钮不存在
                button.state(state)
        self.sidebar.set_busy(busy)

    # ======================================================================= #
    # 测试
    # ======================================================================= #
    def run_test(self):
        entry = self.current_model
        if entry is None:
            messagebox.showinfo("当前无可用模型", "请先训练或导入模型。")
            return
        partition = self.workspace.partition(PARTITION_TEST)
        if partition.image_count() == 0:
            messagebox.showinfo("无法测试", "测试集中没有任何图片。")
            return
        self._stash_canvas()
        try:
            model = entry.ensure_loaded()
        except trainer.TrainError as exc:
            messagebox.showerror("模型加载失败", str(exc))
            return

        entries: list[dict] = []
        for label in partition.labels():
            dataset = partition.dataset(label)
            for number in dataset.indices():
                ink = dataset.ink(number)
                probs = trainer.predict_proba(model, ink, entry.input_mode,
                                              entry.mean, entry.std)
                if probs is None:
                    predicted, confidence = "（空白）", None
                else:
                    best = int(np.argmax(probs))
                    predicted = (entry.labels[best] if best < len(entry.labels)
                                 else str(best))
                    confidence = float(probs[best])
                entries.append({
                    "partition": PARTITION_TEST, "label": label, "number": number,
                    "predicted": predicted, "confidence": confidence,
                    "passed": predicted == label, "ink": ink,
                })

        from ui_test import show_test_results
        show_test_results(self.root, entries, metrics=self.metrics,
                          model_name=entry.name, on_open=self._open_from_result)

    def _open_from_result(self, item: dict):
        self.sidebar.set_partition(item["partition"])
        self.sidebar.expand(item["label"])
        self.sidebar.refresh()
        self.sidebar_open_image(item["partition"], item["label"], item["number"])
        self.sidebar.select(item["partition"], item["label"], item["number"])
        self.root.lift()

    # ======================================================================= #
    # 状态栏 / 关闭
    # ======================================================================= #
    def _update_status(self):
        """状态栏 + 自定义模式里的训练提示。所有改动数据的地方都会调它。"""
        model_name = self.current_model.display if self.current_model else "无可用模型"
        if self.mode == MODE_CUSTOM:
            project = self.workspace.path.name if self.workspace.path else "（未保存）"
            dirty = " *" if self.workspace.dirty else ""
            text = (f"工程：{project}{dirty}    {self.workspace.describe()}    "
                    f"模型：{model_name}    {self.model_info}    "
                    f"显示缩放：{self.scale * 100:.0f}%（{self.dpi:.0f} DPI）")
        else:
            text = (f"模型：{model_name}    {self.model_info}    "
                    f"显示缩放：{self.scale * 100:.0f}%（{self.dpi:.0f} DPI）")
        self._status_full = text        # 完整文字留着: 定窗口宽度时要按它量
        self.status_var.set(self._fit_status_text(text))
        self._refresh_train_hint()

    def _status_text_width(self, text: str) -> int:
        """按状态栏的字号量一段文字有多宽(像素)。"""
        return tkfont.Font(font=self.status_lbl.cget("font")).measure(text)

    def _fit_status_text(self, text: str) -> str:
        """把状态栏文字裁到钉死的宽度以内, 超出的部分用省略号收尾。

        窗口宽度是固定的(见 `_apply_page_size`), 状态栏太长会被窗口边缘切掉半个字;
        截断至少看得出"后面还有"。
        """
        limit = getattr(self, "_status_width", 0)
        if limit <= 0:
            return text
        limit -= self.metrics.px(4)                 # 留一点余量
        font = tkfont.Font(font=self.status_lbl.cget("font"))
        if font.measure(text) <= limit:
            return text                             # 放得下就原样显示
        ellipsis = "…"
        room = limit - font.measure(ellipsis)
        cut = len(text)
        while cut > 0 and font.measure(text[:cut]) > room:
            cut -= 1
        return text[:cut] + ellipsis

    def _refresh_title(self):
        if self.mode == MODE_CUSTOM:
            name = self.workspace.path.name if self.workspace.path else "未命名工程"
            mark = " *" if self.workspace.dirty else ""
            self.root.title(f"字符识别平台 — {name}{mark}")
        else:
            self.root.title("字符识别平台")

    def on_close(self):
        if self.training:
            if not messagebox.askokcancel(
                    "正在训练",
                    "训练仍在进行中，确定要退出吗？\n"
                    "（训练将被中断，结果不会保存）"):
                return
        if self.workspace.dirty:
            answer = messagebox.askyesnocancel(
                "退出",
                "当前工程存在未保存的改动，退出前是否保存？\n\n"
                "选择「是」保存为 .ptp 文件；选择「否」直接退出。")
            if answer is None:
                return
            if answer and not self.save_project():
                return
        self.root.destroy()


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
