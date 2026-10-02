"""自定义训练模式的侧边栏 —— 三段式。

    上: 分区选择框(训练集 / 测试集)
    中: 滚动区, 每个数据集一个可折叠块, 块里是该数据集下的图片(缩略图 + 编号)
    下: 新建数据集 / 删除数据集 / 模型参数 / 打开工程

图片打开方式是**单击或双击列表项**(不用标签页)。所有业务动作都交给
`controller`(主窗口)处理, 本文件只负责画界面和转发事件。
"""

import tkinter as tk
from tkinter import messagebox, ttk

from dataset import PARTITION_NAMES, PARTITIONS, is_valid_label
from ui_widgets import (
    COLOR_MUTED, ScrollFrame, ThumbCache, ask_single_char, ask_yes_no, fit_size,
    label_color,
)

THUMB_BOX = 30          # 缩略图显示区(逻辑像素)
PANEL_WIDTH = 250       # 侧边栏宽度(逻辑像素)
HEADER_BG = "#f4f7fa"   # 数据集卡片默认底色
HEADER_HOVER = "#e0eef9"    # 悬浮
HEADER_PRESS = "#cce4f7"    # 按下
CARD_BORDER = "#d8dee6"


class DatasetSidebar(ttk.Frame):
    """侧边栏。`controller` 需要提供下面这些方法(见 app.DigitApp 的实现):

        sidebar_context() -> (partition, label|None, number|None)
        sidebar_partition_changed(partition)
        sidebar_open_image(partition, label, number, focus_canvas=False)
        sidebar_create_dataset(partition, label)
        sidebar_delete_dataset(label)
        sidebar_save_project()
        sidebar_open_project()
    """

    def __init__(self, parent, controller, metrics, workspace):
        super().__init__(parent, padding=0)
        self.controller = controller
        self.metrics = metrics
        self.workspace = workspace
        self.thumbs = ThumbCache(self)
        self.expanded: set[str] = set()      # 已展开的数据集名
        self._partition = PARTITIONS[0]
        self._selected: tuple[str, int] | None = None

        px = metrics.px
        width = px(PANEL_WIDTH)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # ---------------------------- 上: 分区 ---------------------------- #
        top = ttk.LabelFrame(self, text="分区", padding=px(6))
        top.grid(row=0, column=0, sticky="ew")
        self.partition_var = tk.StringVar(value=self._partition)
        for index, name in enumerate(PARTITIONS):
            ttk.Radiobutton(
                top, text=PARTITION_NAMES[name], value=name,
                variable=self.partition_var, command=self._on_partition_change,
            ).grid(row=0, column=index, sticky="w", padx=(0, px(12)))

        # ---------------------------- 中: 数据集 ---------------------------- #
        self.scroll = ScrollFrame(self, width=width, height=px(230))
        self.scroll.grid(row=1, column=0, sticky="nsew", pady=(px(6), px(6)))

        style = ttk.Style(self)
        self._thumb_px = px(THUMB_BOX)
        style.configure("Sidebar.Treeview", rowheight=self._thumb_px + px(6),
                        font=("Segoe UI", 10))

        # ---------------------------- 下: 按钮 ---------------------------- #
        # 训练/测试/模型参数在自定义模式页的右侧栏里, 这里只放数据集与工程操作
        bottom = ttk.Frame(self)
        bottom.grid(row=2, column=0, sticky="ew")
        for column in range(2):
            bottom.columnconfigure(column, weight=1)
        self.btn_new = ttk.Button(bottom, text="新建数据集", command=self._new_dataset)
        self.btn_new.grid(row=0, column=0, sticky="ew", pady=(0, px(4)),
                          padx=(0, px(3)))
        self.btn_delete = ttk.Button(bottom, text="删除数据集",
                                     command=self._delete_dataset)
        self.btn_delete.grid(row=0, column=1, sticky="ew", pady=(0, px(4)),
                             padx=(px(3), 0))
        self.btn_save = ttk.Button(bottom, text="保存工程（.ptp）",
                                   command=lambda: self.controller.sidebar_save_project())
        self.btn_save.grid(row=1, column=0, sticky="ew", padx=(0, px(3)))
        self.btn_project = ttk.Button(bottom, text="打开工程（.ptp）",
                                      command=lambda: self.controller.sidebar_open_project())
        self.btn_project.grid(row=1, column=1, sticky="ew", padx=(px(3), 0))

        self.refresh()

    # ------------------------------ 状态 ------------------------------ #
    @property
    def partition(self) -> str:
        return self._partition

    def selected_label(self) -> str | None:
        if self._selected is None:
            return None
        return self._selected[0]

    def selected_number(self) -> int | None:
        if self._selected is None:
            return None
        return self._selected[1]

    def get_context(self) -> tuple[str, str | None, int | None]:
        return self._partition, self.selected_label(), self.selected_number()

    def set_partition(self, name: str) -> None:
        if name not in PARTITIONS or name == self._partition:
            return
        self._partition = name
        self.partition_var.set(name)
        self.refresh()

    def clear_selection(self) -> None:
        self._selected = None

    def forget(self, label: str) -> None:
        """某个数据集没了: 清掉和它相关的选中态与展开态, 免得后面拿着旧名字用。"""
        if self._selected is not None and self._selected[0] == label:
            self._selected = None
        if self._open_label == label:
            self._open_label = None
        self.drop_thumbnails(label)

    def select(self, partition: str, label: str, number: int) -> None:
        """选中某张图。属于别分区时先切过去, 否则选中态会和画面不一致。"""
        if partition != self._partition:
            self.set_partition(partition)
        self._selected = (label, number)
        self._sync_tree_selection()

    def set_busy(self, busy: bool) -> None:
        """训练进行中禁掉会改动数据的入口(工程按钮仍可用)。"""
        state = ["disabled"] if busy else ["!disabled"]
        for button in (self.btn_new, self.btn_delete):
            button.state(state)

    # ------------------------------ 事件 ------------------------------ #
    def _on_partition_change(self):
        self._partition = self.partition_var.get()
        self.refresh()
        self.controller.sidebar_partition_changed(self._partition)

    def _new_dataset(self):
        # 只在**当前分区**里查重: 同一个类别本来就该在训练集和测试集里各有一份,
        # 拿整个工作区的名字去拦, 会导致训练集建过的名字在测试集里建不了。
        taken = set(self.workspace.partition(self._partition).labels())
        label = ask_single_char(
            self, "新建数据集", "数据集名称（单个字符，即分类标签）",
            taken, f"将在{PARTITION_NAMES[self._partition]}中新建数据集，"
                   "并打开一张空白画板。")
        if label is None:
            return
        if not is_valid_label(label):
            return
        self.controller.sidebar_create_dataset(self._partition, label)
        self._open_label = label
        self.refresh()

    def _delete_dataset(self):
        """删除**当前选中的**数据集。

        没有选中就明确提示, 绝不自己挑一个 —— 更不能拿着已经被删掉的名字
        去问"确定删除 xxx 吗, 共 0 张图片"。
        """
        labels = self.workspace.partition(self._partition).labels()
        if not labels:
            messagebox.showinfo(
                "没有可删除的数据集",
                f"{PARTITION_NAMES[self._partition]}中尚未创建数据集。",
                parent=self)
            return

        label = self.selected_label()
        if label is None or label not in labels:
            if label is not None:
                # 选中项已经不在工作区里了(例如刚被删掉) -> 顺手清掉
                self._selected = None
                self.refresh()
            messagebox.showinfo(
                "请先选择数据集",
                "删除前请先在列表中展开并选中一个数据集。\n\n"
                f"（当前{PARTITION_NAMES[self._partition]}包含：{' '.join(labels)}）",
                parent=self)
            return

        train = self.workspace.partition(PARTITIONS[0]).find(label)
        test = self.workspace.partition(PARTITIONS[1]).find(label)
        counts = (0 if train is None else len(train)) + (0 if test is None else len(test))
        if not ask_yes_no(
                self, "删除数据集",
                f"确定要删除数据集「{label}」吗？\n\n"
                f"该操作会同时从训练集与测试集中删除，共 {counts} 张图片，"
                "且无法撤销。"):
            return
        self.controller.sidebar_delete_dataset(label)

    def _on_tree_click(self, event):
        """点列表项打开图片。用鼠标坐标找点了哪一行。"""
        tree = event.widget
        item = tree.identify_row(event.y)
        if not item:
            return
        label, number = self._item_target(tree, item)
        if label is None or number is None:
            return
        tree.selection_set(item)
        tree.focus(item)
        self._selected = (label, number)
        self.controller.sidebar_open_image(self._partition, label, number)

    def _on_tree_activate(self, _event=None):
        label, number = self._resolve_selection()
        if label is None or number is None:
            return
        self._selected = (label, number)
        self.controller.sidebar_open_image(self._partition, label, number,
                                           focus_canvas=True)

    def _item_target(self, tree, item) -> tuple[str | None, int | None]:
        """从 Treeview 的条目还原 (数据集, 编号)。"""
        for label, candidate in getattr(self, "_trees", {}).items():
            if candidate is not tree:
                continue
            values = tree.item(item, "values")
            if not values:
                return label, None
            try:
                return label, int(values[0])
            except (TypeError, ValueError):
                return label, None
        return None, None

    def _toggle_dataset(self, label: str):
        """一次只展开一个数据集: 点开一个, 其它的自动收起来。"""
        if label not in self.workspace.partition(self._partition).labels():
            return                      # 不在当前分区里, 忽略
        self._open_label = None if self._open_label == label else label
        self.refresh()

    @property
    def _open_label(self) -> str | None:
        return getattr(self, "_open", None)

    @_open_label.setter
    def _open_label(self, label):
        self._open = label if isinstance(label, str) else None

    @property
    def expanded(self) -> set[str]:
        """兼容旧接口: 展开集合最多只有一个元素。

        注意这是个**只读快照** —— `expanded.add(...)` 改的是临时集合, 没有任何
        效果。要展开请用 `expand(label)`, 收起用 `collapse()`。
        """
        current = self._open_label
        return {current} if current else set()

    @expanded.setter
    def expanded(self, value):
        if isinstance(value, str):
            self._open_label = value
        else:
            items = list(value) if value else []
            self._open_label = items[0] if items else None

    def expand(self, label: str) -> None:
        """展开某个数据集(一次只展开一个)。"""
        self._open_label = label

    def collapse(self) -> None:
        self._open_label = None

    def _resolve_selection(self) -> tuple[str | None, int | None]:
        """从当前 Treeview 选中项还原出 (数据集, 编号)。"""
        for label, tree in self._trees.items():
            selection = tree.selection()
            if not selection:
                continue
            values = tree.item(selection[0], "values")
            if values:
                try:
                    return label, int(values[0])
                except (TypeError, ValueError):
                    return label, None
        return None, None

    # ------------------------------ 绘制 ------------------------------ #
    def refresh(self):
        """按工作区当前内容重建整个滚动区。"""
        self.scroll.clear()
        self._trees: dict[str, ttk.Treeview] = {}
        partition = self.workspace.partition(self._partition)
        labels = partition.labels()

        if not labels:
            ttk.Label(
                self.scroll.body,
                text=f"{PARTITION_NAMES[self._partition]}中尚未创建数据集\n"
                     "请点击下方「新建数据集」",
                font=("Segoe UI", 9), foreground=COLOR_MUTED, justify="center",
            ).pack(pady=self.metrics.px(20))
            return

        for label in labels:
            self._build_dataset_block(label, partition.dataset(label))
        self._sync_tree_selection()
        self.scroll.scroll_to_top()

    def _build_dataset_block(self, label: str, dataset):
        px = self.metrics.px
        expanded = self._open_label == label
        block = ttk.Frame(self.scroll.body)
        block.pack(fill="x", pady=(0, px(4)))

        self._build_header(block, label, len(dataset), expanded)

        if not expanded:
            return

        holder = ttk.Frame(block)
        holder.pack(fill="x", padx=(px(12), 0))
        tree = ttk.Treeview(holder, columns=("number",), show="tree",
                            selectmode="browse", height=min(12, max(2, len(dataset))),
                            style="Sidebar.Treeview")
        tree.column("#0", width=px(THUMB_BOX) + px(10), stretch=False, anchor="center")
        tree.column("number", width=px(60), stretch=False, anchor="w")
        tree.pack(side="left", fill="x", expand=True)
        scroll = ttk.Scrollbar(holder, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        if len(dataset) > 12:
            scroll.pack(side="right", fill="y")

        for number in dataset.indices():
            ink = dataset.ink(number)
            height, width = ink.shape
            box = fit_size(width, height, self._thumb_px)
            photo = self.thumbs.get((self._partition, label, number), ink,
                                    max(box), self.metrics.scale)
            tree.insert("", "end", iid=f"{label}\x00{number}",
                        text="", image=photo, values=(number,))

        # 不用 <<TreeviewSelect>>: 选择态由 refresh() 重建控件时会再次触发它,
        # 那个"控件刚被销毁又来事件"的回环会把 Tk 搞崩(表现为未响应后闪退)。
        # 直接处理鼠标点击/回车, 逻辑清楚也不会自激。
        tree.bind("<ButtonRelease-1>", self._on_tree_click)
        tree.bind("<Double-1>", self._on_tree_activate)
        tree.bind("<Return>", self._on_tree_activate)
        self._trees[label] = tree

    def _build_header(self, parent, label: str, count: int, expanded: bool):
        """数据集卡头: 8px 圆角, 悬浮 #e0eef9, 按下 #cce4f7, 覆盖整张卡片。

        用 Canvas 自绘(ttk 的 Treeview/Label 给不了圆角, 选中高亮也是系统蓝、
        盖不满整行, 看着很碎)。鼠标位置用 `find_withtag("current")` 判断,
        比 <Enter>/<Leave> 更稳: 卡片本身就是那个 item。
        """
        px = self.metrics.px
        height = px(30)
        canvas = tk.Canvas(parent, height=height, highlightthickness=0,
                           bg=self._bg(parent), cursor="hand2")
        canvas.pack(fill="x")
        state = {"bg": HEADER_BG, "hover": False, "pressed": False}
        canvas._card_state = state          # 防止闭包被回收

        def draw():
            width = max(10, canvas.winfo_width())
            canvas.delete("all")
            radius = min(px(8), height // 2, width // 2)
            points = [
                radius, 1, width - radius, 1, width - 1, 1,
                width - 1, radius, width - 1, height - radius, width - 1, height - 1,
                width - radius, height - 1, radius, height - 1, 1, height - 1,
                1, height - radius, 1, radius, 1, 1,
            ]
            canvas.create_polygon(points, smooth=True, splinesteps=12,
                                  fill=state["bg"], outline=CARD_BORDER,
                                  width=1, tags="card")
            arrow = "▾" if expanded else "▸"
            canvas.create_text(px(10), height / 2, anchor="w",
                               text=f"{arrow}  {label}", font=("Segoe UI", 11, "bold"),
                               fill=label_color(label))
            canvas.create_text(width - px(10), height / 2, anchor="e",
                               text=str(count), font=("Segoe UI", 9),
                               fill=COLOR_MUTED)

        def on_motion(_event=None):
            over = canvas.find_withtag("current")
            inside = bool(over)
            if inside == state["hover"]:
                return
            state["hover"] = inside
            state["bg"] = HEADER_PRESS if (inside and state["pressed"]) else \
                (HEADER_HOVER if inside else HEADER_BG)
            draw()

        def on_press(_event=None):
            state["pressed"] = True
            if state["hover"]:
                state["bg"] = HEADER_PRESS
                draw()

        def on_release(event=None):
            state["pressed"] = False
            if state["hover"]:
                state["bg"] = HEADER_HOVER
                draw()
                self._toggle_dataset(label)

        canvas.bind("<Configure>", lambda _e: draw())
        canvas.bind("<Motion>", on_motion)
        canvas.bind("<Enter>", on_motion)
        canvas.bind("<Leave>", lambda _e: (state.update(hover=False, bg=HEADER_BG),
                                           draw()))
        canvas.bind("<Button-1>", on_press)
        canvas.bind("<ButtonRelease-1>", on_release)
        draw()

    def _bg(self, widget) -> str:
        try:
            return widget.cget("bg") or "#f0f0f0"
        except tk.TclError:
            return "#f0f0f0"

    def _sync_tree_selection(self):
        """把 `self._selected` 反映到 Treeview 上。

        纯展示: 用 `selection_set` 只改变高亮, 不绑定任何 select 事件, 所以
        不会反过来触发打开图片的逻辑。
        """
        if not getattr(self, "_trees", None):
            return
        for label, tree in self._trees.items():
            wanted = None
            if self._selected and self._selected[0] == label:
                wanted = f"{label}\x00{self._selected[1]}"
            if wanted and tree.exists(wanted):
                tree.selection_set(wanted)
                tree.focus(wanted)
                tree.see(wanted)
            else:
                current = tree.selection()
                if current:
                    tree.selection_remove(*current)

    def drop_thumbnails(self, label: str):
        for name in PARTITIONS:
            self.thumbs.drop_prefix(name, label)

    def drop_thumbnail(self, partition: str, label: str, number: int) -> None:
        """扔掉某一张图的缩略图(图被删掉时用)。"""
        self.thumbs.drop((partition, label, number))

    def update_thumbnail(self, partition: str, label: str, number: int, ink) -> None:
        """只换某一行的缩略图, **不重建整个列表** —— 画板上落笔时实时调用。

        重建(那怕只是 refresh())要销毁再建所有 Treeview, 边画边做会闪; 这里直接
        改那一行的图片。行不在当前分区、数据集没展开、或者图已经被删掉, 都安静返回。
        """
        if partition != self._partition:
            return
        tree = getattr(self, "_trees", {}).get(label)
        iid = f"{label}\x00{number}"
        if tree is None or not tree.exists(iid):
            return
        height, width = ink.shape
        size = max(fit_size(width, height, self._thumb_px))
        key = (partition, label, number)
        # 旧图先攥在手里: 换上新图之后再放, 免得 Tk 那边还在用的图片被提前删掉
        old = self.thumbs.get(key, ink, size, self.metrics.scale)
        photo = self.thumbs.refresh(key, ink, size, self.metrics.scale)
        tree.item(iid, image=photo)
        del old

    def rebuild_thumbnails(self, label: str):
        self.drop_thumbnails(label)
        self.refresh()
