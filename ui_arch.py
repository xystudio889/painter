"""「模型参数」卡片式对话框。

每张卡片代表网络里的一层连接项目:
    Conv  通道数 · 卷积核 · padding · 激活函数
    Pool  池化窗口
    FC    隐藏单元 · 激活函数 · Dropout

输出层不建卡片, 由类别数(= 数据集个数)自动补 `Linear(-> 类别数)`。
顶部可以选输入模式(center 裁剪居中 / raw 整幅缩放), 底部实时显示参数总数,
有错误时"确定"按钮禁用并把错误标在对应卡片下面。
"""

import tkinter as tk
from tkinter import ttk

import arch as arch_mod
from preprocess import INPUT_CENTER, INPUT_MODES
from ui_widgets import COLOR_FAIL, COLOR_MUTED, COLOR_PASS, Dialog, ScrollFrame

# 最上面两行 + 底部按钮的粗略高度, 用来算滚动区该多高
CHROME_HEIGHT = 210


class ArchDialog(Dialog):
    """编辑架构描述。`show()` 返回架构 dict 或 None。"""

    def __init__(self, parent, architecture: dict, num_classes: int,
                 metrics=None, canvas_height: int = 300):
        super().__init__(parent, "模型参数", resizable=True)
        self.architecture = arch_mod.normalize_arch(architecture, num_classes)
        self.num_classes = int(num_classes)
        self.metrics = metrics or self.metrics
        self.canvas_height = canvas_height
        self.card_rows: list[dict] = []
        self.card_errors: list[ttk.Label] = []
        self.result: dict | None = None
        px = self.metrics.px

        body = ttk.Frame(self, padding=px(10))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        # ------------------------------ 顶部 ------------------------------ #
        header = ttk.Frame(body)
        header.grid(row=0, column=0, sticky="ew", pady=(0, px(6)))
        ttk.Label(header, text="输入模式", font=("Segoe UI", 10)).pack(side="left")
        self.mode_var = tk.StringVar(value=self.architecture["input_mode"])
        mode_box = ttk.Combobox(header, textvariable=self.mode_var, state="readonly",
                                values=list(INPUT_MODES), width=10,
                                font=("Segoe UI", 9))
        mode_box.pack(side="left", padx=(px(4), px(12)))
        mode_box.bind("<<ComboboxSelected>>", lambda _e: self._on_mode_change())
        self.mode_hint = ttk.Label(header, text="", font=("Segoe UI", 9),
                                   foreground=COLOR_MUTED)
        self.mode_hint.pack(side="left")

        # ------------------------------ 卡片区 ------------------------------ #
        height = max(px(200), px(canvas_height) - px(CHROME_HEIGHT))
        self.scroll = ScrollFrame(body, width=px(560), height=height)
        self.scroll.grid(row=1, column=0, sticky="nsew")

        # ------------------------------ 底部 ------------------------------ #
        footer = ttk.Frame(body)
        footer.grid(row=2, column=0, sticky="ew", pady=(px(8), 0))
        footer.columnconfigure(0, weight=1)

        self.summary = ttk.Label(footer, text="", font=("Segoe UI", 9),
                                 foreground=COLOR_MUTED)
        self.summary.grid(row=0, column=0, sticky="w")
        self.error_summary = ttk.Label(footer, text="", font=("Segoe UI", 9),
                                       foreground=COLOR_FAIL)
        self.error_summary.grid(row=1, column=0, sticky="w")

        add_row = ttk.Frame(body)
        add_row.grid(row=3, column=0, sticky="w", pady=(px(6), 0))
        ttk.Label(add_row, text="添加卡片：", font=("Segoe UI", 9)).pack(side="left")
        for card_type, text in (("conv", "Conv"), ("pool", "Pool"), ("fc", "FC")):
            ttk.Button(add_row, text=text, width=8,
                       command=lambda t=card_type: self._add_card(t)).pack(
                side="left", padx=(px(4), 0))
        ttk.Separator(add_row, orient="vertical").pack(side="left", fill="y",
                                                       padx=px(10))
        ttk.Button(add_row, text="恢复默认", command=self._reset_default).pack(side="left")

        buttons = ttk.Frame(body)
        buttons.grid(row=4, column=0, sticky="e", pady=(px(10), 0))
        self.ok_button = ttk.Button(buttons, text="确定", command=self._on_ok, width=10)
        self.ok_button.pack(side="right")
        ttk.Button(buttons, text="取消", command=self.cancel, width=10).pack(
            side="right", padx=(0, px(6)))

        self._rebuild_cards()

    # ------------------------------ 输入模式 ------------------------------ #
    def _on_mode_change(self):
        self.architecture["input_mode"] = self.mode_var.get()
        self._validate()

    def mode_hint_text(self) -> str:
        if self.architecture["input_mode"] == "raw":
            return "整幅画布缩放至 28×28（不裁剪、不居中）"
        return "裁剪外接边框后按重心居中至 28×28（与 MNIST 一致）"

    # ------------------------------ 卡片 ------------------------------ #
    def _rebuild_cards(self):
        self.scroll.clear()
        self.card_rows = []
        self.card_errors = []
        for index, card in enumerate(self.architecture["cards"]):
            self._build_card(index, card)
        self._validate()
        self.center()

    def _build_card(self, index: int, card: dict):
        px = self.metrics.px
        row = ttk.Frame(self.scroll.body)
        row.pack(fill="x", pady=(0, px(3)))
        row.columnconfigure(2, weight=1)

        buttons = ttk.Frame(row)
        buttons.grid(row=0, column=0, sticky="nw")
        # 每张卡片的编辑入口都绑在下面这些控件上
        row_widgets: dict[str, tk.Variable] = {}

        def small(text, command, width=3):
            ttk.Button(buttons, text=text, width=width, command=command).pack(
                side="left", padx=(0, px(2)))

        small("↑", lambda i=index: self._move_card(i, -1))
        small("↓", lambda i=index: self._move_card(i, 1))
        small("✕", lambda i=index: self._delete_card(i))

        ttk.Label(row, text=f"{index + 1}.", width=3, anchor="e",
                  font=("Segoe UI", 9)).grid(row=0, column=1, sticky="nw",
                                             padx=(px(4), px(2)))

        controls = ttk.Frame(row)
        controls.grid(row=0, column=2, sticky="nw")

        def spin(parent, name: str, value, low: int, high: int, width=5,
                 values=None):
            ttk.Label(parent, text=name, font=("Segoe UI", 9)).pack(side="left")
            variable = tk.StringVar(value=str(value))
            row_widgets[name] = variable
            if values:
                widget = ttk.Combobox(parent, textvariable=variable, width=width,
                                      state="readonly", font=("Segoe UI", 9),
                                      values=[str(v) for v in values])
            else:
                widget = ttk.Spinbox(parent, textvariable=variable, width=width,
                                     from_=low, to=high, font=("Segoe UI", 9))
            widget.pack(side="left", padx=(px(2), px(10)))
            widget.bind("<<ComboboxSelected>>", lambda _e: self._on_param_change())
            widget.bind("<FocusOut>", lambda _e: self._on_param_change())
            variable.trace_add("write", lambda *_a: self._on_param_change())
            return widget

        card_type = card.get("type")
        ttk.Label(controls, text={"conv": "Conv", "pool": "Pool",
                                  "fc": "FC"}.get(card_type, str(card_type)),
                  font=("Consolas", 10, "bold"), foreground="#0a6ebd").pack(
            side="left", padx=(0, px(10)))
        if card_type == "conv":
            spin(controls, "通道数", card.get("channels", 32),
                 *arch_mod.CHANNEL_RANGE, width=5)
            spin(controls, "卷积核", card.get("kernel", 3), 0, 0, width=3,
                 values=arch_mod.KERNEL_CHOICES)
            spin(controls, "填充", card.get("padding", 1), 0, 2, width=3,
                 values=arch_mod.PADDING_CHOICES)
            spin(controls, "激活", card.get("activation", "relu"), 0, 0, width=6,
                 values=arch_mod.ACTIVATIONS)
        elif card_type == "pool":
            spin(controls, "池化窗口", card.get("window", 2), 2, 4, width=3,
                 values=arch_mod.POOL_CHOICES)
        elif card_type == "fc":
            spin(controls, "隐藏单元", card.get("units", 128), *arch_mod.UNITS_RANGE,
                 width=6)
            spin(controls, "激活", card.get("activation", "relu"), 0, 0, width=6,
                 values=arch_mod.ACTIVATIONS)
            spin(controls, "Dropout", card.get("dropout", 0.0), 0, 1, width=5)

        error = ttk.Label(row, text="", font=("Segoe UI", 9), foreground=COLOR_FAIL)
        self.card_rows.append({"frame": row, "vars": row_widgets, "error": error,
                               "index": index})

    # ------------------------------ 结构动作 ------------------------------ #
    def _add_card(self, card_type: str):
        self.architecture["cards"].append(arch_mod.new_card(card_type))
        self._rebuild_cards()

    def _delete_card(self, index: int):
        if 0 <= index < len(self.architecture["cards"]):
            self.architecture["cards"].pop(index)
            self._rebuild_cards()

    def _move_card(self, index: int, delta: int):
        cards = self.architecture["cards"]
        target = index + delta
        if 0 <= index < len(cards) and 0 <= target < len(cards):
            cards[index], cards[target] = cards[target], cards[index]
            self._rebuild_cards()

    def _reset_default(self):
        """一键回到与 `DigitCNN` 等价的默认架构。"""
        self.architecture = arch_mod.default_arch(self.num_classes)
        self.mode_var.set(self.architecture["input_mode"])
        self._rebuild_cards()

    # ------------------------------ 校验 ------------------------------ #
    def _on_param_change(self):
        for index, row in enumerate(self.card_rows):
            if index >= len(self.architecture["cards"]):
                break
            card = self.architecture["cards"][index]
            for name, variable in row["vars"].items():
                self._store(card, name, variable.get(), card.get("type"))
        self._validate()

    @staticmethod
    def _store(card: dict, name: str, raw: str, card_type: str):
        """把界面上的字符串写回卡片字典(类型转换交给校验去报错)。

        这里的 `name` 就是界面上那个字段标签(见 `_build_card` 里的 `spin(...)`),
        两边必须一字不差 —— 改名时两处一起改。
        """
        raw = raw.strip()
        if card_type == "conv":
            if name == "通道数":
                card["channels"] = ArchDialog._maybe_int(raw)
            elif name == "卷积核":
                card["kernel"] = ArchDialog._maybe_int(raw)
            elif name == "填充":
                card["padding"] = ArchDialog._maybe_int(raw)
            elif name == "激活":
                card["activation"] = raw
        elif card_type == "pool":
            if name == "池化窗口":
                card["window"] = ArchDialog._maybe_int(raw)
        elif card_type == "fc":
            if name == "隐藏单元":
                card["units"] = ArchDialog._maybe_int(raw)
            elif name == "激活":
                card["activation"] = raw
            elif name == "Dropout":
                card["dropout"] = ArchDialog._maybe_float(raw)

    @staticmethod
    def _maybe_int(raw: str):
        try:
            return int(raw)
        except ValueError:
            return raw            # 交给 validate 报"必须是整数"

    @staticmethod
    def _maybe_float(raw: str):
        try:
            return float(raw)
        except ValueError:
            return raw

    def _validate(self):
        errors = arch_mod.validate_arch(self.architecture)
        by_card: dict[int, list[str]] = {}
        global_errors: list[str] = []
        for error in errors:
            if error.index is None:
                global_errors.append(error.message)
            else:
                by_card.setdefault(error.index, []).append(error.message)

        for row in self.card_rows:
            index = row["index"]
            messages = by_card.get(index)
            label = row["error"]
            if messages:
                label.configure(text="   ".join(f"⚠ {m}" for m in messages))
                label.grid(row=1, column=2, sticky="w", padx=self.metrics.px(6))
            else:
                label.grid_remove()

        if global_errors:
            self.error_summary.configure(text="⚠ " + "；".join(global_errors))
        else:
            self.error_summary.configure(text="")

        try:
            self.summary.configure(text=arch_mod.describe_arch(self.architecture),
                                   foreground=COLOR_MUTED)
        except arch_mod.ArchError as exc:
            self.summary.configure(text=f"架构不合法：{exc}", foreground=COLOR_FAIL)

        self.mode_hint.configure(text=self.mode_hint_text())
        if errors:
            self.ok_button.state(["disabled"])
        else:
            self.ok_button.state(["!disabled"])

    def _on_ok(self):
        self._on_param_change()
        if arch_mod.validate_arch(self.architecture):
            return
        self.architecture["num_classes"] = self.num_classes
        self.architecture["source"] = "custom"
        self.result = self.architecture
        self.destroy()
