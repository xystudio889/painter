"""「测试」结果小窗口。

顶部是"通过 X / 不通过 Y"的汇总, 下面是可滚动的明细列表, 每行形如:

    a/图片0 -> a     ✓ 通过
    b/图片1 -> c     ✗ 不通过(期望 b)

点某一行会在主窗口里打开那张图, 方便定位错例。
"""

import tkinter as tk
from tkinter import ttk

from ui_widgets import (
    COLOR_FAIL, COLOR_MUTED, COLOR_PASS, Dialog, ScrollFrame, fit_size,
    label_color,
)

ROW_THUMB = 34   # 明细行里的小图边长(逻辑像素)


class TestResultWindow(Dialog):
    """展示一次测试集检验的结果。

    entries: [{"partition", "label", "number", "predicted",
               "confidence", "passed", "ink"}]
    """

    def __init__(self, parent, entries: list[dict], metrics=None,
                 title: str = "测试结果", model_name: str = "",
                 on_open=None):
        super().__init__(parent, title, resizable=True, modal=False)
        self.entries = entries
        self.metrics = metrics or self.metrics
        self.on_open = on_open
        self.photos: list[tk.PhotoImage] = []      # 防止被 GC
        px = self.metrics.px

        passed = sum(1 for item in entries if item["passed"])
        failed = len(entries) - passed
        rate = (passed / len(entries) * 100) if entries else 0.0

        body = ttk.Frame(self, padding=px(12))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        # ------------------------------ 汇总 ------------------------------ #
        head = ttk.Frame(body)
        head.grid(row=0, column=0, sticky="ew", pady=(0, px(8)))

        summary = ttk.Frame(head)
        summary.pack(side="left")
        self.pass_label = ttk.Label(
            summary, text=f"通过 {passed}", font=("Segoe UI", 18, "bold"),
            foreground=COLOR_PASS)
        self.pass_label.pack(side="left")
        ttk.Label(summary, text=" / ", font=("Segoe UI", 18),
                  foreground=COLOR_MUTED).pack(side="left")
        self.fail_label = ttk.Label(
            summary, text=f"不通过 {failed}", font=("Segoe UI", 18, "bold"),
            foreground=COLOR_FAIL if failed else COLOR_MUTED)
        self.fail_label.pack(side="left")

        detail = f"共 {len(entries)} 张，通过率 {rate:.1f}%"
        if model_name:
            detail = f"模型 {model_name}    " + detail
        ttk.Label(head, text=detail, font=("Segoe UI", 9),
                  foreground=COLOR_MUTED).pack(side="left", padx=(px(14), 0))

        # ------------------------------ 明细 ------------------------------ #
        box = ttk.LabelFrame(body, text="明细（双击可在主窗口中打开该图片）",
                             padding=px(6))
        box.grid(row=1, column=0, sticky="nsew")
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)

        self.scroll = ScrollFrame(box, width=px(560), height=px(320))
        self.scroll.grid(row=0, column=0, sticky="nsew")

        if not entries:
            ttk.Label(self.scroll.body, text="测试集中暂无图片",
                      font=("Segoe UI", 10), foreground=COLOR_MUTED).pack(
                pady=px(20))
        else:
            # 不通过的排在前面, 方便先看错例
            ordered = sorted(self.entries, key=lambda item: item["passed"])
            for item in ordered:
                self._build_row(item)

        # ------------------------------ 关闭 ------------------------------ #
        footer = ttk.Frame(body)
        footer.grid(row=2, column=0, sticky="ew", pady=(px(8), 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer,
                  text="提示：将容易误识别的样本补充到训练集后重新训练，通常可以改善通过率",
                  font=("Segoe UI", 9), foreground=COLOR_MUTED).grid(
            row=0, column=0, sticky="w")
        ttk.Button(footer, text="关闭", width=10, command=self._close).grid(
            row=0, column=1)

        self.protocol("WM_DELETE_WINDOW", self._close)
        self.center()

    def _close(self):
        self.result = None
        self.destroy()

    def _build_row(self, item: dict):
        px = self.metrics.px
        row = ttk.Frame(self.scroll.body)
        row.pack(fill="x", pady=(0, px(2)))

        color = COLOR_PASS if item["passed"] else COLOR_FAIL
        mark = "✓" if item["passed"] else "✗"
        text = f"{item['label']}／图片{item['number']}  →  {item['predicted']}"
        if not item["passed"]:
            text += f"   （期望 {item['label']}）"
        confidence = item.get("confidence")
        if confidence is not None:
            text += f"   {confidence * 100:.1f}%"

        label = tk.Label(row, text=f" {mark} {text}", anchor="w", justify="left",
                         font=("Segoe UI", 10), fg=color, bg=self._row_bg(row),
                         padx=px(6), pady=px(4), cursor="hand2")
        label.pack(side="left", fill="x", expand=True)

        ink = item.get("ink")
        if ink is not None:
            from PIL import ImageTk
            from preprocess import ink_to_pil
            height, width = ink.shape
            thumb_px = px(ROW_THUMB)
            box = fit_size(width, height, thumb_px)
            photo = ImageTk.PhotoImage(
                ink_to_pil(ink).resize(
                    (max(1, int(box[0] * self.metrics.scale)),
                     max(1, int(box[1] * self.metrics.scale)))),
                master=self)
            self.photos.append(photo)
            thumb = tk.Label(row, image=photo, bg=self._row_bg(row))
            thumb.pack(side="right", padx=px(4))
            thumb.bind("<Double-1>", lambda _e, it=item: self._open(it))

        badge = tk.Label(row, text=f" {item['label']} ",
                         font=("Consolas", 10, "bold"), fg="white",
                         bg=label_color(item["label"]), padx=px(4))
        badge.pack(side="right", padx=px(4))

        for widget in (label, badge):
            widget.bind("<Double-1>", lambda _e, it=item: self._open(it))

    def _open(self, item: dict):
        """在主窗口打开这张图。结果窗口是置顶但不抢焦点的, 主窗口仍可操作。"""
        if self.on_open is None:
            return
        owner = None
        try:
            owner = self.grab_current()
        except tk.TclError:
            owner = None
        if owner is not None and owner is not self:
            # 主窗口此刻还握着模态 grab, 先松开, 否则用户点不动主窗口
            try:
                owner.grab_release()
                self.after(50, lambda: owner.grab_set() if owner.winfo_exists()
                           else None)
            except tk.TclError:
                pass
        self.on_open(item)

    def _row_bg(self, widget) -> str:
        try:
            return widget.cget("bg") or "#f0f0f0"
        except tk.TclError:
            return "#f0f0f0"


def show_test_results(parent, entries, metrics=None, model_name: str = "",
                      on_open=None) -> TestResultWindow:
    window = TestResultWindow(parent, entries, metrics=metrics,
                              model_name=model_name, on_open=on_open)
    window.show()
    return window
