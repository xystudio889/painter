"""自绘的下拉控件(模型选择 / 参考模型 / 数据集选择都用它)。

为什么不用 `ttk.Combobox`: 它的下拉列表由 Windows 主题绘制 —— 纯蓝选中、
盖不满整行、圆角/间距/配色都改不动, 也塞不进缩略图。
这里用「按钮 + 弹出 Toplevel」自己画:

    · 淡蓝标题栏
    · 卡片 8px 圆角, 卡片间 4px 间隔, 覆盖整张卡片
    · 悬浮 #e0eef9, 按下 #cce4f7, 选中态高亮 + ✓
    · 列表项可以带缩略图(数据集选择器用)
"""

import tkinter as tk
from tkinter import ttk

# 配色
TITLE_BG = "#dceaf7"        # 标题栏淡蓝
TITLE_FG = "#1a5276"
PANEL_BG = "#ffffff"
BORDER = "#b9d4ea"
HOVER_BG = "#e0eef9"
PRESS_BG = "#cce4f7"
SELECT_BG = "#d6e9f9"
NORMAL_BG = "#ffffff"
TEXT_FG = "#1a1a1a"

RADIUS = 8
GAP = 4
PAD_X = 10
MIN_WIDTH = 250
MAX_HEIGHT = 320
ROW_HEIGHT = 28
THUMB_BOX = 34


def round_rect(canvas: tk.Canvas, x1, y1, x2, y2, radius: int, **kwargs):
    """Canvas 上的圆角矩形(Tk 原生没有圆角)。"""
    radius = max(0, min(int(radius), int((x2 - x1) / 2), int((y2 - y1) / 2)))
    points = [
        x1 + radius, y1, x2 - radius, y1, x2, y1,
        x2, y1 + radius, x2, y2 - radius, x2, y2,
        x2 - radius, y2, x1 + radius, y2, x1, y2,
        x1, y2 - radius, x1, y1 + radius, x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, splinesteps=12, **kwargs)


class DropdownButton(ttk.Frame):
    """自绘的「下拉按钮 + 弹出卡片列表」。

    items_provider(): 返回列表, 每项 {"text": str, "thumb": PIL.Image|None}
    on_choose(index): 用户选中某项
    side_button / side_command: 右侧那个按钮(例如「导入」)
    """

    def __init__(self, parent, items_provider, on_choose, metrics=None,
                 side_button: str | None = None, side_command=None,
                 empty_text: str = "(无)", title: str = "", **kwargs):
        super().__init__(parent, **kwargs)
        self.items_provider = items_provider
        self.on_choose = on_choose
        self.metrics = metrics
        self.empty_text = empty_text
        self.list_title = title
        self.items: list[dict] = []
        self.index = 0
        self._forced_text: str | None = None
        self._popup: tk.Toplevel | None = None
        self._cards: list[dict] = []
        self._thumbs: list[tk.PhotoImage] = []
        px = self.metrics.px if metrics else (lambda v: int(v))

        self.columnconfigure(0, weight=1)
        outer = tk.Frame(self, bg=BORDER)
        outer.grid(row=0, column=0, sticky="ew")
        self.button = tk.Canvas(outer, height=px(28), highlightthickness=0,
                                bg=NORMAL_BG, cursor="hand2")
        self.button.pack(fill="x", padx=1, pady=1)
        self.button.bind("<Configure>", lambda _e: self._draw_button())
        self.button.bind("<Button-1>", lambda _e: self.toggle())
        self.button.bind("<Enter>", lambda _e: self._button_bg(HOVER_BG))
        self.button.bind("<Leave>", lambda _e: self._button_bg(NORMAL_BG))

        self.side_button = None
        if side_button is not None:
            self.side_button = ttk.Button(self, text=side_button, width=6,
                                          command=side_command)
            self.side_button.grid(row=0, column=1, sticky="e", padx=(px(4), 0))

    # ------------------------------ 数据 ------------------------------ #
    def set_items(self, items: list[dict], index: int = 0):
        self.items = list(items)
        self.index = max(0, min(index, len(items) - 1)) if items else 0
        self._draw_button()
        if self._popup is not None:
            self._build_cards()

    @property
    def current(self) -> dict | None:
        if not self.items:
            return None
        return self.items[self.index]

    def set_text(self, text: str | None):
        """不走列表, 直接指定按钮上显示的文字。传 None 恢复自动。"""
        self._forced_text = text
        self._draw_button()

    def _text(self) -> str:
        if self._forced_text:
            return self._forced_text
        item = self.current
        return item.get("text", "") if item else self.empty_text

    def _button_bg(self, color: str):
        if self._popup is None:
            self.button.configure(bg=color)

    def _draw_button(self):
        px = self.metrics.px if self.metrics else (lambda v: int(v))
        width = max(60, self.button.winfo_width())
        height = max(22, self.button.winfo_height())
        self.button.delete("all")
        self.button.create_text(px(8), height / 2, anchor="w", text=self._text(),
                                font=("Segoe UI", 10), fill=TEXT_FG,
                                width=max(20, width - px(26)))
        self.button.create_text(width - px(12), height / 2, anchor="center",
                                text="▾", font=("Segoe UI", 9), fill="#555")

    # ------------------------------ 弹出 ------------------------------ #
    def toggle(self):
        if self._popup is None:
            self.open()
        else:
            self.close()

    def open(self):
        if self._popup is not None:
            return
        try:
            items = self.items_provider()
        except Exception:                     # noqa: BLE001 - 弹不出来别拖垮主界面
            return
        self.items = list(items or [])
        if not self.items:
            return                            # 没东西可选就不弹
        if not 0 <= self.index < len(self.items):
            self.index = 0
        px = self.metrics.px if self.metrics else (lambda v: int(v))
        self.button.configure(bg=PRESS_BG)
        self.update_idletasks()

        popup = tk.Toplevel(self)
        popup.overrideredirect(True)
        popup.transient(self.winfo_toplevel())
        popup.configure(bg=BORDER)
        self._popup = popup

        width = max(px(MIN_WIDTH), self.button.winfo_width() + px(40))
        title_h = px(24) if self.list_title else 0
        row_h = px(ROW_HEIGHT) + (px(16) if self._has_thumbs() else 0)
        content_h = row_h * len(self.items) + px(8)
        height = min(px(MAX_HEIGHT) + title_h, content_h + title_h + px(4))
        x = self.button.winfo_rootx()
        y = self.button.winfo_rooty() + self.button.winfo_height() + px(2)
        screen_h = popup.winfo_screenheight()
        screen_w = popup.winfo_screenwidth()
        if x + width > screen_w:
            x = max(0, screen_w - width - px(4))
        if y + height > screen_h - px(10):
            y = max(px(4), self.button.winfo_rooty() - height - px(2))
        popup.geometry(f"{width}x{height}+{x}+{y}")

        holder = tk.Frame(popup, bg=PANEL_BG)
        holder.pack(fill="both", expand=True, padx=1, pady=1)
        holder.rowconfigure(1, weight=1)
        holder.columnconfigure(0, weight=1)

        if self.list_title:
            tk.Label(holder, text=self.list_title, anchor="w", bg=TITLE_BG,
                     fg=TITLE_FG, font=("Segoe UI", 9, "bold"),
                     padx=px(8), pady=px(2)).grid(
                row=0, column=0, columnspan=2, sticky="ew")

        self.canvas = tk.Canvas(holder, highlightthickness=0, bg=PANEL_BG)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        if content_h > height - title_h:
            self.scrollbar = ttk.Scrollbar(holder, orient="vertical",
                                           command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=self.scrollbar.set)
            self.scrollbar.grid(row=1, column=1, sticky="ns")
        self.body = tk.Frame(self.canvas, bg=PANEL_BG)
        self._window = self.canvas.create_window((0, 0), window=self.body,
                                                 anchor="nw")
        self.body.bind("<Configure>", lambda _e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(
            self._window, width=e.width))

        popup.bind("<Escape>", lambda _e: self.close())
        self._build_cards()
        try:
            # 抓取输入: 点弹窗外面会先落到本窗口, 于是自动收起
            popup.grab_set()
            popup.focus_set()
        except tk.TclError:
            pass

    def _has_thumbs(self) -> bool:
        return any(item.get("thumb") is not None for item in self.items)

    def close(self):
        if self._popup is None:
            return
        popup, self._popup = self._popup, None
        self._cards = []
        self._thumbs = []
        try:
            popup.grab_release()
        except tk.TclError:
            pass
        popup.destroy()
        self._button_bg(NORMAL_BG)
        self._draw_button()

    # ------------------------------ 卡片 ------------------------------ #
    def _build_cards(self):
        if self._popup is None:
            return
        px = self.metrics.px if self.metrics else (lambda v: int(v))
        for child in self.body.winfo_children():
            child.destroy()
        self._cards = []
        self._thumbs = []
        for index, item in enumerate(self.items):
            self._build_card(index, item, px)
        self.canvas.yview_moveto(0)

    def _build_card(self, index: int, item: dict, px):
        from PIL import Image, ImageTk

        selected = index == self.index
        bg = SELECT_BG if selected else NORMAL_BG
        holder = tk.Frame(self.body, bg=PANEL_BG)
        holder.pack(fill="x", pady=(px(GAP) // 2, px(GAP) // 2))
        canvas = tk.Canvas(holder, height=px(ROW_HEIGHT), highlightthickness=0,
                           bg=PANEL_BG, cursor="hand2")
        canvas.pack(fill="x")

        card = {"index": index, "canvas": canvas, "bg": bg, "selected": selected,
                "px": px, "item": item}

        thumb = item.get("thumb")
        if thumb is not None:
            size = px(THUMB_BOX)
            image = thumb if thumb.size == (size, size) else thumb.resize(
                (size, size), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image, master=self)
            self._thumbs.append(photo)
            card["photo"] = photo

        text_x = px(PAD_X) + (px(THUMB_BOX) + px(14) if "photo" in card else 0)
        mark = "✓ " if selected else "   "
        card["text_item"] = canvas.create_text(
            text_x, px(ROW_HEIGHT) / 2, anchor="w",
            text=mark + str(item.get("text", "")), font=("Segoe UI", 10),
            fill=TEXT_FG)
        self._cards.append(card)

        canvas.bind("<Configure>", lambda _e, i=index: self._draw_card(i))
        canvas.bind("<Enter>", lambda _e, i=index: self._hover(i, True))
        canvas.bind("<Leave>", lambda _e, i=index: self._hover(i, False))
        canvas.bind("<Button-1>", lambda _e, i=index: self._press(i))
        canvas.bind("<ButtonRelease-1>", lambda _e, i=index: self._release(i))
        self._draw_card(index)

    def _card(self, index: int) -> dict | None:
        for card in self._cards:
            if card["index"] == index:
                return card
        return None

    def _draw_card(self, index: int):
        card = self._card(index)
        if card is None:
            return
        px = card["px"]
        canvas = card["canvas"]
        width = max(10, canvas.winfo_width())
        height = max(10, canvas.winfo_height())
        canvas.delete("rect")
        round_rect(canvas, 2, 1, width - 2, height - 1, px(RADIUS),
                   fill=card["bg"], outline=BORDER, width=1, tags="rect")
        canvas.tag_lower("rect")
        if "photo" in card:
            canvas.delete("thumb")
            canvas.create_image(px(PAD_X) + px(THUMB_BOX) / 2, height / 2,
                                image=card["photo"], tags="thumb")
            canvas.tag_raise("thumb")
        canvas.tag_raise(card["text_item"])
        canvas.coords(card["text_item"], px(PAD_X) + (
            px(THUMB_BOX) + px(14) if "photo" in card else 0), height / 2)

    def _hover(self, index: int, entering: bool):
        card = self._card(index)
        if card is None or card["selected"]:
            return
        card["bg"] = HOVER_BG if entering else NORMAL_BG
        self._draw_card(index)

    def _press(self, index: int):
        card = self._card(index)
        if card is None:
            return
        card["bg"] = PRESS_BG
        self._draw_card(index)

    def _release(self, index: int):
        """单选: 选中这张, 其它一律取消。"""
        if index < 0 or index >= len(self.items):
            return
        self.index = index
        for card in self._cards:
            card["selected"] = card["index"] == index
            card["bg"] = SELECT_BG if card["selected"] else NORMAL_BG
            self._draw_card(card["index"])
        self._draw_button()
        self.close()
        self.on_choose(index)


def labelled_dropdown(parent, label: str, metrics, **kwargs):
    """带标题的一行下拉: [标题] 换行 [下拉 + 侧边按钮]。返回 (容器, 控件)。"""
    px = metrics.px
    holder = ttk.Frame(parent)
    holder.columnconfigure(0, weight=1)
    ttk.Label(holder, text=label, font=("Segoe UI", 10)).grid(
        row=0, column=0, sticky="w")
    widget = DropdownButton(holder, metrics=metrics, **kwargs)
    widget.grid(row=1, column=0, sticky="ew", pady=(px(2), 0))
    return holder, widget
