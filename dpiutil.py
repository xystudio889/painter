"""高 DPI 支持与尺寸换算。

Windows 上如果不开进程 DPI 感知, 系统会把 96 DPI 渲染出的位图整体拉伸,
文字和线条都会发虚。这里集中处理"逻辑像素 -> 物理像素"以及跨显示器重排。

所有尺寸常量都是 96 DPI 下的**逻辑像素**, 由 `Metrics.px()` 换算成物理像素。
"""

import sys
import tkinter as tk

BASE_DPI = 96.0


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


class Metrics:
    """一个 DPI 缩放上下文, 供主窗口与各对话框共用。

    用法:
        m = Metrics(root)
        m.px(10)          # 逻辑像素 -> 物理像素
        m.scale           # 当前缩放比(1.0 = 96 DPI)
        m.apply(root)     # 设置 tk scaling, 让点值字号一起放大
    """

    def __init__(self, window: tk.Misc):
        self.set(get_window_dpi(window))

    def set(self, dpi: float) -> None:
        self.dpi = float(dpi)
        self.scale = max(self.dpi, BASE_DPI) / BASE_DPI

    def px(self, value: float) -> int:
        """逻辑像素 -> 物理像素。"""
        return max(1, int(round(value * self.scale)))

    def px0(self, value: float) -> int:
        """和 px 一样, 但允许返回 0(例如 padding 为 0)。"""
        return max(0, int(round(value * self.scale)))

    def apply(self, window: tk.Misc) -> None:
        """把 DPI 应用到 Tk(字号基准 + 本窗口)。"""
        try:
            window.tk.call("tk", "scaling", self.dpi / 72.0)
        except tk.TclError:
            pass

    def refresh(self, window: tk.Misc) -> bool:
        """重新探测窗口 DPI。变化了返回 True。"""
        dpi = get_window_dpi(window)
        if abs(dpi - self.dpi) < 1.0:
            return False
        self.set(dpi)
        self.apply(window)
        return True


def center_on_parent(window: tk.Toplevel, parent: tk.Misc) -> None:
    """把对话框居中到父窗口上。两者都还没映射时忽略。"""
    try:
        window.update_idletasks()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        w, h = window.winfo_reqwidth(), window.winfo_reqheight()
        if pw <= 1 or ph <= 1:
            return
        x = px + (pw - w) // 2
        y = py + (ph - h) // 3
        # 别把对话框推到屏幕外面
        sw, sh = window.winfo_screenwidth(), window.winfo_screenheight()
        x = max(0, min(x, sw - w))
        y = max(0, min(y, sh - h))
        window.geometry(f"+{x}+{y}")
    except tk.TclError:
        pass
