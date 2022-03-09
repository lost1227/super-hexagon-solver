import sys

if sys.platform == "win32":
    import win32gui

    def find_window_coords():
        coords = None
        def callback(hwnd, extra):
            nonlocal coords
            text = win32gui.GetWindowText(hwnd)
            if text != "Super Hexagon":
                return
            rect = win32gui.GetWindowRect(hwnd)
            x = rect[0] + 10
            y = rect[1] + 40
            w = rect[2] - 10 - x
            h = rect[3] - 50 - y
            coords = {"top":y, "left":x, "width":w, "height":h}
        win32gui.EnumWindows(callback, None)

        return coords
elif sys.platform == "linux":
    from gi.repository import Gtk, Wnck
    Gtk.init([])

    def find_window_coords():
        screen = Wnck.Screen.get_default()
        screen.force_update()  # recommended per Wnck documentation
        window_list = screen.get_windows()

        hex = None
        for window in window_list:
            if window.get_name() == "Super Hexagon":
                hex = window

        if hex is None:
            return None
        
        geometry = hex.get_geometry()
        return {"top":geometry.yp + 36, "left":geometry.xp, "width":geometry.widthp, "height":geometry.heightp - 36}
else:
    raise Exception("Unsupported platform")
