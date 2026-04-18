#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import tkinter as tk


_STAGES_RU = [
    "Подготовка файлов",
    "Загрузка модели",
    "Инициализация",
    "Прогрев нейросети",
    "Готово",
]
_STAGES_EN = [
    "Preparing files",
    "Loading model",
    "Initializing",
    "Warming up",
    "Ready",
]

_HINTS_RU = [
    "Это нужно только при первом запуске",
    "После загрузки приложение откроется быстрее",
    "Данные обрабатываются локально на вашем устройстве",
    "Осталось совсем немного",
]
_HINTS_EN = [
    "This is only needed on the first run",
    "Next launch will be much faster",
    "All data is processed locally on your device",
    "Almost there",
]

_MAIN_TITLE = {
    "ru": "Подготавливаем модель\nраспознавания поз",
    "en": "Loading pose recognition\nmodel",
}
_DEFAULT_SUB = {
    "ru": "Подготавливаем файлы модели",
    "en": "Preparing model files",
}
_FIRST_LAUNCH = {
    "ru": "При первом запуске загрузка занимает до 30 секунд",
    "en": "First launch may take up to 30 seconds",
}


class ModelLoadingOverlay:
    def __init__(self, root: tk.Tk, colors: dict,
                 lang: str = "ru") -> None:
        self.root: tk.Tk = root
        self.colors: dict = colors
        self.lang: str = lang
        self._frame: tk.Frame | None = None
        self._spinner_cv: tk.Canvas | None = None
        self._prog_cv: tk.Canvas | None = None
        self._ring_cv: tk.Canvas | None = None
        self._title_var: tk.StringVar | None = None
        self._sub_var: tk.StringVar | None = None
        self._model_var: tk.StringVar | None = None
        self._stage_var: tk.StringVar | None = None
        self._hint_var: tk.StringVar | None = None
        self._prog_var: tk.DoubleVar | None = None
        self._pct_lbl: tk.Label | None = None
        self._hint_lbl: tk.Label | None = None
        self._first_lbl: tk.Label | None = None
        self._anim_id: int | None = None
        self._dot_id: int | None = None
        self._hint_id: int | None = None
        self._angle: float = 0.0
        self._angle2: float = 0.0
        self._dots: int = 0
        self._pulse: float = 0.0
        self._hint_idx: int = 0
        self._hint_fade: float = 1.0
        self._hint_fading: bool = False
        self.current_value: float = 0.0
        self._model_name: str = ""
        self._success_pulse: float = 0.0
        self._success_mode: bool = False

    # ── Public ────────────────────────────────────────────────────────────

    def show(self, title: str = "",
             subtitle: str = "",
             model_name: str = "") -> None:
        if self._frame and self._frame.winfo_exists():
            self.hide()
        self.current_value = 0.0
        self._success_mode = False
        self._model_name = model_name or self._extract_model(title)
        self._build(title, subtitle)

    def hide(self, delay_ms: int = 0) -> None:
        if delay_ms > 0:
            self.root.after(delay_ms, self._do_hide)
        else:
            self._do_hide()

    def _do_hide(self) -> None:
        self._stop_anim()
        if self._frame and self._frame.winfo_exists():
            self._frame.destroy()
        self._frame = None
        self._spinner_cv = None
        self._prog_cv = None
        self._ring_cv = None
        self._title_var = None
        self._sub_var = None
        self._model_var = None
        self._stage_var = None
        self._hint_var = None
        self._prog_var = None
        self._pct_lbl = None
        self._first_lbl = None
        self.current_value = 0.0

    def update(self, pct: float = None, subtitle: str = "") -> None:
        if pct is not None:
            self.set_progress(float(pct), subtitle)
        elif subtitle:
            self.set_subtitle(subtitle)

    def set_progress(self, pct: float, subtitle: str = "") -> None:
        self.current_value = max(0.0, min(100.0, float(pct)))
        if self._prog_var:
            self._prog_var.set(self.current_value)
        if subtitle and self._sub_var:
            self._sub_var.set(self._humanize(subtitle))
        self._update_stage()
        if self._frame and self._frame.winfo_exists():
            try:
                self._draw_progress()
                self._draw_ring()
            except Exception:
                pass
        if self.current_value >= 100.0 and not self._success_mode:
            self._success_mode = True
            self._success_pulse = 0.0

    def set_subtitle(self, text: str) -> None:
        if self._sub_var:
            self._sub_var.set(self._humanize(text))

    def set_title(self, text: str) -> None:
        if self._title_var:
            self._title_var.set(text)

    # ── Смена языка (публичный метод) ─────────────────────────────────────

    def set_lang(self, lang: str) -> None:
        """Переключить язык на лету, если оверлей открыт."""
        self.lang = lang
        if not (self._frame and self._frame.winfo_exists()):
            return

        # Заголовок
        if self._title_var:
            self._title_var.set(_MAIN_TITLE.get(lang, _MAIN_TITLE["en"]))

        # Этап
        self._update_stage()

        # Подсказка «первый запуск»
        if self._first_lbl:
            try:
                self._first_lbl.config(
                    text=_FIRST_LAUNCH.get(lang, _FIRST_LAUNCH["en"]))
            except Exception:
                pass

        # Хинт — сразу подставить текущий
        if self._hint_var:
            hints = _HINTS_RU if lang == "ru" else _HINTS_EN
            idx = self._hint_idx % len(hints)
            self._hint_var.set(hints[idx])

        # Subtitle — перегуманизировать текущий текст
        if self._sub_var:
            cur = self._sub_var.get().rstrip(".")
            self._sub_var.set(self._humanize(cur))

    # ── Humanize ──────────────────────────────────────────────────────────

    def _humanize(self, raw: str) -> str:
        r = raw.lower()
        if self.lang == "ru":
            if any(s in r for s in ["подготов", "локальн", "источник",
                                     "preparing", "local"]):
                return "Подготавливаем файлы модели"
            if any(s in r for s in ["загруж", "loading", "скачив"]):
                return "Загружаем модель распознавания поз"
            if any(s in r for s in ["инициал", "initial"]):
                return "Инициализируем нейросеть"
            if any(s in r for s in ["прогрев", "warmup", "warm"]):
                return "Прогреваем модель — почти готово"
            if any(s in r for s in ["готов", "ready", "done"]):
                return "Модель готова к работе"
            if any(s in r for s in ["ошибка", "error"]):
                return f"Ошибка: {raw}"
            return raw
        else:
            if any(s in r for s in ["подготов", "локальн", "local",
                                     "preparing"]):
                return "Preparing model files"
            if any(s in r for s in ["загруж", "loading"]):
                return "Loading pose recognition model"
            if any(s in r for s in ["инициал", "initial"]):
                return "Initializing neural network"
            if any(s in r for s in ["прогрев", "warmup", "warm"]):
                return "Warming up — almost ready"
            if any(s in r for s in ["готов", "ready", "done"]):
                return "Model is ready"
            if any(s in r for s in ["ошибка", "error"]):
                return f"Error: {raw}"
            return raw

    @staticmethod
    def _extract_model(title: str) -> str:
        import re
        m = re.search(r"(yolo[\w\d\-]+\.pt)", title, re.I)
        return m.group(1) if m else ""

    def _update_stage(self) -> None:
        if not self._stage_var:
            return
        pct = self.current_value
        stages = _STAGES_RU if self.lang == "ru" else _STAGES_EN
        if pct < 15:
            s = stages[0]
        elif pct < 45:
            s = stages[1]
        elif pct < 75:
            s = stages[2]
        elif pct < 95:
            s = stages[3]
        else:
            s = stages[4]
        self._stage_var.set(s)

    # ── Build ─────────────────────────────────────────────────────────────

    def _build(self, title: str, subtitle: str) -> None:
        c = self.colors
        lang = self.lang
        ru = (lang == "ru")

        self._frame = tk.Frame(self.root, bg=c.get("bg", "#0a0c10"))
        self._frame.place(x=0, y=0, relwidth=1, relheight=1)
        self._frame.lift()

        card = tk.Frame(
            self._frame,
            bg=c.get("card", "#14171c"),
            padx=52, pady=36)
        card.place(relx=0.5, rely=0.48, anchor="center")

        # Логотип
        logo_f = tk.Frame(card, bg=c.get("card", "#14171c"))
        logo_f.pack(pady=(0, 0))

        tk.Label(logo_f,
                 text="PARALLEL FINDER",
                 font=("Inter", 16, "bold"),
                 bg=c.get("card", "#14171c"),
                 fg=c.get("accent", "#3b82f6")).pack()

        tk.Label(logo_f,
                 text="Alpha v13",
                 font=("Inter", 8),
                 bg=c.get("card", "#14171c"),
                 fg=self._blend(
                     c.get("text_secondary", "#6b7280"),
                     c.get("card", "#14171c"), 0.3)).pack(pady=(1, 0))

        # Спиннер + кольцо
        spin_wrap = tk.Frame(card, bg=c.get("card", "#14171c"))
        spin_wrap.pack(pady=(18, 18))

        self._ring_cv = tk.Canvas(
            spin_wrap, width=110, height=110,
            bg=c.get("card", "#14171c"),
            highlightthickness=0)
        self._ring_cv.grid(row=0, column=0)

        self._spinner_cv = tk.Canvas(
            spin_wrap, width=110, height=110,
            bg=c.get("card", "#14171c"),
            highlightthickness=0)
        self._spinner_cv.grid(row=0, column=0)

        # Заголовок (локализованный)
        self._title_var = tk.StringVar(
            value=_MAIN_TITLE.get(lang, _MAIN_TITLE["en"]))
        tk.Label(card,
                 textvariable=self._title_var,
                 font=("Inter", 13, "bold"),
                 bg=c.get("card", "#14171c"),
                 fg=c.get("text", "#ffffff"),
                 justify="center").pack(pady=(0, 4))

        self._model_var = tk.StringVar(value=self._model_name or "")
        tk.Label(card,
                 textvariable=self._model_var,
                 font=("Inter", 9),
                 bg=c.get("card", "#14171c"),
                 fg=c.get("text_secondary", "#6b7280")).pack(pady=(0, 14))

        # Этап
        stages = _STAGES_RU if ru else _STAGES_EN
        self._stage_var = tk.StringVar(value=stages[0])
        tk.Label(card,
                 textvariable=self._stage_var,
                 font=("Inter", 10, "bold"),
                 bg=c.get("card", "#14171c"),
                 fg=c.get("accent", "#3b82f6")).pack(pady=(0, 3))

        self._sub_var = tk.StringVar(
            value=self._humanize(subtitle) if subtitle
            else _DEFAULT_SUB.get(lang, _DEFAULT_SUB["en"]))
        tk.Label(card,
                 textvariable=self._sub_var,
                 font=("Inter", 10),
                 bg=c.get("card", "#14171c"),
                 fg=c.get("text_secondary", "#6b7280")).pack(pady=(0, 16))

        # Прогресс-бар
        prog_outer = tk.Frame(card, bg=c.get("card", "#14171c"))
        prog_outer.pack(fill=tk.X, pady=(0, 6))

        pct_row = tk.Frame(prog_outer, bg=c.get("card", "#14171c"))
        pct_row.pack(fill=tk.X, pady=(0, 4))
        self._pct_lbl = tk.Label(
            pct_row, text="0%",
            font=("Inter", 11, "bold"),
            bg=c.get("card", "#14171c"),
            fg=c.get("accent", "#3b82f6"))
        self._pct_lbl.pack(side=tk.RIGHT)

        self._prog_var = tk.DoubleVar(value=0.0)
        self._prog_cv = tk.Canvas(
            prog_outer, width=320, height=14,
            bg=c.get("card", "#14171c"),
            highlightthickness=0)
        self._prog_cv.pack()

        # Подсказки
        hints = _HINTS_RU if ru else _HINTS_EN
        self._hint_var = tk.StringVar(value=hints[0])
        self._hint_lbl = tk.Label(
            card,
            textvariable=self._hint_var,
            font=("Inter", 9),
            bg=c.get("card", "#14171c"),
            fg=self._blend(
                c.get("text_secondary", "#6b7280"),
                c.get("card", "#14171c"), 0.2),
            wraplength=300,
            justify="center")
        self._hint_lbl.pack(pady=(10, 0))

        # «Первый запуск» — храним ссылку для обновления языка
        self._first_lbl = tk.Label(
            card,
            text=_FIRST_LAUNCH.get(lang, _FIRST_LAUNCH["en"]),
            font=("Inter", 8),
            bg=c.get("card", "#14171c"),
            fg=self._blend(
                c.get("text_secondary", "#6b7280"),
                c.get("card", "#14171c"), 0.35))
        self._first_lbl.pack(pady=(6, 0))

        self._draw_progress()
        self._draw_ring()
        self._start_anim()

    # ── Animation ─────────────────────────────────────────────────────────

    def _start_anim(self) -> None:
        self._angle = 0.0
        self._angle2 = 0.0
        self._pulse = 0.0
        self._dots = 0
        self._hint_idx = 0
        self._hint_fade = 1.0
        self._hint_fading = False
        self._tick()
        self._tick_dots()
        self._tick_hint()

    def _stop_anim(self) -> None:
        for attr in ("_anim_id", "_dot_id", "_hint_id"):
            v = getattr(self, attr, None)
            if v:
                try:
                    self.root.after_cancel(v)
                except Exception:
                    pass
                setattr(self, attr, None)

    def _tick(self) -> None:
        if not (self._frame and self._frame.winfo_exists()):
            return

        speed1 = 2.2 + self.current_value / 100 * 2.0
        speed2 = 1.6 + self.current_value / 100 * 1.2

        if self._success_mode:
            speed1 = 6.0
            speed2 = 4.0
            self._success_pulse = (self._success_pulse + 0.08) % 1.0

        self._angle = (self._angle + speed1) % 360.0
        self._angle2 = (self._angle2 - speed2) % 360.0
        self._pulse = (self._pulse + 0.032) % 1.0

        try:
            self._draw_spinner()
            self._draw_progress()
            self._draw_ring()
        except Exception:
            pass

        self._anim_id = self.root.after(16, self._tick)

    def _tick_dots(self) -> None:
        if not (self._frame and self._frame.winfo_exists()):
            return
        if self._sub_var:
            raw = self._sub_var.get().rstrip(".")
            _animated = [
                "Подготавлива", "Загружа", "Инициализ", "Прогрева",
                "Preparing",    "Loading", "Initializ", "Warming",
            ]
            if any(raw.startswith(s) for s in _animated):
                self._sub_var.set(raw + "." * ((self._dots % 3) + 1))
        self._dots += 1
        self._dot_id = self.root.after(520, self._tick_dots)

    def _tick_hint(self) -> None:
        if not (self._frame and self._frame.winfo_exists()):
            return
        hints = _HINTS_RU if self.lang == "ru" else _HINTS_EN
        self._hint_idx = (self._hint_idx + 1) % len(hints)
        self._hint_var.set(hints[self._hint_idx])
        self._hint_id = self.root.after(4000, self._tick_hint)

    # ── Draw spinner ──────────────────────────────────────────────────────

    def _draw_spinner(self) -> None:
        c = self.colors
        cv = self._spinner_cv
        if not cv:
            return
        cv.delete("all")

        cx, cy = 55, 55
        R1 = 28
        R2 = 19

        accent = c.get("accent",    "#3b82f6")
        glow = c.get("glow",      "#1d4ed8")
        track = c.get("highlight", "#1e293b")
        bg = c.get("card",      "#14171c")

        if self._success_mode:
            t = min(1.0, self._success_pulse * 3)
            accent = self._blend("#22c55e", accent, t)
            glow = self._blend("#16a34a", glow,   t)

        brightness = 0.7 + self.current_value / 100 * 0.3

        self._arc(cv, cx, cy, R1, 0, 359.9, track, 3)
        self._arc(cv, cx, cy, R2, 0, 359.9,
                  self._blend(track, bg, 0.5), 2)

        N1, ARC1 = 20, 210
        for i in range(N1):
            fade = i / (N1 - 1)
            start = (self._angle - ARC1 + i * ARC1 / N1) % 360
            ext = ARC1 / N1 + 0.6
            gc = self._blend(glow,   bg, 1.0 - fade * 0.65 * brightness)
            self._arc(cv, cx, cy, R1, start, ext, gc,  7)
            col = self._blend(accent, bg, 1.0 - fade * 0.88 * brightness)
            self._arc(cv, cx, cy, R1, start, ext, col, 3)

        N2, ARC2 = 14, 150
        for i in range(N2):
            fade = i / (N2 - 1)
            start = (self._angle2 - ARC2 + i * ARC2 / N2) % 360
            ext = ARC2 / N2 + 0.5
            gc = self._blend(glow,   bg, 1.0 - fade * 0.55 * brightness)
            self._arc(cv, cx, cy, R2, start, ext, gc,  5)
            col = self._blend(accent, bg, 1.0 - fade * 0.85 * brightness)
            self._arc(cv, cx, cy, R2, start, ext, col, 2)

        ar = math.radians(self._angle)
        hx = cx + R1 * math.cos(ar)
        hy = cy - R1 * math.sin(ar)
        for sz, a in [(10, 0.2), (7, 0.45), (4, 0.8)]:
            col = self._blend(accent, bg, 1.0 - a * brightness)
            cv.create_oval(hx-sz, hy-sz, hx+sz, hy+sz,
                           fill=col, outline="")
        cv.create_oval(hx-2.5, hy-2.5, hx+2.5, hy+2.5,
                       fill="#ffffff", outline="")

        if self._success_mode:
            sp = self._success_pulse
            pr = 4 + 18 * math.sin(sp * math.pi) if sp < 0.5 else 4
        else:
            pr = 5 + 2.5 * math.sin(self._pulse * math.pi * 2)

        for sz, a in [(pr+5, 0.12), (pr+2, 0.30), (pr, 0.60)]:
            col = self._blend(accent, bg, 1.0 - a * brightness)
            cv.create_oval(cx-sz, cy-sz, cx+sz, cy+sz,
                           fill=col, outline="")
        cv.create_oval(cx-2.5, cy-2.5, cx+2.5, cy+2.5,
                       fill="#ffffff", outline="")

    # ── Draw ring ─────────────────────────────────────────────────────────

    def _draw_ring(self) -> None:
        cv = self._ring_cv
        if not cv:
            return
        cv.delete("all")

        c = self.colors
        cx, cy = 55, 55
        R = 50
        pct = self.current_value

        accent = c.get("accent",    "#3b82f6")
        glow = c.get("glow",      "#1d4ed8")
        track = c.get("highlight", "#1e293b")
        bg = c.get("card",      "#14171c")

        if self._success_mode:
            t = min(1.0, self._success_pulse * 3)
            accent = self._blend("#22c55e", accent, t)
            glow = self._blend("#16a34a", glow,   t)

        self._arc(cv, cx, cy, R, 0, 359.9, track, 4)

        if pct > 0:
            extent = pct / 100 * 359.9
            for i in range(4, 0, -1):
                gc = self._blend(glow, bg, 1.0 - i / 4 * 0.5)
                self._arc(cv, cx, cy, R, 90, extent, gc, 4 + i * 2)
            self._arc(cv, cx, cy, R, 90, extent, accent, 4)

            end_angle = math.radians(90 - extent)
            ex = cx + R * math.cos(end_angle)
            ey = cy - R * math.sin(end_angle)
            for sz, a in [(8, 0.25), (5, 0.6)]:
                col = self._blend(accent, bg, 1.0 - a)
                cv.create_oval(ex-sz, ey-sz, ex+sz, ey+sz,
                               fill=col, outline="")
            cv.create_oval(ex-3, ey-3, ex+3, ey+3,
                           fill="#ffffff", outline="")

    # ── Draw progress bar ─────────────────────────────────────────────────

    def _draw_progress(self) -> None:
        cv = self._prog_cv
        if not cv:
            return
        c = self.colors
        pct = self._prog_var.get() if self._prog_var else 0.0
        W, H = 320, 14
        r = H // 2

        cv.delete("all")

        bg_col = c.get("highlight", "#1e293b")
        fill_col = c.get("accent",    "#3b82f6")
        glow_col = c.get("glow",      "#1d4ed8")
        card_col = c.get("card",      "#14171c")

        if self._success_mode:
            t = min(1.0, self._success_pulse * 3)
            fill_col = self._blend("#22c55e", fill_col, t)
            glow_col = self._blend("#16a34a", glow_col, t)

        self._rr(cv, 0, 0, W, H, r, bg_col)
        fill_w = max(0, int(W * pct / 100))

        if fill_w >= r * 2:
            for i in range(5, 0, -1):
                glow_i = max(0, i - 2)
                gc = self._blend(glow_col, card_col, 1.0 - i / 5 * 0.55)
                self._rr(cv, 0, -glow_i, fill_w, H + glow_i, r, gc)

            self._rr(cv, 0, 0, fill_w, H, r, fill_col)

            shine = self._lighter(fill_col, 0.30)
            self._rr(cv, 2, 1, fill_w - 2, H // 2 + 1, r - 1, shine)

            t_pos = (math.sin(self._pulse * math.pi * 2) + 1) / 2
            bx = int(r + (fill_w - r * 2) * t_pos)
            bw = 28
            left = bx - bw // 2
            right = bx + bw // 2

            if left > r and right < fill_w - r:
                ph_center = self._lighter(fill_col, 0.40)
                ph_edge = self._lighter(fill_col, 0.15)
                mid = bw // 3
                cv.create_rectangle(
                    left,        2, left + mid,  H - 2,
                    fill=ph_edge,   outline="")
                cv.create_rectangle(
                    left + mid,  2, right - mid, H - 2,
                    fill=ph_center, outline="")
                cv.create_rectangle(
                    right - mid, 2, right,       H - 2,
                    fill=ph_edge,   outline="")

        elif fill_w > 0:
            cv.create_rectangle(0, 0, fill_w, H,
                                fill=fill_col, outline="")

        if self._pct_lbl:
            try:
                self._pct_lbl.config(text=f"{pct:.0f}%")
            except Exception:
                pass

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _arc(cv, cx, cy, r, start, extent, color, width) -> None:
        cv.create_arc(
            cx-r, cy-r, cx+r, cy+r,
            start=start, extent=extent,
            outline=color, width=width,
            style="arc")

    @staticmethod
    def _rr(cv, x1, y1, x2, y2, r, color) -> None:
        r = max(0, min(int(r), (x2-x1)//2, (y2-y1)//2))
        cv.create_arc(x1,     y1,     x1+r*2, y1+r*2,
                      start=90,  extent=90,  fill=color, outline="")
        cv.create_arc(x2-r*2, y1,     x2,     y1+r*2,
                      start=0,   extent=90,  fill=color, outline="")
        cv.create_arc(x1,     y2-r*2, x1+r*2, y2,
                      start=180, extent=90,  fill=color, outline="")
        cv.create_arc(x2-r*2, y2-r*2, x2,     y2,
                      start=270, extent=90,  fill=color, outline="")
        cv.create_rectangle(x1+r, y1,   x2-r, y2,   fill=color, outline="")
        cv.create_rectangle(x1,   y1+r, x2,   y2-r, fill=color, outline="")

    @staticmethod
    def _h2r(h: str):
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    @staticmethod
    def _r2h(r, g, b) -> str:
        return (f"#{max(0,min(255,int(r))):02x}"
                f"{max(0,min(255,int(g))):02x}"
                f"{max(0,min(255,int(b))):02x}")

    def _blend(self, c1: str, c2: str, t: float) -> str:
        t  = max(0.0, min(1.0, t))
        r1, g1, b1 = self._h2r(c1)
        r2, g2, b2 = self._h2r(c2)
        return self._r2h(
            r1 + (r2-r1)*t,
            g1 + (g2-g1)*t,
            b1 + (b2-b1)*t)

    def _lighter(self, c: str, f: float) -> str:
        r, g, b = self._h2r(c)
        return self._r2h(
            r + (255-r)*f,
            g + (255-g)*f,
            b + (255-b)*f)