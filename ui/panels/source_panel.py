from __future__ import annotations

import os
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from tkinter import filedialog, messagebox
from typing import Callable

import cv2

from ui.app_state import AppState
from ui.widgets.glow_button import GlowButton
from ui.widgets.smooth_scroll import SmoothScrollMixin


def _fmt_hms(secs: float) -> str:
    s = max(0.0, float(secs))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    ss = int(s % 60)
    return f"{h:02d}:{m:02d}:{ss:02d}"


def _fmt_num(n: int) -> str:
    return f"{n / 1000:.1f}K" if n >= 1000 else str(n)


# ── Строки локализации ────────────────────────────────────────────────────────

_STRINGS: dict[str, dict[str, str]] = {
    # Заголовок
    "title":              {"ru": "Источник",                              "en": "Source"},
    # Инфо-строки
    "files_count":        {"ru": "Файлов",                               "en": "Files"},
    "duration":           {"ru": "Длительность",                         "en": "Duration"},
    "frames":             {"ru": "Кадров",                               "en": "Frames"},
    # Список
    "no_videos":          {"ru": "Видео не добавлены",                   "en": "No videos added"},
    # Подсказка drag&drop
    "drop_hint":          {"ru": "📁 Перетащите видео сюда или нажмите кнопки ниже",
                                  "en": "📁 Drop videos here or use buttons below"},
    "drop_hint_loaded":   {"ru": "✓ {} видео загружено",                 "en": "✓ {} video(s) loaded"},
    # Кнопки
    "btn_select_video":   {"ru": "Выбрать видео",                        "en": "Select video"},
    "btn_select_folder":  {"ru": "Выбрать папку",                        "en": "Select folder"},
    "btn_clear":          {"ru": "Очистить",                             "en": "Clear"},
    "btn_remove":         {"ru": "Удалить выбранный",                    "en": "Remove selected"},
    # Диалоги
    "dlg_select_video":   {"ru": "Выберите видео",                       "en": "Select video"},
    "dlg_video_filter":   {"ru": "Видео",                                "en": "Video"},
    "dlg_all_files":      {"ru": "Все файлы",                            "en": "All files"},
    "dlg_select_folder":  {"ru": "Выберите папку с видео",               "en": "Select folder with videos"},
    "dlg_empty_title":    {"ru": "Пусто",                                "en": "Empty"},
    "dlg_empty_msg":      {"ru": "В папке нет поддерживаемых видеофайлов.",
                                  "en": "No supported video files found in the folder."},
    # Фото-панель
    "photo_check":        {"ru": "Найти человека по фото",               "en": "Find person by photo"},
    "photo_drop":         {"ru": "📷  Перетащите 1–2 фото\nили Ctrl+V из буфера",
                                  "en": "📷  Drop 1–2 photos here\nor Ctrl+V from clipboard"},
    "photo_none":         {"ru": "Фото не выбраны",                      "en": "No photos selected"},
    "photo_browse_title": {"ru": "Выберите 1–2 фото",                    "en": "Select 1–2 photos"},
    "photo_img_filter":   {"ru": "Изображения",                          "en": "Images"},
    "btn_clear_photos":   {"ru": "Очистить фото",                        "en": "Clear photos"},
    # Кадров прочерк
    "frames_dash":        {"ru": "Кадров: —",                            "en": "Frames: —"},
}


def _s(key: str, lang: str) -> str:
    return _STRINGS.get(key, {}).get(lang, key)


# ─────────────────────────────────────────────────────────────────────────────

class SourcePanel(tk.Frame, SmoothScrollMixin):
    """Левая панель — очередь видео и фото-поиск."""

    ROW_H = 28

    def __init__(self, parent, state: AppState,
                 colors: dict, callbacks: dict,
                 video_extensions, is_video_file_fn: Callable,
                 lang: str = "ru") -> None:
        super().__init__(parent, bg=colors["bg"])
        self.state            = state
        self.colors           = colors
        self.callbacks        = callbacks
        self.video_extensions = video_extensions
        self.is_video_file    = is_video_file_fn
        self._lang            = lang
        self._executor        = ThreadPoolExecutor(max_workers=4)

        # Ссылки на виджеты, которые нужно обновлять при смене языка
        self._title_label:       tk.Label | None = None
        self._src_count_var:     tk.StringVar | None = None
        self._src_dur_var:       tk.StringVar | None = None
        self._src_frm_var:       tk.StringVar | None = None
        self._drop_hint:         tk.Label | None = None
        self._btn_select_video:  GlowButton | None = None
        self._btn_select_folder: GlowButton | None = None
        self._btn_clear:         GlowButton | None = None
        self._btn_remove:        GlowButton | None = None
        self._photo_check:       tk.Checkbutton | None = None
        self._photo_drop:        tk.Label | None = None
        self._photo_list_var:    tk.StringVar | None = None
        self._btn_clear_photos:  GlowButton | None = None

        # Счётчики для stat-строк (чтобы обновлять при смене языка)
        self._last_n:            int   = 0
        self._last_dur:          float = 0.0
        self._last_frames:       int   = 0
        self._last_frames_valid: bool  = False

        self._build()

    # ── Построение UI ─────────────────────────────────────────────────────

    def _build(self) -> None:
        c     = self.colors
        lang  = self._lang

        card  = tk.Frame(self, bg=c["card"])
        card.pack(fill=tk.X, pady=(0, 10))
        inner = tk.Frame(card, bg=c["card"], padx=16, pady=14)
        inner.pack(fill=tk.X)

        # Заголовок
        self._title_label = tk.Label(
            inner,
            text=_s("title", lang),
            font=("Inter", 13, "bold"),
            bg=c["card"], fg=c["text"])
        self._title_label.pack(anchor="w", pady=(0, 8))

        # Инфо-строки
        info = tk.Frame(inner, bg=c["card"])
        info.pack(fill=tk.X, pady=(0, 8))

        self._src_count_var = tk.StringVar(
            value=f'{_s("files_count", lang)}: 0')
        self._src_dur_var   = tk.StringVar(
            value=f'{_s("duration", lang)}: 00:00:00')
        self._src_frm_var   = tk.StringVar(
            value=f'{_s("frames", lang)}: 0')

        for v in (self._src_count_var,
                  self._src_dur_var,
                  self._src_frm_var):
            tk.Label(info, textvariable=v,
                     font=("Inter", 9),
                     bg=c["card"],
                     fg=c["text_secondary"]).pack(anchor="w")

        # Canvas-список
        list_outer = tk.Frame(inner, bg=c["bg"],
                              highlightbackground=c["border"],
                              highlightthickness=1)
        list_outer.pack(fill=tk.X, pady=(0, 8))

        self._list_canvas = tk.Canvas(
            list_outer, bg=c["bg"],
            highlightthickness=0, height=90)
        sb = tk.Scrollbar(list_outer, orient=tk.VERTICAL,
                          command=self._list_canvas.yview)
        self._list_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._list_canvas.pack(side=tk.LEFT,
                               fill=tk.BOTH, expand=True)

        self._bind_smooth_scroll(
            self._list_canvas, self._list_canvas.yview_scroll)
        self._list_canvas.bind("<Configure>", self._render_list)
        self._list_canvas.bind("<Button-1>",  self._on_list_click)

        self._list_selected: int = -1

        # Подсказка drag&drop
        self._drop_hint = tk.Label(
            inner,
            text=_s("drop_hint", lang),
            font=("Inter", 9),
            bg=c["card"],
            fg=c["text_secondary"],
            wraplength=370)
        self._drop_hint.pack(anchor="w", pady=(0, 8))

        # Кнопки
        btn_row = tk.Frame(inner, bg=c["card"])
        btn_row.pack(fill=tk.X, pady=(0, 4))

        self._btn_select_video = GlowButton(
            btn_row,
            text=_s("btn_select_video", lang),
            command=self._select_video,
            bg_color=c["accent"],
            hover_color=c["accent_hover"],
            width=115, height=30)
        self._btn_select_video.pack(side=tk.LEFT, padx=(0, 6))

        self._btn_select_folder = GlowButton(
            btn_row,
            text=_s("btn_select_folder", lang),
            command=self._select_folder,
            bg_color=c["accent"],
            hover_color=c["accent_hover"],
            width=115, height=30)
        self._btn_select_folder.pack(side=tk.LEFT, padx=(0, 6))

        self._btn_clear = GlowButton(
            btn_row,
            text=_s("btn_clear", lang),
            command=self._clear,
            bg_color=c["highlight"],
            hover_color=c["border"],
            width=80, height=30)
        self._btn_clear.pack(side=tk.LEFT)

        self._btn_remove = GlowButton(
            inner,
            text=_s("btn_remove", lang),
            command=self._remove_selected,
            bg_color=c["highlight"],
            hover_color=c["border"],
            width=160, height=26,
            font=("Inter", 9, "bold"))
        self._btn_remove.pack(anchor="w", pady=(0, 0))

        self._build_photo_panel(inner)

    def _build_photo_panel(self, parent: tk.Widget) -> None:
        c    = self.colors
        lang = self._lang

        tk.Frame(parent, bg=c["border"], height=1).pack(
            fill=tk.X, pady=10)

        hdr = tk.Frame(parent, bg=c["card"])
        hdr.pack(fill=tk.X)

        self._use_photo_var = tk.BooleanVar(value=False)
        self._photo_check = tk.Checkbutton(
            hdr,
            text=_s("photo_check", lang),
            variable=self._use_photo_var,
            command=self._toggle_photo,
            bg=c["card"], fg=c["text"],
            selectcolor=c["bg"],
            activebackground=c["card"],
            font=("Inter", 10))
        self._photo_check.pack(anchor="w")

        self._photo_panel = tk.Frame(parent, bg=c["card"])

        # Drop-зона для фото
        drop_outer = tk.Frame(
            self._photo_panel, bg=c["highlight"],
            highlightbackground=c["border"],
            highlightthickness=1)
        drop_outer.pack(fill=tk.X, pady=(6, 4))

        self._photo_drop = tk.Label(
            drop_outer,
            text=_s("photo_drop", lang),
            font=("Inter", 9),
            bg=c["highlight"],
            fg=c["text_secondary"],
            pady=14, cursor="hand2")
        self._photo_drop.pack(fill=tk.X)
        self._photo_drop.bind(
            "<Button-1>", lambda _e: self._browse_photo())

        self._photo_list_var = tk.StringVar(
            value=_s("photo_none", lang))
        tk.Label(self._photo_panel,
                 textvariable=self._photo_list_var,
                 font=("Inter", 9),
                 bg=c["card"],
                 fg=c["text_secondary"],
                 wraplength=360).pack(anchor="w", pady=2)

        self._btn_clear_photos = GlowButton(
            self._photo_panel,
            text=_s("btn_clear_photos", lang),
            command=self._clear_photos,
            bg_color=c["highlight"],
            hover_color=c["border"],
            width=120, height=24,
            font=("Inter", 9, "bold"))
        self._btn_clear_photos.pack(anchor="w", pady=2)

    # ── Публичный API: смена языка ─────────────────────────────────────────

    def set_lang(self, lang: str) -> None:
        """Обновляет все тексты панели без полной перестройки."""
        self._lang = lang

        # Заголовок
        if self._title_label:
            self._title_label.config(text=_s("title", lang))

        # Инфо-строки (обновляем с текущими значениями)
        self._refresh_stat_labels()

        # Подсказка
        self._refresh_drop_hint()

        # Кнопки
        if self._btn_select_video:
            self._btn_select_video.set_text(
                _s("btn_select_video", lang))
        if self._btn_select_folder:
            self._btn_select_folder.set_text(
                _s("btn_select_folder", lang))
        if self._btn_clear:
            self._btn_clear.set_text(_s("btn_clear", lang))
        if self._btn_remove:
            self._btn_remove.set_text(_s("btn_remove", lang))

        # Фото-панель
        if self._photo_check:
            self._photo_check.config(
                text=_s("photo_check", lang))
        if self._photo_drop:
            self._photo_drop.config(
                text=_s("photo_drop", lang))
        if self._btn_clear_photos:
            self._btn_clear_photos.set_text(
                _s("btn_clear_photos", lang))

        # Обновляем статус фото
        self._refresh_photo_list_label()

        # Перерисовываем список (текст "Видео не добавлены")
        self._render_list()

    # ── Вспомогательные методы обновления текстов ─────────────────────────

    def _refresh_stat_labels(self) -> None:
        lang = self._lang
        n    = self._last_n

        if self._src_count_var:
            self._src_count_var.set(
                f'{_s("files_count", lang)}: {n}')
        if self._src_dur_var:
            self._src_dur_var.set(
                f'{_s("duration", lang)}: '
                f'{_fmt_hms(self._last_dur)}')
        if self._src_frm_var:
            if self._last_frames_valid:
                self._src_frm_var.set(
                    f'{_s("frames", lang)}: '
                    f'{_fmt_num(self._last_frames)}')
            else:
                self._src_frm_var.set(
                    f'{_s("frames", lang)}: '
                    + ("—" if n > 0 else "0"))

    def _refresh_drop_hint(self) -> None:
        lang = self._lang
        n    = self._last_n
        if self._drop_hint:
            if n > 0:
                self._drop_hint.config(
                    text=_s("drop_hint_loaded", lang).format(n))
            else:
                self._drop_hint.config(
                    text=_s("drop_hint", lang))

    def _refresh_photo_list_label(self) -> None:
        if not self._photo_list_var:
            return
        lang = self._lang
        if self.state.ref_photos:
            self._photo_list_var.set(
                "\n".join(os.path.basename(p)
                          for p in self.state.ref_photos))
        else:
            self._photo_list_var.set(_s("photo_none", lang))

    # ── Фото-панель ───────────────────────────────────────────────────────

    def _toggle_photo(self) -> None:
        val = self._use_photo_var.get()
        self.state.use_photo_search = val    
        if val:
            self._photo_panel.pack(fill=tk.X, pady=(4, 0))
        else:
            self._photo_panel.pack_forget()

    def _browse_photo(self) -> None:
        lang  = self._lang
        paths = filedialog.askopenfilenames(
            title=_s("photo_browse_title", lang),
            filetypes=[
                (_s("photo_img_filter", lang),
                 "*.jpg *.jpeg *.png *.bmp *.webp")])
        self._add_photos(list(paths))

    def _add_photos(self, paths: list[str]) -> None:
        for p in paths:
            if (p not in self.state.ref_photos
                    and len(self.state.ref_photos) < 2):
                self.state.ref_photos.append(p)
        self._refresh_photo_list_label()

    def _clear_photos(self) -> None:
        self.state.ref_photos.clear()
        self._refresh_photo_list_label()

    # ── Список видео (Canvas) ─────────────────────────────────────────────

    def _render_list(self, _e=None) -> None:
        self._list_canvas.delete("all")
        c    = self.colors
        lang = self._lang
        cw   = self._list_canvas.winfo_width() or 380
        n    = len(self.state.video_queue)

        if n == 0:
            self._list_canvas.create_text(
                cw // 2, 45,
                text=_s("no_videos", lang),
                fill=c["text_secondary"],
                font=("Inter", 9))
            return

        total_h = n * self.ROW_H
        self._list_canvas.config(
            scrollregion=(0, 0, cw, max(total_h, 90)))

        view_top = self._list_canvas.yview()[0]
        y_off    = view_top * total_h
        ch       = self._list_canvas.winfo_height() or 90

        start = max(0, int(y_off // self.ROW_H) - 1)
        end   = min(n, start + int(ch // self.ROW_H) + 3)

        for i in range(start, end):
            y   = i * self.ROW_H - y_off
            sel = i == self._list_selected

            bg = c["active_row"] if sel else (
                c["highlight"] if i % 2 == 0 else c["bg"])
            self._list_canvas.create_rectangle(
                0, y, cw, y + self.ROW_H - 1,
                fill=bg, outline="")

            name = os.path.basename(self.state.video_queue[i])
            self._list_canvas.create_text(
                8, y + self.ROW_H // 2,
                text=f"{i + 1}.",
                anchor="w",
                fill=c["text_secondary"],
                font=("Inter", 8))
            self._list_canvas.create_text(
                32, y + self.ROW_H // 2,
                text=name,
                anchor="w",
                fill=c["text"] if sel else c["text_secondary"],
                font=("Inter", 9,
                      "bold" if sel else "normal"))
            self._list_canvas.create_line(
                0, y + self.ROW_H - 1,
                cw, y + self.ROW_H - 1,
                fill=c["border"])

    def _on_list_click(self, event: tk.Event) -> None:
        view_top = self._list_canvas.yview()[0]
        n        = len(self.state.video_queue)
        total_h  = n * self.ROW_H
        abs_y    = event.y + view_top * total_h
        i        = int(abs_y // self.ROW_H)
        if 0 <= i < n:
            self._list_selected = i
            self._render_list()

    # ── Очередь ───────────────────────────────────────────────────────────

    def _select_video(self) -> None:
        lang    = self._lang
        ext_str = " ".join(f"*{e}" for e in self.video_extensions)
        paths   = filedialog.askopenfilenames(
            title=_s("dlg_select_video", lang),
            filetypes=[
                (_s("dlg_video_filter", lang), ext_str),
                (_s("dlg_all_files",    lang), "*.*")])
        if paths:
            self.add_to_queue(list(paths))

    def _select_folder(self) -> None:
        lang   = self._lang
        folder = filedialog.askdirectory(
            title=_s("dlg_select_folder", lang))
        if not folder:
            return
        found = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if self.is_video_file(os.path.join(folder, f))])
        if found:
            self.add_to_queue(found)
        else:
            messagebox.showwarning(
                _s("dlg_empty_title", lang),
                _s("dlg_empty_msg",   lang))

    def add_to_queue(self, paths: list[str]) -> None:
        for p in paths:
            if p not in self.state.video_queue:
                self.state.video_queue.append(p)
        self.state.batch_mode = len(self.state.video_queue) > 1
        self.refresh_ui()
        if "on_queue_changed" in self.callbacks:
            self.callbacks["on_queue_changed"]()

    def _remove_selected(self) -> None:
        i = self._list_selected
        if 0 <= i < len(self.state.video_queue):
            self.state.video_queue.pop(i)
            self._list_selected = max(0, i - 1)
            self.refresh_ui()
            if "on_queue_changed" in self.callbacks:
                self.callbacks["on_queue_changed"]()

    def _clear(self) -> None:
        self.state.video_queue.clear()
        self.state.batch_mode = False
        self._list_selected   = -1
        self.refresh_ui()
        if "on_queue_changed" in self.callbacks:
            self.callbacks["on_queue_changed"]()

    def refresh_ui(self) -> None:
        lang = self._lang
        n    = len(self.state.video_queue)

        self._last_n            = n
        self._last_frames_valid = False

        self._render_list()
        self._refresh_drop_hint()

        if self._src_count_var:
            self._src_count_var.set(
                f'{_s("files_count", lang)}: {n}')

        paths = list(self.state.video_queue)
        if paths:
            self._executor.submit(
                self._load_durations_async, paths)
        else:
            # Сброс при очистке
            self._last_dur          = 0.0
            self._last_frames       = 0
            self._last_frames_valid = False
            self._refresh_stat_labels()

    def _load_durations_async(self, paths: list[str]) -> None:
        total_dur    = 0.0
        total_frames = 0
        frames_valid = False

        for p in paths:
            if p in self.state.video_durations_cache:
                dur = self.state.video_durations_cache[p]
            else:
                try:
                    cap = cv2.VideoCapture(p)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    dur = frm / fps if fps > 0 else 0.0
                    cap.release()
                    self.state.video_durations_cache[p] = dur
                    total_frames += frm
                    frames_valid  = True
                except Exception:
                    dur = 0.0
            total_dur += dur

        # Сохраняем для повторного использования при смене языка
        self._last_dur          = total_dur
        self._last_frames       = total_frames
        self._last_frames_valid = frames_valid

        try:
            lang = self._lang
            if self._src_dur_var:
                self._src_dur_var.set(
                    f'{_s("duration", lang)}: '
                    f'{_fmt_hms(total_dur)}')
            if self._src_frm_var:
                if frames_valid:
                    self._src_frm_var.set(
                        f'{_s("frames", lang)}: '
                        f'{_fmt_num(total_frames)}')
                else:
                    self._src_frm_var.set(
                        f'{_s("frames", lang)}: —')
        except Exception:
            pass