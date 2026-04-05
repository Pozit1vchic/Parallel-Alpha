#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ui/controllers/export_controller.py
"""
from __future__ import annotations

import json
import os
from tkinter import filedialog, messagebox

import cv2

from ui.app_state import AppState
from utils.locales import t


def _fmt_hms(secs: float) -> str:
    s  = max(0.0, float(secs))
    h  = int(s // 3600)
    m  = int((s % 3600) // 60)
    ss = int(s % 60)
    return f"{h:02d}:{m:02d}:{ss:02d}"


class ExportController:

    def __init__(
        self,
        state:             AppState,
        app_display_name:  str,
        app_short_version: str,
        app_build_version: str,
        app_author:        str,
    ) -> None:
        self.state             = state
        self.app_display_name  = app_display_name
        self.app_short_version = app_short_version
        self.app_build_version = app_build_version
        self.app_author        = app_author
        self._exporting        = False  # защита от двойного клика

    # ── Общая проверка ────────────────────────────────────────────────────

    def _check(self) -> bool:
        if not self.state.matches:
            messagebox.showwarning(
                t("warning"), t("err_no_results_export")
            )
            return False
        if self._exporting:
            return False
        return True

    def _check_overwrite(self, path: str) -> bool:
        if os.path.exists(path):
            return messagebox.askyesno(
                t("warning"), t("err_file_overwrite")
            )
        return True

    # ── JSON ─────────────────────────────────────────────────────────────

    def export_json(self) -> None:
        if not self._check():
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        if not self._check_overwrite(path):
            return

        self._exporting = True
        try:
            clean = []
            for m in self.state.matches:
                row = {
                    k: v for k, v in m.items()
                    if k not in ("vec", "_marked_bad")
                }
                row["t1_hms"] = _fmt_hms(row.get("t1", 0.0))
                row["t2_hms"] = _fmt_hms(row.get("t2", 0.0))
                clean.append(row)

            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "matches": clean,
                    "meta": {
                        "app":     self.app_display_name,
                        "version": self.app_short_version,
                        "build":   self.app_build_version,
                        "author":  self.app_author,
                        "total":   len(clean),
                    },
                }, f, indent=2, ensure_ascii=False)

            messagebox.showinfo(
                t("info"), f"JSON сохранён:\n{path}"
            )
        except PermissionError:
            messagebox.showerror(
                t("error"), t("err_no_access")
            )
        except OSError as e:
            if "space" in str(e).lower() or "disk" in str(e).lower():
                messagebox.showerror(
                    t("error"), t("err_no_disk_space")
                )
            else:
                messagebox.showerror(t("error"), str(e))
        finally:
            self._exporting = False

    # ── TXT ──────────────────────────────────────────────────────────────

    def export_txt(self) -> None:
        if not self._check():
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt")],
        )
        if not path:
            return
        if not self._check_overwrite(path):
            return

        self._exporting = True
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    f"{self.app_display_name} "
                    f"{self.app_short_version} — РЕЗУЛЬТАТЫ\n"
                )
                f.write("=" * 52 + "\n\n")
                for i, m in enumerate(self.state.matches):
                    try:
                        v1 = os.path.basename(
                            self.state.video_queue[m["v1_idx"]]
                        )
                        v2 = os.path.basename(
                            self.state.video_queue[m["v2_idx"]]
                        )
                    except (IndexError, KeyError):
                        v1 = v2 = "unknown"
                    f.write(
                        f"{i+1:04d}.  {m.get('sim', 0):.1%}  "
                        f"{v1} @ {_fmt_hms(m.get('t1', 0))}  ↔  "
                        f"{v2} @ {_fmt_hms(m.get('t2', 0))}\n"
                    )

            messagebox.showinfo(
                t("info"), f"TXT сохранён:\n{path}"
            )
        except PermissionError:
            messagebox.showerror(t("error"), t("err_no_access"))
        except Exception as e:
            messagebox.showerror(t("error"), str(e))
        finally:
            self._exporting = False

    # ── EDL ──────────────────────────────────────────────────────────────

    def export_edl(self) -> None:
        if not self._check():
            return
        if not self.state.video_queue:
            messagebox.showwarning(
                t("warning"), t("err_no_results_export")
            )
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".edl",
            filetypes=[("Edit Decision List", "*.edl")],
        )
        if not path:
            return
        if not self._check_overwrite(path):
            return

        self._exporting = True
        try:
            # Определить FPS
            try:
                cap = cv2.VideoCapture(self.state.video_queue[0])
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()
            except Exception:
                fps = 30.0

            def tc(seconds: float) -> str:
                s  = max(0.0, seconds)
                h  = int(s / 3600)
                mm = int((s % 3600) / 60)
                ss = int(s % 60)
                fr = min(int((s - int(s)) * fps), int(fps) - 1)
                return f"{h:02d}:{mm:02d}:{ss:02d}:{fr:02d}"

            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    f"TITLE: {self.app_display_name} Export\n"
                    f"FCM: NON-DROP FRAME\n\n"
                )
                tl = 3600.0
                for i, m in enumerate(self.state.matches):
                    try:
                        c1 = os.path.basename(
                            self.state.video_queue[m["v1_idx"]]
                        )
                        c2 = os.path.basename(
                            self.state.video_queue[m["v2_idx"]]
                        )
                    except (IndexError, KeyError):
                        c1 = c2 = "unknown"

                    t1 = m.get("t1", 0.0)
                    t2 = m.get("t2", 0.0)
                    f.write(
                        f"{i*2+1:03d}  AX       V     C        "
                        f"{tc(t1)} {tc(t1+2)} "
                        f"{tc(tl)} {tc(tl+2)}\n"
                        f"* FROM CLIP NAME: {c1}\n"
                        f"{i*2+2:03d}  AX       V     C        "
                        f"{tc(t2)} {tc(t2+2)} "
                        f"{tc(tl+2)} {tc(tl+4)}\n"
                        f"* FROM CLIP NAME: {c2}\n\n"
                    )
                    tl += 5.0

            messagebox.showinfo(
                t("info"), f"EDL сохранён:\n{path}"
            )
        except PermissionError:
            messagebox.showerror(t("error"), t("err_no_access"))
        except Exception as e:
            messagebox.showerror(t("error"), str(e))
        finally:
            self._exporting = False