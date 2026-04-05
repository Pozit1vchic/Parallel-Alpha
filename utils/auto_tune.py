#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""utils/auto_tune.py — Автоматическая настройка параметров анализа."""
from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Optional

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@dataclass
class SystemProfile:
    ram_total_gb:     float = 0.0
    ram_available_gb: float = 0.0
    cpu_cores_physical: int = 1
    cpu_cores_logical:  int = 1
    cpu_freq_mhz:     float = 0.0
    has_gpu:          bool  = False
    gpu_name:         str   = ""
    gpu_vram_gb:      float = 0.0
    gpu_vram_free_gb: float = 0.0
    gpu_compute_cap:  tuple = (0, 0)
    os_name:          str   = ""
    is_laptop:        bool  = False


@dataclass
class TuneResult:
    batch_size:      int   = 8
    chunk_size:      int   = 1000
    max_results:     int   = 500
    quality:         str   = "medium"
    quality_ru:      str   = "Средне"
    quality_en:      str   = "Medium"
    device:          str   = "cpu"
    half_precision:  bool  = False
    num_workers:     int   = 0
    prefetch_factor: int   = 2
    pin_memory:      bool  = False
    reason:          str   = ""
    warnings:        list  = field(default_factory=list)
    profile:         SystemProfile = field(
        default_factory=SystemProfile)


def _get_system_profile() -> SystemProfile:
    p = SystemProfile()
    p.os_name = platform.system()

    if _HAS_PSUTIL:
        try:
            vm = psutil.virtual_memory()
            p.ram_total_gb     = vm.total     / (1024 ** 3)
            p.ram_available_gb = vm.available / (1024 ** 3)
        except Exception:
            pass

        try:
            p.cpu_cores_physical = psutil.cpu_count(logical=False) or 1
            p.cpu_cores_logical  = psutil.cpu_count(logical=True)  or 1
        except Exception:
            pass

        try:
            freq = psutil.cpu_freq()
            if freq:
                p.cpu_freq_mhz = freq.current
        except Exception:
            pass

        try:
            battery    = psutil.sensors_battery()
            p.is_laptop = battery is not None
        except Exception:
            pass

    if _HAS_TORCH:
        try:
            if torch.cuda.is_available():
                p.has_gpu  = True
                p.gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                p.gpu_vram_gb     = props.total_memory / (1024 ** 3)
                p.gpu_compute_cap = (props.major, props.minor)
                free, _ = torch.cuda.mem_get_info(0)
                p.gpu_vram_free_gb = free / (1024 ** 3)
        except Exception:
            p.has_gpu = False

    return p


_QUALITY_LABELS: dict[str, dict[str, str]] = {
    "fast":    {"ru": "Быстро",   "en": "Fast"},
    "medium":  {"ru": "Средне",   "en": "Medium"},
    "maximum": {"ru": "Максимум", "en": "Maximum"},
}


def auto_tune(override_quality: Optional[str] = None) -> TuneResult:
    """
    Подбирает batch_size, chunk_size, quality под железо.

    ┌──────────────┬──────────┬───────────────────────────────────┐
    │  GPU VRAM    │  RAM     │  Результат                        │
    ├──────────────┼──────────┼───────────────────────────────────┤
    │  ≥ 16 GB     │  ≥ 32 G  │  batch=64  chunk=8000  maximum   │
    │  ≥ 10 GB     │  ≥ 24 G  │  batch=48  chunk=6000  maximum   │
    │  ≥ 8 GB      │  ≥ 16 G  │  batch=32  chunk=4000  maximum   │
    │  ≥ 6 GB      │  ≥ 12 G  │  batch=24  chunk=3000  medium    │
    │  ≥ 4 GB      │  ≥ 8 G   │  batch=16  chunk=2000  medium    │
    │  ≥ 2 GB      │  ≥ 6 G   │  batch=8   chunk=1500  medium    │
    │  < 2 GB      │  any     │  batch=4   chunk=1000  fast      │
    │  No GPU      │  ≥ 16 G  │  batch=8   chunk=2000  medium    │
    │  No GPU      │  ≥ 8 G   │  batch=4   chunk=1500  fast      │
    │  No GPU      │  < 8 G   │  batch=2   chunk=800   fast      │
    └──────────────┴──────────┴───────────────────────────────────┘
    """
    profile = _get_system_profile()
    result  = TuneResult(profile=profile)
    warns   = result.warnings

    vram   = profile.gpu_vram_free_gb if profile.has_gpu else 0.0
    ram    = profile.ram_available_gb
    laptop = 0.75 if profile.is_laptop else 1.0

    if profile.has_gpu:
        result.device     = "cuda"
        result.pin_memory = True
        result.num_workers = min(4, profile.cpu_cores_physical)
        cc = profile.gpu_compute_cap
        result.half_precision = (cc[0] >= 7)

        eff = vram * laptop

        if eff >= 16:
            result.batch_size  = 64
            result.chunk_size  = 8000
            result.max_results = 2000
            result.quality     = "maximum"
            result.reason = (
                f"Флагманская GPU ({vram:.1f} GB VRAM) — "
                f"максимальные настройки")

        elif eff >= 10:
            result.batch_size  = 48
            result.chunk_size  = 6000
            result.max_results = 1500
            result.quality     = "maximum"
            result.reason = (
                f"Мощная GPU ({vram:.1f} GB VRAM)")

        elif eff >= 8:
            result.batch_size  = 32
            result.chunk_size  = 4000
            result.max_results = 1000
            result.quality     = "maximum"
            result.reason = (
                f"Высокопроизводительная GPU ({vram:.1f} GB VRAM)")

        elif eff >= 6:
            result.batch_size  = 24
            result.chunk_size  = 3000
            result.max_results = 800
            result.quality     = "medium"
            result.reason = (
                f"Средняя GPU ({vram:.1f} GB VRAM)")

        elif eff >= 4:
            result.batch_size  = 16
            result.chunk_size  = 2000
            result.max_results = 600
            result.quality     = "medium"
            result.reason = (
                f"Достаточно VRAM ({vram:.1f} GB)")

        elif eff >= 2:
            result.batch_size  = 8
            result.chunk_size  = 1500
            result.max_results = 400
            result.quality     = "medium"
            result.reason = (
                f"Ограниченная VRAM ({vram:.1f} GB)")
            warns.append(
                f"Мало VRAM ({vram:.1f} GB) — quality снижен до Medium")

        else:
            result.batch_size  = 4
            result.chunk_size  = 1000
            result.max_results = 200
            result.quality     = "fast"
            result.reason = (
                f"Очень мало VRAM ({vram:.1f} GB)")
            warns.append(
                "Рекомендуется закрыть другие GPU-приложения")

        # RAM-корректировка
        if ram < 4:
            result.chunk_size = min(result.chunk_size, 800)
            warns.append(
                f"Критически мало RAM ({ram:.1f} GB) — chunk уменьшен")
        elif ram < 8:
            result.chunk_size = min(result.chunk_size, 1500)
            warns.append(
                f"Мало RAM ({ram:.1f} GB) — chunk ограничен")

    else:
        # CPU режим
        result.device         = "cpu"
        result.half_precision = False
        result.pin_memory     = False
        result.num_workers    = min(2, profile.cpu_cores_physical)

        eff = ram * laptop

        if eff >= 16:
            result.batch_size  = 8
            result.chunk_size  = 2000
            result.max_results = 500
            result.quality     = "medium"
            result.reason = (
                f"CPU режим, много RAM ({ram:.1f} GB)")

        elif eff >= 8:
            result.batch_size  = 4
            result.chunk_size  = 1500
            result.max_results = 300
            result.quality     = "fast"
            result.reason = (
                f"CPU режим, достаточно RAM ({ram:.1f} GB)")

        else:
            result.batch_size  = 2
            result.chunk_size  = 800
            result.max_results = 200
            result.quality     = "fast"
            result.reason = (
                f"CPU режим, мало RAM ({ram:.1f} GB)")
            warns.append(
                "Рекомендуется GPU для приемлемой скорости")

        warns.append(
            "GPU не обнаружена — используется CPU (значительно медленнее)")

    # Ноутбук
    if profile.is_laptop:
        warns.append(
            "Ноутбук — параметры снижены на 25% для защиты от перегрева")

    # Переопределение качества пользователем
    if override_quality:
        result.quality = override_quality

    # Заполняем локализованные строки качества
    q_labels = _QUALITY_LABELS.get(result.quality, _QUALITY_LABELS["medium"])
    result.quality_ru = q_labels["ru"]
    result.quality_en = q_labels["en"]

    return result


def auto_tune_to_config(lang: str = "ru") -> dict:
    """
    Возвращает dict готовый для config.json и AnalysisController.
    """
    r = auto_tune()

    quality_str = r.quality_ru if lang == "ru" else r.quality_en

    return {
        "batch_size":         r.batch_size,
        "chunk_size":         r.chunk_size,
        "max_unique_results": r.max_results,
        "quality":            quality_str,
        "device":             r.device,
        "half_precision":     r.half_precision,
        "num_workers":        r.num_workers,
        "pin_memory":         r.pin_memory,
        "_reason":            r.reason,
        "_warnings":          r.warnings,
        "_profile": {
            "gpu":            r.profile.gpu_name,
            "vram_total_gb":  round(r.profile.gpu_vram_gb, 1),
            "vram_free_gb":   round(r.profile.gpu_vram_free_gb, 1),
            "ram_total_gb":   round(r.profile.ram_total_gb, 1),
            "ram_avail_gb":   round(r.profile.ram_available_gb, 1),
            "cpu_cores":      r.profile.cpu_cores_physical,
            "cpu_freq_mhz":   round(r.profile.cpu_freq_mhz, 0),
            "is_laptop":      r.profile.is_laptop,
            "half_fp16":      r.half_precision,
            "compute_cap":    r.profile.gpu_compute_cap,
        },
    }


def print_tune_report(lang: str = "ru") -> None:
    """Печатает отчёт автонастройки в консоль."""
    cfg = auto_tune_to_config(lang)
    p   = cfg["_profile"]
    sep = "─" * 50

    print(sep)
    if lang == "ru":
        print("  Автонастройка Parallel Finder")
        print(sep)
        print(f"  GPU    : {p['gpu'] or 'не найдена'}")
        print(f"  VRAM   : {p['vram_free_gb']} / {p['vram_total_gb']} GB")
        print(f"  RAM    : {p['ram_avail_gb']} / {p['ram_total_gb']} GB")
        print(f"  CPU    : {p['cpu_cores']} ядер  {p['cpu_freq_mhz']:.0f} MHz")
        print(f"  FP16   : {'да' if p['half_fp16'] else 'нет'}")
        print(sep)
        print(f"  batch  : {cfg['batch_size']}")
        print(f"  chunk  : {cfg['chunk_size']}")
        print(f"  quality: {cfg['quality']}")
        print(f"  device : {cfg['device']}")
        print(sep)
        print(f"  {cfg['_reason']}")
        for w in cfg["_warnings"]:
            print(f"  ⚠  {w}")
    else:
        print("  Parallel Finder Auto-Tune")
        print(sep)
        print(f"  GPU    : {p['gpu'] or 'not found'}")
        print(f"  VRAM   : {p['vram_free_gb']} / {p['vram_total_gb']} GB")
        print(f"  RAM    : {p['ram_avail_gb']} / {p['ram_total_gb']} GB")
        print(f"  CPU    : {p['cpu_cores']} cores  {p['cpu_freq_mhz']:.0f} MHz")
        print(f"  FP16   : {'yes' if p['half_fp16'] else 'no'}")
        print(sep)
        print(f"  batch  : {cfg['batch_size']}")
        print(f"  chunk  : {cfg['chunk_size']}")
        print(f"  quality: {cfg['quality']}")
        print(f"  device : {cfg['device']}")
        print(sep)
        print(f"  {cfg['_reason']}")
        for w in cfg["_warnings"]:
            print(f"  ⚠  {w}")
    print(sep)