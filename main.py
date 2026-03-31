#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

config_path = os.path.join(os.path.dirname(__file__), "pythonlibs_path.txt")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        libs_path = f.read().strip()
        if libs_path:
            sys.path.insert(0, libs_path)

from ui.main_window import ParallelFinderApp

if __name__ == "__main__":
    app = ParallelFinderApp()
    app.run()