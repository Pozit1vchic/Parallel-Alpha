from __future__ import annotations

"""Application launcher for Parallel Finder.

This module bootstraps the desktop application, configures logging,
installs global exception hooks, resolves the current main window class,
and starts the Qt event loop. The implementation is intentionally tolerant
to small project layout differences so it can work with both rewritten and
legacy package structures.
"""

import argparse
import importlib
import logging
import os
import sys
import threading
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType
from typing import Any

# Keep project-local imports resolvable when launching from an arbitrary cwd.
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from PySide6.QtCore import QTimer, QtMsgType, qInstallMessageHandler
    from PySide6.QtWidgets import QApplication, QMessageBox
except Exception as exc:  # pragma: no cover - fatal startup guard
    print(f"PySide6 import failed: {exc}", file=sys.stderr)
    raise

try:
    from utils.constants import LOG_DIR as _PROJECT_LOG_DIR
except Exception:
    _PROJECT_LOG_DIR = str(ROOT_DIR / "logs")


APP_NAME = "Parallel Finder"
APP_DISPLAY_NAME = "Parallel Finder"
APP_ORGANIZATION = "Parallel Finder"
DEFAULT_LOG_DIR = Path(_PROJECT_LOG_DIR)


class LauncherError(RuntimeError):
    """Fatal launcher-level error."""


def _parse_args() -> argparse.Namespace:
    """Parse launcher command line arguments."""
    parser = argparse.ArgumentParser(description="Parallel Finder launcher")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Explicit root log level",
    )
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_LOG_DIR),
        help="Directory where rotating application logs will be written",
    )
    parser.add_argument(
        "--project",
        default="",
        help="Optional project file to open after the main window is shown",
    )
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="Disable CUDA via environment variables before app startup",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear preview/project cache on startup if a ProjectManager is available",
    )
    return parser.parse_args()


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean flag from environment variables."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def setup_environment(*, debug: bool = False, safe_mode: bool = False) -> None:
    """Apply launcher-level environment configuration.

    Parameters
    ----------
    debug:
        Enables a small set of debug-oriented environment flags.
    safe_mode:
        Disables CUDA device exposure to force CPU-safe execution paths.
    """
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    if debug:
        os.environ.setdefault("PF_DEBUG", "1")

    if safe_mode:
        os.environ.setdefault("PF_SAFE_MODE", "1")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def setup_logging(*, debug: bool = False, log_level: str = "INFO", log_dir: str | Path = DEFAULT_LOG_DIR) -> Path:
    """Configure rotating file and console logging.

    Returns
    -------
    Path
        The full path to the active log file.
    """
    level_name = "DEBUG" if debug else log_level.upper()
    level = getattr(logging, level_name, logging.INFO)

    log_root = Path(log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    log_file = log_root / "parallel_finder.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    logging.captureWarnings(True)

    return log_file


def _show_fatal_error(message: str, *, title: str = f"{APP_NAME} - Startup Error") -> None:
    """Display a fatal message even when no QApplication exists yet."""
    existing_app = QApplication.instance()
    if existing_app is not None:
        QMessageBox.critical(None, title, message)
        return

    temp_app = QApplication([])
    try:
        QMessageBox.critical(None, title, message)
    finally:
        temp_app.quit()


def _install_exception_hooks() -> None:
    """Install process-wide exception hooks for main and worker threads."""

    def _handle_exception(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
    ) -> None:
        details = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logging.critical("Unhandled exception:\n%s", details)
        _show_fatal_error(f"The application encountered a fatal error:\n\n{details}")

    sys.excepthook = _handle_exception

    if hasattr(threading, "excepthook"):
        def _thread_exception_hook(args: threading.ExceptHookArgs) -> None:
            _handle_exception(args.exc_type, args.exc_value, args.exc_traceback)

        threading.excepthook = _thread_exception_hook


def _install_qt_message_handler() -> None:
    """Redirect Qt internal messages to Python logging."""

    def _qt_handler(message_type: QtMsgType, context: Any, message: str) -> None:
        category = getattr(context, "category", "qt") if context is not None else "qt"
        if message_type == QtMsgType.QtDebugMsg:
            logging.debug("[Qt:%s] %s", category, message)
        elif message_type == QtMsgType.QtInfoMsg:
            logging.info("[Qt:%s] %s", category, message)
        elif message_type == QtMsgType.QtWarningMsg:
            logging.warning("[Qt:%s] %s", category, message)
        elif message_type == QtMsgType.QtCriticalMsg:
            logging.error("[Qt:%s] %s", category, message)
        else:
            logging.critical("[Qt:%s] %s", category, message)

    qInstallMessageHandler(_qt_handler)


def _resolve_main_window_class() -> type[Any]:
    """Resolve the main window class across rewritten and legacy layouts."""
    errors: list[str] = []
    module_candidates = (
        "ui.main_window",
        "ui",
        "main_window",
    )
    class_candidates = (
        "ParallelFinderMainWindow",
        "MainWindow",
        "ParallelFinderUI",
    )

    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{module_name}: {exc}")
            continue

        for class_name in class_candidates:
            candidate = getattr(module, class_name, None)
            if candidate is not None:
                return candidate

    raise LauncherError("Could not resolve main window class. " + " | ".join(errors))


def _resolve_project_manager_class() -> type[Any] | None:
    """Resolve ProjectManager if available in either current or legacy layout."""
    for module_name in ("core.project", "project"):
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001
            continue
        manager_cls = getattr(module, "ProjectManager", None)
        if manager_cls is not None:
            return manager_cls
    return None


def _clear_cache_if_requested() -> None:
    """Clear cache through ProjectManager if it can be resolved."""
    manager_cls = _resolve_project_manager_class()
    if manager_cls is None:
        logging.warning("ProjectManager is unavailable; startup cache clear skipped")
        return

    try:
        manager = manager_cls()
        clear_cache = getattr(manager, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()
            logging.info("Project cache cleared on startup")
        else:
            logging.warning("Resolved ProjectManager has no clear_cache() method")
    except Exception:  # noqa: BLE001
        logging.exception("Failed to clear cache on startup")


def _open_project_if_requested(window: Any, project_path: str | Path | None) -> None:
    """Open a project file if the window exposes a compatible method."""
    if not project_path:
        return

    project = str(project_path)
    if not Path(project).exists():
        logging.warning("Requested project file does not exist: %s", project)
        return

    loader_names = (
        "open_project",
        "load_project",
        "load_project_file",
        "open_project_file",
    )
    for loader_name in loader_names:
        loader = getattr(window, loader_name, None)
        if callable(loader):
            try:
                loader(project)
                logging.info("Opened project via %s: %s", loader_name, project)
                return
            except Exception:  # noqa: BLE001
                logging.exception("Project open failed via %s", loader_name)
                return

    logging.warning("Main window does not expose a project open method; skipped %s", project)


def _call_cleanup(window: Any) -> None:
    """Call cleanup hooks on shutdown if the window implements them."""
    for name in ("cleanup", "shutdown", "dispose"):
        method = getattr(window, name, None)
        if callable(method):
            try:
                method()
                logging.info("Executed window cleanup via %s()", name)
            except Exception:  # noqa: BLE001
                logging.exception("Cleanup via %s() failed", name)
            return


def main() -> int:
    """Application entry point."""
    args = _parse_args()
    debug_mode = args.debug or _env_flag("PF_DEBUG", default=False)
    safe_mode = args.safe_mode or _env_flag("PF_SAFE_MODE", default=False)

    setup_environment(debug=debug_mode, safe_mode=safe_mode)
    log_file = setup_logging(debug=debug_mode, log_level=args.log_level, log_dir=args.log_dir)
    _install_exception_hooks()
    _install_qt_message_handler()

    logging.info("Starting %s", APP_NAME)
    logging.info("Python: %s", sys.version.replace("\n", " "))
    logging.info("Root: %s", ROOT_DIR)
    logging.info("Log file: %s", log_file)
    logging.info("Debug mode: %s", debug_mode)
    logging.info("Safe mode: %s", safe_mode)

    if args.clear_cache:
        _clear_cache_if_requested()

    app: QApplication | None = None
    window: Any = None
    try:
        app = QApplication(sys.argv)
        app.setApplicationName(APP_NAME)
        app.setApplicationDisplayName(APP_DISPLAY_NAME)
        app.setOrganizationName(APP_ORGANIZATION)

        main_window_cls = _resolve_main_window_class()
        window = main_window_cls()

        def _on_about_to_quit() -> None:
            if window is not None:
                _call_cleanup(window)

        app.aboutToQuit.connect(_on_about_to_quit)

        window.show()
        if args.project:
            QTimer.singleShot(0, lambda: _open_project_if_requested(window, args.project))

        return int(app.exec())
    except Exception:  # noqa: BLE001
        details = traceback.format_exc()
        logging.critical("Startup failed:\n%s", details)
        if app is not None:
            QMessageBox.critical(None, f"{APP_NAME} - Startup Error", details)
        else:
            _show_fatal_error(details)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
