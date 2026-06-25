from __future__ import annotations

import os
import select
import signal
import sys
import termios
import time
import tty
from dataclasses import dataclass
from typing import Callable, Iterable

from .models import ProcessDetailInfo, ProcessInfo


KEY_UP = "up"
KEY_DOWN = "down"
KEY_LEFT = "left"
KEY_RIGHT = "right"
KEY_PAGE_UP = "page_up"
KEY_PAGE_DOWN = "page_down"
KEY_ENTER = "enter"
KEY_ESC = "esc"
KEY_CTRL_C = "ctrl_c"
KEY_BACKSPACE = "backspace"

ESCAPE_KEY_SEQUENCES = (
    ("\x1b[A", KEY_UP),
    ("\x1b[B", KEY_DOWN),
    ("\x1b[C", KEY_RIGHT),
    ("\x1b[D", KEY_LEFT),
    ("\x1b[5~", KEY_PAGE_UP),
    ("\x1b[6~", KEY_PAGE_DOWN),
)

MODE_NORMAL = "normal"
MODE_SORT_MENU = "sort_menu"
MODE_KILL_CONFIRM = "kill_confirm"
MODE_SEARCH = "search"
MODE_FILTER = "filter"
MODE_HELP = "help"
MODE_PROCESS_INFO = "process_info"

KILL_CONFIRM_CANCEL = "cancel"
KILL_CONFIRM_SIGTERM = "sigterm"
KILL_CONFIRM_SIGKILL = "sigkill"
KILL_CONFIRM_OPTIONS = (
    KILL_CONFIRM_CANCEL,
    KILL_CONFIRM_SIGTERM,
    KILL_CONFIRM_SIGKILL,
)
KILL_CONFIRM_LABELS = {
    KILL_CONFIRM_CANCEL: "Cancel",
    KILL_CONFIRM_SIGTERM: "SIGTERM",
    KILL_CONFIRM_SIGKILL: "SIGKILL",
}
KILL_CONFIRM_SIGNALS = {
    KILL_CONFIRM_SIGTERM: signal.SIGTERM,
    KILL_CONFIRM_SIGKILL: signal.SIGKILL,
}

SORT_DEFAULT = "default"
SORT_OPTIONS = (
    "gpu",
    "gpu_memory",
    "gpu_memory_percent",
    "cpu",
    "mem",
    "pid",
    "user",
    "time",
    "command",
)
SORT_LABELS = {
    SORT_DEFAULT: "default",
    "gpu": "GPU",
    "gpu_memory": "GPU-MEM",
    "gpu_memory_percent": "%GPU-MEM",
    "cpu": "%CPU",
    "mem": "%MEM",
    "pid": "PID",
    "user": "USER",
    "time": "TIME",
    "command": "COMMAND",
}
DEFAULT_DESCENDING_SORTS = {
    "gpu_memory",
    "gpu_memory_percent",
    "cpu",
    "mem",
    "time",
}
STATUS_MESSAGE_SECONDS = 3.0
HELP_VISIBLE_ROWS = 22
PROCESS_INFO_VISIBLE_ROWS = 24
HELP_ENTRIES = (
    ("?", "Open help / close help", "normal, help"),
    ("k/j or Up/Down", "Scroll popup one row", "help, info"),
    ("h/l or Left/Right", "Page popup up/down", "help, info"),
    ("j/k or Up/Down", "Move process cursor", "normal"),
    ("PgUp/PgDn", "Move process cursor by page", "normal"),
    ("s", "Open sort menu", "normal"),
    ("h/l or arrows", "Move sort/kill menu selection", "sort, kill"),
    ("Enter", "Apply selected sort or kill option", "sort, kill"),
    ("/", "Search processes", "normal"),
    ("n/N", "Next/previous search match", "normal"),
    ("f", "Filter processes", "normal"),
    ("0-9", "Filter processes by GPU id", "normal"),
    ("z", "Zoom process table", "normal"),
    (",/.", "Pan graph older/newer", "normal"),
    ("r", "Reset graph to live", "normal"),
    ("Esc", "Clear filter or cancel active mode", "normal, menus"),
    ("t", "Toggle process tree", "normal"),
    ("p", "Jump to parent process", "tree"),
    ("h / Left", "Jump to previous sibling", "tree"),
    ("l / Right", "Jump to next sibling", "tree"),
    ("x", "Open kill confirmation", "normal"),
    ("i", "Open selected process details", "normal"),
    ("y", "Send SIGTERM in kill confirmation", "kill"),
    ("q", "Quit or cancel menu", "normal, menus"),
    ("Ctrl-C", "Quit", "all"),
)
ProcessSelectionKey = tuple[int, int | None]


@dataclass(slots=True)
class KeyResult:
    quit: bool = False
    changed: bool = False


@dataclass(slots=True)
class ProcessViewState:
    selected_pid: int | None = None
    selected_process_key: ProcessSelectionKey | None = None
    selected_index: int = 0
    scroll_offset: int = 0
    sort_field: str = SORT_DEFAULT
    sort_desc: bool = True
    mode: str = MODE_NORMAL
    sort_menu_index: int = 0
    kill_confirm_index: int = 0
    kill_confirm_pid: int | None = None
    status_message: str = ""
    status_message_expires_at: float | None = None
    viewport_rows: int = 8
    tree_mode: bool = False
    help_scroll_offset: int = 0
    process_info_scroll_offset: int = 0
    process_info_process: ProcessInfo | None = None
    process_info_detail: ProcessDetailInfo | None = None
    process_info_parent: ProcessInfo | None = None
    process_info_child_count: int = 0
    process_info_render_row_count: int = 0
    search_query: str = ""
    search_input: str = ""
    filter_query: str = ""
    filter_input: str = ""
    gpu_filter_index: int | None = None
    process_zoomed: bool = False
    graph_view_offset_seconds: int = 0

    def sorted_processes(self, processes: Iterable[ProcessInfo]) -> list[ProcessInfo]:
        rows = list(processes)
        return self.sort_process_rows(rows)

    def filtered_processes(self, processes: Iterable[ProcessInfo]) -> list[ProcessInfo]:
        rows = list(processes)
        query = self.filter_query.strip()
        if query:
            rows = [proc for proc in rows if process_matches_search(proc, query)]
        if self.gpu_filter_index is not None:
            rows = [proc for proc in rows if proc.gpu_index == self.gpu_filter_index]
        return rows

    def display_processes(
        self,
        processes: Iterable[ProcessInfo],
        process_ancestors: Iterable[ProcessInfo] | None = None,
    ) -> list[ProcessInfo]:
        if self.tree_mode:
            rows = combine_tree_processes(processes, process_ancestors)
            return self.tree_processes(self.filtered_processes(rows))
        return self.sorted_processes(self.filtered_processes(processes))

    def sort_process_rows(self, rows: list[ProcessInfo]) -> list[ProcessInfo]:
        if self.sort_field == SORT_DEFAULT:
            return rows
        rows.sort(key=lambda proc: process_sort_key(proc, self.sort_field), reverse=self.sort_desc)
        return rows

    def tree_processes(self, rows: Iterable[ProcessInfo]) -> list[ProcessInfo]:
        return flatten_process_tree(rows, self.sort_field, self.sort_desc)

    def sync(
        self,
        processes: list[ProcessInfo],
        viewport_rows: int | None = None,
        adjust_scroll: bool = True,
    ) -> None:
        if viewport_rows is not None:
            self.viewport_rows = max(1, viewport_rows)
        if not processes:
            self.selected_pid = None
            self.selected_process_key = None
            self.selected_index = 0
            self.scroll_offset = 0
            return

        match_index = None
        if self.selected_process_key is not None:
            match_index = find_process_index(processes, self.selected_process_key)
        if match_index is None and self.selected_pid is not None:
            match_index = find_process_pid_index(processes, self.selected_pid)
        if match_index is None:
            match_index = clamp(self.selected_index, 0, len(processes) - 1)
        self.select_index(processes, match_index)

        if adjust_scroll:
            self.ensure_selected_visible(len(processes))

    def visible_processes(self, processes: list[ProcessInfo]) -> list[ProcessInfo]:
        self.sync(processes)
        return processes[self.scroll_offset : self.scroll_offset + self.viewport_rows]

    def selected_process(self, processes: list[ProcessInfo]) -> ProcessInfo | None:
        self.sync(processes)
        if not processes:
            return None
        return processes[self.selected_index]

    def selected_synced_process(self, processes: list[ProcessInfo]) -> ProcessInfo | None:
        if not processes:
            return None
        return processes[self.selected_index]

    def selected_visible_index(self) -> int | None:
        if self.selected_pid is None:
            return None
        index = self.selected_index - self.scroll_offset
        if 0 <= index < self.viewport_rows:
            return index
        return None

    def handle_key(
        self,
        key: str,
        processes: list[ProcessInfo],
        kill_func: Callable[[int, signal.Signals], None] = None,
        processes_synced: bool = False,
        gpu_indices: Iterable[int] | None = None,
    ) -> KeyResult:
        kill_func = kill_func or kill_process
        source_processes = processes
        if not processes_synced:
            processes = self.display_processes(source_processes)
            self.sync(processes)

        if key == KEY_CTRL_C:
            return KeyResult(quit=True, changed=True)

        if self.mode == MODE_KILL_CONFIRM:
            return self.handle_kill_confirm_key(key, processes, kill_func, processes_synced=True)

        if self.mode == MODE_SORT_MENU:
            return self.handle_sort_menu_key(key)

        if self.mode == MODE_SEARCH:
            return self.handle_search_key(key, processes, processes_synced=True)

        if self.mode == MODE_FILTER:
            result = self.handle_filter_key(key)
            if result.changed and not processes_synced:
                self.sync(self.display_processes(source_processes))
            return result

        if self.mode == MODE_HELP:
            return self.handle_help_key(key)

        if self.mode == MODE_PROCESS_INFO:
            return self.handle_process_info_key(key)

        if key == "z":
            self.process_zoomed = not self.process_zoomed
            self.clear_status_message()
            return KeyResult(changed=True)

        if is_gpu_filter_key(key):
            if self.apply_gpu_filter_key(key, source_processes, gpu_indices):
                if not processes_synced:
                    self.sync(self.display_processes(source_processes))
                return KeyResult(changed=True)
            return KeyResult()

        if key == KEY_ESC and self.process_zoomed:
            self.process_zoomed = False
            self.clear_status_message()
            return KeyResult(changed=True)

        if key == KEY_ESC and self.has_filter():
            self.clear_filter()
            if not processes_synced:
                self.sync(self.display_processes(source_processes))
            return KeyResult(changed=True)

        if key == "q":
            return KeyResult(quit=True, changed=True)
        if key == "t":
            self.tree_mode = not self.tree_mode
            self.clear_status_message()
            return KeyResult(changed=True)
        if key == "p" and self.tree_mode:
            self.select_parent_process(processes)
            return KeyResult(changed=True)
        if key == "?":
            self.mode = MODE_HELP
            self.help_scroll_offset = 0
            self.clear_status_message()
            return KeyResult(changed=True)
        if key in ("h", KEY_LEFT) and self.tree_mode:
            self.select_sibling_process(processes, direction=-1)
            return KeyResult(changed=True)
        if key in ("l", KEY_RIGHT) and self.tree_mode:
            self.select_sibling_process(processes, direction=1)
            return KeyResult(changed=True)
        if key in ("j", KEY_DOWN):
            self.move_selection(processes, 1)
            return KeyResult(changed=True)
        if key in ("k", KEY_UP):
            self.move_selection(processes, -1)
            return KeyResult(changed=True)
        if key == KEY_PAGE_DOWN:
            self.move_selection(processes, self.viewport_rows)
            return KeyResult(changed=True)
        if key == KEY_PAGE_UP:
            self.move_selection(processes, -self.viewport_rows)
            return KeyResult(changed=True)
        if key == "s":
            self.mode = MODE_SORT_MENU
            self.sort_menu_index = current_sort_menu_index(self.sort_field)
            self.clear_status_message()
            return KeyResult(changed=True)
        if key == "/":
            self.mode = MODE_SEARCH
            self.search_input = ""
            self.clear_status_message()
            return KeyResult(changed=True)
        if key == "f":
            self.mode = MODE_FILTER
            self.filter_input = self.filter_query
            self.clear_status_message()
            return KeyResult(changed=True)
        if key == "n":
            self.search_next(processes, direction=1, processes_synced=True)
            return KeyResult(changed=True)
        if key == "N":
            self.search_next(processes, direction=-1, processes_synced=True)
            return KeyResult(changed=True)
        if key == "x":
            selected = self.selected_synced_process(processes)
            if selected is None:
                self.kill_confirm_pid = None
                self.set_status_message("No process selected")
            else:
                self.open_kill_confirm(selected)
            return KeyResult(changed=True)
        return KeyResult()

    def open_kill_confirm(self, process: ProcessInfo) -> None:
        self.mode = MODE_KILL_CONFIRM
        self.kill_confirm_index = 0
        self.kill_confirm_pid = process.pid
        self.clear_status_message()

    def close_kill_confirm(self) -> None:
        self.mode = MODE_NORMAL
        self.kill_confirm_pid = None
        self.clear_status_message()

    def open_process_info(
        self,
        process: ProcessInfo,
        detail: ProcessDetailInfo,
        parent: ProcessInfo | None = None,
        child_count: int = 0,
    ) -> None:
        self.process_info_process = process
        self.process_info_detail = detail
        self.process_info_parent = parent
        self.process_info_child_count = max(0, child_count)
        self.process_info_scroll_offset = 0
        self.process_info_render_row_count = 0
        self.mode = MODE_PROCESS_INFO
        self.clear_status_message()

    def handle_process_info_key(self, key: str) -> KeyResult:
        if key in (KEY_ESC, "i"):
            self.mode = MODE_NORMAL
            self.clear_status_message()
            return KeyResult(changed=True)
        if key in ("k", KEY_UP):
            self.process_info_scroll_offset = max(0, self.process_info_scroll_offset - 1)
            return KeyResult(changed=True)
        if key in ("j", KEY_DOWN):
            self.process_info_scroll_offset = min(
                max_process_info_scroll_offset(self),
                self.process_info_scroll_offset + 1,
            )
            return KeyResult(changed=True)
        if key in ("h", KEY_LEFT):
            self.process_info_scroll_offset = max(0, self.process_info_scroll_offset - PROCESS_INFO_VISIBLE_ROWS)
            return KeyResult(changed=True)
        if key in ("l", KEY_RIGHT):
            self.process_info_scroll_offset = min(
                max_process_info_scroll_offset(self),
                self.process_info_scroll_offset + PROCESS_INFO_VISIBLE_ROWS,
            )
            return KeyResult(changed=True)
        return KeyResult()

    def handle_search_key(
        self,
        key: str,
        processes: list[ProcessInfo],
        processes_synced: bool = False,
    ) -> KeyResult:
        if key == KEY_ESC:
            self.mode = MODE_NORMAL
            self.search_input = ""
            self.clear_status_message()
            return KeyResult(changed=True)
        if key == KEY_ENTER:
            query = self.search_input.strip()
            self.mode = MODE_NORMAL
            self.search_input = ""
            self.search_query = query
            self.search_next(processes, direction=1, processes_synced=processes_synced)
            return KeyResult(changed=True)
        if key == KEY_BACKSPACE:
            self.search_input = self.search_input[:-1]
            return KeyResult(changed=True)
        if is_printable_key(key):
            self.search_input += key
            return KeyResult(changed=True)
        return KeyResult()

    def handle_filter_key(self, key: str) -> KeyResult:
        if key == KEY_ESC:
            self.mode = MODE_NORMAL
            self.clear_filter()
            return KeyResult(changed=True)
        if key == KEY_ENTER:
            self.mode = MODE_NORMAL
            self.filter_input = self.filter_query
            self.clear_status_message()
            return KeyResult(changed=True)
        if key == KEY_BACKSPACE:
            self.filter_input = self.filter_input[:-1]
            self.apply_filter_input()
            return KeyResult(changed=True)
        if is_printable_key(key):
            self.filter_input += key
            self.apply_filter_input()
            return KeyResult(changed=True)
        return KeyResult()

    def handle_help_key(self, key: str) -> KeyResult:
        if key in (KEY_ESC, "?"):
            self.mode = MODE_NORMAL
            self.clear_status_message()
            return KeyResult(changed=True)
        if key in ("k", KEY_UP):
            self.help_scroll_offset = max(0, self.help_scroll_offset - 1)
            return KeyResult(changed=True)
        if key in ("j", KEY_DOWN):
            self.help_scroll_offset = min(max_help_scroll_offset(), self.help_scroll_offset + 1)
            return KeyResult(changed=True)
        if key in ("h", KEY_LEFT):
            self.help_scroll_offset = max(0, self.help_scroll_offset - HELP_VISIBLE_ROWS)
            return KeyResult(changed=True)
        if key in ("l", KEY_RIGHT):
            self.help_scroll_offset = min(max_help_scroll_offset(), self.help_scroll_offset + HELP_VISIBLE_ROWS)
            return KeyResult(changed=True)
        return KeyResult()

    def apply_filter_input(self) -> None:
        self.filter_query = self.filter_input.strip()

    def apply_gpu_filter_key(
        self,
        key: str,
        processes: Iterable[ProcessInfo],
        gpu_indices: Iterable[int] | None = None,
    ) -> bool:
        gpu_index = int(key)
        if gpu_index not in available_gpu_indices(processes, gpu_indices):
            return False
        self.gpu_filter_index = gpu_index
        self.clear_status_message()
        return True

    def has_filter(self) -> bool:
        return bool(self.filter_query.strip()) or self.gpu_filter_index is not None

    def clear_filter(self) -> None:
        self.filter_input = ""
        self.filter_query = ""
        self.gpu_filter_index = None
        self.clear_status_message()

    def handle_kill_confirm_key(
        self,
        key: str,
        processes: list[ProcessInfo],
        kill_func: Callable[[int, signal.Signals], None],
        processes_synced: bool = False,
    ) -> KeyResult:
        if key in ("h", "k", KEY_UP, KEY_LEFT):
            self.kill_confirm_index = max(0, self.kill_confirm_index - 1)
            return KeyResult(changed=True)
        if key in ("j", "l", KEY_DOWN, KEY_RIGHT):
            self.kill_confirm_index = min(len(KILL_CONFIRM_OPTIONS) - 1, self.kill_confirm_index + 1)
            return KeyResult(changed=True)
        if key in (KEY_ESC, "q", "n", "N"):
            self.close_kill_confirm()
            return KeyResult(changed=True)
        if key in ("y", "Y"):
            self.kill_confirm_index = KILL_CONFIRM_OPTIONS.index(KILL_CONFIRM_SIGTERM)
        elif key == KEY_ENTER:
            option = KILL_CONFIRM_OPTIONS[self.kill_confirm_index]
            if option == KILL_CONFIRM_CANCEL:
                self.close_kill_confirm()
                return KeyResult(changed=True)
        else:
            return KeyResult()

        option = KILL_CONFIRM_OPTIONS[self.kill_confirm_index]
        kill_signal = KILL_CONFIRM_SIGNALS[option]
        target_pid = self.kill_confirm_pid
        self.mode = MODE_NORMAL
        self.kill_confirm_pid = None
        if target_pid is None:
            self.set_status_message("No process selected")
            return KeyResult(changed=True)
        if not any(proc.pid == target_pid for proc in processes):
            self.set_status_message(f"PID {target_pid} is no longer running")
            return KeyResult(changed=True)
        try:
            kill_func(target_pid, kill_signal)
        except ProcessLookupError:
            self.set_status_message(f"PID {target_pid} is no longer running")
        except PermissionError:
            self.set_status_message(f"Permission denied killing PID {target_pid}")
        except OSError as exc:
            self.set_status_message(f"Failed to kill PID {target_pid}: {exc}")
        else:
            self.set_status_message(f"Sent {kill_signal.name} to PID {target_pid}")
        return KeyResult(changed=True)

    def handle_sort_menu_key(self, key: str) -> KeyResult:
        if key in ("h", "k", KEY_UP, KEY_LEFT):
            self.sort_menu_index = max(0, self.sort_menu_index - 1)
            return KeyResult(changed=True)
        if key in ("j", "l", KEY_DOWN, KEY_RIGHT):
            self.sort_menu_index = min(len(SORT_OPTIONS) - 1, self.sort_menu_index + 1)
            return KeyResult(changed=True)
        if key in (KEY_ESC, "q"):
            self.mode = MODE_NORMAL
            self.clear_status_message()
            return KeyResult(changed=True)
        if key == KEY_ENTER:
            field = SORT_OPTIONS[self.sort_menu_index]
            if self.sort_field == field:
                self.sort_desc = not self.sort_desc
            else:
                self.sort_field = field
                self.sort_desc = field in DEFAULT_DESCENDING_SORTS
            self.mode = MODE_NORMAL
            self.clear_status_message()
            return KeyResult(changed=True)
        return KeyResult()

    def move_selection(self, processes: list[ProcessInfo], delta: int) -> None:
        if not processes:
            self.sync(processes)
            return
        self.select_index(processes, clamp(self.selected_index + delta, 0, len(processes) - 1))
        if abs(delta) == 1:
            return
        self.ensure_selected_visible(len(processes))

    def select_parent_process(self, processes: list[ProcessInfo]) -> None:
        selected = self.selected_synced_process(processes)
        if selected is None or selected.ppid is None:
            self.set_status_message("No visible parent process")
            return
        parent_index = find_process_pid_index(processes, selected.ppid)
        if parent_index is None:
            self.set_status_message("No visible parent process")
            return
        self.select_index(processes, parent_index)
        self.ensure_selected_visible(len(processes))
        self.clear_status_message()

    def select_sibling_process(self, processes: list[ProcessInfo], direction: int) -> None:
        if not processes:
            self.sync(processes)
            return
        parent_keys = visible_tree_parent_keys(processes)
        selected_parent_key = parent_keys[self.selected_index]
        if direction >= 0:
            sibling_indices = range(self.selected_index + 1, len(processes))
        else:
            sibling_indices = range(self.selected_index - 1, -1, -1)
        for index in sibling_indices:
            if parent_keys[index] != selected_parent_key:
                continue
            self.select_index(processes, index)
            self.ensure_selected_visible(len(processes))
            self.clear_status_message()
            return
        self.set_status_message("No visible sibling process")

    def search_next(self, processes: list[ProcessInfo], direction: int, processes_synced: bool = False) -> bool:
        query = self.search_query.strip()
        if not query:
            self.set_status_message("No search query")
            return False
        match_index = self.search_match_index(processes, query, direction, processes_synced=processes_synced)
        if match_index is None:
            self.set_status_message(f"No matches for: {query}")
            return False
        self.select_index(processes, match_index)
        self.ensure_selected_visible(len(processes))
        self.set_status_message(f"Search: {query}")
        return True

    def search_match_index(
        self,
        processes: list[ProcessInfo],
        query: str,
        direction: int,
        processes_synced: bool = False,
    ) -> int | None:
        if not processes:
            return None
        if not processes_synced:
            self.sync(processes)
        step = 1 if direction >= 0 else -1
        start = self.selected_index
        for offset in range(1, len(processes) + 1):
            index = (start + offset * step) % len(processes)
            if process_matches_search(processes[index], query):
                return index
        return None

    def ensure_selected_visible(self, process_count: int) -> None:
        max_scroll = max(0, process_count - self.viewport_rows)
        if self.selected_index < self.scroll_offset:
            self.scroll_offset = self.selected_index
        elif self.selected_index >= self.scroll_offset + self.viewport_rows:
            self.scroll_offset = self.selected_index - self.viewport_rows + 1
        self.scroll_offset = clamp(self.scroll_offset, 0, max_scroll)

    def select_index(self, processes: list[ProcessInfo], index: int) -> None:
        self.selected_index = index
        proc = processes[index]
        self.selected_pid = proc.pid
        self.selected_process_key = process_selection_key(proc)

    def sort_direction_label(self) -> str:
        if self.sort_field == SORT_DEFAULT:
            return ""
        return "desc" if self.sort_desc else "asc"

    def sort_label(self) -> str:
        label = SORT_LABELS.get(self.sort_field, self.sort_field)
        direction = self.sort_direction_label()
        return f"{label} {direction}".strip()

    def title(self, process_count: int) -> str:
        return self.process_title(process_count)

    def process_title(self, process_count: int) -> str:
        label = "Process Tree" if self.tree_mode else "Processes"
        if process_count <= 0:
            return f"{label}  0/0"
        return f"{label}  {self.selected_index + 1}/{process_count}"

    def caption(self) -> str:
        parts = []
        if self.status_message:
            parts.append(self.status_message)
        if self.gpu_filter_index is not None:
            parts.append(f"GPU: {self.gpu_filter_index}")
        if self.filter_query.strip() and self.mode != MODE_FILTER:
            parts.append(f"Filter: {self.filter_query.strip()}")
        return "   ".join(parts)

    def set_status_message(self, message: str, now: float | None = None) -> None:
        self.status_message = message
        self.status_message_expires_at = (time.monotonic() if now is None else now) + STATUS_MESSAGE_SECONDS

    def clear_status_message(self) -> None:
        self.status_message = ""
        self.status_message_expires_at = None

    def expire_status_message(self, now: float | None = None) -> bool:
        if not self.status_message or self.status_message_expires_at is None:
            return False
        current_time = time.monotonic() if now is None else now
        if current_time < self.status_message_expires_at:
            return False
        self.clear_status_message()
        return True


class TerminalKeyboard:
    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stdin
        self.fd: int | None = None
        self.original_attrs = None
        self.enabled = False
        self.pending_input = ""

    def __enter__(self) -> TerminalKeyboard:
        if hasattr(self.stream, "isatty") and self.stream.isatty():
            self.fd = self.stream.fileno()
            self.original_attrs = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self.enabled = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled and self.fd is not None and self.original_attrs is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original_attrs)

    def read_keys(self, timeout: float = 0.0) -> list[str]:
        if not self.enabled or self.fd is None:
            return []
        readable, _, _ = select.select([self.fd], [], [], max(0.0, timeout))
        if not readable:
            if self.pending_input:
                keys, self.pending_input = parse_key_input(self.pending_input, flush_incomplete=True)
                return keys
            return []
        data = os.read(self.fd, 64)
        keys, self.pending_input = parse_key_input(self.pending_input + data.decode(errors="ignore"))
        return keys


def parse_keys(data: bytes | str) -> list[str]:
    keys, _pending = parse_key_input(data, flush_incomplete=True)
    return keys


def parse_key_input(data: bytes | str, flush_incomplete: bool = False) -> tuple[list[str], str]:
    text = data.decode(errors="ignore") if isinstance(data, bytes) else data
    keys: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\x03":
            keys.append(KEY_CTRL_C)
            index += 1
            continue
        if char in ("\x7f", "\b"):
            keys.append(KEY_BACKSPACE)
            index += 1
            continue
        if char in ("\r", "\n"):
            keys.append(KEY_ENTER)
            index += 1
            continue
        if char != "\x1b":
            keys.append(char)
            index += 1
            continue

        sequence = text[index:]
        matched = False
        for raw, key in ESCAPE_KEY_SEQUENCES:
            if sequence.startswith(raw):
                keys.append(key)
                index += len(raw)
                matched = True
                break
        if matched:
            continue
        if not flush_incomplete and any(raw.startswith(sequence) for raw, _key in ESCAPE_KEY_SEQUENCES):
            return keys, sequence
        keys.append(KEY_ESC)
        index += 1
    return keys, ""


def is_printable_key(key: str) -> bool:
    return len(key) == 1 and key.isprintable()


def is_gpu_filter_key(key: str) -> bool:
    return len(key) == 1 and "0" <= key <= "9"


def available_gpu_indices(
    processes: Iterable[ProcessInfo],
    gpu_indices: Iterable[int] | None = None,
) -> set[int]:
    if gpu_indices is not None:
        return {index for index in gpu_indices if 0 <= index <= 9}
    return {proc.gpu_index for proc in processes if proc.gpu_index is not None and 0 <= proc.gpu_index <= 9}


def process_matches_search(proc: ProcessInfo, query: str) -> bool:
    needle = query.lower()
    fields = (
        proc.args,
        proc.command,
        proc.name,
        proc.user,
        str(proc.pid),
    )
    return needle in " ".join(str(field or "") for field in fields).lower()


def process_selection_key(proc: ProcessInfo) -> ProcessSelectionKey:
    return (proc.pid, proc.gpu_index)


def find_process_index(processes: list[ProcessInfo], key: ProcessSelectionKey) -> int | None:
    for index, proc in enumerate(processes):
        if process_selection_key(proc) == key:
            return index
    return None


def find_process_pid_index(processes: list[ProcessInfo], pid: int) -> int | None:
    for index, proc in enumerate(processes):
        if proc.pid == pid:
            return index
    return None


def combine_tree_processes(
    processes: Iterable[ProcessInfo],
    process_ancestors: Iterable[ProcessInfo] | None = None,
) -> list[ProcessInfo]:
    rows: list[ProcessInfo] = []
    seen_keys: set[ProcessSelectionKey] = set()
    process_pids: set[int] = set()
    for proc in processes:
        key = process_selection_key(proc)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        process_pids.add(proc.pid)
        rows.append(proc)

    for proc in process_ancestors or ():
        if proc.pid in process_pids:
            continue
        key = process_selection_key(proc)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows.append(proc)
    return rows


def flatten_process_tree(
    processes: Iterable[ProcessInfo],
    sort_field: str = SORT_DEFAULT,
    sort_desc: bool = True,
) -> list[ProcessInfo]:
    rows = list(processes)
    if not rows:
        return []

    rows_by_key: dict[ProcessSelectionKey, ProcessInfo] = {}
    key_by_pid: dict[int, ProcessSelectionKey] = {}
    ordered_keys: list[ProcessSelectionKey] = []
    for proc in rows:
        key = process_selection_key(proc)
        if key in rows_by_key:
            continue
        rows_by_key[key] = proc
        ordered_keys.append(key)
        key_by_pid.setdefault(proc.pid, key)

    root_keys: list[ProcessSelectionKey] = []
    children_by_parent: dict[ProcessSelectionKey, list[ProcessSelectionKey]] = {}
    for key in ordered_keys:
        proc = rows_by_key[key]
        parent_key = key_by_pid.get(proc.ppid or -1)
        if parent_key is None or parent_key == key:
            root_keys.append(key)
            continue
        children_by_parent.setdefault(parent_key, []).append(key)

    flattened: list[ProcessInfo] = []
    visited: set[ProcessSelectionKey] = set()

    def append_branch(key: ProcessSelectionKey) -> None:
        if key in visited:
            return
        visited.add(key)
        flattened.append(rows_by_key[key])
        for child_key in sorted_tree_keys(children_by_parent.get(key, []), rows_by_key, sort_field, sort_desc):
            append_branch(child_key)

    for key in sorted_tree_keys(root_keys, rows_by_key, sort_field, sort_desc):
        append_branch(key)
    for key in ordered_keys:
        append_branch(key)

    return flattened


def sorted_tree_keys(
    keys: list[ProcessSelectionKey],
    rows_by_key: dict[ProcessSelectionKey, ProcessInfo],
    sort_field: str,
    sort_desc: bool,
) -> list[ProcessSelectionKey]:
    if sort_field == SORT_DEFAULT:
        return sorted(keys, key=lambda key: default_tree_sort_key(rows_by_key[key]))
    return sorted(keys, key=lambda key: process_sort_key(rows_by_key[key], sort_field), reverse=sort_desc)


def default_tree_sort_key(proc: ProcessInfo) -> tuple[int, bool, int]:
    gpu_index = proc.gpu_index if proc.gpu_index is not None else 9999
    return (proc.pid, proc.gpu_index is None, gpu_index)


def visible_tree_parent_keys(processes: list[ProcessInfo]) -> list[ProcessSelectionKey | None]:
    key_by_pid: dict[int, ProcessSelectionKey] = {}
    for proc in processes:
        key_by_pid.setdefault(proc.pid, process_selection_key(proc))

    parent_keys: list[ProcessSelectionKey | None] = []
    for proc in processes:
        parent_key = key_by_pid.get(proc.ppid or -1)
        if parent_key == process_selection_key(proc):
            parent_key = None
        parent_keys.append(parent_key)
    return parent_keys


def process_sort_key(proc: ProcessInfo, field: str):
    if field == "gpu":
        return (proc.gpu_index is None, proc.gpu_index if proc.gpu_index is not None else 9999, proc.pid)
    if field == "gpu_memory":
        return (proc.gpu_memory_bytes, proc.pid)
    if field == "gpu_memory_percent":
        return (proc.gpu_memory_percent, proc.pid)
    if field == "cpu":
        return (proc.cpu_percent if proc.cpu_percent is not None else -1.0, proc.pid)
    if field == "mem":
        return (proc.host_mem_percent if proc.host_mem_percent is not None else -1.0, proc.pid)
    if field == "pid":
        return (proc.pid,)
    if field == "user":
        return ((proc.user or "").lower(), proc.pid)
    if field == "time":
        return (elapsed_seconds(proc.elapsed), proc.pid)
    if field == "command":
        return ((proc.args or proc.command or proc.name or "").lower(), proc.pid)
    return (proc.pid,)


def elapsed_seconds(value: str) -> int:
    text = str(value or "").strip()
    if not text or text == "-":
        return 0
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        days = int(day_text) if day_text.isdigit() else 0
    parts = text.split(":")
    if not all(part.isdigit() for part in parts):
        return days * 86400
    numbers = [int(part) for part in parts]
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
    elif len(numbers) == 2:
        hours = 0
        minutes, seconds = numbers
    elif len(numbers) == 1:
        hours = 0
        minutes = 0
        seconds = numbers[0]
    else:
        return days * 86400
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def current_sort_menu_index(sort_field: str) -> int:
    try:
        return SORT_OPTIONS.index(sort_field)
    except ValueError:
        return 0


def max_help_scroll_offset() -> int:
    return max(0, len(HELP_ENTRIES) - HELP_VISIBLE_ROWS)


def max_process_info_scroll_offset(process_state: ProcessViewState) -> int:
    row_count = process_state.process_info_render_row_count or process_info_row_count(process_state)
    return max(0, row_count - PROCESS_INFO_VISIBLE_ROWS)


def process_info_row_count(process_state: ProcessViewState) -> int:
    if process_state.process_info_process is None:
        return 1
    detail = process_state.process_info_detail
    count = 15
    if detail is None:
        return count + 1
    count += 10
    if detail.error:
        count += 1
    return count


def kill_process(pid: int, kill_signal: signal.Signals = signal.SIGTERM) -> None:
    os.kill(pid, kill_signal)


def clamp(value: int, low: int, high: int) -> int:
    return min(high, max(low, value))
