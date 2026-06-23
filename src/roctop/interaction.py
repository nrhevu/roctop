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

from .models import ProcessInfo


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

MODE_NORMAL = "normal"
MODE_SORT_MENU = "sort_menu"
MODE_KILL_CONFIRM = "kill_confirm"
MODE_SEARCH = "search"

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


@dataclass(slots=True)
class KeyResult:
    quit: bool = False
    changed: bool = False


@dataclass(slots=True)
class ProcessViewState:
    selected_pid: int | None = None
    selected_index: int = 0
    scroll_offset: int = 0
    sort_field: str = SORT_DEFAULT
    sort_desc: bool = True
    mode: str = MODE_NORMAL
    sort_menu_index: int = 0
    kill_confirm_index: int = 0
    status_message: str = ""
    status_message_expires_at: float | None = None
    viewport_rows: int = 8
    search_query: str = ""
    search_input: str = ""

    def sorted_processes(self, processes: Iterable[ProcessInfo]) -> list[ProcessInfo]:
        rows = list(processes)
        if self.sort_field == SORT_DEFAULT:
            return rows
        rows.sort(key=lambda proc: process_sort_key(proc, self.sort_field), reverse=self.sort_desc)
        return rows

    def sync(self, processes: list[ProcessInfo], viewport_rows: int | None = None) -> None:
        if viewport_rows is not None:
            self.viewport_rows = max(1, viewport_rows)
        if not processes:
            self.selected_pid = None
            self.selected_index = 0
            self.scroll_offset = 0
            return

        if self.selected_pid is not None:
            for index, proc in enumerate(processes):
                if proc.pid == self.selected_pid:
                    self.selected_index = index
                    break
            else:
                self.selected_index = clamp(self.selected_index, 0, len(processes) - 1)
                self.selected_pid = processes[self.selected_index].pid
        else:
            self.selected_index = clamp(self.selected_index, 0, len(processes) - 1)
            self.selected_pid = processes[self.selected_index].pid

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
    ) -> KeyResult:
        kill_func = kill_func or kill_process
        if not processes_synced:
            self.sync(processes)

        if key == KEY_CTRL_C:
            return KeyResult(quit=True, changed=True)

        if self.mode == MODE_KILL_CONFIRM:
            return self.handle_kill_confirm_key(key, processes, kill_func, processes_synced=True)

        if self.mode == MODE_SORT_MENU:
            return self.handle_sort_menu_key(key)

        if self.mode == MODE_SEARCH:
            return self.handle_search_key(key, processes, processes_synced=True)

        if key == "q":
            return KeyResult(quit=True, changed=True)
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
        if key == "n":
            self.search_next(processes, direction=1, processes_synced=True)
            return KeyResult(changed=True)
        if key == "N":
            self.search_next(processes, direction=-1, processes_synced=True)
            return KeyResult(changed=True)
        if key == "x":
            selected = self.selected_synced_process(processes)
            if selected is None:
                self.set_status_message("No process selected")
            else:
                self.mode = MODE_KILL_CONFIRM
                self.kill_confirm_index = 0
                self.clear_status_message()
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

    def handle_kill_confirm_key(
        self,
        key: str,
        processes: list[ProcessInfo],
        kill_func: Callable[[int, signal.Signals], None],
        processes_synced: bool = False,
    ) -> KeyResult:
        if key in ("j", KEY_DOWN, KEY_LEFT):
            self.kill_confirm_index = max(0, self.kill_confirm_index - 1)
            return KeyResult(changed=True)
        if key in ("k", KEY_UP, KEY_RIGHT):
            self.kill_confirm_index = min(len(KILL_CONFIRM_OPTIONS) - 1, self.kill_confirm_index + 1)
            return KeyResult(changed=True)
        if key in (KEY_ESC, "q", "n", "N"):
            self.mode = MODE_NORMAL
            self.clear_status_message()
            return KeyResult(changed=True)
        if key in ("y", "Y"):
            self.kill_confirm_index = KILL_CONFIRM_OPTIONS.index(KILL_CONFIRM_SIGTERM)
        elif key == KEY_ENTER:
            option = KILL_CONFIRM_OPTIONS[self.kill_confirm_index]
            if option == KILL_CONFIRM_CANCEL:
                self.mode = MODE_NORMAL
                self.clear_status_message()
                return KeyResult(changed=True)
        else:
            return KeyResult()

        option = KILL_CONFIRM_OPTIONS[self.kill_confirm_index]
        kill_signal = KILL_CONFIRM_SIGNALS[option]
        selected = self.selected_synced_process(processes) if processes_synced else self.selected_process(processes)
        self.mode = MODE_NORMAL
        if selected is None:
            self.set_status_message("No process selected")
            return KeyResult(changed=True)
        try:
            kill_func(selected.pid, kill_signal)
        except ProcessLookupError:
            self.set_status_message(f"PID {selected.pid} is no longer running")
        except PermissionError:
            self.set_status_message(f"Permission denied killing PID {selected.pid}")
        except OSError as exc:
            self.set_status_message(f"Failed to kill PID {selected.pid}: {exc}")
        else:
            self.set_status_message(f"Sent {kill_signal.name} to PID {selected.pid}")
        return KeyResult(changed=True)

    def handle_sort_menu_key(self, key: str) -> KeyResult:
        if key in ("j", KEY_DOWN, KEY_LEFT):
            self.sort_menu_index = max(0, self.sort_menu_index - 1)
            return KeyResult(changed=True)
        if key in ("k", KEY_UP, KEY_RIGHT):
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
        self.selected_index = clamp(self.selected_index + delta, 0, len(processes) - 1)
        self.selected_pid = processes[self.selected_index].pid
        self.ensure_selected_visible(len(processes))

    def search_next(self, processes: list[ProcessInfo], direction: int, processes_synced: bool = False) -> bool:
        query = self.search_query.strip()
        if not query:
            self.set_status_message("No search query")
            return False
        match_index = self.search_match_index(processes, query, direction, processes_synced=processes_synced)
        if match_index is None:
            self.set_status_message(f"No matches for: {query}")
            return False
        self.selected_index = match_index
        self.selected_pid = processes[match_index].pid
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
        if process_count <= 0:
            return "Processes  0/0"
        return f"Processes  {self.selected_index + 1}/{process_count}"

    def caption(self) -> str:
        return self.status_message

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
            return []
        data = os.read(self.fd, 64)
        return parse_keys(data)


def parse_keys(data: bytes | str) -> list[str]:
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
        for raw, key in (
            ("\x1b[A", KEY_UP),
            ("\x1b[B", KEY_DOWN),
            ("\x1b[C", KEY_RIGHT),
            ("\x1b[D", KEY_LEFT),
            ("\x1b[5~", KEY_PAGE_UP),
            ("\x1b[6~", KEY_PAGE_DOWN),
        ):
            if sequence.startswith(raw):
                keys.append(key)
                index += len(raw)
                matched = True
                break
        if not matched:
            keys.append(KEY_ESC)
            index += 1
    return keys


def is_printable_key(key: str) -> bool:
    return len(key) == 1 and key.isprintable()


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


def kill_process(pid: int, kill_signal: signal.Signals = signal.SIGTERM) -> None:
    os.kill(pid, kill_signal)


def clamp(value: int, low: int, high: int) -> int:
    return min(high, max(low, value))
