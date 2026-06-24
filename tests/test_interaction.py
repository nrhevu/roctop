from __future__ import annotations

import signal
import unittest

from roctop.interaction import (
    KEY_BACKSPACE,
    KEY_DOWN,
    KEY_ENTER,
    KEY_LEFT,
    KEY_PAGE_DOWN,
    KEY_PAGE_UP,
    KEY_RIGHT,
    KEY_UP,
    MODE_FILTER,
    MODE_KILL_CONFIRM,
    MODE_NORMAL,
    MODE_SEARCH,
    MODE_SORT_MENU,
    ProcessViewState,
    STATUS_MESSAGE_SECONDS,
    elapsed_seconds,
    parse_keys,
)
from roctop.models import ProcessInfo


def proc(pid: int, **kwargs) -> ProcessInfo:
    return ProcessInfo(gpu_index=kwargs.pop("gpu_index", 0), pid=pid, **kwargs)


class InteractionTests(unittest.TestCase):
    def test_parse_keys_maps_arrows_pages_enter_and_escape(self) -> None:
        self.assertEqual(
            parse_keys(b"j\x1b[A\x1b[B\x1b[C\x1b[D\x1b[5~\x1b[6~\x7f\r\x1b"),
            [
                "j",
                KEY_UP,
                KEY_DOWN,
                KEY_RIGHT,
                KEY_LEFT,
                KEY_PAGE_UP,
                KEY_PAGE_DOWN,
                KEY_BACKSPACE,
                KEY_ENTER,
                "esc",
            ],
        )

    def test_cursor_movement_and_page_keys_clamp(self) -> None:
        processes = [proc(pid) for pid in range(100, 106)]
        state = ProcessViewState(viewport_rows=3)
        state.sync(processes)

        state.handle_key("j", processes)
        self.assertEqual(state.selected_pid, 101)
        state.handle_key(KEY_DOWN, processes)
        self.assertEqual(state.selected_pid, 102)
        state.handle_key(KEY_PAGE_DOWN, processes)
        self.assertEqual(state.selected_pid, 105)
        self.assertEqual(state.scroll_offset, 3)
        state.handle_key("k", processes)
        self.assertEqual(state.selected_pid, 104)
        state.handle_key(KEY_PAGE_UP, processes)
        self.assertEqual(state.selected_pid, 101)
        state.handle_key(KEY_UP, processes)
        self.assertEqual(state.selected_pid, 100)

    def test_cursor_tracks_duplicate_pid_by_gpu_row(self) -> None:
        processes = [
            proc(100, gpu_index=0),
            proc(100, gpu_index=1),
            proc(101, gpu_index=2),
        ]
        state = ProcessViewState(viewport_rows=3)
        state.sync(processes)

        state.handle_key("j", processes)
        self.assertEqual(state.selected_pid, 100)
        self.assertEqual(state.selected_index, 1)

        state.sync(processes)
        self.assertEqual(state.selected_pid, 100)
        self.assertEqual(state.selected_index, 1)

    def test_sort_menu_applies_field_and_toggles_direction(self) -> None:
        processes = [
            proc(1, cpu_percent=1.0),
            proc(2, cpu_percent=90.0),
            proc(3, cpu_percent=30.0),
        ]
        state = ProcessViewState(viewport_rows=3)
        state.handle_key("s", processes)
        self.assertEqual(state.mode, MODE_SORT_MENU)
        for _ in range(3):
            state.handle_key("l", processes)
        state.handle_key(KEY_ENTER, processes)
        self.assertEqual(state.mode, MODE_NORMAL)
        self.assertEqual(state.sort_field, "cpu")
        self.assertTrue(state.sort_desc)
        self.assertEqual([row.pid for row in state.sorted_processes(processes)], [2, 3, 1])

        state.handle_key("s", processes)
        state.handle_key(KEY_ENTER, processes)
        self.assertFalse(state.sort_desc)
        self.assertEqual([row.pid for row in state.sorted_processes(processes)], [1, 3, 2])

    def test_sort_menu_uses_down_right_for_right_and_up_left_for_left(self) -> None:
        processes = [proc(1)]
        state = ProcessViewState(viewport_rows=3)
        state.handle_key("s", processes)
        self.assertEqual(state.sort_menu_index, 0)
        state.handle_key(KEY_DOWN, processes)
        self.assertEqual(state.sort_menu_index, 1)
        state.handle_key(KEY_RIGHT, processes)
        self.assertEqual(state.sort_menu_index, 2)
        state.handle_key(KEY_UP, processes)
        self.assertEqual(state.sort_menu_index, 1)
        state.handle_key(KEY_LEFT, processes)
        self.assertEqual(state.sort_menu_index, 0)

    def test_sort_menu_uses_h_l_for_left_right(self) -> None:
        processes = [proc(1)]
        state = ProcessViewState(viewport_rows=3)
        state.handle_key("s", processes)

        state.handle_key("l", processes)
        self.assertEqual(state.sort_menu_index, 1)
        state.handle_key("l", processes)
        self.assertEqual(state.sort_menu_index, 2)
        state.handle_key("h", processes)
        self.assertEqual(state.sort_menu_index, 1)
        state.handle_key("h", processes)
        self.assertEqual(state.sort_menu_index, 0)

    def test_cursor_tracks_selected_pid_after_sort_refresh(self) -> None:
        processes = [
            proc(1, cpu_percent=1.0),
            proc(2, cpu_percent=90.0),
            proc(3, cpu_percent=30.0),
        ]
        state = ProcessViewState(selected_pid=3, sort_field="cpu", sort_desc=True, viewport_rows=2)
        sorted_processes = state.sorted_processes(processes)
        state.sync(sorted_processes)
        self.assertEqual(state.selected_pid, 3)
        self.assertEqual(state.selected_index, 1)

    def test_tree_mode_toggles_with_t(self) -> None:
        processes = [proc(100)]
        state = ProcessViewState(viewport_rows=3)

        result = state.handle_key("t", processes)
        self.assertTrue(result.changed)
        self.assertTrue(state.tree_mode)

        state.handle_key("t", processes)
        self.assertFalse(state.tree_mode)

    def test_tree_mode_keeps_parent_first_and_sorts_siblings(self) -> None:
        processes = [
            proc(12, ppid=10, cpu_percent=1.0, args="child-low"),
            proc(11, ppid=10, cpu_percent=90.0, args="child-high"),
        ]
        ancestors = [ProcessInfo(gpu_index=None, pid=10, args="parent")]
        state = ProcessViewState(tree_mode=True, sort_field="cpu", sort_desc=True, viewport_rows=4)

        display = state.display_processes(processes, ancestors)

        self.assertEqual([row.pid for row in display], [10, 11, 12])

    def test_tree_mode_default_sort_uses_pid_order(self) -> None:
        processes = [
            proc(12, ppid=10, args="child-high-pid"),
            proc(11, ppid=10, args="child-low-pid"),
        ]
        ancestors = [ProcessInfo(gpu_index=None, pid=10, args="parent")]
        state = ProcessViewState(tree_mode=True, viewport_rows=4)

        display = state.display_processes(processes, ancestors)

        self.assertEqual([row.pid for row in display], [10, 11, 12])

    def test_tree_filter_keeps_only_matching_rows_as_roots(self) -> None:
        processes = [
            proc(11, ppid=10, args="demo::train"),
            proc(12, ppid=10, args="demo::serve"),
        ]
        ancestors = [ProcessInfo(gpu_index=None, pid=10, args="launcher")]
        state = ProcessViewState(tree_mode=True, filter_query="train", viewport_rows=4)

        display = state.display_processes(processes, ancestors)

        self.assertEqual([row.pid for row in display], [11])

    def test_search_mode_commits_query_and_matches_command_pid_or_user(self) -> None:
        processes = [
            proc(100, user="alice", args="demo::trainer --batch-size 64"),
            proc(101, user="bob", args="demo::serve --port 3000"),
        ]
        state = ProcessViewState(viewport_rows=3)

        state.handle_key("/", processes)
        self.assertEqual(state.mode, MODE_SEARCH)
        for key in "serve":
            state.handle_key(key, processes)
        state.handle_key(KEY_ENTER, processes)
        self.assertEqual(state.mode, MODE_NORMAL)
        self.assertEqual(state.search_query, "serve")
        self.assertEqual(state.selected_pid, 101)
        self.assertEqual(state.status_message, "Search: serve")

        state.handle_key("/", processes)
        self.assertEqual(state.search_input, "")
        for key in "ALICE":
            state.handle_key(key, processes)
        state.handle_key(KEY_ENTER, processes)
        self.assertEqual(state.selected_pid, 100)

        state.handle_key("/", processes)
        self.assertEqual(state.search_input, "")
        for key in "101":
            state.handle_key(key, processes)
        state.handle_key(KEY_ENTER, processes)
        self.assertEqual(state.selected_pid, 101)

    def test_search_escape_keeps_previous_query(self) -> None:
        processes = [proc(100, args="demo::trainer"), proc(101, args="demo::serve")]
        state = ProcessViewState(search_query="trainer", viewport_rows=3)

        state.handle_key("/", processes)
        self.assertEqual(state.search_input, "")
        for key in " new":
            state.handle_key(key, processes)
        state.handle_key("esc", processes)

        self.assertEqual(state.mode, MODE_NORMAL)
        self.assertEqual(state.search_query, "trainer")
        self.assertEqual(state.search_input, "")

    def test_filter_mode_applies_query_realtime_and_keeps_it_on_enter(self) -> None:
        processes = [
            proc(100, args="demo::trainer"),
            proc(101, args="demo::serve"),
            proc(102, user="alice", args="demo::helper"),
        ]
        state = ProcessViewState(selected_pid=100, viewport_rows=3)

        state.handle_key("f", processes)
        self.assertEqual(state.mode, MODE_FILTER)
        self.assertEqual(state.filter_input, "")
        for key in "serve":
            state.handle_key(key, processes)

        self.assertEqual(state.filter_query, "serve")
        self.assertEqual([row.pid for row in state.display_processes(processes)], [101])
        self.assertEqual(state.selected_pid, 101)

        state.handle_key(KEY_BACKSPACE, processes)
        self.assertEqual(state.filter_input, "serv")
        self.assertEqual(state.filter_query, "serv")
        self.assertEqual([row.pid for row in state.display_processes(processes)], [101])

        state.handle_key(KEY_ENTER, processes)
        self.assertEqual(state.mode, MODE_NORMAL)
        self.assertEqual(state.filter_query, "serv")
        self.assertEqual(state.filter_input, "serv")

    def test_filter_mode_prefills_existing_query_and_escape_clears_filter(self) -> None:
        processes = [proc(100, args="demo::trainer"), proc(101, args="demo::serve")]
        state = ProcessViewState(filter_query="serve", viewport_rows=3)

        state.handle_key("f", processes)
        self.assertEqual(state.mode, MODE_FILTER)
        self.assertEqual(state.filter_input, "serve")
        state.handle_key("r", processes)
        self.assertEqual(state.filter_query, "server")
        self.assertEqual(state.selected_pid, None)

        state.handle_key("esc", processes)
        self.assertEqual(state.mode, MODE_NORMAL)
        self.assertEqual(state.filter_query, "")
        self.assertEqual(state.filter_input, "")
        self.assertEqual([row.pid for row in state.display_processes(processes)], [100, 101])

    def test_escape_clears_active_filter_without_reopening_filter_mode(self) -> None:
        processes = [proc(100, args="demo::trainer"), proc(101, args="demo::serve")]
        state = ProcessViewState(filter_query="serve", filter_input="serve", viewport_rows=3)
        state.sync(state.display_processes(processes))
        self.assertEqual(state.mode, MODE_NORMAL)
        self.assertEqual([row.pid for row in state.display_processes(processes)], [101])

        result = state.handle_key("esc", processes)

        self.assertTrue(result.changed)
        self.assertEqual(state.mode, MODE_NORMAL)
        self.assertEqual(state.filter_query, "")
        self.assertEqual(state.filter_input, "")
        self.assertEqual([row.pid for row in state.display_processes(processes)], [100, 101])

    def test_search_next_and_previous_wrap_in_sorted_order(self) -> None:
        processes = [
            proc(1, cpu_percent=10.0, args="demo::worker low"),
            proc(2, cpu_percent=90.0, args="demo::worker high"),
            proc(3, cpu_percent=50.0, args="demo::other"),
        ]
        state = ProcessViewState(
            selected_pid=2,
            sort_field="cpu",
            sort_desc=True,
            search_query="worker",
            viewport_rows=2,
        )
        sorted_processes = state.sorted_processes(processes)
        state.sync(sorted_processes)

        state.handle_key("n", sorted_processes)
        self.assertEqual(state.selected_pid, 1)
        state.handle_key("N", sorted_processes)
        self.assertEqual(state.selected_pid, 2)

    def test_search_status_for_missing_query_or_match(self) -> None:
        processes = [proc(100, args="demo::trainer"), proc(101, args="demo::serve")]
        state = ProcessViewState(selected_pid=100, viewport_rows=3)

        state.handle_key("n", processes)
        self.assertEqual(state.selected_pid, 100)
        self.assertEqual(state.status_message, "No search query")

        state.search_query = "missing"
        state.handle_key("n", processes)
        self.assertEqual(state.selected_pid, 100)
        self.assertEqual(state.status_message, "No matches for: missing")

    def test_status_message_expires_after_three_seconds(self) -> None:
        state = ProcessViewState()
        state.set_status_message("Search: demo", now=10.0)

        self.assertEqual(state.status_message_expires_at, 10.0 + STATUS_MESSAGE_SECONDS)
        self.assertFalse(state.expire_status_message(now=12.9))
        self.assertEqual(state.status_message, "Search: demo")
        self.assertTrue(state.expire_status_message(now=13.0))
        self.assertEqual(state.status_message, "")
        self.assertIsNone(state.status_message_expires_at)

    def test_kill_confirm_can_cancel_or_send_selected_signal(self) -> None:
        processes = [proc(42)]
        calls: list[tuple[int, signal.Signals]] = []
        state = ProcessViewState(viewport_rows=3)
        state.sync(processes)

        def record(pid: int, kill_signal: signal.Signals) -> None:
            calls.append((pid, kill_signal))

        state.handle_key("x", processes)
        self.assertEqual(state.mode, MODE_KILL_CONFIRM)
        self.assertEqual(state.kill_confirm_index, 0)
        self.assertEqual(state.status_message, "")
        state.handle_key(KEY_ENTER, processes, kill_func=record)
        self.assertEqual(state.mode, MODE_NORMAL)
        self.assertEqual(calls, [])
        self.assertEqual(state.status_message, "")

        state.handle_key("x", processes)
        state.handle_key(KEY_RIGHT, processes, kill_func=record)
        self.assertEqual(state.kill_confirm_index, 1)
        state.handle_key(KEY_ENTER, processes, kill_func=record)
        self.assertEqual(calls, [(42, signal.SIGTERM)])
        self.assertIn("Sent SIGTERM", state.status_message)

        state.handle_key("x", processes)
        state.handle_key(KEY_RIGHT, processes, kill_func=record)
        state.handle_key(KEY_RIGHT, processes, kill_func=record)
        self.assertEqual(state.kill_confirm_index, 2)
        state.handle_key(KEY_ENTER, processes, kill_func=record)
        self.assertEqual(calls[-1], (42, signal.SIGKILL))
        self.assertIn("Sent SIGKILL", state.status_message)

    def test_kill_confirm_uses_h_l_for_left_right(self) -> None:
        processes = [proc(42)]
        state = ProcessViewState(viewport_rows=3)
        state.sync(processes)

        state.handle_key("x", processes)
        state.handle_key("l", processes)
        self.assertEqual(state.kill_confirm_index, 1)
        state.handle_key("l", processes)
        self.assertEqual(state.kill_confirm_index, 2)
        state.handle_key("h", processes)
        self.assertEqual(state.kill_confirm_index, 1)
        state.handle_key("h", processes)
        self.assertEqual(state.kill_confirm_index, 0)

    def test_kill_confirm_uses_down_right_and_up_left(self) -> None:
        processes = [proc(42)]
        state = ProcessViewState(viewport_rows=3)
        state.sync(processes)

        state.handle_key("x", processes)
        state.handle_key(KEY_DOWN, processes)
        self.assertEqual(state.kill_confirm_index, 1)
        state.handle_key(KEY_DOWN, processes)
        self.assertEqual(state.kill_confirm_index, 2)
        state.handle_key(KEY_UP, processes)
        self.assertEqual(state.kill_confirm_index, 1)
        state.handle_key(KEY_UP, processes)
        self.assertEqual(state.kill_confirm_index, 0)

    def test_kill_errors_become_status_messages(self) -> None:
        processes = [proc(42)]
        state = ProcessViewState(viewport_rows=3)
        state.sync(processes)

        def deny(_pid: int, _kill_signal: signal.Signals) -> None:
            raise PermissionError

        state.handle_key("x", processes)
        state.handle_key(KEY_RIGHT, processes, kill_func=deny)
        state.handle_key(KEY_ENTER, processes, kill_func=deny)
        self.assertIn("Permission denied", state.status_message)

    def test_elapsed_seconds_parses_ps_etime_formats(self) -> None:
        self.assertEqual(elapsed_seconds("01:02"), 62)
        self.assertEqual(elapsed_seconds("03:01:02"), 10862)
        self.assertEqual(elapsed_seconds("1-03:01:02"), 97262)


if __name__ == "__main__":
    unittest.main()
