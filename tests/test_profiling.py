from __future__ import annotations

import unittest
from unittest.mock import patch

from roctop import profiling


class ProfilingTests(unittest.TestCase):
    def test_profile_output_failure_does_not_break_application(self) -> None:
        with (
            patch.object(profiling, "_PROFILE_ENABLED", True),
            patch("builtins.print", side_effect=ValueError("closed stream")),
        ):
            with profiling.profile_span("test"):
                pass


if __name__ == "__main__":
    unittest.main()
