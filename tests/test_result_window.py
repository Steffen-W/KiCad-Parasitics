"""Background work of the result window (the window itself needs a display)."""

from __future__ import annotations

from analysis_progress import Stage


def test_task_reports_progress_until_cancelled(no_wx: None) -> None:
    from result_window import _progress_text, _Task

    task = _Task()

    assert task.report(Stage.PATH, 1, 3)
    assert _progress_text(*task.progress) == "Step 2 of 4: Finding path (1 of 3 done)"
    task.cancel.set()
    assert not task.report(Stage.PATH, 2, 3)
