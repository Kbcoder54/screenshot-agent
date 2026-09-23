"""
Tests for file_watcher.FileWatcher.

Drives `_check()` directly (rather than the background thread + sleep loop)
so the tests are fast and deterministic -- no timing flakiness.
"""

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from file_watcher import FileWatcher


@pytest.fixture
def watch_dir(tmp_path):
    return tmp_path


@pytest.fixture
def events():
    log = {"new": [], "modified": [], "deleted": []}
    return log


@pytest.fixture
def watcher(watch_dir, events):
    return FileWatcher(
        watch_dir,
        on_new=lambda p: events["new"].append(p),
        on_modified=lambda p: events["modified"].append(p),
        on_deleted=lambda p: events["deleted"].append(p),
    )


def write_image(dir_path: Path, name: str, content: bytes = b"fake image bytes") -> Path:
    p = dir_path / name
    p.write_bytes(content)
    return p


def test_detects_new_image(watch_dir, watcher, events):
    write_image(watch_dir, "a.jpg")
    watcher._check()
    assert len(events["new"]) == 1
    assert events["new"][0].endswith("a.jpg")


def test_ignores_non_image_files(watch_dir, watcher, events):
    write_image(watch_dir, "notes.txt")
    watcher._check()
    assert events["new"] == []


def test_detects_modification(watch_dir, watcher, events):
    path = write_image(watch_dir, "a.jpg")
    watcher._check()
    events["new"].clear()

    # Force an mtime change so the "modified" branch definitely fires,
    # regardless of filesystem timestamp resolution.
    new_mtime = path.stat().st_mtime + 5
    path.write_bytes(b"different bytes, different size")
    os.utime(path, (new_mtime, new_mtime))

    watcher._check()
    assert len(events["modified"]) == 1
    assert events["new"] == []


def test_detects_deletion(watch_dir, watcher, events):
    path = write_image(watch_dir, "a.jpg")
    watcher._check()
    path.unlink()
    watcher._check()
    assert len(events["deleted"]) == 1


def test_no_events_when_nothing_changes(watch_dir, watcher, events):
    write_image(watch_dir, "a.jpg")
    watcher._check()
    events["new"].clear()

    watcher._check()  # nothing changed
    assert events["new"] == []
    assert events["modified"] == []
    assert events["deleted"] == []


def test_multiple_image_extensions_detected(watch_dir, watcher, events):
    for name in ("a.jpg", "b.PNG", "c.gif", "d.webp"):
        write_image(watch_dir, name)
    watcher._check()
    assert len(events["new"]) == 4


def test_case_insensitive_extension(watch_dir, watcher, events):
    write_image(watch_dir, "a.JPG")
    watcher._check()
    assert len(events["new"]) == 1


def test_scan_survives_vanished_folder(watch_dir, watcher):
    """_scan() must not raise if the watched folder disappears mid-run."""
    import shutil

    shutil.rmtree(watch_dir)
    result = watcher._scan()
    assert result == {}


def test_start_stop_runs_in_background(watch_dir, events):
    watcher = FileWatcher(
        watch_dir,
        on_new=lambda p: events["new"].append(p),
        interval=0.05,
    )
    watcher.start()
    try:
        write_image(watch_dir, "a.jpg")
        time.sleep(0.3)
        assert len(events["new"]) == 1
    finally:
        watcher.stop()


def test_no_callback_set_does_not_crash(watch_dir):
    watcher = FileWatcher(watch_dir)  # no callbacks at all
    write_image(watch_dir, "a.jpg")
    watcher._check()  # should not raise
