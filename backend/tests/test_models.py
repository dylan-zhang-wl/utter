"""P1 Task 8 — the tier-aware model downloader.

The load-bearing test is test_present_model_is_not_redownloaded. The author's
machine already holds mlx-community/whisper-large-v3-turbo (1.6G) in
~/.cache/huggingface, and re-fetching it because the code failed to notice would
be a real cost on a real connection, not a hypothetical.

huggingface_hub is patched everywhere. Nothing here touches the network or the
real cache.
"""

import pytest

from backend import models
from backend.catalog import UnknownTier
from backend.hardware import Hardware


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """Stub snapshot_download, counting calls and simulating a cache."""
    state = {"downloads": [], "cached": set()}

    def snapshot_download(repo_id, **kwargs):
        if kwargs.get("local_files_only"):
            if repo_id not in state["cached"]:
                raise models.LocalEntryNotFoundError(f"{repo_id} not cached")
            return str(tmp_path / repo_id.replace("/", "--"))

        state["downloads"].append({"repo_id": repo_id, **kwargs})
        state["cached"].add(repo_id)
        path = tmp_path / repo_id.replace("/", "--")
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    monkeypatch.setattr(models, "snapshot_download", snapshot_download)
    return state


def hw(kind="apple_silicon"):
    return Hardware(kind=kind, ram_gb=16)


def test_ensure_model_returns_a_local_path(fake_hub):
    path = models.ensure_model("balanced", hardware=hw())
    assert path
    assert "whisper-large-v3-turbo" in str(path)


def test_resolves_the_tier_through_the_catalog(fake_hub):
    models.ensure_model("minimal", hardware=hw())
    assert fake_hub["downloads"][0]["repo_id"] == "mlx-community/whisper-base-mlx-q4"


def test_hardware_changes_which_repo_is_fetched(fake_hub):
    models.ensure_model("balanced", hardware=hw("cpu"))
    assert fake_hub["downloads"][0]["repo_id"] == (
        "dropbox-dash/faster-whisper-large-v3-turbo"
    )


def test_present_model_is_not_redownloaded(fake_hub):
    """The one that matters. large-v3-turbo is already on this machine; a
    downloader that re-fetches 1.6G because it did not check is a real bug."""
    fake_hub["cached"].add("mlx-community/whisper-large-v3-turbo")

    models.ensure_model("balanced", hardware=hw())

    assert fake_hub["downloads"] == [], "must not hit the network for a cached model"


def test_is_downloaded_reports_cache_state(fake_hub):
    assert models.is_downloaded("balanced", hardware=hw()) is False
    fake_hub["cached"].add("mlx-community/whisper-large-v3-turbo")
    assert models.is_downloaded("balanced", hardware=hw()) is True


def test_progress_callback_fires(fake_hub):
    seen = []
    models.ensure_model("balanced", hardware=hw(), on_progress=seen.append)

    assert seen, "the UI needs something to show during a multi-gigabyte fetch"
    assert any("whisper-large-v3-turbo" in str(m) for m in seen)


def test_progress_callback_reports_completion(fake_hub):
    seen = []
    models.ensure_model("balanced", hardware=hw(), on_progress=seen.append)
    assert seen[-1].done is True


def test_progress_is_not_reported_for_a_cached_model(fake_hub):
    fake_hub["cached"].add("mlx-community/whisper-large-v3-turbo")
    seen = []
    models.ensure_model("balanced", hardware=hw(), on_progress=seen.append)

    assert all(m.done for m in seen), "a cache hit should not look like a download"


def test_failure_raises_model_download_error(monkeypatch):
    def boom(*_a, **_k):
        raise OSError("connection reset")

    monkeypatch.setattr(models, "snapshot_download", boom)

    with pytest.raises(models.ModelDownloadError) as exc:
        models.ensure_model("balanced", hardware=hw())
    assert "connection reset" in str(exc.value)


def test_failure_names_the_repo(monkeypatch):
    """The user has to be able to tell which of four tiers failed."""
    def boom(*_a, **_k):
        raise OSError("connection reset")

    monkeypatch.setattr(models, "snapshot_download", boom)

    with pytest.raises(models.ModelDownloadError) as exc:
        models.ensure_model("high", hardware=hw())
    assert "whisper-large-v3-mlx" in str(exc.value)


def test_failed_download_is_not_registered_as_present(monkeypatch, fake_hub):
    """A half-finished fetch must not make is_downloaded() lie — the next run
    would then hand a truncated directory to the model loader."""
    calls = {"n": 0}
    original = models.snapshot_download

    def flaky(repo_id, **kwargs):
        if kwargs.get("local_files_only"):
            return original(repo_id, **kwargs)
        calls["n"] += 1
        raise OSError("connection reset")

    monkeypatch.setattr(models, "snapshot_download", flaky)

    with pytest.raises(models.ModelDownloadError):
        models.ensure_model("balanced", hardware=hw())

    assert models.is_downloaded("balanced", hardware=hw()) is False


def test_unknown_tier_propagates(fake_hub):
    with pytest.raises(UnknownTier):
        models.ensure_model("enormous", hardware=hw())


def test_disk_usage_lists_only_cached_tiers(fake_hub):
    fake_hub["cached"].add("mlx-community/whisper-large-v3-turbo")
    usage = models.disk_usage(hardware=hw())

    by_tier = {u.tier: u for u in usage}
    assert by_tier["balanced"].present is True
    assert by_tier["high"].present is False
    assert by_tier["balanced"].size_mb == 1614


def test_disk_usage_covers_every_tier(fake_hub):
    from backend.catalog import TIERS

    assert {u.tier for u in models.disk_usage(hardware=hw())} == set(TIERS)
