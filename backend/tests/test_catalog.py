"""P1 Task 4 — tier to concrete model, resolved against hardware.

Users pick "balanced"; the platform decides whether that means an MLX repo or a
CTranslate2 one. Design §5.3.

The matrix must have no holes: a machine kind with no entry for some tier means
a user picking that tier gets an exception at download time, on hardware the
author does not own.
"""

import json
import urllib.request

import pytest

from backend import catalog
from backend.hardware import Hardware

KINDS = ["apple_silicon", "cuda", "cpu"]


def hw(kind, ram_gb=16):
    return Hardware(kind=kind, ram_gb=ram_gb)


def test_balanced_on_apple_silicon_is_the_cached_turbo_model():
    """This exact repo is already in ~/.cache/huggingface and was benchmarked
    at ~1s per utterance in design §3. Changing it invalidates §3."""
    entry = catalog.resolve("balanced", hw("apple_silicon"))
    assert entry.repo == "mlx-community/whisper-large-v3-turbo"
    assert entry.runtime == "mlx"


def test_balanced_off_apple_silicon_is_not_mlx():
    """铁律 6. MLX has no backend outside Apple Silicon."""
    for kind in ("cuda", "cpu"):
        assert catalog.resolve("balanced", hw(kind)).runtime == "ctranslate2"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("tier", catalog.TIERS)
def test_every_tier_resolves_on_every_hardware(tier, kind):
    entry = catalog.resolve(tier, hw(kind))
    assert entry.repo
    assert "/" in entry.repo, "must be a HuggingFace repo id, not a bare name"
    assert entry.runtime in ("mlx", "ctranslate2")
    assert entry.size_mb > 0


@pytest.mark.parametrize("kind", KINDS)
def test_mlx_appears_only_on_apple_silicon(kind):
    runtimes = {catalog.resolve(t, hw(kind)).runtime for t in catalog.TIERS}
    if kind == "apple_silicon":
        assert runtimes == {"mlx"}
    else:
        assert "mlx" not in runtimes


def test_unknown_tier_raises():
    with pytest.raises(catalog.UnknownTier):
        catalog.resolve("enormous", hw("apple_silicon"))


def test_unknown_tier_names_the_valid_ones():
    """The error is read by a user who typed a tier at the CLI."""
    with pytest.raises(catalog.UnknownTier) as exc:
        catalog.resolve("enormous", hw("cpu"))
    for tier in catalog.TIERS:
        assert tier in str(exc.value)


@pytest.mark.parametrize("tier", catalog.TIERS)
def test_ui_metadata_is_present(tier):
    entry = catalog.resolve(tier, hw("apple_silicon"))
    assert entry.label
    assert entry.blurb
    assert entry.size_mb > 0


def test_balanced_is_marked_recommended():
    assert catalog.resolve("balanced", hw("apple_silicon")).recommended is True
    assert catalog.resolve("high", hw("apple_silicon")).recommended is False


def test_blurbs_do_not_claim_bigger_is_slower():
    """Design §5.3 warns against this explicitly.

    large-v3-turbo has 809M parameters but only 4 decoder layers against
    large-v3's 32, so it is both big AND fast. A blurb implying a linear
    size/speed tradeoff would push users away from the recommended tier.
    """
    balanced = catalog.resolve("balanced", hw("apple_silicon"))
    assert balanced.size_mb > catalog.resolve("light", hw("apple_silicon")).size_mb
    assert "快" in balanced.blurb, "the balanced tier must be sold on speed"


def test_sizes_are_measured_not_guessed():
    """Real figures, checked against the HF API on 2026-08-10.

    The design's original estimates were roughly half the truth for the top two
    tiers (1.5G/0.8G against an actual 3.1G/1.6G). Users read this number when
    deciding whether to download on a metered connection.
    """
    assert catalog.resolve("high", hw("apple_silicon")).size_mb == 3084
    assert catalog.resolve("balanced", hw("apple_silicon")).size_mb == 1614


def test_all_tiers_lists_in_declared_order():
    entries = catalog.all_tiers(hw("apple_silicon"))
    assert [e.tier for e in entries] == list(catalog.TIERS)


def test_all_tiers_is_hardware_specific():
    mac = {e.repo for e in catalog.all_tiers(hw("apple_silicon"))}
    pc = {e.repo for e in catalog.all_tiers(hw("cpu"))}
    assert not (mac & pc)


@pytest.mark.network
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("tier", catalog.TIERS)
def test_every_repo_actually_exists(tier, kind):
    """Opt-in: `pytest -m network`. Catches renamed or deleted repos, which
    otherwise surface as a 404 during a user's first-run download."""
    repo = catalog.resolve(tier, hw(kind)).repo
    url = f"https://huggingface.co/api/models/{repo}"
    with urllib.request.urlopen(url, timeout=30) as response:
        assert json.load(response)["id"] == repo
