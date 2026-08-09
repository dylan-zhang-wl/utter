"""P1 Task 2 — API keys in the macOS Keychain (铁律 4).

The keyring backend is replaced with an in-memory stub throughout. These tests
must never write to the real Keychain: a test run should not leave credentials
on the developer's machine, and CI has no Keychain at all.
"""

import logging

import pytest

from backend import secrets


class FakeKeyring:
    """Stands in for the `keyring` module's four functions we use."""

    def __init__(self):
        self.store: dict[tuple[str, str], str] = {}

    def set_password(self, service, name, value):
        self.store[(service, name)] = value

    def get_password(self, service, name):
        return self.store.get((service, name))

    def delete_password(self, service, name):
        try:
            del self.store[(service, name)]
        except KeyError:
            raise secrets.keyring.errors.PasswordDeleteError(name)


@pytest.fixture
def fake(monkeypatch):
    stub = FakeKeyring()
    monkeypatch.setattr(secrets.keyring, "set_password", stub.set_password)
    monkeypatch.setattr(secrets.keyring, "get_password", stub.get_password)
    monkeypatch.setattr(secrets.keyring, "delete_password", stub.delete_password)
    return stub


def test_round_trip(fake):
    secrets.set_secret("openai_api_key", "sk-abc123")
    assert secrets.get_secret("openai_api_key") == "sk-abc123"


def test_missing_secret_returns_none(fake):
    assert secrets.get_secret("never_set") is None


def test_delete_unset_secret_is_a_noop(fake):
    secrets.delete_secret("never_set")  # must not raise


def test_delete_removes(fake):
    secrets.set_secret("groq_api_key", "gsk-xyz")
    secrets.delete_secret("groq_api_key")
    assert secrets.get_secret("groq_api_key") is None


def test_uses_the_project_service_name(fake):
    secrets.set_secret("openai_api_key", "sk-abc")
    assert ("com.dylan.utter", "openai_api_key") in fake.store


def test_empty_value_is_stored_as_absent(fake):
    """An empty string is a cleared key, not a key whose value is "".

    Otherwise a user who blanks the field in the UI gets an auth error from the
    provider instead of the honest "no key configured".
    """
    secrets.set_secret("openai_api_key", "")
    assert secrets.get_secret("openai_api_key") is None


def test_backend_failure_reads_as_absent(monkeypatch, caplog):
    """A locked or unavailable Keychain must not crash the app.

    Same reasoning as config.load(): degrade to "unset" so the user reaches a
    screen where they can do something about it.
    """
    def boom(*_a, **_k):
        raise secrets.keyring.errors.KeyringError("locked")

    monkeypatch.setattr(secrets.keyring, "get_password", boom)
    with caplog.at_level(logging.WARNING):
        assert secrets.get_secret("openai_api_key") is None


def test_nothing_logs_the_value(fake, caplog):
    """铁律 4. Not even truncated — a prefix still identifies a leaked key."""
    with caplog.at_level(logging.DEBUG):
        secrets.set_secret("openai_api_key", "sk-supersecret-value")
        secrets.get_secret("openai_api_key")
        secrets.delete_secret("openai_api_key")

    blob = caplog.text
    assert "sk-supersecret-value" not in blob
    assert "sk-super" not in blob
    assert "sk-" not in blob


def test_set_none_clears(fake):
    secrets.set_secret("openai_api_key", "sk-abc")
    secrets.set_secret("openai_api_key", None)
    assert secrets.get_secret("openai_api_key") is None
