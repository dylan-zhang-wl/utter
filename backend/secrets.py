"""API keys in the OS credential store (铁律 4).

On macOS `keyring` resolves to the login Keychain; on Windows to the Credential
Manager; on Linux to whatever Secret Service is present. Nothing here ever
writes a key to a file, and nothing here ever logs a value.

Named `secrets` inside the `backend` package, which shadows the stdlib module of
that name for anything doing a relative import. Nothing in this project uses
stdlib `secrets`; absolute imports elsewhere still resolve to it normally.
"""

from __future__ import annotations

import logging

import keyring
import keyring.errors

log = logging.getLogger(__name__)

SERVICE = "com.dylan.utter"


def set_secret(name: str, value: str | None) -> None:
    """Store a secret. An empty or None value clears it instead.

    Blanking a field in the UI should leave the app reporting "no key
    configured", not sending an empty key to a provider and surfacing that as an
    authentication error the user cannot interpret.
    """
    if not value:
        delete_secret(name)
        return

    try:
        keyring.set_password(SERVICE, name, value)
    except keyring.errors.KeyringError:
        # Deliberately no `exc_info`: some backends put the attempted value into
        # the exception's string form.
        log.warning("could not store %r in the keychain", name)


def get_secret(name: str) -> str | None:
    """Return a stored secret, or None if unset or unreachable.

    A locked or missing credential store reads as "unset" rather than raising.
    Same reasoning as config.load(): the app must still start, because the
    screen where the user fixes this is inside the app.
    """
    try:
        value = keyring.get_password(SERVICE, name)
    except keyring.errors.KeyringError:
        log.warning("could not read %r from the keychain", name)
        return None

    return value or None


def delete_secret(name: str) -> None:
    """Remove a secret. Deleting something that was never set is not an error."""
    try:
        keyring.delete_password(SERVICE, name)
    except keyring.errors.PasswordDeleteError:
        pass
    except keyring.errors.KeyringError:
        log.warning("could not delete %r from the keychain", name)
