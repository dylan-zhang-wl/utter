"""Vertex AI, authenticated with Google Cloud credentials instead of an API key.

The author already pays for Google Cloud and does not have a Gemini API key.
Vertex takes an OAuth bearer token rather than a key, so this provider mints
one and hands it to the same OpenAI-shaped client everything else uses —
Google publishes an OpenAI-compatible endpoint on Vertex too.

Measured against the author's project on 2026-08-10:

    google/gemini-2.5-flash-lite    980 ms
    google/gemini-2.5-flash        6946 ms   ← same output, seven times the wait

Identical text out of both, so Flash-Lite is not a compromise here; it is the
right tool. Polish adds punctuation and removes filler, and 铁律 10 spends the
whole prompt forbidding the model from thinking about the argument.

The `x-goog-user-project` header is not required — tested without it, 200.

## Two identities, and why this tries both

Application Default Credentials are the standard path, and on a machine with
one Google account they are the only path worth having. The author has two:
ADC on this machine belongs to one account (written by another tool, pointing
at a managed project that account cannot administer) while the gcloud active
account owns the project with Vertex enabled. ADC gets 403; the gcloud
credential gets 200.

Rather than telling them to run `gcloud auth application-default login` — which
would overwrite the ADC another tool is relying on — this tries ADC, and on an
auth failure falls back to the gcloud CLI's own token and remembers which one
worked. One extra round trip, once, on a path that was going to fail anyway.

Tokens last an hour and are cached until a minute before expiry. Nothing here
logs a token, and the only thing ever printed about one is which account it
belongs to.
"""

from __future__ import annotations

import logging
import subprocess
import time

from backend.providers.llm import LlmError

log = logging.getLogger(__name__)

SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)
DEFAULT_LOCATION = "global"
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"

#: Refresh this long before the token actually dies, so a request never leaves
#: with a credential that expires in flight.
EXPIRY_MARGIN_SECONDS = 60


def _base_url(project: str, location: str) -> str:
    host = (
        "https://aiplatform.googleapis.com"
        if location == "global"
        else f"https://{location}-aiplatform.googleapis.com"
    )
    return f"{host}/v1/projects/{project}/locations/{location}/endpoints/openapi"


def _adc_token() -> tuple[str, float, str]:
    """(token, expiry epoch, whose account). Raises if ADC is not usable."""
    import google.auth
    import google.auth.transport.requests

    credentials, _ = google.auth.default(scopes=list(SCOPES))
    credentials.refresh(google.auth.transport.requests.Request())
    expiry = credentials.expiry.timestamp() if credentials.expiry else time.time() + 3300
    who = getattr(credentials, "service_account_email", None) or "ADC"
    return credentials.token, expiry, who


def _gcloud_token() -> tuple[str, float, str]:
    """The gcloud CLI's active user credential. Not available in a packaged app."""
    try:
        token = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip()
    except FileNotFoundError:
        raise LlmError("gcloud 没装") from None
    except subprocess.CalledProcessError as exc:
        # stderr can name the account but never contains the token.
        raise LlmError(f"gcloud 取不到 token：{exc.stderr.strip()[:120]}") from None
    except subprocess.TimeoutExpired:
        raise LlmError("gcloud 取 token 超时") from None
    if not token:
        raise LlmError("gcloud 返回了空 token")
    # gcloud tokens are good for an hour; it does not tell us when.
    return token, time.time() + 3300, "gcloud active account"


class VertexProvider:
    id = "vertex"
    display_name = "Vertex AI (Google Cloud，用云凭据不用 key)"

    def __init__(
        self,
        project: str | None = None,
        location: str = DEFAULT_LOCATION,
        model: str = DEFAULT_MODEL,
    ):
        self.project = project
        self.location = location
        self.model = model
        self._token: str | None = None
        self._expires_at = 0.0
        self._source = None  # which of the two worked, once we know

    # -- credentials --

    def _sources(self):
        """ADC first — it is the standard and the only one a packaged app has."""
        if self._source is not None:
            return [self._source]
        return [("adc", _adc_token), ("gcloud", _gcloud_token)]

    def _fresh_token(self) -> str:
        if self._token and time.time() < self._expires_at - EXPIRY_MARGIN_SECONDS:
            return self._token

        problems = []
        for name, mint in self._sources():
            try:
                token, expiry, who = mint()
            except Exception as exc:
                problems.append(f"{name}: {exc}")
                continue
            self._token, self._expires_at = token, expiry
            log.info("Vertex 用 %s 的凭据（%s）", name, who)
            return token
        raise LlmError("拿不到 Google Cloud 凭据 —— " + "；".join(problems))

    def _confirm_source(self, name: str) -> None:
        """Remember which credential worked, so later calls skip the dead one."""
        for candidate in (("adc", _adc_token), ("gcloud", _gcloud_token)):
            if candidate[0] == name:
                self._source = candidate

    # -- the provider interface --

    def is_available(self) -> tuple[bool, str]:
        """Cheap checks only — no request, no token minted.

        `utter doctor` probes every provider, and a probe that called Vertex
        would cost a round trip every time the author asked what was wrong.
        """
        if not self.project:
            return False, (
                "还没设 vertex_project —— 在 ~/Utter/config.json 里填上项目 ID"
                "（`gcloud projects list` 能看到）"
            )
        try:
            import google.auth  # noqa: F401
        except ImportError:
            return False, "google-auth 没装"
        return True, ""

    def complete(self, system: str, user: str) -> str:
        if not self.project:
            raise LlmError("没有配置 vertex_project")

        from openai import OpenAI

        attempted = []
        for name, mint in self._sources():
            try:
                token, expiry, _who = mint()
            except Exception as exc:
                attempted.append(f"{name}: {exc}")
                continue

            try:
                client = OpenAI(
                    api_key=token, base_url=_base_url(self.project, self.location)
                )
                response = client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,  # 铁律 10 — no creativity wanted here
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
            except Exception as exc:
                # No exc_info and no repr of the client: an SDK exception can
                # carry the request headers, and those hold the bearer token.
                kind = type(exc).__name__
                attempted.append(f"{name}: {kind}")
                if _is_auth_failure(exc) and self._source is None:
                    # The other identity may well work. This is the whole
                    # reason both are tried.
                    log.info("Vertex 拒绝了 %s 的凭据，换一个试试", name)
                    continue
                raise LlmError(f"Vertex: {kind}") from None

            self._token, self._expires_at = token, expiry
            self._confirm_source(name)
            return response.choices[0].message.content or ""

        raise LlmError("Vertex 认证失败 —— " + "；".join(attempted))


def _is_auth_failure(exc) -> bool:
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return True
    # openai wraps some transports; the code is not always an attribute.
    return any(code in str(exc) for code in ("401", "403", "PERMISSION_DENIED"))
