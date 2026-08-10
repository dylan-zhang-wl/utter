"""Build Utter.app.

    ~/.venvs/utter/bin/python packaging/build_app.py

## What this is not

Not a self-contained, distributable bundle. It references the venv at
~/.venvs/utter rather than freezing it, for two reasons that are worth being
explicit about instead of discovering later:

  * MLX ships Metal shader libraries, and freezing those with PyInstaller or
    py2app is its own project. The bundle would also be well over 2 GB before
    a single model was downloaded.
  * Distributing to anyone else needs Apple notarisation, which needs a paid
    Developer account. Without it macOS Gatekeeper refuses the app on any
    machine but this one, whatever we build.

So: this solves "stop making me keep a terminal open", which is the actual
request. A distributable build is a separate piece of work and the blockers
above are the reason it is not five more minutes of effort.

## The signature is the part that matters

macOS ties Accessibility and Microphone permission to a code signature. Ad-hoc
signing (`codesign -s -`) produces a *different* signature every build, so
macOS treats each rebuild as a new application and every permission the author
granted disappears — silently, since a hotkey without Accessibility receives
nothing and raises nothing.

The fix is a self-signed certificate that persists. `--make-cert` creates one;
after that every build carries the same identity and permissions survive.
"""

from __future__ import annotations

import argparse
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
APP_NAME = "Utter"
BUNDLE_ID = "com.dylan.utter"  # matches the Keychain service already in use
CERT_NAME = "Utter Self-Signed"
#: Not a secret — it only protects the PKCS12 file for the seconds it exists
#: in a temp directory. `security import` refuses an empty one.
_P12_PASSPHRASE = "utter"
DEFAULT_DEST = Path("/Applications")
VENV_PYTHON = Path.home() / ".venvs" / "utter" / "bin" / "python"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


# --- the certificate ----------------------------------------------------------


def signing_identity() -> str | None:
    """The persistent certificate, if it has been created."""
    try:
        listing = run(["security", "find-identity", "-v", "-p", "codesigning"]).stdout
    except subprocess.CalledProcessError:
        return None
    for line in listing.splitlines():
        if CERT_NAME in line:
            return CERT_NAME
    return None


def make_certificate() -> int:
    """Create a self-signed code-signing certificate in the login keychain.

    Scripted rather than clicked through Keychain Access because the steps are
    fiddly and getting one of them wrong produces a certificate that signs but
    does not persist. macOS may ask for the login password — that is the
    keychain, not us.
    """
    if signing_identity():
        print(f"「{CERT_NAME}」已经存在，不用重建。")
        return 0

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        key, crt = tmp / "utter.key", tmp / "utter.crt"
        config = tmp / "openssl.cnf"
        config.write_text(
            "[req]\ndistinguished_name=dn\nx509_extensions=v3\nprompt=no\n"
            f"[dn]\nCN={CERT_NAME}\n"
            "[v3]\nbasicConstraints=critical,CA:false\n"
            "keyUsage=critical,digitalSignature\n"
            "extendedKeyUsage=critical,codeSigning\n"
        )
        run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(crt),
            "-days", "3650", "-config", str(config),
        ])

        # Three details, each of which produced a different unhelpful failure
        # before it was right:
        #
        #  * -keypbe/-certpbe/-macalg: OpenSSL 3 defaults to encryption macOS's
        #    `security` cannot read, and it reports that as "MAC verification
        #    failed (wrong password?)" — which sends you looking at passwords.
        #  * a non-empty passphrase: `security import` rejects an empty one
        #    with the same misleading message.
        #  * add-trusted-cert: without it the certificate imports fine and
        #    `find-identity -p codesigning` still reports zero identities,
        #    because an untrusted certificate cannot sign.
        p12 = tmp / "utter.p12"
        run([
            "openssl", "pkcs12", "-export", "-inkey", str(key), "-in", str(crt),
            "-out", str(p12), "-passout", f"pass:{_P12_PASSPHRASE}",
            "-keypbe", "PBE-SHA1-3DES", "-certpbe", "PBE-SHA1-3DES", "-macalg", "sha1",
        ])
        keychain = Path.home() / "Library/Keychains/login.keychain-db"
        run([
            "security", "import", str(p12), "-k", str(keychain),
            "-P", _P12_PASSPHRASE, "-T", "/usr/bin/codesign", "-A",
        ])
        # User trust domain, not admin: no sudo, no password prompt.
        run([
            "security", "add-trusted-cert", "-r", "trustRoot", "-p", "codeSign",
            "-k", str(keychain), str(crt),
        ])

    if not signing_identity():
        print("证书建好了但 codesign 看不到它，可能需要在「钥匙串访问」里把它设为「始终信任」。")
        return 1
    print(f"✓ 「{CERT_NAME}」已创建。以后每次重建都用同一个签名，权限不会再丢。")
    return 0


# --- the bundle ---------------------------------------------------------------


def build(dest_dir: Path, sign: bool = True) -> Path:
    if not VENV_PYTHON.exists():
        raise SystemExit(f"找不到 {VENV_PYTHON} —— 先按 CLAUDE.md 建好环境。")

    app = dest_dir / f"{APP_NAME}.app"
    contents = app / "Contents"
    macos, resources = contents / "MacOS", contents / "Resources"

    if app.exists():
        shutil.rmtree(app)
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)

    (contents / "Info.plist").write_bytes(plistlib.dumps({
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": _version(),
        "CFBundleVersion": _version(),
        # Menu-bar only: no Dock icon, no ⌘Tab entry. Utter is a tool you talk
        # to, not a window you switch to.
        "LSUIElement": True,
        "LSMinimumSystemVersion": "13.0",
        # macOS shows this in the microphone prompt. A blank one reads as
        # sinister; this one says why.
        "NSMicrophoneUsageDescription":
            "Utter 需要麦克风来把你说的话转成文字。音频只在内存里存在几秒，转完即弃，不会写入任何文件。",
        "NSAppleEventsUsageDescription":
            "Utter 用它把文字粘贴到你正在编辑的窗口。",
    }))

    # The launcher, with the interpreter path compiled in.
    source = REPO / "packaging" / "launcher.c"
    run([
        "clang", "-O2", "-arch", _arch(),
        f'-DUTTER_PYTHON="{VENV_PYTHON}"',
        "-o", str(macos / APP_NAME), str(source),
    ])

    # PYTHONPATH via a .pth-style shim would be fragile; the launcher runs
    # `python -m backend.app`, so the repo has to be importable. It is:
    # `pip install -e .` put a link to it on the venv's path.
    (resources / "README.txt").write_text(
        f"Utter.app runs `{VENV_PYTHON} -m backend.app`.\n"
        f"Source: {REPO}\n"
        f"Log:    ~/Utter/utter.log\n"
        "This bundle is not self-contained; see packaging/build_app.py.\n"
    )

    identity = signing_identity() if sign else None
    if identity:
        run(["codesign", "--force", "--sign", identity, "--timestamp=none",
             "--options", "runtime", str(app)])
        print(f"✓ 用「{identity}」签名，权限不会因重建而丢失。")
    else:
        run(["codesign", "--force", "--sign", "-", str(app)])
        print(
            "⚠ 用的是 ad-hoc 签名。每次重建 macOS 都会当成一个新 app，\n"
            "  辅助功能权限要重新授权一次。跑一次这个就好了：\n"
            f"    {VENV_PYTHON} packaging/build_app.py --make-cert"
        )
    return app


def _version() -> str:
    try:
        return run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]).stdout.strip()
    except Exception:
        return "0"


def _arch() -> str:
    import platform

    return "arm64" if platform.machine() == "arm64" else "x86_64"


# --- login item ---------------------------------------------------------------


LAUNCH_AGENT = Path.home() / "Library/LaunchAgents/com.dylan.utter.plist"


def install_login_item(app: Path) -> None:
    """Start Utter at login, via a LaunchAgent.

    A LaunchAgent rather than a Login Item because it restarts the app if it
    dies, and because it can be removed with one file deletion rather than a
    trip through System Settings.
    """
    LAUNCH_AGENT.parent.mkdir(parents=True, exist_ok=True)
    LAUNCH_AGENT.write_bytes(plistlib.dumps({
        "Label": BUNDLE_ID,
        "ProgramArguments": [str(app / "Contents/MacOS" / APP_NAME)],
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "StandardErrorPath": str(Path.home() / "Utter" / "launchd.log"),
    }))
    subprocess.run(["launchctl", "unload", str(LAUNCH_AGENT)],
                   capture_output=True, text=True)
    subprocess.run(["launchctl", "load", str(LAUNCH_AGENT)],
                   capture_output=True, text=True)
    print(f"✓ 开机自启已开启（{LAUNCH_AGENT}）")


def remove_login_item() -> None:
    subprocess.run(["launchctl", "unload", str(LAUNCH_AGENT)],
                   capture_output=True, text=True)
    LAUNCH_AGENT.unlink(missing_ok=True)
    print("✓ 开机自启已关闭")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Utter.app")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--make-cert", action="store_true",
                        help="create the persistent signing certificate, once")
    parser.add_argument("--login-item", action="store_true", help="start at login")
    parser.add_argument("--no-login-item", action="store_true")
    parser.add_argument("--no-sign", action="store_true")
    args = parser.parse_args()

    if args.make_cert:
        return make_certificate()
    if args.no_login_item:
        remove_login_item()
        return 0

    app = build(args.dest, sign=not args.no_sign)
    print(f"✓ 建好了：{app}")
    if args.login_item:
        install_login_item(app)
    print(f"\n打开它：open '{app}'\n日志：  ~/Utter/utter.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
