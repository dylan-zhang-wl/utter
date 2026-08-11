"""The .app entry point — Utter with no terminal.

`utter dictate` is the same daemon, but it dies with the window it was launched
from and prints to a terminal the author has to keep open. This module is what
Utter.app runs instead: same daemon, output to a log file, and a first-run
check that explains what macOS still needs granted rather than failing silently.

## Everything that can fail before there is a UI

A menu-bar app that exits during startup leaves nothing on screen at all — no
window, no icon, no error. So every failure here ends in a dialog: NSAlert if
AppKit is up, and a log file either way. The one thing this must never do is
quit quietly.

## The permissions

Two, and they behave differently. The microphone prompts itself, once, the
first time a stream opens. Accessibility does not prompt at all — it has to be
granted by hand in System Settings, and until it is, the hotkey silently
receives nothing (backend/hotkey.py). So the first run offers to open the
right settings pane, which is not the one with the same name.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path

LOG_PATH = Path.home() / "Utter" / "utter.log"

#: Module level, because _notify and alert are called from paths that have no
#: local `log` — the first version referenced one and would have raised
#: NameError inside the very handler meant to keep startup from failing silently.
log = logging.getLogger("utter.app")


def _set_up_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )


def alert(title: str, message: str, *, settings_url: str | None = None) -> None:
    """Say something the user can act on, with no terminal to say it in."""
    try:
        import AppKit

        panel = AppKit.NSAlert.alloc().init()
        panel.setMessageText_(title)
        panel.setInformativeText_(message)
        panel.addButtonWithTitle_("打开设置" if settings_url else "好")
        if settings_url:
            panel.addButtonWithTitle_("以后再说")
        AppKit.NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        chosen = panel.runModal()
        if settings_url and chosen == AppKit.NSAlertFirstButtonReturn:
            AppKit.NSWorkspace.sharedWorkspace().openURL_(
                AppKit.NSURL.URLWithString_(settings_url)
            )
    except Exception:  # pragma: no cover - no window server
        print(f"{title}\n{message}", file=sys.stderr)


def _notify(title: str, body: str) -> None:
    """A Notification Centre banner. Non-blocking, unlike alert()."""
    import subprocess

    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification {body!r} with title {title!r}'],
            check=False, capture_output=True, timeout=10,
        )
    except Exception:  # pragma: no cover
        log.debug("could not post a notification", exc_info=True)


def bundle_identity() -> str | None:
    """What macOS thinks this process is.

    The whole reason Contents/MacOS/Utter is a compiled stub that execs Python
    rather than a shell script: permissions are granted to a bundle identifier,
    and if this returns None then they would be granted to whatever launched
    us — which is how the author's Accessibility grant ended up on Terminal.
    """
    try:
        import AppKit

        return AppKit.NSBundle.mainBundle().bundleIdentifier()
    except Exception:  # pragma: no cover
        return None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    _set_up_logging()

    if "--check" in argv:
        # Diagnostics that only make sense from inside the bundle.
        from backend.hotkey import HotkeyListener

        trusted, reason = HotkeyListener(on_event=lambda _e: None).is_available()
        print(f"bundle id        {bundle_identity()}")
        print(f"executable       {sys.executable}")
        print(f"resources        {os.environ.get('UTTER_APP_RESOURCES', '(unset)')}")
        print(f"accessibility    {'granted' if trusted else 'NOT granted'}")
        if not trusted:
            print(reason)
        return 0

    log.info("Utter.app starting; bundle=%s python=%s", bundle_identity(), sys.executable)

    try:
        return _run(log)
    except Exception:
        # A menu-bar app that dies during startup leaves nothing on screen.
        log.exception("startup failed")
        alert(
            "Utter 启动失败",
            f"{traceback.format_exc(limit=3)}\n\n完整日志：{LOG_PATH}",
        )
        return 1


def _wait_for_accessibility(log) -> bool:
    """Ask for the permission, then wait for it. Do not exit.

    The first build did exit — checked the permission, showed a dialog,
    returned 1. From the Finder that is indistinguishable from the app not
    working at all: no window, no Dock icon, nothing on screen, and the log is
    somewhere the author has no reason to look.

    Worse, the advice in the dialog was wrong. It said to relaunch after
    granting, because backend/hotkey.py says the permission is read once at
    launch. That is true of the *event tap*, not of AXIsProcessTrusted, which
    updates the moment the switch is flipped. So the app can simply wait, and
    start the listener when the answer changes.
    """
    import time

    from backend.hotkey import SETTINGS_URL, HotkeyListener

    probe = HotkeyListener(on_event=lambda _e: None)
    if probe.is_available()[0]:
        return True

    log.warning("accessibility not granted; waiting")
    alert(
        "Utter 需要「辅助功能」权限",
        "没有它，热键收不到任何按键 —— 而且不会报任何错。\n\n"
        "在打开的设置页里，把名单中的 Utter 打开。\n"
        "注意不是系统设置里那个同名的「辅助功能」功能页。\n\n"
        "打开开关后回到这里，Utter 会自己就绪，不用重开。",
        settings_url=SETTINGS_URL,
    )

    # Ten minutes is long enough to find the pane, short enough that a
    # forgotten app does not sit spinning forever.
    for _ in range(300):
        if probe.is_available()[0]:
            log.info("accessibility granted; continuing")
            return True
        time.sleep(2)
    log.error("accessibility still not granted after 10 minutes; giving up")
    alert("Utter 没等到授权", "一直没拿到「辅助功能」权限，先退出了。授权后重新打开就行。")
    return False


def _run(log) -> int:
    from backend.config import DEFAULT_DIR, load
    from backend.instance_lock import InstanceLock

    lock = InstanceLock(DEFAULT_DIR / "dictate.pid")
    owner = lock.acquire()
    if owner is not None:
        alert("Utter 已经在运行", owner.message())
        return 1

    config = load()

    # Accessibility never prompts on its own, and without it the hotkey
    # receives nothing and raises nothing.
    if not _wait_for_accessibility(log):
        lock.release()
        return 1

    from backend.cli import build_polish
    from backend.daemon import DictationDaemon
    from backend.overlay import Overlay
    from backend.providers.stt import NoProviderAvailable, get_stt_provider

    try:
        stt = get_stt_provider(preferred=config.stt_provider)
    except NoProviderAvailable as exc:
        alert("Utter 找不到可用的语音模型", f"{exc}\n\n日志：{LOG_PATH}")
        lock.release()
        return 1

    daemon = DictationDaemon(
        config=config,
        stt=stt,
        polish=build_polish(config),
        polish_factory=build_polish,
        on_text=lambda u: log.info("dictated: %s", u.text),
        overlay=Overlay(),
    )

    try:
        daemon.start()
    except Exception as exc:
        alert("Utter 启动失败", f"{exc}\n\n日志：{LOG_PATH}")
        lock.release()
        return 1

    log.info("ready; hotkey=%s", config.hotkey_push)

    # Say so out loud, once.
    #
    # A menu-bar app with no Dock icon and no window gives the user nothing to
    # look at, and if the menu bar is crowded macOS silently hides the icon
    # rather than shrinking anything. The author double-clicked a running,
    # healthy Utter and reported "I can't open it any more", which is the
    # correct conclusion from the evidence they had: nothing appeared.
    _notify(
        "Utter 就绪",
        f"按住 {config.hotkey_push} 说一句；双击 {config.hotkey_toggle} 边说边出字。"
        "图标在菜单栏（波形）。",
    )
    try:
        from backend.cli import _run_with_ui

        _run_with_ui(daemon)
    finally:
        daemon.stop()
        lock.release()
        log.info("Utter.app stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
