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


def microphone_status() -> str:
    """notDetermined / denied / authorized, as macOS sees this bundle.

    Worth its own function because the failure it detects is invisible: an app
    without microphone permission does not get an error when it opens a stream.
    It gets **exact zeros**, forever. The author pressed the hotkey and Utter
    said "no speech" — which was true, and said nothing about why.

    A real microphone in a silent room returns room tone around 0.001. A peak
    of precisely 0.0000 over six seconds is not a quiet room; it is macOS
    declining.
    """
    try:
        import AVFoundation as AV

        return {
            0: "未询问", 1: "受限", 2: "已拒绝", 3: "已授权",
        }.get(AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeAudio), "?")
    except Exception:  # pragma: no cover - bindings missing
        return "查不到"


def request_microphone(log) -> bool:
    """Ask for the microphone, and wait for the answer.

    Unlike Accessibility, this one *can* be prompted for — but only if
    something asks. Opening a PortAudio stream does not ask: it just receives
    silence. So Utter has to ask explicitly, once, at startup.
    """
    import threading

    try:
        import AVFoundation as AV
    except ImportError:  # pragma: no cover
        return True  # cannot check; let the recording try

    status = AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeAudio)
    if status == 3:
        return True
    if status in (1, 2):
        log.error("microphone permission denied for this bundle")
        alert(
            "Utter 没有麦克风权限",
            "macOS 拒绝麦克风时不会报错，只会一直给静音 —— 所以听写会说"
            "「没听到」，而其实是权限的问题。\n\n"
            "去「系统设置 → 隐私与安全性 → 麦克风」，把 Utter 打开。",
            settings_url="x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
        )
        return False

    log.info("requesting microphone permission")
    answered = threading.Event()
    granted = {"ok": False}

    def done(ok):
        granted["ok"] = bool(ok)
        answered.set()

    AV.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AV.AVMediaTypeAudio, done)
    answered.wait(120)
    log.info("microphone permission: %s", "granted" if granted["ok"] else "refused")
    return granted["ok"]


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
        print(f"microphone       {microphone_status()}")
        if not trusted:
            print(reason)
        return 0

    if "--probe" in argv:
        # The bare minimum status item, run from inside the bundle. Terminal
        # and bundle behave differently and this is the only way to see which.
        import threading, time

        import AppKit

        app = AppKit.NSApplication.sharedApplication()
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
        item = AppKit.NSStatusBar.systemStatusBar().statusItemWithLength_(
            AppKit.NSVariableStatusItemLength)
        def frame():
            w = item.button().window()
            return w.frame().size if w else None

        steps = []
        steps.append(("刚创建", frame()))
        item.button().setTitle_("ZZ")
        steps.append(("设了文字标题", frame()))
        img = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            "waveform", "Utter")
        img.setTemplate_(True)
        item.button().setImage_(img)
        steps.append(("设了图标", frame()))
        item.button().setTitle_("")
        steps.append(("清空文字（menubar.py 就是这样）", frame()))
        m = AppKit.NSMenu.alloc().init()
        m.addItem_(AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("测试", None, ""))
        item.setMenu_(m)
        steps.append(("挂上菜单", frame()))

        if "--with-hotkey" in argv:
            # pynput installs a CGEventTap and spins its own CFRunLoop. It has
            # already killed this process once (two listeners plus a Controller
            # = SIGABRT), so it is the first thing to suspect.
            from backend.hotkey import HotkeyListener

            HotkeyListener(on_event=lambda _e: None, combination="<alt_l>").start()
            steps.append(("启动 pynput 热键监听", frame()))

        def report():
            for label, size in steps:
                print(f"  {label:<32} {size}", flush=True)
            time.sleep(4)
            print(f"  4 秒后                            {frame()}", flush=True)
            AppKit.NSApp().terminate_(None)

        threading.Thread(target=report, daemon=True).start()
        AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(0.2, True, lambda _t: None)
        app.run()
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
    """Put the icon on screen first, then do everything slow behind it.

    Order matters more than it looks. Every version of this before now did the
    slow work first and created the menu bar item last, so double-clicking
    Utter produced nothing on screen for as long as it took — measured at 79
    seconds on a cold page cache, and there is no Dock icon and no window to
    say otherwise. The author reported the app would not open, which was the
    only conclusion the evidence supported.

    Three things are slow and all of them are now behind the run loop:
    resolving the speech provider (a network call to Hugging Face), waiting on
    the Accessibility grant, and warming the model. None of them needs the main
    thread.
    """
    import threading

    import AppKit

    from backend.config import DEFAULT_DIR, load
    from backend.daemon import DictationDaemon
    from backend.instance_lock import InstanceLock
    from backend.menubar import MenuBar
    from backend.overlay import Overlay

    lock = InstanceLock(DEFAULT_DIR / "dictate.pid")
    owner = lock.acquire()
    if owner is not None:
        alert("Utter 已经在运行", owner.message())
        return 1

    config = load()

    # `stt=None` until the provider resolves. The menu reads it with getattr and
    # a default, so it renders fine in the meantime.
    daemon = DictationDaemon(
        config=config,
        stt=None,
        on_text=lambda u: log.info("dictated: %s", u.text),
        overlay=Overlay(),
    )

    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    menu = MenuBar(daemon, on_quit=daemon.stop)
    menu.install()
    menu.set_status("启动中…")

    def bring_up():
        from backend.cli import build_polish
        from backend.providers.stt import NoProviderAvailable, get_stt_provider

        if not _wait_for_accessibility(log):
            menu.set_status("⚠ 没有辅助功能权限，热键不工作")
            return

        menu.set_status("启动中…（正在申请麦克风权限）")
        if not request_microphone(log):
            menu.set_status("⚠ 没有麦克风权限，听不到声音")
            return

        menu.set_status("启动中…（正在找语音模型）")
        try:
            daemon.stt = get_stt_provider(preferred=config.stt_provider)
        except NoProviderAvailable as exc:
            menu.set_status("⚠ 找不到语音模型")
            alert("Utter 找不到可用的语音模型", f"{exc}\n\n日志：{LOG_PATH}")
            return

        daemon.polish = build_polish(config)
        daemon.polish_factory = build_polish

        menu.set_status("启动中…（正在预热模型，首次可能要一分钟）")
        try:
            daemon.start()
        except Exception as exc:
            log.exception("daemon failed to start")
            menu.set_status(f"⚠ 启动失败：{exc}")
            alert("Utter 启动失败", f"{exc}\n\n日志：{LOG_PATH}")
            return

        menu.set_status(None)
        log.info("ready; hotkey=%s", config.hotkey_push)
        _notify(
            "Utter 就绪",
            f"按住 {config.hotkey_push} 说一句；双击 {config.hotkey_toggle} 边说边出字。",
        )

    threading.Thread(target=bring_up, daemon=True, name="utter-start").start()

    import signal

    signal.signal(signal.SIGINT, lambda *_: app.terminate_(None))
    # A no-op timer keeps the run loop responsive to signals; without it AppKit
    # can sit in mach_msg and ignore Ctrl-C entirely.
    AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(0.3, True, lambda _t: None)

    try:
        app.run()
    finally:
        daemon.stop()
        lock.release()
        log.info("Utter.app stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
