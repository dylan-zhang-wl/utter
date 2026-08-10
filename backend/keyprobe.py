"""`utter keys` — find a hotkey that nothing else on this machine has taken.

Why this exists instead of an automatic conflict check.

macOS lets you read its own 35-odd system shortcuts, and nothing else. There is
no API that enumerates the global hotkeys other applications have registered —
each app stores them in its own private format, and many register at runtime
with no on-disk trace at all. On the author's machine that is WeChat, Claude,
Chrome, Obsidian, Word, WPS, Doubao, Gemini, ChatGPT, Zotero and Tencent
Meeting, all running at once.

So a promise of "we will warn you about conflicts" would be a lie. What is
honest, and more useful anyway, is to make testing a candidate key take one
second: press it, and see both what Utter would receive and whether the system
already claims it. Whether WeChat takes it, only pressing it in WeChat will tell.
"""

from __future__ import annotations

import subprocess
import time

from pynput import keyboard

from backend.hotkey import _SIDED, _accessibility_trusted

# Reverse of _SIDED, so a pressed key can be printed as something the user can
# paste straight into config.json.
_SPEC_FOR_KEY = {key: f"<{name}>" for name, key in _SIDED.items()}

# Curated, and honestly incomplete. Everything here was either observed by the
# author or is documented macOS behaviour. There is no way to generate it.
KNOWN_CLAIMS = {
    "<alt_r>": "微信的语音键（作者 2026-08-10 实测冲突）",
    "<alt_l>": "双击被 Claude 占用（作者实测）；单独按住通常没问题",
    "<fn>": "macOS 自己的听写与表情面板",
    "<cmd_l>": "长按会触发部分应用的快捷键面板",
    "<ctrl_l>": "macOS 听写的可选触发键之一",
}


def _system_shortcuts() -> int:
    """How many macOS system shortcuts are enabled. Readable, unlike app ones."""
    try:
        out = subprocess.run(
            ["defaults", "read", "com.apple.symbolichotkeys", "AppleSymbolicHotKeys"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        return out.count("enabled = 1")
    except Exception:
        return 0


def _describe(key) -> tuple[str, str]:
    """(config spec, human label) for a pressed key."""
    if key in _SPEC_FOR_KEY:
        return _SPEC_FOR_KEY[key], str(key).replace("Key.", "")
    if isinstance(key, keyboard.Key):
        return f"<{key.name}>", key.name
    char = getattr(key, "char", None)
    if char:
        return char, f"'{char}'"
    return "", repr(key)


def run(out, seconds: float = 60.0) -> int:
    if not _accessibility_trusted():
        from backend.hotkey import PERMISSION_HINT

        print(PERMISSION_HINT, file=out)
        return 1

    print(
        "按键探测器。按下任意键，看 Utter 收到的是什么。\n"
        f"macOS 系统快捷键当前启用 {_system_shortcuts()} 条（系统级的能查，"
        "别的 app 抢了什么查不到——见下）。\n"
        "\n"
        "  找一个没人用的键，建议这样试：先在这里按，记下它的写法；\n"
        "  再切到微信、Claude、Word 里按同一个键，看有没有反应。\n"
        "  两边都安静的那个就能用。\n"
        "\n"
        f"Ctrl-C 退出（或 {seconds:.0f} 秒后自动结束）。\n",
        file=out,
    )

    seen: dict[str, float] = {}

    def on_press(key):
        spec, label = _describe(key)
        now = time.perf_counter()
        # Key repeat would otherwise scroll the screen while a key is held.
        if spec and now - seen.get(spec, 0.0) < 0.5:
            return
        seen[spec] = now

        note = KNOWN_CLAIMS.get(spec, "")
        warning = f"   ⚠ {note}" if note else ""
        config_hint = f'  →  "{spec}"' if spec else "   (无法用作热键)"
        print(f"  {label:<14}{config_hint}{warning}", file=out)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    try:
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()

    print(
        "\n把选好的写进 ~/Utter/config.json：\n"
        '    "hotkey_push":   "<alt_r>"     按住说，短插入\n'
        '    "hotkey_toggle": "<cmd_r>"     双击开始／再双击停止，长口述\n'
        "\n任一项设成 null 即关闭该手势。",
        file=out,
    )
    return 0
