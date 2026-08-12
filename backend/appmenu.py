"""The application menu — the one that makes keyboard shortcuts exist.

A menu-bar app with `LSUIElement` shows no menu bar, so it is easy to conclude
it does not need an `NSMenu`. It does. Key equivalents are dispatched *through*
the main menu, so without one:

  * ⌘W does not close the window and ⌘Q does not quit — the two shortcuts every
    macOS user tries first.
  * **⌘V does not paste.** The settings window has a field for an API key, and
    an API key is always pasted, never typed. Without an Edit menu that field
    silently refuses the one interaction it exists for.

The menu is never drawn, because an accessory app has no menu bar. It is built
entirely so that six key combinations behave the way they do in every other
Mac application.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def _item(menu, title, action, key, *, modifiers=None, target=None):
    import AppKit

    entry = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        title, action, key
    )
    if modifiers is not None:
        entry.setKeyEquivalentModifierMask_(modifiers)
    if target is not None:
        entry.setTarget_(target)
    menu.addItem_(entry)
    return entry


def _submenu(parent, title):
    import AppKit

    holder = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, None, "")
    menu = AppKit.NSMenu.alloc().initWithTitle_(title)
    parent.addItem_(holder)
    parent.setSubmenu_forItem_(menu, holder)
    return menu


def install(open_settings_target=None) -> bool:
    """Build and install the main menu. Returns whether it worked.

    `open_settings_target` is an object responding to `openWindow:`; ⌘, is
    wired to it, because on macOS ⌘, opens settings in every application and
    people press it without thinking.
    """
    try:
        import AppKit

        root = AppKit.NSMenu.alloc().init()

        app_menu = _submenu(root, "Utter")
        _item(app_menu, "关于 Utter", "orderFrontStandardAboutPanel:", "")
        app_menu.addItem_(AppKit.NSMenuItem.separatorItem())
        if open_settings_target is not None:
            _item(app_menu, "设置…", "openWindow:", ",", target=open_settings_target)
            app_menu.addItem_(AppKit.NSMenuItem.separatorItem())
        _item(app_menu, "隐藏 Utter", "hide:", "h")
        _item(app_menu, "退出 Utter", "terminate:", "q")

        # Edit. Without this the API key field cannot be pasted into, which is
        # the only way anyone ever enters an API key.
        edit = _submenu(root, "编辑")
        _item(edit, "撤销", "undo:", "z")
        _item(edit, "重做", "redo:", "Z")
        edit.addItem_(AppKit.NSMenuItem.separatorItem())
        _item(edit, "剪切", "cut:", "x")
        _item(edit, "拷贝", "copy:", "c")
        _item(edit, "粘贴", "paste:", "v")
        _item(edit, "全选", "selectAll:", "a")

        window = _submenu(root, "窗口")
        _item(window, "最小化", "performMiniaturize:", "m")
        # performClose: rather than close: — it goes through windowShouldClose:,
        # which is where the window hides instead of dying.
        _item(window, "关闭", "performClose:", "w")

        AppKit.NSApp().setMainMenu_(root)
        AppKit.NSApp().setWindowsMenu_(window)
        return True
    except Exception:
        log.warning("could not install the application menu", exc_info=True)
        return False
