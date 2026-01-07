# banner.py
from __future__ import annotations

import os
import sys

_ASCII = r"""
██████╗ ██████╗ ██████╗ ███████╗███╗   ██╗███████╗████████╗
██╔══██╗██╔══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝╚══██╔══╝
██║  ██║██████╔╝██║  ██║█████╗  ██╔██╗ ██║█████╗     ██║
██║  ██║██╔═══╝ ██║  ██║██╔══╝  ██║╚██╗██║██╔══╝     ██║
██████╔╝██║     ██████╔╝██║     ██║ ╚████║███████╗   ██║
╚═════╝ ╚═╝     ╚═════╝ ╚═╝     ╚═╝  ╚═══╝╚══════╝   ╚═╝
"""

def _ansi(code: str) -> str:
    return f"\x1b[{code}m"

RESET = _ansi("0")
BOLD = _ansi("1")
DIM = _ansi("2")

# Light gray in most terminals (ANSI "bright black")
LIGHT_GRAY = _ansi("90")


def print_banner(
    *,
    app: str = "DPDFNet",
    powered_by: str = "CEVA Inc.",
    version: str | None = None,
    use_unicode_box: bool = True,
) -> None:
    """
    Prints a left-aligned ASCII banner in a frame.
    Respects:
      - NO_BANNER=1/true/yes
      - non-TTY stdout (won't pollute logs/pipes)
    """
    if os.getenv("NO_BANNER", "").lower() in {"1", "true", "yes"}:
        return
    if not sys.stdout.isatty():
        return

    title = f"{app}" + (f" v{version}" if version else "")
    tagline = f"Powered by {powered_by}"

    content_lines = [ln.rstrip("\n") for ln in _ASCII.strip("\n").splitlines()]
    content_lines += ["", title, tagline]

    # Frame width is the longest visible line
    inner_w = max(len(ln) for ln in content_lines)

    if use_unicode_box:
        tl, tr, bl, br = "┌", "┐", "└", "┘"
        h, v = "─", "│"
    else:
        tl = tr = bl = br = "+"
        h, v = "-", "|"

    top = f"{tl}{h * (inner_w + 2)}{tr}"
    bot = f"{bl}{h * (inner_w + 2)}{br}"

    framed = [top]
    for ln in content_lines:
        framed.append(f"{v} {ln.ljust(inner_w)} {v}")
    framed.append(bot)

    # Styling: logo block light gray; title bold light gray; tagline dim light gray
    # We'll re-walk lines to apply styles without messing the frame alignment.
    styled = []
    for i, ln in enumerate(framed):
        if i == 0 or i == len(framed) - 1:
            styled.append(f"{LIGHT_GRAY}{ln}{RESET}")
            continue

        # Lines inside frame:
        # content_lines index is i-1
        ci = i - 1
        raw = content_lines[ci]

        if raw == title:
            # replace just the raw part with bold
            styled_ln = ln.replace(raw, f"{BOLD}{LIGHT_GRAY}{raw}{RESET}{LIGHT_GRAY}", 1)
            styled.append(f"{LIGHT_GRAY}{styled_ln}{RESET}")
        elif raw == tagline:
            styled_ln = ln.replace(raw, f"{DIM}{LIGHT_GRAY}{raw}{RESET}{LIGHT_GRAY}", 1)
            styled.append(f"{LIGHT_GRAY}{styled_ln}{RESET}")
        else:
            styled.append(f"{LIGHT_GRAY}{ln}{RESET}")

    print("\n".join(styled) + "\n", flush=True)
