"""Capture real screenshots of the running Gradio app with Playwright.

Launches nothing itself — start the app first (``python app.py``), then:

    python scripts/capture_screenshots.py

Writes PNGs to screenshots/. Requires: ``pip install playwright`` and
``python -m playwright install chromium``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "screenshots"
URL = "http://localhost:7860"

CAESAR_EXAMPLE = (
    "WKH TXLFN EURZQ IRA MXPSV RYHU WKH ODCB GRJ ZKLOH WKH VLJQDO "
    "RIILFHU GHFRGHV WKH LQWHUFHSWHG PHVVDJH DW GDZQ"
)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1400})
        page.goto(URL, wait_until="networkidle", timeout=60000)
        time.sleep(2)

        # --- Detect Mode with a real result ---
        page.get_by_role("tab", name="Detect Mode").click()
        time.sleep(0.5)
        box = page.get_by_placeholder("Example: WKLV LV D FDHVDU FLSKHU...")
        box.click()
        box.fill("")
        # Type a suffix so Gradio's (Svelte) input handler registers the value —
        # a bare .fill() sets the DOM value without firing the input event, so
        # the backend analyzes stale text.
        box.fill(CAESAR_EXAMPLE[:-1])
        box.type(CAESAR_EXAMPLE[-1])
        page.wait_for_timeout(400)
        page.get_by_role("button", name="Analyze like a detective").click()
        # Wait until the report actually contains the decoded plaintext, not a
        # stale/partial state — otherwise we screenshot before the result renders.
        page.wait_for_function(
            "document.body.innerText.includes('THE QUICK BROWN FOX JUMPS')",
            timeout=20000,
        )
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / "01-detect-mode.png"), full_page=True)
        print("captured 01-detect-mode.png")

        # --- The other tabs (UI state) ---
        tabs = {
            "Try Decode": "02-try-decode.png",
            "Solve Substitution": "03-solve-substitution.png",
            "Compare Mode": "04-compare-mode.png",
            "Challenge Mode": "05-challenge-mode.png",
            "About / Model Status": "06-about.png",
        }
        for tab_name, fname in tabs.items():
            try:
                page.get_by_role("tab", name=tab_name).click()
                page.wait_for_timeout(1200)
                page.screenshot(path=str(OUT / fname), full_page=True)
                print(f"captured {fname}")
            except Exception as exc:  # noqa: BLE001
                print(f"skip {tab_name}: {exc}", file=sys.stderr)

        browser.close()


if __name__ == "__main__":
    main()
