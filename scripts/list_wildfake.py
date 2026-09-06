"""List WildFake repo files via Playwright (ModelScope CN is JS-heavy)."""

from __future__ import annotations

import argparse
import re

from playwright.sync_api import sync_playwright

BASE = "https://modelscope.cn/datasets/hy2628982280/WildFake"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="", help="subfolder under Files, e.g. Images")
    ap.add_argument("--timeout", type=int, default=60000)
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"{BASE}/summary", wait_until="domcontentloaded", timeout=args.timeout)
        page.wait_for_timeout(4000)

        # Click "Files and versions" tab
        for sel in ["text=Files and versions", "text=Files", "text=文件"]:
            try:
                page.click(sel, timeout=5000)
                break
            except Exception:
                continue
        page.wait_for_timeout(4000)

        # Drill into subfolder if requested (click folder row links)
        if args.path:
            for part in args.path.split("/"):
                try:
                    page.click(f"text={part}", timeout=8000)
                    page.wait_for_timeout(3000)
                except Exception as e:
                    print(f"[warn] could not open {part}: {e}")
                    break

        # Dump visible file rows: name + size-ish text
        body = page.inner_text("body")
        print("===== PAGE TEXT (first 4000 chars) =====")
        print(body[:4000])
        print("===== LINKS =====")
        seen = set()
        for a in page.query_selector_all("a"):
            try:
                txt = (a.inner_text() or "").strip().replace("\n", " ")[:120]
                href = a.get_attribute("href") or ""
            except Exception:
                continue
            if not txt or (txt, href) in seen:
                continue
            seen.add((txt, href))
            if re.search(r"zip|Images|label|split|csv|json|part|SD|GAN|real|fake", txt, re.I) or (
                href and "WildFake" in href
            ):
                print(f"- {txt}  [{href}]")
        print(f"===== URL now: {page.url} =====")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
