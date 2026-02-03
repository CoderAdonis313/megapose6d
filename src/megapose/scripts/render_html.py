#!/usr/bin/env python3
import argparse
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import pathname2url


def file_url(path: Path) -> str:
    # Safe file:// URL (handles spaces, etc.)
    return urljoin("file:", pathname2url(str(path.resolve())))


def find_firefox() -> str:
    firefox = shutil.which("firefox")
    if not firefox:
        raise SystemExit(
            "Firefox not found on PATH. Install Firefox or ensure `firefox` is available."
        )
    return firefox


def html_to_png_firefox(firefox_path: str, html_path: Path, png_path: Path, timeout: int = 30) -> None:
    # Firefox screenshots the rendered page (viewport). Output path must end with .png.
    url = file_url(html_path)
    cmd = [
        firefox_path,
        "--headless",
        "--screenshot",
        "--window-size=720,1280",
        str(png_path),
        url,
    ]
    subprocess.run(cmd, check=True, timeout=timeout)


def main():
    ap = argparse.ArgumentParser(description="Convert all .html/.htm files in a folder to .png using Firefox headless.")
    ap.add_argument("folder", help="Folder containing .html files")
    ap.add_argument("--timeout", type=int, default=30, help="Per-file timeout in seconds (default: 30)")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")

    firefox_path = find_firefox()

    html_files = sorted(folder.glob("*.html")) + sorted(folder.glob("*.htm"))
    if not html_files:
        print(f"No .html/.htm files found in: {folder}")
        return

    for html in html_files:
        png = html.with_suffix(".png")
        try:
            print(f"[firefox] {html.name} -> {png.name}")
            html_to_png_firefox(firefox_path, html, png, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout: {html.name}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed: {html.name} (exit={e.returncode})")

    print("Done.")


if __name__ == "__main__":
    main()
