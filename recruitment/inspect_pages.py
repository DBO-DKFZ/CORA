#!/usr/bin/env python3
"""
Quick inspector – dumps HTML snippets so we can find the right selectors.
Run this on your machine where Jameda/Doctolib are reachable.
"""
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def inspect(name, url):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  {url}")
    print('='*60)
    r = requests.get(url, headers=HEADERS, timeout=20)
    print(f"Status: {r.status_code}   Final URL: {r.url}")
    soup = BeautifulSoup(r.text, "lxml")

    # Print all unique tag+class combos that appear more than once
    # (these are likely the repeated card elements)
    from collections import Counter
    tags = Counter()
    for tag in soup.find_all(True):
        cls = " ".join(tag.get("class", []))[:60]
        if cls:
            tags[f"<{tag.name} class='{cls}'>"] += 1

    print("\nTop repeated elements (likely card containers):")
    for sig, count in tags.most_common(20):
        if count >= 3:
            print(f"  {count:3d}x  {sig}")

    # Also dump first 4000 chars of raw HTML for manual inspection
    print("\n--- RAW HTML (first 4000 chars) ---")
    print(r.text[:4000])

    # Save full HTML to file
    fname = name.replace(" ", "_").replace("/", "_") + ".html"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(r.text)
    print(f"\n  Full HTML saved to: {fname}")

inspect("Jameda Heidelberg",
        "https://www.jameda.de/arzte/?s=Heidelberg&fachrichtung=Allgemeinmedizin&page=1")

inspect("Doctolib Heidelberg",
        "https://www.doctolib.de/allgemeinmedizin/heidelberg")

