#!/usr/bin/env python3
"""
GP Email Scraper for German Hausarzt Practices
------------------------------------------------
Strategy:
  1. For each location (city name or postcode), geocode it via the OSM
     Nominatim API to get a lat/lon centre point.
  2. Query the OSM Overpass API for general-practitioner practices
     (amenity=doctors, healthcare:speciality~general) within a radius.
     OSM exposes the practice name, address, phone, website and — for
     ~10-40% of practices — the email address directly.
  3. For practices without a tagged email, visit their website and
     extract the email from the homepage / Impressum / Kontakt page.
  4. Save results to CSV.

Why OSM instead of Jameda/Doctolib:
  Jameda's old search URL now 404s (merged into Docplanner) and Doctolib
  is a booking platform that deliberately does NOT publish practice emails.
  OSM is free, needs no API key, is ToS-clean, and tags website/email
  directly — which is exactly what we need.

Usage:
    python gp_scraper.py --locations "Heidelberg" "69115" "Mannheim" --output results.csv
"""

import argparse
import csv
import re
import time
import random
import logging
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE
)

# A descriptive User-Agent with a contact address is required by both
# Nominatim and Overpass usage policies (and avoids 406/429 responses).
HEADERS = {
    "User-Agent": (
        "gp-recruitment-research/1.0 "
        "(tirtha.chanda@dkfz-heidelberg.de; DKFZ study recruitment)"
    ),
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
}

# Domains that commonly appear in emails but aren't real practice contacts
SPAM_DOMAINS = {
    "example.com", "example.de", "domain.de", "beispiel.de", "beispiel.com",
    "sentry.io", "w3.org", "schema.org", "google.com", "google.de",
    "facebook.com", "instagram.com", "fonts.gstatic.com", "jquery.com",
    "cloudflare.com", "wordpress.com", "wordpress.org", "support.microsoft.com",
    # Platform / CMS / tracking boilerplate that appears on practice sites
    "wixpress.com", "sentry.wixpress.com", "wix.com", "doctolib.de",
    "doctolib.fr", "osmfoundation.org", "one.com", "de.one.com", "jimdo.com",
}

# Substrings that mark an address as machine boilerplate, not a real contact
SPAM_SUBSTRINGS = ("sentry", "wixpress", "@2x.", "@3x.")

# File/asset extensions wrongly captured by the email regex (e.g. foo@2x.jpg)
ASSET_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp",
    ".css", ".js", ".json", ".ico", ".woff", ".woff2", ".ttf",
)

DELAY_RANGE = (1.0, 2.5)  # seconds between website requests – be polite

# OSM endpoints
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class Practice:
    name: str
    address: str
    city: str
    postcode: str
    phone: str = ""
    website: str = ""
    emails: list[str] = field(default_factory=list)
    source: str = ""


# ── HTTP helper ───────────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def get(url: str, timeout: int = 15) -> Optional[requests.Response]:
    try:
        r = SESSION.get(url, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return r
    except Exception as e:
        log.debug("GET %s failed: %s", url, e)
        return None


def polite_sleep():
    time.sleep(random.uniform(*DELAY_RANGE))


# ── OSM: geocode + Overpass practice search ───────────────────────────────────
def geocode(location: str) -> Optional[tuple[float, float]]:
    """Resolve a city name or postcode to (lat, lon) via Nominatim."""
    params = {
        "q": f"{location}, Germany",
        "format": "json",
        "limit": "1",
        "countrycodes": "de",
    }
    try:
        r = SESSION.get(NOMINATIM_URL, params=params, timeout=20)
        r.raise_for_status()
        results = r.json()
        if not results:
            log.warning("  Could not geocode '%s'", location)
            return None
        lat, lon = float(results[0]["lat"]), float(results[0]["lon"])
        log.info("  Geocoded '%s' → (%.4f, %.4f)", location, lat, lon)
        return lat, lon
    except Exception as e:
        log.warning("  Geocoding '%s' failed: %s", location, e)
        return None


def _tag(tags: dict, *keys: str) -> str:
    """Return the first non-empty value among the given tag keys."""
    for k in keys:
        v = tags.get(k)
        if v:
            return v
    return ""


def search_osm(location: str, radius_km: float = 8.0,
               include_all_doctors: bool = False) -> list[Practice]:
    """
    Find GP practices near `location` via OSM Overpass.

    By default only practices tagged as general practitioners
    (healthcare:speciality~general) are returned. Set include_all_doctors
    to also pull untagged/other doctors (broader but noisier).
    """
    coords = geocode(location)
    if not coords:
        return []
    lat, lon = coords
    radius_m = int(radius_km * 1000)

    speciality_filter = "" if include_all_doctors else '["healthcare:speciality"~"general"]'
    query = f"""
    [out:json][timeout:60];
    (
      node["amenity"="doctors"]{speciality_filter}(around:{radius_m},{lat},{lon});
      way["amenity"="doctors"]{speciality_filter}(around:{radius_m},{lat},{lon});
    );
    out tags center 400;
    """

    log.info("  Querying OSM Overpass for GPs near '%s' (%.0f km) ...",
             location, radius_km)
    try:
        r = SESSION.post(OVERPASS_URL, data={"data": query}, timeout=90)
        r.raise_for_status()
        elements = r.json().get("elements", [])
    except Exception as e:
        log.warning("  Overpass query for '%s' failed: %s", location, e)
        return []

    practices = []
    for el in elements:
        tags = el.get("tags", {})
        street = _tag(tags, "addr:street")
        house = _tag(tags, "addr:housenumber")
        addr_line = " ".join(p for p in (street, house) if p)
        email = _tag(tags, "email", "contact:email")

        p = Practice(
            name=_tag(tags, "name", "operator") or "Unknown",
            address=addr_line,
            city=_tag(tags, "addr:city") or location,
            postcode=_tag(tags, "addr:postcode"),
            phone=_tag(tags, "phone", "contact:phone"),
            website=_tag(tags, "website", "contact:website"),
            emails=[email.lower()] if email else [],
            source=f"osm/{location}",
        )
        practices.append(p)

    n_email = sum(1 for p in practices if p.emails)
    n_site = sum(1 for p in practices if p.website)
    log.info("  Found %d practices near '%s' (%d with email tag, %d with website)",
             len(practices), location, n_email, n_site)
    return practices


# ── Email extraction from practice websites ───────────────────────────────────
CONTACT_PATH_HINTS = ["kontakt", "contact", "impressum", "imprint", "ueber-uns", "team"]

def extract_emails_from_page(html: str, base_url: str) -> list[str]:
    """Find all plausible email addresses in an HTML page."""
    # 1. mailto: links (most reliable)
    soup = BeautifulSoup(html, "lxml")
    emails = set()
    for a in soup.select("a[href^='mailto:']"):
        addr = a["href"].replace("mailto:", "").split("?")[0].strip()
        if addr:
            emails.add(addr.lower())

    # 2. Regex over visible text + raw HTML (catches obfuscated ones)
    for m in EMAIL_RE.finditer(html):
        emails.add(m.group(0).lower())

    # Filter junk
    return [
        e for e in emails
        if not any(e.endswith("@" + d) or e.endswith("." + d) for d in SPAM_DOMAINS)
        and not any(s in e for s in SPAM_SUBSTRINGS)
        and not e.endswith(ASSET_EXTENSIONS)
        and "." in e.split("@")[-1]
        and len(e) < 80
    ]


def scrape_emails_from_website(url: str) -> list[str]:
    """Visit a practice website, try homepage + contact page, collect emails."""
    if not url:
        return []

    all_emails: set[str] = set()

    # Homepage
    r = get(url)
    if not r:
        return []
    all_emails.update(extract_emails_from_page(r.text, url))

    # If no email found yet, hunt for a contact/impressum sub-page
    if not all_emails:
        soup = BeautifulSoup(r.text, "lxml")
        for a in soup.select("a[href]"):
            href = a.get("href", "").lower()
            if any(hint in href for hint in CONTACT_PATH_HINTS):
                full_url = urljoin(url, a["href"])
                # Stay on same domain
                if urlparse(full_url).netloc == urlparse(url).netloc:
                    polite_sleep()
                    r2 = get(full_url)
                    if r2:
                        all_emails.update(extract_emails_from_page(r2.text, full_url))
                    break  # one contact page is enough

    return list(all_emails)


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run(locations: list[str], output: str, radius_km: float,
        include_all_doctors: bool):
    all_practices: list[Practice] = []

    for loc in locations:
        all_practices.extend(
            search_osm(loc, radius_km=radius_km,
                       include_all_doctors=include_all_doctors)
        )
        polite_sleep()

    # De-duplicate practices that appear for overlapping locations
    seen: set = set()
    unique: list[Practice] = []
    for p in all_practices:
        key = (p.name.lower(), p.website.lower(), p.address.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    all_practices = unique

    log.info("Total unique practices found: %d", len(all_practices))
    log.info("Now visiting practice websites to fill in missing emails …")

    for i, p in enumerate(all_practices, 1):
        if p.emails:
            continue  # already have an OSM-tagged email
        if p.website:
            log.info("  [%d/%d] %s → %s", i, len(all_practices), p.name, p.website)
            p.emails = scrape_emails_from_website(p.website)
            log.info("         Emails: %s", p.emails or "(none found)")
            polite_sleep()
        else:
            log.debug("  [%d/%d] %s – no website listed", i, len(all_practices), p.name)

    # Write CSV
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Address", "City", "Postcode", "Phone", "Website", "Emails", "Source"])
        for p in all_practices:
            writer.writerow([
                p.name,
                p.address,
                p.city,
                p.postcode,
                p.phone,
                p.website,
                "; ".join(p.emails),
                p.source,
            ])

    log.info("✓ Saved %d records to %s", len(all_practices), output)

    # Summary
    with_email = [p for p in all_practices if p.emails]
    log.info("  Practices with at least one email: %d / %d", len(with_email), len(all_practices))


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape GP emails from OpenStreetMap + practice websites")
    parser.add_argument(
        "--locations", nargs="+", required=True,
        metavar="LOC",
        help="Postcodes or city names to search, e.g. --locations Heidelberg Mannheim 69115",
    )
    parser.add_argument(
        "--output", default="gp_emails.csv",
        help="Output CSV file path (default: gp_emails.csv)",
    )
    parser.add_argument(
        "--radius-km", type=float, default=8.0,
        help="Search radius around each location in km (default: 8)",
    )
    parser.add_argument(
        "--all-doctors", action="store_true",
        help="Include all doctors, not just those tagged as GPs (broader, noisier)",
    )
    args = parser.parse_args()
    run(
        locations=args.locations,
        output=args.output,
        radius_km=args.radius_km,
        include_all_doctors=args.all_doctors,
    )
