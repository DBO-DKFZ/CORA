#!/usr/bin/env python3
"""Driver: run the GP scraper over the 100 most populous German cities.

Processes ONE city at a time (OSM lookup + website email scraping for that
city's practices) and rewrites the combined CSV after every city, so
intermediate results are never lost.

Robustness:
  * Overpass queries are retried with exponential backoff on transient
    failures (504 Gateway Timeout / rate limiting).
  * A randomized delay is inserted BETWEEN cities to stay within the
    Overpass/Nominatim usage policies.
  * A sidecar file (top100_done_cities.txt) records fully-processed cities.
    On resume, those cities are skipped; a city whose Overpass query keeps
    failing is NOT marked done, so it will be retried on the next run.
"""

import csv
import logging
import os
import random
import time

from gp_scraper import (
    Practice,
    search_osm,
    scrape_emails_from_website,
    polite_sleep,
)

log = logging.getLogger("run_top100")

OUTPUT = "top100_gp_emails.csv"
DONE_FILE = "top100_done_cities.txt"
RADIUS_KM = 8.0

INTER_CITY_DELAY = (5.0, 10.0)   # seconds between cities
OVERPASS_RETRIES = 4             # attempts per city before giving up
OVERPASS_BACKOFF = 15.0          # base backoff seconds (×attempt)

# Top 100 German cities by population (approx., descending).
CITIES = [
    "Berlin", "Hamburg", "München", "Köln", "Frankfurt am Main",
    "Stuttgart", "Düsseldorf", "Leipzig", "Dortmund", "Essen",
    "Bremen", "Dresden", "Hannover", "Nürnberg", "Duisburg",
    "Bochum", "Wuppertal", "Bielefeld", "Bonn", "Münster",
    "Mannheim", "Karlsruhe", "Augsburg", "Wiesbaden", "Mönchengladbach",
    "Gelsenkirchen", "Aachen", "Braunschweig", "Chemnitz", "Kiel",
    "Halle (Saale)", "Magdeburg", "Freiburg im Breisgau", "Krefeld", "Mainz",
    "Lübeck", "Erfurt", "Oberhausen", "Rostock", "Kassel",
    "Hagen", "Potsdam", "Saarbrücken", "Hamm", "Ludwigshafen am Rhein",
    "Mülheim an der Ruhr", "Oldenburg", "Osnabrück", "Leverkusen", "Heidelberg",
    "Darmstadt", "Solingen", "Herne", "Neuss", "Regensburg",
    "Paderborn", "Ingolstadt", "Offenbach am Main", "Fürth", "Würzburg",
    "Ulm", "Heilbronn", "Pforzheim", "Wolfsburg", "Göttingen",
    "Bottrop", "Reutlingen", "Koblenz", "Bremerhaven", "Bergisch Gladbach",
    "Erlangen", "Remscheid", "Trier", "Jena", "Salzgitter",
    "Moers", "Siegen", "Hildesheim", "Cottbus", "Gütersloh",
    "Kaiserslautern", "Witten", "Gera", "Iserlohn", "Schwerin",
    "Düren", "Zwickau", "Ratingen", "Esslingen am Neckar", "Marl",
    "Lünen", "Velbert", "Hanau", "Ludwigsburg", "Tübingen",
    "Minden", "Flensburg", "Konstanz", "Worms", "Wilhelmshaven",
]

HEADER = ["Name", "Address", "City", "Postcode", "Phone", "Website", "Emails", "Source"]


def practice_key(p: Practice) -> tuple:
    return (p.name.lower(), p.website.lower(), p.address.lower())


def load_existing() -> tuple[list[list[str]], set]:
    """Return (existing CSV rows, set of practice keys)."""
    rows: list[list[str]] = []
    keys: set = set()
    if not os.path.exists(OUTPUT):
        return rows, keys
    with open(OUTPUT, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if not row:
                continue
            rows.append(row)
            name, address, _city, _pc, _phone, website, _emails, _source = row
            keys.add((name.lower(), website.lower(), address.lower()))
    return rows, keys


def load_done_cities() -> set:
    if not os.path.exists(DONE_FILE):
        return set()
    with open(DONE_FILE, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_done(city: str):
    with open(DONE_FILE, "a", encoding="utf-8") as f:
        f.write(city + "\n")


def write_csv(rows: list[list[str]]):
    """Atomically rewrite the full CSV (temp file + rename)."""
    tmp = OUTPUT + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)
    os.replace(tmp, OUTPUT)


def search_with_retry(city: str) -> tuple[list[Practice], bool]:
    """Query OSM for a city, retrying on transient (empty) results.

    Returns (practices, ok). ok is False only if every attempt came back
    empty — treated as a failure so the city is retried on a later run.
    """
    for attempt in range(1, OVERPASS_RETRIES + 1):
        practices = search_osm(city, radius_km=RADIUS_KM, include_all_doctors=False)
        if practices:
            return practices, True
        if attempt < OVERPASS_RETRIES:
            backoff = OVERPASS_BACKOFF * attempt
            log.warning("  '%s' returned no practices (attempt %d/%d) — "
                        "backing off %.0fs", city, attempt, OVERPASS_RETRIES, backoff)
            time.sleep(backoff)
    return [], False


def main():
    rows, seen_keys = load_existing()
    done_cities = load_done_cities()
    if done_cities:
        log.info("Resuming: %d cities already done, %d rows so far",
                 len(done_cities), len(rows))

    for ci, city in enumerate(CITIES, 1):
        if city in done_cities:
            log.info("[%d/%d] %s — already done, skipping", ci, len(CITIES), city)
            continue

        log.info("[%d/%d] === %s ===", ci, len(CITIES), city)
        practices, ok = search_with_retry(city)
        if not ok:
            log.error("[%d/%d] %s — Overpass kept failing; leaving for a later "
                      "retry and moving on", ci, len(CITIES), city)
            time.sleep(random.uniform(*INTER_CITY_DELAY))
            continue

        new_for_city = 0
        for p in practices:
            key = practice_key(p)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            if not p.emails and p.website:
                p.emails = scrape_emails_from_website(p.website)
                polite_sleep()

            rows.append([
                p.name, p.address, p.city, p.postcode,
                p.phone, p.website, "; ".join(p.emails), p.source,
            ])
            new_for_city += 1

        # Persist after every city so nothing is lost on a crash.
        write_csv(rows)
        mark_done(city)
        with_email = sum(1 for r in rows if r[6])
        log.info("[%d/%d] %s done: +%d new practices, %d rows total (%d with email). Saved.",
                 ci, len(CITIES), city, new_for_city, len(rows), with_email)

        # Be polite to the OSM APIs between cities.
        time.sleep(random.uniform(*INTER_CITY_DELAY))

    log.info("✓ All cities processed. %d total records in %s", len(rows), OUTPUT)


if __name__ == "__main__":
    main()
