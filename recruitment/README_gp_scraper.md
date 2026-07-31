# GP Email Scraper – German Hausärzte

Scrapes practice websites found via **Jameda** and/or **Doctolib** for email
addresses, given a list of postcodes or city names.

## Setup

```bash
pip install requests beautifulsoup4 lxml
```

## Usage

```bash
# Basic – search Heidelberg and Mannheim
python gp_scraper.py --locations Heidelberg Mannheim --output results.csv

# By postcode
python gp_scraper.py --locations 69115 69120 68161 --output results.csv

# Mix of both, only use Jameda, scrape up to 5 pages per location
python gp_scraper.py \
  --locations Heidelberg Mannheim Karlsruhe Freiburg \
  --sources jameda \
  --max-pages 5 \
  --output gp_emails.csv
```

## Output CSV columns

| Column   | Description                              |
|----------|------------------------------------------|
| Name     | Doctor/practice name                     |
| Address  | Street address from directory listing    |
| City     | Search location used                     |
| Postcode | Postcode (if available)                  |
| Phone    | Phone number from listing                |
| Website  | Practice website URL                     |
| Emails   | Semicolon-separated extracted emails     |
| Source   | Which directory + location it came from  |

## How it works

1. **Directory search** – For each location, Jameda and/or Doctolib are queried
   for Allgemeinmedizin/Hausarzt listings. Practice cards are parsed for name,
   address, phone, and any direct website link.

2. **Email extraction** – For each practice with a website, the script visits
   the homepage. If no email is found there, it follows the first link whose
   path contains `kontakt`, `impressum`, or `contact`. Emails are extracted
   from `mailto:` links first, then via regex over the raw HTML.

3. **Rate limiting** – A random 1.5–3.5 second delay is inserted between every
   HTTP request so as not to overload servers.

## Notes & caveats

- **Many practices don't list their website** on Jameda/Doctolib, in which case
  no email can be scraped. For those rows, `Website` and `Emails` will be blank.
  You may want to manually look up the practice name + city on Google.

- **Impressum emails** (required by German law) are the most reliable — they're
  usually the practice email and the script specifically hunts for `/impressum`
  pages.

- **Anti-bot measures** – Jameda and Doctolib occasionally return CAPTCHAs or
  block scrapers. If you get many 403s, try adding a longer `--delay` or running
  at a different time.

- **Legal note** – Email addresses from Impressum pages are publicly disclosed
  for contact purposes under German TMG § 5. Academic recruitment emails to
  listed addresses are generally permissible, but keep a record of opt-outs
  and comply with DSGVO (include an opt-out in your email).
