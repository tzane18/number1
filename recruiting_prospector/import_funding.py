#!/usr/bin/env python3
"""
import_funding.py — turn funding data into prospect rows.

A fresh Seed / Series A / Series B raise is the single strongest "call them now"
signal (the ICP engine weights it heavily), because the company just got cash and
an urgent mandate to scale headcount. This script normalizes funding data from the
common formats into the CSV shape the rest of the toolkit uses:

  * a Crunchbase (or PitchBook / Tracxn) CSV export  -> --format csv (auto-detected)
  * any funding CSV with sensible headers            -> --format csv
  * a funding-news RSS/Atom feed (file or URL)       -> --format rss

Output columns match prospector.py, so you can either:
  * import it as new companies:   python3 prospector.py import-csv funding.csv
  * OR enrich companies you already have (e.g. from fetch_ats.py) so the funding
    signal lands on the same record as the live hiring signal:
        python3 prospector.py enrich funding.csv

No third-party dependencies — Python 3.8+ standard library only.
"""

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

OUTPUT_FIELDS = [
    "name", "domain", "linkedin_url", "industry", "hq_location",
    "headcount", "headcount_growth_6mo", "funding_stage", "last_funding_date",
    "last_funding_amount_usd", "open_roles", "hard_roles",
    "primary_hiring_function", "in_house_recruiters",
    "on_recruiting_marketplace", "source", "notes",
]


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------

def normalize_name(name):
    """Trim legal suffixes so 'Acme Robotics, Inc.' matches 'Acme Robotics'."""
    if not name:
        return name
    n = str(name).strip()
    n = re.sub(r",?\s*(incorporated|inc\.?|llc|l\.l\.c\.?|ltd\.?|limited|corp\.?|"
               r"corporation|gmbh|s\.a\.?|plc)\s*$", "", n, flags=re.I).strip()
    return n


def normalize_stage(s):
    if not s:
        return None
    t = str(s).strip().lower()
    if "pre" in t and "seed" in t:
        return "Pre-seed"
    if "angel" in t:
        return "Pre-seed"
    if "seed" in t:
        return "Seed"
    m = re.search(r"series\s+([a-k])", t)
    if m:
        letter = m.group(1).lower()
        return f"Series {letter.upper()}" if letter in ("a", "b") else "Series C+"
    if "ipo" in t or "public" in t or "post-ipo" in t:
        return "Public"
    if "bootstrap" in t or "profitable" in t:
        return "Bootstrapped"
    return str(s).strip().title()


def parse_amount_usd(s):
    """'$12M' / '1.5B' / '500K' / '12,000,000' -> int USD."""
    if s is None:
        return None
    t = str(s).strip().lower().replace("$", "").replace(",", "")
    t = t.replace("usd", "").replace("us", "").strip()
    if not t or t in ("-", "n/a", "na", "undisclosed", "unknown"):
        return None
    m = re.match(r"^([\d.]+)\s*(b|bn|billion|m|mm|million|k|thousand)?", t)
    if not m:
        return None
    try:
        num = float(m.group(1))
    except ValueError:
        return None
    unit = (m.group(2) or "").strip()
    mult = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "million": 1e6,
            "b": 1e9, "bn": 1e9, "billion": 1e9}.get(unit, 1)
    return int(round(num * mult))


def parse_headcount(s):
    """'11-50' -> 30 (midpoint); '10001+' -> 10001; '42' -> 42."""
    if not s:
        return None
    t = str(s).strip().replace(",", "")
    m = re.match(r"^(\d+)\s*(?:-|–|to)\s*(\d+)$", t)
    if m:
        return int(round((int(m.group(1)) + int(m.group(2))) / 2))
    m = re.match(r"^(\d+)\s*\+$", t)
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d+)$", t)
    if m:
        return int(m.group(1))
    return None


_DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y",
                 "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y", "%Y-%m"]


def parse_date(s):
    """Return ISO YYYY-MM-DD, or None."""
    if not s:
        return None
    t = str(s).strip()
    from datetime import datetime
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(t, fmt).date().isoformat()
        except ValueError:
            continue
    try:                                   # RSS pubDate, e.g. "Wed, 10 Jun 2026 08:00:00 GMT"
        return parsedate_to_datetime(t).date().isoformat()
    except (TypeError, ValueError, IndexError):
        return None


def domain_from_url(u):
    if not u:
        return None
    t = str(u).strip().lower()
    t = re.sub(r"^https?://", "", t)
    t = re.sub(r"^www\.", "", t)
    t = t.split("/")[0].split("?")[0].strip()
    return t or None


# ---------------------------------------------------------------------------
# CSV import with fuzzy header detection (handles Crunchbase & most exports)
# ---------------------------------------------------------------------------

# canonical field -> alias headers (lowercased). Order = priority.
_HEADER_ALIASES = {
    "name": ["organization name", "company name", "company", "organization",
             "startup", "name"],
    "domain": ["website", "organization name url", "company website", "url",
               "domain", "web"],
    "funding_stage": ["last funding type", "funding stage", "funding type",
                      "last round", "round", "stage"],
    "last_funding_amount_usd": ["last funding amount currency (in usd)",
                                "last funding amount (in usd)", "money raised (in usd)",
                                "total funding amount (in usd)", "amount raised (usd)",
                                "last funding amount", "amount raised", "funding amount",
                                "amount"],
    "last_funding_date": ["last funding date", "announced date", "funding date",
                          "date announced", "date"],
    "headcount": ["number of employees", "employee count", "employees",
                  "company size", "headcount", "size"],
    "hq_location": ["headquarters location", "headquarters", "hq location",
                    "location", "city", "hq"],
    "industry": ["industries", "industry", "categories", "category", "sector"],
}


def _norm_header(h):
    return re.sub(r"\s+", " ", str(h or "").strip().lower())


def detect_columns(fieldnames):
    """Map our canonical fields to the source CSV's actual headers."""
    headers = {_norm_header(h): h for h in fieldnames}
    used, mapping = set(), {}
    # Pass 1: exact alias matches (most reliable).
    for field, aliases in _HEADER_ALIASES.items():
        for alias in aliases:
            if alias in headers and headers[alias] not in used:
                mapping[field] = headers[alias]
                used.add(headers[alias])
                break
    # Pass 2: substring matches for anything still unmapped.
    for field, aliases in _HEADER_ALIASES.items():
        if field in mapping:
            continue
        for alias in aliases:
            for norm, orig in headers.items():
                if orig in used:
                    continue
                if alias in norm:
                    mapping[field] = orig
                    used.add(orig)
                    break
            if field in mapping:
                break
    return mapping


def _row_from_mapping(raw, mapping, source_label):
    name = normalize_name(raw.get(mapping.get("name", ""), "")) if "name" in mapping else ""
    if not name:
        return None
    row = {f: "" for f in OUTPUT_FIELDS}
    row["name"] = name
    if "domain" in mapping:
        row["domain"] = domain_from_url(raw.get(mapping["domain"])) or ""
    if "industry" in mapping:
        row["industry"] = (raw.get(mapping["industry"]) or "").strip()
    if "hq_location" in mapping:
        row["hq_location"] = (raw.get(mapping["hq_location"]) or "").strip()
    if "headcount" in mapping:
        hc = parse_headcount(raw.get(mapping["headcount"]))
        row["headcount"] = hc if hc is not None else ""
    if "funding_stage" in mapping:
        row["funding_stage"] = normalize_stage(raw.get(mapping["funding_stage"])) or ""
    if "last_funding_date" in mapping:
        row["last_funding_date"] = parse_date(raw.get(mapping["last_funding_date"])) or ""
    if "last_funding_amount_usd" in mapping:
        amt = parse_amount_usd(raw.get(mapping["last_funding_amount_usd"]))
        row["last_funding_amount_usd"] = amt if amt is not None else ""
    row["source"] = source_label
    bits = []
    if row["funding_stage"]:
        bits.append(row["funding_stage"])
    if row["last_funding_amount_usd"]:
        bits.append(f"${int(row['last_funding_amount_usd'])/1e6:.1f}M")
    if row["last_funding_date"]:
        bits.append(f"raised {row['last_funding_date']}")
    row["notes"] = "; ".join(bits) + (" (funding import)" if bits else "funding import")
    return row


def parse_csv(text_or_path, source_label="funding:csv", is_path=True):
    fh = open(text_or_path, newline="", encoding="utf-8-sig") if is_path \
        else _StringIO(text_or_path)
    try:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return []
        mapping = detect_columns(reader.fieldnames)
        if "name" not in mapping:
            raise ValueError(f"could not find a company-name column in headers: "
                             f"{reader.fieldnames}")
        rows = []
        for raw in reader:
            r = _row_from_mapping(raw, mapping, source_label)
            if r:
                rows.append(r)
        return rows
    finally:
        fh.close()


# ---------------------------------------------------------------------------
# RSS / Atom funding-feed import (heuristic extraction from headlines)
# ---------------------------------------------------------------------------

_RAISE_VERBS = (r"raises|raised|lands|secures|closes|nets|bags|scores|"
                r"picks up|snags|grabs|announces|pulls in")
_RE_COMPANY = re.compile(r"^\s*(.+?)\s+(?:" + _RAISE_VERBS + r")\b", re.I)
_RE_AMOUNT = re.compile(r"\$\s*([\d.,]+)\s*(billion|million|thousand|bn|mm|[bmk])?\b", re.I)
_RE_STAGE = re.compile(r"(pre-?seed|seed|series\s+[a-k]|angel round|angel)", re.I)


def _strip_ns(tag):
    return tag.split("}", 1)[-1].lower()


def parse_rss(text_or_path, source_label="funding:rss", is_path=True):
    data = open(text_or_path, encoding="utf-8").read() if is_path else text_or_path
    root = ET.fromstring(data)
    items = []
    for el in root.iter():
        if _strip_ns(el.tag) in ("item", "entry"):
            items.append(el)
    rows = []
    for item in items:
        title = pub = summary = link = ""
        for child in item:
            tag = _strip_ns(child.tag)
            txt = (child.text or "").strip()
            if tag == "title":
                title = txt
            elif tag in ("pubdate", "published", "updated", "date"):
                pub = pub or txt
            elif tag in ("description", "summary", "content"):
                summary = summary or txt
            elif tag == "link":
                link = link or (child.get("href") or txt)
        blob = f"{title}. {summary}"
        r = _row_from_headline(title, blob, pub, link, source_label)
        if r:
            rows.append(r)
    return rows


def _row_from_headline(title, blob, pub, link, source_label):
    cm = _RE_COMPANY.search(title) or _RE_COMPANY.search(blob)
    name = normalize_name(cm.group(1)) if cm else None
    if not name or len(name) > 80:
        return None
    row = {f: "" for f in OUTPUT_FIELDS}
    row["name"] = name

    am = _RE_AMOUNT.search(blob)
    if am:
        amt = parse_amount_usd(am.group(1) + (am.group(2) or ""))
        if amt:
            row["last_funding_amount_usd"] = amt
    sm = _RE_STAGE.search(blob)
    if sm:
        row["funding_stage"] = normalize_stage(sm.group(1)) or ""
    d = parse_date(pub)
    if d:
        row["last_funding_date"] = d
    row["source"] = source_label
    bits = [b for b in (row["funding_stage"],
                        f"${int(row['last_funding_amount_usd'])/1e6:.1f}M"
                        if row["last_funding_amount_usd"] else "",
                        f"raised {row['last_funding_date']}"
                        if row["last_funding_date"] else "") if b]
    row["notes"] = "; ".join(bits + ["from funding feed"])
    return row


# small helper so parse_csv can read a string in the self-test
class _StringIO:
    def __init__(self, text):
        import io
        self._io = io.StringIO(text)

    def __iter__(self):
        return iter(self._io)

    def close(self):
        self._io.close()


# ---------------------------------------------------------------------------
def _fetch_url(url):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "recruiting-prospector/1.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.read().decode("utf-8", errors="replace")


def write_output(rows, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} companies -> {out_path}")
    print("\nNext, either:")
    print(f"  python3 prospector.py import-csv {out_path}     # add as new prospects")
    print(f"  python3 prospector.py enrich     {out_path}     # merge into existing companies")


def _detect_format(path):
    p = path.lower()
    if p.endswith((".rss", ".xml", ".atom")):
        return "rss"
    return "csv"


def run(inp, out_path, fmt):
    is_url = str(inp).lower().startswith(("http://", "https://"))
    if fmt == "auto":
        fmt = "rss" if (is_url and "rss" in inp.lower()) else \
              ("rss" if not is_url and _detect_format(inp) == "rss" else "csv")
    if is_url:
        text = _fetch_url(inp)
        rows = parse_rss(text, f"funding:{fmt}", is_path=False) if fmt == "rss" \
            else parse_csv(text, "funding:csv", is_path=False)
    else:
        rows = parse_rss(inp) if fmt == "rss" else parse_csv(inp)
    if not rows:
        sys.exit("No funding rows parsed. Check the file format / headers.")
    write_output(rows, out_path)


# ---------------------------------------------------------------------------
def _selftest():
    checks = []
    # 1) Crunchbase-style CSV export.
    cb = ("Organization Name,Website,Last Funding Type,"
          "Last Funding Amount Currency (in USD),Last Funding Date,"
          "Number of Employees,Headquarters Location,Industries\n"
          "\"Acme Robotics, Inc.\",http://www.acme.co/careers,Series A,"
          "15000000,2026-05-01,11-50,\"San Francisco, California\",Robotics\n"
          "Brightloop,https://brightloop.io,Seed,\"6,000,000\","
          "06/10/2026,1-10,Remote,Data Analytics\n")
    rows = parse_csv(cb, is_path=False)
    r0 = {x["name"]: x for x in rows}
    checks.append(("CB parsed 2 rows", len(rows) == 2))
    checks.append(("CB strips 'Inc.' -> 'Acme Robotics'", "Acme Robotics" in r0))
    checks.append(("CB domain acme.co", r0["Acme Robotics"]["domain"] == "acme.co"))
    checks.append(("CB stage Series A", r0["Acme Robotics"]["funding_stage"] == "Series A"))
    checks.append(("CB amount 15000000", r0["Acme Robotics"]["last_funding_amount_usd"] == 15000000))
    checks.append(("CB headcount midpoint 30", r0["Acme Robotics"]["headcount"] == 30))
    checks.append(("CB date normalized", r0["Acme Robotics"]["last_funding_date"] == "2026-05-01"))
    checks.append(("CB '6,000,000' -> 6000000", r0["Brightloop"]["last_funding_amount_usd"] == 6000000))
    checks.append(("CB MM/DD/YYYY -> ISO", r0["Brightloop"]["last_funding_date"] == "2026-06-10"))

    # 2) RSS funding feed.
    rss = ("<rss><channel>"
           "<item><title>Northwind AI raises $12M Series A to scale its platform</title>"
           "<pubDate>Wed, 01 Jul 2026 08:00:00 GMT</pubDate>"
           "<description>The startup closed a Series A led by a16z.</description></item>"
           "<item><title>Tinyseed Labs lands $2.5M seed round</title>"
           "<pubDate>Mon, 15 Jun 2026 09:00:00 GMT</pubDate></item>"
           "</channel></rss>")
    rr = {x["name"]: x for x in parse_rss(rss, is_path=False)}
    checks.append(("RSS extracted Northwind AI", "Northwind AI" in rr))
    checks.append(("RSS Northwind stage Series A", rr.get("Northwind AI", {}).get("funding_stage") == "Series A"))
    checks.append(("RSS Northwind $12M", rr.get("Northwind AI", {}).get("last_funding_amount_usd") == 12000000))
    checks.append(("RSS Northwind date", rr.get("Northwind AI", {}).get("last_funding_date") == "2026-07-01"))
    checks.append(("RSS Tinyseed seed $2.5M", rr.get("Tinyseed Labs", {}).get("last_funding_amount_usd") == 2500000))

    # 3) unit checks
    checks.append(("$1.5B", parse_amount_usd("$1.5B") == 1_500_000_000))
    checks.append(("500K", parse_amount_usd("500K") == 500_000))
    checks.append(("Series D -> Series C+", normalize_stage("Series D") == "Series C+"))
    checks.append(("Pre-Seed", normalize_stage("Pre-Seed") == "Pre-seed"))

    ok = True
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok = ok and passed
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description="Normalize funding data into prospect rows.")
    p.add_argument("input", nargs="?", help="funding CSV/RSS file, or an RSS feed URL")
    p.add_argument("-o", "--out", default="funding_prospects.csv")
    p.add_argument("--format", choices=["auto", "csv", "rss"], default="auto")
    p.add_argument("--selftest", action="store_true", help="run offline tests, no network")
    args = p.parse_args()
    if args.selftest:
        sys.exit(_selftest())
    if not args.input:
        p.error("provide a funding CSV/RSS file or URL, or use --selftest")
    run(args.input, args.out, args.format)


if __name__ == "__main__":
    main()
