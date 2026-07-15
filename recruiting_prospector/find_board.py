#!/usr/bin/env python3
"""
find_board.py — turn a list of COMPANY NAMES into a ready-to-fetch watchlist.

The one manual step in prospecting is finding each company's job-board "slug".
This automates it: give it company names, and for each it guesses likely slugs,
probes the public Greenhouse / Lever / Ashby endpoints, and — when one answers
with live jobs — records the board, slug, role count, and a sample title so you
can eyeball that it's the right company.

Usage:
    # names one per line in a file:
    python3 find_board.py companies.txt -o my_watchlist.csv

    # or names straight on the command line:
    python3 find_board.py --names "Ramp" "Vercel" "Retool" -o my_watchlist.csv

    python3 find_board.py --selftest      # offline tests, no network

Then:
    python3 fetch_ats.py my_watchlist.csv -o batch.csv
    python3 prospector.py import-csv batch.csv

Reuses fetch_ats.py's endpoints/parsers, so anything it finds fetch_ats can pull.
No third-party dependencies — Python 3.8+ standard library only.
"""

import argparse
import csv
import re
import sys
import time

import fetch_ats as fa


_LEGAL = re.compile(r"\b(inc|inc\.|llc|ltd|limited|corp|corporation|co|gmbh|"
                    r"technologies|technology|labs|software|systems|the)\b", re.I)


def slug_candidates(name):
    """Generate likely board slugs for a company name, most-specific first."""
    base = name.lower().strip()
    base = base.replace("&", " and ")
    base = re.sub(r"[^a-z0-9 ]", " ", base)          # drop punctuation
    full_words = [w for w in base.split() if w]
    core_words = [w for w in full_words if not _LEGAL.fullmatch(w)] or full_words

    cands = []

    def add(s):
        s = s.strip("-")
        if s and s not in cands:
            cands.append(s)

    add("".join(full_words))          # "acmerobotics"
    add("".join(core_words))          # core-only concatenation
    add("-".join(core_words))         # "acme-robotics"
    add("-".join(full_words))
    if core_words:                    # single leading word — least precise, last
        add(core_words[0])
    return cands[:5]


def probe_slug(slug, retries=1):
    """Try this slug against each ATS. Return (ats, postings) on the first hit."""
    for ats, spec in fa.ATS.items():
        try:
            data = fa.fetch_json(spec["url"].format(token=slug), retries=retries)
            postings = spec["parse"](data)
            if postings:
                return ats, postings
        except Exception:
            continue
    return None


def resolve(name, delay=0.3):
    """Find the board for one company name, or None."""
    for slug in slug_candidates(name):
        hit = probe_slug(slug)
        if hit:
            ats, postings = hit
            titled = [p for p in postings if p.get("title")]
            return {
                "name": name.strip(),
                "ats": ats,
                "token": slug,
                "open_roles": len(titled),
                "sample_title": titled[0]["title"] if titled else "",
            }
        time.sleep(delay)
    return None


def read_names(path):
    names = []
    with open(path, encoding="utf-8-sig") as fh:
        # Accept a plain list (one name per line) OR a CSV with a 'name' column.
        first = fh.readline()
        fh.seek(0)
        if "," in first and "name" in first.lower():
            for row in csv.DictReader(fh):
                n = (row.get("name") or "").strip()
                if n:
                    names.append(n)
        else:
            for line in fh:
                n = line.strip()
                if n and not n.startswith("#"):
                    names.append(n)
    return names


def run(names, out_path, delay):
    found, missed = [], []
    for i, name in enumerate(names, 1):
        r = resolve(name, delay=delay)
        if r:
            found.append(r)
            print(f"[{i}/{len(names)}] FOUND {name:<28} -> {r['ats']}:{r['token']}  "
                  f"({r['open_roles']} roles; e.g. \"{r['sample_title']}\")")
        else:
            missed.append(name)
            print(f"[{i}/{len(names)}] ----- {name:<28} -> no public board found "
                  f"(check spelling, or find the slug manually)")
        if i < len(names):
            time.sleep(delay)

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "ats", "token", "funding_stage", "last_funding_date",
                    "last_funding_amount_usd", "headcount", "in_house_recruiters",
                    "on_recruiting_marketplace"])
        for r in found:
            w.writerow([r["name"], r["ats"], r["token"], "", "", "", "", "", ""])

    print(f"\nMatched {len(found)}/{len(names)} companies -> {out_path}")
    if missed:
        print(f"Not found ({len(missed)}): {', '.join(missed)}")
        print("  For these, open the careers page, click Apply on a job, and read the")
        print("  slug from the boards.greenhouse.io / jobs.lever.co / jobs.ashbyhq.com URL.")
    print(f"\nReview the sample titles above, then:  "
          f"python3 fetch_ats.py {out_path} -o batch.csv")


def _selftest():
    checks = []
    c1 = slug_candidates("Acme Robotics")
    checks.append(("'Acme Robotics' -> acmerobotics", "acmerobotics" in c1))
    checks.append(("'Acme Robotics' -> acme-robotics", "acme-robotics" in c1))
    c2 = slug_candidates("Bright Loop, Inc.")
    checks.append(("strips 'Inc.' punctuation", "brightloop" in c2))
    c3 = slug_candidates("Ramp")
    checks.append(("single word -> ramp", c3[0] == "ramp"))
    c4 = slug_candidates("Data & AI Co")
    checks.append(("'&' -> 'and'", any("dataandai" in c for c in c4)))
    checks.append(("most-specific candidate first (not bare first word)",
                   slug_candidates("Northwind Systems")[0] == "northwindsystems"))
    ok = True
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok = ok and passed
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(
        description="Resolve company names to job-board slugs -> watchlist CSV.")
    p.add_argument("file", nargs="?", help="text file of company names (one per line) "
                                           "or a CSV with a 'name' column")
    p.add_argument("--names", nargs="+", help="company names directly on the CLI")
    p.add_argument("-o", "--out", default="my_watchlist.csv")
    p.add_argument("--delay", type=float, default=0.3, help="seconds between probes")
    p.add_argument("--selftest", action="store_true", help="offline tests, no network")
    args = p.parse_args()

    if args.selftest:
        sys.exit(_selftest())
    names = list(args.names) if args.names else (read_names(args.file) if args.file else [])
    if not names:
        p.error("provide a names file, --names ..., or --selftest")
    run(names, args.out, args.delay)


if __name__ == "__main__":
    main()
