#!/usr/bin/env python3
"""
fetch_ats.py — turn public ATS job boards into prospect rows.

Startups host their open roles on Greenhouse, Lever, or Ashby, and each of those
exposes a PUBLIC JSON endpoint (the same one their own careers page calls to render
the job list). Reading those endpoints is allowed — unlike scraping LinkedIn — so
this is the compliant way to get a live hiring-demand signal automatically.

Given a watchlist of companies (name + which board + the board's token), this script:
  * pulls every open posting,
  * counts open roles and how many are senior/technical ("hard"),
  * infers the primary hiring function from the mix of roles,
  * and writes a CSV in exactly the shape `prospector.py import-csv` expects.

Signals an ATS board can't tell you — funding, headcount, in-house recruiter count —
you add as optional columns in the watchlist and they pass straight through, so you
can layer in funding-news / data-provider info alongside the live hiring data.

Usage:
    python3 fetch_ats.py watchlist.csv -o prospects_from_ats.csv
    python3 fetch_ats.py --selftest          # parse built-in fixtures, no network

Watchlist CSV columns (only name, ats, token are required):
    name, ats, token,
    domain, hq_location, headcount, headcount_growth_6mo,
    funding_stage, last_funding_date, last_funding_amount_usd,
    in_house_recruiters, on_recruiting_marketplace, industry
  ats is one of: greenhouse | lever | ashby
  token is the board slug, e.g. Greenhouse boards-api.greenhouse.io/v1/boards/<token>/jobs

No third-party dependencies — Python 3.8+ standard library only.
"""

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

USER_AGENT = "recruiting-prospector/1.0 (+public-job-board-reader)"
REQUEST_TIMEOUT = 25

# Passed straight through from the watchlist into the output (ATS can't supply these).
PASSTHROUGH_FIELDS = [
    "domain", "hq_location", "headcount", "headcount_growth_6mo",
    "funding_stage", "last_funding_date", "last_funding_amount_usd",
    "in_house_recruiters", "on_recruiting_marketplace", "industry",
]

# Output columns must match prospector.py's COMPANY_FIELDS.
OUTPUT_FIELDS = [
    "name", "domain", "linkedin_url", "industry", "hq_location",
    "headcount", "headcount_growth_6mo", "funding_stage", "last_funding_date",
    "last_funding_amount_usd", "open_roles", "hard_roles",
    "primary_hiring_function", "in_house_recruiters",
    "on_recruiting_marketplace", "source", "notes",
]


# ---------------------------------------------------------------------------
# Classifiers: title (+ department hint) -> function, and is-this-role-hard.
# ---------------------------------------------------------------------------

# Order matters: earlier buckets win when keywords overlap.
_FUNCTION_KEYWORDS = [
    ("Engineering", ["software engineer", "engineer", "developer", " swe", "sre",
                     "devops", "infrastructure", "backend", "back-end", "frontend",
                     "front-end", "full stack", "full-stack", "platform", "mobile",
                     "ios ", "android", "security engineer", "qa ", "architect",
                     "machine learning", " ml ", "ai engineer", "data engineer"]),
    ("Data", ["data scientist", "data analyst", "analytics", "data science",
              "business intelligence", " bi "]),
    ("Product", ["product manager", "product owner", "technical product",
                 "program manager, product", "head of product", "vp product"]),
    ("Design", ["designer", "ux", "ui ", "user experience", "product design",
                "brand design", "graphic"]),
    ("Sales", ["account executive", " ae ", "sdr", "bdr", "sales", "account manager",
               "revenue", "business development", "gtm", "go-to-market"]),
    ("Marketing", ["marketing", "growth", "demand gen", "content", "seo", "brand ",
                   "communications", "social media", "pr "]),
    ("Customer", ["customer success", "customer support", "support engineer",
                  "solutions engineer", "csm", "implementation", "onboarding specialist"]),
    ("Finance", ["finance", "accounting", "accountant", "controller", "fp&a", "treasury"]),
    ("People", ["recruiter", "talent acquisition", "people ops", "human resources",
                " hr ", "people partner"]),
    ("Operations", ["operations", " ops", "business operations", "program manager",
                    "project manager", "chief of staff", "logistics", "supply chain"]),
]

_SENIOR_KEYWORDS = ["senior", "sr.", "sr ", "staff", "principal", "lead ", " lead",
                    "head of", "director", "vp ", "vice president", "chief", "architect",
                    "manager"]
_JUNIOR_KEYWORDS = ["junior", "jr.", "jr ", "intern", "internship", "entry",
                    "new grad", "new-grad", "apprentice", "associate", "trainee"]


def _pad(s):
    return f" {s.lower().strip()} "


def classify_function(title, dept_hint=""):
    hay = _pad(title) + _pad(dept_hint)
    for func, keywords in _FUNCTION_KEYWORDS:
        for kw in keywords:
            if kw in hay:
                return func
    return "Other"


def is_hard_role(title, function):
    """Hard-to-fill = senior/leadership, OR a technical IC role (not junior).

    These are exactly the searches contingency recruiters get hired for."""
    hay = _pad(title)
    if any(j in hay for j in _JUNIOR_KEYWORDS):
        return False
    if any(s in hay for s in _SENIOR_KEYWORDS):
        return True
    return function in ("Engineering", "Data")


# ---------------------------------------------------------------------------
# Per-ATS parsers: raw JSON -> normalized [{title, location, function}].
# Kept separate from the network layer so they're unit-testable offline.
# ---------------------------------------------------------------------------

def parse_greenhouse(data):
    out = []
    for job in data.get("jobs", []):
        title = job.get("title", "") or ""
        loc = ((job.get("location") or {}).get("name")) or ""
        func = classify_function(title)
        out.append({"title": title, "location": loc, "function": func})
    return out


def parse_lever(data):
    out = []
    for job in data:
        title = job.get("text", "") or ""
        cats = job.get("categories") or {}
        loc = cats.get("location", "") or ""
        dept = f"{cats.get('team','')} {cats.get('department','')}"
        func = classify_function(title, dept)
        out.append({"title": title, "location": loc, "function": func})
    return out


def parse_ashby(data):
    out = []
    for job in data.get("jobs", []):
        title = job.get("title", "") or ""
        loc = job.get("location", "") or ""
        dept = f"{job.get('department','')} {job.get('team','')}"
        func = classify_function(title, dept)
        out.append({"title": title, "location": loc, "function": func})
    return out


ATS = {
    "greenhouse": {
        "url": "https://boards-api.greenhouse.io/v1/boards/{token}/jobs",
        "parse": parse_greenhouse,
    },
    "lever": {
        "url": "https://api.lever.co/v0/postings/{token}?mode=json",
        "parse": parse_lever,
    },
    "ashby": {
        "url": "https://api.ashbyhq.com/posting-api/job-board/{token}",
        "parse": parse_ashby,
    },
}


# ---------------------------------------------------------------------------
# Network layer
# ---------------------------------------------------------------------------

def fetch_json(url, retries=3):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (404, 400):     # bad token — don't bother retrying
                raise
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    if last:
        raise last


def fetch_company(ats, token):
    ats = ats.strip().lower()
    if ats not in ATS:
        raise ValueError(f"unknown ats '{ats}' (use: {', '.join(ATS)})")
    spec = ATS[ats]
    data = fetch_json(spec["url"].format(token=token.strip()))
    return spec["parse"](data)


# ---------------------------------------------------------------------------
# Roll a company's postings up into one prospect row.
# ---------------------------------------------------------------------------

def summarize(meta, postings):
    postings = [p for p in postings if p.get("title")]
    open_roles = len(postings)
    hard_roles = sum(1 for p in postings if is_hard_role(p["title"], p["function"]))

    func_counts = Counter(p["function"] for p in postings if p["function"] != "Other")
    primary_function = func_counts.most_common(1)[0][0] if func_counts else ""

    # Most common concrete location (skip blanks / "remote"-only noise for the label).
    loc_counts = Counter((p["location"] or "").strip() for p in postings if p.get("location"))
    top_location = loc_counts.most_common(1)[0][0] if loc_counts else ""

    hiring_own_recruiter = any(p["function"] == "People" and
                               ("recruit" in p["title"].lower() or
                                "talent" in p["title"].lower())
                               for p in postings)

    row = {f: "" for f in OUTPUT_FIELDS}
    row["name"] = meta["name"]
    row["open_roles"] = open_roles
    row["hard_roles"] = hard_roles
    row["primary_hiring_function"] = primary_function
    row["source"] = f"{meta['ats'].lower()}:{meta['token']}"

    for f in PASSTHROUGH_FIELDS:
        if meta.get(f):
            row[f] = meta[f]
    if not row["hq_location"] and top_location:
        row["hq_location"] = top_location

    notes = [f"{open_roles} live roles via {meta['ats'].lower()}"]
    if hiring_own_recruiter:
        notes.append("posting a recruiter/talent role — building in-house; act soon")
    row["notes"] = "; ".join(notes)
    return row


# ---------------------------------------------------------------------------
# Watchlist -> output CSV
# ---------------------------------------------------------------------------

def read_watchlist(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = []
        for r in csv.DictReader(fh):
            r = {(k or "").strip(): (v.strip() if isinstance(v, str) else v)
                 for k, v in r.items()}
            if not r.get("name") or not r.get("ats") or not r.get("token"):
                continue
            rows.append(r)
        return rows


def run(watchlist_path, out_path, delay=0.5):
    companies = read_watchlist(watchlist_path)
    if not companies:
        sys.exit("Watchlist is empty or missing required columns (name, ats, token).")
    results, failures = [], []
    for i, meta in enumerate(companies, 1):
        label = f"{meta['name']} ({meta['ats']}:{meta['token']})"
        try:
            postings = fetch_company(meta["ats"], meta["token"])
            row = summarize(meta, postings)
            results.append(row)
            print(f"[{i}/{len(companies)}] OK   {label} -> "
                  f"{row['open_roles']} roles, {row['hard_roles']} hard, "
                  f"{row['primary_hiring_function'] or '—'}")
        except Exception as e:
            failures.append((label, f"{type(e).__name__}: {e}"))
            print(f"[{i}/{len(companies)}] FAIL {label} -> {type(e).__name__}: {e}",
                  file=sys.stderr)
        if delay and i < len(companies):
            time.sleep(delay)

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        w.writerows(results)

    print(f"\nWrote {len(results)} companies -> {out_path}")
    if failures:
        print(f"{len(failures)} failed:")
        for label, err in failures:
            print(f"   - {label}: {err}")
    print("\nNext:  python3 prospector.py import-csv " + out_path)


# ---------------------------------------------------------------------------
# Offline self-test: prove the parsers/classifiers work without any network.
# ---------------------------------------------------------------------------

def _selftest():
    gh = {"jobs": [
        {"title": "Senior Backend Engineer", "location": {"name": "Remote - US"}},
        {"title": "Staff Machine Learning Engineer", "location": {"name": "SF"}},
        {"title": "Account Executive", "location": {"name": "NYC"}},
        {"title": "Product Designer", "location": {"name": "Remote"}},
        {"title": "Marketing Intern", "location": {"name": "SF"}},
    ]}
    lever = [
        {"text": "Head of Sales", "categories": {"team": "Sales", "location": "Austin"}},
        {"text": "Software Engineer, Platform", "categories": {"team": "Engineering", "location": "Remote"}},
        {"text": "Technical Recruiter", "categories": {"team": "People", "location": "SF"}},
    ]
    ashby = {"jobs": [
        {"title": "Principal Data Scientist", "location": "Remote", "department": "Data"},
        {"title": "Junior Frontend Developer", "location": "Boston", "department": "Engineering"},
    ]}

    ghp, lvp, ashp = parse_greenhouse(gh), parse_lever(lever), parse_ashby(ashby)
    checks = []

    r = summarize({"name": "GH Co", "ats": "greenhouse", "token": "ghco"}, ghp)
    checks.append(("GH open_roles==5", r["open_roles"] == 5))
    checks.append(("GH hard_roles==2 (2 sr eng, AE excluded, intern excluded)",
                   r["hard_roles"] == 2))
    checks.append(("GH primary==Engineering", r["primary_hiring_function"] == "Engineering"))

    r2 = summarize({"name": "Lever Co", "ats": "lever", "token": "lc"}, lvp)
    checks.append(("Lever hard includes Head of Sales + SWE",
                   r2["hard_roles"] == 2))
    checks.append(("Lever flags in-house recruiter note",
                   "recruiter" in r2["notes"].lower()))

    r3 = summarize({"name": "Ashby Co", "ats": "ashby", "token": "ac"}, ashp)
    checks.append(("Ashby Principal Data Scientist is hard",
                   is_hard_role("Principal Data Scientist", "Data")))
    checks.append(("Ashby Junior dev is NOT hard",
                   not is_hard_role("Junior Frontend Developer", "Engineering")))
    checks.append(("Ashby hard_roles==1", r3["hard_roles"] == 1))

    ok = True
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok = ok and passed
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description="Fetch live openings from public ATS boards.")
    p.add_argument("watchlist", nargs="?", help="watchlist CSV (name, ats, token, ...)")
    p.add_argument("-o", "--out", default="prospects_from_ats.csv",
                   help="output CSV (default: prospects_from_ats.csv)")
    p.add_argument("--delay", type=float, default=0.5,
                   help="seconds between requests (be polite; default 0.5)")
    p.add_argument("--selftest", action="store_true",
                   help="run offline parser tests and exit (no network)")
    args = p.parse_args()

    if args.selftest:
        sys.exit(_selftest())
    if not args.watchlist:
        p.error("provide a watchlist CSV, or use --selftest")
    run(args.watchlist, args.out, args.delay)


if __name__ == "__main__":
    main()
