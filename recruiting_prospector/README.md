# Recruiting Prospector

A database + qualification engine for a **contingency recruiter** looking to win new
clients — the kind of early-stage startups and small companies that post roles on
**Paraform** and need outside recruiting help. You feed it companies; it automatically
figures out **which ones are worth your time**, **who to reach on LinkedIn**, and
**what to say** — then hands you a ranked, ready-to-send outreach list.

It's a single SQLite file and pure-Python CLI. No servers, no accounts, no dependencies
beyond Python 3.8+.

---

## What it does

1. **Stores** prospect companies and their hiring signals (size, funding, open roles,
   in-house recruiters, etc.) in a local SQLite database.
2. **Scores** every company 0–100 on how likely it is to *need and pay for* a
   contingency recruiter — with a fully transparent, tunable breakdown of *why*.
3. **Targets the right person** — for each company it recommends the exact title to
   contact based on size and what they're hiring for (founder vs. head of talent vs.
   the hiring manager) and builds the LinkedIn search URL to find them.
4. **Writes the outreach** — a personalized LinkedIn connection note and first message
   per prospect, keyed off their strongest signal (fresh funding, a pile of open roles,
   hard-to-fill technical roles).
5. **Exports** all of the above as one ranked CSV you can work top-down.

### What "automatic" honestly means here

The **qualification, targeting, and message-writing is fully automatic** — that's the
90% of prospecting that is tedious research and prioritization. And `fetch_ats.py`
(below) **automatically pulls live open roles** from Greenhouse / Lever / Ashby to build
your input CSV. What is *not* automated is scraping LinkedIn/Paraform directly: doing
that violates their Terms of Service and gets accounts banned — so *you* send the
LinkedIn messages the tool drafts. That keeps you compliant while the database does the
heavy lifting. Other signals (funding, headcount) come from funding feeds or a data
provider like Apollo/Crunchbase — see **[SOURCES.md](SOURCES.md)**.

## Automatically pull live openings (Greenhouse / Lever / Ashby)

Most startups host their jobs on Greenhouse, Lever, or Ashby, and each exposes a
**public JSON endpoint** (the same one their careers page uses). `fetch_ats.py` reads
those — which is allowed, unlike scraping LinkedIn — and turns them into a ready-to-import
CSV, deriving `open_roles`, `hard_roles` (senior/technical), and `primary_hiring_function`
straight from the live postings.

```bash
# 1. Make a watchlist of companies + which board they use (see data/watchlist.example.csv)
#    Find the board "slug" in a company's careers URL:
#      jobs.lever.co/SLUG   ·   boards.greenhouse.io/SLUG   ·   jobs.ashbyhq.com/SLUG

# 2. Pull their live openings into a CSV
python3 fetch_ats.py data/watchlist.example.csv -o prospects_from_ats.csv

# 3. Feed it straight into the database (auto-scores)
python3 prospector.py import-csv prospects_from_ats.csv
python3 prospector.py rank --tier A

# 4. Each week, pull just what's newly worth working:
python3 prospector.py digest --your-name "Your Name"

# Offline check that the parsers work, no network needed:
python3 fetch_ats.py --selftest
```

The watchlist's optional columns (`funding_stage`, `headcount`, `in_house_recruiters`, …)
pass straight through, so you can layer funding-news / data-provider info onto the live
hiring signal in one file. Re-running `fetch_ats.py` weekly keeps the hiring numbers
fresh; `import-csv` updates existing companies in place.

> **Note on this sandbox:** if you're running inside a restricted Claude Code web session,
> outbound access to those job-board hosts may be blocked by the environment's network
> policy (you'll see a 403). Run `fetch_ats.py` from your own machine, or use a web
> environment whose network policy allows public sites. The `--selftest` works anywhere.

## Don't know the board slugs? Just use company names

`find_board.py` takes company **names**, auto-discovers each one's job board
(Greenhouse / Lever / Ashby) by probing the public endpoints, and writes the
watchlist for you — so you skip the manual slug hunt entirely.

```bash
# names one per line in a file (see data/example_companies.txt):
python3 find_board.py data/example_companies.txt -o my_watchlist.csv

# or straight on the command line:
python3 find_board.py --names "Ramp" "Vercel" "Retool" -o my_watchlist.csv

python3 find_board.py --selftest      # offline check, no network
```

It prints a sample job title for each match so you can confirm it found the right
company, then you run `fetch_ats.py my_watchlist.csv` as usual. Companies it can't
resolve are listed so you can grab those slugs by hand.

## Add the funding signal (Crunchbase / funding news)

A fresh raise is the strongest "reach out now" trigger. `import_funding.py` normalizes
funding data — a **Crunchbase CSV export**, any funding CSV (it auto-detects the columns),
or a **funding-news RSS feed** — into the same row shape.

```bash
# Crunchbase / PitchBook / Tracxn CSV export (auto-detects headers):
python3 import_funding.py crunchbase_export.csv -o funding.csv

# A funding-news RSS feed (file or URL) — pulls "X raises $Y Series Z" headlines:
python3 import_funding.py https://example.com/funding.rss -o funding.csv

python3 import_funding.py --selftest      # offline parser tests
```

Then merge it onto the companies you already have — this is what puts the funding signal
and the live hiring signal on the **same scored record**:

```bash
python3 prospector.py enrich funding.csv            # fills empty fields only
python3 prospector.py enrich funding.csv --insert-missing   # also add new companies
```

`enrich` matches by company name (ignoring `Inc.`/`LLC` suffixes), fills in funding,
headcount, location, etc., and **never overwrites** the `open_roles` / `hard_roles` /
`primary_hiring_function` that `fetch_ats.py` derived from the job boards. Affected
companies are automatically re-scored. Unmatched names are reported so you can fix
spelling or add them.

**The full compliant pipeline:**

```
fetch_ats.py  (live open roles) ─┐
                                 ├─►  prospects in SQLite  ─►  rank / export  ─►  LinkedIn
import_funding.py (fresh raises)─┘        (auto-scored)
```

---

## Quick start

```bash
cd recruiting_prospector

# 1. Create the database
python3 prospector.py init

# 2. Load the included sample prospects (or your own CSV — same columns)
python3 prospector.py import-csv data/seed_prospects.csv

# 3. See your best-fit targets
python3 prospector.py rank --limit 15

# 4. Deep-dive one company (score breakdown + who to contact + the messages)
python3 prospector.py show "Nimbus Data" --your-name "Your Name"

# 5. Export a ranked outreach list to work through
python3 prospector.py export my_list.csv --tier A --your-name "Your Name"
```

> The companies in `data/seed_prospects.csv` are **synthetic examples** so the tool works
> out of the box — replace them with real prospects. See **[SOURCES.md](SOURCES.md)** for
> where to get real data.

---

## Commands

| Command | What it does |
|---|---|
| `init` | Create the SQLite database (`prospects.db`). |
| `import-csv <file>` | Import/refresh companies from a CSV (re-imports are safe — updates by name+domain). Auto-scores unless `--no-score`. |
| `enrich <file> [--insert-missing] [--overwrite]` | Merge funding/other data into existing companies by name; fills empty fields, re-scores, and won't clobber live hiring signals. |
| `score` | Recompute fit scores for everyone. |
| `rank [--tier A] [--limit N]` | List companies best-fit first. |
| `digest [--days N] [--peek]` | Show only the **new / newly-qualified** A/B prospects since your last digest run — your weekly "who to work now" list. |
| `show <id or name> [--your-name X]` | Full detail on one company: score bars, reasons, who to contact, and the drafted outreach. |
| `export <file.csv> [--tier A] [--limit N] [--your-name X]` | Write a ranked, ready-to-send outreach list. |
| `add "<Name>" [--headcount N ...]` | Add a single company by hand (and score it). |
| `stats` | Database + outreach-pipeline snapshot. |

Run `python3 prospector.py <command> --help` for the flags on any command.

---

## How the fit score works

Every company gets 0–100 points across six components (weights in parentheses). This is
the whole thesis of *who needs a contingency recruiter*, encoded as math:

| Component (max) | Rewards |
|---|---|
| **Size fit** (20) | The sweet spot ~11–150 employees: real budget, but usually no full talent team. Penalizes both too-early (<5) and enterprise (>500). |
| **Funding** (20) | Recent Seed/Series A/B raises — cash plus urgency to scale. Recency matters: a raise 2 months ago outscores one 2 years ago. |
| **Hiring demand** (25) | Volume of open roles, how many are hard-to-fill (senior/technical), and hiring intensity (roles per 100 staff). |
| **Recruiting gap** (20) | Open roles vs. in-house recruiters. Zero recruiters + many roles = your biggest opening. |
| **Marketplace signal** (10) | Already uses Paraform or a similar marketplace = proven willingness to pay a placement fee. |
| **Growth** (5) | Headcount growth over the last 6 months. |

Tiers: **A** ≥ 75 · **B** 60–74 · **C** 45–59 · **D** < 45.

The logic lives in [`scoring.py`](scoring.py) and is meant to be tuned — as you win
clients, adjust the `WEIGHTS` and thresholds to match the companies that actually say yes.
Because every point is attributed to a named reason, you can always see *why* a company
ranks where it does.

### Who to contact (also automatic)

`recommend_contact()` maps company size + hiring function to the right title:

- **≤ 30 people** → Founder / CEO (they own hiring)
- **31–80** → Head of Talent if they have one, else Founder
- **81–200** → Head of Talent / VP People + the hiring manager (VP Eng, VP Sales, …)
- **> 200** → Talent Acquisition Lead / Recruiting Manager for the relevant function

---

## Data model

One SQLite file, five tables (full schema in [`schema.sql`](schema.sql)):

- **`companies`** — the prospects and their signals
- **`roles`** — individual open jobs (optional granular demand data)
- **`contacts`** — the humans you reach out to
- **`scores`** — the computed fit score, tier, and reasons
- **`outreach`** — your LinkedIn pipeline per company (stages: `connection_sent →
  accepted → messaged → replied → meeting → won/lost`)

It's plain SQLite, so you can also query it directly:

```bash
sqlite3 prospects.db "SELECT name, fit_score, tier FROM v_prospects LIMIT 10;"
```

## CSV format

The importer reads these columns (only `name` is required; extras are ignored, missing
ones become empty). See `data/seed_prospects.csv` for a filled-in example.

```
name, domain, linkedin_url, industry, hq_location, headcount,
headcount_growth_6mo, funding_stage, last_funding_date,
last_funding_amount_usd, open_roles, hard_roles,
primary_hiring_function, in_house_recruiters,
on_recruiting_marketplace, source, notes
```

## Files

```
recruiting_prospector/
├── prospector.py               # the CLI (init, import, enrich, score, rank, show, export, add, stats)
├── find_board.py               # company NAMES -> auto-discovered job-board watchlist
├── fetch_ats.py                # pull live roles from Greenhouse/Lever/Ashby -> import CSV
├── import_funding.py           # normalize Crunchbase CSV / funding RSS -> enrich CSV
├── scoring.py                  # the ICP fit-score engine + contact recommender
├── outreach.py                 # LinkedIn message + search-URL generator
├── schema.sql                  # SQLite schema
├── data/seed_prospects.csv     # synthetic sample prospects (replace with real data)
├── data/watchlist.example.csv  # template for fetch_ats.py
├── SOURCES.md                  # where to get real prospect data, compliantly
└── README.md
```
