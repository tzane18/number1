# Where to get prospects (and how to keep the database fed)

The engine in this project does the hard part — deciding *who is worth your time*
and *who to message* — but it needs a stream of companies to judge. Here is how to
feed it without violating anyone's Terms of Service.

> **The honest constraint.** A bot that live-scrapes LinkedIn or Paraform breaks
> their Terms of Service and gets accounts banned. So the loop is: pull company +
> hiring data from sources that *allow* programmatic or export access, drop it into
> a CSV, and let `prospector.py` qualify and rank it. You (a human) then do the
> actual LinkedIn outreach it hands you. That keeps you compliant and still saves
> the 90% of the work that is research and prioritization.

## The signals that matter (map these into the CSV columns)

| Column | What it tells the engine | Where to find it |
|---|---|---|
| `open_roles`, `hard_roles` | Active hiring demand | Company careers pages, Greenhouse/Lever/Ashby public boards, Wellfound, YC's Work at a Startup, LinkedIn Jobs |
| `funding_stage`, `last_funding_date`, `last_funding_amount_usd` | Cash + urgency to scale | Crunchbase, PitchBook, TechCrunch, "[X] raises" news, VC portfolio pages |
| `headcount`, `headcount_growth_6mo` | Size sweet spot + momentum | LinkedIn company page, Crunchbase, company "About" |
| `in_house_recruiters` | Whether they can cover hiring themselves | LinkedIn people search: `"Recruiter" OR "Talent" at <company>` |
| `on_recruiting_marketplace` | Proven willingness to pay an agency fee | Paraform's public company list, mentions of agency/contract recruiting |
| `primary_hiring_function` | Who to route the message to | The functions that dominate their open roles |

## Sources, ranked by how easy they are to pull

1. **Public ATS job boards (best signal, easiest to pull).** Most startups host jobs
   on Greenhouse (`boards.greenhouse.io/<company>`), Lever (`jobs.lever.co/<company>`),
   or Ashby. These have public JSON endpoints. A company with 8+ roles on Greenhouse
   and no "Recruiter" job posted is a textbook target.
2. **YC's Work at a Startup / Wellfound (AngelList).** Curated early-stage companies
   that are actively hiring — exactly the Paraform demographic. Filter by stage/size.
3. **Funding news.** Crunchbase (has an API/CSV export on paid tiers), plus free feeds
   from TechCrunch, Axios Pro Rata, and VC portfolio pages. A fresh Seed/Series A raise
   is the single strongest "call them now" trigger — that's why the engine weights it.
4. **Paraform itself.** Browse the companies posting roles there. Any company posting
   on Paraform is, by definition, open to contingency recruiters — mark
   `on_recruiting_marketplace = 1` and it jumps up your list.
5. **Data providers (paid, fully automatable, ToS-clean).** Apollo.io, Clearbit,
   Crunchbase API, People Data Labs, LinkedIn Sales Navigator exports. These *do*
   allow programmatic pulls and enrichment. If you want real automation, this is the
   compliant path: schedule a weekly pull → export CSV → `import-csv` → `export`.

## The included fetcher: `fetch_ats.py`

For source #1 (public ATS boards) this repo ships a working fetcher. Put your target
companies in a watchlist (see `data/watchlist.example.csv`) with the board slug from
their careers URL (`jobs.lever.co/SLUG`, `boards.greenhouse.io/SLUG`,
`jobs.ashbyhq.com/SLUG`), then:

```bash
python3 fetch_ats.py my_watchlist.csv -o prospects_from_ats.csv
python3 prospector.py import-csv prospects_from_ats.csv
```

It pulls live postings and fills in `open_roles`, `hard_roles`, and
`primary_hiring_function` automatically; funding/headcount columns in the watchlist pass
straight through. (`python3 fetch_ats.py --selftest` verifies the parsers offline.)

## A weekly routine (about 30 minutes)

1. Refresh live hiring data: `python3 fetch_ats.py my_watchlist.csv -o batch.csv`
   (and/or add newly-funded companies from a funding feed into the same CSV shape).
2. `python3 prospector.py import-csv batch.csv`  ← re-imports are safe; it updates
   existing companies by `name + domain` and adds new ones.
3. `python3 prospector.py rank --tier A --limit 20` to see this week's best targets.
4. `python3 prospector.py export week_of_2026_07_20.csv --your-name "Your Name"`.
5. Work the list on LinkedIn: use `find_person_url` to reach the right person, send the
   `connection_note`, then the `first_message` once they accept.
6. Log what happens (see the `outreach` table / `stats`) so you learn which tiers convert.

## Automating the pull (optional)

The `import-csv` command is the integration point. Anything that can produce a CSV with
the columns above can feed this database on a schedule:

- A small script hitting Greenhouse/Lever public boards for a watchlist of companies.
- A cron job calling the Crunchbase or Apollo API for "raised in the last 30 days,
  11–200 employees, tech" and writing the CSV.
- A Zapier/Make flow that appends new funding announcements to a Google Sheet you export.

Keep the human in the loop for the LinkedIn step — that is both the compliant choice and
the one that actually books meetings.
