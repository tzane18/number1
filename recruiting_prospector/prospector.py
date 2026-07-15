#!/usr/bin/env python3
"""
prospector.py — command-line control center for your recruiting-client database.

A single SQLite file (prospects.db) holds prospect companies, their hiring
signals, the right contact, an auto-computed fit score, and your LinkedIn
pipeline. This CLI is how you drive it.

Typical workflow:
    python3 prospector.py init                       # create the database
    python3 prospector.py import-csv data/seed_prospects.csv
    python3 prospector.py score                       # rank everyone by fit
    python3 prospector.py rank --limit 15             # see your best targets
    python3 prospector.py show 1                       # deep-dive one company
    python3 prospector.py export outreach_list.csv --your-name "Alex Rivera"
    python3 prospector.py stats                        # pipeline snapshot

No external dependencies — Python 3.8+ standard library only.
"""

import argparse
import csv
import json
import os
import sqlite3
import sys

import scoring
import outreach as outreach_mod
from import_funding import normalize_name

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "prospects.db")
SCHEMA_PATH = os.path.join(HERE, "schema.sql")

# CSV columns we understand on import. Extra columns are ignored; missing ones
# become NULL. Only `name` is required.
COMPANY_FIELDS = [
    "name", "domain", "linkedin_url", "industry", "hq_location",
    "headcount", "headcount_growth_6mo", "funding_stage", "last_funding_date",
    "last_funding_amount_usd", "open_roles", "hard_roles",
    "primary_hiring_function", "in_house_recruiters",
    "on_recruiting_marketplace", "source", "notes",
]
_INT_FIELDS = {"headcount", "last_funding_amount_usd", "open_roles", "hard_roles",
               "in_house_recruiters", "on_recruiting_marketplace"}
_FLOAT_FIELDS = {"headcount_growth_6mo"}


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _coerce(field, value):
    if value is None or value == "":
        return None
    if field in _INT_FIELDS:
        try:
            return int(float(value))
        except ValueError:
            return None
    if field in _FLOAT_FIELDS:
        try:
            return float(value)
        except ValueError:
            return None
    return str(value).strip()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args):
    fresh = not os.path.exists(DB_PATH)
    with open(SCHEMA_PATH) as fh:
        schema = fh.read()
    conn = connect()
    conn.executescript(schema)
    conn.commit()
    conn.close()
    print(f"{'Created' if fresh else 'Ensured'} database at {DB_PATH}")


def _upsert_company(conn, record):
    cols = [f for f in COMPANY_FIELDS if f in record]
    placeholders = ", ".join("?" for _ in cols)
    collist = ", ".join(cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "name")
    sql = (f"INSERT INTO companies ({collist}) VALUES ({placeholders}) "
           f"ON CONFLICT(name, domain) DO UPDATE SET {updates}, updated_at=datetime('now')")
    conn.execute(sql, [record[c] for c in cols])


def cmd_import_csv(args):
    if not os.path.exists(args.file):
        sys.exit(f"File not found: {args.file}")
    conn = connect()
    n = 0
    with open(args.file, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            name = (raw.get("name") or "").strip()
            if not name:
                continue
            record = {}
            for f in COMPANY_FIELDS:
                if f in raw:
                    record[f] = _coerce(f, raw[f])
            record["name"] = name
            record.setdefault("domain", record.get("domain"))
            _upsert_company(conn, record)
            n += 1
    conn.commit()
    conn.close()
    print(f"Imported/updated {n} companies from {args.file}")
    if not args.no_score:
        cmd_score(argparse.Namespace())
        print("(auto-scored; pass --no-score to skip)")


def cmd_score(args):
    conn = connect()
    rows = conn.execute("SELECT * FROM companies").fetchall()
    for row in rows:
        result = scoring.score_company(row)
        conn.execute(
            "INSERT INTO scores (company_id, fit_score, tier, subscores_json, reasons_json, scored_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now')) "
            "ON CONFLICT(company_id) DO UPDATE SET fit_score=excluded.fit_score, "
            "tier=excluded.tier, subscores_json=excluded.subscores_json, "
            "reasons_json=excluded.reasons_json, scored_at=datetime('now')",
            (row["id"], result["fit_score"], result["tier"],
             json.dumps(result["subscores"]), json.dumps(result["reasons"])),
        )
    conn.commit()
    conn.close()
    print(f"Scored {len(rows)} companies.")


def _fetch_ranked(conn, tier=None, limit=None):
    sql = ("SELECT c.*, s.fit_score, s.tier, s.reasons_json "
           "FROM companies c JOIN scores s ON s.company_id = c.id ")
    params = []
    if tier:
        sql += "WHERE s.tier = ? "
        params.append(tier.upper())
    sql += "ORDER BY s.fit_score DESC "
    if limit:
        sql += "LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


def cmd_rank(args):
    conn = connect()
    rows = _fetch_ranked(conn, args.tier, args.limit)
    conn.close()
    if not rows:
        print("No scored companies yet. Run: import-csv then score.")
        return
    print(f"{'#':>2}  {'Score':>5} {'Tier':>4}  {'HC':>4} {'Roles':>5}  {'Stage':<10} Company")
    print("-" * 78)
    for i, r in enumerate(rows, 1):
        print(f"{i:>2}  {r['fit_score']:>5.1f}  {r['tier']:>3}  "
              f"{(r['headcount'] or '—'):>4} {(r['open_roles'] or '—'):>5}  "
              f"{(r['funding_stage'] or '—'):<10} {r['name']}")


def _resolve_company(conn, ident):
    if str(ident).isdigit():
        row = conn.execute("SELECT * FROM companies WHERE id=?", (int(ident),)).fetchone()
        if row:
            return row
    return conn.execute("SELECT * FROM companies WHERE name LIKE ? LIMIT 1",
                        (f"%{ident}%",)).fetchone()


def cmd_show(args):
    conn = connect()
    c = _resolve_company(conn, args.company)
    if not c:
        conn.close()
        sys.exit(f"No company matching '{args.company}'")
    s = conn.execute("SELECT * FROM scores WHERE company_id=?", (c["id"],)).fetchone()
    rec = scoring.recommend_contact(c["headcount"], c["primary_hiring_function"],
                                    c["in_house_recruiters"])
    kit = outreach_mod.build_outreach_kit(
        c, primary_title=rec["primary_title"], your_name=args.your_name)
    conn.close()

    print("=" * 70)
    print(f"  {c['name']}   (id {c['id']})")
    print("=" * 70)
    print(f"  Industry : {c['industry'] or '—'}    Location: {c['hq_location'] or '—'}")
    print(f"  Size     : {c['headcount'] or '?'} employees"
          f"    Stage: {c['funding_stage'] or '?'}    "
          f"Raised: {c['last_funding_date'] or '?'}")
    print(f"  Hiring   : {c['open_roles'] or 0} open ({c['hard_roles'] or 0} hard) "
          f"in {c['primary_hiring_function'] or '?'}    "
          f"In-house recruiters: {c['in_house_recruiters'] if c['in_house_recruiters'] is not None else '?'}")
    if s:
        print(f"\n  FIT SCORE: {s['fit_score']}/100   (Tier {s['tier']})")
        subs = json.loads(s["subscores_json"])
        for k, v in subs.items():
            bar = "#" * int(round(v))
            print(f"    {k:<20} {v:>5.1f}  {bar}")
        print("\n  Why:")
        for reason in json.loads(s["reasons_json"]):
            print(f"    • {reason}")

    print(f"\n  WHO TO CONTACT: {rec['primary_title']}")
    if rec["backup_titles"]:
        print(f"    backups: {', '.join(rec['backup_titles'])}")
    print(f"    find them: {kit['find_person_url']}")

    print("\n  ── LinkedIn connection note ─────────────────────────")
    print("   " + kit["connection_note"])
    print("\n  ── First message (after they accept) ────────────────")
    for line in kit["first_message"].splitlines():
        print("   " + line)
    print()


def cmd_export(args):
    conn = connect()
    rows = _fetch_ranked(conn, args.tier, args.limit)
    out_cols = [
        "rank", "company", "fit_score", "tier", "headcount", "funding_stage",
        "open_roles", "hard_roles", "primary_hiring_function", "in_house_recruiters",
        "target_title", "find_person_url", "connection_note", "first_message",
        "top_reasons", "company_linkedin",
    ]
    with open(args.file, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            rec = scoring.recommend_contact(r["headcount"], r["primary_hiring_function"],
                                            r["in_house_recruiters"])
            kit = outreach_mod.build_outreach_kit(
                r, primary_title=rec["primary_title"], your_name=args.your_name)
            reasons = json.loads(r["reasons_json"]) if r["reasons_json"] else []
            w.writerow({
                "rank": i, "company": r["name"], "fit_score": r["fit_score"],
                "tier": r["tier"], "headcount": r["headcount"],
                "funding_stage": r["funding_stage"], "open_roles": r["open_roles"],
                "hard_roles": r["hard_roles"],
                "primary_hiring_function": r["primary_hiring_function"],
                "in_house_recruiters": r["in_house_recruiters"],
                "target_title": rec["primary_title"],
                "find_person_url": kit["find_person_url"],
                "connection_note": kit["connection_note"],
                "first_message": kit["first_message"],
                "top_reasons": " | ".join(reasons[:3]),
                "company_linkedin": r["linkedin_url"] or kit["company_url"],
            })
    conn.close()
    print(f"Exported {len(rows)} ranked prospects -> {args.file}")


def cmd_add(args):
    conn = connect()
    record = {"name": args.name}
    for f in COMPANY_FIELDS:
        v = getattr(args, f, None)
        if v is not None:
            record[f] = _coerce(f, v)
    _upsert_company(conn, record)
    conn.commit()
    row = _resolve_company(conn, args.name)
    result = scoring.score_company(row)
    conn.execute(
        "INSERT INTO scores (company_id, fit_score, tier, subscores_json, reasons_json) "
        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(company_id) DO UPDATE SET "
        "fit_score=excluded.fit_score, tier=excluded.tier, "
        "subscores_json=excluded.subscores_json, reasons_json=excluded.reasons_json, "
        "scored_at=datetime('now')",
        (row["id"], result["fit_score"], result["tier"],
         json.dumps(result["subscores"]), json.dumps(result["reasons"])))
    conn.commit()
    conn.close()
    print(f"Added '{args.name}' — fit score {result['fit_score']} (Tier {result['tier']})")


def _rescore(conn, company_id):
    row = conn.execute("SELECT * FROM companies WHERE id=?", (company_id,)).fetchone()
    result = scoring.score_company(row)
    conn.execute(
        "INSERT INTO scores (company_id, fit_score, tier, subscores_json, reasons_json) "
        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(company_id) DO UPDATE SET "
        "fit_score=excluded.fit_score, tier=excluded.tier, "
        "subscores_json=excluded.subscores_json, reasons_json=excluded.reasons_json, "
        "scored_at=datetime('now')",
        (company_id, result["fit_score"], result["tier"],
         json.dumps(result["subscores"]), json.dumps(result["reasons"])))
    return result


# Fields enrich may fill. Deliberately excludes the live-hiring signals
# (open_roles, hard_roles, primary_hiring_function) so a funding import never
# clobbers what fetch_ats.py learned from the job boards.
_ENRICH_FIELDS = [
    "domain", "linkedin_url", "industry", "hq_location", "headcount",
    "headcount_growth_6mo", "funding_stage", "last_funding_date",
    "last_funding_amount_usd", "in_house_recruiters", "on_recruiting_marketplace",
]


def cmd_enrich(args):
    if not os.path.exists(args.file):
        sys.exit(f"File not found: {args.file}")
    conn = connect()
    companies = conn.execute("SELECT * FROM companies").fetchall()
    by_name = {}
    for c in companies:
        by_name.setdefault(normalize_name(c["name"]).lower(), c)

    matched = inserted = unmatched = 0
    touched = set()
    unmatched_names = []

    with open(args.file, newline="", encoding="utf-8-sig") as fh:
        for raw in csv.DictReader(fh):
            name = (raw.get("name") or "").strip()
            if not name:
                continue
            key = normalize_name(name).lower()
            existing = by_name.get(key)

            if existing is None:
                if args.insert_missing:
                    record = {"name": name}
                    for f in COMPANY_FIELDS:
                        if f in raw and raw[f] not in (None, ""):
                            record[f] = _coerce(f, raw[f])
                    _upsert_company(conn, record)
                    conn.commit()
                    row = _resolve_company(conn, name)
                    by_name[key] = row
                    touched.add(row["id"])
                    inserted += 1
                else:
                    unmatched += 1
                    unmatched_names.append(name)
                continue

            # Fill enrichable fields that are empty (or all, with --overwrite).
            updates, applied = {}, []
            for f in _ENRICH_FIELDS:
                if f not in raw:
                    continue
                new_val = _coerce(f, raw[f])
                if new_val in (None, ""):
                    continue
                cur = existing[f] if f in existing.keys() else None
                if args.overwrite or cur in (None, ""):
                    updates[f] = new_val
                    applied.append(f)
            if updates:
                sets = ", ".join(f"{k}=?" for k in updates)
                conn.execute(f"UPDATE companies SET {sets}, updated_at=datetime('now') "
                             f"WHERE id=?", list(updates.values()) + [existing["id"]])
                touched.add(existing["id"])
                matched += 1
                if args.verbose:
                    print(f"  enriched {existing['name']}: {', '.join(applied)}")

    conn.commit()
    for cid in touched:
        _rescore(conn, cid)
    conn.commit()
    conn.close()

    print(f"Enriched {matched} existing companies"
          + (f", inserted {inserted} new" if inserted else "")
          + f". {unmatched} unmatched.")
    if unmatched and not args.insert_missing:
        print("  Unmatched (use --insert-missing to add them, or fix name spelling):")
        for n in unmatched_names[:15]:
            print(f"    - {n}")
        if len(unmatched_names) > 15:
            print(f"    ... and {len(unmatched_names) - 15} more")
    print(f"Re-scored {len(touched)} companies.")


def cmd_digest(args):
    """This run's new / newly-qualified prospects — the fresh targets to work."""
    conn = connect()
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")

    if args.since:
        cutoff, basis = args.since, f"since {args.since}"
    elif args.days:
        cutoff = conn.execute("SELECT datetime('now', ?)", (f"-{args.days} days",)).fetchone()[0]
        basis = f"last {args.days} days"
    else:
        row = conn.execute("SELECT value FROM meta WHERE key='last_digest_at'").fetchone()
        if row:
            cutoff, basis = row[0], f"since last digest ({row[0]} UTC)"
        else:
            cutoff = conn.execute("SELECT datetime('now', '-7 days')").fetchone()[0]
            basis = "last 7 days (no previous digest)"

    tiers = [t.strip().upper() for t in (args.tiers or "A,B").split(",") if t.strip()]
    ph = ",".join("?" for _ in tiers)
    rows = conn.execute(
        f"SELECT c.*, s.fit_score, s.tier, s.reasons_json, o.status AS ostatus "
        f"FROM companies c JOIN scores s ON s.company_id=c.id "
        f"LEFT JOIN outreach o ON o.company_id=c.id "
        f"WHERE (c.created_at > ? OR s.scored_at > ?) AND s.tier IN ({ph}) "
        f"ORDER BY s.fit_score DESC",
        [cutoff, cutoff] + tiers).fetchall()

    if not args.include_contacted:
        rows = [r for r in rows if (r["ostatus"] in (None, "not_started"))]

    print("=" * 72)
    print(f"  PROSPECT DIGEST — {basis}")
    print(f"  Tier {'/'.join(tiers)}"
          + ("" if args.include_contacted else ", not yet contacted")
          + f"  ·  {len(rows)} target(s)")
    print("=" * 72)

    if not rows:
        print("  Nothing new to work. Refresh data (fetch_ats / import_funding + enrich)\n"
              "  or widen the window: digest --days 30")
    for i, r in enumerate(rows, 1):
        rec = scoring.recommend_contact(r["headcount"], r["primary_hiring_function"],
                                        r["in_house_recruiters"])
        kit = outreach_mod.build_outreach_kit(r, primary_title=rec["primary_title"],
                                              your_name=args.your_name)
        reasons = json.loads(r["reasons_json"]) if r["reasons_json"] else []
        size = f"{r['headcount']}ppl" if r["headcount"] else "size ?"
        print(f"\n {i}. {r['name']}   [{r['tier']} · {r['fit_score']:.0f}]   "
              f"{size} · {r['funding_stage'] or 'stage ?'} · "
              f"{r['open_roles'] or 0} open roles")
        print(f"     → contact: {rec['primary_title']}")
        print(f"       find:    {kit['find_person_url']}")
        if reasons:
            print(f"       why:     {reasons[0]}")

    if not args.peek and rows is not None:
        now = conn.execute("SELECT datetime('now')").fetchone()[0]
        conn.execute("INSERT INTO meta(key, value) VALUES('last_digest_at', ?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (now,))
        conn.commit()
        print(f"\n  (marked digest run at {now} UTC — next digest shows what's new after this;"
              f" use --peek to look without marking)")
    conn.close()


def cmd_stats(args):
    conn = connect()
    total = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    print(f"Companies in database: {total}")
    if total == 0:
        conn.close()
        return
    print("\nBy tier:")
    for row in conn.execute(
            "SELECT COALESCE(tier,'unscored') t, COUNT(*) n, ROUND(AVG(fit_score),1) avg "
            "FROM companies c LEFT JOIN scores s ON s.company_id=c.id "
            "GROUP BY t ORDER BY t"):
        print(f"  Tier {row['t']:<9} {row['n']:>4} companies   (avg score {row['avg'] or '—'})")
    print("\nOutreach pipeline:")
    pipe = conn.execute(
        "SELECT status, COUNT(*) n FROM outreach GROUP BY status").fetchall()
    if not pipe:
        print("  (no outreach logged yet)")
    for row in pipe:
        print(f"  {row['status']:<18} {row['n']:>4}")
    conn.close()


def build_parser():
    p = argparse.ArgumentParser(
        description="Recruiting-client prospecting database & qualification engine.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the SQLite database").set_defaults(func=cmd_init)

    pi = sub.add_parser("import-csv", help="import/refresh companies from a CSV")
    pi.add_argument("file")
    pi.add_argument("--no-score", action="store_true", help="don't auto-score after import")
    pi.set_defaults(func=cmd_import_csv)

    sub.add_parser("score", help="(re)compute fit scores for all companies").set_defaults(func=cmd_score)

    pr = sub.add_parser("rank", help="list companies best-fit first")
    pr.add_argument("--tier", help="filter to a tier (A/B/C/D)")
    pr.add_argument("--limit", type=int)
    pr.set_defaults(func=cmd_rank)

    ps = sub.add_parser("show", help="full detail + outreach kit for one company")
    ps.add_argument("company", help="company id or name fragment")
    ps.add_argument("--your-name", default="[Your Name]")
    ps.set_defaults(func=cmd_show)

    pe = sub.add_parser("export", help="write a ranked outreach action list to CSV")
    pe.add_argument("file")
    pe.add_argument("--tier")
    pe.add_argument("--limit", type=int)
    pe.add_argument("--your-name", default="[Your Name]")
    pe.set_defaults(func=cmd_export)

    pa = sub.add_parser("add", help="add a single company via flags")
    pa.add_argument("name")
    for f in COMPANY_FIELDS:
        if f != "name":
            pa.add_argument("--" + f.replace("_", "-"), dest=f, default=None)
    pa.set_defaults(func=cmd_add)

    pn = sub.add_parser("enrich",
                        help="merge funding/other data into EXISTING companies by name "
                             "(fills empty fields; won't clobber live hiring signals)")
    pn.add_argument("file")
    pn.add_argument("--overwrite", action="store_true",
                    help="overwrite existing values instead of only filling blanks")
    pn.add_argument("--insert-missing", action="store_true",
                    help="add companies that don't match any existing record")
    pn.add_argument("--verbose", action="store_true")
    pn.set_defaults(func=cmd_enrich)

    pd = sub.add_parser("digest",
                        help="show new / newly-qualified prospects since the last digest run")
    pd.add_argument("--days", type=int, help="look back N days instead of since last digest")
    pd.add_argument("--since", help="explicit cutoff date/time (YYYY-MM-DD)")
    pd.add_argument("--tiers", default="A,B", help="tiers to include (default A,B)")
    pd.add_argument("--include-contacted", action="store_true",
                    help="also show prospects already in outreach")
    pd.add_argument("--peek", action="store_true",
                    help="don't mark this as a digest run (look without advancing the window)")
    pd.add_argument("--your-name", default="[Your Name]")
    pd.set_defaults(func=cmd_digest)

    sub.add_parser("stats", help="database + pipeline snapshot").set_defaults(func=cmd_stats)
    return p


def main():
    args = build_parser().parse_args()
    if args.command != "init" and not os.path.exists(DB_PATH):
        sys.exit("No database yet. Run:  python3 prospector.py init")
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # stdout was closed early (e.g. piped to `head`); exit quietly.
        try:
            sys.stdout.close()
        except Exception:
            pass
