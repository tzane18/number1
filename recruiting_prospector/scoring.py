"""
scoring.py — the Ideal Customer Profile (ICP) engine.

Given a company's signals, produce a 0-100 "fit score" answering:
    "How likely is this company to need — and pay for — a contingency recruiter,
     of the kind that lists roles on Paraform?"

The score is fully transparent: every point is attributed to a named component
with a human-readable reason, so you know WHY a company ranks where it does and
can tune the weights to match the clients you actually win.

The thesis (who needs contingency recruiting):
  * Small enough that they don't have a full in-house recruiting team...
  * ...but funded / growing enough to have real hiring budget and urgency.
  * Lots of open roles relative to their size (hiring is a bottleneck).
  * Roles that are hard to fill (senior, technical, specialized).
  * A gap between the hiring they need to do and the recruiters they have.
  * Signals they're open to outside recruiters (already on a marketplace, etc.).

Weights (max points) — total 100:
    size_fit            20
    funding             20
    hiring_demand       25
    recruiting_gap      20
    marketplace_signal  10
    growth               5
"""

from datetime import date, datetime


WEIGHTS = {
    "size_fit": 20,
    "funding": 20,
    "hiring_demand": 25,
    "recruiting_gap": 20,
    "marketplace_signal": 10,
    "growth": 5,
}


def _num(x, default=0):
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _months_since(iso_date):
    """Whole-ish months between iso_date and today. None if unparseable."""
    if not iso_date:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            d = datetime.strptime(str(iso_date).strip(), fmt).date()
            break
        except ValueError:
            d = None
    if d is None:
        return None
    today = date.today()
    return (today.year - d.year) * 12 + (today.month - d.month)


# ---------------------------------------------------------------------------
# Component scorers. Each returns (points, reason_or_None).
# ---------------------------------------------------------------------------

def _score_size(hc):
    """Sweet spot: big enough to have budget, small enough to lack a talent team."""
    if hc is None or hc <= 0:
        return 6, "Headcount unknown (assumed small/early)"
    hc = int(hc)
    if hc <= 4:
        return 4, f"{hc} employees — likely too early to pay agency fees"
    if hc <= 10:
        return 13, f"{hc} employees — early but may need help with first key hires"
    if hc <= 50:
        return 20, f"{hc} employees — prime size: real budget, rarely a full talent team"
    if hc <= 150:
        return 18, f"{hc} employees — scaling, often out-hires its in-house recruiters"
    if hc <= 300:
        return 11, f"{hc} employees — usually has some in-house talent, but overflow is common"
    if hc <= 600:
        return 6, f"{hc} employees — likely a talent team; contingency only for niche roles"
    return 3, f"{hc} employees — enterprise; low fit for contingency"


def _score_funding(stage, last_date, amount):
    """Recently funded startups have cash + urgency to scale headcount fast."""
    stage = (stage or "").strip().lower()
    stage_pts = {
        "pre-seed": 8, "preseed": 8,
        "seed": 14,
        "series a": 16, "a": 16,
        "series b": 15, "b": 15,
        "series c+": 9, "series c": 9, "c": 9,
        "bootstrapped": 6, "profitable": 6,
        "public": 3,
    }.get(stage, 8)

    months = _months_since(last_date)
    if months is None:
        recency_mult, recency_txt = 0.75, "funding date unknown"
    elif months <= 3:
        recency_mult, recency_txt = 1.0, f"raised ~{months} mo ago — actively scaling now"
    elif months <= 9:
        recency_mult, recency_txt = 0.95, f"raised ~{months} mo ago — deploying capital"
    elif months <= 18:
        recency_mult, recency_txt = 0.75, f"raised ~{months} mo ago"
    else:
        recency_mult, recency_txt = 0.45, f"last raise ~{months} mo ago — may be past peak hiring"

    pts = round(min(WEIGHTS["funding"], stage_pts * recency_mult), 1)
    label = stage.title() if stage else "Unknown stage"
    amt = ""
    if amount:
        amt = f", ${int(amount)/1_000_000:.0f}M" if amount >= 1_000_000 else f", ${int(amount):,}"
    reason = f"{label}{amt} ({recency_txt})"
    return pts, reason


def _score_demand(open_roles, hard_roles, headcount):
    """More open roles — especially hard ones, and especially relative to size."""
    open_roles = int(_num(open_roles))
    hard_roles = int(_num(hard_roles))
    if open_roles <= 0:
        return 2, "No open roles on file — no active demand signal"

    # Absolute volume (up to 12 pts, saturating around ~15 roles).
    volume = min(12.0, open_roles * 0.9)

    # Hard-to-fill roles are where contingency recruiters win (up to 8 pts).
    hard = min(8.0, hard_roles * 1.5)

    # Hiring intensity: roles per 100 employees (up to 5 pts). Signals hiring is
    # a genuine bottleneck rather than routine backfill.
    if headcount and headcount > 0:
        intensity = min(5.0, (open_roles / headcount) * 100 * 0.5)
    else:
        intensity = 3.0  # unknown size + open roles = assume intense

    pts = round(min(WEIGHTS["hiring_demand"], volume + hard + intensity), 1)
    bits = [f"{open_roles} open roles"]
    if hard_roles:
        bits.append(f"{hard_roles} senior/technical (hard to fill)")
    if headcount and headcount > 0:
        bits.append(f"{open_roles/headcount*100:.0f} openings per 100 staff")
    return pts, "; ".join(bits)


def _score_recruiting_gap(open_roles, in_house_recruiters):
    """The gap between hiring needed and recruiters they have = your opening."""
    open_roles = int(_num(open_roles))
    rec = _num(in_house_recruiters, default=None)

    if open_roles <= 0:
        return 4, None
    if rec is None:
        return 14, "In-house recruiting unknown — assume capacity gap"
    rec = int(rec)
    if rec == 0:
        return 20, "No in-house recruiters — every role is on founders/managers"
    load = open_roles / rec  # open roles per internal recruiter
    if load >= 8:
        return 18, f"~{load:.0f} open roles per in-house recruiter — badly overloaded"
    if load >= 5:
        return 15, f"~{load:.0f} open roles per in-house recruiter — overloaded"
    if load >= 3:
        return 10, f"~{load:.0f} open roles per in-house recruiter — stretched"
    return 5, f"~{load:.0f} open roles per in-house recruiter — reasonably staffed"


def _score_marketplace(on_marketplace):
    """Already using an agency marketplace = proven willingness to pay a fee."""
    if on_marketplace in (1, "1", True, "true", "yes"):
        return 10, "Already uses a recruiting marketplace — warm to contingency"
    return 3, None


def _score_growth(growth_6mo):
    g = _num(growth_6mo, default=None)
    if g is None:
        return 2, None
    if g >= 0.5:
        return 5, f"+{g*100:.0f}% headcount in 6 mo — hypergrowth"
    if g >= 0.25:
        return 4, f"+{g*100:.0f}% headcount in 6 mo — fast growth"
    if g >= 0.1:
        return 3, f"+{g*100:.0f}% headcount in 6 mo — steady growth"
    if g > 0:
        return 2, f"+{g*100:.0f}% headcount in 6 mo"
    return 0, "Flat/declining headcount"


def _tier(score):
    if score >= 75:
        return "A"
    if score >= 60:
        return "B"
    if score >= 45:
        return "C"
    return "D"


def score_company(row):
    """
    row: a dict-like with company fields (sqlite3.Row works).
    Returns dict: fit_score, tier, subscores {component: points}, reasons [str].
    """
    g = row.__getitem__ if hasattr(row, "keys") else row.get

    def f(key, default=None):
        try:
            v = g(key)
        except (KeyError, IndexError):
            v = default
        return default if v is None else v

    subscores, reasons = {}, []

    parts = {
        "size_fit": _score_size(f("headcount")),
        "funding": _score_funding(f("funding_stage"), f("last_funding_date"),
                                  f("last_funding_amount_usd")),
        "hiring_demand": _score_demand(f("open_roles"), f("hard_roles"), f("headcount")),
        "recruiting_gap": _score_recruiting_gap(f("open_roles"), f("in_house_recruiters")),
        "marketplace_signal": _score_marketplace(f("on_recruiting_marketplace")),
        "growth": _score_growth(f("headcount_growth_6mo")),
    }

    total = 0.0
    for name, (pts, reason) in parts.items():
        subscores[name] = round(pts, 1)
        total += pts
        if reason:
            reasons.append(reason)

    total = round(min(100.0, total), 1)
    return {
        "fit_score": total,
        "tier": _tier(total),
        "subscores": subscores,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Who do you actually contact? Depends on size + what they're hiring for.
# ---------------------------------------------------------------------------

_FUNCTION_MANAGER = {
    "engineering": "VP Engineering / CTO",
    "software": "VP Engineering / CTO",
    "product": "VP Product / CPO",
    "design": "Head of Design",
    "sales": "VP Sales / CRO",
    "gtm": "VP Sales / CRO",
    "marketing": "VP Marketing / CMO",
    "data": "Head of Data / VP Engineering",
    "operations": "COO / Head of Ops",
    "ops": "COO / Head of Ops",
    "finance": "CFO / VP Finance",
}


def recommend_contact(headcount, primary_function, in_house_recruiters=None):
    """
    Return the ideal title(s) to reach on LinkedIn for this company, given its
    size and what it's hiring for. Small companies -> founders; larger -> talent
    leaders and the specific hiring manager.
    """
    hc = int(_num(headcount)) if headcount else 0
    func = (primary_function or "").strip().lower()
    manager = _FUNCTION_MANAGER.get(func, f"VP {primary_function}" if primary_function else "Hiring Manager")
    has_talent_team = _num(in_house_recruiters, default=0) and _num(in_house_recruiters) > 0

    if hc == 0:
        primary, backups = "Founder / CEO", [manager, "Head of Talent"]
    elif hc <= 30:
        primary, backups = "Founder / CEO / Co-founder", [manager]
    elif hc <= 80:
        if has_talent_team:
            primary, backups = "Head of Talent / Head of People", ["Founder / CEO", manager]
        else:
            primary, backups = "Founder / CEO", ["Head of People (if any)", manager]
    elif hc <= 200:
        primary, backups = "Head of Talent / VP People", [manager, "Talent Acquisition Lead"]
    else:
        primary, backups = f"Talent Acquisition Lead ({primary_function or 'relevant function'})", \
                           ["Recruiting Manager", manager]

    # De-dupe while preserving order.
    seen, ordered = set(), []
    for t in [primary] + backups:
        if t and t not in seen:
            seen.add(t)
            ordered.append(t)
    return {"primary_title": ordered[0], "backup_titles": ordered[1:]}


if __name__ == "__main__":
    # Tiny smoke test.
    demo = {
        "name": "Demo", "headcount": 35, "funding_stage": "Series A",
        "last_funding_date": date.today().replace(month=max(1, date.today().month - 2)).isoformat(),
        "last_funding_amount_usd": 12_000_000, "open_roles": 9, "hard_roles": 5,
        "primary_hiring_function": "Engineering", "in_house_recruiters": 0,
        "on_recruiting_marketplace": 1, "headcount_growth_6mo": 0.4,
    }
    import json
    print(json.dumps(score_company(demo), indent=2))
    print(recommend_contact(35, "Engineering", 0))
