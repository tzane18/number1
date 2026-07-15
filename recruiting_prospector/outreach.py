"""
outreach.py — turn a scored prospect into ready-to-send LinkedIn outreach.

Produces, per company:
  * a LinkedIn people-search URL to FIND the right person, and
  * a connection note (< 300 chars, LinkedIn's limit) and a first message,
    personalized off the strongest signal we have (fresh funding, or a pile of
    open roles, or hard-to-fill technical roles).

Everything is a plain template you can edit. Nothing is sent anywhere — this
just drafts the words. You review, personalize the first line, and send.
"""

from urllib.parse import quote


def linkedin_people_search_url(company_name, title):
    """A LinkedIn search that surfaces the right person at the company."""
    q = f"{company_name} {title}".strip()
    return "https://www.linkedin.com/search/results/people/?keywords=" + quote(q)


def linkedin_company_url(company_name):
    return "https://www.linkedin.com/search/results/companies/?keywords=" + quote(company_name)


def _pick_hook(company):
    """Choose the single strongest, most personal reason to reach out now."""
    g = company.__getitem__ if hasattr(company, "keys") else company.get

    def f(key, default=None):
        try:
            v = g(key)
        except (KeyError, IndexError):
            v = default
        return default if v is None else v

    stage = (f("funding_stage") or "").strip()
    open_roles = int(f("open_roles") or 0)
    hard_roles = int(f("hard_roles") or 0)
    func = (f("primary_hiring_function") or "").strip()
    growth = f("headcount_growth_6mo")

    # Priority order: recent raise > many hard roles > lots of open roles > growth.
    if stage and f("last_funding_date"):
        return (f"Congrats on the {stage} raise — scaling the team fast is usually the "
                f"next scramble, especially on {func or 'key'} roles.")
    if hard_roles >= 3 and func:
        return (f"Saw you're hiring several senior {func} roles — those are exactly the "
                f"hard-to-fill searches I run for startups.")
    if open_roles >= 4:
        plural = "roles" if open_roles != 1 else "role"
        return (f"Noticed you've got {open_roles} open {plural} right now — that's a lot of "
                f"hiring load for a team your size.")
    if growth and float(growth) >= 0.25:
        return "Saw the team's grown fast lately — hiring at that pace is hard to keep up with."
    return (f"I recruit for startups like yours{(' on ' + func + ' roles') if func else ''} and "
            f"wanted to reach out.")


def build_connection_note(company, contact_name=None):
    """LinkedIn connection request note (kept under 300 chars)."""
    g = company.__getitem__ if hasattr(company, "keys") else company.get
    name = (contact_name or "").split(" ")[0] if contact_name else None
    greeting = f"Hi {name}," if name else "Hi,"
    hook = _pick_hook(company)
    note = (f"{greeting} {hook} I'm an independent recruiter who fills roles on "
            f"contingency (you only pay on a hire). Open to connecting?")
    if len(note) > 300:
        note = note[:297].rstrip() + "..."
    return note


def build_first_message(company, contact_name=None, your_name="[Your Name]",
                        specialty=None):
    """The follow-up message after they accept the connection."""
    g = company.__getitem__ if hasattr(company, "keys") else company.get

    def f(key, default=None):
        try:
            v = g(key)
        except (KeyError, IndexError):
            v = default
        return default if v is None else v

    name = (contact_name or "").split(" ")[0] if contact_name else "there"
    company_name = f("name") or "your team"
    func = (f("primary_hiring_function") or "").strip()
    hook = _pick_hook(company)
    spec = specialty or (f"{func} and technical roles" if func else "startup roles")

    return (
        f"Thanks for connecting, {name}!\n\n"
        f"{hook}\n\n"
        f"Quick context on me: I'm a contingency recruiter specializing in {spec} for "
        f"early-stage startups. I work the way Paraform-style engagements do — I only "
        f"get paid when you actually hire someone I send, so there's zero risk to trying me "
        f"on a role or two.\n\n"
        f"If filling {('a role on your ' + func + ' team') if func else 'an open role'} at {company_name} "
        f"is on your plate this quarter, would you be open to a quick 15-min call to see if "
        f"I can help? Happy to start with your hardest-to-fill seat.\n\n"
        f"Best,\n{your_name}"
    )


def build_outreach_kit(company, contact_name=None, primary_title=None,
                       your_name="[Your Name]", specialty=None):
    """Everything you need to act on one prospect."""
    g = company.__getitem__ if hasattr(company, "keys") else company.get
    company_name = (g("name") if hasattr(company, "keys") else company.get("name")) or "Company"
    title = primary_title or "Founder / CEO"
    return {
        "find_person_url": linkedin_people_search_url(company_name, title),
        "company_url": linkedin_company_url(company_name),
        "target_title": title,
        "connection_note": build_connection_note(company, contact_name),
        "first_message": build_first_message(company, contact_name, your_name, specialty),
    }


if __name__ == "__main__":
    demo = {
        "name": "Nimbus Data", "funding_stage": "Series A",
        "last_funding_date": "2026-05-01", "open_roles": 9, "hard_roles": 5,
        "primary_hiring_function": "Engineering", "headcount_growth_6mo": 0.4,
    }
    kit = build_outreach_kit(demo, primary_title="Founder / CEO", your_name="Alex Rivera")
    for k, v in kit.items():
        print(f"\n=== {k} ===\n{v}")
