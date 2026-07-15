-- Recruiting Prospector — database schema
-- SQLite (no server required). One file: prospects.db
--
-- Model of the world:
--   companies   -> the prospect (a startup/SMB that might need a contingency recruiter)
--   roles       -> the open jobs that company is trying to fill (the demand signal)
--   contacts    -> the human you actually reach out to on LinkedIn
--   scores      -> the auto-computed "how good a fit is this" (0-100) + why
--   outreach    -> your LinkedIn pipeline for each company/contact

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS companies (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    name                   TEXT NOT NULL,
    domain                 TEXT,
    linkedin_url           TEXT,
    industry               TEXT,
    hq_location            TEXT,
    headcount              INTEGER,          -- current employee count
    headcount_growth_6mo   REAL,             -- fractional, e.g. 0.30 = +30% in 6 months
    funding_stage          TEXT,             -- Pre-seed, Seed, Series A, Series B, Series C+, Bootstrapped
    last_funding_date      TEXT,             -- ISO date YYYY-MM-DD
    last_funding_amount_usd INTEGER,
    open_roles             INTEGER,          -- total open positions right now
    hard_roles             INTEGER,          -- of those, senior/technical/specialized (harder to fill)
    primary_hiring_function TEXT,            -- Engineering, Sales, Product, Design, GTM, Ops, ...
    in_house_recruiters    INTEGER,          -- size of internal talent team (0 = none)
    on_recruiting_marketplace INTEGER,       -- 0/1 known to use Paraform or similar agency marketplace
    source                 TEXT,             -- where this record came from (job board, funding feed, ...)
    notes                  TEXT,
    created_at             TEXT DEFAULT (datetime('now')),
    updated_at             TEXT DEFAULT (datetime('now')),
    UNIQUE(name, domain)
);

CREATE TABLE IF NOT EXISTS roles (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id   INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    title        TEXT,
    function     TEXT,          -- Engineering, Sales, ...
    seniority    TEXT,          -- Junior, Mid, Senior, Lead, Exec
    location     TEXT,
    posted_date  TEXT,
    url          TEXT
);

CREATE TABLE IF NOT EXISTS contacts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id    INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    full_name     TEXT,
    title         TEXT,
    role_type     TEXT,         -- Founder/CEO, Head of Talent, Hiring Manager, ...
    linkedin_url  TEXT,
    is_primary    INTEGER DEFAULT 0,   -- 1 = best person to contact first
    priority_rank INTEGER,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS scores (
    company_id     INTEGER PRIMARY KEY REFERENCES companies(id) ON DELETE CASCADE,
    fit_score      REAL,          -- 0-100
    tier           TEXT,          -- A / B / C / D
    subscores_json TEXT,          -- JSON: component -> points
    reasons_json   TEXT,          -- JSON: list of human-readable reasons
    scored_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS outreach (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id       INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    contact_id       INTEGER REFERENCES contacts(id) ON DELETE SET NULL,
    channel          TEXT DEFAULT 'linkedin',
    -- pipeline stages: not_started -> connection_sent -> accepted -> messaged
    --                  -> replied -> meeting -> won / lost / nurture
    status           TEXT DEFAULT 'not_started',
    connection_note  TEXT,
    first_message    TEXT,
    last_touch_date  TEXT,
    next_action      TEXT,
    next_action_date TEXT,
    created_at       TEXT DEFAULT (datetime('now')),
    updated_at       TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_roles_company    ON roles(company_id);
CREATE INDEX IF NOT EXISTS idx_contacts_company ON contacts(company_id);
CREATE INDEX IF NOT EXISTS idx_outreach_company ON outreach(company_id);
CREATE INDEX IF NOT EXISTS idx_outreach_status  ON outreach(status);

-- Convenient view: everything you need for an outreach list, ranked.
CREATE VIEW IF NOT EXISTS v_prospects AS
SELECT
    c.id,
    c.name,
    c.headcount,
    c.funding_stage,
    c.open_roles,
    c.hard_roles,
    c.primary_hiring_function,
    c.in_house_recruiters,
    s.fit_score,
    s.tier,
    o.status  AS outreach_status,
    c.linkedin_url
FROM companies c
LEFT JOIN scores   s ON s.company_id  = c.id
LEFT JOIN outreach o ON o.company_id  = c.id
ORDER BY s.fit_score DESC;
