#!/usr/bin/env python3
"""
generate_fintech_sqlite.py

Generates a realistic fintech payments dataset in SQLite.

Usage:
    python3 generate_fintech_sqlite.py --db fintech.db --num_users 100000 --num_txns 3000000

Notes:
- Uses Python standard library only.
- Inserts in chunks to avoid memory blow-up.
- Adjust NUM_USERS and NUM_TXNS to scale dataset size.
"""

import argparse
import sqlite3
import random
import uuid
import time
from datetime import datetime, timedelta
import math
import itertools

# -------------------------
# Configuration / Defaults
# -------------------------
DEFAULT_DB = "fintech.db"
DEFAULT_NUM_USERS = 100_000
DEFAULT_NUM_TXNS = 3_000_000
BATCH_USERS = 10_000
BATCH_TXNS = 50_000
BATCH_ATTEMPTS = 50_000
COHORT_MONTHS = 6

# Tunables (you can change these at top-level or via code)
TXN_TYPE_DISTR = {
    "merchant_qr": 0.70,
    "p2p": 0.15,
    "billpay": 0.10,
    "recharge": 0.05
}
GLOBAL_SUCCESS_RATE = 0.89  # between 0.85 - 0.92 (you can randomize per-run)
GLOBAL_FRAUD_BASE = 0.03  # base fraud prob (will be modified by amount, user age, instrument)
POWER_USER_FRACTION = 0.20
POWER_USER_TXN_SHARE = 0.60  # approx 60% of txns should originate from power users
INCENTIVE_TXNS = 3  # first N txns eligible for incentives
NEW_USER_DAYS = 7  # <7 days considered "new user"
FRAUD_INSTRUMENTS = {"card_prepaid", "card_credit"}  # instrument types contributing to higher fraud
INSTRUMENT_TYPES = ["card_credit", "card_debit", "card_prepaid", "upi", "wallet", "bank_account"]
GATEWAY_COUNT = 6
MERCHANT_COUNT = min(20000, DEFAULT_NUM_TXNS // 30)  # scale with txns (but not huge)
INSTRUMENTS_PER_USER = 2  # how many instruments each user may have
MAX_ATTEMPTS_SUCCESS = (1, 2)  # inclusive range
MAX_ATTEMPTS_FAIL = (1, 3)
REVENUE_ROWS_PER_TXN = (1, 3)  # inclusive
RETRY_INCREASE_FOR_FAILED = 0.15  # increases attempts probability for failures
DORMANT_USER_PCT = 0.08  # percent of users who stop after 1 txn
POWER_USER_WEIGHT = 5.0  # multiplier weight for power users when sampling user IDs

# Amount distribution parameters (log-normal helps heavy-tail)
AMOUNT_MEDIAN = 200.0
AMOUNT_SIGMA = 1.2  # higher sigma => heavier tail

# Time distribution
NOW = datetime.utcnow()
COHORT_START = NOW - timedelta(days=COHORT_MONTHS * 30)  # approx 6 months back

# -------------------------
# Utilities
# -------------------------
def rand_timestamp_between(start: datetime, end: datetime) -> datetime:
    """Return random timestamp between start and end biased to business hours and month-end."""
    total_seconds = int((end - start).total_seconds())
    r = random.random()
    # bias: more likely later in timeframe (slight positive skew)
    r = r ** 0.9
    secs = int(r * total_seconds)
    ts = start + timedelta(seconds=secs)
    # apply intraday bias: more likely during 8am-9pm
    if random.random() < 0.8:
        hour = int(random.gauss(15, 4)) % 24
        ts = ts.replace(hour=hour, minute=random.randint(0,59), second=random.randint(0,59), microsecond=0)
    else:
        ts = ts.replace(hour=random.randint(0,23), minute=random.randint(0,59), second=random.randint(0,59), microsecond=0)
    # month-end spike for billpay: if chosen, move closer to month end
    if random.random() < 0.12:
        # move to last 4 days of that month
        month = ts.month
        year = ts.year
        # find last day
        next_month = month % 12 + 1
        next_month_year = year + (1 if next_month == 1 else 0)
        last_day = (datetime(next_month_year, next_month, 1) - timedelta(days=1)).day
        day = random.randint(max(26, last_day-3), last_day)
        ts = ts.replace(day=day)
    return ts

def lognormal_amount(median=AMOUNT_MEDIAN, sigma=AMOUNT_SIGMA):
    """Log-normal sampling producing positive skewed amounts."""
    # median = exp(mu)
    mu = math.log(median)
    val = random.lognormvariate(mu, sigma)
    # cap max amount for realism
    if val > 100000:
        val = 100000 + random.random() * 5000
    return round(val, 2)

def choose_txn_type():
    r = random.random()
    cum = 0.0
    for t, p in TXN_TYPE_DISTR.items():
        cum += p
        if r <= cum:
            return t
    return "merchant_qr"

# -------------------------
# Schema SQL
# -------------------------
CREATE_TABLE_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS dim_user (
    user_id INTEGER PRIMARY KEY,
    uuid TEXT UNIQUE,
    created_ts TEXT,
    is_power_user INTEGER,
    dormant_after_one_txn INTEGER,
    first_name TEXT,
    last_name TEXT,
    email TEXT
);

CREATE TABLE IF NOT EXISTS dim_merchant (
    merchant_id INTEGER PRIMARY KEY,
    name TEXT,
    category TEXT,
    created_ts TEXT
);

CREATE TABLE IF NOT EXISTS dim_instrument (
    instrument_id INTEGER PRIMARY KEY,
    user_id INTEGER,
    inst_type TEXT,
    last4 TEXT,
    provider TEXT,
    created_ts TEXT,
    FOREIGN KEY (user_id) REFERENCES dim_user(user_id)
);

CREATE TABLE IF NOT EXISTS dim_gateway (
    gateway_id INTEGER PRIMARY KEY,
    name TEXT,
    region TEXT,
    created_ts TEXT
);

CREATE TABLE IF NOT EXISTS dim_txn_type (
    txn_type_id INTEGER PRIMARY KEY,
    txn_type TEXT UNIQUE,
    description TEXT
);

CREATE TABLE IF NOT EXISTS fact_transactions (
    txn_id INTEGER PRIMARY KEY,
    user_id INTEGER,
    merchant_id INTEGER,
    instrument_id INTEGER,
    gateway_id INTEGER,
    txn_type TEXT,
    amount REAL,
    currency TEXT,
    status TEXT,
    created_ts TEXT,
    attempt_count INTEGER,
    is_fraud INTEGER DEFAULT 0,
    is_incentive_applied INTEGER DEFAULT 0,
    kyc_level TEXT,
    FOREIGN KEY (user_id) REFERENCES dim_user(user_id),
    FOREIGN KEY (merchant_id) REFERENCES dim_merchant(merchant_id),
    FOREIGN KEY (instrument_id) REFERENCES dim_instrument(instrument_id),
    FOREIGN KEY (gateway_id) REFERENCES dim_gateway(gateway_id)
);

CREATE TABLE IF NOT EXISTS fact_payment_attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id INTEGER,
    attempt_no INTEGER,
    gateway_id INTEGER,
    status TEXT,
    attempt_ts TEXT,
    response_code TEXT,
    FOREIGN KEY (txn_id) REFERENCES fact_transactions(txn_id),
    FOREIGN KEY (gateway_id) REFERENCES dim_gateway(gateway_id)
);

CREATE TABLE IF NOT EXISTS fact_revenue (
    revenue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id INTEGER,
    revenue_component TEXT,
    amount REAL,
    created_ts TEXT,
    FOREIGN KEY (txn_id) REFERENCES fact_transactions(txn_id)
);

CREATE TABLE IF NOT EXISTS fact_incentives (
    incentive_id INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id INTEGER,
    user_id INTEGER,
    incentive_type TEXT,
    amount REAL,
    created_ts TEXT,
    FOREIGN KEY (txn_id) REFERENCES fact_transactions(txn_id),
    FOREIGN KEY (user_id) REFERENCES dim_user(user_id)
);

CREATE TABLE IF NOT EXISTS fact_fraud_loss (
    fraud_id INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id INTEGER,
    detected_ts TEXT,
    loss_amount REAL,
    reason TEXT,
    FOREIGN KEY (txn_id) REFERENCES fact_transactions(txn_id)
);

CREATE TABLE IF NOT EXISTS fact_kyc_events (
    kyc_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    kyc_level TEXT,
    kyc_ts TEXT,
    status TEXT,
    FOREIGN KEY (user_id) REFERENCES dim_user(user_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON fact_transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_transactions_txn_id ON fact_transactions(txn_id);
CREATE INDEX IF NOT EXISTS idx_transactions_created_ts ON fact_transactions(created_ts);
CREATE INDEX IF NOT EXISTS idx_transactions_status ON fact_transactions(status);
CREATE INDEX IF NOT EXISTS idx_transactions_gateway_id ON fact_transactions(gateway_id);

CREATE INDEX IF NOT EXISTS idx_attempts_txn_id ON fact_payment_attempts(txn_id);
CREATE INDEX IF NOT EXISTS idx_attempts_attempt_ts ON fact_payment_attempts(attempt_ts);
"""

# -------------------------
# Generator Implementation
# -------------------------
def create_schema(conn):
    cur = conn.cursor()
    cur.executescript(CREATE_TABLE_SQL)
    conn.commit()

def seed_dim_txn_type(conn):
    cur = conn.cursor()
    types = list(TXN_TYPE_DISTR.keys())
    cur.executemany("INSERT OR IGNORE INTO dim_txn_type (txn_type, description) VALUES (?, ?)",
                    [(t, f"{t} payments") for t in types])
    conn.commit()

def generate_users(conn, num_users):
    cur = conn.cursor()
    created = 0
    user_insert_sql = "INSERT INTO dim_user (user_id, uuid, created_ts, is_power_user, dormant_after_one_txn, first_name, last_name, email) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    next_id = 1
    power_user_count = max(1, int(num_users * POWER_USER_FRACTION))
    power_user_ids = set(range(1, power_user_count+1))  # first chunk are power users for simplicity
    # shuffle power set to avoid extreme clustering
    # decide dormants randomly
    dormant_ids = set(random.sample(range(1, num_users+1), int(num_users * DORMANT_USER_PCT)))
    start = COHORT_START
    end = NOW
    rows = []
    for uid in range(1, num_users+1):
        created_ts = rand_timestamp_between(start, end).isoformat()
        is_power = 1 if uid in power_user_ids else 0
        dormant = 1 if uid in dormant_ids else 0
        # lightweight fake names/emails
        fn = f"User{uid}"
        ln = "X"
        email = f"user{uid}@example.com"
        rows.append((uid, str(uuid.uuid4()), created_ts, is_power, dormant, fn, ln, email))
        if len(rows) >= BATCH_USERS:
            cur.executemany(user_insert_sql, rows)
            conn.commit()
            rows = []
    if rows:
        cur.executemany(user_insert_sql, rows)
        conn.commit()

    # create some initial KYC events randomly
    kyc_sql = "INSERT INTO fact_kyc_events (user_id, kyc_level, kyc_ts, status) VALUES (?, ?, ?, ?)"
    kyc_rows = []
    for uid in range(1, num_users+1):
        # 60% users have KYC level 1, 25% level 2, rest none or pending
        r = random.random()
        if r < 0.6:
            level = "basic"
            status = "completed"
        elif r < 0.85:
            level = "enhanced"
            status = "completed"
        else:
            level = "basic"
            status = "pending"
        kyc_rows.append((uid, level, rand_timestamp_between(COHORT_START, NOW).isoformat(), status))
        if len(kyc_rows) >= BATCH_USERS:
            cur.executemany(kyc_sql, kyc_rows)
            conn.commit()
            kyc_rows = []
    if kyc_rows:
        cur.executemany(kyc_sql, kyc_rows)
        conn.commit()

def generate_merchants(conn, merchant_count):
    cur = conn.cursor()
    rows = []
    categories = ["retail", "grocery", "entertainment", "utilities", "education", "travel", "services"]
    for mid in range(1, merchant_count+1):
        name = f"Merchant_{mid}"
        cat = random.choice(categories)
        ts = rand_timestamp_between(COHORT_START, NOW).isoformat()
        rows.append((mid, name, cat, ts))
        if len(rows) >= 1000:
            cur.executemany("INSERT INTO dim_merchant (merchant_id, name, category, created_ts) VALUES (?, ?, ?, ?)", rows)
            conn.commit()
            rows = []
    if rows:
        cur.executemany("INSERT INTO dim_merchant (merchant_id, name, category, created_ts) VALUES (?, ?, ?, ?)", rows)
        conn.commit()

def generate_gateways(conn, gateway_count):
    cur = conn.cursor()
    rows = []
    regions = ["IN", "US", "EU", "APAC", "LATAM"]
    for gid in range(1, gateway_count+1):
        rows.append((gid, f"Gateway_{gid}", random.choice(regions), rand_timestamp_between(COHORT_START, NOW).isoformat()))
    cur.executemany("INSERT INTO dim_gateway (gateway_id, name, region, created_ts) VALUES (?, ?, ?, ?)", rows)
    conn.commit()

def generate_instruments(conn, num_users):
    cur = conn.cursor()
    rows = []
    inst_id = 1
    for uid in range(1, num_users+1):
        count = random.choice([1, INSTRUMENTS_PER_USER]) if random.random() < 0.8 else 1
        for _ in range(count):
            itype = random.choices(INSTRUMENT_TYPES, weights=[2,2,1,4,1,3], k=1)[0]
            last4 = f"{random.randint(0,9999):04d}"
            provider = "Bank" if "bank" in itype else ("WalletCo" if "wallet" in itype else "CardNet")
            ts = rand_timestamp_between(COHORT_START, NOW).isoformat()
            rows.append((inst_id, uid, itype, last4, provider, ts))
            inst_id += 1
            if len(rows) >= 5000:
                cur.executemany("INSERT INTO dim_instrument (instrument_id, user_id, inst_type, last4, provider, created_ts) VALUES (?, ?, ?, ?, ?, ?)", rows)
                conn.commit()
                rows = []
    if rows:
        cur.executemany("INSERT INTO dim_instrument (instrument_id, user_id, inst_type, last4, provider, created_ts) VALUES (?, ?, ?, ?, ?, ?)", rows)
        conn.commit()

def prepare_user_sampling_weights(conn, num_users):
    """Return list of user ids and weights to sample users respecting power user skew."""
    # 20% are power users -> 60% txns
    # We'll assign weights: power users weight=POWER_USER_WEIGHT, others=1
    power_count = int(num_users * POWER_USER_FRACTION)
    power_ids = list(range(1, power_count+1))
    normal_ids = list(range(power_count+1, num_users+1))
    ids = power_ids + normal_ids
    weights = [POWER_USER_WEIGHT]*len(power_ids) + [1.0]*len(normal_ids)
    # normalize not necessary for random.choices
    return ids, weights

def user_created_ts(conn, user_id):
    cur = conn.cursor()
    cur.execute("SELECT created_ts FROM dim_user WHERE user_id = ?", (user_id,))
    r = cur.fetchone()
    return datetime.fromisoformat(r[0]) if r and r[0] else COHORT_START

def pick_instrument_for_user(conn, user_id):
    cur = conn.cursor()
    cur.execute("SELECT instrument_id, inst_type FROM dim_instrument WHERE user_id = ? LIMIT 10", (user_id,))
    rows = cur.fetchall()
    if not rows:
        # fallback create a bank instrument
        return None, "bank_account"
    inst = random.choice(rows)
    return inst[0], inst[1]

def generate_transactions_and_related(conn, num_txns, num_users, merchant_count):
    cur = conn.cursor()

    user_ids, user_weights = prepare_user_sampling_weights(conn, num_users)
    gateway_ids = [row[0] for row in cur.execute("SELECT gateway_id FROM dim_gateway").fetchall()]
    merchant_ids = [row[0] for row in cur.execute("SELECT merchant_id FROM dim_merchant").fetchall()]

    txn_insert_sql = """INSERT INTO fact_transactions
    (txn_id, user_id, merchant_id, instrument_id, gateway_id, txn_type, amount, currency, status, created_ts, attempt_count, is_fraud, is_incentive_applied, kyc_level)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    attempt_insert_sql = """INSERT INTO fact_payment_attempts (txn_id, attempt_no, gateway_id, status, attempt_ts, response_code) VALUES (?, ?, ?, ?, ?, ?)"""
    revenue_insert_sql = "INSERT INTO fact_revenue (txn_id, revenue_component, amount, created_ts) VALUES (?, ?, ?, ?)"
    incentive_insert_sql = "INSERT INTO fact_incentives (txn_id, user_id, incentive_type, amount, created_ts) VALUES (?, ?, ?, ?, ?)"
    fraud_insert_sql = "INSERT INTO fact_fraud_loss (txn_id, detected_ts, loss_amount, reason) VALUES (?, ?, ?, ?)"

    txn_rows = []
    attempt_rows = []
    revenue_rows = []
    incentive_rows = []
    fraud_rows = []

    txn_id = 1

    # Keep a counter of how many txns each user has for incentive logic and new-user behavior
    user_txn_count = [0] * (num_users + 1)

    # Pre-compute target numbers per txn_type for more control (optional)
    types = list(TXN_TYPE_DISTR.keys())
    type_probs = [TXN_TYPE_DISTR[t] for t in types]

    # For faster user info lookup, load created_ts for all users
    cur.execute("SELECT user_id, created_ts, is_power_user FROM dim_user")
    usermeta = {r[0]: (datetime.fromisoformat(r[1]), r[2]) for r in cur.fetchall()}

    success_rate = GLOBAL_SUCCESS_RATE

    for batch_start in range(0, num_txns, BATCH_TXNS):
        batch_end = min(num_txns, batch_start + BATCH_TXNS)
        for _ in range(batch_start, batch_end):
            # choose user with power weight
            user = random.choices(user_ids, weights=user_weights, k=1)[0]
            u_created_ts, u_is_power = usermeta.get(user, (COHORT_START, 0))

            # Dormant logic: if user designated dormant and already had 1 txn, skip generating more for them
            if u_is_power == 0:
                # not strictly necessary but helps simulate dormancy
                pass

            # choose txn type
            txn_type = choose_txn_type()

            # amount
            amount = lognormal_amount()
            # scale amount per txn_type
            if txn_type == "billpay":
                amount *= 1.8
            elif txn_type == "p2p":
                amount *= 0.9
            elif txn_type == "recharge":
                amount = max(10.0, min(1000.0, amount * 0.25))

            amount = round(amount, 2)

            # pick merchant (billpay may have special merchant)
            merchant = random.choice(merchant_ids) if merchant_ids else None

            # pick instrument
            inst_id, inst_type = pick_instrument_for_user(conn, user)
            # fallback
            if inst_id is None:
                inst_id = None
                inst_type = "bank_account"

            # pick gateway
            gateway = random.choice(gateway_ids)

            # created timestamp
            created_ts = rand_timestamp_between(COHORT_START, NOW)

            # new user logic: if txn within NEW_USER_DAYS of user's created
            user_age_days = (created_ts - u_created_ts).days
            is_new_user = user_age_days < NEW_USER_DAYS

            # user txn ordinal
            user_txn_count[user] += 1
            ordinal = user_txn_count[user]

            # base success probability may be lowered for new user first 3 txns
            success_prob = success_rate
            if is_new_user and ordinal <= 3:
                # reduce success prob
                success_prob -= 0.25  # heavy initial failure propensity
            # also simulate random slight variability by instrument/provider
            if inst_type in ("card_prepaid", "card_credit"):
                success_prob -= 0.02

            status = "success" if random.random() < success_prob else "failed"

            # Payment attempts logic
            if status == "success":
                attempts = random.randint(MAX_ATTEMPTS_SUCCESS[0], MAX_ATTEMPTS_SUCCESS[1])
            else:
                # failed txns more retries
                attempts = random.randint(MAX_ATTEMPTS_FAIL[0], MAX_ATTEMPTS_FAIL[1])
                # small extra chance for extra retries
                if random.random() < 0.12:
                    attempts += 1
                    attempts = min(attempts, 5)

            # fraud decision: only on success per requirement
            is_fraud = 0
            if status == "success":
                # base fraud on amount, newness, inst_type
                fraud_prob = GLOBAL_FRAUD_BASE
                if amount > 2000:
                    fraud_prob += min(0.25, (amount - 2000)/20000)
                if is_new_user:
                    fraud_prob *= 1.8
                if inst_type in FRAUD_INSTRUMENTS:
                    fraud_prob *= 1.7
                # slight random noise
                if random.random() < fraud_prob:
                    is_fraud = 1

            # incentive logic (first N txns)
            is_incentive = 1 if ordinal <= INCENTIVE_TXNS and status == "success" else 0

            # kyc level (for analytics)
            kyc_level = random.choice(["basic", "enhanced"]) if random.random() < 0.8 else "none"

            txn_rows.append((txn_id, user, merchant, inst_id, gateway, txn_type, amount, "INR", status, created_ts.isoformat(), attempts, is_fraud, is_incentive, kyc_level))

            # payment attempts rows
            for attempt_no in range(1, attempts+1):
                attempt_time = created_ts + timedelta(seconds=attempt_no * random.randint(10, 300))
                # attempts have status distribution: last attempt determines txn status
                if attempt_no < attempts:
                    a_status = "failed" if status == "failed" else ("success" if random.random() < 0.85 else "failed")
                else:
                    a_status = status
                response_code = "00" if a_status == "success" else random.choice(["05", "91", "99"])
                attempt_rows.append((txn_id, attempt_no, gateway, a_status, attempt_time.isoformat(), response_code))

            # revenue rows (1-3) only for successful txns
            if status == "success":
                num_rev_rows = random.randint(REVENUE_ROWS_PER_TXN[0], REVENUE_ROWS_PER_TXN[1])
                # revenue split proportional to type
                base_cut = {
                    "merchant_qr": 0.02,
                    "p2p": 0.012,
                    "billpay": 0.04,
                    "recharge": 0.03
                }.get(txn_type, 0.02)
                # total revenue is a small slice of amount
                total_revenue = round(amount * base_cut, 2)
                # split across components
                splits = []
                if num_rev_rows == 1:
                    splits = [("commission", total_revenue)]
                else:
                    # create components: commission, gst, gateway_fee etc.
                    comps = ["commission", "gateway_fee", "tax"]
                    chosen = comps[:num_rev_rows]
                    # random proportional split
                    parts = [random.random() for _ in range(num_rev_rows)]
                    ssum = sum(parts)
                    for i, comp in enumerate(chosen):
                        amt = round(total_revenue * parts[i] / ssum, 2)
                        splits.append((comp, amt))
                for comp, amt in splits:
                    revenue_rows.append((txn_id, comp, amt, created_ts.isoformat()))

            # incentives rows for first INCENTIVE_TXNS
            if is_incentive:
                # simple cashback model: 1-3% of amount capped
                inc_amount = round(min(500.0, amount * random.uniform(0.01, 0.03)), 2)
                incentive_rows.append((txn_id, user, "cashback", inc_amount, created_ts.isoformat()))

            # fraud loss if marked
            if is_fraud:
                loss = round(amount * random.uniform(0.5, 0.95), 2)
                fraud_rows.append((txn_id, datetime.utcnow().isoformat(), loss, "suspicious_activity"))

            # commit in batches
            txn_id += 1

        # Bulk insert batch
        cur.executemany(txn_insert_sql, txn_rows)
        cur.executemany(attempt_insert_sql, attempt_rows)
        if revenue_rows:
            cur.executemany(revenue_insert_sql, revenue_rows)
        if incentive_rows:
            cur.executemany(incentive_insert_sql, incentive_rows)
        if fraud_rows:
            cur.executemany(fraud_insert_sql, fraud_rows)
        conn.commit()

        # reset buffers
        txn_rows = []
        attempt_rows = []
        revenue_rows = []
        incentive_rows = []
        fraud_rows = []

        # progress print
        print(f"Inserted txns up to {batch_end}")

def main():
    parser = argparse.ArgumentParser(description="Generate Fintech SQLite dataset")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite DB filename")
    parser.add_argument("--num_users", type=int, default=DEFAULT_NUM_USERS, help="Number of users")
    parser.add_argument("--num_txns", type=int, default=DEFAULT_NUM_TXNS, help="Number of transactions")
    args = parser.parse_args()

    num_users = args.num_users
    num_txns = args.num_txns

    print(f"Creating DB {args.db} with {num_users} users and {num_txns} txns...")

    conn = sqlite3.connect(args.db, timeout=30)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    create_schema(conn)
    seed_dim_txn_type(conn)

    print("Generating users...")
    generate_users(conn, num_users)
    print("Generating merchants...")
    generate_merchants(conn, MERCHANT_COUNT)
    print("Generating gateways...")
    generate_gateways(conn, GATEWAY_COUNT)
    print("Generating instruments...")
    generate_instruments(conn, num_users)

    print("Generating transactions and related tables (this can take a while)...")
    generate_transactions_and_related(conn, num_txns, num_users, MERCHANT_COUNT)

    print("Done. VACUUM recommended if you want to compact DB.")
    conn.close()

if __name__ == "__main__":
    main()
