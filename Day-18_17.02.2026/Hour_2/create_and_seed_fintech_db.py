#!/usr/bin/env python3

import argparse
import sqlite3
import random
from datetime import datetime, timedelta
import uuid

# ------------------------
# CONFIG
# ------------------------

NUM_USERS = 200
NUM_MERCHANTS = 60
DAYS_SPEND = 90
AVG_TXNS_PER_USER = 6

START_DATE = datetime(2025, 1, 1)

CHANNELS = ["google_ads", "meta_ads", "organic", "referral"]
TIERS = ["smb", "mid_market", "enterprise"]
MCC_CODES = ["5411", "5732", "5812", "4111", "4900"]  # grocery, electronics, restaurant, transport, utilities

# ------------------------
# DB Schema
# ------------------------

DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
  user_id INTEGER PRIMARY KEY,
  signup_ts TEXT NOT NULL,
  kyc_ts TEXT,
  activation_ts TEXT,
  acq_channel TEXT,
  campaign_id TEXT,
  adset_id TEXT,
  geo TEXT,
  device_id TEXT,
  risk_flag INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS marketing_spend (
  spend_date TEXT,
  channel TEXT,
  campaign_id TEXT,
  adset_id TEXT,
  spend REAL,
  clicks INTEGER,
  impressions INTEGER,
  PRIMARY KEY (spend_date, channel, campaign_id)
);

CREATE TABLE IF NOT EXISTS merchants (
  merchant_id INTEGER PRIMARY KEY,
  mcc TEXT,
  merchant_tier TEXT,
  risk_flag INTEGER DEFAULT 0,
  created_ts TEXT
);

CREATE TABLE IF NOT EXISTS transactions (
  txn_id INTEGER PRIMARY KEY,
  user_id INTEGER,
  merchant_id INTEGER,
  txn_ts TEXT,
  amount REAL,
  status TEXT,
  rev_interchange REAL,
  rev_mdr_share REAL,
  processing_fee REAL,
  cashback_cost REAL,
  mcc TEXT,
  FOREIGN KEY (user_id) REFERENCES users(user_id),
  FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id)
);

CREATE TABLE IF NOT EXISTS refunds (
  refund_id INTEGER PRIMARY KEY,
  txn_id INTEGER,
  refund_ts TEXT,
  refund_amount REAL,
  refund_reason TEXT,
  refund_status TEXT,
  FOREIGN KEY (txn_id) REFERENCES transactions(txn_id)
);
"""

# ------------------------
# Helper Functions
# ------------------------

def random_ts(start, days_range):
    return (start + timedelta(days=random.randint(0, days_range),
                              hours=random.randint(0, 23),
                              minutes=random.randint(0, 59))).isoformat()

def create_schema(conn):
    conn.executescript(DDL)
    conn.commit()

# ------------------------
# Seed Data
# ------------------------

def seed_merchants(conn):
    for mid in range(1, NUM_MERCHANTS + 1):
        conn.execute("""
            INSERT INTO merchants
            VALUES (?, ?, ?, ?, ?)
        """, (
            mid,
            random.choice(MCC_CODES),
            random.choice(TIERS),
            1 if random.random() < 0.1 else 0,
            random_ts(START_DATE, 120)
        ))
    conn.commit()

def seed_marketing_spend(conn):
    for d in range(DAYS_SPEND):
        date = (START_DATE + timedelta(days=d)).date().isoformat()
        for channel in CHANNELS:
            campaign_id = f"{channel}_camp_1"
            conn.execute("""
                INSERT INTO marketing_spend
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                date,
                channel,
                campaign_id,
                "adset_1",
                round(random.uniform(500, 3000), 2),
                random.randint(200, 2000),
                random.randint(5000, 20000)
            ))
    conn.commit()

def seed_users(conn):
    for uid in range(1, NUM_USERS + 1):
        signup_ts = random_ts(START_DATE, 60)

        kyc_ts = None
        if random.random() < 0.8:
            kyc_ts = (datetime.fromisoformat(signup_ts) +
                      timedelta(days=random.randint(0, 5))).isoformat()

        activation_ts = None
        if random.random() < 0.65:
            activation_ts = (datetime.fromisoformat(signup_ts) +
                             timedelta(days=random.randint(0, 10))).isoformat()

        conn.execute("""
            INSERT INTO users
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            uid,
            signup_ts,
            kyc_ts,
            activation_ts,
            random.choice(CHANNELS),
            "google_ads_camp_1",
            "adset_1",
            random.choice(["IN-KA", "IN-MH", "US-CA"]),
            str(uuid.uuid4())[:12],
            1 if random.random() < 0.1 else 0
        ))
    conn.commit()

def seed_transactions(conn):
    txn_id = 1
    refund_id = 1

    users = conn.execute("SELECT user_id, activation_ts FROM users").fetchall()
    merchants = conn.execute("SELECT merchant_id, mcc FROM merchants").fetchall()

    for user_id, activation_ts in users:
        if activation_ts is None:
            continue

        num_txns = random.randint(1, AVG_TXNS_PER_USER + 3)

        for _ in range(num_txns):
            merchant_id, mcc = random.choice(merchants)
            amount = round(random.uniform(50, 5000), 2)

            status = random.choices(
                ["success", "failed", "reversed"],
                weights=[0.82, 0.13, 0.05]
            )[0]

            rev_interchange = round(amount * 0.015, 2)
            rev_mdr_share = round(amount * 0.005, 2)
            processing_fee = round(amount * 0.004, 2)
            cashback_cost = round(amount * 0.01 if random.random() < 0.3 else 0, 2)

            txn_ts = random_ts(datetime.fromisoformat(activation_ts), 60)

            conn.execute("""
                INSERT INTO transactions
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                txn_id,
                user_id,
                merchant_id,
                txn_ts,
                amount,
                status,
                rev_interchange,
                rev_mdr_share,
                processing_fee,
                cashback_cost,
                mcc
            ))

            # Refund logic only on successful txns
            if status == "success" and random.random() < 0.15:
                conn.execute("""
                    INSERT INTO refunds
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    refund_id,
                    txn_id,
                    random_ts(datetime.fromisoformat(txn_ts), 30),
                    round(amount * random.uniform(0.2, 1.0), 2),
                    random.choice(["customer_request", "fraud", "duplicate"]),
                    random.choice(["approved", "rejected"])
                ))
                refund_id += 1

            txn_id += 1

    conn.commit()

# ------------------------
# Main
# ------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="fintech_learn.db")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    create_schema(conn)
    seed_merchants(conn)
    seed_marketing_spend(conn)
    seed_users(conn)
    seed_transactions(conn)

    conn.close()

    print("Database created with realistic sample data.")
    print("You now have:")
    print("- 200 users")
    print("- 60 merchants")
    print("- 90 days of spend")
    print("- ~1200+ transactions")
    print("- ~150 refunds")

if __name__ == "__main__":
    main()
