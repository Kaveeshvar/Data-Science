# gen_fintech_data.py
import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------ Controls (tune these) ------------
NUM_USERS = 10_000
NUM_TXNS = 200_000             # "small but real"
MAX_ATTEMPTS_PER_TXN = 3

START_DATE = datetime(2025, 7, 1)
END_DATE   = datetime(2025, 12, 31)

CHANNELS = ["organic", "paid_search", "paid_social", "referral", "partners"]
CITIES = ["BLR", "HYD", "DEL", "MUM", "CHE", "PUN", "KOL"]
RISK_SEG = ["low", "med", "high"]

TXN_TYPES = ["merchant_qr", "billpay", "p2p", "recharge"]
TXN_TYPE_PROBS = [0.70, 0.15, 0.10, 0.05]

INSTRUMENTS = ["UPI", "CARD", "WALLET_BAL"]
INSTRUMENT_PROBS = [0.75, 0.15, 0.10]

GATEWAYS = ["gpay", "razorpay", "payu", "cashfree"]
GATEWAY_PROBS = [0.35, 0.30, 0.20, 0.15]

# Base success rates by instrument (attempt-level); txn success emerges from attempts
BASE_SUCCESS_BY_INSTR = {"UPI": 0.90, "CARD": 0.93, "WALLET_BAL": 0.98}

# New users have more friction in first 7 days
NEW_USER_PENALTY = 0.06

# Fraud modeling: higher for new users + high amount + high risk segment
BASE_FRAUD_RATE = 0.018  # ~1.8% of successful txns (small but meaningful)

# Incentives: only for first 3 successful txns of a user
INCENTIVE_COVERAGE = 0.40  # of eligible first-3 txns
INCENTIVE_AMT = {"merchant_qr": 2.0, "billpay": 8.0, "p2p": 1.0, "recharge": 3.0}

# Revenue per txn type for successful txns (split into 1-2 components)
REV_PER_TXN = {"merchant_qr": 0.10, "billpay": 0.80, "p2p": 0.02, "recharge": 0.25}
REV_COMPONENTS = {
    "merchant_qr": ["mdr"],
    "billpay": ["commission", "fee"],
    "p2p": ["fee"],
    "recharge": ["commission"]
}

# Variable cost per successful txn type
COST_PER_TXN = {"merchant_qr": 0.04, "billpay": 0.20, "p2p": 0.01, "recharge": 0.08}

# ------------ Helpers ------------
def rand_ts(start: datetime, end: datetime) -> datetime:
    delta = end - start
    seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=seconds)

def iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")

# ------------ 1) Users + KYC ------------
users = []
kyc_events = []

for uid in range(1, NUM_USERS + 1):
    signup_ts = rand_ts(START_DATE, END_DATE - timedelta(days=7))
    channel = random.choices(CHANNELS, weights=[40, 20, 15, 15, 10], k=1)[0]
    city = random.choice(CITIES)
    risk = random.choices(RISK_SEG, weights=[70, 22, 8], k=1)[0]

    # KYC: approval probability depends on risk
    kyc_prob = {"low": 0.85, "med": 0.75, "high": 0.60}[risk]
    kyc_done = random.random() < kyc_prob

    kyc_status = "not_started"
    kyc_approved_ts = None

    if kyc_done:
        kyc_status = "approved"
        # approval happens 0-4 days after signup
        appr = signup_ts + timedelta(hours=random.randint(1, 96))
        kyc_approved_ts = iso(appr)

        kyc_events.append({
            "event_id": f"kyc_{uid}_1",
            "user_id": uid,
            "event_ts": iso(signup_ts + timedelta(hours=random.randint(0, 6))),
            "status": "started"
        })
        kyc_events.append({
            "event_id": f"kyc_{uid}_2",
            "user_id": uid,
            "event_ts": iso(appr),
            "status": "approved"
        })
    else:
        # Some start but fail
        if random.random() < 0.35:
            kyc_status = "rejected"
            rej = signup_ts + timedelta(hours=random.randint(1, 72))
            kyc_events.append({
                "event_id": f"kyc_{uid}_1",
                "user_id": uid,
                "event_ts": iso(signup_ts + timedelta(hours=random.randint(0, 6))),
                "status": "started"
            })
            kyc_events.append({
                "event_id": f"kyc_{uid}_2",
                "user_id": uid,
                "event_ts": iso(rej),
                "status": "rejected"
            })

    users.append({
        "user_id": uid,
        "signup_ts": iso(signup_ts),
        "channel": channel,
        "city": city,
        "risk_segment": risk,
        "kyc_status": kyc_status,
        "kyc_approved_ts": kyc_approved_ts
    })

dim_users = pd.DataFrame(users)
fact_kyc = pd.DataFrame(kyc_events)

# ------------ 2) Transactions + Attempts + Revenue + Incentives + Fraud ------------
# User activity skew: lognormal -> heavy tail
activity_weights = np.random.lognormal(mean=0.0, sigma=1.0, size=NUM_USERS)
activity_probs = activity_weights / activity_weights.sum()
user_ids = np.random.choice(np.arange(1, NUM_USERS + 1), size=NUM_TXNS, p=activity_probs)

tx_rows = []
attempt_rows = []
rev_rows = []
incentive_rows = []
fraud_rows = []

# track first-3 successful txns per user
success_count = {uid: 0 for uid in range(1, NUM_USERS + 1)}

# quick user lookup
user_signup = dict(zip(dim_users.user_id, pd.to_datetime(dim_users.signup_ts)))
user_risk = dict(zip(dim_users.user_id, dim_users.risk_segment))

for i, uid in enumerate(user_ids, start=1):
    txn_id = f"txn_{i}"
    u_signup = user_signup[uid]
    txn_ts = rand_ts(max(u_signup, START_DATE), END_DATE)

    txn_type = random.choices(TXN_TYPES, weights=TXN_TYPE_PROBS, k=1)[0]
    instr = random.choices(INSTRUMENTS, weights=INSTRUMENT_PROBS, k=1)[0]

    # Amount distributions by txn type (skewed)
    if txn_type == "merchant_qr":
        amt = float(np.random.lognormal(mean=5.3, sigma=0.6))   # ~200-400 typical, heavy tail
    elif txn_type == "billpay":
        amt = float(np.random.lognormal(mean=6.3, sigma=0.5))   # ~500-1500 typical
    elif txn_type == "p2p":
        amt = float(np.random.lognormal(mean=5.0, sigma=0.7))   # ~120-300
    else:  # recharge
        amt = float(np.random.lognormal(mean=5.6, sigma=0.4))   # ~250-500

    amt = round(max(10.0, min(amt, 50_000.0)), 2)

    # Determine attempt-level success probability with new-user penalty
    days_since_signup = (txn_ts - u_signup).days
    p_success = BASE_SUCCESS_BY_INSTR[instr]
    if days_since_signup <= 7:
        p_success = max(0.50, p_success - NEW_USER_PENALTY)

    # simulate attempts
    num_attempts = random.randint(1, MAX_ATTEMPTS_PER_TXN)
    gateway = random.choices(GATEWAYS, weights=GATEWAY_PROBS, k=1)[0]
    final_status = "fail"
    error_code = None
    attempt_ts = txn_ts

    for a in range(1, num_attempts + 1):
        attempt_id = f"att_{i}_{a}"
        # failures more likely earlier attempts
        attempt_success = random.random() < p_success
        if attempt_success:
            status = "success"
            final_status = "success"
            error_code = None
        else:
            status = "fail"
            # simplified error codes
            error_code = random.choice(["U01_TIMEOUT", "U02_BANK_DOWN", "U03_INSUFF_FUNDS", "U04_UPI_ERR", "C01_3DS_FAIL"])
            attempt_ts = attempt_ts + timedelta(seconds=random.randint(10, 180))

        attempt_rows.append({
            "attempt_id": attempt_id,
            "txn_id": txn_id,
            "attempt_ts": iso(attempt_ts),
            "gateway": gateway,
            "status": status,
            "error_code": error_code
        })

        if final_status == "success":
            break

    # merchant_id only for merchant_qr
    merchant_id = f"m_{random.randint(1, 2000)}" if txn_type == "merchant_qr" else None

    tx_rows.append({
        "txn_id": txn_id,
        "user_id": uid,
        "txn_ts": iso(txn_ts),
        "txn_type": txn_type,
        "instrument_type": instr,
        "merchant_id": merchant_id,
        "amount": amt,
        "status": final_status
    })

    if final_status == "success":
        # revenue rows (1-2 components)
        base_rev = REV_PER_TXN[txn_type]
        comps = REV_COMPONENTS[txn_type]
        if len(comps) == 1:
            rev_rows.append({"txn_id": txn_id, "revenue_type": comps[0], "revenue_amount": round(base_rev, 4)})
        else:
            # split revenue into two parts
            split = random.uniform(0.3, 0.7)
            rev_rows.append({"txn_id": txn_id, "revenue_type": comps[0], "revenue_amount": round(base_rev * split, 4)})
            rev_rows.append({"txn_id": txn_id, "revenue_type": comps[1], "revenue_amount": round(base_rev * (1 - split), 4)})

        # variable cost as negative revenue not stored here; you'll compute in SQL using COST_PER_TXN
        # incentives for first 3 successful txns
        success_count[uid] += 1
        if success_count[uid] <= 3 and random.random() < INCENTIVE_COVERAGE:
            incentive_rows.append({
                "incentive_id": f"inc_{txn_id}",
                "user_id": uid,
                "txn_id": txn_id,
                "campaign_id": "new_user_cashback",
                "incentive_amount": round(INCENTIVE_AMT[txn_type], 2)
            })

        # fraud correlated with amount + new user + risk segment
        risk = user_risk[uid]
        fraud_p = BASE_FRAUD_RATE
        if days_since_signup <= 7:
            fraud_p *= 1.8
        if risk == "high":
            fraud_p *= 2.0
        elif risk == "med":
            fraud_p *= 1.3
        if amt >= 5000:
            fraud_p *= 1.8

        if random.random() < min(0.25, fraud_p):
            loss = round(min(amt, random.uniform(0.2, 1.0) * amt), 2)
            fraud_rows.append({
                "fraud_id": f"fraud_{txn_id}",
                "txn_id": txn_id,
                "user_id": uid,
                "flagged_ts": iso(txn_ts + timedelta(hours=random.randint(1, 72))),
                "confirmed_flag": 1,
                "loss_amount": loss
            })

fact_tx = pd.DataFrame(tx_rows)
fact_attempts = pd.DataFrame(attempt_rows)
fact_rev = pd.DataFrame(rev_rows)
fact_inc = pd.DataFrame(incentive_rows)
fact_fraud = pd.DataFrame(fraud_rows)

# ------------ Save CSVs ------------
dim_users.to_csv(os.path.join(OUT_DIR, "dim_users.csv"), index=False)
fact_kyc.to_csv(os.path.join(OUT_DIR, "fact_kyc_events.csv"), index=False)
fact_tx.to_csv(os.path.join(OUT_DIR, "fact_transactions.csv"), index=False)
fact_attempts.to_csv(os.path.join(OUT_DIR, "fact_payment_attempts.csv"), index=False)
fact_rev.to_csv(os.path.join(OUT_DIR, "fact_revenue.csv"), index=False)
fact_inc.to_csv(os.path.join(OUT_DIR, "fact_incentives.csv"), index=False)
fact_fraud.to_csv(os.path.join(OUT_DIR, "fact_fraud.csv"), index=False)

print("✅ Generated CSVs in ./data/")
print("Rows:")
print("users:", len(dim_users))
print("kyc_events:", len(fact_kyc))
print("txns:", len(fact_tx))
print("attempts:", len(fact_attempts))
print("revenue_rows:", len(fact_rev))
print("incentives:", len(fact_inc))
print("fraud:", len(fact_fraud))
