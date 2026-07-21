"""
Sends a batch of realistic, varied requests to the running /predict endpoint —
useful for populating the dashboard and getting drift_check.py enough data to
work with (it needs 30+ logged predictions).

Usage:
    python scripts/generate_traffic.py            # sends 50 requests
    python scripts/generate_traffic.py --n 100     # sends 100 requests
"""
import argparse
import random
import sys

import requests

API_URL = "http://127.0.0.1:8000/predict"
API_KEY = "dev-key-change-me"


def random_payload():
    return {
        "tenure_months": random.randint(0, 72),
        "monthly_charges": round(random.uniform(15, 250), 2),
        "total_charges": round(random.uniform(0, 15000), 2),
        "contract_type": random.choice(["month-to-month", "one-year", "two-year"]),
        "internet_service": random.choice(["DSL", "Fiber optic", "No"]),
        "has_tech_support": random.choice([True, False]),
        "senior_citizen": random.choice([True, False]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of requests to send")
    args = parser.parse_args()

    headers = {"Content-Type": "application/json", "X-API-Key": API_KEY}
    sent, failed = 0, 0

    for i in range(args.n):
        payload = random_payload()
        try:
            r = requests.post(API_URL, json=payload, headers=headers, timeout=10)
            if r.status_code == 200:
                sent += 1
            else:
                failed += 1
                print(f"  request {i}: HTTP {r.status_code} — {r.text[:150]}")
        except requests.exceptions.ConnectionError:
            print("Could not connect. Is the server running at " + API_URL + " ?")
            sys.exit(1)

        if (i + 1) % 10 == 0:
            print(f"  sent {i + 1}/{args.n}...")

    print(f"\nDone: {sent} succeeded, {failed} failed.")
    print("Check the dashboard: http://127.0.0.1:8000/dashboard")
    print("Or run: python src/monitoring/drift_check.py")


if __name__ == "__main__":
    main()
