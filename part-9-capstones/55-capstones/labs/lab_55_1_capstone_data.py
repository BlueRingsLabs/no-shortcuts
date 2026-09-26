# %% [markdown]
# # Capstone data: a small online shop, with the usual problems
#
# Generates the raw data for Capstones A and B: a fictional online shop's operational exports and event streams, over
# nine months, with the problems real sources have. The problems are planted on purpose and listed (behind a spoiler
# block) in 55.1. Finding them is part of the capstone. The generator is seeded: the same arguments give the same
# bytes, so your pipeline's outputs can be compared with other people's.
#
#     python part-9-capstones/55-capstones/labs/lab_55_1_capstone_data.py --out capstone-data --customers 20000
#
# Files (all UTF-8; timestamps ISO 8601):
#   customers_cdc.jsonl   change-data-capture log of the customers table: op (c/u/d), ts, before, after
#   products.csv          current product catalog
#   price_changes.csv     every price change (product_id, new_price, changed_at)
#   order_events.jsonl    order lifecycle events from a queue (created, paid, shipped, delivered, refunded)
#   order_items.csv       lines of each order
#   web_events.jsonl      clickstream: page views, searches, add-to-cart, with a session id
#   support_tickets.csv   customer support tickets with free text
# With no arguments it writes a small version to a temporary directory and checks it, which is what CI runs.

# %%
import argparse
import csv
import hashlib
import json
import random
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

START = datetime(2025, 1, 1, tzinfo=timezone.utc)
DAYS = 273                                                                          # to the end of September
CATEGORIES = {"candles": (8, 40), "soap": (4, 15), "diffusers": (15, 60), "gift sets": (25, 90), "accessories": (3, 25)}
COUNTRIES = ["ES", "FR", "DE", "IT", "PT", "NL", "IE", "BE"]
CHANNELS = ["organic", "paid search", "social", "newsletter"]


def iso(t):
    return t.isoformat().replace("+00:00", "Z")


def generate(out: Path, n_customers: int, seed: int = 55):
    r = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    day = lambda d: START + timedelta(days=d)

    # products and prices; a catalog-wide price rise on day 150 (SCD type 2 territory)
    products, price_changes = [], []
    for pid in range(1, 301):
        cat = r.choice(list(CATEGORIES))
        lo, hi = CATEGORIES[cat]
        price = round(r.uniform(lo, hi), 2)
        products.append({"product_id": f"P{pid:04d}", "category": cat, "name": f"{cat.title()} #{pid}",
                         "price_eur": price})
        price_changes.append({"product_id": f"P{pid:04d}", "new_price_eur": price, "changed_at": iso(START)})
        for d in sorted(r.sample(range(10, DAYS), r.randint(0, 3)) + ([150] if r.random() < 0.8 else [])):
            price = round(price * (1.12 if d == 150 else r.uniform(0.85, 1.1)), 2)
            price_changes.append({"product_id": f"P{pid:04d}", "new_price_eur": price, "changed_at": iso(day(d))})
    price_at = {}
    for pc in sorted(price_changes, key=lambda x: x["changed_at"]):
        price_at.setdefault(pc["product_id"], []).append((pc["changed_at"], pc["new_price_eur"]))

    def price_on(pid, t):
        p = price_at[pid][0][1]
        for ts, v in price_at[pid]:
            if ts <= iso(t):
                p = v
        return p

    # customers, via a CDC log; a partner channel appears on day 120 with very different behavior
    cdc, customers = [], []
    for cid in range(1, n_customers + 1):
        d0 = r.randint(0, DAYS - 1)
        channel = "partner marketplace" if d0 >= 120 and r.random() < 0.35 else r.choice(CHANNELS)
        c = {"customer_id": cid, "email": f"customer{cid}@example.test", "country": r.choice(COUNTRIES),
             "channel": channel, "birth_year": r.randint(1950, 2006), "created_at": iso(day(d0) + timedelta(seconds=r.randint(0, 86399)))}
        if r.random() < 0.01:
            c["country"] = None                                                     # missing
        if d0 >= 200:
            c["marketing_opt_in"] = r.random() < 0.4                                # schema change on day 200
        # engagement: how often they buy; partner customers buy once and vanish
        c["_rate"] = (0.004 if channel == "partner marketplace" else r.choice([0.01, 0.03, 0.08])) * r.uniform(0.5, 1.5)
        customers.append(c)
        public = {k: v for k, v in c.items() if not k.startswith("_")}
        cdc.append({"op": "c", "ts": c["created_at"], "before": None, "after": public})
        t = day(d0)
        while r.random() < 0.3:                                                     # updates: moves, email changes
            t += timedelta(days=r.randint(1, 60))
            if t >= day(DAYS):
                break
            before = dict(public)
            if r.random() < 0.5:
                public["country"] = r.choice(COUNTRIES)
            else:
                public["email"] = f"customer{cid}.{r.randint(1, 99)}@example.test"
            cdc.append({"op": "u", "ts": iso(t), "before": before, "after": dict(public)})
        if r.random() < 0.02:                                                       # GDPR deletions
            t2 = t + timedelta(days=r.randint(1, 90))
            if t2 < day(DAYS):
                cdc.append({"op": "d", "ts": iso(t2), "before": dict(public), "after": None})
                c["_deleted"] = t2

    # orders and their events; web sessions around them
    order_events, items, web, tickets = [], [], [], []
    oid, eid, sid = 0, 0, 0
    for c in customers:
        t = datetime.fromisoformat(c["created_at"].replace("Z", "+00:00"))
        end = min(day(DAYS), c.get("_deleted", day(DAYS)))
        rate = c["_rate"]
        first = True
        while True:
            t += timedelta(days=r.expovariate(rate) if not first else r.uniform(0, 2))
            first = False
            if t >= end:
                break
            sid += 1
            session = f"S{sid:08d}"
            for k in range(r.randint(2, 12)):                                       # browsing before the order
                web.append({"session_id": session, "customer_id": c["customer_id"] if r.random() > 0.1 else None,
                            "ts": iso(t - timedelta(minutes=30 - 2 * k)), "event": r.choice(["page_view", "page_view", "search", "add_to_cart"]),
                            "product_id": f"P{r.randint(1, 300):04d}"})
            if r.random() < 0.25:                                                   # abandoned session, no order
                continue
            oid += 1
            order_id = f"O{oid:08d}"
            n_lines = r.choice([1, 1, 1, 2, 2, 3, 4])
            total = 0.0
            for _ in range(n_lines):
                p = r.choice(products)
                qty = r.choice([1, 1, 1, 2, 3])
                if r.random() < 0.001:
                    qty = -qty                                                      # a bug upstream
                unit = price_on(p["product_id"], t)
                if day(97) <= t < day(98):
                    unit = round(unit * 100)                                        # one day exported in cents
                items.append({"order_id": order_id, "product_id": p["product_id"], "quantity": qty, "unit_price_eur": unit})
                total += qty * unit
            refunded = r.random() < (0.12 if c["channel"] == "partner marketplace" else 0.04)
            steps = [("created", 0), ("paid", r.uniform(0.01, 0.3)), ("shipped", r.uniform(12, 48)), ("delivered", r.uniform(48, 150))]
            if refunded:
                steps.append(("refunded", r.uniform(160, 500)))
            for status, hours in steps:
                te = t + timedelta(hours=hours)
                if te >= day(DAYS):
                    break
                ingested = te + timedelta(seconds=r.uniform(1, 30))
                if r.random() < 0.02:
                    ingested += timedelta(hours=r.uniform(24, 72))                  # late arrivals
                ts_field = iso(te)
                if day(210) <= te < day(211) and r.random() < 0.9:                  # a producer switched to local time
                    ts_field = (te + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")   # ...and dropped the offset
                eid += 1
                ev = {"event_id": f"E{eid:09d}", "order_id": order_id, "customer_id": c["customer_id"], "status": status,
                      "event_ts": ts_field, "ingested_ts": iso(ingested), "amount_eur": round(total, 2) if status in ("paid", "refunded") else None}
                order_events.append(ev)
                if r.random() < 0.01:
                    order_events.append(dict(ev))                                   # at-least-once delivery: duplicates
            if r.random() < 0.03:
                topic = r.choice(["late delivery", "broken item", "wrong item", "refund status", "account"])
                tickets.append({"ticket_id": f"T{len(tickets) + 1:06d}", "customer_id": c["customer_id"], "order_id": order_id,
                                "opened_at": iso(t + timedelta(days=r.uniform(2, 20))), "topic": topic,
                                "text": f"Hi, about order {order_id}: {topic}. My email is {c['email']}, please call me back."})

    order_events.sort(key=lambda e: e["ingested_ts"])                              # the queue delivers in arrival order
    cdc.sort(key=lambda e: e["ts"])
    web.sort(key=lambda e: e["ts"])

    def write_jsonl(name, rows):
        with open(out / name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    def write_csv(name, rows):
        with open(out / name, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    write_jsonl("customers_cdc.jsonl", cdc)
    write_csv("products.csv", products)
    write_csv("price_changes.csv", price_changes)
    write_jsonl("order_events.jsonl", order_events)
    write_csv("order_items.csv", items)
    write_jsonl("web_events.jsonl", web)
    write_csv("support_tickets.csv", tickets)
    return {"customers": len(customers), "cdc": len(cdc), "orders": oid, "order_events": len(order_events),
            "order_items": len(items), "web_events": len(web), "tickets": len(tickets)}


def fingerprint(out: Path):
    h = hashlib.sha256()
    for p in sorted(out.iterdir()):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()[:16]


def check(out: Path):
    """Sanity checks that the planted problems are really there (without saying where: see 55.1)."""
    events = [json.loads(l) for l in open(out / "order_events.jsonl", encoding="utf-8")]
    ids = [e["event_id"] for e in events]
    dup_rate = 1 - len(set(ids)) / len(ids)
    no_offset = sum(1 for e in events if not e["event_ts"].endswith("Z"))
    late = sum(1 for e in events if e["event_ts"].endswith("Z") and
               (datetime.fromisoformat(e["ingested_ts"].replace("Z", "+00:00")) -
                datetime.fromisoformat(e["event_ts"].replace("Z", "+00:00"))).total_seconds() > 86400)
    ops = [json.loads(l)["op"] for l in open(out / "customers_cdc.jsonl", encoding="utf-8")]
    items = list(csv.DictReader(open(out / "order_items.csv", encoding="utf-8")))
    return {"duplicate event rate": round(dup_rate, 4), "events without a UTC offset": no_offset,
            "events ingested more than a day late": late, "CDC ops": {o: ops.count(o) for o in "cud"},
            "negative quantities": sum(int(i["quantity"]) < 0 for i in items)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: a temporary one, then deleted)")
    ap.add_argument("--customers", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=55)
    args = ap.parse_args()
    tmp = None
    if args.out is None:
        tmp = tempfile.TemporaryDirectory()
        args.out = Path(tmp.name)
    counts = generate(args.out, args.customers, args.seed)
    print(f"wrote {args.out}:")
    for k, v in counts.items():
        print(f"  {k:14s} {v:>10,d}")
    stats = check(args.out)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"fingerprint {fingerprint(args.out)} (same arguments, same fingerprint)")
    if tmp is not None:                                                               # CI: check determinism too
        with tempfile.TemporaryDirectory() as again:
            generate(Path(again), args.customers, args.seed)
            assert fingerprint(Path(again)) == fingerprint(args.out), "generator is not deterministic"
    assert 0.005 < stats["duplicate event rate"] < 0.02 and stats["events without a UTC offset"] > 0
    assert stats["events ingested more than a day late"] > 0 and stats["CDC ops"]["d"] > 0
    print("\nAll checks passed.")
