# %% [markdown]
# # Lab 14.1: A partitioned log with consumer groups
#
# A small in-process imitation of Kafka's model (topics, partitions, keys, offsets, consumer groups, rebalances), then
# a consumer that crashes at the worst moment under three strategies:
#
# 1. Keyed partitioning keeps per-key order; there's no global order.
# 2. Consumer groups split partitions; independent groups each read everything; replay from an offset.
# 3. At-most-once loses records, at-least-once duplicates them, at-least-once + idempotent sink is correct.
# 4. Lag.

# %%
import hashlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


def stable_hash(key: str) -> int:
    return int.from_bytes(hashlib.md5(key.encode()).digest()[:8], "big")


@dataclass
class Record:
    offset: int
    key: str
    value: dict


class Topic:
    def __init__(self, name, partitions):
        self.name = name
        self.logs = [[] for _ in range(partitions)]

    def produce(self, key, value):
        p = stable_hash(key) % len(self.logs)            # same key -> same partition -> ordered
        log = self.logs[p]
        log.append(Record(len(log), key, value))
        return p, len(log) - 1

    def read(self, partition, offset, max_records=10):
        return self.logs[partition][offset: offset + max_records]

    def end_offsets(self):
        return [len(l) for l in self.logs]


class ConsumerGroup:
    """Tracks committed offsets per partition and assigns partitions to live members."""

    def __init__(self, topic, name):
        self.topic, self.name = topic, name
        self.committed = defaultdict(int)
        self.members = []

    def join(self, member):
        self.members.append(member)
        return self.assignment()

    def leave(self, member):
        self.members.remove(member)
        return self.assignment()

    def assignment(self):                                   # a simple range assignor
        parts = range(len(self.topic.logs))
        return {m: [p for p in parts if p % len(self.members) == i] for i, m in enumerate(self.members)}

    def lag(self):
        return sum(end - self.committed[p] for p, end in enumerate(self.topic.end_offsets()))

# %% [markdown]
# ## 1. Keys and ordering

# %%
rng = np.random.default_rng(14)
orders = Topic("orders", partitions=4)
accounts = [f"acct-{i}" for i in range(20)]
sent = defaultdict(list)
for seq in range(2000):
    acct = accounts[rng.integers(0, len(accounts))]
    orders.produce(acct, {"acct": acct, "seq": seq, "amount": float(rng.gamma(2, 30))})
    sent[acct].append(seq)

for acct in accounts:
    parts = {p for p, log in enumerate(orders.logs) for r in log if r.key == acct}
    assert len(parts) == 1, "all records of a key live in one partition"
    p = parts.pop()
    seqs = [r.value["seq"] for r in orders.logs[p] if r.key == acct]
    assert seqs == sent[acct], "and they're in production order"
interleaved = [r.value["seq"] for log in orders.logs for r in log]
assert interleaved != sorted(interleaved), "but there's no global order across partitions"
print("per-partition sizes:", orders.end_offsets(), "(keys spread by hash)")

# %% [markdown]
# ## 2. Consumer groups, independent readers, replay

# %%
g = ConsumerGroup(orders, "fraud-model")
g.join("c1"); a = g.join("c2")
print("assignment with 2 consumers:", a)
for m in ("c3", "c4", "c5"):
    a5 = g.join(m)
idle = [m for m, ps in a5.items() if not ps]
print("assignment with 5 consumers:", a5, "-> idle:", idle)
assert len(idle) == 1, "4 partitions can keep at most 4 consumers busy"

def consume_all(topic, group, member_parts, sink):
    for p in member_parts:
        while True:
            batch = topic.read(p, group.committed[p])
            if not batch:
                break
            for r in batch:
                sink.append(r.value["seq"])
            group.committed[p] = batch[-1].offset + 1


lake_group, fraud_group = ConsumerGroup(orders, "lake-sink"), ConsumerGroup(orders, "fraud")
lake_sink, fraud_sink = [], []
consume_all(orders, lake_group, range(4), lake_sink)
consume_all(orders, fraud_group, range(4), fraud_sink)
assert sorted(lake_sink) == sorted(fraud_sink) == list(range(2000)), "each group reads everything, independently"

replay = []
lake_group.committed[0] = 0                                  # rewind partition 0 (a bug fix needs reprocessing)
consume_all(orders, lake_group, [0], replay)
assert len(replay) == orders.end_offsets()[0]
print(f"replayed partition 0 from offset 0: {len(replay)} records")

# %% [markdown]
# ## 3. Crashes and delivery semantics
#
# The consumer processes a batch of 10 records, crashing right after processing record #7 of one batch. We compare
# three strategies by the totals they end up with in the sink.

# %%
payments = Topic("payments", partitions=1)
for i in range(100):
    payments.produce("k", {"event_id": f"evt-{i}", "amount": 10.0})
TRUE_TOTAL = 100 * 10.0


def run(strategy, crash_at_offset=57):
    group = ConsumerGroup(payments, strategy)
    sink_rows, sink_by_id = [], {}
    crashed = False
    while True:
        batch = payments.read(0, group.committed[0], 10)
        if not batch:
            break
        if strategy == "at_most_once":
            group.committed[0] = batch[-1].offset + 1          # commit first
        for r in batch:
            if not crashed and r.offset == crash_at_offset:
                crashed = True                                  # crash mid-batch; the "new consumer" restarts the loop
                break
            if strategy == "idempotent":
                sink_by_id[r.value["event_id"]] = r.value["amount"]   # upsert by event id
            else:
                sink_rows.append(r.value["amount"])
        else:
            if strategy != "at_most_once":
                group.committed[0] = batch[-1].offset + 1       # commit after processing
            continue
        # after a crash we just loop again from the last committed offset
    total = sum(sink_by_id.values()) if strategy == "idempotent" else sum(sink_rows)
    return total


results = {s: run(s) for s in ("at_most_once", "at_least_once", "idempotent")}
for s, total in results.items():
    print(f"{s:14s} total {total:7.1f}  (true {TRUE_TOTAL})")
assert results["at_most_once"] < TRUE_TOTAL, "records lost"
assert results["at_least_once"] > TRUE_TOTAL, "records duplicated"
assert results["idempotent"] == TRUE_TOTAL, "effectively once"

# %% [markdown]
# ## 4. Lag

# %%
g = ConsumerGroup(orders, "slow-consumer")
consume_all(orders, g, [0, 1], [])                              # only half the partitions are keeping up
print(f"lag of a group that only drains partitions 0 and 1: {g.lag():,} records")
assert g.lag() == orders.end_offsets()[2] + orders.end_offsets()[3]

print("\nAll checks passed.")
