# Shared lab helpers for Part II

`shop.py` generates the "Rat & Co." online shop: customers, products, orders, order items and a clickstream, deterministic
for a given seed, with an optional `dirty=True` mode that injects the kind of garbage real data has (duplicates, orphans,
bad emails, inconsistent codes, negative quantities, prices stored as strings). Lesson 10.3 asks you to find all of it.

Run `python shop.py` to see the table sizes.
