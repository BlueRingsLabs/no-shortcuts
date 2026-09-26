# %% [markdown]
# # Lab 54.3: Governance, documentation and regulation
#
# 1. A model card generated from what the registry and the evaluation already know (48.1, 50.1, 54.1), with a check
#    that refuses to register a model whose card has empty sections.
# 2. A first-pass triage of use cases against the EU AI Act's risk tiers, to decide which ones go to legal review
#    and with which obligations in view. It routes work; it is not legal advice.

# %%

# %% [markdown]
# ## 1. A model card from metadata

# %%
REQUIRED = ["model details", "intended use", "out of scope", "training data", "evaluation", "slices and fairness",
            "limitations", "human oversight", "contact"]

metadata = {
    "model details": {"name": "loan-default-scorer", "version": 7, "type": "gradient-boosted trees",
                      "trained": "2026-09-01", "code": "git 3f2c9a1", "data": "sha256 8d49...a28 (loans 2021-2025)"},
    "intended use": "Rank personal-loan applications by estimated default risk to prioritize manual underwriting review.",
    "out of scope": "Automated rejection without human review; business loans; applicants under 18; any use outside the EU.",
    "training data": "1.2M applications 2021-2025 with 12-month repayment outcomes; excludes applications declined "
                     "before 2023 (no outcomes), which under-represents thin-file applicants.",
    "evaluation": {"test period": "2025-07 to 2025-12", "AUC": 0.781, "precision at review capacity (150/day)": 0.34},
    "slices and fairness": {"approval-rate gap (age < 25 vs 25+)": 0.08, "TPR gap (age < 25 vs 25+)": 0.03,
                            "calibration gap (worst slice)": 0.02},
    "limitations": "Trained on a period without a recession; performance under macroeconomic shock is untested. "
                   "Selective labels (declined applicants have no outcome) bias estimates for the riskiest segment.",
    "human oversight": "",                                                           # forgotten
    "contact": "credit-ml@example.test",
}


def render_card(meta):
    lines = [f"# Model card: {meta['model details']['name']} v{meta['model details']['version']}", ""]
    for section in REQUIRED:
        lines.append(f"## {section.capitalize()}")
        v = meta.get(section)
        if isinstance(v, dict):
            lines += [f"- **{k}**: {val}" for k, val in v.items()]
        else:
            lines.append(v or "")
        lines.append("")
    return "\n".join(lines)


def check_card(meta):
    return [s for s in REQUIRED if not meta.get(s)]


card = render_card(metadata)
print(card[:700] + "\n...")
missing = check_card(metadata)
print(f"registration check: missing sections {missing} -> refused")
metadata["human oversight"] = ("Scores go to underwriters with the top three contributing features; underwriters decide and "
                               "record the reason; weekly sample review of disagreements; applicants can request a human review.")
print(f"after completing the card: missing {check_card(metadata)} -> registered")
print("a card generated from the registry, the evaluation report and the fairness audit stays in sync with the model; a")
print("card written by hand once goes stale by the next retraining. Make the card a release artifact, checked like a test.")
assert missing == ["human oversight"] and check_card(metadata) == []

# %% [markdown]
# ## 2. EU AI Act triage
#
# Regulation (EU) 2024/1689 sorts AI systems by risk: prohibited practices (Article 5), high-risk systems (Annex III
# areas, and safety components of regulated products under Annex I), transparency obligations (Article 50), and
# everything else. General-purpose AI models have their own obligations. The questions below are a simplified first
# pass. Obligations phase in over 2025-2027 and the timeline has been the subject of amendment proposals: check the
# current text with counsel.

# %%
PROHIBITED = {"emotion recognition at work or school", "social scoring", "untargeted facial image scraping",
              "manipulation exploiting vulnerabilities", "biometric categorisation of sensitive traits"}
ANNEX_III = {"biometric identification", "critical infrastructure", "education access or assessment",
             "employment or worker management", "creditworthiness of individuals", "life or health insurance pricing",
             "public benefits eligibility", "law enforcement", "migration and border control", "administration of justice"}


def triage(use_case):
    if use_case["practice"] in PROHIBITED:
        return "prohibited", ["do not build or deploy in the EU; escalate to legal immediately"]
    if use_case["area"] in ANNEX_III or use_case.get("safety component of a regulated product"):
        return "high-risk", ["risk management system", "data governance and quality", "technical documentation",
                             "logging", "transparency to deployers", "human oversight", "accuracy, robustness and "
                             "cybersecurity", "conformity assessment and registration before deployment"]
    duties = []
    if use_case.get("interacts with people"):
        duties.append("tell people they're interacting with an AI system")
    if use_case.get("generates synthetic content"):
        duties.append("mark generated content as AI-generated (machine-readable)")
    if duties:
        return "transparency obligations", duties
    return "minimal risk", ["no specific obligations; voluntary codes of conduct; other law still applies (GDPR...)"]


cases = [
    {"name": "loan default scorer (this card)", "area": "creditworthiness of individuals", "practice": ""},
    {"name": "CV screening for hiring", "area": "employment or worker management", "practice": ""},
    {"name": "customer-support chatbot", "area": "customer service", "practice": "", "interacts with people": True},
    {"name": "marketing image generator", "area": "marketing", "practice": "", "generates synthetic content": True},
    {"name": "spam filter", "area": "email", "practice": ""},
    {"name": "employee mood detection from webcams", "area": "employment or worker management",
     "practice": "emotion recognition at work or school"},
]
print()
results = {}
for c in cases:
    tier, duties = triage(c)
    results[c["name"]] = tier
    print(f"{c['name']:38s} -> {tier}")
    print(f"{'':41s}{'; '.join(duties)}")
print("\nthe tier depends on the use, not the algorithm: the same gradient-boosted model is minimal-risk as a spam filter and")
print("high-risk as a credit scorer. Do this triage at design time (47.1), because high-risk obligations shape the system")
print("(logging, oversight, documentation), and they're expensive to retrofit.")
assert results["loan default scorer (this card)"] == "high-risk" and results["employee mood detection from webcams"] == "prohibited"
assert results["spam filter"] == "minimal risk" and results["customer-support chatbot"] == "transparency obligations"

print("\nAll checks passed.")
