"""Heterogeneous tool-planning task families with held-out phrasing templates.

Each family fixes the capability types an answer must be grounded in. Train
and test splits use *disjoint* templates, so routers that learn from text
(classifier, supervised router, rules written from training examples) are
evaluated on unseen phrasings.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

FAMILIES: dict[str, frozenset[str]] = {
    "revenue_growth": frozenset({"internal_metrics", "numeric_result"}),
    "cohort_analysis": frozenset({"internal_metrics", "numeric_result"}),
    "kpi_today": frozenset({"internal_metrics"}),
    "policy_lookup": frozenset({"internal_docs"}),
    "incident_triage": frozenset({"logs", "internal_docs"}),
    "market_compare": frozenset({"internal_metrics", "market_data", "numeric_result"}),
    "stock_quote": frozenset({"market_data"}),
    "breaking_news": frozenset({"web_facts"}),
    "claim_check": frozenset({"web_facts", "numeric_result"}),
    "general_knowledge": frozenset(),
    "metric_explainer": frozenset({"internal_docs", "internal_metrics"}),
}

WEIGHTS = {f: (1.5 if f == "general_knowledge" else 1.0) for f in FAMILIES}

SLOTS = {
    "metric": ["revenue", "sales", "gross margin", "ARR", "net bookings", "signups", "churn", "order volume"],
    "period": ["Q1", "Q2", "Q3", "Q4", "last quarter", "the previous quarter", "H1", "last month"],
    "segment": ["enterprise", "SMB", "EMEA", "APAC", "self-serve", "partner-led"],
    "policy": ["travel", "expense", "remote work", "data retention", "parental leave", "security"],
    "topic": ["reimbursement limits", "approval steps", "exceptions", "eligibility", "deadlines"],
    "service": ["checkout", "auth", "payments", "search", "billing", "notifications"],
    "company": ["Acme Corp", "Globex", "Initech", "Hooli", "Vandelay", "Stark Industries"],
    "ticker": ["ACME", "GLBX", "INIT", "HOOL", "VNDL", "STRK"],
    "entity": ["the EU AI Act", "the chip export rules", "the central bank", "the port strike", "the merger talks"],
    "claim": [
        "global smartphone shipments fell 12% last year",
        "average US rent rose by a third since 2020",
        "EV sales doubled in two years",
        "air travel recovered to 95% of 2019 levels",
    ],
    "concept": ["photosynthesis", "a hash table", "inflation", "the Pythagorean theorem", "TCP", "a p95 latency"],
}

TEMPLATES: dict[str, dict[str, list[str]]] = {
    "revenue_growth": {
        "train": [
            "What was our {metric} growth from {period} to the quarter after?",
            "Calculate the percentage change in {metric} between {period} and now",
            "How much did {metric} grow quarter over quarter in {period}?",
            "Compute {metric} growth rate for {period}",
        ],
        "test": [
            "By what percent did our {metric} increase versus {period}?",
            "Give me the QoQ delta of {metric} relative to {period}, as a percentage",
        ],
    },
    "cohort_analysis": {
        "train": [
            "Run a cohort retention analysis of {segment} customers who joined in {period}",
            "Fit a trend line to weekly {metric} for the {segment} segment",
            "What is the median and variance of {metric} per {segment} account in {period}?",
            "Break down {metric} by signup cohort for {segment} and compute retention curves",
        ],
        "test": [
            "Model the decay of {segment} cohorts from {period} and estimate 12-week retention",
            "Do a regression of {metric} against account age for {segment}",
        ],
    },
    "kpi_today": {
        "train": [
            "What is today's {metric} so far?",
            "Show the live {metric} number for the {segment} segment right now",
            "Current real-time {metric} for today",
            "How many {metric} have we booked today, up to this minute?",
        ],
        "test": [
            "Give me the up-to-the-minute {metric} figure for {segment}",
            "Where does {metric} stand as of this morning?",
        ],
    },
    "policy_lookup": {
        "train": [
            "What does our {policy} policy say about {topic}?",
            "Find the section of the {policy} handbook covering {topic}",
            "According to the internal {policy} guidelines, what are the {topic}?",
            "Look up {topic} in the company {policy} policy document",
        ],
        "test": [
            "Am I allowed to do this under the {policy} rules? I need the {topic}",
            "Quote the {topic} clause from our {policy} policy",
        ],
    },
    "incident_triage": {
        "train": [
            "Why is the {service} service throwing errors? Check the logs and the runbook",
            "Investigate the {service} outage using error logs and incident docs",
            "Find the root cause of {service} latency spikes from logs and the playbook",
            "Triage the {service} incident: pull recent exceptions and the on-call runbook",
        ],
        "test": [
            "{service} is failing for users, what do the stack traces and our runbooks say?",
            "Diagnose the {service} 500s with log evidence and the documented remediation",
        ],
    },
    "market_compare": {
        "train": [
            "Compare our {metric} growth with {company}'s latest reported numbers",
            "How does our {period} {metric} stack up against {company} stock performance?",
            "Compute the ratio of our {metric} to {company}'s market cap",
            "Benchmark our {metric} trend against {company} share price over {period}",
        ],
        "test": [
            "Are we outpacing {company}? Contrast our {metric} with their market data for {period}",
            "Quantify the gap between our {metric} and {company}'s valuation",
        ],
    },
    "stock_quote": {
        "train": [
            "What is the current share price of {ticker}?",
            "Get the latest stock quote for {company}",
            "What's {ticker} trading at right now?",
            "Show the market price and volume for {ticker}",
        ],
        "test": [
            "Quote me {company} ({ticker}) on the exchange",
            "How much is one share of {ticker} worth at the moment?",
        ],
    },
    "breaking_news": {
        "train": [
            "What is the latest news about {entity}?",
            "Summarise today's headlines on {entity}",
            "Any recent developments regarding {entity}?",
            "Search the web for current reporting on {entity}",
        ],
        "test": [
            "Catch me up on what happened this week with {entity}",
            "What are journalists saying about {entity} right now?",
        ],
    },
    "claim_check": {
        "train": [
            "Is it true that {claim}? Verify the numbers",
            "Fact-check this statistic: {claim}",
            "Validate the claim that {claim} and compute the actual percentage",
            "Check sources online to confirm whether {claim}",
        ],
        "test": [
            "Someone told me {claim} - does the arithmetic hold up against published data?",
            "Verify with public figures: {claim}",
        ],
    },
    "general_knowledge": {
        "train": [
            "Explain {concept} in simple terms",
            "What is {concept}?",
            "Give me an intuitive definition of {concept}",
            "How would you teach {concept} to a new hire?",
        ],
        "test": [
            "I keep hearing about {concept}, what does it mean?",
            "Describe {concept} with an example",
        ],
    },
    "metric_explainer": {
        "train": [
            "Why did {metric} change in {period}? Use the numbers and the board memo",
            "Explain the movement in {metric} for {period} with data and internal commentary",
            "Pull {metric} for {period} and the planning doc that discusses it",
            "What drove {metric} in {period} according to our reports and figures?",
        ],
        "test": [
            "Give the {period} {metric} figure along with the narrative from the internal review",
            "Reconcile our {metric} numbers for {period} with what the strategy memo claims",
        ],
    },
}

ROLES = (("analyst", 0.65), ("finance", 0.25), ("admin", 0.10))


@dataclass(frozen=True)
class Task:
    id: int
    query: str
    family: str
    required: frozenset[str]
    role: str


def _fill(template: str, rng: random.Random) -> str:
    out = template
    for k, vals in SLOTS.items():
        token = "{" + k + "}"
        while token in out:
            out = out.replace(token, rng.choice(vals), 1)
    return out


def generate(n: int, seed: int, split: str = "test", families: dict[str, float] | None = None) -> list[Task]:
    rng = random.Random(f"tasks-{split}-{seed}")
    weights = families or WEIGHTS
    names = list(weights)
    roles, rw = zip(*ROLES, strict=True)
    tasks = []
    for i in range(n):
        fam = rng.choices(names, weights=[weights[f] for f in names])[0]
        q = _fill(rng.choice(TEMPLATES[fam][split]), rng)
        tasks.append(Task(i, q, fam, FAMILIES[fam], rng.choices(roles, weights=rw)[0]))
    return tasks
