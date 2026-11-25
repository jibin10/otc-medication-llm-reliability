"""Select a compact source-quality audit without limiting benchmark admission."""

import json
from collections import defaultdict
from hashlib import sha256


def stratified_review_sample(rows: list[dict], per_stratum: int = 3) -> list[dict]:
    strata = defaultdict(list)
    for row in rows:
        if row["partition"] == "test":
            continue
        for reason in json.loads(row["reasons_json"]):
            strata[(row["support_tier"], reason)].append(row)
    selected = {
        row["product_ndc"]: row
        for values in strata.values()
        for row in sorted(values, key=lambda r: sha256(r["product_ndc"].encode()).hexdigest())[
            :per_stratum
        ]
    }
    return [selected[key] for key in sorted(selected)]
