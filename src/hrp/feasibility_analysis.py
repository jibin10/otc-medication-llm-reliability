# Find identical OTC names associated with different ingredients

import re
from collections import Counter, defaultdict
from itertools import combinations, pairwise

from rapidfuzz.fuzz import ratio

EXACT_AGREEMENT = {"exact_base_names", "exact_precise_names"}


def name_key(value: str) -> str:
    from hrp.data_cleaning import normalize_name

    return re.sub(r"[^\w]+", " ", normalize_name(value).replace("&", " and ")).strip()


def discover_name_links(products: dict[str, dict]) -> tuple[list[tuple], dict]:
    """Find exact aliases and near names using rare tokens and character grams."""
    by_name = defaultdict(list)
    for ndc, product in sorted(products.items()):
        if name := name_key(product.get("display_name", product.get("name", ""))):
            by_name[name].append(ndc)
    names = sorted(by_name)
    tokens = {name: set(name.split()) for name in names}
    frequencies = Counter(t for values in tokens.values() for t in values)
    common = {t for t, n in frequencies.items() if n > max(40, len(names) // 500)}
    common |= {
        "mg",
        "ml",
        "g",
        "mcg",
        "tablet",
        "tablets",
        "capsule",
        "capsules",
        "cream",
        "gel",
        "solution",
        "spray",
        "oral",
        "topical",
        "and",
        "with",
        "for",
        "the",
        "of",
        "in",
        "usp",
    }
    blocks = defaultdict(set)
    for name in names:
        compact = name.replace(" ", "")
        for token in tokens[name] - common:
            if len(token) >= 4 and not token.isdigit():
                blocks[("token", token)].add(name)
        for gram in {compact[i : i + 4] for i in range(len(compact) - 3)}:
            blocks[("gram", gram)].add(name)
    proposals = set()
    omitted_blocks = Counter()
    for (kind, _), members in sorted(blocks.items()):
        if len(members) > (128 if kind == "token" else 48):
            omitted_blocks[kind] += 1
            continue
        proposals.update(combinations(sorted(members), 2))
    links = []
    for name, ndcs in sorted(by_name.items()):
        for left, right in pairwise(ndcs):
            links.append((left, right, "identical_or_punctuation_alias"))
    accepted_names = []
    rejected = 0
    for left, right in sorted(proposals):
        a, b = tokens[left] - common, tokens[right] - common
        score = ratio(left, right)
        subset = len(a & b) >= 2 and (a <= b or b <= a)
        if subset or (score >= 92 and min(len(left), len(right)) >= 6):
            accepted_names.append((left, right))
            links.append(
                (
                    by_name[left][0],
                    by_name[right][0],
                    "distinctive_token_variant" if subset else "near_name_co_partition",
                )
            )
        else:
            rejected += 1
    return links, {
        "catalog_products_considered": len(products),
        "distinct_names": len(names),
        "name_pair_proposals": len(proposals),
        "accepted_near_name_links": len(accepted_names),
        "rejected_unrelated_name_pairs": rejected,
        "oversized_blocks_skipped": dict(omitted_blocks),
        "policy": "All identical punctuation aliases. Near names share two rare tokens with a "
        "subset relation or have character similarity at least 92. Four-character gram "
        "postings up to 48 and rare-token postings up to 128 generate proposals. "
        "These are conservative co-partition links, not verified marketing families.",
    }


def distinct_ingredient_pair(products: list[dict]) -> tuple[dict, dict] | None:
    """Select one reproducible pair with different ingredient sets."""
    first_by_set = {}

    for product in sorted(products, key=lambda item: item["product_ndc"]):
        first_by_set.setdefault(product["in_ids"], product)

    if len(first_by_set) < 2:
        return None

    examples = list(first_by_set.values())
    return examples[0], examples[1]
