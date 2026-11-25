def grams(text, n=3):
    text = " " + normalize_name(text) + " "
    return {text[i:i+n] for i in range(max(0, len(text) - n + 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def agreement(wanted, found):
    return -1.0 if not wanted or not found else float(wanted == found)


def candidate_features(item, candidate):
    check_source(item)
    query = item["request"]
    names = [candidate[key] for key in ("display_name", "brand_name", "generic_name")]
    name = normalize_name(query.get("name"))
    native = candidate["fda_active_ingredients"]
    label_text = " ".join(candidate["spl"]["active_ingredient_text"])
    strengths = [quantity_key(i.get("strength", "")) for i in native]
    requested = [quantity_key(s) for s in query.get("native_strengths", [])]
    valid_strengths = [contract.canonical(k) for k in strengths if k]
    valid_requested = [contract.canonical(k) for k in requested if k]
    label_words = set(normalize_name(label_text).split())
    source = candidate["provenance"]
    out = {
        "name_ratio": max(SequenceMatcher(None, name, normalize_name(n)).ratio() for n in names),
        "name_exact": int(any(name == normalize_name(n) for n in names) and bool(name)),
        "name_grams": max(jaccard(grams(name), grams(n)) for n in names),
        "form_agreement": agreement(normalize_name(query.get("dosage_form")),
                                    normalize_name(candidate["dosage_form"])),
        "route_agreement": agreement(route_signature(query.get("route")), route_signature(candidate["route"])),
        "strength_agreement": agreement(sorted(valid_requested), sorted(valid_strengths))
                              if len(valid_requested) == len(requested) else -1.0,
        "query_ndc_given": int(bool(query.get("product_ndc"))),
        "ndc_agreement": agreement(query.get("product_ndc"), candidate["product_ndc"]),
        "ingredient_count": len(native),
        "parsed_strength_fraction": len(valid_strengths) / max(1, len(native)),
        "label_word_count": len(label_words),
        "label_name_fraction": np.mean([
            set(normalize_name(i["name"]).split()) <= label_words for i in native
        ]) if native else 0.0,
        "rxnorm_pair_count": len(candidate["rxnorm"]["ingredient_pairs"]),
        "rxnorm_name_ratio": SequenceMatcher(None, normalize_name(candidate["generic_name"]),
                                             normalize_name(candidate["rxnorm"]["product_name"])).ratio(),
        "version_confirmed": int(source["original_versions_confirmed"]),
        "native_name_confirmed": int(source["raw_ndc_name_matches_catalog"]),
        "native_form_route_confirmed": int(source.get("raw_form_route_confirmed", False)),
        "rxnorm_pairs_confirmed": int(candidate["rxnorm"]["cached_pairs_confirmed"]),
        "missing_label": int(not label_text),
        "missing_rxnorm": int(not candidate["rxnorm"]["product_rxcui"]),
        "missing_native": int(not native),
        "missing_form": int(not candidate["dosage_form"]),
        "missing_route": int(not candidate["route"]),
        "candidate_form": normalize_name(candidate["dosage_form"]) or "missing",
        "candidate_route": " ".join(route_signature(candidate["route"]) or []) or "missing"
    }
    return out


def request_features(item, resolver):
    check_source(item)
    query, candidates = item["request"], item["candidates"]
    out = {
        "candidate_count": len(candidates),
        "ingredient_diversity": len({ingredient_key(c, resolver) for c in candidates}),
        "form_diversity": len({form_key(c, resolver) for c in candidates}),
        "query_form": normalize_name(query.get("dosage_form")) or "missing",
        "query_route": " ".join(route_signature(query.get("route")) or []) or "missing",
        "query_name_length": len(query.get("name", "")),
        "query_strength_count": len(query.get("native_strengths", []))
    }
    for key in sorted(QUERY_FIELDS):
        out["missing_query_" + key] = int(not query.get(key))
    rows = [candidate_features(item, c) for c in candidates]
    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float, np.number))] if rows else []
    for key in numeric:
        values = np.asarray([r[key] for r in rows], dtype=float)
        for name, value in (("min", values.min()), ("max", values.max()),
                            ("mean", values.mean()), ("std", values.std())):
            out[key + "_" + name] = float(value)
    return out


def feature_rows(items, resolver):
    request_dicts = [request_features(item, resolver) for item in items]
    candidate_dicts = [candidate_features(item, c) for item in items for c in item["candidates"]]
    return request_dicts, candidate_dicts
