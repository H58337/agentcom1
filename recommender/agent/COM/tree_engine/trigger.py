def infer_primary_trigger(args, proposal_name, shortlist, uncertainty_points, prior_hint="", history_count=0):
    shortlist = list(shortlist or [])
    uncertainty_points = [str(x) for x in (uncertainty_points or [])]
    proposal_norm = str(proposal_name or "").strip().lower()
    prior_norm = str(prior_hint or "").strip().lower()

    if int(history_count or 0) < int(getattr(args, "MIN_ITEM_LIST_LENGTH", 5) or 5):
        return "cold-start"
    if proposal_norm and prior_norm and proposal_norm != prior_norm:
        return "internal-prior-conflict"
    if "novelty_justification" in uncertainty_points:
        return "novelty-uncertainty"
    if "candidate_comparison" in uncertainty_points or len(shortlist) >= 2:
        return "candidate-conflict"
    if "preference_alignment" in uncertainty_points:
        return "cold-start"
    return "candidate-conflict" if shortlist else "cold-start"
