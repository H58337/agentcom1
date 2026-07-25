from recommender.agent.COM.tree_engine.schemas import build_evaluation_result


class InteractionEvaluator:
    @staticmethod
    def _to_int_set(values):
        out = set()
        for value in values or []:
            try:
                out.add(int(value))
            except (TypeError, ValueError):
                continue
        return out

    def evaluate(
        self,
        initial_item_name,
        final_item_name,
        gt_items,
        initial_confidence,
        final_confidence,
        prev_uncertainty,
        curr_uncertainty,
        path,
        initial_item_id=None,
        final_item_id=None,
        gt_item_ids=None,
    ):
        gt_ids = self._to_int_set(gt_item_ids)
        if gt_ids and initial_item_id is not None and final_item_id is not None:
            initial_correct = int(initial_item_id) in gt_ids
            final_correct = int(final_item_id) in gt_ids
        else:
            # Compatibility fallback for callers that do not expose stable item IDs.
            gt_names = set(str(x) for x in (gt_items or []))
            initial_correct = bool(str(initial_item_name) in gt_names)
            final_correct = bool(str(final_item_name) in gt_names)

        if initial_correct and final_correct:
            outcome = "TT"
        elif (not initial_correct) and final_correct:
            outcome = "WT"
        elif initial_correct and (not final_correct):
            outcome = "TW"
        else:
            outcome = "WW"

        prev_set = set(str(x) for x in (prev_uncertainty or []))
        curr_set = set(str(x) for x in (curr_uncertainty or []))
        reduction = {point: ("reduced" if point not in curr_set else "remaining") for point in prev_set}

        return build_evaluation_result(
            outcome_signal=outcome,
            confidence_change={
                "before": int(initial_confidence or 0),
                "after": int(final_confidence or 0),
            },
            uncertainty_reduction=reduction,
            path_level_feedback={
                "path": dict(path or {}),
                "outcome_signal": outcome,
            },
            branch_level_feedback={
                'why': str((path or {}).get('why', "") or ""),
                "who": str((path or {}).get("who", "") or ""),
                "how": str((path or {}).get("how", "") or ""),
            },
        )
