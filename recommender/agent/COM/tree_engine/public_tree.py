from pathlib import Path
import re
import shutil

from recommender.agent.COM.tree_engine.utils import append_jsonl, dump_json, load_json, safe_read_text


INITIAL_ACTIVE_NODES = {
    'why': [
        ("cold-start", "Use when the user history or profile evidence is too sparse to make a stable decision."),
        ("candidate-conflict", "Use when multiple HesitationSet candidates are hard to distinguish."),
        ("novelty-uncertainty", "Use when the current proposal may be valuable exploration but might also deviate from stable preference."),
        ("internal-prior-conflict", "Use when the internal proposal and prior recommendation disagree."),
    ],
    "what": [
        ("reduce_hesitation_set", "Use when the user's task is to remove or down-rank candidates that clearly do not fit."),
        ("find_interested_subset", "Use when the user's task is to keep a shorter subset of candidates that remain interesting."),
        ("compare_remaining_candidates", "Use when the user's task is to compare remaining candidates or test rival claims."),
        ("evidence_gap_check", "Use when the user's task is to identify insufficient original reasons and specify what supplementary evidence is needed."),
        ("reasoning_check", "Use when the user's task is to verify whether their own initial reasoning or assumptions are actually valid."),
        ("none", "Use when the user's natural-language task cannot yet be mapped to a stable what node."),
    ],
    "who": [
        ("trusted-advisors", "Use direct trust advisors first; internally distinguish one-way trust, mutual trust, single/multi trust, and history-similar versus history-dissimilar trusted users."),
        ("similar-users", "Use non-trust users with similar historical item preferences."),
        ("experienced-users", "Use users who interacted with the proposal or HesitationSet items."),
        ("topk-advisors", "Use friend-of-friend advisors from the social graph, mainly for exploration or when direct trust is unavailable."),
    ],
    "how": [
        ("single-advisor", "One friend independently answers the UserTask from their own evidence and understanding of the user."),
        ("multi-cooperative", "Multiple friends share the UserTask, think independently first, then use memory to complement, refine, or integrate each other's points."),
        ("multi-competitive", "Multiple friends use the UserTask competitively: the first makes a claim, and later friends must question or rebut a prior claim before the user decides."),
    ],
}


CANDIDATE_EXTENSION_NODES = {
    "who": [
        ("weak-tie-advisors", "Use weak-tie advisors to bring new information from outside the user's close trust circle."),
        ("domain-experts", "Use domain experts when the item category requires specialized knowledge."),
    ],
    "how": [
        ("evidence-aggregation", "Deprecated alias candidate for multi-cooperative evidence collection."),
        ("devils-advocate", "Deprecated alias candidate for multi-competitive challenge."),
        ("counterfactual-comparison", "Deprecated alias candidate for multi-competitive comparison."),
    ],
}


INITIAL_PATTERNS = [
    ("validate-then-debate", "A higher-order pattern that first validates a proposal, then debates remaining conflict."),
    ("competition-then-validation", "A higher-order pattern that first runs competition, then validates the winner."),
]


DEPRECATED_NODE_ALIASES = {
    "who": {},
    "what": {
        "warning": "reduce_hesitation_set",
        "promotion": "find_interested_subset",
        "feedback-repair": "evidence_gap_check",
        "evaluation-only": "none",
        "verify": "reasoning_check",
        "reasoning": "reasoning_check",
        "reduce_hesitation_set/action-confirmation/elicit-priority-and-brand-weight": "reduce_hesitation_set/action-confirmation/elicit-music-preference-priority",
    },
    "how": {
        "single": "single-advisor",
        "single-warning": "single-advisor",
        "single-promotion": "single-advisor",
        "single-evaluation-only": "single-advisor",
        "single-evaluation-warning": "single-advisor",
        "single-evaluation-promotion": "single-advisor",
        "cooperative": "multi-cooperative",
        "cooperative-inquiry": "multi-cooperative",
        "multi-cooperative-warning": "multi-cooperative",
        "multi-cooperative-promotion": "multi-cooperative",
        "feedback-cooperative-repair": "multi-cooperative",
        "feedback-focused-repair": "multi-cooperative",
        "competitive": "multi-competitive",
        "debate": "multi-competitive",
        "pairwise-debate": "multi-competitive",
        "multi-candidate-debate": "multi-competitive",
        "multi-competitive-warning": "multi-competitive",
        "multi-competitive-promotion": "multi-competitive",
        "feedback-competitive-repair": "multi-competitive",
        "multi-cooperative/cooperative-evidence-elicitation/elicit-priority-brand-weight": "multi-cooperative/cooperative-evidence-elicitation/elicit-music-preference-priority",
    },
}

DEFAULT_SELECTION_PROFILES = {
    ('why', "candidate-conflict"): {
        "requires": ["has_comparable_candidates=true"],
        "prefers": ["focus_candidate_count>=2", "score_gap_small=true"],
        "do_not_use_why": ["focus_candidate_count<=1"],
        "selection_prior": 1.0,
    },
    ('why', "cold-start"): {
        "requires": [],
        "prefers": ["history_sparsity=sparse", "history_count<=3"],
        "do_not_use_why": ["history_sparsity=rich"],
        "selection_prior": 0.7,
    },
    ('why', "internal-prior-conflict"): {
        "requires": ["proposal_conflicts_with_prior=true"],
        "prefers": ["score_gap_small=true"],
        "do_not_use_why": [],
        "selection_prior": 1.0,
    },
    ('why', "novelty-uncertainty"): {
        "requires": ["proposal_is_novel=true"],
        "prefers": ["history_sparsity!=sparse"],
        "do_not_use_why": [],
        "selection_prior": 0.8,
    },
    ("what", "compare_remaining_candidates"): {
        "requires": ["requires_candidate_contrast=true"],
        "prefers": ["asks_compare_candidates=true", "mentioned_candidate_count>=2"],
        "do_not_use_why": [],
        "selection_prior": 1.0,
    },
    ("what", "reduce_hesitation_set"): {
        "requires": [],
        "prefers": ["asks_reduce_options=true"],
        "do_not_use_why": [],
        "selection_prior": 0.8,
    },
    ("what", "reasoning_check"): {
        "requires": [],
        "prefers": ["requires_reason_validation=true", "asks_verify_reasoning=true"],
        "do_not_use_why": [],
        "selection_prior": 0.8,
    },
    ("what", "evidence_gap_check"): {
        "requires": [],
        "prefers": ["requires_evidence_gap_check=true", "asks_missing_evidence=true"],
        "do_not_use_why": [],
        "selection_prior": 0.8,
    },
    ("what", "find_interested_subset"): {
        "requires": [],
        "prefers": ["asks_positive_subset=true"],
        "do_not_use_why": [],
        "selection_prior": 0.7,
    },
    ("what", "none"): {
        "requires": [],
        "prefers": [],
        "do_not_use_why": [
            "asks_compare_candidates=true",
            "asks_reduce_options=true",
            "asks_verify_reasoning=true",
            "asks_missing_evidence=true",
            "asks_positive_subset=true",
        ],
        "selection_prior": 0.05,
    },
}


DEFAULT_SUMMARY_HINTS = {
    ("what", "reduce_hesitation_set"): {
        "task_focus": "summarize why advisors think specific candidates can be removed",
        "important_output_fields": ["RemoveSet", "CandidateView", "RiskReason", "TaskAnswer"],
        "preserve_interaction_fields": ["ChallengeOrSupportPrevious", "ResponseToPrevious", "Correction"],
    },
    ("what", "find_interested_subset"): {
        "task_focus": "summarize why advisors think specific candidates may interest the requester",
        "important_output_fields": ["InterestedSet", "CandidateView", "InterestReason", "TaskAnswer"],
        "preserve_interaction_fields": ["ChallengeOrSupportPrevious", "ResponseToPrevious", "Correction"],
    },
    ("what", "compare_remaining_candidates"): {
        "task_focus": "summarize candidate-level comparison, tradeoffs, and unresolved comparison axes",
        "important_output_fields": ["ComparedSet", "StrongerCandidates", "WeakerCandidates", "CandidateView", "KeyTradeoff", "TaskAnswer"],
        "preserve_interaction_fields": ["ChallengeOrSupportPrevious", "ResponseToPrevious", "Correction"],
    },
    ("what", "evidence_gap_check"): {
        "task_focus": "summarize what evidence or supplementary reason is missing or added",
        "important_output_fields": ["EvidenceGapSet", "CandidateView", "SupplementReason", "TaskAnswer"],
        "preserve_interaction_fields": ["ChallengeOrSupportPrevious", "ResponseToPrevious", "Correction"],
    },
    ("what", "reasoning_check"): {
        "task_focus": "summarize whether the user's reason or assumptions are reliable, weak, mixed, or corrected",
        "important_output_fields": ["CandidateView", "TaskAnswer", "Correction"],
        "preserve_interaction_fields": ["ChallengeOrSupportPrevious", "ResponseToPrevious", "Correction"],
    },
    ("what", "none"): {
        "task_focus": "summarize the natural-language advisor task without voting",
        "important_output_fields": ["CandidateView", "TaskAnswer"],
        "preserve_interaction_fields": ["ChallengeOrSupportPrevious", "ResponseToPrevious", "Correction"],
    },
}


def infer_communication_shape(how):
    how = str(how or "").strip()
    how = (DEPRECATED_NODE_ALIASES.get("how", {}) or {}).get(how, how)
    if "/" in how:
        how = how.split("/", 1)[0]
    if how == "single-advisor":
        return "single"
    if how.startswith("multi-") or how.startswith("mc-") or "cooperative" in how or how in {"reconciliation"}:
        return "multi"
    return "single"


def infer_communication_family(how):
    how = str(how or "").strip()
    how = (DEPRECATED_NODE_ALIASES.get("how", {}) or {}).get(how, how)
    if "/" in how:
        how = how.split("/", 1)[0]
    if how == "single-advisor":
        return "single"
    if how == "multi-cooperative" or "cooperative" in how or how in {"reconciliation"}:
        return "cooperative"
    if how == "multi-competitive" or how.startswith("mc-"):
        return "competitive"
    return "single"


def infer_communication_intent(how):
    how = str(how or "").strip()
    how = (DEPRECATED_NODE_ALIASES.get("how", {}) or {}).get(how, how)
    if "/" in how:
        how = how.split("/", 1)[0]
    return "user-task"


def advisor_output_format_for_how(how):
    how = str(how or "").strip()
    how = (DEPRECATED_NODE_ALIASES.get("how", {}) or {}).get(how, how)
    if "/" in how:
        how = how.split("/", 1)[0]
    if how == "multi-cooperative" or "cooperative" in how or how in {"reconciliation"}:
        return ["ResponseToPrevious: <agreement, refinement, correction, or integration of PreviousFriendViews, or none>"]
    if how == "multi-competitive" or how.startswith("mc-"):
        return ["ChallengeOrSupportPrevious: <for advisor 1: new_claim; for later advisors: rebut/question/correct one specific previous claim; never generic agreement or none when PreviousFriendViews exist>"]
    return []


def task_output_format_for_what(what):
    what = str(what or "").strip()
    what = what.split("/", 1)[0] if "/" in what else what
    formats = {
        "reduce_hesitation_set": [
            "RemoveSet: <exact HesitationSet candidates you would remove or down-rank, or none>",
            "CandidateView:\n- <exact removed candidate> | <remove> | <specific evidence or criterion used to exclude this candidate>",
            "RiskReason: <the exclusion criterion and evidence for every RemoveSet candidate, or none>",
            "TaskAnswer: <direct answer to Task>",
        ],
        "find_interested_subset": [
            "InterestedSet: <exact HesitationSet candidates the user may stay interested in, or none>",
            "CandidateView:\n- <exact interested candidate> | <interest> | <specific evidence or criterion showing why the user may stay interested>",
            "InterestReason: <the interest criterion and evidence for every InterestedSet candidate, or none>",
            "TaskAnswer: <direct answer to Task>",
        ],
        "compare_remaining_candidates": [
            "ComparedSet: <exact HesitationSet candidates you actually compared>",
            "StrongerCandidates: <exact candidates with stronger fit, or none>",
            "WeakerCandidates: <exact candidates with weaker fit/risk, or none>",
            "CandidateView:\n- <exact candidate> | <stronger/weaker/mixed/unclear> | <short reason>",
            "KeyTradeoff: <the key comparison criterion, why it matters for this user, and which side it favors>",
            "TaskAnswer: <direct answer to Task>",
        ],
        "evidence_gap_check": [
            "EvidenceGapSet: <exact HesitationSet candidates whose current/original reason is insufficient, or none>",
            "CandidateView:\n- <exact candidate needing more evidence> | <evidence_gap> | <what current/original reason is insufficient and what supplementary reason or evidence is needed>",
            "SupplementReason: <the missing or supplementary reason/evidence needed for every EvidenceGapSet candidate, or none>",
            "TaskAnswer: <direct answer to Task>",
        ],
        "reasoning_check": [
            "ReliableReasons: <initial hesitation reasons or assumptions that are well supported, or none>",
            "WeakReasons: <initial hesitation reasons or assumptions that are weak, unsupported, or need correction, or none>",
            "CandidateView:\n- <exact candidate> | <reason_reliable/reason_weak/mixed/unclear> | <which initial reason was checked and why>",
            "Correction: <how the user should revise the initial reasoning; do not select a final item>",
            "TaskAnswer: <direct answer to Task>",
            "AskUser: <specific question for the user, or none>",
        ],
    }
    return list(formats.get(what) or [
        "RelevantCandidates: <exact HesitationSet candidates your answer covers, or none>",
        "UnclearSet: <exact HesitationSet candidates you cannot judge yet, or none>",
        "CandidateView:\n- <exact candidate> | <support/risk/unclear> | <short reason>",
        "TaskAnswer: <direct answer to Task as far as possible>",
        "AskUser: <specific question for the user, or none>",
    ])


NODE_ACTION_RULES = {
    ('why', "cold-start"): {
        "action": "Start evidence-expansion communication for sparse user-history or weak profile support.",
        "rule": "Advisor speech should surface broad preference clues, nearby analogue evidence, and explicit uncertainty instead of producing overconfident conclusions.",
    },
    ('why', "candidate-conflict"): {
        "action": "Start HesitationSet-conflict communication focused on direct comparison among plausible candidates.",
        "rule": "Select this when HesitationSet contains multiple plausible candidates that are hard to distinguish. Advisor speech must compare candidate fit directly; generic praise is insufficient. If two candidates dominate, pair naturally with pairwise-debate; if three or more candidates remain plausible, pair naturally with multi-candidate-debate.",
    },
    ('why', "novelty-uncertainty"): {
        "action": "Start novelty-risk communication to judge whether the proposal is useful exploration or preference drift.",
        "rule": "Advisor speech should discuss both exploration value and mismatch risk, and should connect novelty back to the user's historical preference evidence.",
    },
    ('why', "internal-prior-conflict"): {
        "action": "Start conflict analysis between the user's current proposal and the prior recommendation evidence.",
        "rule": "Advisor speech should compare the internal proposal and PriorHint without assuming either is automatically correct.",
    },
    ("who", "trusted-advisors"): {
        "action": "Select explicit social-edge trusted advisors as speakers.",
        "rule": "Advisor selection should prioritize direct trust. Within trusted advisors, distinguish mutual trust from one-way trust, single trusted source from multiple trusted sources, and history-similar trusted users from history-dissimilar trusted users. Mutual and history-similar trusted users should carry the strongest initial trust, while history-dissimilar trusted users should speak as trusted but exploratory evidence.",
    },
    ("who", "similar-users"): {
        "action": "Select users with similar historical preferences as speakers.",
        "rule": "Advisor speech should ground its recommendation in historical preference similarity and item-overlap evidence, not friendship, popularity, or invented expertise. Use this mainly when direct trust is unavailable or insufficient and the target user's history is rich enough to support similarity matching.",
    },
    ("who", "experienced-users"): {
        "action": "Select users with direct experience on the proposal or HesitationSet candidates.",
        "rule": "Advisor speech should prioritize item-level experience with proposal or HesitationSet items and candidate-specific discrimination over broad profile commentary. Use this when both social evidence and long-term history evidence are too sparse, or when a specific candidate needs direct experience evidence.",
    },
    ("who", "topk-advisors"): {
        "action": "Select top-ranked friend-of-friend advisors from the social graph.",
        "rule": "Advisor speech should present friend-of-friend evidence with calibrated confidence. Topk-advisors means social two-hop advisors, not recommender topK items. Use this mainly for exploration, social expansion, or fallback when direct trusted advisors are unavailable.",
    },
    ("how", "single-warning"): {
        "action": "Use one advisor to read the whole HesitationSet and warn about candidates the user is likely not interested in.",
        "rule": "Select this when only one advisor is available or one advisor has enough evidence for risk screening. The advisor should first consider the whole HesitationSet and the user's initial reasoning, then name a ShrinkSet of candidates to remove or down-rank when concrete user-centered mismatch evidence supports it.",
    },
    ("how", "single-promotion"): {
        "action": "Use one advisor to read the whole HesitationSet and identify a shorter set of candidates the user is likely interested in.",
        "rule": "Select this when only one advisor is available or one advisor has strong positive evidence. The advisor should first consider the whole HesitationSet and the user's initial reasoning, then name a RetainSet shorter than the original HesitationSet whenever evidence allows.",
    },
    ("how", "multi-cooperative-warning"): {
        "action": "Use multiple advisors cooperatively to screen the HesitationSet, flag risky options, and narrow the user's hesitation.",
        "rule": "Select this when the user has a broad HesitationSet and needs collaborative risk screening rather than a debate or a final item choice. Each advisor should first read the whole HesitationSet and the user's own initial reasoning, form an independent judgment from their evidence and understanding of the requester, then use group memory to agree, refine, or challenge earlier points. The goal is cooperative hesitation-set reduction: identify candidates the user is likely not interested in, not collective opposition to one favorite.",
    },
    ("how", "multi-cooperative-promotion"): {
        "action": "Use multiple advisors cooperatively to add complementary positive evidence and decide which candidates deserve to stay in the HesitationSet.",
        "rule": "Select this when the user needs support coverage rather than item-selection debate. Each advisor should first read the whole HesitationSet and the user's own initial reasoning, form an independent retained-set judgment from their evidence and understanding of the requester, then use group memory to agree, refine, or challenge earlier points. The goal is a shorter set of candidates the user is likely interested in, with weak or missing support preserved as unresolved.",
    },
    ("how", "multi-competitive-warning"): {
        "action": "Use multiple advisors competitively to make or challenge risk claims.",
        "rule": "Select this when the main need is to compare which candidate is most worth excluding. Advisors should propose a risk claim, challenge or refine earlier risk claims, prefer a different candidate only when they can make a stronger case, and identify the strongest risk evidence. The goal is risk-claim competition, not coverage of every candidate.",
    },
    ("how", "multi-competitive-promotion"): {
        "action": "Use multiple advisors competitively to make or challenge support claims.",
        "rule": "Select this when multiple candidates remain plausible and the user needs to know which positive evidence is strongest. Advisors should propose a support claim, challenge or refine earlier support claims, prefer a different candidate only with stronger evidence, and decide whether one candidate clearly deserves promotion or the conflict remains unresolved.",
    },
    ("how", "feedback-cooperative-repair"): {
        "action": "Run a cooperative follow-up round with the same friends to answer the user's explicit remaining questions.",
        "rule": "Use after DecisionState=continue when the next round should be collaborative. Advisors must answer FeedbackToAdvisors first, reuse prior discussion memory, add complementary evidence when they have it, and say unresolved when a requested comparison still lacks evidence.",
    },
    ("how", "feedback-competitive-repair"): {
        "action": "Run a competitive follow-up round with the same friends to answer the user's explicit remaining questions by testing rival claims.",
        "rule": "Use after DecisionState=continue when the next round should compare or challenge competing claims. Advisors must answer FeedbackToAdvisors first, reuse prior discussion memory, challenge weak previous claims when justified, and preserve unresolved issues instead of forcing a final item choice.",
    },
    ("candidate_extensions", "weak-tie-advisors"): {
        "action": "Trial weak-tie advisors to add broader social evidence.",
        "rule": "Advisor speech should mark weak-tie evidence as exploratory and lower priority than direct trust or strong similarity evidence.",
    },
    ("candidate_extensions", "domain-experts"): {
        "action": "Trial domain-expert evidence when specialized item interpretation matters.",
        "rule": "Advisor speech should add domain-specific discriminators while tying every claim back to the target user's preference fit.",
    },
    ("candidate_extensions", "evidence-aggregation"): {
        "action": "Trial broad evidence aggregation without direct rebuttal.",
        "rule": "Advisor speech should preserve evidence coverage and consistency without staging debate or collapsing disagreement too early.",
    },
    ("candidate_extensions", "devils-advocate"): {
        "action": "Trial a challenge path against an apparently strong proposal.",
        "rule": "Advisor speech should search for concrete counter-evidence and hidden mismatch risks without opposing for style alone.",
    },
    ("candidate_extensions", "counterfactual-comparison"): {
        "action": "Trial explicit counterfactual comparison against the strongest target-like alternative.",
        "rule": "Advisor speech should compare the current selected item with a strong alternative and explain which failure mode would make the current choice wrong.",
    },
    ("patterns", "validate-then-debate"): {
        "action": "Coordinate a pattern that validates first and debates only if important conflict remains.",
        "rule": "The pattern should preserve the reason for escalation and avoid unnecessary debate after validation already stabilizes the choice.",
    },
    ("patterns", "competition-then-validation"): {
        "action": "Coordinate a pattern that finds a competitive winner and then validates it.",
        "rule": "The pattern should treat competition as candidate discovery and validation as a separate stability check.",
    },
}


NODE_INSTRUCTION_LIBRARY = {
    ('why', "cold-start"): {
        "role": "Interpret the current interaction as a sparse-evidence situation.",
        "objective": "Push the communication toward evidence expansion, preference elicitation, and conservative stabilization rather than overconfident early finalization.",
        "required_actions": [
            "Treat missing or weak history/profile support as a real limitation rather than pretending any HesitationSet candidate is already justified.",
            "Ask the communication path to surface broad preference fit, nearby analogues, and disconfirming evidence for weak assumptions.",
            "Prefer explanations that connect each HesitationSet candidate to stable preference clues instead of niche item trivia.",
        ],
        "output_contract": [
            "Make uncertainty explicit when evidence is thin.",
            "State which user preference dimensions remain underspecified.",
            "Avoid claiming strong certainty unless later evidence truly closes the gap.",
        ],
        "handoff": [
            "If stronger supporting evidence appears, pass a stabilized summary to later nodes.",
            "If evidence stays thin, preserve uncertainty for possible continued communication.",
        ],
    },
    ('why', "candidate-conflict"): {
        "role": "Interpret the interaction as a HesitationSet conflict between multiple plausible candidates.",
        "objective": "Maximize separability between the strongest candidates and force explicit comparison instead of vague support.",
        "required_actions": [
            "Center the communication on the HesitationSet head-to-head conflict, not on generic candidate praise.",
            "Require every contribution to say why one candidate is more suitable than at least one rival candidate.",
            "Prefer concrete differentiators such as style fit, novelty risk, preference alignment, and HesitationSet tradeoffs.",
        ],
        "output_contract": [
            "Expose which candidate currently leads and why.",
            "Preserve any unresolved comparison dimensions as explicit uncertainty.",
            "Do not allow a third unrelated candidate to hijack the discussion.",
        ],
        "handoff": [
            "If a clear margin emerges, hand off a ranked comparison result.",
            "If the gap remains narrow, mark the conflict as still active.",
        ],
    },
    ('why', "novelty-uncertainty"): {
        "role": "Interpret the interaction as a novelty-risk judgment.",
        "objective": "Test whether the proposal is a justified exploration move or an avoidable deviation from the user's stable preference.",
        "required_actions": [
            "Balance exploration value against preference drift risk.",
            "Demand evidence for why the novel candidate is still anchored to the user's known tastes.",
            "Surface both upside and failure modes instead of treating novelty as automatically good or bad.",
        ],
        "output_contract": [
            "Explain whether novelty is supported, weakly supported, or unsupported.",
            "Preserve any remaining concern about stability or preference mismatch.",
            "Prefer evidence tied to similar listening or interaction patterns over abstract genre labels alone.",
        ],
        "handoff": [
            "If novelty is justified, pass forward a positive but calibrated recommendation.",
            "If novelty remains risky, retain that uncertainty for the next decision step.",
        ],
    },
    ('why', "internal-prior-conflict"): {
        "role": "Interpret the interaction as a conflict between the user's internal proposal and an external prior recommendation.",
        "objective": "Force explicit comparison between the current internal proposal and the prior-driven alternative without assuming either side is correct.",
        "required_actions": [
            "Compare proposal and prior candidate directly and name the disagreement source.",
            "Ask whether the prior is trustworthy for this user state or whether the internal proposal better reflects current evidence.",
            "Look for reasons one side might be overconfident, outdated, or weakly grounded.",
        ],
        "output_contract": [
            "State whether the conflict is resolved in favor of proposal, prior, or still unresolved.",
            "Preserve explicit rationale for trusting one side over the other.",
            "Keep the disagreement visible for arbitration if the evidence stays mixed.",
        ],
        "handoff": [
            "If the conflict resolves, pass a clear keep-or-switch recommendation.",
            "If not, carry the conflict forward as a first-class uncertainty.",
        ],
    },
    ("who", "trusted-advisors"): {
        "role": "Speak as an explicitly trusted social-edge advisor.",
        "objective": "Provide high-priority direct-trust evidence while distinguishing mutual trust, one-way trust, trust multiplicity, and history/preference similarity.",
        "required_actions": [
            "Treat direct trust as the strongest bootstrap who source when trusted users exist.",
            "Mark whether the trusted relation is mutual or one-way when that metadata is available.",
            "Give extra weight to trusted advisors who are also history-similar to the target user.",
            "If a trusted advisor is history-dissimilar, keep the trust value but label the item-fit evidence as exploratory or weaker.",
            "If multiple trusted advisors are present, preserve distinct positions instead of collapsing them into one fake consensus.",
            "If a HesitationSet candidate is weak, use the trust position to deliver credible counter-evidence rather than automatic agreement.",
        ],
        "output_contract": [
            "Tie each recommendation to direct-trust evidence and item-fit evidence separately.",
            "Expose trust subtype: mutual-trust or one-way-trust, single-trust or multi-trust, history-similar or history-dissimilar.",
            "Make support or opposition explicit.",
            "Avoid drifting into generic item summaries detached from trust value.",
        ],
        "handoff": [
            "Pass forward whether direct trust strengthened or weakened any HesitationSet candidate.",
        ],
    },
    ("who", "similar-users"): {
        "role": "Speak as an advisor selected for high preference similarity.",
        "objective": "Ground the recommendation in overlap between the advisor's historical taste and the target user's history.",
        "required_actions": [
            "Use this path mainly when direct trusted advisors are unavailable, insufficient, or clearly less preference-similar.",
            "Explain why the recommendation fits because of preference similarity, not because of friendship or raw popularity.",
            "Highlight historical overlap signals when defending or opposing a candidate.",
            "Avoid pretending to be a direct friend or domain expert.",
        ],
        "output_contract": [
            "Frame evidence around taste alignment and similar preference trajectories.",
            "Make clear which HesitationSet candidate the similarity evidence supports or weakens.",
            "Preserve mismatch clues when the similar-user signal is weaker than expected.",
        ],
        "handoff": [
            "Pass a clear statement of whether similarity evidence reinforces or challenges each relevant HesitationSet candidate.",
        ],
    },
    ("who", "experienced-users"): {
        "role": "Speak as an advisor chosen because they interacted with HesitationSet candidates.",
        "objective": "Use item-specific experience to discriminate among HesitationSet candidates.",
        "required_actions": [
            "Use this path mainly when social evidence and long-term similarity evidence are sparse, or when candidate-level experience is the missing signal.",
            "Prioritize concrete experience with HesitationSet items over generic taste commentary.",
            "If the path is comparative, favor advisors whose experience can meaningfully represent different HesitationSet candidates.",
            'Why experience coverage is partial, state the limitation instead of overclaiming certainty.',
        ],
        "output_contract": [
            "Anchor support or opposition in direct candidate experience.",
            "Name which candidate the experience most strongly supports.",
            "Expose gaps when the advisor only knows one side of the conflict.",
        ],
        "handoff": [
            "Hand off a candidate-focused evidence bundle that can drive comparison or novelty judgment.",
        ],
    },
    ("who", "topk-advisors"): {
        "role": "Speak as a top-ranked two-hop social advisor selected through the friend-of-friend graph.",
        "objective": "Bring socially adjacent but not directly trusted evidence into the decision, with appropriate caution.",
        "required_actions": [
            "Treat two-hop social evidence as socially relevant but weaker than direct trust.",
            "Explain why the recommendation is useful despite the weaker social tie.",
            "Do not treat topk-advisors as recommender topK item evidence.",
            "Do not present yourself as a direct friend; keep the friend-of-friend nature implicit in the evidence style.",
        ],
        "output_contract": [
            "Provide structured support or opposition grounded in near-social evidence.",
            "Make it clear when two-hop evidence is enough to challenge a HesitationSet candidate.",
            "Avoid overstating certainty just because social adjacency exists.",
        ],
        "handoff": [
            "Pass a calibrated social-neighborhood view of the HesitationSet candidates.",
        ],
    },
    ("how", "single-warning"): {
        "role": "Run single-advisor communication as risk screening over the whole HesitationSet.",
        "objective": "Help the user shrink the HesitationSet by identifying candidates they are likely not interested in.",
        "required_actions": [
            "Read the whole HesitationSet and UserInitialChoiceContext before giving advice.",
            "Make an independent friend judgment from the advisor's own evidence and understanding of the requester.",
            "Use ShrinkSet to name one or more candidates to remove or down-rank only when concrete mismatch evidence supports it.",
            "If no candidate can be safely removed, keep ShrinkSet as none and explain the missing evidence.",
            "Do not evaluate a hidden CurrentProposal; the advisor-visible decision space is only the HesitationSet.",
        ],
        "output_contract": [
            "Advice should be caution, keep, or unresolved.",
            "SuggestedItem names the primary candidate behind the warning; ShrinkSet carries the broader removal/down-rank set.",
            "IndependentThinking must summarize the whole-set judgment.",
            "ReasonForUser must explain why the warning helps this requester decide.",
        ],
        "advisor_output_format": advisor_output_format_for_how("single-warning"),
        "handoff": [
            "Pass ShrinkSet, retained candidates, and unresolved missing evidence to the user's post-feedback decision.",
        ],
    },
    ("how", "single-promotion"): {
        "role": "Run single-advisor communication as positive retained-set selection over the whole HesitationSet.",
        "objective": "Help the user keep a shorter set of candidates they are likely interested in.",
        "required_actions": [
            "Read the whole HesitationSet and UserInitialChoiceContext before giving advice.",
            "Make an independent friend judgment from the advisor's own evidence and understanding of the requester.",
            "Use RetainSet to name a shorter subset of candidates that fit the requester when evidence allows.",
            "If the retained set cannot be shortened, keep RetainSet as none and explain what evidence is missing.",
            "Do not evaluate a hidden CurrentProposal; the advisor-visible decision space is only the HesitationSet.",
        ],
        "output_contract": [
            "Advice should be keep, switch, or unresolved.",
            "SuggestedItem names the primary candidate behind the recommendation; RetainSet carries the shorter interested set.",
            "IndependentThinking must summarize the whole-set judgment.",
            "ReasonForUser must explain why the retained set fits this requester.",
        ],
        "advisor_output_format": advisor_output_format_for_how("single-promotion"),
        "handoff": [
            "Pass RetainSet, weak-support candidates, and unresolved missing evidence to the user's post-feedback decision.",
        ],
    },
    ("how", "multi-cooperative-warning"): {
        "role": "Run the communication as cooperative risk screening over the HesitationSet.",
        "objective": "Help the user shrink the HesitationSet by identifying which candidates should be removed, retained, or left unresolved because evidence is missing.",
        "required_actions": [
            "Treat multi-cooperative-warning as cooperative exclusion / hesitation-set reduction over the whole HesitationSet, not collective opposition to a protected favorite.",
            "No HesitationSet candidate should be treated as automatically protected just because it was the Stage1 top candidate internally.",
            "Each advisor should read the whole HesitationSet and UserInitialChoiceContext, then make an independent friend judgment using their own evidence and understanding of the requester before looking at group memory.",
            "After independent thinking, use AdvisorWorkingMemory to agree with, refine, challenge, or add to earlier friends' points.",
            "If an earlier warning matches the advisor's own judgment, the advisor may agree, but should explain the independent evidence or reasoning behind that agreement.",
            "Silent HesitationSet candidates and AdvisorKnownFocusItems are useful places to look, but advisors should not invent new concerns just to avoid overlap.",
            "ShrinkSet may contain multiple candidates the user is likely not interested in; eliminate or down-rank only with concrete user-centered mismatch evidence.",
            "If no candidate can be safely removed, keep ShrinkSet as none and explain why the set cannot be narrowed yet.",
            "If an advisor lacks evidence for a candidate, say MissingEvidence instead of turning silence into a negative claim.",
            "The desired handoff is a narrower retained HesitationSet plus explicit eliminated candidates and unresolved gaps.",
        ],
        "output_contract": [
            "Advice should usually be caution, keep, or unresolved; switch is not the goal of this mode.",
            "SuggestedItem names the primary candidate behind the advice; ShrinkSet carries the broader removal/down-rank set.",
            "IndependentThinking must summarize the advisor's whole-set judgment before group-memory response.",
            "ReasonForUser should explain how the advice narrows the user's hesitation.",
            "Concern should be candidate-specific; generic opposition to one anchored favorite is not enough.",
            "MissingEvidence should name silent HesitationSet candidates or AdvisorKnownFocusItems gaps that still need coverage.",
        ],
        "advisor_output_format": advisor_output_format_for_how("multi-cooperative-warning"),
        "handoff": [
            "Pass eliminated candidates, retained candidates, and unresolved evidence gaps to the user's post-feedback decision.",
            "If the HesitationSet cannot be narrowed, preserve the specific missing evidence rather than forcing consensus.",
        ],
    },
    ("how", "multi-cooperative-promotion"): {
        "role": "Run the communication as cooperative positive-evidence coverage over the HesitationSet.",
        "objective": "Help the user understand which candidates deserve to remain under consideration by giving support evidence grounded in each advisor's own judgment.",
        "required_actions": [
            "Treat multi-cooperative-promotion as shared support coverage, not a competition for one forced winner.",
            "Read the whole HesitationSet and UserInitialChoiceContext, then make an independent retained-set judgment from your own evidence and understanding of the requester before looking at group memory.",
            "After independent thinking, use AdvisorWorkingMemory to agree with, refine, challenge, or add to earlier friends' points.",
            "If an earlier support point matches your own judgment, you may agree, but explain your independent reason instead of mechanically paraphrasing it.",
            "Silent promising HesitationSet candidates and AdvisorKnownFocusItems are useful places to look, but do not invent support just to sound different.",
            "RetainSet should be shorter than the original HesitationSet whenever evidence allows; include candidates the user is likely interested in.",
            "Support a candidate only with concrete user-centered fit evidence; otherwise mark the support weak or unresolved.",
            "Do not over-promote any one candidate simply because it appeared stronger in the user's initial private reasoning.",
            "The desired handoff is a clearer retained set: candidates with direct support, candidates with weak support, and candidates still missing evidence.",
        ],
        "output_contract": [
            "Advice should usually be keep, switch, or unresolved depending on whether the support is strong enough.",
            "SuggestedItem names the primary candidate behind the advice; RetainSet carries the shorter set of candidates still worth considering.",
            "IndependentThinking must summarize the advisor's whole-set judgment before group-memory response.",
            "ReasonForUser should explain the advisor's own support evidence and how it relates to earlier advisor points.",
            "Concern should name the limitation of this support evidence when it is not decisive.",
            "MissingEvidence should name support gaps for promising silent candidates or AdvisorKnownFocusItems.",
        ],
        "advisor_output_format": advisor_output_format_for_how("multi-cooperative-promotion"),
        "handoff": [
            "Pass supported candidates, weakly supported candidates, and unresolved positive-evidence gaps to the user's post-feedback decision.",
            "If no candidate has decisive support, preserve the strongest retained candidates rather than forcing a winner.",
        ],
    },
    ("how", "multi-competitive-warning"): {
        "role": "Run the communication as competition among risk claims.",
        "objective": "Help the user decide which HesitationSet candidate is most worth excluding by comparing and challenging risk evidence.",
        "required_actions": [
            "Treat multi-competitive-warning as risk-claim competition, not cooperative coverage.",
            "Read the whole HesitationSet and UserInitialChoiceContext, then form an independent risk ranking before looking at group memory.",
            "After independent thinking, use AdvisorWorkingMemory to judge whether to agree, challenge, refine, or propose another risk claim.",
            "You may agree with a previous risk claim when your own evidence supports it; explain why instead of mechanically paraphrasing.",
            "Prefer a different candidate only when your own evidence makes its risk stronger than an earlier claim.",
            "Acknowledge when a previous risk claim is stronger than yours; competition can accept a better claim.",
            "ShrinkSet may contain the candidate or candidates whose risk makes them worth removing or down-ranking for this user.",
            "If risk evidence is too weak to exclude any candidate, output unresolved rather than inventing a loser.",
        ],
        "output_contract": [
            "Advice should be caution or unresolved; the mode is about exclusion risk, not final recommendation.",
            "SuggestedItem names the primary candidate attached to your strongest risk claim; ShrinkSet carries the broader removal/down-rank set.",
            "IndependentThinking must summarize the advisor's whole-set risk judgment before group-memory response.",
            "ReasonForUser should state the risk claim in user-centered terms.",
            "Concern should compare this risk against a rival risk claim when prior memory makes that useful.",
            "ResponseToPrevious should identify the earlier claim being agreed with, challenged, accepted, or refined when relevant.",
        ],
        "advisor_output_format": advisor_output_format_for_how("multi-competitive-warning"),
        "handoff": [
            "Pass the strongest risk claim, challenged weak risk claims, and remaining exclusion uncertainty to the user's post-feedback decision.",
            "Do not treat vote count as the winner; preserve why the strongest risk evidence is stronger.",
        ],
    },
    ("how", "multi-competitive-promotion"): {
        "role": "Run the communication as competition among support claims.",
        "objective": "Help the user decide which HesitationSet candidate has the strongest positive evidence by comparing and challenging support claims.",
        "required_actions": [
            "Treat multi-competitive-promotion as support-claim competition, not cooperative support coverage.",
            "Read the whole HesitationSet and UserInitialChoiceContext, then form an independent support ranking before looking at group memory.",
            "After independent thinking, use AdvisorWorkingMemory to judge whether to agree, challenge, refine, or propose another support claim.",
            "You may agree with a previous support claim when your own evidence supports it; explain why instead of mechanically paraphrasing.",
            "Prefer a different candidate only when your own evidence makes its support stronger than an earlier claim.",
            "RetainSet should be shorter than the original HesitationSet whenever evidence allows; include candidates the user is likely interested in.",
            "Challenge shallow support such as brand/category matching when another candidate has more direct user-fit evidence.",
            "If no candidate clearly wins the support competition, output unresolved and preserve the remaining comparison gap.",
        ],
        "output_contract": [
            "Advice should be keep, switch, or unresolved depending on whether your support claim beats rivals.",
            "SuggestedItem names the primary candidate attached to your strongest support claim; RetainSet carries the shorter set of candidates still worth considering.",
            "IndependentThinking must summarize the advisor's whole-set support judgment before group-memory response.",
            "ReasonForUser should state the support claim in user-centered terms.",
            "Concern should explain why a rival support claim is weaker or what prevents a clear winner when relevant.",
            "ResponseToPrevious should identify the earlier claim being agreed with, challenged, accepted, or refined when relevant.",
        ],
        "advisor_output_format": advisor_output_format_for_how("multi-competitive-promotion"),
        "handoff": [
            "Pass the strongest support claim, rejected weak support claims, and remaining comparison uncertainty to the user's post-feedback decision.",
            "Do not treat vote count as the winner; preserve why the strongest positive evidence is stronger.",
        ],
    },
    ("how", "feedback-cooperative-repair"): {
        "role": "Run follow-up communication as cooperative continued conversation with the same friends.",
        "objective": "Answer the user's FeedbackToAdvisors directly while using previous-round advisor memory as shared context.",
        "required_actions": [
            "Treat previous_user_feedback as the primary task and answer it before adding any generic advice.",
            "Use previous_round_discussion_memory to remember what each friend already said in earlier rounds.",
            "Use current_round_discussion_memory to avoid losing the flow inside this round.",
            "Cooperate by adding complementary evidence, clarifying earlier evidence, or saying which requested comparison remains unresolved.",
            "Do not restart first-round generic evaluation or repeat only the previously supported candidate.",
        ],
        "output_contract": [
            "AnsweredFeedback must name the user question being answered.",
            "ReasonForUser must directly address the requested candidate comparison or evidence gap.",
            "ResponseToPrevious must say how this relates to previous-round advisor memory.",
            "MissingEvidence must preserve any part of the user request still unanswered.",
        ],
        "advisor_output_format": advisor_output_format_for_how("feedback-cooperative-repair"),
        "handoff": [
            "Pass a compact cooperative repair result that states what was answered, what changed, and what remains unresolved.",
        ],
    },
    ("how", "feedback-competitive-repair"): {
        "role": "Run follow-up communication as competitive continued conversation with the same friends.",
        "objective": "Answer the user's FeedbackToAdvisors by testing rival claims, challenging weak assumptions, and comparing unresolved candidates.",
        "required_actions": [
            "Treat previous_user_feedback as the primary task and answer it before adding any generic advice.",
            "Use previous_round_discussion_memory to identify claims that should be accepted, challenged, or refined.",
            "Use current_round_discussion_memory to continue the debate rather than restarting it.",
            "Compete by testing which claim best answers the user's question, not by opposing for style.",
            "Do not repeat only the previous winner unless you explain why it beats the named rivals in the user's feedback.",
        ],
        "output_contract": [
            "AnsweredFeedback must name the user question being answered.",
            "ReasonForUser must directly compare or challenge the claims named in the user request.",
            "ResponseToPrevious must identify the previous claim accepted, challenged, or refined.",
            "MissingEvidence must preserve any unresolved comparison.",
        ],
        "advisor_output_format": advisor_output_format_for_how("feedback-competitive-repair"),
        "handoff": [
            "Pass a compact competitive repair result with accepted claims, challenged claims, and unresolved comparison gaps.",
        ],
    },
    ("how", "cooperative-inquiry"): {
        "role": "Run the communication as cooperative inquiry when the user needs clarification rather than conflict.",
        "objective": "Clarify missing evidence, ask and answer concrete doubts, and summarize agreement plus unresolved uncertainty without forcing a winner.",
        "required_actions": [
            "Use this mode when CandidateEvidence is sparse, uncertain, or mostly explanatory rather than truly conflicting.",
            "Each advisor should give one candidate-level observation with evidence, confidence, and missing evidence.",
            'Why disagreeing, ask a concrete question or doubt instead of attacking.',
            "A questioned advisor should answer the doubt from available target history, target profile, candidate evidence, or explicitly say evidence is lacking.",
            "After at most two discussion passes, state agreement, remaining doubts, and whether the user should keep, switch, or continue.",
            "If nobody is convinced, output unresolved or partially_resolved; do not invent consensus.",
        ],
        "output_contract": [
            "Produce AgreedEvidence, CandidateEvidenceCoverage, QuestionsAnswered, UnresolvedDoubts, SilentCandidates, and SupportOnlyWarnings.",
            "State what evidence is still missing if the discussion cannot settle the choice.",
            "Avoid theatrical confrontation, naive vote counts, or unnecessary item-selection framing.",
        ],
        "handoff": [
            "Pass a compact evidence package to the user; if unresolved, preserve retained candidates and unanswered questions.",
        ],
    },
    ("how", "pairwise-debate"): {
        "role": "Run the communication as a bounded two-candidate debate.",
        "objective": "Differentiate the two strongest conflicting candidates by requiring direct challenge, answer, rebuttal, and closing.",
        "required_actions": [
            "Use this mode only when two serious candidates remain or when the discussion has narrowed to two.",
            "Each side should defend a candidate only when it can ground the defense in evidence, and should challenge rival points when the challenge is meaningful.",
            "Each challenged side should answer from evidence or say the evidence is lacking rather than mechanically repeating its opening claim.",
            "A rebuttal should mention the opponent's actual point when it is responding to one.",
            "Limit friend discussion to two passes by default and three only when the second pass introduces genuinely new evidence.",
            "If neither side wins, output unresolved with both retained candidates, the main conflict, and unresolved questions.",
            "Do not recommend a third unrelated option outside the current HesitationSet.",
            "Make the disagreement explicit, concrete, and user-specific rather than abstract or generic.",
        ],
        "output_contract": [
            "Return CandidateA and CandidateB evidence, answered challenges, unresolved risks, DebateResult, and AdviceToUser.",
            "Preserve unresolved comparison dimensions if the debate does not settle them.",
            "Do not treat support count as the result; evaluate which candidate survived direct comparison better.",
        ],
        "handoff": [
            "Pass contrastive evidence to the user; if unresolved, pass retained candidates and the strongest open question.",
        ],
    },
    ("how", "multi-candidate-debate"): {
        "role": "Run the communication as multi-candidate elimination debate.",
        "objective": "Reduce three or more plausible candidates into a shorter retained set by finding weak fits, missing evidence, and avoidable mistakes before trying to name a winner.",
        "required_actions": [
            "Use this mode when three or more HesitationSet candidates remain plausible and at least two advisors are available.",
            "Do not assign each advisor to blindly represent one candidate; every advisor should consider the whole HesitationSet.",
            "Screening phase: identify candidates that seem least supported by target history/profile/advisor evidence.",
            "Challenge phase: challenge weak reasoning links or mismatch risks for candidates likely to be removed.",
            "Answer phase: if another advisor doubts an elimination, answer the doubt using target-user evidence.",
            "Elimination phase: remove weak candidates only with explicit item-level reasons; silence is missing evidence, not elimination.",
            "If two candidates remain, recommend pairwise-debate for the next round.",
            "If nobody is convinced, output partially_resolved or unresolved instead of a fake winner.",
            "Do not introduce candidates outside HesitationSet.",
            "Keep the interaction recommendation-focused so the final summary can expose eliminated candidates, retained candidates, unresolved questions, and next state.",
        ],
        "output_contract": [
            "Return ScreeningEvidence, ChallengesAndAnswers, EliminatedCandidates, RetainedCandidates, UnresolvedQuestions, and NextCommunicationState.",
            "Surface why candidates were retained or removed, not just one consensus sentence.",
            "Do not collapse disagreement into equal scores or naive vote counts.",
        ],
        "handoff": [
            "Pass a compact multi-candidate evidence summary; if only two candidates remain, hand off to pairwise-debate.",
        ],
    },
    ("candidate_extensions", "weak-tie-advisors"): {
        "role": "Introduce weak-tie social evidence from outside the direct and two-hop comfort zone.",
        "objective": "Test whether broader social diversity adds useful signal without overwhelming core preference evidence.",
        "required_actions": [
            "Bring in novel but still socially adjacent evidence.",
            "Treat weak-tie evidence as exploratory and lower-priority than trusted evidence.",
            "Avoid finalizing purely on weak-tie novelty unless later validation supports it.",
        ],
        "output_contract": [
            "Clearly label the value and risk of weak-tie advice.",
        ],
        "handoff": [
            "Use this node mainly as a trial extension until repeated success justifies activation.",
        ],
    },
    ("candidate_extensions", "domain-experts"): {
        "role": "Inject specialized knowledge when the item domain demands expertise beyond ordinary preference evidence.",
        "objective": "Improve recommendation quality in cases where expert interpretation matters more than ordinary social or similarity evidence.",
        "required_actions": [
            "Focus on domain-specific discriminators that ordinary users may miss.",
            "Avoid replacing user preference with pure expert taste.",
            "Keep expert guidance tied back to the target user's likely fit.",
        ],
        "output_contract": [
            "Provide expert-level justification while remaining recommendation-centric.",
        ],
        "handoff": [
            "Use as a low-weight trial node until enough successful trajectories justify broader activation.",
        ],
    },
    ("candidate_extensions", "evidence-aggregation"): {
        "role": "Aggregate multiple advisor signals without staging direct rebuttal.",
        "objective": "Collect broad evidence coverage first, then summarize support, opposition, and gaps.",
        "required_actions": [
            "Preserve multiple viewpoints without forcing debate structure.",
            "Emphasize count, coverage, and evidence consistency.",
            "Do not collapse nuanced evidence into a single sentence too early.",
        ],
        "output_contract": [
            "Return structured evidence blocks suitable for later arbitration or evaluation.",
        ],
        "handoff": [
            "Use as an inactive extension until validated through repeated trajectory patterns.",
        ],
    },
    ("candidate_extensions", "devils-advocate"): {
        "role": "Actively challenge an apparently high-confidence HesitationSet candidate to search for hidden weaknesses.",
        "objective": "Stress-test apparently strong recommendations before the system becomes overconfident.",
        "required_actions": [
            "Search for counter-evidence, overlooked rivals, and preference mismatch risks.",
            "Do not oppose a candidate just for style; oppose only with concrete reasoning.",
            "Preserve useful support evidence even while challenging the current direction.",
        ],
        "output_contract": [
            "Return explicit challenge points and say whether they truly destabilize the candidate.",
        ],
        "handoff": [
            "Use as a guarded extension node with low-probability trial insertion.",
        ],
    },
    ("patterns", "validate-then-debate"): {
        "role": "Coordinate a higher-order pattern where a strong HesitationSet candidate is first validated, then debated only if conflict remains.",
        "objective": "Avoid unnecessary debate when validation already stabilizes the choice, but escalate to debate when validation leaves residual conflict.",
        "required_actions": [
            "Treat validation as the first gate and debate as a conditional second phase.",
            "Only escalate when important uncertainty survives the validation phase.",
            "Preserve path-to-path transition rationale for later evaluation.",
        ],
        "output_contract": [
            "Describe what validation settled and what debate still had to resolve.",
        ],
        "handoff": [
            "Keep this pattern in trial mode until repeated successful trajectories support activation.",
        ],
    },
    ("patterns", "competition-then-validation"): {
        "role": "Coordinate a higher-order pattern where competitive candidate selection is followed by winner validation.",
        "objective": "First surface a strong winner under competition, then check whether that winner truly deserves finalization.",
        "required_actions": [
            "Treat competition as candidate discovery and validation as stability checking.",
            "Do not let competition alone guarantee finalization.",
            "Record when the validation step overturns or weakens the competitive winner.",
        ],
        "output_contract": [
            "State both the competition winner and the outcome of later validation.",
        ],
        "handoff": [
            "Keep this pattern as a trial path until validated by offline trajectory evidence.",
        ],
    },
}


class PublicTreeStore:
    def __init__(self, base_dir, refresh_layout_on_load=True, dataset=""):
        self.base_dir = Path(base_dir)
        self.index_dir = self.base_dir / "indexes"
        self._cache = None
        self.refresh_layout_on_load = bool(refresh_layout_on_load)
        self.dataset = str(dataset or self.base_dir.name or "").strip()

    @staticmethod
    def _all_known_nodes():
        rows = []
        for level, items in INITIAL_ACTIVE_NODES.items():
            for node_id, description in items:
                rows.append((level, node_id, description, "active"))
        return rows

    @staticmethod
    def _valid_lifecycle(status):
        return str(status or "").strip() in {"sprout", "active", "withered"}

    @staticmethod
    def _who_policy_like_node_names():
        return {
            "distinct-perspective",
            "distinct-perspective-trusted",
            "high-reliability",
            "high-reliability-trusted",
            "disagreement-seeking",
            "evidence-rich",
            "evidence-seeking",
            "counterargument-seeking",
        }

    @staticmethod
    def infer_who_subgroup_metadata(node_id, node=None):
        node_id = str(node_id or "").strip()
        node = dict(node or {}) if isinstance(node, dict) else {}
        if "/" not in node_id:
            return {
                "who_node_kind": str(node.get("who_node_kind", "") or "source"),
                "advisor_source": node_id,
                "retrieval_constraints": dict(node.get("retrieval_constraints", {}) or {}),
            }
        parts = [part for part in node_id.split("/") if part]
        source = str(node.get("advisor_source", "") or parts[0])
        leaf_tokens = set()
        for part in parts[1:]:
            leaf_tokens.update(tok for tok in re.split(r"[^a-z0-9]+", part.lower()) if tok)
        constraints = dict(node.get("retrieval_constraints", {}) or {})
        if "mutual" in leaf_tokens:
            constraints.setdefault("trust_relation", "mutual-trust")
        elif "one" in leaf_tokens and "way" in leaf_tokens:
            constraints.setdefault("trust_relation", "one-way-trust")
        if "multi" in leaf_tokens and "trust" in leaf_tokens:
            constraints.setdefault("trust_scope", "multi-trust")
        elif "single" in leaf_tokens and "trust" in leaf_tokens:
            constraints.setdefault("trust_scope", "single-trust")
        if "dissimilar" in leaf_tokens or "diverse" in leaf_tokens:
            constraints.setdefault("history_similarity", "dissimilar")
        elif "similar" in leaf_tokens or "nearest" in leaf_tokens or "neighbor" in leaf_tokens:
            constraints.setdefault("history_similarity", "similar")
        if "item" in leaf_tokens and "experienced" in leaf_tokens:
            constraints.setdefault("requires_item_experience", True)
        if "two" in leaf_tokens and "hop" in leaf_tokens:
            constraints.setdefault("hop", 2)
        return {
            "who_node_kind": str(node.get("who_node_kind", "") or "source_subgroup"),
            "advisor_source": source,
            "retrieval_constraints": constraints,
        }

    @staticmethod
    def infer_trial_anchor_node(layer, node_id, node=None, tree_nodes=None):
        layer = str(layer or "").strip()
        node_id = str(node_id or "").strip().strip("/")
        node = dict(node or {}) if isinstance(node, dict) else {}
        explicit = str(node.get("trial_anchor_node", "") or "").strip().strip("/")
        tree_nodes = dict(tree_nodes or {})

        def valid(anchor):
            anchor = str(anchor or "").strip().strip("/")
            if not anchor or anchor == node_id or "/" in anchor:
                return ""
            if tree_nodes and anchor not in tree_nodes:
                return ""
            return anchor

        explicit = valid(explicit)
        if explicit:
            return explicit

        text = " ".join(
            [
                node_id,
                str(node.get("description", "") or ""),
                str(node.get("use_why", "") or ""),
                str(node.get("if_selected", "") or ""),
                str(node.get("evidence_pattern", "") or ""),
                str(node.get("skill_body", "") or ""),
            ]
        ).lower()
        tokens = set(tok for tok in re.split(r"[^a-z0-9]+", text) if tok)
        if layer == "what":
            if {"candidate", "contrast"} & tokens or "evidence" in tokens or "compare" in tokens:
                anchor = valid("compare_remaining_candidates")
                if anchor:
                    return anchor
            if "reduce" in tokens or "hesitation" in tokens:
                anchor = valid("reduce_hesitation_set")
                if anchor:
                    return anchor
        if layer == "how":
            if {"opposition", "oppose", "rebuttal", "rebut", "counterargument", "competitive", "challenge"} & tokens:
                anchor = valid("multi-competitive")
                if anchor:
                    return anchor
            if {"cooperative", "coverage", "consensus", "support"} & tokens:
                anchor = valid("multi-cooperative")
                if anchor:
                    return anchor
        if layer == 'why':
            if "candidate" in tokens and "conflict" in tokens:
                anchor = valid("candidate-conflict")
                if anchor:
                    return anchor
            if ("prior" in tokens or "internal" in tokens) and "conflict" in tokens:
                anchor = valid("internal-prior-conflict")
                if anchor:
                    return anchor
        return ""

    @staticmethod
    def _is_selectable_status(status, stage="test"):
        status = str(status or "active").strip()
        if status == "active":
            return True
        if status == "sprout":
            return str(stage or "").strip().lower() == "train"
        return False

    @staticmethod
    def _safe_node_id(node_id):
        node_id = str(node_id or "").strip().strip("/")
        node_id = node_id.replace("\\", "/")
        if not node_id:
            return ""
        parts = [part for part in node_id.split("/") if part]
        safe_parts = []
        for part in parts:
            part = part.strip()
            if not re.match(r"^[a-z0-9][a-z0-9_-]*$", part):
                return ""
            safe_parts.append(part)
        return "/".join(safe_parts)

    @staticmethod
    def _node_depth(node_id):
        return len([part for part in str(node_id or "").strip("/").split("/") if part])

    def _node_dir_for_id(self, level, node_id):
        safe = self._safe_node_id(node_id)
        if not safe:
            return None
        return self.base_dir / str(level) / Path(*safe.split("/"))

    def _node_instruction_config(self, node_level, node_id, description, status):
        config = dict(NODE_INSTRUCTION_LIBRARY.get((node_level, node_id), {}) or {})
        has_custom_config = bool(config)
        action_rule = dict(NODE_ACTION_RULES.get((node_level, node_id), {}) or {})
        config.setdefault("action", action_rule.get("action", f"Execute the {node_id} communication-tree node."))
        config.setdefault("rule", action_rule.get("rule", str(description or "")))
        level = str(node_level or "")
        config.setdefault("use_why", str(description or ""))
        if level == 'why':
            config.setdefault(
                "if_selected",
                "Communication is triggered; next choose a how mode that fits this uncertainty and then choose who should answer it.",
            )
        elif level == "who":
            config.setdefault("if_selected", action_rule.get("action", f"Retrieve advisors from the {node_id} source."))
            config.setdefault("advisor_role", str(node_id or "advisor"))
        elif level == "how":
            config.setdefault("if_selected", action_rule.get("action", f"Use the {node_id} communication mode."))
        if str(node_level) == "how" and not has_custom_config:
            shape = infer_communication_shape(node_id)
            family = infer_communication_family(node_id)
            config.update(
                {
                    "role": {
                        "single": "Run one friend as an independent advisor.",
                        "cooperative": "Run multiple friends as a cooperative team around the same UserTask.",
                        "competitive": "Run multiple friends as claim-makers and claim-testers around the same UserTask.",
                    }.get(family, f"Execute {node_id} as the Communication Mode node."),
                    "objective": {
                        "single": "Give one grounded friend answer to the UserTask.",
                        "cooperative": "Let friends think independently, then complement, refine, integrate, or agree with useful earlier points.",
                        "competitive": "Let the first friend make a claim, then require later friends to question or rebut a prior claim before stating their own judgment.",
                    }.get(family, str(description or "")),
                    "required_actions": {
                        "single": [
                            "Read UserTask, the full HesitationSet, and UserInitialChoiceContext before answering.",
                            "Answer as the requester's friend from your own evidence and understanding of the requester.",
                            "Do not choose a final item for the user; provide advice, uncertainty, and missing evidence.",
                        ],
                        "cooperative": [
                            "All advisors share the same UserTask.",
                            "Each advisor should think independently before reading group memory.",
                            "After independent thinking, use group memory to complement, refine, integrate, or agree with earlier useful points.",
                            "Do not invent a different opinion just to avoid overlap; justified agreement is allowed.",
                            "Keep continuity across follow-up rounds with previous_round_discussion_memory.",
                        ],
                        "competitive": [
                            "All advisors answer the same UserTask through competing claims.",
                            "Each advisor should form an independent claim before reading group memory.",
                            "Advisor 1 opens with a clear candidate-level claim.",
                            "Every later advisor must identify one previous advisor claim and question, rebut, or correct it before giving their own judgment.",
                            "Do not answer with generic agreement, generic support, or a summary of prior views; if a prior claim is strong, challenge its weakest assumption, missing evidence, overreach, or untested comparison.",
                            "Keep continuity across follow-up rounds with previous_round_discussion_memory.",
                        ],
                    }.get(family, []),
                    "output_contract": [
                        "UserTask defines what content to answer; how only defines organization.",
                        "Use exact HesitationSet item names and never introduce outside items.",
                        "Return the task-specific output fields supplied by the selected what node.",
                        "If evidence is partial, answer as far as possible and name MissingEvidence.",
                    ],
                    "advisor_output_format": advisor_output_format_for_how(node_id),
                    "handoff": [
                        "Pass concise advisor evidence, group-memory responses, and remaining gaps to the user's post-feedback decision.",
                    ],
                }
            )
        config.setdefault("role", f"Execute the {node_id} node inside the Public Communication Skill Tree.")
        config.setdefault("objective", description)
        config.setdefault(
            "required_actions",
            [
                "Follow the node intent faithfully.",
                "Keep the recommendation tied to the current communication path.",
                "Return structured, recommendation-relevant evidence rather than generic filler.",
            ],
        )
        config.setdefault(
            "output_contract",
            [
                "Make support, opposition, and remaining uncertainty explicit.",
                "Stay compatible with later feedback absorption and arbitration.",
            ],
        )
        if str(node_level) == "what":
            config["role"] = "Interpret the user's natural-language communication task."
            config["objective"] = {
                "reduce_hesitation_set": "Help advisors answer which HesitationSet candidates can be removed, down-ranked, retained, or left unresolved.",
                "find_interested_subset": "Help advisors answer which smaller subset the user is likely interested in keeping.",
                        "compare_remaining_candidates": "Help advisors compare remaining candidates or test rival claims.",
                        "evidence_gap_check": "Help advisors identify insufficient original reasons and specify what supplementary evidence is needed.",
                        "reasoning_check": "Help advisors verify whether the user's initial reasoning or assumptions are valid.",
                        "none": "Preserve an unmapped natural-language task for execution and later evolution.",
            }.get(str(node_id), "Decide what the friends should answer, while how decides how they organize.")
            config["required_actions"] = [
                "Use UserTask as the content goal.",
                "Keep the advisor answer inside the HesitationSet.",
                "Do not select who or how; this node only defines what the advisors should answer.",
                "If the task cannot be mapped, preserve it as none so later evolution can learn a new branch.",
            ]
            config["output_contract"] = [
                "Task output requirements are passed to advisors through the selected what node.",
                "The user still makes the final item decision after seeing advisor evidence.",
                "Do not turn what into an advisor organization rule.",
            ]
            config["task_output_format"] = task_output_format_for_what(node_id)
            config["summary_hints"] = DEFAULT_SUMMARY_HINTS.get(
                ("what", node_id),
                DEFAULT_SUMMARY_HINTS[("what", "none")],
            )
            config["handoff"] = [
                "Pass the task-specific output format to advisors together with UserTask.",
            ]
            config.setdefault(
                "if_selected",
                "Use this task type to decide what advisor evidence should answer.",
            )
        if str(node_level) == "how":
            config.setdefault("advisor_output_format", advisor_output_format_for_how(node_id))
            config.setdefault(
                "summary_hints",
                {
                    "task_focus": "preserve how-specific advisor interaction signals",
                    "important_output_fields": [],
                    "preserve_interaction_fields": list(config.get("advisor_output_format", []) or []),
                },
            )
        config.setdefault(
            "handoff",
            [
                "Pass forward only evidence that helps the next stage make a better decision.",
            ],
        )
        config["description"] = str(description or "")
        config["status"] = str(status or "active")
        config["node_level"] = str(node_level or "")
        config["node_id"] = str(node_id or "")
        return config

    def _node_skill_markdown(self, node_id, description, node_level, status):
        cfg = self._node_instruction_config(node_level=node_level, node_id=node_id, description=description, status=status)

        def bullet_block(items):
            return "\n".join([f"- {str(item)}" for item in (items or [])]) if items else "- none"

        status_note = {
            "sprout": "This node is sprout and may only be trialed during train.",
            "active": "This node is active and may be selected during normal path selection.",
            "withered": "This node is withered and should not be selected by default, but remains for audit history.",
            "trial": "This node is in trial mode and should be inserted cautiously with explicit evaluation.",
            "inactive": "This node is inactive by default and should only appear through controlled trial insertion.",
        }.get(str(status), "This node belongs to the Public Communication Skill Tree.")

        body = (
            f"---\n"
            f"name: {node_id}\n"
            f"description: {description}\n"
            f"level: {node_level}\n"
            f"status: {status}\n"
            f"---\n\n"
            f"# {node_id}\n\n"
            f"## Use When\n"
            f"{cfg.get('use_why', description)}\n\n"
            f"## If Selected\n"
            f"{cfg.get('if_selected', cfg.get('action', 'Use this node in the selected communication path.'))}\n\n"
        )
        if str(node_level) == 'why':
            body += (
                "## Trigger Role\n"
                "This why node is a trigger condition only. If it does not match the user's current DecisionState, communication should be skipped rather than forced.\n\n"
            )
        if str(node_level) == "who":
            body += (
                "## Advisor Role Label\n"
                f"{cfg.get('advisor_role', node_id)}\n\n"
                "## Retrieval Policy\n"
                f"{cfg.get('rule', description)}\n\n"
            )
        if str(node_level) == "what":
            body += (
                "## User Task Type\n"
                f"{cfg['role']}\n\n"
                "### Goal\n"
                f"{cfg['objective']}\n\n"
                "### Required Actions\n"
                f"{bullet_block(cfg.get('required_actions'))}\n\n"
                "### Advisor Output Format For This Task\n"
                f"{bullet_block(cfg.get('task_output_format'))}\n\n"
            )
        if str(node_level) == "how":
            body += (
                "## Advisor Communication Skill\n"
                f"{cfg['role']}\n\n"
                "### Goal\n"
                f"{cfg['objective']}\n\n"
                "### Required Actions\n"
                f"{bullet_block(cfg.get('required_actions'))}\n\n"
                "### Output Contract\n"
                f"{bullet_block(cfg.get('output_contract'))}\n\n"
                "### Advisor Output Format\n"
                f"{bullet_block(cfg.get('advisor_output_format'))}\n\n"
                "### Stop And Handoff\n"
                f"{bullet_block(cfg.get('handoff'))}\n\n"
            )
        body += (
            "## Runtime Status\n"
            f"{status_note}\n"
        )
        return body

    def _node_openai_yaml(self, node_id, description):
        return (
            "interface:\n"
            f"  display_name: \"{node_id}\"\n"
            f"  short_description: \"{description}\"\n"
            f"  default_prompt: \"Execute the {node_id} node as an instruction-bearing component inside the Public Communication Skill Tree.\"\n\n"
            "policy:\n"
            "  allow_implicit_invocation: true\n"
        )

    def _node_spec(self, node_id, description, node_level, status, cfg=None):
        cfg = dict(cfg or self._node_instruction_config(node_level, node_id, description, status))
        spec = {
            "node_id": str(node_id),
            "node_level": str(node_level),
            "status": str(status),
            "description": str(description),
            "use_why": str(cfg.get("use_why", description)),
            "if_selected": str(cfg.get("if_selected", cfg.get("action", ""))),
            "advisor_role": str(cfg.get("advisor_role", node_id if node_level == "who" else "")),
            "advisor_output_format": list(cfg.get("advisor_output_format", []) or []) if node_level == "how" else [],
            "task_output_format": list(cfg.get("task_output_format", []) or []) if node_level == "what" else [],
            "summary_hints": dict(cfg.get("summary_hints", {}) or {}) if node_level in {"what", "how"} else {},
            "selection_prior": 1.0 if status == "active" else (0.15 if status == "sprout" else 0.0),
            "applicability_condition": {},
            "execution_hint": {},
            "evolution_state": {
                "tt": 0,
                "wt": 0,
                "tw": 0,
                "ww": 0,
            },
        }
        if self.dataset:
            spec.setdefault("dataset", self.dataset)
            spec.setdefault("source_dataset", self.dataset)
            spec.setdefault("dataset_scope", [self.dataset])
        profile = dict(cfg.get("selection_profile", {}) or DEFAULT_SELECTION_PROFILES.get((node_level, node_id), {}) or {})
        if profile and node_level in {'why', "what"}:
            spec["selection_profile"] = profile
        return spec

    def _write_or_refresh_node(self, level, node_id, description, status="active", refresh_existing=True):
        node_dir = self._node_dir_for_id(level, node_id)
        if node_dir is None:
            return
        (node_dir / "agents").mkdir(parents=True, exist_ok=True)
        (node_dir / "references").mkdir(parents=True, exist_ok=True)
        cfg = self._node_instruction_config(node_level=level, node_id=node_id, description=description, status=status)

        skill_path = node_dir / "SKILL.md"
        if refresh_existing or not skill_path.exists():
            skill_path.write_text(
                self._node_skill_markdown(node_id=node_id, description=description, node_level=level, status=status),
                encoding="utf-8",
            )
        agent_path = node_dir / "agents" / "openai.yaml"
        if refresh_existing or not agent_path.exists():
            agent_path.write_text(
                self._node_openai_yaml(node_id=node_id, description=description),
                encoding="utf-8",
            )

        spec_path = node_dir / "references" / "skill.json"
        if not refresh_existing and spec_path.exists():
            return
        existing = load_json(spec_path, default={}) or {}
        merged = self._node_spec(node_id, description, level, status, cfg=cfg)
        merged["applicability_condition"] = dict(existing.get("applicability_condition", {}) or {})
        merged["execution_hint"] = dict(existing.get("execution_hint", {}) or {})
        merged["evolution_state"] = dict(existing.get("evolution_state", {}) or merged["evolution_state"])
        if "selection_profile" in existing:
            merged["selection_profile"] = dict(existing.get("selection_profile", {}) or {})
        if "summary_hints" in existing:
            merged["summary_hints"] = dict(existing.get("summary_hints", {}) or {})
        if "selection_prior" in existing:
            merged["selection_prior"] = existing["selection_prior"]
        dump_json(spec_path, merged)
        lifecycle_path = node_dir / "references" / "lifecycle.json"
        lifecycle = load_json(lifecycle_path, default={}) or {}
        if not lifecycle:
            lifecycle = {
                "status": str(status or "active"),
                "support": 0,
                "tt": 0,
                "wt": 0,
                "tw": 0,
                "ww": 0,
                "trial_rounds": 0,
                "useful_final_t_count": 0,
                "parent_node": "/".join(str(node_id or "").split("/")[:-1]),
                **(
                    {
                        "dataset": self.dataset,
                        "source_dataset": self.dataset,
                        "dataset_scope": [self.dataset],
                    }
                    if self.dataset
                    else {}
                ),
                "promotion_history": [],
            }
            dump_json(lifecycle_path, lifecycle)

    def ensure_layout(self, refresh_existing=True):
        for folder in ['why', "what", "who", "how", "indexes"]:
            (self.base_dir / folder).mkdir(parents=True, exist_ok=True)

        for level, node_id, description, status in self._all_known_nodes():
            self._write_or_refresh_node(level, node_id, description, status=status, refresh_existing=refresh_existing)

        self.ensure_indexes()

    def reset_to_initial_layout(self):
        """Reset the public tree to the code-defined initial active nodes.

        This removes all generated child/sibling nodes and clears evolution
        indexes. It is intentionally stronger than ensure_layout(refresh=True),
        which only rewrites known initial node files.
        """
        initial_by_level = {
            level: {str(node_id) for node_id, _desc in rows}
            for level, rows in INITIAL_ACTIVE_NODES.items()
        }
        for folder in ['why', "what", "who", "how", "indexes"]:
            (self.base_dir / folder).mkdir(parents=True, exist_ok=True)

        for level, initial_nodes in initial_by_level.items():
            level_dir = self.base_dir / level
            for child in list(level_dir.iterdir()):
                if not child.is_dir():
                    continue
                if child.name not in initial_nodes:
                    shutil.rmtree(child, ignore_errors=True)
                    continue
                for nested in list(child.iterdir()):
                    if nested.is_dir() and nested.name not in {"agents", "references"}:
                        shutil.rmtree(nested, ignore_errors=True)

        for level, node_id, description, status in self._all_known_nodes():
            node_dir = self._node_dir_for_id(level, node_id)
            if node_dir is None:
                continue
            lifecycle_path = node_dir / "references" / "lifecycle.json"
            spec_path = node_dir / "references" / "skill.json"
            if lifecycle_path.exists():
                lifecycle_path.unlink()
            if spec_path.exists():
                spec_path.unlink()
            self._write_or_refresh_node(level, node_id, description, status=status, refresh_existing=True)

        self.reset_runtime_indexes()
        self._cache = None

    def ensure_indexes(self):
        active_nodes = {
            level: [node_id for node_id, _ in rows]
            for level, rows in INITIAL_ACTIVE_NODES.items()
        }
        defaults = {
            "active_nodes.json": active_nodes,
            "path_stats.json": {},
            "fine_path_stats.json": {},
            "branch_stats.json": {},
            "fine_branch_stats.json": {},
            "node_stats.json": {},
            "risky_paths.json": {},
            "sprout_trial_stats.json": {},
        }
        for filename, payload in defaults.items():
            path = self.index_dir / filename
            if not path.exists():
                dump_json(path, payload)
            elif filename in ["active_nodes.json"]:
                current = load_json(path, default={}) or {}
                changed = False
                for obsolete_level in ["patterns", "candidate_extensions"]:
                    if obsolete_level in current:
                        current.pop(obsolete_level, None)
                        changed = True
                for level, nodes in payload.items():
                    if list(current.get(level, []) or []) != list(nodes):
                        current[level] = list(nodes)
                        changed = True
                if changed:
                    dump_json(path, current)
        for filename in [
            "node_proposals.jsonl",
            "tree_patch_proposals.jsonl",
            "tree_patch_prompt_io.jsonl",
            "tree_patch_invalid.jsonl",
            "tree_batch_errors.jsonl",
            "round_failure_reflections.jsonl",
            "round_failure_clusters.jsonl",
            "tree_patch_candidates.jsonl",
            "tree_patch_critic.jsonl",
            "tree_patch_repair_io.jsonl",
            "sprout_trial_records.jsonl",
            "lifecycle_updates.jsonl",
            "tree_evolution_buffer.jsonl",
            "failure_diagnoses.jsonl",
            "evolution_log.jsonl",
        ]:
            path = self.index_dir / filename
            if not path.exists():
                path.touch()

    def reset_runtime_indexes(self):
        active_nodes = {
            level: [node_id for node_id, _ in rows]
            for level, rows in INITIAL_ACTIVE_NODES.items()
        }
        payloads = {
            "active_nodes.json": active_nodes,
            "path_stats.json": {},
            "fine_path_stats.json": {},
            "branch_stats.json": {},
            "fine_branch_stats.json": {},
            "node_stats.json": {},
            "risky_paths.json": {},
            "sprout_trial_stats.json": {},
        }
        for filename, payload in payloads.items():
            dump_json(self.index_dir / filename, payload)
        for filename in [
            "node_proposals.jsonl",
            "tree_patch_proposals.jsonl",
            "tree_patch_prompt_io.jsonl",
            "tree_patch_invalid.jsonl",
            "tree_batch_errors.jsonl",
            "round_failure_reflections.jsonl",
            "round_failure_clusters.jsonl",
            "tree_patch_candidates.jsonl",
            "tree_patch_critic.jsonl",
            "tree_patch_repair_io.jsonl",
            "sprout_trial_records.jsonl",
            "lifecycle_updates.jsonl",
            "tree_evolution_buffer.jsonl",
            "failure_diagnoses.jsonl",
            "evolution_log.jsonl",
        ]:
            path = self.index_dir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        for path in self.index_dir.glob("tree_evolution_buffer_*.jsonl"):
            path.write_text("", encoding="utf-8")
        processed_dir = self.index_dir / "processed_batches"
        processed_dir.mkdir(parents=True, exist_ok=True)
        for child in processed_dir.glob("*.jsonl"):
            child.unlink()
        self._cache = None

    def _load_node_dir(self, node_dir):
        spec_path = node_dir / "references" / "skill.json"
        skill_md_path = node_dir / "SKILL.md"
        spec = load_json(spec_path, default={}) or {}
        body = safe_read_text(skill_md_path, default="")
        lifecycle = load_json(node_dir / "references" / "lifecycle.json", default={}) or {}
        if lifecycle:
            spec.setdefault("lifecycle", lifecycle)
            spec["status"] = str(lifecycle.get("status", spec.get("status", "active")) or "active")
        spec["skill_path"] = str(skill_md_path)
        spec["skill_dir"] = str(node_dir)
        spec["skill_body"] = body
        if self.dataset:
            spec.setdefault("dataset", self.dataset)
            spec.setdefault("source_dataset", self.dataset)
            spec.setdefault("dataset_scope", [self.dataset])
        return spec

    def _iter_node_dirs(self, level_dir):
        if not level_dir.exists():
            return []
        rows = []
        for skill_path in sorted(level_dir.rglob("SKILL.md")):
            node_dir = skill_path.parent
            parts = set(part.lower() for part in node_dir.parts)
            if "agents" in parts or "references" in parts:
                continue
            rows.append(node_dir)
        return rows

    def load_tree(self, force_reload=False, refresh_layout=None):
        if self._cache is not None and not force_reload:
            return self._cache
        if refresh_layout is None:
            refresh_layout = self.refresh_layout_on_load
        self.ensure_layout(refresh_existing=bool(refresh_layout))
        tree = {}
        for level in ['why', "what", "who", "how"]:
            level_dir = self.base_dir / level
            nodes = {}
            for node_dir in self._iter_node_dirs(level_dir):
                node = self._load_node_dir(node_dir)
                rel_id = str(node_dir.relative_to(level_dir)).replace("\\", "/")
                node_id = str(node.get("node_id", rel_id) or rel_id)
                nodes[node_id] = node
            tree[level] = nodes
        tree["indexes"] = {
            "active_nodes": load_json(self.index_dir / "active_nodes.json", default={}) or {},
            "path_stats": load_json(self.index_dir / "path_stats.json", default={}) or {},
            "fine_path_stats": load_json(self.index_dir / "fine_path_stats.json", default={}) or {},
            "branch_stats": load_json(self.index_dir / "branch_stats.json", default={}) or {},
            "fine_branch_stats": load_json(self.index_dir / "fine_branch_stats.json", default={}) or {},
            "node_stats": load_json(self.index_dir / "node_stats.json", default={}) or {},
            "risky_paths": load_json(self.index_dir / "risky_paths.json", default={}) or {},
        }
        self._cache = tree
        return tree

    def get_active_nodes(self, level):
        tree = self.load_tree()
        active_ids = set((tree.get("indexes", {}).get("active_nodes", {}) or {}).get(level, []))
        return {
            node_id: node
            for node_id, node in (tree.get(level, {}) or {}).items()
            if node_id in active_ids and str(node.get("status", "active") or "active") == "active"
        }

    def get_selectable_nodes(self, level, stage="test", include_children=True):
        tree = self.load_tree()
        active_ids = set((tree.get("indexes", {}).get("active_nodes", {}) or {}).get(level, []))
        rows = {}
        for node_id, node in (tree.get(level, {}) or {}).items():
            if not include_children and "/" in str(node_id):
                continue
            status = str(node.get("status", "active") or "active")
            has_lifecycle = bool(node.get("lifecycle"))
            is_generated_or_indexed = str(node_id) in active_ids or "/" in str(node_id) or has_lifecycle
            if is_generated_or_indexed and self._is_selectable_status(status, stage=stage):
                rows[node_id] = node
        return rows

    def get_active_skill_options(self, levels=None):
        levels = list(levels or ['why', "what", "who", "how"])
        options = {}
        for level in levels:
            rows = []
            for node_id, node in self.get_active_nodes(level).items():
                rows.append(
                    {
                        "node_id": str(node_id),
                        "status": str(node.get("status", "active") or "active"),
                        "use_why": str(node.get("use_why", "") or node.get("description", "") or ""),
                        "if_selected": str(node.get("if_selected", "") or node.get("action", "") or ""),
                    }
                )
            options[level] = rows
        return options

    @staticmethod
    def _contains_forbidden_text(value, forbidden_terms):
        text = str(value or "")
        low = text.lower()
        for term in forbidden_terms or []:
            term = str(term or "").strip()
            if len(term) >= 3 and term.lower() in low:
                return True
        return False

    @staticmethod
    def _markdown_has_required_sections(skill_md, layer):
        text = str(skill_md or "")
        required = ["---", "## Use When", "## If Selected", "## Runtime Status"]
        if str(layer) == "what":
            required.append("### Advisor Output Format For This Task")
            required.append("CandidateView")
        if str(layer) == "how":
            required.append("## Advisor Communication Skill")
            required.append("### Output Contract")
        if str(layer) == "who":
            required.append("## Retrieval Policy")
        if str(layer) == 'why':
            required.append("Communication Trigger")
        return all(token in text for token in required)

    def validate_tree_patch(self, patch, forbidden_terms=None):
        patch = dict(patch or {})
        op = str(patch.get("operation", "") or "").strip()
        layer = str(patch.get("layer", "") or "").strip()
        if op not in {"add_child_node", "add_sibling_node", "mark_withered"}:
            return False, "invalid operation"
        if layer not in {'why', "what", "who", "how"}:
            return False, "invalid layer"
        if op == "mark_withered":
            node_id = self._safe_node_id(patch.get("node_id", ""))
            if not node_id:
                return False, "invalid node_id"
            if node_id not in (self.load_tree().get(layer, {}) or {}):
                return False, "node not found"
            return True, "ok"

        new_node_id = self._safe_node_id(patch.get("new_node_id", ""))
        parent = self._safe_node_id(patch.get("parent_node", ""))
        if not new_node_id:
            return False, "invalid new_node_id"
        full_node_id = f"{parent}/{new_node_id}" if op == "add_child_node" and parent and not new_node_id.startswith(parent + "/") else new_node_id
        full_node_id = self._safe_node_id(full_node_id)
        if not full_node_id:
            return False, "invalid full node id"
        if self._node_depth(full_node_id) > 3:
            return False, "node depth exceeds max depth 3"
        if op == "add_child_node":
            if not parent:
                return False, "missing parent_node for add_child_node"
            if parent not in (self.load_tree().get(layer, {}) or {}):
                return False, "parent_node not found"
            if layer == "who":
                leaf = full_node_id.split("/")[-1].strip().lower()
                if any(policy_name in leaf for policy_name in self._who_policy_like_node_names()):
                    return False, "who child must name an advisor subgroup, not an abstract selection policy"
        if layer == "who" and op == "add_sibling_node":
            return False, "new root who sources are not enabled; add a subgroup under an existing who source"
        if op == "add_sibling_node" and layer in {'why', "what", "how"}:
            tree_nodes = self.load_tree().get(layer, {}) or {}
            skill_json_for_anchor = dict(patch.get("skill_json", {}) or {})
            if patch.get("trial_anchor_node"):
                skill_json_for_anchor["trial_anchor_node"] = patch.get("trial_anchor_node")
            anchor = self.infer_trial_anchor_node(layer, full_node_id, skill_json_for_anchor, tree_nodes=tree_nodes)
            if not anchor:
                return False, "missing or invalid trial_anchor_node for sibling parent"
        if full_node_id in (self.load_tree().get(layer, {}) or {}):
            return False, "node already exists"
        status = str(patch.get("status", "") or "active").strip() or "active"
        if status not in {"active", "sprout"}:
            return False, "new node status must be active or sprout"
        skill_md = str(patch.get("skill_md", "") or "")
        skill_json = dict(patch.get("skill_json", {}) or {})
        if not self._markdown_has_required_sections(skill_md, layer):
            return False, "SKILL.md missing required sections"
        skill_status = str(skill_json.get("status", "") or status).strip() or status
        if skill_status != status:
            return False, "skill_json status must match patch status"
        if str(skill_json.get("node_level", "") or "") != layer:
            return False, "skill_json node_level mismatch"
        if self._contains_forbidden_text(skill_md, forbidden_terms) or self._contains_forbidden_text(skill_json, forbidden_terms):
            return False, "contains forbidden item/user-specific term"
        return True, "ok"

    def apply_tree_patch(self, patch, forbidden_terms=None, source="llm_tree_patch"):
        ok, reason = self.validate_tree_patch(patch, forbidden_terms=forbidden_terms)
        patch = dict(patch or {})
        event = {
            "ts": 0,
            "source": str(source or "llm_tree_patch"),
            "valid": bool(ok),
            "reason": str(reason),
            "patch": patch,
        }
        import time

        event["ts"] = int(time.time())
        if not ok:
            append_jsonl(self.index_dir / "tree_patch_invalid.jsonl", event)
            return {"applied": False, "reason": reason}

        op = str(patch.get("operation", "") or "")
        layer = str(patch.get("layer", "") or "")
        if op == "mark_withered":
            node_id = self._safe_node_id(patch.get("node_id", ""))
            node_dir = self._node_dir_for_id(layer, node_id)
            spec_path = node_dir / "references" / "skill.json"
            spec = load_json(spec_path, default={}) or {}
            previous = str(spec.get("status", "active") or "active")
            spec["status"] = "withered"
            dump_json(spec_path, spec)
            lifecycle_path = node_dir / "references" / "lifecycle.json"
            lifecycle = load_json(lifecycle_path, default={}) or {}
            previous = str(lifecycle.get("status", previous) or previous)
            lifecycle["status"] = "withered"
            lifecycle["withered_reason"] = str(patch.get("why_withered", "") or reason)
            replacement_hint = str(patch.get("replacement_hint", "") or "")
            repair_hint = str(patch.get("repair_hint", "") or replacement_hint or "")
            if repair_hint:
                hints = list(lifecycle.get("repair_hints", []) or [])
                if repair_hint not in hints:
                    hints.append(repair_hint)
                lifecycle["repair_hints"] = hints[-12:]
            failure_pattern = str(patch.get("failure_pattern", "") or patch.get("why_withered", "") or "")
            if failure_pattern:
                patterns = list(lifecycle.get("failure_patterns", []) or [])
                if failure_pattern not in patterns:
                    patterns.append(failure_pattern)
                lifecycle["failure_patterns"] = patterns[-12:]
            lifecycle.setdefault("revision_memory", []).append(
                {
                    "ts": int(time.time()),
                    "event": "marked_withered",
                    "reason": str(patch.get("why_withered", "") or reason),
                    "repair_hint": repair_hint,
                    "replacement_hint": replacement_hint,
                }
            )
            lifecycle["revision_memory"] = list(lifecycle.get("revision_memory", []) or [])[-20:]
            lifecycle.setdefault("promotion_history", []).append(
                {
                    "ts": int(time.time()),
                    "from": previous,
                    "to": "withered",
                    "reason": str(patch.get("why_withered", "") or reason),
                    "replacement_hint": replacement_hint,
                }
            )
            dump_json(lifecycle_path, lifecycle)
            skill_path = node_dir / "SKILL.md"
            body = safe_read_text(skill_path, default="")
            if body:
                body = re.sub(r"(?m)^status:\s*\S+\s*$", "status: withered", body, count=1)
                skill_path.write_text(body, encoding="utf-8")
            append_jsonl(self.index_dir / "lifecycle_updates.jsonl", event)
            self._cache = None
            return {"applied": True, "operation": op, "node_id": node_id}

        parent = self._safe_node_id(patch.get("parent_node", ""))
        new_node_id = self._safe_node_id(patch.get("new_node_id", ""))
        full_node_id = f"{parent}/{new_node_id}" if op == "add_child_node" and parent and not new_node_id.startswith(parent + "/") else new_node_id
        full_node_id = self._safe_node_id(full_node_id)
        node_dir = self._node_dir_for_id(layer, full_node_id)
        (node_dir / "agents").mkdir(parents=True, exist_ok=True)
        (node_dir / "references").mkdir(parents=True, exist_ok=True)
        skill_md = str(patch.get("skill_md", "") or "")
        skill_json = dict(patch.get("skill_json", {}) or {})
        status = str(patch.get("status", "") or skill_json.get("status", "") or "active").strip() or "active"
        if status not in {"active", "sprout"}:
            status = "active"
        skill_md = re.sub(r"(?m)^status:\s*\S+\s*$", f"status: {status}", skill_md, count=1)
        if status == "active":
            skill_md = skill_md.replace(
                "This node is sprout and may only be trialed during train.",
                "This node is active in the public tree but should be prioritized only for users whose communication_route_skill explicitly includes it, unless global_default is true.",
            )
        skill_json["node_id"] = full_node_id
        skill_json["node_level"] = layer
        skill_json["status"] = status
        if self.dataset:
            skill_json.setdefault("dataset", self.dataset)
            skill_json.setdefault("source_dataset", self.dataset)
            skill_json.setdefault("dataset_scope", [self.dataset])
        skill_json.setdefault("global_default", False)
        skill_json.setdefault("route_injection_only", True)
        if layer == "who":
            who_meta = self.infer_who_subgroup_metadata(full_node_id, skill_json)
            skill_json.setdefault("who_node_kind", who_meta.get("who_node_kind"))
            skill_json.setdefault("advisor_source", who_meta.get("advisor_source"))
            constraints = dict(who_meta.get("retrieval_constraints", {}) or {})
            constraints.update(dict(skill_json.get("retrieval_constraints", {}) or {}))
            skill_json["retrieval_constraints"] = constraints
        if layer == "what":
            base_node_id = str(full_node_id).split("/", 1)[0] if "/" in str(full_node_id) else str(full_node_id)
            skill_json.setdefault(
                "summary_hints",
                DEFAULT_SUMMARY_HINTS.get(("what", base_node_id), DEFAULT_SUMMARY_HINTS[("what", "none")]),
            )
        if layer == "how":
            skill_json.setdefault(
                "summary_hints",
                {
                    "task_focus": "preserve how-specific advisor interaction signals",
                    "important_output_fields": [],
                    "preserve_interaction_fields": list(skill_json.get("advisor_output_format", []) or []),
                },
            )
        if op == "add_sibling_node" and layer in {'why', "what", "how"}:
            tree_nodes = self.load_tree(force_reload=True).get(layer, {}) or {}
            anchor_source = "llm" if str(skill_json.get("trial_anchor_node", "") or patch.get("trial_anchor_node", "") or "").strip() else "fallback_inferred"
            if patch.get("trial_anchor_node"):
                skill_json["trial_anchor_node"] = patch.get("trial_anchor_node")
            anchor = self.infer_trial_anchor_node(layer, full_node_id, skill_json, tree_nodes=tree_nodes)
            if anchor:
                skill_json["trial_anchor_node"] = anchor
                skill_json["trial_anchor_source"] = str(skill_json.get("trial_anchor_source", "") or patch.get("trial_anchor_source", "") or anchor_source)
        skill_json.setdefault("selection_prior", 0.7 if status == "active" else 0.15)
        if layer in {'why', "what"}:
            skill_json.setdefault(
                "selection_profile",
                {
                    "requires": [],
                    "prefers": [],
                    "do_not_use_why": [],
                    "selection_prior": skill_json.get("selection_prior", 0.15),
                },
            )
        skill_json.setdefault("evolution_state", {"tt": 0, "wt": 0, "tw": 0, "ww": 0})
        (node_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
        (node_dir / "agents" / "openai.yaml").write_text(
            self._node_openai_yaml(full_node_id, skill_json.get("description", patch.get("why_needed", ""))),
            encoding="utf-8",
        )
        dump_json(node_dir / "references" / "skill.json", skill_json)
        dump_json(
            node_dir / "references" / "lifecycle.json",
            {
                "status": status,
                "support": 0,
                "tt": 0,
                "wt": 0,
                "tw": 0,
                "ww": 0,
                "trial_rounds": 0,
                "trial_count": 0,
                "useful_final_t_count": 0,
                "helpful_count": 0,
                "harmful_count": 0,
                "ineffective_count": 0,
                "neutral_count": 0,
                "severe_quality_issue_count": 0,
                "success_patterns": [],
                "failure_patterns": [],
                "repair_hints": [],
                "withered_reason": "",
                "revision_memory": [],
                "parent_node": parent,
                **(
                    {
                        "dataset": str(skill_json.get("dataset", "") or ""),
                        "source_dataset": str(skill_json.get("source_dataset", "") or ""),
                        "dataset_scope": list(skill_json.get("dataset_scope", []) or []),
                    }
                    if skill_json.get("dataset") or skill_json.get("source_dataset") or skill_json.get("dataset_scope")
                    else {}
                ),
                "promotion_history": [
                    {
                        "ts": int(time.time()),
                        "from": "none",
                        "to": status,
                        "reason": str(patch.get("why_needed", "") or ""),
                    }
                ],
            },
        )
        append_jsonl(self.index_dir / "tree_patch_proposals.jsonl", event)
        self._cache = None
        return {"applied": True, "operation": op, "node_id": full_node_id}

    def build_path_skill_payload(self, path):
        tree = self.load_tree()
        payload = {}
        for level in ['why', "what", "who", "how"]:
            node_id = str((path or {}).get(level, "") or "")
            node_id = (DEPRECATED_NODE_ALIASES.get(level, {}) or {}).get(node_id, node_id)
            node = (tree.get(level, {}) or {}).get(node_id, {}) or {}
            if not node:
                continue
            parent_node = "/".join(node_id.split("/")[:-1]) if "/" in node_id else ""
            parent = (tree.get(level, {}) or {}).get(parent_node, {}) or {}
            payload[level] = {
                "node_id": node_id,
                "parent_node": parent_node,
                "status": str(node.get("status", "") or ""),
                "use_why": str(node.get("use_why", "") or node.get("description", "") or ""),
                "if_selected": str(node.get("if_selected", "") or node.get("action", "") or ""),
                "advisor_role": str(node.get("advisor_role", "") or (node_id if level == "who" else "")),
                "task_output_format": list(node.get("task_output_format", []) or []) if level == "what" else [],
                "advisor_output_format": list(node.get("advisor_output_format", []) or []) if level == "how" else [],
                "summary_hints": dict(node.get("summary_hints", {}) or {}) if level in {"what", "how"} else {},
                "selection_prior": node.get("selection_prior", ""),
                "selection_profile": dict(node.get("selection_profile", {}) or {}) if level in {'why', "what"} else {},
                "skill_path": str(node.get("skill_path", "") or ""),
                "skill_body": str(node.get("skill_body", "") or ""),
                "parent_status": str(parent.get("status", "") or "") if parent else "",
                "parent_use_why": str(parent.get("use_why", "") or parent.get("description", "") or "") if parent else "",
                "parent_if_selected": str(parent.get("if_selected", "") or parent.get("action", "") or "") if parent else "",
            }
            if level in {'why', "what", "how"} and "/" not in node_id:
                anchor = self.infer_trial_anchor_node(level, node_id, node, tree_nodes=tree.get(level, {}) or {})
                if anchor:
                    payload[level]["trial_anchor_node"] = anchor
                    payload[level]["trial_anchor_source"] = str(node.get("trial_anchor_source", "") or "fallback_inferred")
            if level == "who":
                payload[level].update(self.infer_who_subgroup_metadata(node_id, node))
        who_branch = str((path or {}).get("who_branch", "") or "")
        if who_branch:
            branch_node = (tree.get("who", {}) or {}).get(who_branch, {}) or {}
            if branch_node:
                branch_meta = self.infer_who_subgroup_metadata(who_branch, branch_node)
                payload["who_branch"] = {
                    "node_id": who_branch,
                    "status": str(branch_node.get("status", "") or ""),
                    "use_why": str(branch_node.get("use_why", "") or branch_node.get("description", "") or ""),
                    "if_selected": str(branch_node.get("if_selected", "") or branch_node.get("action", "") or ""),
                    "advisor_role": str(branch_node.get("advisor_role", "") or ""),
                    "who_node_kind": str(branch_meta.get("who_node_kind", "") or ""),
                    "advisor_source": str(branch_meta.get("advisor_source", "") or ""),
                    "retrieval_constraints": dict(branch_meta.get("retrieval_constraints", {}) or {}),
                }
        return payload
