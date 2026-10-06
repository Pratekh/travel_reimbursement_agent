
import json
import os
from datetime import date, datetime
from typing import Any, Dict, List, Literal, TypedDict

import pandas as pd
import streamlit as st

try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# ============================================================
# APP CONFIG
# ============================================================
st.set_page_config(
    page_title="Travel Reimbursement Approval Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; }
    .top-title { font-size: 2rem; font-weight: 800; margin-bottom: 0.15rem; }
    .top-subtitle { color: #64748b; margin-bottom: 1rem; }
    .metric-card {
        padding: 1rem;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,.05);
    }
    .decision {
        display:inline-block; padding:.35rem .7rem; border-radius:999px;
        font-weight:700; font-size:.85rem;
    }
    .approve { background:#dcfce7; color:#166534; }
    .partial { background:#fef3c7; color:#92400e; }
    .reject { background:#fee2e2; color:#991b1b; }
    .manual { background:#dbeafe; color:#1e40af; }
    .trace-box {
        border-left: 4px solid #64748b; padding: .65rem 1rem;
        margin:.35rem 0; background:#f8fafc; border-radius:0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# POLICY - TRANSCRIBED FROM ASSIGNMENT APPENDIX A
# ============================================================
POLICY = {
    "POL-CAT-01": {
        "title": "Eligible categories",
        "rule": "Airfare economy, lodging, meals subject to per-diem, ground transport, and conference/registration fees are reimbursable when incurred for documented business purpose."
    },
    "POL-CAT-02": {
        "title": "Ineligible items",
        "rule": "Alcohol/minibar, spa/gym/personal entertainment, in-room movies, personal shopping/gifts, traffic fines/penalties/late fees, and personal non-business expenses are never reimbursable."
    },
    "POL-PD-01": {
        "title": "Meals cap",
        "rule": "Meals maximum $75 per day. Amount above daily cap is deducted."
    },
    "POL-PD-02": {
        "title": "Lodging cap",
        "rule": "Lodging maximum $200 per night. Amount above nightly cap is deducted."
    },
    "POL-PD-03": {
        "title": "Ground transport cap",
        "rule": "Ground transport maximum $50 per day. Amount above daily cap is deducted."
    },
    "POL-AIR-01": {
        "title": "Airfare class",
        "rule": "Only economy airfare is reimbursable. Business/first class is a policy exception and must go to Manual Review."
    },
    "POL-RCT-01": {
        "title": "Receipt required",
        "rule": "Any single line item above $25 requires an attached itemized receipt. Airfare and lodging always require a receipt."
    },
    "POL-RCT-02": {
        "title": "Missing receipt",
        "rule": "Missing required receipt routes the claim to Manual Review; it is not silently rejected."
    },
    "POL-APR-01": {
        "title": "Auto-approve tier",
        "rule": "Total reimbursable amount <= $500 may be auto-approved when fully compliant."
    },
    "POL-APR-02": {
        "title": "Manager tier",
        "rule": "Total reimbursable amount > $500 and <= $2,000 is approvable when fully compliant."
    },
    "POL-APR-03": {
        "title": "Director/manual tier",
        "rule": "Total reimbursable amount > $2,000 must be routed to Manual Review."
    },
    "POL-TIME-01": {
        "title": "Submission window",
        "rule": "Claims must be submitted within 30 days of the expense date. Late claims route to Manual Review."
    },
}

# ============================================================
# SAMPLE CLAIMS - APPENDIX B
# ============================================================
SAMPLE_CLAIMS = [
    {
        "claim_id": "CLM-001",
        "employee": "A. Rivera",
        "trip_start": "2026-06-10",
        "trip_end": "2026-06-12",
        "submitted": "2026-06-20",
        "items": [
            {"category": "airfare", "description": "Round-trip economy airfare", "amount": 420.00, "receipt_attached": True},
            {"category": "lodging", "description": "Hotel, 2 nights @ $180", "amount": 360.00, "receipt_attached": True, "nights": 2},
            {"category": "meals", "description": "Meals, 3 days @ ~$60/day", "amount": 180.00, "receipt_attached": True, "days": 3},
            {"category": "conference_fees", "description": "Conference registration", "amount": 150.00, "receipt_attached": True},
        ],
    },
    {
        "claim_id": "CLM-002",
        "employee": "B. Osei",
        "trip_start": "2026-06-14",
        "trip_end": "2026-06-15",
        "submitted": "2026-06-25",
        "items": [
            {"category": "spa", "description": "Hotel spa package", "amount": 300.00, "receipt_attached": True},
            {"category": "minibar", "description": "In-room minibar", "amount": 80.00, "receipt_attached": True},
        ],
    },
    {
        "claim_id": "CLM-003",
        "employee": "C. Nakamura",
        "trip_start": "2026-06-08",
        "trip_end": "2026-06-10",
        "submitted": "2026-06-22",
        "items": [
            {"category": "airfare", "description": "Round-trip economy airfare", "amount": 300.00, "receipt_attached": True},
            {"category": "lodging", "description": "Hotel, 2 nights @ $250", "amount": 500.00, "receipt_attached": True, "nights": 2},
            {"category": "meals", "description": "Meals, 2 days @ $70/day", "amount": 140.00, "receipt_attached": True, "days": 2},
        ],
    },
    {
        "claim_id": "CLM-004",
        "employee": "D. Fischer",
        "trip_start": "2026-06-16",
        "trip_end": "2026-06-18",
        "submitted": "2026-06-28",
        "items": [
            {"category": "airfare", "description": "Business-class international airfare", "amount": 2400.00, "receipt_attached": True},
            {"category": "lodging", "description": "Hotel, 3 nights", "amount": 600.00, "receipt_attached": False, "nights": 3},
        ],
    },
    {
        "claim_id": "CLM-005",
        "employee": "E. Haddad",
        "trip_start": "2026-06-11",
        "trip_end": "2026-06-11",
        "submitted": "2026-06-24",
        "items": [
            {"category": "meals", "description": "Client dinner for 4 (business development)", "amount": 220.00, "receipt_attached": False, "days": 1},
        ],
    },
]


# ============================================================
# AGENT STATE
# ============================================================
class AgentState(TypedDict, total=False):
    claim: Dict[str, Any]
    policy_context: Dict[str, Any]
    receipt_check: Dict[str, Any]
    limit_check: Dict[str, Any]
    eligibility_check: Dict[str, Any]
    threshold_check: Dict[str, Any]
    timeliness_check: Dict[str, Any]
    decision: Dict[str, Any]
    trace: List[Dict[str, Any]]
    tools_used: List[str]
    llm_used: bool


# ============================================================
# TOOL FUNCTIONS
# ============================================================
def tool_policy_lookup(claim: Dict[str, Any]) -> Dict[str, Any]:
    categories = {i["category"] for i in claim["items"]}
    refs = set(["POL-CAT-01", "POL-CAT-02", "POL-RCT-01", "POL-RCT-02",
                "POL-APR-01", "POL-APR-02", "POL-APR-03", "POL-TIME-01"])

    if "meals" in categories:
        refs.add("POL-PD-01")
    if "lodging" in categories:
        refs.add("POL-PD-02")
    if "ground_transport" in categories:
        refs.add("POL-PD-03")
    if "airfare" in categories:
        refs.add("POL-AIR-01")

    return {"refs": sorted(refs), "rules": {r: POLICY[r] for r in sorted(refs)}}


def tool_receipt_completeness(claim: Dict[str, Any]) -> Dict[str, Any]:
    missing = []
    required = []

    for item in claim["items"]:
        amount = float(item["amount"])
        category = item["category"]
        receipt_required = amount > 25 or category in {"airfare", "lodging"}
        if receipt_required:
            required.append(item["description"])
            if not item.get("receipt_attached", False):
                missing.append({
                    "category": category,
                    "description": item["description"],
                    "amount": amount,
                })

    return {
        "required_receipts": required,
        "missing_receipts": missing,
        "complete": len(missing) == 0,
    }


def tool_limit_checker(claim: Dict[str, Any]) -> Dict[str, Any]:
    results = []
    total_claimed = 0.0
    total_approved_after_caps = 0.0
    total_deducted = 0.0

    for item in claim["items"]:
        amount = float(item["amount"])
        category = item["category"]
        total_claimed += amount

        approved = amount
        deduction = 0.0
        cap = None
        basis = "No per-item cap"

        if category == "meals":
            days = max(int(item.get("days", 1)), 1)
            cap = 75.0 * days
            basis = f"$75/day × {days} day(s)"
        elif category == "lodging":
            nights = max(int(item.get("nights", 1)), 1)
            cap = 200.0 * nights
            basis = f"$200/night × {nights} night(s)"
        elif category == "ground_transport":
            days = max(int(item.get("days", 1)), 1)
            cap = 50.0 * days
            basis = f"$50/day × {days} day(s)"

        if cap is not None and amount > cap:
            approved = cap
            deduction = amount - cap

        total_approved_after_caps += approved
        total_deducted += deduction
        results.append({
            "category": category,
            "description": item["description"],
            "claimed": round(amount, 2),
            "approved_after_cap": round(approved, 2),
            "deducted": round(deduction, 2),
            "cap": cap,
            "basis": basis,
        })

    return {
        "items": results,
        "total_claimed": round(total_claimed, 2),
        "total_after_caps": round(total_approved_after_caps, 2),
        "total_cap_deduction": round(total_deducted, 2),
    }


def tool_eligibility_checker(claim: Dict[str, Any]) -> Dict[str, Any]:
    ineligible_categories = {
        "alcohol", "minibar", "spa", "gym", "personal_entertainment",
        "in_room_movies", "personal_shopping", "gifts", "traffic_fines",
        "penalties", "late_fees", "personal_expense"
    }

    exceptions = []
    ineligible = []

    for item in claim["items"]:
        cat = item["category"].lower().strip()
        if cat in ineligible_categories:
            ineligible.append(item)
        if cat == "airfare" and "business" in item["description"].lower():
            exceptions.append({
                "type": "airfare_class_exception",
                "description": item["description"],
                "policy_ref": "POL-AIR-01",
            })

    return {
        "ineligible_items": ineligible,
        "policy_exceptions": exceptions,
        "has_ineligible": bool(ineligible),
        "has_exception": bool(exceptions),
    }


def tool_approval_threshold(total_reimbursable: float) -> Dict[str, Any]:
    total = float(total_reimbursable)
    if total <= 500:
        return {"tier": "AUTO_APPROVE", "manual_required": False, "policy_ref": "POL-APR-01"}
    if total <= 2000:
        return {"tier": "MANAGER", "manual_required": False, "policy_ref": "POL-APR-02"}
    return {"tier": "DIRECTOR_MANUAL_REVIEW", "manual_required": True, "policy_ref": "POL-APR-03"}


def tool_timeliness_checker(claim: Dict[str, Any]) -> Dict[str, Any]:
    # Assignment policy is based on expense date. For a multi-day trip,
    # use the latest expense/trip date as the final expense date.
    expense_date = datetime.strptime(claim["trip_end"], "%Y-%m-%d").date()
    submitted = datetime.strptime(claim["submitted"], "%Y-%m-%d").date()
    days = (submitted - expense_date).days
    return {
        "expense_date": str(expense_date),
        "submitted": str(submitted),
        "days_after_expense": days,
        "within_30_days": days <= 30,
    }


# ============================================================
# OPTIONAL LLM - USED ONLY FOR EXPLANATION / AGENT REASONING
# ============================================================
def get_llm():
    key = os.getenv("OPENAI_API_KEY")
    if not key or not OPENAI_AVAILABLE:
        return None
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, temperature=0)


def llm_reasoning_summary(claim: Dict[str, Any], tool_results: Dict[str, Any]) -> str:
    """
    Optional GenAI layer. The deterministic policy engine remains authoritative.
    This prevents an LLM hallucination from changing money/decision calculations.
    """
    llm = get_llm()
    if llm is None:
        return ""

    prompt = f"""
You are the reasoning layer of a travel reimbursement approval agent.
Use ONLY the supplied policy tool results. Do not invent policy rules.
Summarize why the final policy engine should choose its decision.

CLAIM:
{json.dumps(claim, indent=2)}

TOOL RESULTS:
{json.dumps(tool_results, indent=2)}

Return 2-4 concise sentences. Mention the most important policy references.
"""
    try:
        response = llm.invoke(prompt)
        return response.content if isinstance(response.content, str) else str(response.content)
    except Exception:
        return ""


# ============================================================
# DECISION ENGINE
# ============================================================
def build_decision(state: AgentState) -> Dict[str, Any]:
    claim = state["claim"]
    receipts = state["receipt_check"]
    limits = state["limit_check"]
    eligibility = state["eligibility_check"]
    threshold = state["threshold_check"]
    timeliness = state["timeliness_check"]

    refs = set(state["policy_context"]["refs"])
    missing_docs = [m["description"] for m in receipts["missing_receipts"]]

    # Policy exception / ambiguity always wins.
    if eligibility["has_exception"]:
        decision = "MANUAL_REVIEW"
        approved = 0.0
        deducted = 0.0
        reason = "Business/first-class airfare is a policy exception and requires manual review; no automatic reimbursement amount is finalized."
        refs.add("POL-AIR-01")
        confidence = 0.98
    elif not timeliness["within_30_days"]:
        decision = "MANUAL_REVIEW"
        approved = 0.0
        deducted = 0.0
        reason = "The claim was submitted more than 30 days after the latest trip expense date."
        refs.add("POL-TIME-01")
        confidence = 0.99
    elif missing_docs:
        decision = "MANUAL_REVIEW"
        approved = 0.0
        deducted = 0.0
        reason = "A required receipt is missing. The policy requires manual review rather than silent rejection."
        refs.add("POL-RCT-02")
        confidence = 0.99
    elif eligibility["has_ineligible"] and limits["total_after_caps"] == 0:
        decision = "REJECT"
        approved = 0.0
        deducted = limits["total_claimed"]
        reason = "All claimed items are ineligible under the travel reimbursement policy."
        refs.add("POL-CAT-02")
        confidence = 0.99
    elif eligibility["has_ineligible"]:
        decision = "PARTIAL_APPROVE"
        # Remove ineligible items from approval.
        ineligible_amount = sum(float(i["amount"]) for i in eligibility["ineligible_items"])
        approved = max(limits["total_after_caps"] - ineligible_amount, 0.0)
        deducted = limits["total_claimed"] - approved
        reason = "Eligible items are reimbursed, while ineligible personal/non-reimbursable items are deducted."
        refs.add("POL-CAT-02")
        confidence = 0.98
    elif threshold["manual_required"]:
        decision = "MANUAL_REVIEW"
        approved = 0.0
        deducted = 0.0
        reason = "The post-cap reimbursable amount exceeds $2,000 and therefore requires director/manual review."
        refs.add("POL-APR-03")
        confidence = 0.99
    elif limits["total_cap_deduction"] > 0:
        decision = "PARTIAL_APPROVE"
        approved = limits["total_after_caps"]
        deducted = limits["total_cap_deduction"]
        reason = "The claim is otherwise compliant, but one or more category limits were exceeded; excess amounts are deducted."
        confidence = 0.99
    else:
        decision = "APPROVE"
        approved = limits["total_after_caps"]
        deducted = 0.0
        reason = "All items are eligible, required receipts are present, the claim is timely, and the reimbursable amount is within an approvable tier."
        confidence = 0.99

    if eligibility["has_ineligible"]:
        refs.add("POL-CAT-02")
    if limits["total_cap_deduction"] > 0:
        if any(x["category"] == "meals" and x["deducted"] > 0 for x in limits["items"]):
            refs.add("POL-PD-01")
        if any(x["category"] == "lodging" and x["deducted"] > 0 for x in limits["items"]):
            refs.add("POL-PD-02")
        if any(x["category"] == "ground_transport" and x["deducted"] > 0 for x in limits["items"]):
            refs.add("POL-PD-03")

    return {
        "claim_id": claim["claim_id"],
        "decision": decision,
        "approved_amount": round(approved, 2),
        "deducted_amount": round(deducted, 2),
        "missing_docs": missing_docs,
        "policy_refs": sorted(refs),
        "confidence": round(confidence, 2),
        "explanation": reason,
        "tools_used": state.get("tools_used", []),
    }


# ============================================================
# LANGGRAPH
# ============================================================
def make_graph():
    if not LANGGRAPH_AVAILABLE:
        return None

    def intake_node(state: AgentState):
        return {
            "trace": [{
                "node": "intake",
                "status": "completed",
                "detail": f"Accepted claim {state['claim']['claim_id']}"
            }]
        }

    def policy_node(state: AgentState):
        result = tool_policy_lookup(state["claim"])
        return {
            "policy_context": result,
            "tools_used": state.get("tools_used", []) + ["policy_lookup"],
            "trace": state.get("trace", []) + [{
                "node": "policy_lookup",
                "status": "completed",
                "detail": f"Retrieved {len(result['refs'])} policy references"
            }]
        }

    def receipt_node(state: AgentState):
        result = tool_receipt_completeness(state["claim"])
        return {
            "receipt_check": result,
            "tools_used": state.get("tools_used", []) + ["receipt_completeness_check"],
            "trace": state.get("trace", []) + [{
                "node": "receipt_check",
                "status": "completed",
                "detail": f"{len(result['missing_receipts'])} required receipt(s) missing"
            }]
        }

    def limit_node(state: AgentState):
        result = tool_limit_checker(state["claim"])
        return {
            "limit_check": result,
            "tools_used": state.get("tools_used", []) + ["per_diem_limit_checker"],
            "trace": state.get("trace", []) + [{
                "node": "limit_check",
                "status": "completed",
                "detail": f"Post-cap amount ${result['total_after_caps']:.2f}; cap deduction ${result['total_cap_deduction']:.2f}"
            }]
        }

    def eligibility_node(state: AgentState):
        result = tool_eligibility_checker(state["claim"])
        return {
            "eligibility_check": result,
            "tools_used": state.get("tools_used", []) + ["eligibility_checker"],
            "trace": state.get("trace", []) + [{
                "node": "eligibility_check",
                "status": "completed",
                "detail": f"Ineligible={result['has_ineligible']}, exception={result['has_exception']}"
            }]
        }

    def threshold_node(state: AgentState):
        result = tool_approval_threshold(state["limit_check"]["total_after_caps"])
        return {
            "threshold_check": result,
            "timeliness_check": tool_timeliness_checker(state["claim"]),
            "tools_used": state.get("tools_used", []) + ["approval_threshold_check", "timeliness_checker"],
            "trace": state.get("trace", []) + [{
                "node": "threshold_check",
                "status": "completed",
                "detail": f"Tier={result['tier']}; manual_required={result['manual_required']}"
            }]
        }

    def decision_node(state: AgentState):
        decision = build_decision(state)

        tool_results = {
            "policy": state["policy_context"],
            "receipts": state["receipt_check"],
            "limits": state["limit_check"],
            "eligibility": state["eligibility_check"],
            "threshold": state["threshold_check"],
            "timeliness": state["timeliness_check"],
            "decision": decision,
        }
        llm_summary = llm_reasoning_summary(state["claim"], tool_results)
        if llm_summary:
            decision["explanation"] = f"{decision['explanation']} GenAI reasoning: {llm_summary}"
            llm_used = True
        else:
            llm_used = False

        return {
            "decision": decision,
            "llm_used": llm_used,
            "trace": state.get("trace", []) + [{
                "node": "agent_decision",
                "status": "completed",
                "detail": f"{decision['decision']} with confidence {decision['confidence']}"
            }]
        }

    def validate_node(state: AgentState):
        result = validate_result(state["decision"])
        return {
            "decision": result,
            "trace": state.get("trace", []) + [{
                "node": "output_validator",
                "status": "completed",
                "detail": "Strict output schema validation passed"
            }]
        }

    workflow = StateGraph(AgentState)
    workflow.add_node("intake", intake_node)
    workflow.add_node("policy_lookup", policy_node)
    workflow.add_node("receipt_check", receipt_node)
    workflow.add_node("limit_check", limit_node)
    workflow.add_node("eligibility_check", eligibility_node)
    workflow.add_node("threshold_check", threshold_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("validate", validate_node)

    workflow.add_edge(START, "intake")
    workflow.add_edge("intake", "policy_lookup")
    workflow.add_edge("policy_lookup", "receipt_check")
    workflow.add_edge("receipt_check", "limit_check")
    workflow.add_edge("limit_check", "eligibility_check")
    workflow.add_edge("eligibility_check", "threshold_check")
    workflow.add_edge("threshold_check", "decision")
    workflow.add_edge("decision", "validate")
    workflow.add_edge("validate", END)

    return workflow.compile()


GRAPH = make_graph()


# ============================================================
# VALIDATION + EXECUTION
# ============================================================
REQUIRED_FIELDS = [
    "claim_id", "decision", "approved_amount", "deducted_amount",
    "missing_docs", "policy_refs", "confidence", "explanation", "tools_used"
]
ALLOWED_DECISIONS = {"APPROVE", "PARTIAL_APPROVE", "REJECT", "MANUAL_REVIEW"}


def validate_result(result: Dict[str, Any]) -> Dict[str, Any]:
    missing = [k for k in REQUIRED_FIELDS if k not in result]
    if missing:
        raise ValueError(f"Missing output fields: {missing}")

    if result["decision"] not in ALLOWED_DECISIONS:
        raise ValueError(f"Invalid decision: {result['decision']}")

    if not isinstance(result["approved_amount"], (int, float)):
        raise ValueError("approved_amount must be numeric")
    if not isinstance(result["deducted_amount"], (int, float)):
        raise ValueError("deducted_amount must be numeric")
    if not isinstance(result["missing_docs"], list):
        raise ValueError("missing_docs must be a list")
    if not isinstance(result["policy_refs"], list):
        raise ValueError("policy_refs must be a list")
    if not isinstance(result["tools_used"], list):
        raise ValueError("tools_used must be a list")
    if not (0 <= float(result["confidence"]) <= 1):
        raise ValueError("confidence must be between 0 and 1")

    return {k: result[k] for k in REQUIRED_FIELDS}


def run_claim(claim: Dict[str, Any]) -> Dict[str, Any]:
    if GRAPH is None:
        # Deterministic fallback keeps the app runnable if LangGraph is not installed.
        policy = tool_policy_lookup(claim)
        receipts = tool_receipt_completeness(claim)
        limits = tool_limit_checker(claim)
        eligibility = tool_eligibility_checker(claim)
        threshold = tool_approval_threshold(limits["total_after_caps"])
        timeliness = tool_timeliness_checker(claim)
        state = {
            "claim": claim,
            "policy_context": policy,
            "receipt_check": receipts,
            "limit_check": limits,
            "eligibility_check": eligibility,
            "threshold_check": threshold,
            "timeliness_check": timeliness,
            "tools_used": [
                "policy_lookup", "receipt_completeness_check",
                "per_diem_limit_checker", "eligibility_checker",
                "approval_threshold_check", "timeliness_checker"
            ]
        }
        result = build_decision(state)
        result = validate_result(result)
        return {
            "result": result,
            "trace": [{"node": "fallback_engine", "status": "completed",
                       "detail": "LangGraph unavailable; deterministic tool workflow used"}],
            "raw_state": state,
            "llm_used": False,
        }

    initial = {
        "claim": claim,
        "trace": [],
        "tools_used": [],
        "llm_used": False,
    }
    final_state = GRAPH.invoke(initial)
    return {
        "result": final_state["decision"],
        "trace": final_state.get("trace", []),
        "raw_state": final_state,
        "llm_used": final_state.get("llm_used", False),
    }


def run_all_claims(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [run_claim(c)["result"] for c in claims]


# ============================================================
# INPUT NORMALIZATION
# ============================================================
def normalize_claims(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        if "claims" in payload:
            payload = payload["claims"]
        else:
            payload = [payload]

    if not isinstance(payload, list):
        raise ValueError("Input must be a claim object, a list of claims, or {'claims': [...]}.")

    required = {"claim_id", "trip_start", "trip_end", "submitted", "items"}
    for c in payload:
        missing = required - set(c.keys())
        if missing:
            raise ValueError(f"{c.get('claim_id', 'UNKNOWN')}: missing {sorted(missing)}")
        if not isinstance(c["items"], list):
            raise ValueError(f"{c['claim_id']}: items must be a list")
    return payload


def claims_to_dataframe(claims: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for claim in claims:
        for item in claim["items"]:
            row = {
                "claim_id": claim["claim_id"],
                "employee": claim.get("employee", ""),
                "trip_start": claim["trip_start"],
                "trip_end": claim["trip_end"],
                "submitted": claim["submitted"],
                **item,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def dataframe_to_claims(df: pd.DataFrame) -> List[Dict[str, Any]]:
    required = {"claim_id", "trip_start", "trip_end", "submitted", "category", "description", "amount", "receipt_attached"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    claims = {}
    for _, row in df.iterrows():
        cid = str(row["claim_id"])
        if cid not in claims:
            claims[cid] = {
                "claim_id": cid,
                "employee": str(row.get("employee", "")),
                "trip_start": str(row["trip_start"]),
                "trip_end": str(row["trip_end"]),
                "submitted": str(row["submitted"]),
                "items": [],
            }

        item = {
            "category": str(row["category"]),
            "description": str(row["description"]),
            "amount": float(row["amount"]),
            "receipt_attached": bool(row["receipt_attached"]),
        }

        if "nights" in df.columns and pd.notna(row.get("nights")):
            item["nights"] = int(row["nights"])
        if "days" in df.columns and pd.notna(row.get("days")):
            item["days"] = int(row["days"])

        claims[cid]["items"].append(item)

    return list(claims.values())


# ============================================================
# UI HELPERS
# ============================================================
def decision_badge(decision: str):
    cls = {
        "APPROVE": "approve",
        "PARTIAL_APPROVE": "partial",
        "REJECT": "reject",
        "MANUAL_REVIEW": "manual",
    }[decision]
    st.markdown(f'<span class="decision {cls}">{decision}</span>', unsafe_allow_html=True)


def show_result_table(results: List[Dict[str, Any]]):
    df = pd.DataFrame(results)
    if df.empty:
        st.info("No results yet.")
        return

    display = df[[
        "claim_id", "decision", "approved_amount",
        "deducted_amount", "confidence"
    ]].copy()
    display["approved_amount"] = display["approved_amount"].map(lambda x: f"${x:,.2f}")
    display["deducted_amount"] = display["deducted_amount"].map(lambda x: f"${x:,.2f}")
    st.dataframe(display, use_container_width=True, hide_index=True)


# ============================================================
# SESSION STATE
# ============================================================
if "claims" not in st.session_state:
    st.session_state.claims = SAMPLE_CLAIMS
if "results" not in st.session_state:
    st.session_state.results = []
if "traces" not in st.session_state:
    st.session_state.traces = {}
if "selected_claim" not in st.session_state:
    st.session_state.selected_claim = None


# ============================================================
# HEADER + MENU BAR
# ============================================================
st.markdown('<div class="top-title">✈️ Travel Reimbursement Approval Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="top-subtitle">Agentic AI prototype • LangGraph workflow • Policy-grounded decisions • Audit-ready output</div>',
    unsafe_allow_html=True
)

menu = st.radio(
    "Navigation",
    ["Dashboard", "Evaluate Claim", "Claims & Results", "Agent Trace", "Design Notes"],
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()


# ============================================================
# DASHBOARD
# ============================================================
if menu == "Dashboard":
    if not st.session_state.results:
        st.session_state.results = run_all_claims(st.session_state.claims)

    results = st.session_state.results
    df = pd.DataFrame(results)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Claims Evaluated", len(results))
    with c2:
        st.metric("Approved", int((df["decision"] == "APPROVE").sum()))
    with c3:
        st.metric("Partial Approvals", int((df["decision"] == "PARTIAL_APPROVE").sum()))
    with c4:
        st.metric("Manual / Reject", int(df["decision"].isin(["MANUAL_REVIEW", "REJECT"]).sum()))

    st.subheader("Decision Breakdown")
    counts = df["decision"].value_counts()
    st.bar_chart(counts)

    a, b = st.columns(2)
    with a:
        st.subheader("Financial Summary")
        st.metric("Total Claimed", f"${sum(float(c.get('amount', 0)) for cl in st.session_state.claims for c in cl['items']):,.2f}")
        st.metric("Total Approved", f"${df['approved_amount'].sum():,.2f}")
        st.metric("Total Deducted", f"${df['deducted_amount'].sum():,.2f}")

    with b:
        st.subheader("Agent Status")
        st.success("LangGraph workflow active" if LANGGRAPH_AVAILABLE else "Deterministic fallback active")
        st.write("LLM explanation layer:", "Available" if get_llm() else "Optional / not configured")
        st.write("Policy source:", "Assignment Appendix A")
        st.write("Claims source:", "Assignment Appendix B")

    st.subheader("Results")
    show_result_table(results)

    st.download_button(
        "Download JSON results",
        data=json.dumps(results, indent=2),
        file_name="travel_reimbursement_results.json",
        mime="application/json",
    )


# ============================================================
# EVALUATE CLAIM
# ============================================================
elif menu == "Evaluate Claim":
    st.subheader("Evaluate a Claim")

    input_mode = st.radio(
        "Claim input",
        ["Sample claims", "JSON", "CSV"],
        horizontal=True,
    )

    claims = st.session_state.claims

    if input_mode == "Sample claims":
        options = {c["claim_id"]: c for c in SAMPLE_CLAIMS}
        cid = st.selectbox("Select sample claim", list(options.keys()))
        claim = options[cid]
        st.json(claim)

        if st.button("▶ Run Agent", type="primary", use_container_width=True):
            with st.spinner("Running LangGraph agent..."):
                run = run_claim(claim)
            st.session_state.results = [
                r for r in st.session_state.results if r["claim_id"] != cid
            ] + [run["result"]]
            st.session_state.traces[cid] = run
            st.session_state.selected_claim = cid
            st.success("Evaluation completed.")

    elif input_mode == "JSON":
        default_json = json.dumps(SAMPLE_CLAIMS[0], indent=2)
        raw = st.text_area("Paste claim JSON", value=default_json, height=380)
        if st.button("▶ Evaluate JSON", type="primary"):
            try:
                parsed = normalize_claims(json.loads(raw))
                all_runs = []
                with st.spinner("Running agent..."):
                    for c in parsed:
                        run = run_claim(c)
                        all_runs.append(run)
                        st.session_state.traces[c["claim_id"]] = run
                st.session_state.results = [r["result"] for r in all_runs]
                st.success(f"Evaluated {len(all_runs)} claim(s).")
            except Exception as e:
                st.error(str(e))

    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            try:
                df_upload = pd.read_csv(uploaded)
                st.dataframe(df_upload.head(20), use_container_width=True)
                if st.button("▶ Evaluate CSV", type="primary"):
                    parsed = dataframe_to_claims(df_upload)
                    all_runs = []
                    with st.spinner("Running agent..."):
                        for c in parsed:
                            run = run_claim(c)
                            all_runs.append(run)
                            st.session_state.traces[c["claim_id"]] = run
                    st.session_state.results = [r["result"] for r in all_runs]
                    st.success(f"Evaluated {len(all_runs)} claim(s).")
            except Exception as e:
                st.error(str(e))

    if st.session_state.selected_claim and st.session_state.selected_claim in st.session_state.traces:
        run = st.session_state.traces[st.session_state.selected_claim]
        result = run["result"]
        st.divider()
        st.subheader(f"Agent Result — {result['claim_id']}")
        decision_badge(result["decision"])

        m1, m2, m3 = st.columns(3)
        m1.metric("Approved", f"${result['approved_amount']:,.2f}")
        m2.metric("Deducted", f"${result['deducted_amount']:,.2f}")
        m3.metric("Confidence", f"{result['confidence']:.0%}")

        st.write("**Explanation:**", result["explanation"])
        st.write("**Missing documents:**", result["missing_docs"] or "None")
        st.write("**Policy references:**", ", ".join(result["policy_refs"]))
        st.write("**Tools used:**", ", ".join(result["tools_used"]))


# ============================================================
# CLAIMS & RESULTS
# ============================================================
elif menu == "Claims & Results":
    st.subheader("Claims & Results")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("▶ Evaluate all 5 assignment claims", type="primary"):
            with st.spinner("Running all claims through the LangGraph workflow..."):
                runs = []
                for claim in SAMPLE_CLAIMS:
                    run = run_claim(claim)
                    runs.append(run)
                    st.session_state.traces[claim["claim_id"]] = run
                st.session_state.claims = SAMPLE_CLAIMS
                st.session_state.results = [x["result"] for x in runs]
            st.success("All 5 claims evaluated.")

    with col2:
        if st.button("↺ Reset to assignment samples"):
            st.session_state.claims = SAMPLE_CLAIMS
            st.session_state.results = run_all_claims(SAMPLE_CLAIMS)
            st.session_state.traces = {}
            st.success("Reset complete.")

    if st.session_state.results:
        show_result_table(st.session_state.results)

        st.subheader("Detailed Results")
        for result in st.session_state.results:
            with st.expander(f"{result['claim_id']} — {result['decision']}"):
                st.json(result)
    else:
        st.info("Click 'Evaluate all 5 assignment claims' to generate results.")


# ============================================================
# AGENT TRACE
# ============================================================
elif menu == "Agent Trace":
    st.subheader("Agent Trace / Audit Trail")

    if not st.session_state.traces:
        st.info("Evaluate a claim first. The trace will appear here.")
    else:
        cid = st.selectbox("Claim", list(st.session_state.traces.keys()))
        run = st.session_state.traces[cid]

        st.write("**Architecture:** Intake → Policy Lookup → Receipt Check → Limit Check → Eligibility Check → Threshold Check → Agent Decision → Output Validator")

        for i, step in enumerate(run["trace"], 1):
            st.markdown(
                f"""
                <div class="trace-box">
                    <b>{i}. {step['node']}</b><br>
                    Status: {step['status']}<br>
                    {step['detail']}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("**Tools used:**", ", ".join(run["result"]["tools_used"]))
        st.write("**LLM explanation layer used:**", "Yes" if run.get("llm_used") else "No — deterministic policy reasoning used")


# ============================================================
# DESIGN NOTES
# ============================================================
else:
    st.subheader("Design Notes & Reasoning")

    st.markdown("""
### Architecture

The application uses a **LangGraph state machine** to orchestrate an agentic reimbursement workflow:

1. **Intake** — accepts a claim from sample data, JSON, or CSV.
2. **Policy Lookup Tool** — retrieves only the relevant `POL-*` rules.
3. **Receipt Completeness Tool** — checks whether required receipts exist.
4. **Per-Diem / Limit Tool** — calculates capped reimbursable amounts and deductions.
5. **Eligibility Tool** — detects ineligible categories and policy exceptions.
6. **Approval Threshold Tool** — determines auto-approve, manager, or director/manual tier.
7. **Timeliness Tool** — checks the 30-day submission rule.
8. **Agent Decision** — combines tool outputs using policy precedence.
9. **Output Validator** — guarantees the exact required output fields and allowed decision values.

### Why deterministic policy logic is authoritative

The LLM is optional and is used only for a concise reasoning summary. Monetary calculations,
policy IDs, receipt handling, and the final decision are controlled by explicit policy tools.
This reduces hallucination risk and makes the prototype reproducible.

### Manual Review strategy

Manual Review has priority whenever:
- a required receipt is missing;
- airfare is business/first class;
- the post-cap reimbursable amount exceeds $2,000;
- the claim is late;
- the case contains a policy ambiguity or exception.

The assignment explicitly says to prefer Manual Review over forcing a decision.

### Assignment sample expectations

- **CLM-001:** APPROVE — $1,110.00.
- **CLM-002:** REJECT — $0.00 approved because spa and minibar are ineligible.
- **CLM-003:** PARTIAL_APPROVE — $840.00 approved and $100.00 deducted for lodging above the $200/night cap.
- **CLM-004:** MANUAL_REVIEW — business-class airfare is a policy exception, lodging receipt is missing, and the claim exceeds $2,000.
- **CLM-005:** MANUAL_REVIEW — the meal amount requires a receipt and the receipt is missing.

### Trade-offs

This is intentionally a lightweight prototype, matching the assignment's 2–3 day scope.
For production, the policy should be stored in a governed policy service/vector store, tool calls
should have observability, claims should have immutable audit IDs, and approval actions should
be protected by RBAC and human authorization.
""")


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("Travel Reimbursement Approval Agent • Assignment prototype • Policy-grounded LangGraph workflow")
