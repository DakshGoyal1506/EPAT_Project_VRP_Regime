"""
Broker-readiness package for Phase 11.

This package is deliberately paper-only. It must not expose live order
placement functions. Phase 11 converts research signals into auditable
paper signals and paper order intents only.
"""

PHASE = "phase_11"
MODE = "paper_signal_only"

APPROVED_STRATEGIES = (
    "unconditional_full",
    "threshold_hard_filter",
    "threshold_defensive",
    "hmm_prob_linear",
    "hmm_prob_linear_carry",
    "mar_prob_linear",
    "mar_prob_linear_carry",
)

FORBIDDEN_ACTIVE_STRATEGY_MODELS = (
    "msvol",
    "msgarch",
)

BROKER_STATUS_TAXONOMY = (
    "IBRIDGEPY_NOT_INSTALLED",
    "IBRIDGEPY_IMPORT_OK",
    "BROKER_CONNECTION_NOT_ATTEMPTED",
    "BROKER_CONNECTION_FAILED",
    "BROKER_DATA_UNAVAILABLE",
    "BROKER_DATA_AVAILABLE",
)

FINAL_STATUS_TAXONOMY = (
    "ALLOWED_PAPER_INTENT",
    "BLOCKED_BY_KILL_SWITCH",
    "BLOCKED_STALE_SIGNAL",
    "BLOCKED_CONFIG_SAFETY",
    "BLOCKED_RISK_LIMIT",
    "BLOCKED_MISSING_SIGNAL",
    "BLOCKED_BROKER_DATA",
    "NO_SIGNAL",
    "STAY_FLAT",
    "BROKER_INSPECTION_ONLY",
)

RESEARCH_PROXY_WARNING = (
    "Phase 10 backtest returns are research-layer proxy units, not executable "
    "option-trading account returns. Phase 11 does not infer starting capital, "
    "margin, option contracts, or live order size from Phase 10."
)

__all__ = [
    "PHASE",
    "MODE",
    "APPROVED_STRATEGIES",
    "FORBIDDEN_ACTIVE_STRATEGY_MODELS",
    "BROKER_STATUS_TAXONOMY",
    "FINAL_STATUS_TAXONOMY",
    "RESEARCH_PROXY_WARNING",
]