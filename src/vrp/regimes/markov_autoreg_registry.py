"""
Registry and configuration utilities for Phase 7 Markov autoregression.

This module intentionally contains no statsmodels fitting code. It only handles:
- config loading
- model spec expansion
- target-column resolution
- path resolution
- forbidden-input policy helpers
- output suffix construction

The actual model implementation belongs in:
    src/vrp/regimes/markov_autoreg.py

The dedicated runner belongs in:
    scripts/train_markov_autoreg.py
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


SUPPORTED_MARKETS = {"US", "INDIA"}


@dataclass(frozen=True)
class MARModelSpec:
    """A single Markov-autoregression candidate specification."""

    target: str
    order: int
    n_states: int
    switching_ar: bool
    switching_trend: bool
    switching_variance: bool
    primary: bool = False

    def suffix(self) -> str:
        """
        Stable suffix used for model-specific output files.

        Example:
            target=vrp_har, order=1, k=2, switching_variance=True
            -> vrp_har_order1_k2_sv
        """
        variance_suffix = "sv" if self.switching_variance else "constvar"
        ar_suffix = "sar" if self.switching_ar else "constar"
        trend_suffix = "strend" if self.switching_trend else "consttrend"

        # Keep primary approved filename compact when using standard spec.
        if self.switching_ar and self.switching_trend:
            return f"{self.target}_order{self.order}_k{self.n_states}_{variance_suffix}"

        return (
            f"{self.target}_order{self.order}_k{self.n_states}_"
            f"{ar_suffix}_{trend_suffix}_{variance_suffix}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MARTrainTestSplitConfig:
    method: str
    train_fraction: float
    min_train_observations: int
    min_test_observations: int


@dataclass(frozen=True)
class MARFitConfig:
    maxiter: int
    em_iter: int
    search_reps: int
    cov_type: str
    disp: bool


@dataclass(frozen=True)
class MARValidationConfig:
    min_train_state_occupancy: float
    min_test_state_occupancy: float
    near_absorbing_transition_threshold: float
    probability_row_sum_tolerance: float
    min_available_fraction_after_warmup: float
    ar_explosive_abs_phi_threshold: float


@dataclass(frozen=True)
class MARInputPolicy:
    allowed_target_candidates: tuple[str, ...] = field(default_factory=tuple)
    forbidden_exact_columns: tuple[str, ...] = field(default_factory=tuple)
    forbidden_prefixes: tuple[str, ...] = field(default_factory=tuple)
    forbidden_substrings: tuple[str, ...] = field(default_factory=tuple)
    diagnostic_only_allowed_after_assignment: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MARPaths:
    processed_input: Mapping[str, str]
    threshold_regimes: Mapping[str, str]
    hmm_regimes: Mapping[str, str]

    model_specific_output_template: str
    primary_alias_output_template: str

    model_specific_model_template: str
    primary_alias_model_template: str

    report_dir_template: str
    figure_dir_template: str

    metadata_template: str
    candidate_ranking_template: str
    no_lookahead_audit_template: str
    ar_stability_template: str
    state_summary_template: str
    transition_matrix_template: str
    duration_summary_template: str
    state_by_year_template: str
    probability_audit_template: str
    hmm_agreement_template: str
    threshold_agreement_template: str

    global_regime_model_comparison: str


@dataclass(frozen=True)
class MARConfig:
    model_name: str
    implementation: str
    output_prefix: str
    random_seed: int

    primary_model: MARModelSpec
    target_columns: Mapping[str, str]
    target_availability_rules: Mapping[str, Mapping[str, Any]]
    target_transform: Mapping[str, Mapping[str, Any]]

    train_test_split: MARTrainTestSplitConfig
    fit: MARFitConfig
    validation: MARValidationConfig
    input_policy: MARInputPolicy
    paths: MARPaths

    raw: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["raw"] = dict(self.raw)
        return data


def load_markov_autoreg_config(
    path: str | Path = "configs/model_markov_autoreg.yaml",
) -> MARConfig:
    """Load and validate the Phase 7 Markov-autoregression YAML config."""
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Markov-autoreg config not found: {config_path}. "
            "Expected configs/model_markov_autoreg.yaml"
        )

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config must parse to a dictionary: {config_path}")

    primary_model = _parse_model_spec(raw.get("primary_model", {}), primary=True)

    split_raw = _required_mapping(raw, "train_test_split")
    fit_raw = _required_mapping(raw, "fit")
    validation_raw = _required_mapping(raw, "validation")
    input_policy_raw = _required_mapping(raw, "input_policy")
    paths_raw = _required_mapping(raw, "paths")

    cfg = MARConfig(
        model_name=str(_required(raw, "model_name")),
        implementation=str(_required(raw, "implementation")),
        output_prefix=str(raw.get("output_prefix", "mar")),
        random_seed=int(raw.get("random_seed", 42)),
        primary_model=primary_model,
        target_columns=dict(_required_mapping(raw, "target_columns")),
        target_availability_rules=dict(raw.get("target_availability_rules", {})),
        target_transform=dict(raw.get("target_transform", {})),
        train_test_split=MARTrainTestSplitConfig(
            method=str(_required(split_raw, "method")),
            train_fraction=float(_required(split_raw, "train_fraction")),
            min_train_observations=int(_required(split_raw, "min_train_observations")),
            min_test_observations=int(_required(split_raw, "min_test_observations")),
        ),
        fit=MARFitConfig(
            maxiter=int(_required(fit_raw, "maxiter")),
            em_iter=int(_required(fit_raw, "em_iter")),
            search_reps=int(_required(fit_raw, "search_reps")),
            cov_type=str(_required(fit_raw, "cov_type")),
            disp=bool(_required(fit_raw, "disp")),
        ),
        validation=MARValidationConfig(
            min_train_state_occupancy=float(_required(validation_raw, "min_train_state_occupancy")),
            min_test_state_occupancy=float(_required(validation_raw, "min_test_state_occupancy")),
            near_absorbing_transition_threshold=float(
                _required(validation_raw, "near_absorbing_transition_threshold")
            ),
            probability_row_sum_tolerance=float(
                _required(validation_raw, "probability_row_sum_tolerance")
            ),
            min_available_fraction_after_warmup=float(
                _required(validation_raw, "min_available_fraction_after_warmup")
            ),
            ar_explosive_abs_phi_threshold=float(
                _required(validation_raw, "ar_explosive_abs_phi_threshold")
            ),
        ),
        input_policy=MARInputPolicy(
            allowed_target_candidates=tuple(input_policy_raw.get("allowed_target_candidates", [])),
            forbidden_exact_columns=tuple(input_policy_raw.get("forbidden_exact_columns", [])),
            forbidden_prefixes=tuple(input_policy_raw.get("forbidden_prefixes", [])),
            forbidden_substrings=tuple(input_policy_raw.get("forbidden_substrings", [])),
            diagnostic_only_allowed_after_assignment=tuple(
                input_policy_raw.get("diagnostic_only_allowed_after_assignment", [])
            ),
        ),
        paths=MARPaths(
            processed_input=dict(_required_mapping(paths_raw, "processed_input")),
            threshold_regimes=dict(_required_mapping(paths_raw, "threshold_regimes")),
            hmm_regimes=dict(_required_mapping(paths_raw, "hmm_regimes")),
            model_specific_output_template=str(_required(paths_raw, "model_specific_output_template")),
            primary_alias_output_template=str(_required(paths_raw, "primary_alias_output_template")),
            model_specific_model_template=str(_required(paths_raw, "model_specific_model_template")),
            primary_alias_model_template=str(_required(paths_raw, "primary_alias_model_template")),
            report_dir_template=str(_required(paths_raw, "report_dir_template")),
            figure_dir_template=str(_required(paths_raw, "figure_dir_template")),
            metadata_template=str(_required(paths_raw, "metadata_template")),
            candidate_ranking_template=str(_required(paths_raw, "candidate_ranking_template")),
            no_lookahead_audit_template=str(_required(paths_raw, "no_lookahead_audit_template")),
            ar_stability_template=str(_required(paths_raw, "ar_stability_template")),
            state_summary_template=str(_required(paths_raw, "state_summary_template")),
            transition_matrix_template=str(_required(paths_raw, "transition_matrix_template")),
            duration_summary_template=str(_required(paths_raw, "duration_summary_template")),
            state_by_year_template=str(_required(paths_raw, "state_by_year_template")),
            probability_audit_template=str(_required(paths_raw, "probability_audit_template")),
            hmm_agreement_template=str(_required(paths_raw, "hmm_agreement_template")),
            threshold_agreement_template=str(_required(paths_raw, "threshold_agreement_template")),
            global_regime_model_comparison=str(_required(paths_raw, "global_regime_model_comparison")),
        ),
        raw=raw,
    )

    validate_model_spec(cfg.primary_model, cfg)
    return cfg


def expand_candidate_specs(cfg: MARConfig) -> list[MARModelSpec]:
    """
    Expand candidate grid from config.

    The primary model is included first. Duplicate specs are removed while
    preserving order.
    """
    grid_raw = _required_mapping(cfg.raw, "candidate_models")

    specs: list[MARModelSpec] = [cfg.primary_model]

    targets = list(_required(grid_raw, "targets"))
    orders = list(_required(grid_raw, "orders"))
    n_states_values = list(_required(grid_raw, "n_states"))
    switching_ar_values = list(_required(grid_raw, "switching_ar"))
    switching_trend_values = list(_required(grid_raw, "switching_trend"))
    switching_variance_values = list(_required(grid_raw, "switching_variance"))

    for target, order, n_states, switching_ar, switching_trend, switching_variance in product(
        targets,
        orders,
        n_states_values,
        switching_ar_values,
        switching_trend_values,
        switching_variance_values,
    ):
        spec = MARModelSpec(
            target=str(target),
            order=int(order),
            n_states=int(n_states),
            switching_ar=bool(switching_ar),
            switching_trend=bool(switching_trend),
            switching_variance=bool(switching_variance),
            primary=False,
        )
        validate_model_spec(spec, cfg)
        specs.append(spec)

    return _dedupe_specs(specs)


def validate_model_spec(spec: MARModelSpec, cfg: MARConfig) -> None:
    """Validate one candidate specification against Phase 7 constraints."""
    if spec.target not in cfg.target_columns:
        raise ValueError(
            f"Unknown MAR target '{spec.target}'. "
            f"Known targets: {sorted(cfg.target_columns.keys())}"
        )

    if spec.order < 1:
        raise ValueError(f"MAR order must be >= 1, got {spec.order}")

    if spec.n_states not in {2, 3}:
        raise ValueError(
            f"Phase 7 only supports n_states in {{2, 3}}, got {spec.n_states}"
        )

    if not spec.switching_ar:
        raise ValueError("Phase 7 approved grid requires switching_ar=True")

    if not spec.switching_trend:
        raise ValueError("Phase 7 approved grid requires switching_trend=True")

    if spec.order != 1:
        raise ValueError("Phase 7 approved grid allows order=1 only")

    target_col = resolve_target_column(spec.target, cfg)
    allowed_cols = set(cfg.input_policy.allowed_target_candidates)
    if target_col not in allowed_cols:
        raise ValueError(
            f"Target column '{target_col}' for target '{spec.target}' is not in "
            f"allowed target candidates: {sorted(allowed_cols)}"
        )


def resolve_target_column(target: str, cfg: MARConfig) -> str:
    """Map target alias to concrete DataFrame column."""
    try:
        return str(cfg.target_columns[target])
    except KeyError as exc:
        raise KeyError(
            f"Unknown target alias '{target}'. "
            f"Known targets: {sorted(cfg.target_columns.keys())}"
        ) from exc


def target_availability_rule(target: str, cfg: MARConfig) -> dict[str, Any]:
    """Return optional target availability rule for a target alias."""
    rule = cfg.target_availability_rules.get(target, {})
    if rule is None:
        return {}
    return dict(rule)


def target_transform_rule(target: str, cfg: MARConfig) -> dict[str, Any]:
    """Return target transform rule for a target alias."""
    rule = cfg.target_transform.get(target, {"method": "none"})
    if rule is None:
        return {"method": "none"}
    return dict(rule)


def normalize_market(market: str) -> str:
    """Normalize and validate market code."""
    market_norm = market.upper()
    if market_norm not in SUPPORTED_MARKETS:
        raise ValueError(f"Unsupported market '{market}'. Expected one of {sorted(SUPPORTED_MARKETS)}")
    return market_norm


def market_lower(market: str) -> str:
    return normalize_market(market).lower()


def processed_input_path(market: str, cfg: MARConfig) -> Path:
    market_norm = normalize_market(market)
    try:
        return Path(cfg.paths.processed_input[market_norm])
    except KeyError as exc:
        raise KeyError(f"No processed input path configured for market {market_norm}") from exc


def threshold_regimes_path(market: str, cfg: MARConfig) -> Path:
    market_norm = normalize_market(market)
    try:
        return Path(cfg.paths.threshold_regimes[market_norm])
    except KeyError as exc:
        raise KeyError(f"No threshold regimes path configured for market {market_norm}") from exc


def hmm_regimes_path(market: str, cfg: MARConfig) -> Path:
    market_norm = normalize_market(market)
    try:
        return Path(cfg.paths.hmm_regimes[market_norm])
    except KeyError as exc:
        raise KeyError(f"No HMM regimes path configured for market {market_norm}") from exc


def model_specific_output_path(market: str, spec: MARModelSpec, cfg: MARConfig) -> Path:
    return _format_path(
        cfg.paths.model_specific_output_template,
        market=market,
        suffix=spec.suffix(),
    )


def primary_alias_output_path(market: str, cfg: MARConfig) -> Path:
    return _format_path(cfg.paths.primary_alias_output_template, market=market)


def model_specific_model_path(market: str, spec: MARModelSpec, cfg: MARConfig) -> Path:
    return _format_path(
        cfg.paths.model_specific_model_template,
        market=market,
        suffix=spec.suffix(),
    )


def primary_alias_model_path(market: str, cfg: MARConfig) -> Path:
    return _format_path(cfg.paths.primary_alias_model_template, market=market)


def report_dir(market: str, cfg: MARConfig) -> Path:
    return _format_path(cfg.paths.report_dir_template, market=market)


def figure_dir(market: str, cfg: MARConfig) -> Path:
    return _format_path(cfg.paths.figure_dir_template, market=market)


def metadata_path(market: str, cfg: MARConfig) -> Path:
    return _format_path(cfg.paths.metadata_template, market=market)


def candidate_ranking_path(market: str, cfg: MARConfig) -> Path:
    return _format_path(cfg.paths.candidate_ranking_template, market=market)


def no_lookahead_audit_path(market: str, cfg: MARConfig) -> Path:
    return _format_path(cfg.paths.no_lookahead_audit_template, market=market)


def ar_stability_path(market: str, cfg: MARConfig) -> Path:
    return _format_path(cfg.paths.ar_stability_template, market=market)


def state_summary_path(market: str, cfg: MARConfig) -> Path:
    return _format_path(cfg.paths.state_summary_template, market=market)


def transition_matrix_path(market: str, cfg: MARConfig) -> Path:
    return _format_path(cfg.paths.transition_matrix_template, market=market)


def duration_summary_path(market: str, cfg: MARConfig) -> Path:
    return _format_path(cfg.paths.duration_summary_template, market=market)


def state_by_year_path(market: str, cfg: MARConfig) -> Path:
    return _format_path(cfg.paths.state_by_year_template, market=market)


def probability_audit_path(market: str, cfg: MARConfig) -> Path:
    return _format_path(cfg.paths.probability_audit_template, market=market)


def hmm_agreement_path(market: str, cfg: MARConfig) -> Path:
    return _format_path(cfg.paths.hmm_agreement_template, market=market)


def threshold_agreement_path(market: str, cfg: MARConfig) -> Path:
    return _format_path(cfg.paths.threshold_agreement_template, market=market)


def global_regime_model_comparison_path(cfg: MARConfig) -> Path:
    return Path(cfg.paths.global_regime_model_comparison)


def ensure_output_directories(market: str, spec: MARModelSpec, cfg: MARConfig) -> None:
    """Create parent directories needed by Phase 7 outputs."""
    paths = [
        model_specific_output_path(market, spec, cfg),
        primary_alias_output_path(market, cfg),
        model_specific_model_path(market, spec, cfg),
        primary_alias_model_path(market, cfg),
        metadata_path(market, cfg),
        candidate_ranking_path(market, cfg),
        no_lookahead_audit_path(market, cfg),
        ar_stability_path(market, cfg),
        state_summary_path(market, cfg),
        transition_matrix_path(market, cfg),
        duration_summary_path(market, cfg),
        state_by_year_path(market, cfg),
        probability_audit_path(market, cfg),
        hmm_agreement_path(market, cfg),
        threshold_agreement_path(market, cfg),
        global_regime_model_comparison_path(cfg),
    ]

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    report_dir(market, cfg).mkdir(parents=True, exist_ok=True)
    figure_dir(market, cfg).mkdir(parents=True, exist_ok=True)


def is_forbidden_input_column(column: str, cfg: MARConfig) -> bool:
    """
    Return True if a column is forbidden as a MAR model input.

    This is intentionally conservative. Diagnostic columns may still be joined
    after MAR state assignment by later diagnostics code.
    """
    col = str(column)
    col_lower = col.lower()

    if col in cfg.input_policy.forbidden_exact_columns:
        return True

    for prefix in cfg.input_policy.forbidden_prefixes:
        if col_lower.startswith(prefix.lower()):
            return True

    for substring in cfg.input_policy.forbidden_substrings:
        if substring.lower() in col_lower:
            return True

    return False


def find_forbidden_input_columns(
    columns: Iterable[str],
    cfg: MARConfig,
    allowed_columns: Iterable[str] | None = None,
) -> list[str]:
    """
    Find forbidden columns among a candidate model-input column set.

    Parameters
    ----------
    columns:
        Columns proposed for model preparation.
    cfg:
        Loaded MAR config.
    allowed_columns:
        Optional allow-list. Columns in this list are ignored even if their
        names contain conservative substrings. Use sparingly.
    """
    allowed = set(allowed_columns or [])
    forbidden: list[str] = []

    for col in columns:
        if col in allowed:
            continue
        if is_forbidden_input_column(col, cfg):
            forbidden.append(col)

    return forbidden


def require_no_forbidden_input_columns(
    columns: Iterable[str],
    cfg: MARConfig,
    allowed_columns: Iterable[str] | None = None,
) -> None:
    """Raise if any forbidden model-input columns are present."""
    forbidden = find_forbidden_input_columns(columns, cfg, allowed_columns=allowed_columns)
    if forbidden:
        raise ValueError(
            "Forbidden Phase 7 MAR input column(s) detected: "
            f"{forbidden}. These columns are diagnostic/label/future data and "
            "must not be used to fit or filter the Markov autoregression model."
        )


def raw_filtered_probability_columns(n_states: int) -> list[str]:
    return [f"mar_filtered_prob_raw_state_{i}" for i in range(int(n_states))]


def raw_smoothed_diagnostic_probability_columns(n_states: int) -> list[str]:
    return [f"mar_diagnostic_smoothed_prob_raw_state_{i}" for i in range(int(n_states))]


def economic_probability_columns() -> list[str]:
    return [
        "mar_filtered_prob_calm",
        "mar_filtered_prob_transition",
        "mar_filtered_prob_stress",
    ]


def next_session_probability_columns() -> list[str]:
    return [
        "mar_filtered_prob_calm_for_next_session",
        "mar_filtered_prob_transition_for_next_session",
        "mar_filtered_prob_stress_for_next_session",
    ]


def expected_signal_columns() -> list[str]:
    return [
        "mar_signal_observation_date",
        "mar_signal_available_after_close_date",
        "mar_signal_trade_date",
        "mar_model_observation_available",
        "mar_state_for_next_session",
        "mar_state_name_for_next_session",
        "mar_filtered_prob_calm_for_next_session",
        "mar_filtered_prob_transition_for_next_session",
        "mar_filtered_prob_stress_for_next_session",
    ]


def _parse_model_spec(raw: Mapping[str, Any], primary: bool) -> MARModelSpec:
    if not isinstance(raw, Mapping):
        raise ValueError("Model spec must be a mapping")

    return MARModelSpec(
        target=str(_required(raw, "target")),
        order=int(_required(raw, "order")),
        n_states=int(_required(raw, "n_states")),
        switching_ar=bool(_required(raw, "switching_ar")),
        switching_trend=bool(_required(raw, "switching_trend")),
        switching_variance=bool(_required(raw, "switching_variance")),
        primary=bool(raw.get("primary", primary)),
    )


def _dedupe_specs(specs: Iterable[MARModelSpec]) -> list[MARModelSpec]:
    seen: set[tuple[Any, ...]] = set()
    out: list[MARModelSpec] = []

    for spec in specs:
        key = (
            spec.target,
            spec.order,
            spec.n_states,
            spec.switching_ar,
            spec.switching_trend,
            spec.switching_variance,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)

    return out


def _format_path(template: str, market: str, suffix: str | None = None) -> Path:
    market_norm = normalize_market(market)
    values = {
        "market": market_norm,
        "market_lower": market_norm.lower(),
        "suffix": suffix or "",
    }
    return Path(template.format(**values))


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required config key: {key}")
    return mapping[key]


def _required_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _required(mapping, key)
    if not isinstance(value, Mapping):
        raise TypeError(f"Config key '{key}' must be a mapping")
    return value