#!/usr/bin/env python3
"""
Parameters Module

Contains the Parameters dataclass for model configuration with validation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Union
import logging

from . import config as cfg
from .utils import _coerce_float_scalar, should_reraise
from .taste_distribution import TasteDistribution, get_or_create_taste_distribution
from .automation_model import AutomationModel

logger = logging.getLogger(__name__)


@dataclass
class Parameters:
    """Model parameters with validation"""

    human_only: bool = field(default_factory=lambda: False)

    # Production function parameters
    rho_coding_labor: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['rho_coding_labor'])
    direct_input_exp_cap_ces_params: bool = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['direct_input_exp_cap_ces_params'])
    rho_experiment_capacity: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['rho_experiment_capacity'])
    alpha_experiment_capacity: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['alpha_experiment_capacity'])
    r_software: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['r_software'])
    software_progress_rate_at_reference_year: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['software_progress_rate_at_reference_year'])
    experiment_compute_exponent: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['experiment_compute_exponent'])

    # Automation parameters
    automation_fraction_at_coding_automation_anchor: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['automation_fraction_at_coding_automation_anchor'])
    automation_anchors: Optional[Dict[float, float]] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['automation_anchors'])
    automation_model: Optional[AutomationModel] = field(default_factory=lambda: None)
    automation_interp_type: str = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['automation_interp_type'])
    automation_logistic_asymptote: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['automation_logistic_asymptote'])
    # DERIVED, not primary: the present-day PARALLEL coding-labor multiplier.
    # The primary container for "how much AI uplift there is" is
    # `serial_uplift_at_reference_date` (serial space, at `uplift_reference_date`).
    # DERIVED, never an input: init=False keeps it out of api/index.py's
    # _KNOWN_PARAM_KEYS, so a stale caller stating it is dropped rather than
    # silently overriding the serial container. Materialized in __post_init__.
    swe_multiplier_at_present_day: float = field(default=None, init=False)
    # AI Research Taste sigmoid parameters
    ai_research_taste_at_coding_automation_anchor: float = field(default=None, init=False)  # Computed from _sd
    # Always specify the superhuman-coder taste as SD within the human range
    ai_research_taste_at_coding_automation_anchor_sd: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('ai_research_taste_at_coding_automation_anchor_sd', 0.5))
    ai_research_taste_slope: float = field(default_factory=lambda: cfg.TASTE_SLOPE_DEFAULTS.get(cfg.DEFAULT_TASTE_SCHEDULE_TYPE, cfg.DEFAULT_PARAMETERS['ai_research_taste_slope']))
    taste_schedule_type: str = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['taste_schedule_type'])
    progress_at_aa: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('progress_at_aa'))
    ac_time_horizon_minutes: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['ac_time_horizon_minutes'])
    horizon_extrapolation_type: str = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['horizon_extrapolation_type'])

    # AC anchor mode: how progress_at_aa is determined (see progress_model/ac_anchor.py)
    ac_anchor_mode: str = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('ac_anchor_mode', 'time-horizon'))
    # Revenue AC-anchor mode: annual revenue ($/yr) at revenue_reference_date
    # (None => present_day), its annual growth factor (e.g. 3.0 = triples per
    # year) at that same date, and the AC-defining level
    present_revenue: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('present_revenue', 1e10))
    present_revenue_growth_rate: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('present_revenue_growth_rate', 3.0))
    revenue_required_for_ac: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('revenue_required_for_ac', 1e12))
    # ECI AC-anchor mode: ECI score at eci_reference_date (None => present_day),
    # its instantaneous growth rate (points/year) at that same date, and the
    # AC-defining level. The (value, rate) pair also defines the
    # capability_ecis metric in every mode.
    eci_at_present_day: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('eci_at_present_day', 162.0))
    eci_growth_rate: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('eci_growth_rate', 15.0))
    eci_required_for_ac: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('eci_required_for_ac', 250.0))
    # PER-OBSERVABLE REFERENCE DATES (elicitation vintages). Each observable's
    # value (and its companion per-year rate) binds at its own date on the
    # anchoring mapping, following the uplift_reference_date convention:
    # None => present_day, which is bit-identical to the historical behavior.
    # These carry WHEN an observable was measured; present_day remains the
    # simulation's "now" (anchor stocks, initial conditions, default date).
    horizon_reference_date: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('horizon_reference_date'))
    revenue_reference_date: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('revenue_reference_date'))
    eci_reference_date: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('eci_reference_date'))
    # 'uplift-solved' AC-anchor mode: the model's OWN endogenous serial coding
    # uplift M* at uplift_reference_date (None => present_day), and the doubling
    # time (anchor-months) of the excess (M - 1) there. The logistic schedule's
    # (k, x0) are solved to satisfy both, making progress_at_aa an output.
    # The current AI coding uplift (serial space, at `uplift_reference_date`). An
    # ordinary defaulted parameter: callers may pass None, which __post_init__
    # normalizes to the default belief, so downstream code never has to ask whether
    # a value was stated or defaulted.
    serial_uplift_at_reference_date: float = field(
        default_factory=lambda: cfg.DEFAULT_PARAMETERS['serial_uplift_at_reference_date'])
    excess_uplift_doubling_time: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('excess_uplift_doubling_time', 12.0))
    uplift_reference_date: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('uplift_reference_date'))
    # Optional: when set, automation_logistic_asymptote is SOLVED rather than used
    serial_uplift_at_ac: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('serial_uplift_at_ac'))

    # Manual horizon fitting parameters
    present_day: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['present_day'])
    present_horizon: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['present_horizon'])
    present_doubling_time: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['present_doubling_time'])
    doubling_difficulty_growth_factor: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['doubling_difficulty_growth_factor'])

    # Normalization
    coding_labor_normalization: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['coding_labor_normalization'])

    # exp capacity pseudoparameters
    inf_labor_asymptote: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['inf_labor_asymptote'])
    inf_compute_asymptote: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['inf_compute_asymptote'])
    labor_anchor_exp_cap: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['labor_anchor_exp_cap'])
    compute_anchor_exp_cap: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['compute_anchor_exp_cap'])
    inv_compute_anchor_exp_cap: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['inv_compute_anchor_exp_cap'])
    # penalty on parallel coding labor in exp capacity CES
    parallel_penalty: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['parallel_penalty'])

    # Research taste distribution parameter
    median_to_top_taste_multiplier: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['median_to_top_taste_multiplier'])
    top_percentile: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['top_percentile'])
    taste_limit: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['taste_limit'])
    taste_limit_smoothing: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['taste_limit_smoothing'])
    taste_distribution: Optional[TasteDistribution] = field(default=None, init=False)  # Initialized in __post_init__
    # Median-to-best multipliers for general capability milestones
    strat_ai_m2b: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['strat_ai_m2b'])
    ted_ai_m2b: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['ted_ai_m2b'])

    # Benchmarks and gaps mode
    include_gap: Union[str, bool] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['include_gap'])
    gap_years: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['gap_years'])

    # Coding labor mode and optimal CES params (see docs/optimal-ces-coding-labor.md)
    coding_labor_mode: str = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('coding_labor_mode', 'simple_ces'))
    coding_automation_efficiency_slope: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('coding_automation_efficiency_slope', 1.0))
    optimal_ces_eta_init: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('optimal_ces_eta_init', 1.0))
    optimal_ces_grid_size: int = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('optimal_ces_grid_size', 4096))
    optimal_ces_frontier_tail_eps: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('optimal_ces_frontier_tail_eps', 1e-6))
    optimal_ces_frontier_cap: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('optimal_ces_frontier_cap'))
    max_serial_coding_labor_multiplier: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('max_serial_coding_labor_multiplier'))

    # SIE mode
    sos_mode: bool = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['sos_mode'])
    sos_start_milestone: str = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['sos_start_milestone'])

    def __post_init__(self):
        """Validate and sanitize parameters after initialization"""
        # Normalize taste parameters early so downstream numpy ops see scalars
        self.median_to_top_taste_multiplier = _coerce_float_scalar(
            self.median_to_top_taste_multiplier, "median_to_top_taste_multiplier")
        self.top_percentile = _coerce_float_scalar(self.top_percentile, "top_percentile")
        self.taste_limit = _coerce_float_scalar(self.taste_limit, "taste_limit")
        self.taste_limit_smoothing = _coerce_float_scalar(self.taste_limit_smoothing, "taste_limit_smoothing")

        # Sanitize elasticity parameters
        if not np.isfinite(self.rho_coding_labor):
            logger.warning(f"Non-finite rho_coding_labor: {self.rho_coding_labor}, setting to 0")
            self.rho_coding_labor = 0.0


        if not np.isfinite(self.r_software):
            logger.warning(f"Non-finite r_software: {self.r_software}, setting to 1.0")
            self.r_software = 1.0

        # Sanitize automation parameters
        if not np.isfinite(self.automation_fraction_at_coding_automation_anchor):
            logger.warning(f"Non-finite automation_fraction_at_coding_automation_anchor: {self.automation_fraction_at_coding_automation_anchor}, setting to {cfg.DEFAULT_PARAMETERS['automation_fraction_at_coding_automation_anchor']}")
            self.automation_fraction_at_coding_automation_anchor = cfg.DEFAULT_PARAMETERS['automation_fraction_at_coding_automation_anchor']

        # Sanitize AI research taste parameters
        # Always compute ai_research_taste_at_coding_automation_anchor from _sd
        if self.ai_research_taste_at_coding_automation_anchor_sd is None:
            self.ai_research_taste_at_coding_automation_anchor_sd = cfg.DEFAULT_PARAMETERS.get('ai_research_taste_at_coding_automation_anchor_sd', 0.5)

        if not np.isfinite(self.ai_research_taste_at_coding_automation_anchor_sd):
            logger.warning(f"Non-finite ai_research_taste_at_coding_automation_anchor_sd: {self.ai_research_taste_at_coding_automation_anchor_sd}, using default")
            self.ai_research_taste_at_coding_automation_anchor_sd = cfg.DEFAULT_PARAMETERS.get('ai_research_taste_at_coding_automation_anchor_sd', 0.5)

        # Initialize taste distribution (used throughout the model)
        # Use cached version to avoid expensive reinitialization
        try:
            self.taste_distribution = get_or_create_taste_distribution(
                top_percentile=self.top_percentile,
                median_to_top_gap=self.median_to_top_taste_multiplier,
                taste_limit=self.taste_limit,
                taste_limit_smoothing=self.taste_limit_smoothing
            )
        except Exception as e:
            logger.error(f"Failed to initialize TasteDistribution: {e}")
            raise

        # Convert SD to taste value using the taste distribution
        try:
            converted_taste = self.taste_distribution.get_taste_at_sd(float(self.ai_research_taste_at_coding_automation_anchor_sd))
            if np.isfinite(converted_taste):
                self.ai_research_taste_at_coding_automation_anchor = float(converted_taste)
            else:
                logger.warning(f"Converted taste is non-finite, using fallback")
                self.ai_research_taste_at_coding_automation_anchor = cfg.DEFAULT_PARAMETERS.get('ai_research_taste_at_coding_automation_anchor_fallback', 0.95)
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Failed converting ai_research_taste_at_coding_automation_anchor_sd to taste: {e}, using fallback")
            self.ai_research_taste_at_coding_automation_anchor = cfg.DEFAULT_PARAMETERS.get('ai_research_taste_at_coding_automation_anchor_fallback', 0.95)

        if not np.isfinite(self.ai_research_taste_slope):
            logger.warning(f"Non-finite ai_research_taste_slope: {self.ai_research_taste_slope}, setting to 1.0")
            self.ai_research_taste_slope = 1.0
        # Sanitize time horizon parameter
        if not np.isfinite(self.ac_time_horizon_minutes) or self.ac_time_horizon_minutes <= 0:
            logger.warning(f"Invalid ac_time_horizon_minutes: {self.ac_time_horizon_minutes}, setting to {cfg.DEFAULT_PARAMETERS['ac_time_horizon_minutes']}")
            self.ac_time_horizon_minutes = cfg.DEFAULT_PARAMETERS['ac_time_horizon_minutes']

        # Sanitize parallel_penalty
        if not np.isfinite(self.parallel_penalty):
            logger.warning(f"Non-finite parallel_penalty: {self.parallel_penalty}, setting to {cfg.DEFAULT_PARAMETERS['parallel_penalty']}")
            self.parallel_penalty = cfg.DEFAULT_PARAMETERS['parallel_penalty']
        else:
            self.parallel_penalty = float(np.clip(self.parallel_penalty, cfg.PARALLEL_PENALTY_MIN, cfg.PARALLEL_PENALTY_MAX))

        # Sanitize categorical parameters
        if self.horizon_extrapolation_type not in cfg.HORIZON_EXTRAPOLATION_TYPES:
            logger.warning(f"Invalid horizon_extrapolation_type: {self.horizon_extrapolation_type}, setting to default")
            self.horizon_extrapolation_type = cfg.DEFAULT_HORIZON_EXTRAPOLATION_TYPE
        if self.ac_anchor_mode not in cfg.AC_ANCHOR_MODES:
            logger.warning(f"Invalid ac_anchor_mode: {self.ac_anchor_mode}, setting to default")
            self.ac_anchor_mode = cfg.DEFAULT_AC_ANCHOR_MODE

        # Sanitize revenue AC-anchor parameters
        if not np.isfinite(self.present_revenue) or self.present_revenue <= 0:
            logger.warning(f"Invalid present_revenue: {self.present_revenue}, setting to default")
            self.present_revenue = cfg.DEFAULT_PARAMETERS['present_revenue']
        if not np.isfinite(self.present_revenue_growth_rate) or self.present_revenue_growth_rate <= 1.0:
            logger.warning(f"Invalid present_revenue_growth_rate: {self.present_revenue_growth_rate}, setting to default")
            self.present_revenue_growth_rate = cfg.DEFAULT_PARAMETERS['present_revenue_growth_rate']
        if not np.isfinite(self.revenue_required_for_ac) or self.revenue_required_for_ac <= 0:
            logger.warning(f"Invalid revenue_required_for_ac: {self.revenue_required_for_ac}, setting to default")
            self.revenue_required_for_ac = cfg.DEFAULT_PARAMETERS['revenue_required_for_ac']
        if not np.isfinite(self.eci_at_present_day):
            logger.warning(f"Invalid eci_at_present_day: {self.eci_at_present_day}, setting to default")
            self.eci_at_present_day = cfg.DEFAULT_PARAMETERS['eci_at_present_day']
        if not np.isfinite(self.eci_growth_rate) or self.eci_growth_rate <= 0:
            logger.warning(f"Invalid eci_growth_rate: {self.eci_growth_rate}, setting to default")
            self.eci_growth_rate = cfg.DEFAULT_PARAMETERS['eci_growth_rate']
        if not np.isfinite(self.eci_required_for_ac):
            logger.warning(f"Invalid eci_required_for_ac: {self.eci_required_for_ac}, setting to default")
            self.eci_required_for_ac = cfg.DEFAULT_PARAMETERS['eci_required_for_ac']
        # Sanitize 'uplift-solved' AC-anchor parameters
        if self.serial_uplift_at_reference_date is None:
            self.serial_uplift_at_reference_date = float(
                cfg.DEFAULT_PARAMETERS['serial_uplift_at_reference_date'])
        if (not np.isfinite(self.serial_uplift_at_reference_date)
                or self.serial_uplift_at_reference_date <= 1.0):
            raise ValueError(
                "serial_uplift_at_reference_date must be a finite value > 1 (it is a "
                f"multiplier on human coding labor); got "
                f"{self.serial_uplift_at_reference_date!r}. Substituting the default "
                "belief here would silently answer a different question than the caller "
                "asked."
            )
        if not np.isfinite(self.excess_uplift_doubling_time) or self.excess_uplift_doubling_time <= 0:
            logger.warning(f"Invalid excess_uplift_doubling_time: {self.excess_uplift_doubling_time}, setting to default")
            self.excess_uplift_doubling_time = cfg.DEFAULT_PARAMETERS['excess_uplift_doubling_time']
        if self.serial_uplift_at_ac is not None and (
            not np.isfinite(self.serial_uplift_at_ac) or self.serial_uplift_at_ac <= 1.0
        ):
            logger.warning(f"Invalid serial_uplift_at_ac: {self.serial_uplift_at_ac}, disabling the solved-asymptote condition")
            self.serial_uplift_at_ac = None
        if self.uplift_reference_date is not None and not np.isfinite(self.uplift_reference_date):
            logger.warning(f"Invalid uplift_reference_date: {self.uplift_reference_date}, falling back to present_day")
            self.uplift_reference_date = None
        for _date_attr in ('horizon_reference_date', 'revenue_reference_date', 'eci_reference_date'):
            _date_val = getattr(self, _date_attr)
            if _date_val is not None and not np.isfinite(_date_val):
                logger.warning(f"Invalid {_date_attr}: {_date_val}, falling back to present_day")
                setattr(self, _date_attr, None)
        if self.automation_interp_type not in ["linear", "exponential", "logistic"]:
            logger.warning(f"Invalid automation_interp_type: {self.automation_interp_type}, setting to default")
            self.automation_interp_type = cfg.DEFAULT_PARAMETERS['automation_interp_type']

        # Sanitize manual horizon fitting parameters
        if not np.isfinite(self.present_day):
            logger.warning(f"Non-finite present_day: {self.present_day}, setting to default")
            self.present_day = cfg.DEFAULT_present_day

        # --- Uplift consolidation -------------------------------------------
        # Runs AFTER parallel_penalty, present_day, uplift_reference_date and
        # excess_uplift_doubling_time are sanitized, because the serial->parallel
        # conversion needs all four.
        self.swe_multiplier_at_present_day = resolve_swe_multiplier_at_present_day(self)

        # Validate optional parameters - if provided, ensure they're finite and positive
        if self.present_horizon is not None:
            if not np.isfinite(self.present_horizon) or self.present_horizon <= 0:
                logger.warning(f"Invalid present_horizon: {self.present_horizon}, setting to None for optimization")
                self.present_horizon = None

        if self.present_doubling_time is not None:
            if not np.isfinite(self.present_doubling_time) or self.present_doubling_time <= 0:
                logger.warning(f"Invalid present_doubling_time: {self.present_doubling_time}, setting to None for optimization")
                self.present_doubling_time = None

        if self.doubling_difficulty_growth_factor is not None:
            if not np.isfinite(self.doubling_difficulty_growth_factor):
                logger.warning(f"Invalid doubling_difficulty_growth_factor: {self.doubling_difficulty_growth_factor}, setting to None for optimization")
                self.doubling_difficulty_growth_factor = None

        if self.inv_compute_anchor_exp_cap is not None:
            self.compute_anchor_exp_cap = 1 / self.inv_compute_anchor_exp_cap

        # Sanitize include_gap parameter (new API)
        include_gap_bool = False
        try:
            if isinstance(self.include_gap, str):
                val = self.include_gap.strip().lower()
                if val in ("gap", "yes", "true", "1"):  # accept common truthy variants
                    include_gap_bool = True
                elif val in ("no gap", "no", "false", "0"):
                    include_gap_bool = False
                else:
                    include_gap_bool = bool(cfg.DEFAULT_PARAMETERS['include_gap'] == 'gap')
            else:
                include_gap_bool = bool(self.include_gap)
        except Exception as e:
            if should_reraise(e):
                raise
            include_gap_bool = bool(cfg.DEFAULT_PARAMETERS['include_gap'] == 'gap')
        # Normalize include_gap to canonical string for consistency
        self.include_gap = "gap" if include_gap_bool else "no gap"
        if not np.isfinite(self.gap_years) or self.gap_years < 0:
            logger.warning(f"Invalid gap_years: {self.gap_years}, setting to default")
            self.gap_years = cfg.DEFAULT_PARAMETERS['gap_years']

        # Sanitize normalization parameters
        if not np.isfinite(self.coding_labor_normalization) or self.coding_labor_normalization <= 0:
            logger.warning(f"Invalid coding_labor_normalization: {self.coding_labor_normalization}, setting to 1.0")
            self.coding_labor_normalization = 1.0

        # Sanitize coding_labor_mode
        if self.coding_labor_mode not in ("simple_ces", "optimal_ces"):
            logger.warning(f"Invalid coding_labor_mode: {self.coding_labor_mode}, defaulting to 'simple_ces'")
            self.coding_labor_mode = 'simple_ces'
        # Sanitize optimal CES parameters
        try:
            if not np.isfinite(self.coding_automation_efficiency_slope) or self.coding_automation_efficiency_slope <= 0:
                self.coding_automation_efficiency_slope = float(cfg.DEFAULT_PARAMETERS.get('coding_automation_efficiency_slope', 1.0))
            if not np.isfinite(self.optimal_ces_eta_init) or self.optimal_ces_eta_init <= 0:
                self.optimal_ces_eta_init = float(cfg.DEFAULT_PARAMETERS.get('optimal_ces_eta_init', 1.0))
            if not np.isfinite(self.optimal_ces_grid_size) or int(self.optimal_ces_grid_size) < 256:
                self.optimal_ces_grid_size = int(cfg.DEFAULT_PARAMETERS.get('optimal_ces_grid_size', 4096))
            else:
                self.optimal_ces_grid_size = int(self.optimal_ces_grid_size)

            # Apply parallel_penalty transformation to coding_automation_efficiency_slope
            # Transform: slope_transformed = slope_input * (0.5 / parallel_penalty)
            if np.isfinite(self.parallel_penalty) and self.parallel_penalty != 0:
                try:
                    factor = 0.5 / self.parallel_penalty
                    original_slope = self.coding_automation_efficiency_slope
                    self.coding_automation_efficiency_slope = float(self.coding_automation_efficiency_slope * factor)
                    logger.info(f"Transformed coding_automation_efficiency_slope: {original_slope:.6f} * ({factor:.6f}) = {self.coding_automation_efficiency_slope:.6f}")

                    if not np.isfinite(self.coding_automation_efficiency_slope) or self.coding_automation_efficiency_slope <= 0:
                        logger.warning(f"Non-finite result after parallel_penalty transformation, reverting to original value: {original_slope}")
                        self.coding_automation_efficiency_slope = original_slope
                except (OverflowError, ValueError) as e:
                    logger.warning(f"Error applying parallel_penalty transformation to coding_automation_efficiency_slope: {e}, keeping original value")
            bounds_map = getattr(cfg, 'PARAMETER_BOUNDS', {})

            tail_eps_limits = bounds_map.get('optimal_ces_frontier_tail_eps', (1e-12, 1e-1))
            tail_eps = getattr(self, 'optimal_ces_frontier_tail_eps', cfg.DEFAULT_PARAMETERS.get('optimal_ces_frontier_tail_eps', 1e-6))
            if not np.isfinite(tail_eps) or tail_eps <= 0:
                tail_eps = cfg.DEFAULT_PARAMETERS.get('optimal_ces_frontier_tail_eps', 1e-6)
            if isinstance(tail_eps_limits, (list, tuple)) and len(tail_eps_limits) == 2:
                tail_min, tail_max = float(tail_eps_limits[0]), float(tail_eps_limits[1])
            else:
                tail_min, tail_max = 1e-12, 1e-1
            tail_min = max(tail_min, 1e-24)
            tail_max = min(max(tail_max, tail_min * 10.0), 0.5)
            self.optimal_ces_frontier_tail_eps = float(np.clip(tail_eps, tail_min, tail_max))

            cap_limits = bounds_map.get('optimal_ces_frontier_cap', (1.0, 1e30))
            cap_min, cap_max = cap_limits if isinstance(cap_limits, (list, tuple)) and len(cap_limits) == 2 else (1.0, 1e30)
            cap_val = getattr(self, 'optimal_ces_frontier_cap', None)
            if cap_val is not None:
                if not np.isfinite(cap_val) or cap_val <= 0:
                    self.optimal_ces_frontier_cap = None
                else:
                    self.optimal_ces_frontier_cap = float(np.clip(cap_val, cap_min, cap_max))

            max_serial = getattr(self, 'max_serial_coding_labor_multiplier', None)
            if max_serial is not None:
                if not np.isfinite(max_serial) or max_serial <= 0:
                    logger.warning(f"Invalid max_serial_coding_labor_multiplier: {max_serial}, ignoring")
                    self.max_serial_coding_labor_multiplier = None
                else:
                    serial_limits = bounds_map.get('max_serial_coding_labor_multiplier', (1.0, 1e30))
                    serial_min, serial_max = serial_limits if isinstance(serial_limits, (list, tuple)) and len(serial_limits) == 2 else (1.0, 1e30)
                    max_serial = float(np.clip(max_serial, serial_min, serial_max))
                    self.max_serial_coding_labor_multiplier = max_serial
                    penalty = float(self.parallel_penalty)
                    if penalty <= 0:
                        logger.warning("parallel_penalty must be positive to derive optimal_ces_frontier_cap from max_serial_coding_labor_multiplier")
                    else:
                        try:
                            exponent = 1.0 / penalty
                            derived_cap = float(np.power(max_serial, exponent))
                        except (OverflowError, ValueError):
                            derived_cap = None
                        if derived_cap is not None and np.isfinite(derived_cap) and derived_cap > 0:
                            self.optimal_ces_frontier_cap = float(np.clip(derived_cap, cap_min, cap_max))
                        else:
                            logger.warning("Could not derive optimal_ces_frontier_cap from max_serial_coding_labor_multiplier; leaving cap unset")
                            self.optimal_ces_frontier_cap = None

            if self.optimal_ces_frontier_cap is not None:
                rho = float(self.rho_coding_labor)
                denom = 1.0 - rho
                if abs(denom) > 1e-12:
                    exponent = rho / denom
                    try:
                        derived_eps = float(np.power(self.optimal_ces_frontier_cap, exponent))
                    except (OverflowError, ValueError):
                        derived_eps = None
                    if derived_eps is not None and np.isfinite(derived_eps) and derived_eps > 0:
                        self.optimal_ces_frontier_tail_eps = float(np.clip(derived_eps, tail_min, tail_max))
                else:
                    logger.warning("optimal_ces_frontier_cap ignored because rho is too close to 1.0")
        except Exception as e:
            if should_reraise(e):
                raise
            self.coding_labor_mode = 'simple_ces'


# =============================================================================
# The uplift pair: one physical quantity, two containers
# =============================================================================
#
# The current AI coding uplift is described by exactly one number. Two fields
# carry it, in different spaces and at different dates:
#
#   S = serial_uplift_at_reference_date   -- SERIAL space, at t* (PRIMARY)
#   W = swe_multiplier_at_present_day     -- PARALLEL space, at present_day
#                                            (DERIVED / legacy)
#
# S is the source of truth: it is what the Q2 elicitation states, what the
# 'uplift-solved' mode's target M* is, and what the unconditional
# D-reconstruction anchors on. W exists because older URLs, presets, sampling
# configs and tests state the belief in parallel space at the present day.
#
# The two spaces are related by the parallel penalty, and the two DATES by the
# model's own excess-uplift growth law -- the same law
# `uplift_reconstruction.serial_multiplier_path` integrates:
#
#     M(t) - 1 = (M(t*) - 1) * 2 ** ((t - t*) * 12 / D)
#
# so with  g = 2 ** (Δt_months / D),  Δt_months = (t* - present_day) * 12,
#
#     forward  (canonical):  W = (1 + (S - 1) / g) ** (1 / pp)
#     backward (legacy):     S = 1 + (W ** pp - 1) * g
#
# Float note: at Δt == 0 the backward form is `1.0 + (W**pp - 1.0) * 1.0`,
# which is bit-identical to `W**pp` for every W with ulp(W**pp) <= 1 (the
# subtraction is exact, and adding 1.0 back is then exact too). The default
# pipeline is therefore unchanged to the last bit.


def _uplift_parallel_penalty(params) -> float:
    """The parallel->serial exponent, or ValueError when it cannot be used."""
    try:
        pp = float(params.parallel_penalty)
    except (AttributeError, TypeError, ValueError) as e:
        raise ValueError(
            f"parallel_penalty is required to convert between the parallel and "
            f"serial uplift containers, but could not be read: {e}"
        ) from e
    if not np.isfinite(pp) or pp <= 0.0:
        raise ValueError(
            f"parallel_penalty must be finite and > 0 to convert between "
            f"swe_multiplier_at_present_day (parallel space) and "
            f"serial_uplift_at_reference_date (serial space); got {pp!r}. "
            f"Refusing to guess an exponent."
        )
    return pp


def _uplift_growth_factor(params) -> float:
    """g = 2 ** (Δt_months / D): how much the EXCESS uplift grows from
    present_day to the uplift reference date.

    Returns exactly 1.0 when the two dates coincide (the default), WITHOUT
    reading excess_uplift_doubling_time -- there is nothing to extrapolate, so
    a degenerate doubling time is irrelevant. When they do not coincide a
    usable doubling time is mandatory and its absence raises rather than
    silently assuming g == 1 (which is the date-blind bug this replaces).
    """
    try:
        present_day = float(params.present_day)
    except (AttributeError, TypeError, ValueError) as e:
        raise ValueError(f"present_day is required to date the uplift belief: {e}") from e

    reference_date = getattr(params, 'uplift_reference_date', None)
    if reference_date is None:
        return 1.0
    reference_date = float(reference_date)
    if not np.isfinite(reference_date) or not np.isfinite(present_day):
        raise ValueError(
            f"uplift_reference_date ({reference_date!r}) and present_day "
            f"({present_day!r}) must both be finite to date the uplift belief."
        )

    delta_months = (reference_date - present_day) * 12.0
    if delta_months == 0.0:
        return 1.0

    doubling_time = getattr(params, 'excess_uplift_doubling_time', None)
    if doubling_time is None:
        raise ValueError(
            f"uplift_reference_date ({reference_date}) differs from present_day "
            f"({present_day}) by {delta_months:.6f} months, so converting between "
            f"serial_uplift_at_reference_date and swe_multiplier_at_present_day "
            f"needs excess_uplift_doubling_time -- but it is None."
        )
    doubling_time = float(doubling_time)
    if not np.isfinite(doubling_time) or doubling_time <= 0.0:
        raise ValueError(
            f"uplift_reference_date ({reference_date}) differs from present_day "
            f"({present_day}) by {delta_months:.6f} months, so converting between "
            f"serial_uplift_at_reference_date and swe_multiplier_at_present_day "
            f"needs a positive, finite excess_uplift_doubling_time -- got "
            f"{doubling_time!r}. Refusing to assume the excess does not grow."
        )

    growth = float(2.0 ** (delta_months / doubling_time))
    if not np.isfinite(growth) or growth <= 0.0:
        raise ValueError(
            f"The excess-uplift growth factor between present_day ({present_day}) "
            f"and uplift_reference_date ({reference_date}) with "
            f"excess_uplift_doubling_time={doubling_time} months is {growth!r}, "
            f"which is not a usable multiplier."
        )
    return growth


def resolve_swe_multiplier_at_present_day(params) -> float:
    """The present-day PARALLEL coding multiplier, DERIVED from the serial container.

        W = (1 + (S - 1) / g) ** (1 / pp)

    the inverse of reading the serial belief S at its reference date. This is a
    view, not an input: the field is init=False, so nothing can state W and there
    is no pair to reconcile.
    """
    serial = float(params.serial_uplift_at_reference_date)
    penalty = _uplift_parallel_penalty(params)
    growth = _uplift_growth_factor(params)
    return (1.0 + (serial - 1.0) / growth) ** (1.0 / penalty)


