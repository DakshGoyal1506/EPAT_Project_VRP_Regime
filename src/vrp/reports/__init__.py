"""Report generation utilities for the VRP regime project."""

try:
    from vrp.reports.backtest_diagnostics import (
        BacktestDiagnosticsResult,
        VISUAL_INTERPRETATION_WARNING,
        build_backtest_by_strategy_year_table,
        build_backtest_summary_table,
        build_common_start_panel,
        build_common_start_summary_table,
        build_crisis_window_performance_table,
        build_no_lookahead_audit_table,
        build_report_metadata,
        build_tail_summary_table,
        generate_backtest_diagnostics,
        get_common_start_dates,
        load_backtest_panels,
        render_diagnostics_summary,
        write_backtest_diagnostic_figures,
        write_backtest_diagnostic_tables,
    )

    __all__ = [
        "BacktestDiagnosticsResult",
        "VISUAL_INTERPRETATION_WARNING",
        "build_backtest_by_strategy_year_table",
        "build_backtest_summary_table",
        "build_common_start_panel",
        "build_common_start_summary_table",
        "build_crisis_window_performance_table",
        "build_no_lookahead_audit_table",
        "build_report_metadata",
        "build_tail_summary_table",
        "generate_backtest_diagnostics",
        "get_common_start_dates",
        "load_backtest_panels",
        "render_diagnostics_summary",
        "write_backtest_diagnostic_figures",
        "write_backtest_diagnostic_tables",
    ]
except ModuleNotFoundError as exc:
    if exc.name != "matplotlib":
        raise
    __all__: list[str] = []
