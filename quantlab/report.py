"""Layer 5, reporting (QA Section F).

One self-contained HTML file: equity curve, drawdowns, monthly table, risk metrics,
and the QA report. Charts are emitted as inline SVG, so the page renders offline and
survives being emailed, archived, or opened three years later.

The ordering is deliberate: the QA verdict sits at the top, above the returns. If a
strategy failed its checks, that is the first thing a reader should meet, not a
footnote below a flattering equity curve.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import metrics as M

CSS = """
:root { --ink:#1a1a2e; --muted:#6b7280; --line:#e5e7eb; --bg:#ffffff;
        --pass:#0f9d58; --warn:#f4b400; --fail:#db4437; --accent:#5b2c9f; }
* { box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       margin:0; padding:32px; color:var(--ink); background:#f7f7fb; line-height:1.5; }
.wrap { max-width:1080px; margin:0 auto; }
.card { background:var(--bg); border:1px solid var(--line); border-radius:10px;
        padding:24px; margin-bottom:20px; }
h1 { margin:0 0 4px; font-size:26px; }
h2 { font-size:17px; margin:0 0 14px; padding-bottom:8px; border-bottom:2px solid var(--line); }
.sub { color:var(--muted); font-size:14px; margin-bottom:24px; }
.banner { padding:16px 20px; border-radius:8px; font-weight:600; margin-bottom:20px; color:#fff; }
.banner.pass { background:var(--pass); } .banner.fail { background:var(--fail); }
.banner .detail { font-weight:400; font-size:13px; margin-top:6px; opacity:.95; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
.metric { background:#fafafa; border:1px solid var(--line); border-radius:8px; padding:14px; }
.metric .label { font-size:11px; text-transform:uppercase; letter-spacing:.5px; color:var(--muted); }
.metric .value { font-size:22px; font-weight:700; margin-top:4px; font-variant-numeric:tabular-nums; }
.pos { color:var(--pass); } .neg { color:var(--fail); }
table { width:100%; border-collapse:collapse; font-size:13px; }
th { text-align:left; padding:8px 10px; background:#fafafa; border-bottom:2px solid var(--line);
     font-size:11px; text-transform:uppercase; letter-spacing:.4px; color:var(--muted); }
td { padding:8px 10px; border-bottom:1px solid var(--line); font-variant-numeric:tabular-nums; }
.pill { display:inline-block; padding:2px 9px; border-radius:11px; font-size:11px;
        font-weight:700; color:#fff; }
.pill.PASS { background:var(--pass); } .pill.WARN { background:var(--warn); }
.pill.FAIL { background:var(--fail); }
.note { background:#fdf6e3; border-left:3px solid var(--warn); padding:12px 16px;
        font-size:13px; margin-top:14px; border-radius:0 6px 6px 0; }
.chart { width:100%; height:auto; }
.foot { text-align:center; color:var(--muted); font-size:12px; margin-top:8px; }
"""


def _fmt(v, kind="pct"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "&mdash;"
    if kind == "pct":
        return f"{v:.2%}"
    if kind == "num":
        return f"{v:.2f}"
    if kind == "int":
        return f"{int(v):,}"
    return str(v)


def _sparkline(series: pd.Series, width=1000, height=240, color="#5b2c9f",
               fill=True, log_scale=False, zero_base=False) -> str:
    """Inline SVG line chart, with no JavaScript and no CDN: the report stands on its own."""
    s = series.dropna()
    if len(s) < 2:
        return "<p style='color:#6b7280'>Not enough data to plot.</p>"

    y = np.log(s.values) if log_scale and (s > 0).all() else s.values
    x = np.linspace(0, width, len(y))
    lo, hi = float(np.min(y)), float(np.max(y))
    if zero_base:
        hi = max(hi, 0.0)
    span = hi - lo if hi > lo else 1.0
    pad = height * 0.08
    py = height - pad - (y - lo) / span * (height - 2 * pad)

    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, py))
    baseline = height - pad - (0 - lo) / span * (height - 2 * pad) if zero_base else height
    area = (f'<polygon points="0,{baseline:.1f} {pts} {width},{baseline:.1f}" '
            f'fill="{color}" opacity="0.13"/>') if fill else ""

    # Five date ticks only: a dense axis competes with the line for attention.
    n = len(s)
    ticks = ""
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        i = min(n - 1, int(frac * (n - 1)))
        xt = x[i]
        anchor = "start" if frac == 0 else ("end" if frac == 1.0 else "middle")
        ticks += (f'<text x="{xt:.0f}" y="{height-2}" font-size="10" fill="#6b7280" '
                  f'text-anchor="{anchor}">{s.index[i]:%Y-%m}</text>')

    return (f'<svg class="chart" viewBox="0 0 {width} {height}" '
            f'xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">'
            f'<line x1="0" y1="{baseline:.1f}" x2="{width}" y2="{baseline:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
            f'{area}<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>'
            f'{ticks}</svg>')


def _metric_cards(stats: dict) -> str:
    items = [
        ("CAGR", stats.get("cagr"), "pct", True),
        ("Volatility", stats.get("volatility"), "pct", False),
        ("Sharpe", stats.get("sharpe"), "num", True),
        ("Sortino", stats.get("sortino"), "num", True),
        ("Max Drawdown", stats.get("max_drawdown"), "pct", True),
        ("DD Duration", stats.get("max_dd_duration_days"), "int", False),
        ("Calmar", stats.get("calmar"), "num", True),
        ("Win Rate", stats.get("win_rate"), "pct", False),
        ("Skew", stats.get("skew"), "num", False),
        ("Excess Kurtosis", stats.get("excess_kurtosis"), "num", False),
        ("CVaR 95%", stats.get("cvar_95"), "pct", True),
        ("Deflated Sharpe", stats.get("deflated_sharpe"), "num", True),
    ]
    out = []
    for label, val, kind, signed in items:
        cls = ""
        if signed and isinstance(val, (int, float)) and np.isfinite(val):
            cls = "pos" if val > 0 else ("neg" if val < 0 else "")
        suffix = " days" if label == "DD Duration" else ""
        out.append(f'<div class="metric"><div class="label">{label}</div>'
                   f'<div class="value {cls}">{_fmt(val, kind)}{suffix}</div></div>')
    return f'<div class="grid">{"".join(out)}</div>'


def _monthly_table(returns: pd.Series) -> str:
    tbl = M.monthly_returns_table(returns)
    if tbl.empty:
        return "<p>No monthly data.</p>"
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    head = "".join(f"<th>{m}</th>" for m in months)
    rows = ""
    for year, row in tbl.iterrows():
        cells = ""
        for m in range(1, 13):
            v = row.get(m, np.nan)
            if pd.isna(v):
                cells += "<td></td>"
            else:
                cls = "pos" if v > 0 else "neg"
                cells += f'<td class="{cls}">{v:.1%}</td>'
        yr_total = (1 + row.dropna()).prod() - 1
        cls = "pos" if yr_total > 0 else "neg"
        rows += (f"<tr><td><strong>{year}</strong></td>{cells}"
                 f'<td class="{cls}"><strong>{yr_total:.1%}</strong></td></tr>')
    return f"<table><tr><th>Year</th>{head}<th>Total</th></tr>{rows}</table>"


def _df_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "<p style='color:#6b7280'>Not run.</p>"
    d = df.head(max_rows)
    head = "".join(f"<th>{c}</th>" for c in d.columns)
    rows = ""
    for _, r in d.iterrows():
        cells = ""
        for c in d.columns:
            v = r[c]
            if isinstance(v, float):
                cells += f"<td>{v:,.4f}</td>" if abs(v) < 1000 else f"<td>{v:,.0f}</td>"
            else:
                cells += f"<td>{v}</td>"
        rows += f"<tr>{cells}</tr>"
    more = (f"<p style='color:#6b7280;font-size:12px'>Showing {max_rows} of {len(df)} rows.</p>"
            if len(df) > max_rows else "")
    return f"<table><tr>{head}</tr>{rows}</table>{more}"


def _qa_table(qa) -> str:
    df = qa.to_frame()
    if df.empty:
        return "<p>No checks run.</p>"
    rows = ""
    for _, r in df.iterrows():
        rows += (f'<tr><td><strong>{r["section"]}</strong></td><td>{r["check"]}</td>'
                 f'<td><span class="pill {r["status"]}">{r["status"]}</span></td>'
                 f'<td style="font-size:12px">{r["detail"]}</td></tr>')
    return f"<table><tr><th>Sec</th><th>Check</th><th>Status</th><th>Detail</th></tr>{rows}</table>"


def build_report(pr, output_path: str | Path, title: str | None = None) -> Path:
    """Render a ``PipelineResult`` to one standalone HTML file; returns the path written."""
    res, stats, qa = pr.result, pr.stats, pr.qa
    cfg = pr.config
    title = title or f"{res.strategy_name}: Backtest Report"

    equity = res.equity
    dd = M.drawdown_series(res.returns)

    banner_cls = "pass" if qa.passed else "fail"
    banner_txt = "QA PASSED" if qa.passed else f"QA FAILED: {len(qa.failures)} blocking issue(s)"
    banner_detail = qa.summary_line()
    if not qa.passed:
        banner_detail += " (" + "; ".join(c.name for c in qa.failures) + ")"

    bench_block = ""
    if pr.benchmark_returns is not None and "benchmark_cagr" in stats:
        bench_block = f"""
        <div class="card"><h2>Benchmark Comparison ({cfg.benchmark_symbol})</h2>
        <div class="grid">
          <div class="metric"><div class="label">Strategy CAGR</div><div class="value">{_fmt(stats['cagr'])}</div></div>
          <div class="metric"><div class="label">Benchmark CAGR</div><div class="value">{_fmt(stats['benchmark_cagr'])}</div></div>
          <div class="metric"><div class="label">Excess CAGR</div><div class="value {'pos' if stats.get('excess_cagr',0)>0 else 'neg'}">{_fmt(stats.get('excess_cagr'))}</div></div>
          <div class="metric"><div class="label">Beta</div><div class="value">{_fmt(stats.get('beta'),'num')}</div></div>
          <div class="metric"><div class="label">Alpha (ann.)</div><div class="value {'pos' if stats.get('alpha',0)>0 else 'neg'}">{_fmt(stats.get('alpha'))}</div></div>
          <div class="metric"><div class="label">Info Ratio</div><div class="value">{_fmt(stats.get('information_ratio'),'num')}</div></div>
        </div></div>"""

    wf_block = "<p style='color:#6b7280'>Walk-forward not run.</p>"
    if pr.walk_forward is not None and pr.walk_forward.windows:
        wf = pr.walk_forward
        wf_block = (_df_table(wf.summary()) +
                    f"<div class='note'><strong>OOS Sharpe {M.sharpe_ratio(wf.oos_returns):.2f}</strong> "
                    f"across {len(wf.windows)} folds. Train&rarr;test degradation "
                    f"{wf.degradation:.2f}; {wf.consistency:.0%} of folds positive. "
                    "Parameters were selected on training data only, with a "
                    f"{wf.embargo_days}-day embargo before each test window.</div>")

    param_note = ""
    if pr.param_verdict:
        param_note = (f"<div class='note'><strong>Verdict: {pr.param_verdict['verdict'].upper()}</strong> "
                      f"&mdash; {pr.param_verdict['detail']}</div>")

    regime_note = ""
    if pr.regime_verdict:
        regime_note = (f"<div class='note'><strong>Verdict: {pr.regime_verdict['verdict'].upper()}</strong> "
                       f"&mdash; {pr.regime_verdict['detail']}</div>")

    cost_note = ""
    if pr.cost_verdict:
        cost_note = (f"<div class='note'><strong>Verdict: {pr.cost_verdict['verdict'].upper()}</strong> "
                     f"&mdash; {pr.cost_verdict['detail']}</div>")

    strat = qa.context.get("strategy", {})
    uni = qa.context.get("universe", {})

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{title}</title><style>{CSS}</style></head><body><div class="wrap">
<h1>{title}</h1>
<div class="sub">Generated {datetime.now():%Y-%m-%d %H:%M} &middot;
{res.meta.get('start')} to {res.meta.get('end')} &middot;
{stats.get('years',0):.1f} years &middot; universe: {uni.get('name','?')} &middot;
cost model: {res.meta.get('cost_model')}</div>

<div class="banner {banner_cls}">{banner_txt}
  <div class="detail">{banner_detail}</div></div>

<div class="card"><h2>Performance</h2>{_metric_cards(stats)}</div>

<div class="card"><h2>Equity Curve (log scale)</h2>
{_sparkline(equity, log_scale=True)}</div>

<div class="card"><h2>Drawdown</h2>
{_sparkline(dd, color="#db4437", zero_base=True)}
<div class="note">Max drawdown <strong>{_fmt(stats.get('max_drawdown'))}</strong>,
longest underwater stretch <strong>{stats.get('max_dd_duration_days',0)} trading days</strong>
(~{stats.get('max_dd_duration_days',0)/21:.0f} months). Duration is usually the harder
part to live through, not depth.</div></div>

{bench_block}

<div class="card"><h2>Monthly Returns</h2>{_monthly_table(res.returns)}</div>

<div class="card"><h2>Cost Sensitivity &mdash; QA Section G</h2>
{_df_table(pr.cost_table)}{cost_note}</div>

<div class="card"><h2>Walk-Forward &mdash; QA Section C</h2>{wf_block}</div>

<div class="card"><h2>Parameter Sensitivity &mdash; QA Section C</h2>
{_df_table(pr.param_sweep, 15)}{param_note}</div>

<div class="card"><h2>Regime Performance &mdash; QA Section C</h2>
{_df_table(pr.regime_table)}{regime_note}</div>

<div class="card"><h2>Backtest QA Checklist</h2>{_qa_table(qa)}</div>

<div class="card"><h2>Configuration</h2>
<table>
<tr><th>Setting</th><th>Value</th></tr>
<tr><td>Strategy</td><td>{strat.get('name')} {strat.get('params')}</td></tr>
<tr><td>Citation</td><td>{strat.get('citation','n/a')}</td></tr>
<tr><td>Evidence grade</td><td>{strat.get('evidence','ungraded')}</td></tr>
<tr><td>Universe</td><td>{uni.get('name')} ({uni.get('n_symbols')} symbols,
    point-in-time: {uni.get('point_in_time')})</td></tr>
<tr><td>Universe rationale</td><td style="font-size:12px">{uni.get('rationale','')}</td></tr>
<tr><td>Rebalance</td><td>{cfg.rebalance}</td></tr>
<tr><td>Execution</td><td>signal at close, fill at next {cfg.execution_price}
    (lag {cfg.execution_lag} bar)</td></tr>
<tr><td>Cost model</td><td>{cfg.cost_preset}</td></tr>
<tr><td>Sizing</td><td>{cfg.sizing}</td></tr>
<tr><td>Annual turnover</td><td>{res.annual_turnover:.2f}x</td></tr>
<tr><td>Annual cost drag</td><td>{res.cost_drag_annual:.2%}</td></tr>
<tr><td>Trials assumed (DSR)</td><td>{stats.get('n_trials_assumed')}</td></tr>
</table></div>

<p class="foot">Backtested results are hypothetical and do not represent actual trading.
Past performance does not indicate future results. This report is a research artefact,
not investment advice.</p>
</div></body></html>"""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
