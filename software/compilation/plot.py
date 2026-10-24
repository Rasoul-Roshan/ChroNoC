"""Plot resource and compilation-time comparisons from pipeline metrics."""
import argparse
import csv
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
ORDER = ['Joint-CP', 'ExactRoute-CP', 'HeuristicRoute-CP', 'HeuristicRoute-DARSA']
COLORS = dict(zip(ORDER, ['#126782', '#e07a1f', '#499448', '#9d4edd']))
LABELS = {'Joint-CP': 'Joint CP-SAT', 'ExactRoute-CP': 'Exact route + CP-SAT', 'HeuristicRoute-CP': 'Heuristic route + CP-SAT', 'HeuristicRoute-DARSA': 'Heuristic route + DARSA'}
FEASIBLE = {'FEASIBLE', 'OPTIMAL', 'FEASIBLE_TIMEOUT'}
STAGES = [('total_compilation_seconds', 'Total time', '-', 'o', 0.0), ('mapping_search_seconds', 'Mapping time', '--', 's', 1.0), ('routing_seconds', 'Routing time', ':', '^', -1.0)]
MARKER_SIZE = 6.0

def number(row, key):
    raw = row.get(key, '')
    if raw is None or str(raw).strip() == '':
        return None
    try:
        value = float(raw)
    except (ValueError, TypeError):
        raise ValueError(f'Invalid numeric {key}={raw!r}')
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'Invalid {key}={raw!r}')
    return value

def load(args):
    with Path(args.csv).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        rows = list(reader)
    required = {'neurons', 'run', 'method', 'status', 'delay_profile', 'pattern', 'study_schema', 'total_compilation_seconds', 'mapping_search_seconds', 'routing_seconds', 'frame_cycles', 'total_provisioned_bits', 'state_verified'}
    if required - fields:
        raise ValueError('Missing CSV columns: ' + ', '.join(sorted(required - fields)))
    if not rows:
        raise ValueError('Empty input CSV')
    if any((r['study_schema'] != 'chronoc-compilation' for r in rows)):
        raise ValueError('Use compilation study metrics.csv.')
    choices = sorted({r['delay_profile'] for r in rows})
    profile = args.profiles or ('5:20' if '5:20' in choices else choices[0])
    if profile not in choices:
        raise ValueError('Select one --profiles value from ' + str(choices))
    rows = [r for r in rows if r['delay_profile'] == profile and r['pattern'] == args.pattern]
    if not rows:
        raise ValueError('No rows match profile and pattern')
    seen = set()
    grids = set()
    for row in rows:
        if row['method'] not in ORDER:
            raise ValueError('Unexpected pipeline ' + row['method'])
        row['neurons'] = int(row['neurons'])
        if row['neurons'] < 1:
            raise ValueError('Invalid neuron count')
        key = (row['method'], row['neurons'], row['run'])
        if key in seen:
            raise ValueError('Duplicate method/size/run: ' + str(key))
        seen.add(key)
        if row.get('grid'):
            grids.add(row['grid'])
        if row['status'] in FEASIBLE:
            if row['state_verified'].strip().lower() != 'true':
                raise ValueError('Feasible row without verified replay')
            if number(row, 'deadline_violations') not in (None, 0.0):
                raise ValueError('Deadline violations in feasible row')
            for metric in ['frame_cycles', 'total_provisioned_bits']:
                if number(row, metric) is None:
                    raise ValueError('Missing feasible resource metric ' + metric)
    if len(grids) > 1:
        raise ValueError('Plot only one mesh size per figure')
    return (rows, profile)

def aggregate(rows, methods, sizes, metric):
    out = {}
    for method in methods:
        points = []
        for n in sizes:
            group = [r for r in rows if r['method'] == method and r['neurons'] == n]
            feasible = [r for r in group if r['status'] in FEASIBLE]
            eligible = feasible
            if method == 'Joint-CP' and metric in {'mapping_search_seconds', 'routing_seconds'}:
                eligible = []
            vals = [number(r, metric) for r in eligible]
            vals = [v for v in vals if v is not None]
            points.append({'method': method, 'neurons': n, 'metric': metric, 'mean': float(np.mean(vals)) if vals else None, 'sample_sd': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0 if vals else None, 'n_values': len(vals), 'n_feasible': len(feasible), 'n_attempted': len(group)})
        out[method] = points
    return out

def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.linewidth': 0.65, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.fontsize': 5.5, 'axes.spines.top': True, 'axes.spines.right': True})

def axes_style(ax, sizes, ylabel, args):
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlim(-0.58, len(sizes) - 0.42)
    ax.set_xticks(np.arange(len(sizes)))
    ax.set_xticklabels([str(n) for n in sizes], rotation=args.tick_rotation)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.tick_params(direction='out', length=2.5, width=0.6, pad=2)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v:,.0f}' if abs(v) >= 1000 else f'{v:g}'))
    ax.set_axisbelow(True)
    if args.grid:
        ax.grid(axis='both', color='#b8bec5', alpha=0.38, linewidth=0.45)

def curve(ax, entries, x, color, linestyle, marker, args, width=0.9, marker_scale=1.0):
    y = np.array([p['mean'] if p['mean'] is not None else np.nan for p in entries])
    sd = np.array([p['sample_sd'] if p['sample_sd'] is not None else np.nan for p in entries])
    if not np.isfinite(y).any():
        return
    ax.plot(x, y, color=color, linestyle=linestyle, marker=marker, lw=width, ms=args.marker_size * marker_scale, mec='white', mew=0.3, zorder=5 if entries[0]['method'] == 'Joint-CP' else 4, clip_on=False)
    if args.uncertainty == 'sd' and any((p['n_values'] > 1 for p in entries)):
        ax.fill_between(x, np.maximum(0, y - sd), y + sd, color=color, alpha=0.1, zorder=1)

def limit_y(ax, stats, methods, args, upper=None):
    tops = [p['mean'] + (p['sample_sd'] or 0.0) for m in methods for p in stats[m] if p['mean'] is not None]
    high = max(tops, default=0.0)
    if upper is not None and upper < high:
        raise ValueError('Requested y limit would hide a measured point or SD')
    ax.set_ylim(0, upper if upper is not None else max(1.0, 1.1 * high))

def pipeline_handles(methods):
    return [Line2D([], [], color=COLORS[m], lw=1.1, label=LABELS[m]) for m in methods]

def fit_axis_to_canvas(fig, ax, right_reserve=0.0):
    pad = 1.5 / 72.0
    for _ in range(3):
        fig.canvas.draw()
        bounds = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        pos = ax.get_position()
        left = pos.x0 + pad / fig.get_figwidth() - bounds.x0
        bottom = pos.y0 + pad / fig.get_figheight() - bounds.y0
        right = pos.x1 + (1 - (right_reserve + pad) / fig.get_figwidth()) - bounds.x1
        top = pos.y1 + (1 - pad / fig.get_figheight()) - bounds.y1
        if right <= left or top <= bottom:
            raise ValueError('Canvas too small for labels. Increase figure dimensions.')
        ax.set_position([left, bottom, right - left, top - bottom])

def save(fig, out, stem, args, files):
    for ext in ['pdf', 'png']:
        path = out / (stem + '.' + ext)
        fig.savefig(path, dpi=args.dpi, bbox_inches=None, pad_inches=0, facecolor='white')
        files.append(path.name)
    plt.close(fig)

def resource_figure(stats, methods, sizes, args, out, files, metrics=None, stem=None):
    metrics = metrics or [('total_provisioned_bits', 'Provisioned memory (bits)'), ('frame_cycles', 'Clocks per frame')]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(args.resource_width, 2 * args.figure_height if len(metrics) > 1 else args.figure_height), sharex=True, squeeze=False)
    axes = axes[:, 0]
    fig.subplots_adjust(left=0.28, right=0.975, bottom=0.19 if len(metrics) > 1 else 0.32, top=0.96, hspace=0.22)
    x = np.arange(len(sizes), dtype=float)
    for ax, (metric, label) in zip(axes, metrics):
        displayed = stats[metric]
        if metric == 'total_provisioned_bits':
            label = 'Memory (kbit)'
            displayed = {m: [{**p, 'mean': p['mean'] / 1000 if p['mean'] is not None else None, 'sample_sd': p['sample_sd'] / 1000 if p['sample_sd'] is not None else None} for p in entries] for m, entries in stats[metric].items()}
        for i, m in enumerate(ORDER):
            if m not in methods:
                continue
            offset = (i - 1.5) * args.resource_jitter
            curve(ax, displayed[m], x + offset, COLORS[m], '-', 'o', args)
        axes_style(ax, sizes, label, args)
        limit_y(ax, displayed, methods, args)
    axes[-1].set_xlabel('Neurons', labelpad=2)
    if len(metrics) == 1:
        fit_axis_to_canvas(fig, axes[0])
    save(fig, out, stem or 'compilation_resources', args, files)

def timing_figure(stats, methods, sizes, args, out, files):
    fig = plt.figure(figsize=(args.timing_width, args.figure_height))
    ax = fig.add_axes([0.12, 0.3, 0.59, 0.68])
    x = np.arange(len(sizes), dtype=float)
    for i, m in enumerate(ORDER):
        if m not in methods:
            continue
        for metric, label, linestyle, marker, phase in STAGES:
            if m == 'Joint-CP' and metric != 'total_compilation_seconds':
                continue
            offset = (i - 1.5) * args.pipeline_jitter + phase * args.stage_jitter
            curve(ax, stats[metric][m], x + offset, COLORS[m], linestyle, marker, args, width=1.0 if metric == 'total_compilation_seconds' else 0.8, marker_scale=1.15 if marker == '^' else 1.0)
    axes_style(ax, sizes, 'Compilation time (s)', args)
    ax.set_xlabel('Neurons', labelpad=2)
    combined = {m: [p for metric, *_ in STAGES for p in stats[metric][m]] for m in methods}
    limit_y(ax, combined, methods, args, args.time_ymax)
    upper = ax.get_ylim()[1]
    ax.set_ylim(-0.045 * upper, upper)
    ticks = [t for t in ax.get_yticks() if 0 <= t <= upper]
    ax.set_yticks(ticks)
    ax.set_ylim(-0.045 * upper, upper)
    fit_axis_to_canvas(fig, ax, right_reserve=1.06)
    legend_x = ax.get_position().x1 + 2.0 / 72.0 / fig.get_figwidth()
    legend_top = ax.get_position().y1 - 3.0 / 72.0 / fig.get_figheight()
    first = fig.legend(handles=pipeline_handles(methods), loc='upper left', bbox_to_anchor=(legend_x, legend_top), borderaxespad=0, frameon=True, fancybox=False, edgecolor='.70', title='Pipelines', title_fontsize=6, handlelength=1.1, handletextpad=0.35, labelspacing=0.3, borderpad=0.25)
    fig.set_dpi(args.dpi)
    fig.canvas.draw()
    legend_bounds = first.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    overflow = max(0.0, legend_bounds.x1 - (1 - 1.5 / 72.0 / fig.get_figwidth()))
    if overflow:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width - overflow, pos.height])
        legend_x -= overflow
        first.set_bbox_to_anchor((legend_x, legend_top))
        fig.canvas.draw()
    bottom = first.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y0
    stage_handles = [Line2D([], [], color='black', ls=ls, marker=mk, ms=args.marker_size * (1.15 if mk == '^' else 1.0), lw=0.8, label=label) for _, label, ls, mk, _ in STAGES]
    fig.legend(handles=stage_handles, loc='upper left', bbox_to_anchor=(legend_x, bottom - 3.0 / 72.0 / fig.get_figheight()), borderaxespad=0, frameon=True, fancybox=False, edgecolor='.70', title='Phase', title_fontsize=6, handlelength=1.8, handletextpad=0.4, borderpad=0.25, labelspacing=0.25)
    save(fig, out, 'compilation_timing', args, files)

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', required=True)
    p.add_argument('--out', '--outdir', dest='out', required=True)
    p.add_argument('--profiles', default='')
    p.add_argument('--pattern', default='simultaneous')
    p.add_argument('--marker-size', type=float, default=3.0, help='Marker size at final print scale, points')
    p.add_argument('--column-width', type=float, default=3.5, help='LaTeX column width in inches')
    p.add_argument('--resource-jitter', type=float, default=0.16, help='Pipeline separation in categorical x units')
    p.add_argument('--pipeline-jitter', type=float, default=0.23, help='Timing pipeline separation in categorical x units')
    p.add_argument('--stage-jitter', type=float, default=0.115, help='Timing phase offset in categorical x units')
    p.add_argument('--figure-height', type=float, default=1.43, help='Shared height in inches for all three figures')
    p.add_argument('--resource-width', type=float, default=None)
    p.add_argument('--timing-width', type=float, default=None)
    p.add_argument('--tick-rotation', type=float, default=90.0)
    p.add_argument('--time-ymax', type=float, default=None)
    p.add_argument('--uncertainty', choices=['sd', 'none'], default='sd')
    p.add_argument('--grid', action='store_true', default=True, help='Faint grid lines, enabled by default')
    p.add_argument('--no-grid', dest='grid', action='store_false')
    p.add_argument('--combined-metrics', action='store_true', help='Also export stacked memory/clock figure, without a legend')
    p.add_argument('--separate-metrics', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--objective', action='store_true')
    p.add_argument('--dpi', type=int, default=600)
    a = p.parse_args()
    a.resource_width = a.resource_width or 2.0 / 3.5 * a.column_width
    a.timing_width = a.timing_width or 2 * a.resource_width
    if min(a.marker_size, a.resource_width, a.timing_width, a.figure_height, a.dpi) <= 0:
        raise ValueError('Invalid size or DPI')
    if min(a.resource_jitter, a.pipeline_jitter, a.stage_jitter) < 0:
        raise ValueError('Offsets must be nonnegative')
    if 1.5 * a.pipeline_jitter + a.stage_jitter > 0.47 or 1.5 * a.resource_jitter > 0.47:
        raise ValueError('Offsets would overlap neighboring workload categories; reduce jitter')
    if a.time_ymax is not None and a.time_ymax <= 0:
        raise ValueError('time-ymax must be positive')
    return a

def main():
    args = parse_args()
    rows, profile = load(args)
    style()
    methods = [m for m in ORDER if any((r['method'] == m for r in rows))]
    sizes = sorted({r['neurons'] for r in rows})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = ['total_provisioned_bits', 'frame_cycles', 'total_compilation_seconds', 'mapping_search_seconds', 'routing_seconds']
    if args.objective:
        metrics.append('objective_normalized')
    stats = {metric: aggregate(rows, methods, sizes, metric) for metric in metrics}
    files = []
    for metric, label, stem in [('total_provisioned_bits', 'Provisioned memory (bits)', 'compilation_memory'), ('frame_cycles', 'Clocks per frame', 'compilation_frames')]:
        resource_figure(stats, methods, sizes, args, out, files, [(metric, label)], stem)
    timing_figure(stats, methods, sizes, args, out, files)
    if args.combined_metrics:
        resource_figure(stats, methods, sizes, args, out, files)
    if args.objective:
        resource_figure(stats, methods, sizes, args, out, files, [('objective_normalized', 'Normalized resource score')], 'compilation_objective')
if __name__ == '__main__':
    main()
