"""Plot resource and compilation-time scaling from study metrics."""
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import warnings
SCHEMA = 'chronoc-scaling'
COLORS = ['#126782', '#e07a1f', '#499448', '#9d4edd', '#bf4055', '#595959']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']
MARKER_SIZE = 6.0
MARKER_SPACING_PT = 6.5

def parser(doc):
    p = argparse.ArgumentParser(description=doc)
    p.add_argument('--csv', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--pattern', default='simultaneous', choices=['simultaneous', 'staggered', 'silent', 'after_opportunity', 'all_silent'])
    p.add_argument('--profiles', default='', help='Comma-separated delay profiles, e.g. 5:5,5:20; default all')
    p.add_argument('--route-ymin', type=float, default=0.01, help='Shared routing-time axis minimum in seconds')
    p.add_argument('--route-ymax', type=float, default=10.0, help='Shared routing-time axis maximum in seconds')
    return p

def load(a, study):
    rows = list(csv.DictReader(open(a.csv, newline='')))
    if not rows:
        raise ValueError('Empty CSV')
    if any((r.get('schema') != SCHEMA or r.get('study') != study for r in rows)):
        raise ValueError('Wrong CSV schema/study')
    selected = set(a.profiles.split(',')) if a.profiles else None
    rows = [r for r in rows if r['pattern'] == a.pattern and (selected is None or r['delay_profile'] in selected)]
    if not rows:
        raise ValueError('No rows match requested filters')
    for r in rows:
        if r.get('deadline_violations') not in ('', '0', None):
            raise ValueError('Invalid experiment: deadlines')
        if r.get('frame_cycles') and r.get('state_verified') != 'True':
            raise ValueError('Missing state validation')
    Path(a.out).mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    return rows

def save(fig, out, name):
    fig.tight_layout(pad=0.7)
    fig.savefig(Path(out) / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(Path(out) / (name + '.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)

def line(ax, rows, x, y, group='delay_profile', logy=False, percent=False):
    groups = sorted(set((r[group] for r in rows)))
    allx = sorted(set((int(r[x]) for r in rows)))
    for i, key in enumerate(groups):
        mean = []
        sd = []
        for xx in allx:
            subset = [r for r in rows if r[group] == key and int(r[x]) == xx]
            vals = [float(r[y]) for r in subset if r.get(y) not in ('', None)]
            if len({r['run'] for r in subset}) != len(subset):
                raise ValueError('Duplicate run; filter additional study dimensions')
            factor = 100 if percent else 1
            mean.append(np.mean(vals) * factor if vals else np.nan)
            sd.append(np.std(vals, ddof=1) * factor if len(vals) > 1 else 0.0)
        yy = np.asarray(mean)
        ss = np.asarray(sd)
        color = COLORS[i % len(COLORS)]
        label = 'delay ' + key if group == 'delay_profile' else key
        marker = MARKERS[i % len(MARKERS)]
        ax.plot(allx, yy, lw=1.2, label=label, color=color, marker=marker, ms=MARKER_SIZE, markevery=[], mec='white', mew=0.5)
        offset = (i - (len(groups) - 1) / 2) * MARKER_SPACING_PT
        transform = ax.transData + ScaledTranslation(offset / 72.0, 0, ax.figure.dpi_scale_trans)
        ax.plot(allx, yy, linestyle='none', marker=marker, ms=MARKER_SIZE, color=color, mec='white', mew=0.5, transform=transform, zorder=4, label='_nolegend_')
        lower = yy - ss
        if logy:
            lower = np.maximum(lower, 1e-08)
        ax.fill_between(allx, lower, yy + ss, alpha=0.12, color=color)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    if logy:
        ax.set_yscale('log')
    if x in ('neurons', 'max_sources_setting'):
        ax.set_xscale('log', base=2)
    ax.set_xticks(allx)
    ax.set_xticklabels([str(x) for x in allx])
    ax.margins(x=0.1)

def routing_axis(ax, a):
    if not 0 < a.route_ymin < a.route_ymax:
        raise ValueError('Require 0 < route-ymin < route-ymax')
    low, high = (ax.dataLim.y0, ax.dataLim.y1)
    if low < a.route_ymin or high > a.route_ymax:
        warnings.warn('Routing means/SD extend beyond the shared axis. Set the same --route-ymin and --route-ymax for all three timing plots.')
    ax.set_yscale('log')
    ax.set_ylim(a.route_ymin, a.route_ymax)

def scaling(a, study, x, mainstem, timestem):
    rows = load(a, study)
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.55), sharex=True)
    line(axes[0], rows, x, 'total_provisioned_bits', logy=True)
    axes[0].set_ylabel('Provisioned memory (bits)')
    line(axes[1], rows, x, 'frame_cycles', logy=True)
    axes[1].set_ylabel('Clocks / fixed frame')
    axes[0].legend(ncol=2)
    axes[1].set_xlabel('Neurons' if x == 'neurons' else 'Mesh side length')
    save(fig, a.out, mainstem)
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.55), sharex=True)
    for ax, key, label in zip(axes, ['routing_seconds', 'mapping_search_seconds'], ['Route construction (s)', 'Mapping search (s)']):
        line(ax, rows, x, key, logy=True)
        ax.set_ylabel(label)
    routing_axis(axes[0], a)
    axes[0].legend(ncol=2)
    axes[-1].set_xlabel('Neurons' if x == 'neurons' else 'Mesh side length')
    save(fig, a.out, timestem)
if __name__ == '__main__':
    a = parser(__doc__).parse_args()
    scaling(a, 'logical', 'neurons', 'logical_resources', 'logical_timing')
