"""Plot post-mapping structural utilization versus maximum source count."""
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SCHEMA = 'chronoc-scaling'
COLORS = ['red']

def parser(doc):
    p = argparse.ArgumentParser(description=doc)
    p.add_argument('--csv', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--pattern', default='simultaneous', choices=['simultaneous', 'staggered', 'silent', 'after_opportunity', 'all_silent'])
    p.add_argument('--profiles', default='', help='Comma-separated delay profiles, e.g. 5:5,5:20; default all')
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
        ax.plot(allx, yy, linestyle='-', marker=None, lw=1.5, label=label, color=color)
        lower = yy - ss
        if logy:
            lower = np.maximum(lower, 1e-08)
        ax.fill_between(allx, lower, yy + ss, alpha=0.12, color=color)
    ax.grid(False, which='both')
    ax.set_axisbelow(True)
    if logy:
        ax.set_yscale('log')
    if x in ('neurons', 'max_sources_setting'):
        ax.set_xscale('log', base=2)
    ax.set_xticks(allx)
    ax.set_xticklabels([str(x) for x in allx])
EXPORT_DPI = 600
FIGURE_SIZE = (1779 / EXPORT_DPI, 1156 / EXPORT_DPI * (2 / 3))
OUTPUT_STEM = 'structural_utilization'

def configure_style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6, 'axes.labelsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.linewidth': 0.7, 'axes.grid': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': None})

def add_callouts(ax):
    x, y = ax.lines[0].get_data()
    valid = np.flatnonzero(np.isfinite(y))
    if not len(valid):
        return
    first, last = (valid[0], valid[-1])
    middle = valid[np.argmin(np.abs(np.asarray(y)[valid] - 50))]
    for idx in dict.fromkeys([first, middle, last]):
        xv, yv = (x[idx], y[idx])
        if idx == first:
            text = f'Sparse fan-in\n{yv:.1f}% useful'
            offset, ha, va = ((2, 16), 'left', 'bottom')
        elif idx == last:
            text = f'Dense fan-in\n{yv:.1f}% useful'
            offset, ha, va = ((-8, -28), 'right', 'top')
        else:
            text = f'{yv:.1f}% useful\nat $S_{{\\max}}={int(xv)}$'
            offset, ha, va = ((0, 14), 'center', 'bottom')
        ax.annotate(text, xy=(xv, yv), xytext=offset, textcoords='offset points', ha=ha, va=va, fontsize=4.5, linespacing=1.05, color='black', arrowprops=dict(arrowstyle='->', color='0.25', lw=0.7, shrinkA=2, shrinkB=2, mutation_scale=5), bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='0.75', linewidth=0.45, alpha=0.92), annotation_clip=False)

def main():
    p = parser(__doc__)
    p.add_argument('--no-annotations', action='store_true', help='Omit the data-derived arrow callouts')
    a = p.parse_args()
    a.profiles = '1:20'
    rows = load(a, 'utilization')
    configure_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    line(ax, rows, 'max_sources_setting', 'structural_utilization', percent=True)
    ax.set_ylabel('Post-mapping\nutilization (%)', labelpad=3)
    ax.set_xlabel('Maximum sources per neuron', labelpad=2)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 50, 100])
    ax.tick_params(which='major', length=3, width=0.65, pad=2)
    ax.margins(x=0.035)
    if not a.no_annotations:
        add_callouts(ax)
    fig.tight_layout(pad=0.3)
    for ext in ('pdf', 'png'):
        output = Path(a.out) / (OUTPUT_STEM + '.' + ext)
        fig.savefig(output, dpi=EXPORT_DPI, facecolor='white', bbox_inches=None)
    plt.close(fig)
if __name__ == '__main__':
    main()
