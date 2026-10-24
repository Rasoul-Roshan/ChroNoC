"""Plot the deployment comparison from compact CSV metrics."""
import argparse, csv, json, math
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from bundle_io import load_bundle, sha_file
from darsa import RESULT_SCHEMA
METHODS = ['darsa', 'darsa_immediate', 'jin', 'xiao', 'neumap', 'spinemap', 'sneap']
LABELS = ['DARSA periodic', 'DARSA immediate', 'Jin adaptation', 'Xiao adaptation', 'NeuMap adaptation', 'SpiNeMap adaptation', 'SNEAP adaptation']
COLORS = ['#333333', '#999999', '#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9']
METRICS = [('mean_frame_cycles', 'Mean frame cycles'), ('mean_inference_communication_cycles', 'Inference comm. cycles'), ('memory_bits', 'Provisioned storage (bits)'), ('compile_seconds', 'Compilation (s)')]

def load_results(workload_path, result_path, allow_demo):
    bundle = load_bundle(workload_path)
    manifest = bundle[1]
    root = Path(result_path)
    groups = {}
    hashes = {}
    if manifest['synthetic_demo'] and (not allow_demo):
        raise ValueError('Synthetic data require --allow-demo')
    for mode, suffix in [('common-tdm', 'tdm')]:
        rows = []
        for name in METHODS:
            filename = name + '.json' if name.startswith('darsa') else name + '_' + suffix + '.json'
            path = root / filename
            if not path.exists():
                raise FileNotFoundError(f'{path}: run all counterparts with --fabric tdm')
            r = json.loads(path.read_text())
            hashes[str(path)] = sha_file(path)
            if r.get('schema') != RESULT_SCHEMA or r['bundle_id'] != manifest['bundle_id']:
                raise ValueError('Stale or mismatched result: ' + str(path))
            if r['samples'] != manifest['export_samples']:
                raise ValueError('Complete event evaluation required: ' + str(path))
            if r['deadline_violations'] or r['reservation_collisions']:
                raise ValueError('Invalid timing result')
            verified = r['neural_replay_verified_samples']
            if verified not in (0, r['samples']):
                raise ValueError('Partial neural replay: ' + str(path))
            r['neural_replay_verified'] = verified == r['samples']
            if verified:
                if not math.isclose(r['deployed_accuracy'], manifest['export_reference_accuracy'], abs_tol=1e-12):
                    raise ValueError('Reference/deployed accuracy mismatch')
            elif r['deployed_accuracy'] is not None:
                raise ValueError('Unverified run must not report deployed accuracy')
            if not all((math.isfinite(r[x]) and r[x] >= 0 for x in ['mean_frame_cycles', 'mean_inference_communication_cycles'])):
                raise ValueError('Invalid numerical metrics')
            expected_fabric = 'tdm'
            if r['fabric'] != expected_fabric:
                raise ValueError('Fabric/comparison mix-up')
            r['memory_bits'] = r['memory']['total_provisioned_bits']
            r['compile_seconds'] = r['compilation']['total_seconds']
            rows.append(r)
        if len({r['neural_replay_verified'] for r in rows}) != 1:
            raise ValueError('Mixed neural verification modes')
        d, imm = rows[:2]
        if d['mapping'] != imm['mapping'] or d['route_id'] != imm['route_id'] or any((p != 1 for p in imm['periods'])):
            raise ValueError('Immediate baseline must reuse DARSA mapping/route, with P=1')
        if not imm.get('compilation_reused'):
            raise ValueError('Immediate compilation must be marked as reused')
        for r in rows:
            if r['grid'] != d['grid'] or r['core_capacity'] != d['core_capacity']:
                raise ValueError('Unequal hardware capacity')
            if r['memory']['flit_bits'] != d['memory']['flit_bits'] or r['memory']['event_entry_bits'] != d['memory']['event_entry_bits']:
                raise ValueError('Unequal entry formats')
            if r['route_id'] != d['route_id'] or r['T'] != d['T']:
                raise ValueError('Different TDM substrate')
            if not r['method'].lower().startswith('darsa') and 'provenance' not in r:
                raise ValueError('Missing mapper provenance')
            if [x['sample_id'] for x in r['per_inference']] != [x['sample_id'] for x in d['per_inference']]:
                raise ValueError('Different evaluation samples')
        groups[mode] = rows
    return (manifest, groups, hashes)

def plot(groups, out, mode, absolute=False, demo=False):
    nr = len(groups)
    fig, axes = plt.subplots(nr, 4, figsize=(8.8, max(1.6, 1.1 * nr + 0.3)), squeeze=False)
    for row, (manifest, data, _) in enumerate(groups):
        entries = data[mode]
        for col, (metric, title) in enumerate(METRICS):
            ax = axes[row, col]
            values = np.array([r[metric] for r in entries], float)
            if not absolute:
                if values[0] <= 0:
                    raise ValueError('Cannot normalize to a zero DARSA measurement; use --absolute')
                values /= values[0]
            bars = ax.bar(np.arange(len(values)), values, color=COLORS, width=0.78, edgecolor='white', linewidth=0.35)
            bars[1].set_hatch('///')
            ax.set_xticks([])
            ax.grid(axis='y', alpha=0.2)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(title, fontsize=8)
            if col == 0:
                ax.set_ylabel(manifest['dataset'].upper() + ('\nrelative to DARSA' if not absolute else ''), fontsize=8)
            if not absolute:
                ax.axhline(1, color='#555555', linewidth=0.5, linestyle=':')
    handles = [Patch(facecolor=c, label=l, hatch='///' if i == 1 else '') for i, (c, l) in enumerate(zip(COLORS, LABELS))]
    fig.legend(handles=handles, loc='center left', ncol=1, fontsize=6.8, frameon=False, bbox_to_anchor=(0.825, 0.5), borderaxespad=0, handletextpad=0.4, labelspacing=0.4, title='SYNTHETIC DEMO' if demo else None)
    caption = 'Shared TDM substrate; counterparts use immediate reservations'
    if demo:
        caption = 'SYNTHETIC DEMO — ' + caption
    fig.tight_layout(rect=[0, 0, 0.82, 1], pad=0.2, w_pad=0.6, h_pad=0.5)
    stem = 'deployment_comparison'
    fig.savefig(out / (stem + '.pdf'), bbox_inches='tight', pad_inches=0.02)
    fig.savefig(out / (stem + '.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    return stem

def export_metrics(workloads, results, output, allow_demo=False):
    data = [load_results(w, r, allow_demo) for w, r in zip(workloads, results)]
    fields = ['dataset', 'method', 'synthetic_demo', 'neural_replay_verified'] + [m[0] for m in METRICS]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for manifest, groups, _ in data:
            for row in groups['common-tdm']:
                writer.writerow({**{key: row[key] for key, _ in METRICS}, 'dataset': manifest['dataset'], 'method': row['method'], 'synthetic_demo': manifest['synthetic_demo'], 'neural_replay_verified': row['neural_replay_verified']})

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--absolute', action='store_true')
    p.add_argument('--allow-demo', action='store_true')
    a = p.parse_args()
    with open(a.csv, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty CSV')
    data = []
    for dataset in dict.fromkeys((row['dataset'] for row in rows)):
        selected = [r for r in rows if r['dataset'] == dataset]
        if len(selected) != len(METHODS):
            raise ValueError('Expected one row per method and dataset')
        by_method = {r['method'].lower(): r for r in selected}
        if set(by_method) != {m if m.startswith('darsa') else m + '_tdm' for m in METHODS}:
            raise ValueError('Missing or duplicate methods')
        ordered = [by_method[m if m.startswith('darsa') else m + '_tdm'] for m in METHODS]
        for row in ordered:
            for metric, _ in METRICS:
                row[metric] = float(row[metric])
                if not math.isfinite(row[metric]) or row[metric] < 0:
                    raise ValueError('Invalid metric: ' + metric)
        demo = any((r['synthetic_demo'] == 'True' for r in ordered))
        if demo and (not a.allow_demo):
            raise ValueError('Synthetic data require --allow-demo')
        data.append(({'dataset': dataset, 'synthetic_demo': demo}, {'common-tdm': ordered}, {}))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'text.usetex': False})
    plot(data, out, 'common-tdm', a.absolute, any((m['synthetic_demo'] for m, _, _ in data)))
if __name__ == '__main__':
    main()
