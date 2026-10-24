"""Compare deployments using prepared single-spike workload bundles."""
import argparse
from pathlib import Path
import subprocess
import sys
from bundle_io import load_bundle, require_replay_traces
from plot import export_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--workloads', help='Parent directory containing dataset bundles')
    p.add_argument('--datasets', nargs='+', choices=['ucihar', 'nmnist', 'demo'], default=['ucihar', 'nmnist'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--iterations', type=int, default=2000)
    p.add_argument('--route-attempts', type=int, default=100)
    p.add_argument('--verify-neural', action='store_true', help='Replay reference neural execution; requires traces/*.npz')
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    if len(set(a.datasets)) != len(a.datasets):
        p.error('Datasets must be distinct')
    if a.iterations < 0 or a.route_attempts < 1:
        p.error('Invalid iteration or routing budget')
    scripts = Path(__file__).resolve().parent
    repository = scripts.parents[1]
    out = Path(a.out).resolve()
    workloads, results = [], []
    for dataset in a.datasets:
        candidates = ([Path(a.workloads).resolve() / dataset] if a.workloads else
                      [repository / 'results' / 'workloads' / dataset, scripts.parent / 'workloads' / dataset])
        workload = next((path for path in candidates if (path / 'manifest.json').is_file()), None)
        if workload is None:
            raise FileNotFoundError('Prepared workload missing. Checked: ' + ', '.join(map(str, candidates)))
        bundle = load_bundle(workload)
        if bundle[1]['dataset'] != dataset:
            raise ValueError(f'{workload}: dataset metadata mismatch')
        if a.verify_neural:
            require_replay_traces(bundle)
        workloads.append(workload)
        print(f'{dataset}: using {workload}; neural replay={a.verify_neural}', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    verification = '--verify-neural' if a.verify_neural else '--no-verify-neural'
    for dataset, workload in zip(a.datasets, workloads):
        result = out / dataset
        subprocess.run([sys.executable, str(scripts / 'darsa.py'), '--workload', str(workload),
            '--out', str(result), '--seed', str(a.seed), '--iterations', str(a.iterations),
            '--route-attempts', str(a.route_attempts), verification], check=True)
        subprocess.run([sys.executable, str(scripts / 'baselines.py'), '--workload', str(workload),
            '--results', str(result), '--seed', str(a.seed), '--fabric', 'tdm', verification,
            *(['--quick'] if a.quick else [])], check=True)
        results.append(result)
    export_metrics(workloads, results, out / 'metrics.csv', allow_demo='demo' in a.datasets)
    print(f'Metrics: {out / "metrics.csv"}')


if __name__ == '__main__':
    main()
