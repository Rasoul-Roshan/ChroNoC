"""Run physical-grid scaling with a fixed logical workload."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from scaling import *

def main():
    p = common_parser(__doc__)
    p.add_argument('--grids', type=ints, default=ints('2,3,4,5,6'))
    p.add_argument('--neurons', type=int, default=64)
    a = p.parse_args()
    validate_args(a)
    rows = []
    if min(a.grids) < 2:
        raise ValueError('Mesh grid sides must be >=2')
    for run in range(a.runs):
        adj = graph(a.neurons, a.max_sources, a.seed + run)
        for grid in a.grids:
            route = route_for(grid, run, a)
            for lo, hi in a.delay_profiles:
                delays = delays_for(a.neurons, lo, hi, a.seed + 1000 + run)
                cap, m, info = run_mapping(adj, delays, route, a, run)
                meta = {'study': 'grid', 'run': run, 'delay_profile': f'{lo}:{hi}', 'method': 'DARSA'}
                r, detail = evaluate_case(adj, delays, m, route, cap, a, meta, info)
                rows += r
                write_csv(Path(a.out) / 'metrics.csv', rows)
if __name__ == '__main__':
    main()
