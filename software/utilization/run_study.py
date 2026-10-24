"""Run the source fan-in utilization study."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from scaling import *

def main():
    p = common_parser(__doc__)
    p.add_argument('--neurons', type=int, default=1024)
    p.add_argument('--source-counts', type=ints, default=ints('1,2,4,8,16,32,64,128'))
    a = p.parse_args()
    validate_args(a)
    rows = []
    if max(a.source_counts) >= a.neurons:
        raise ValueError('max incoming sources must be <= N-1')
    for run in range(a.runs):
        route = route_for(4, run, a)
        for maximum in a.source_counts:
            adj = graph(a.neurons, maximum, a.seed + run)
            for lo, hi in a.delay_profiles:
                delays = delays_for(a.neurons, lo, hi, a.seed + 1000 + run)
                cap, m, info = run_mapping(adj, delays, route, a, run)
                meta = {'study': 'utilization', 'run': run, 'delay_profile': f'{lo}:{hi}', 'method': 'DARSA', 'max_sources_setting': maximum, 'observed_max_sources': int(adj.sum(axis=0).max())}
                r, detail = evaluate_case(adj, delays, m, route, cap, a, meta, info)
                rows += r
                write_csv(Path(a.out) / 'metrics.csv', rows)
if __name__ == '__main__':
    main()
