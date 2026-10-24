"""Plot logical-workload resource and timing figures."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from plot_scaling import parser, scaling
if __name__ == '__main__':
    a = parser(__doc__).parse_args()
    scaling(a, 'logical', 'neurons', 'logical_resources', 'logical_timing')
