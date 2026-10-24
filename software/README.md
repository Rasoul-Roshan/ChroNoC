# Software

Each deployment/scaling study has a `run_study.py`, a `plot.py`, and reference figures. Commands in
this repository run from its root. The shared `common/` directory contains the
scaling evaluator and plotting helpers.

## Deployment and scaling studies

| Study | Run | Plot |
| --- | --- | --- |
| Deployment comparison | `python software/deployment_comparison/run_study.py --out results/deployment_comparison` | `python software/deployment_comparison/plot.py --csv results/deployment_comparison/metrics.csv --out results/deployment_comparison/figures` |
| Logical scaling | `python software/logical_scaling/run_study.py --out results/logical_scaling` | `python software/logical_scaling/plot.py --csv results/logical_scaling/metrics.csv --out results/logical_scaling/figures` |
| Physical scaling | `python software/physical_scaling/run_study.py --out results/physical_scaling` | `python software/physical_scaling/plot.py --csv results/physical_scaling/metrics.csv --out results/physical_scaling/figures` |
| Utilization | `python software/utilization/run_study.py --out results/utilization` | `python software/utilization/plot.py --csv results/utilization/metrics.csv --out results/utilization/figures` |
| Compilation | `python software/compilation/run_study.py --out results/compilation --delay-profiles 5:20` | `python software/compilation/plot.py --csv results/compilation/metrics.csv --out results/compilation/figures --profiles 5:20` |

