# Single-spike edge workloads

Prepared UCI-HAR activity-recognition and N-MNIST event-vision bundles describe
128-neuron recurrent LIF reservoirs with fixed random weights, fixed axonal
delays of 1--20 timesteps, and trained logistic readouts. Each reservoir neuron
fires at most once per sample; delayed activity does not overlap across samples.

Each dataset directory contains:

- `manifest.json`: model settings, dataset statistics, and reference accuracy.
- `model.npz`: reservoir weights, delays, and readout parameters.
- `events.npz`: firing times, inference horizons, labels, and reference predictions.
- `traces.json`: sample metadata and references to optional neural replay files.

These files support the [deployment comparison](../deployment_comparison/).
