"""Read prepared workload bundles and optionally replay reference neural execution."""
from pathlib import Path
import hashlib, json
import numpy as np
from scipy import sparse
SCHEMA = "chronoc-single-spike"

def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def spike_features(g):
    g = np.asarray(g)
    return np.concatenate([(g >= 0).astype(np.float32), np.where(g >= 0, 1 / (1 + np.maximum(g, 0)), 0).astype(np.float32)])


def run_reservoir(drive, model, on_emit=None):
    drive = np.asarray(drive, dtype=np.float32)
    if drive.ndim != 2 or not len(drive) or (not np.isfinite(drive).all()):
        raise ValueError('Bad drive')
    n = drive.shape[1]
    rec = sparse.csr_matrix(model['rec'])
    d = np.asarray(model['delays'], np.int64)
    if len(d) != n or rec.shape != (n, n):
        raise ValueError('Model dimension mismatch')
    mem = np.zeros(n, np.float32)
    fired = np.zeros(n, bool)
    g = np.full(n, -1, np.int64)
    pending = {}
    digest = hashlib.sha256()
    t = 0
    last_due = -1
    hard_limit = len(drive) + n * int(d.max()) + 2
    while t < len(drive) or pending:
        if t >= hard_limit:
            raise RuntimeError('One-shot tail failed to drain')
        current = drive[t].copy() if t < len(drive) else np.zeros(n, np.float32)
        due = pending.pop(t, [])
        if due:
            current += np.asarray(rec[sorted(due)].sum(axis=0)).ravel().astype(np.float32)
        mem = np.float32(model['beta']) * mem + current
        new = ~fired & (mem >= float(model['threshold']))
        for s in np.flatnonzero(new):
            s = int(s)
            g[s] = t
            if rec.indptr[s + 1] > rec.indptr[s]:
                h = t + int(d[s])
                pending.setdefault(h, []).append(s)
                last_due = max(last_due, h)
                if on_emit is not None:
                    on_emit(s, t, h)
        fired |= new
        mem[fired] = 0
        digest.update(mem.tobytes())
        digest.update(new.tobytes())
        t += 1
    return (g, spike_features(g), t, digest.hexdigest(), last_due)


def predict_features(features, model):
    f = np.asarray(features, dtype=np.float64)
    score = (f - model['readout_mean']) / model['readout_scale'] @ model['readout_coef'].T + model['readout_intercept']
    if score.shape[-1] == 1:
        ids = (score[..., 0] > 0).astype(int)
    else:
        ids = np.argmax(score, axis=-1)
    return np.asarray(model['classes'])[ids]


def load_bundle(path):
    root = Path(path)
    root = root.parent if root.is_file() else root
    m = json.loads((root / 'manifest.json').read_text())
    if m.get('schema') != SCHEMA or m.get('inferences_overlap') or m.get('max_spikes_per_neuron') != 1:
        raise ValueError('Not a nonoverlapping single-spike workload bundle')
    for f, digest in m['files'].items():
        if sha_file(root / f) != digest:
            raise ValueError('Bundle integrity failure: ' + f)
    model = dict(np.load(root / 'model.npz', allow_pickle=False))
    events = dict(np.load(root / 'events.npz', allow_pickle=False))
    g = events['generation']
    n = m['neurons']
    expected_id = hashlib.sha256(json.dumps(m['files'], sort_keys=True).encode()).hexdigest()
    if m['bundle_id'] != expected_id:
        raise ValueError('Bundle identity is inconsistent')
    if model['rec'].shape != (n, n) or model['delays'].shape != (n,) or np.any(model['delays'] < 1):
        raise ValueError('Invalid model graph/delays')
    if len(g) != m['export_samples'] or not len(g):
        raise ValueError('Export sample count mismatch')
    if len(events['horizon']) != len(g) or len(events['labels']) != len(g) or len(events['predictions']) != len(g):
        raise ValueError('Export array length mismatch')
    if not np.isclose(np.mean(events['labels'] == events['predictions']), m['export_reference_accuracy']):
        raise ValueError('Export accuracy metadata mismatch')
    if g.ndim != 2 or g.shape[1] != n or np.any(g < -1) or (not np.issubdtype(g.dtype, np.integer)):
        raise ValueError('Invalid generation matrix')
    if np.any((g >= events['horizon'][:, None]) & (g >= 0)):
        raise ValueError('Event outside inference')
    traces = json.loads((root / 'traces.json').read_text())
    if len(traces) != len(g) or [t['sample_id'] for t in traces] != events['sample_ids'].tolist():
        raise ValueError('Trace metadata/sample mismatch')
    return (root, m, model, events, traces)


def require_replay_traces(bundle):
    root, manifest, _, _, traces = bundle
    missing = [t['file'] for t in traces if not (root / t['file']).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{root}: {len(missing)} replay trace files missing. "
            f"Restore the matching traces/ directory for --verify-neural. First missing: {missing[0]}"
        )
