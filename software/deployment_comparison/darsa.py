"""Construct fixed broadcast routes and evaluate DARSA periodic reservations."""
import argparse, collections, hashlib, json, math, random, time
from pathlib import Path
import sys
import numpy as np
from bundle_io import require_replay_traces, load_bundle, write_json, sha_file, run_reservoir, predict_features
RESULT_SCHEMA = 'chronoc-periodic-results'

def bits(n):
    return max(1, int(math.ceil(math.log2(max(2, int(n))))))

def make_workload(model):
    a = np.asarray(model['rec']) != 0
    n = len(a)
    delays = np.asarray(model['delays'], np.int64)
    names = [f'neuron_{i:06d}' for i in range(n)]
    return {'agents': [{'agent_id': f'a_{j}', 'destination': names[j], 'sources': [names[i] for i in np.flatnonzero(a[:, j])], 'delays': [int(delays[i]) for i in np.flatnonzero(a[:, j])]} for j in range(n)], 'axon_delays': {names[i]: int(delays[i]) for i in np.flatnonzero(a.any(axis=1))}, 'delay_model': 'axonal_per_source'}

def physical_links(grid):
    k = grid * grid
    edges = []
    for p in range(k):
        edges.extend([(p, k + p), (k + p, p)])
        r, c = divmod(p, grid)
        for rr, cc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= rr < grid and 0 <= cc < grid:
                edges.append((k + p, k + rr * grid + cc))
    return edges

def build_routes(grid, seed=42, attempts=100, depth=None):
    if grid == 1:
        return {'grid': 1, 'T': 1, 'starts': [0], 'arrivals': [[0]], 'trees': {'0': []}, 'routing_seconds': 0.0}
    start_clock = time.perf_counter()
    k = grid * grid
    depth = depth or max(32, 4 * k + 4 * grid)
    links = physical_links(grid)
    neighbors = collections.defaultdict(list)
    for u, v in links:
        neighbors[u].append(v)
    rng = random.Random(seed)
    best = None
    for attempt in range(attempts):
        used = set()
        trees = {}
        starts = [0] * k
        arrivals = np.full((k, k), -1, dtype=int)
        order = list(range(k))
        rng.shuffle(order)
        success = True
        for source in order:
            start = rng.randrange(max(1, depth - 2 * grid - 2))
            starts[source] = start
            nodes = {(source, start)}
            tree = []
            remaining = set(range(k)) - {source}
            while remaining:
                anchors = sorted((n for n in nodes if n[0] >= k or n == (source, start)))
                queue = collections.deque(anchors)
                parent = {a: None for a in anchors}
                found = None
                while queue and found is None:
                    u, t = queue.popleft()
                    if t + 1 >= depth:
                        continue
                    for v in neighbors[u]:
                        if v < k and v not in remaining:
                            continue
                        if u < k and (u, t) != (source, start):
                            continue
                        edge = (u, t, v, t + 1)
                        nxt = (v, t + 1)
                        if edge in used or nxt in parent:
                            continue
                        parent[nxt] = (u, t)
                        if v in remaining:
                            found = nxt
                            break
                        queue.append(nxt)
                if found is None:
                    success = False
                    break
                remaining.remove(found[0])
                arrivals[source, found[0]] = found[1]
                cur = found
                branch = []
                while parent[cur] is not None:
                    prev = parent[cur]
                    branch.append([prev[0], prev[1], cur[0], cur[1]])
                    cur = prev
                for edge in reversed(branch):
                    if tuple(edge) not in {tuple(x) for x in tree}:
                        tree.append(edge)
                    nodes.add((edge[0], edge[1]))
                    nodes.add((edge[2], edge[3]))
            if not success:
                break
            used.update(map(tuple, tree))
            trees[str(source)] = tree
            arrivals[source, source] = start
        if success:
            T = int(arrivals.max()) + 1
            candidate = {'grid': grid, 'T': T, 'starts': starts, 'arrivals': arrivals.tolist(), 'trees': trees}
            validate_routes(candidate)
            if best is None or T < best['T']:
                best = candidate
    if best is None:
        raise RuntimeError('No broadcast schedule found; increase --route-depth or --route-attempts')
    best['routing_seconds'] = time.perf_counter() - start_clock
    best['search_attempts'] = attempts
    best['search_depth'] = depth
    best['seed'] = seed
    best['route_id'] = hashlib.sha256(json.dumps({k: v for k, v in best.items() if k != 'routing_seconds'}, sort_keys=True).encode()).hexdigest()
    return best

def validate_routes(route):
    n = int(route['grid']) ** 2
    T = int(route['T'])
    links = set(physical_links(route['grid']))
    used = set()
    for src in range(n):
        edges = [tuple(x) for x in route['trees'][str(src)]]
        starts = route['starts'][src]
        nodes = {(src, starts)}
        pending = edges.copy()
        destinations = []
        if len(edges) != len(set(edges)):
            raise ValueError('Duplicate route edge')
        if used.intersection(edges):
            raise ValueError('Conflicting link-timeslot resources')
        used.update(edges)
        while pending:
            remaining = []
            progress = False
            for u, t, v, t2 in pending:
                if (u, v) not in links or t2 != t + 1 or (not 0 <= t < t2 < T):
                    raise ValueError('Invalid time-space edge')
                if (u, t) not in nodes:
                    remaining.append((u, t, v, t2))
                    continue
                if (v, t2) in nodes:
                    raise ValueError('Reconvergent multicast route')
                nodes.add((v, t2))
                progress = True
                if v < n:
                    destinations.append(v)
                    if route['arrivals'][src][v] != t2:
                        raise ValueError('Arrival matrix mismatch')
            if not progress and remaining:
                raise ValueError('Unreachable route edge')
            pending = remaining
        if sorted(destinations) != [d for d in range(n) if d != src]:
            raise ValueError('Incomplete/duplicate full-grid delivery')
    return True

def calendar(mapping, adj, delays, pes, immediate=False):
    mapping = np.asarray(mapping, int)
    sources = np.flatnonzero(np.asarray(adj).any(axis=1))
    P = np.ones(pes, dtype=int)
    c = np.zeros(pes, dtype=int)
    phase = np.full(len(mapping), -1, int)
    slot = phase.copy()
    for v in range(pes):
        ids = sources[mapping[sources] == v]
        if len(ids):
            P[v] = 1 if immediate else int(np.min(delays[ids]))
            c[v] = math.ceil(len(ids) / P[v])
            phase[ids] = np.arange(len(ids)) % P[v]
            slot[ids] = np.arange(len(ids)) // P[v]
    return {'periods': P, 'counts': c, 'phase': phase, 'subslot': slot, 'omega': max(1, int(c.max()))}

def validate_mapping(mapping, n, pes, capacity):
    m = np.asarray(mapping)
    if m.shape != (n,) or not np.issubdtype(m.dtype, np.integer) or np.any((m < 0) | (m >= pes)):
        raise ValueError('Incomplete/invalid mapping')
    if np.bincount(m, minlength=pes).max() > capacity:
        raise ValueError('Core capacity exceeded')

def memory_model(mapping, adj, delays, cal, route, max_steps, flit_bits=32):
    n = len(mapping)
    pes = int(route['grid']) ** 2
    idbits = bits(n)
    tagbits = bits(max_steps + 1)
    event_bits = idbits + tagbits + 1
    required_bits = event_bits + bits(pes)
    if required_bits > flit_bits:
        raise ValueError(f'{flit_bits}-bit flit too short; need {required_bits} including destination and timing metadata')
    sources = np.flatnonzero(adj.any(axis=1))
    dest_caps = np.zeros(pes, int)
    src_caps = np.zeros(pes, int)
    for s in sources:
        src_caps[mapping[s]] += 1
        targets = np.unique(mapping[np.flatnonzero(adj[s])])
        dest_caps[targets] += 1
    source = int(src_caps.sum()) * event_bits
    destination = int(dest_caps.sum()) * event_bits
    ni = 2 * pes * flit_bits
    calendar_bits = 0
    for v in range(pes):
        calendar_bits += int(cal['periods'][v] * cal['counts'][v]) * (idbits + 1)
        calendar_bits += bits(cal['periods'][v]) + bits(cal['omega'])
    network = 2 * cal['omega'] * len(physical_links(route['grid'])) * flit_bits
    network_model = 'two omega-word slot banks per receiving channel (provisioning assumption; not original RTL PPA)'
    out = {'event_entry_bits': event_bits, 'transport_required_bits': required_bits, 'flit_bits': flit_bits, 'source_retention_bits': source, 'destination_retention_bits': destination, 'calendar_bits': calendar_bits, 'ni_register_bits': ni, 'network_storage_bits': network, 'source_capacity_per_core': src_caps.tolist(), 'destination_capacity_per_core': dest_caps.tolist(), 'network_storage_model': network_model, 'scope': 'communication storage; excludes neuron/synaptic parameters, external input/readout, and combinational logic'}
    out['total_provisioned_bits'] = sum((out[x] for x in ['source_retention_bits', 'destination_retention_bits', 'calendar_bits', 'ni_register_bits', 'network_storage_bits']))
    return out

def optimize_mapping(adj, delays, route, capacity, max_steps, flit_bits, seed, iterations, alpha):
    rng = np.random.default_rng(seed)
    n = len(adj)
    pes = route['grid'] ** 2
    mapping = rng.permutation(np.arange(n) % pes)
    validate_mapping(mapping, n, pes, capacity)

    def raw(m):
        cal = calendar(m, adj, delays, pes)
        mem = memory_model(m, adj, delays, cal, route, max_steps, flit_bits)
        return (route['T'] * cal['omega'], mem['total_provisioned_bits'])
    L0, B0 = raw(mapping)

    def score(m):
        L, B = raw(m)
        return alpha * L / max(1, L0) + (1 - alpha) * B / max(1, B0)
    val = score(mapping)
    best = mapping.copy()
    bestval = val
    trace = []
    start = time.perf_counter()
    for i in range(iterations):
        cand = mapping.copy()
        a = int(rng.integers(n))
        if rng.random() < 0.5 and n > 1:
            b = int(rng.integers(n))
            cand[a], cand[b] = (cand[b], cand[a])
        else:
            dest = int(rng.integers(pes))
            if np.sum(cand == dest) >= capacity:
                continue
            cand[a] = dest
        nv = score(cand)
        temp = 0.15 * (0.001 / 0.15) ** (i / max(1, iterations - 1))
        if nv <= val or rng.random() < math.exp(min(0, (val - nv) / temp)):
            mapping, val = (cand, nv)
        if val < bestval:
            best, bestval = (mapping.copy(), val)
        if i % max(1, iterations // 20) == 0:
            trace.append({'iteration': i, 'best_cost': float(bestval)})
    return (best, {'mapping_seconds': time.perf_counter() - start, 'iterations': iterations, 'seed': seed, 'alpha': alpha, 'L0': L0, 'B0': B0, 'objective': 'alpha*L/L0+(1-alpha)*B/B0', 'progress': trace, 'best_cost': bestval})

def interval_peaks(intervals, pes):
    changes = [collections.defaultdict(int) for _ in range(pes)]
    for pe, start, end in intervals:
        if start < end:
            changes[int(pe)][int(start)] += 1
            changes[int(pe)][int(end)] -= 1
    peaks = []
    for events in changes:
        count = peak = 0
        for t in sorted(events):
            count += events[t]
            peak = max(peak, count)
        if count != 0:
            raise AssertionError('Unbalanced intervals')
        peaks.append(peak)
    return peaks

def evaluate(bundle, mapping, route, capacity, method, immediate=False, flit_bits=32, compile_info=None, verify_neural=True):
    root, manifest, model, ev, traces = bundle
    adj = model['rec'] != 0
    delays = model['delays']
    pes = route['grid'] ** 2
    validate_mapping(mapping, len(adj), pes, capacity)
    validate_routes(route)
    cal = calendar(mapping, adj, delays, pes, immediate)
    bound = manifest['observation_steps_max'] + len(adj) * int(delays.max()) + 2
    mem = memory_model(mapping, adj, delays, cal, route, bound, flit_bits)
    rows = []
    dest_peak = np.zeros(pes, int)
    source_peak = dest_peak.copy()
    observed_pes = {s: np.unique(mapping[np.flatnonzero(adj[s])]).tolist() for s in np.flatnonzero(adj.any(axis=1))}
    all_cycles = []
    validated = correct = 0
    total_events = total_deliveries = 0
    for i, g in enumerate(ev['generation']):
        H = int(ev['horizon'][i])
        src_intervals = []
        dest_intervals = []
        ledger = {}
        boarded = {}
        frame_cycles = np.full(H, route['T'] * cal['omega'], dtype=np.int64)
        injection_occupancy = collections.Counter()
        physical_arrivals = {}
        for s in observed_pes:
            if g[s] < 0:
                continue
            gen = int(g[s])
            h = gen + int(delays[s])
            pe = int(mapping[s])
            f = gen + (int(cal['phase'][s]) - gen) % int(cal['periods'][pe])
            if not gen <= f < h or h >= H:
                raise AssertionError('Deadline or drained horizon violation')
            key = (pe, f, int(cal['subslot'][s]))
            injection_occupancy[key] += 1
            if injection_occupancy[key] != 1:
                raise AssertionError('Reservation collision')
            if cal['subslot'][s] >= cal['omega']:
                raise AssertionError('Subslot exceeds expansion')
            boarded[s] = f
            src_intervals.append((pe, gen, f + 1))
            for dst in observed_pes[s]:
                if (s, dst) in ledger:
                    raise AssertionError('Duplicate source event at target core')
                ledger[s, dst] = (gen, f + 1, h)
                dest_intervals.append((dst, f + 1, h))
                within = int(route['arrivals'][pe][dst]) * cal['omega'] + int(cal['subslot'][s])
                if within >= frame_cycles[f]:
                    raise AssertionError('Route leaves boarding frame')
                physical_arrivals[s, dst] = within
        if set(physical_arrivals) != set(ledger):
            raise AssertionError('Delivery-set mismatch')
        sp = interval_peaks(src_intervals, pes)
        dp = interval_peaks(dest_intervals, pes)
        if np.any(np.array(sp) > mem['source_capacity_per_core']) or np.any(np.array(dp) > mem['destination_capacity_per_core']):
            raise AssertionError('Provisioned event storage exceeded')
        source_peak = np.maximum(source_peak, sp)
        dest_peak = np.maximum(dest_peak, dp)
        prediction = None
        if verify_neural:
            trace = traces[i]
            path = root / trace['file']
            if sha_file(path) != trace['sha256']:
                raise ValueError('Trace checksum failure')
            with np.load(path, allow_pickle=False) as rec:
                if not np.array_equal(rec['generation'], g):
                    raise ValueError('Trace/index mismatch')

                def on_emit(s, gen, h):
                    if g[s] != gen:
                        raise AssertionError('Deployed generation differs')
                    for dst in observed_pes[s]:
                        tag, ready, due = ledger[s, dst]
                        if tag != gen or due != h or ready > due:
                            raise AssertionError('Incorrect logical delivery')
                gg, feat, hh, state_hash, _ = run_reservoir(rec['drive'], model, on_emit)
                if not np.array_equal(gg, g) or hh != H or state_hash != trace['state_sha256']:
                    raise AssertionError('Neural-state/spike-train mismatch')
                prediction = int(predict_features(feat, model))
                if prediction != int(ev['predictions'][i]):
                    raise AssertionError('Prediction mismatch')
                correct += int(prediction == int(ev['labels'][i]))
                validated += 1
        total_events += len(boarded)
        total_deliveries += len(ledger)
        all_cycles.extend(map(int, frame_cycles))
        observation = int(traces[i]['observation_steps'])
        rows.append({'sample_id': int(ev['sample_ids'][i]), 'logical_frames': H, 'frame_cycles': frame_cycles.tolist(), 'communication_cycles': int(frame_cycles.sum()), 'tail_communication_cycles': int(frame_cycles[observation:].sum()), 'source_peak_entries': sp, 'destination_peak_entries': dp, 'fired_transmitting_sources': len(boarded), 'target_core_deliveries': len(ledger), 'deployed_prediction': prediction, 'label': int(ev['labels'][i])})
    compile_info = compile_info or {}
    return {'schema': RESULT_SCHEMA, 'bundle_id': manifest['bundle_id'], 'dataset': manifest['dataset'], 'synthetic_demo': manifest['synthetic_demo'], 'method': method, 'fabric': 'tdm', 'transport_model': 'fixed space-time slot/subslot model', 'validation_level': ('software event/neural replay; not RTL or silicon' if verify_neural else 'stored-event communication checks; neural replay not performed'), 'boarding_policy': 'immediate' if immediate else 'static_periodic', 'grid': route['grid'], 'core_capacity': capacity, 'mapping': list(map(int, mapping)), 'route_id': route.get('route_id', 'single-core'), 'T': route['T'], 'periods': cal['periods'].tolist(), 'phase': cal['phase'].tolist(), 'subslot': cal['subslot'].tolist(), 'omega': cal['omega'], 'memory': mem, 'compilation': compile_info, 'mean_frame_cycles': float(np.mean(all_cycles)), 'p99_frame_cycles': float(np.quantile(all_cycles, 0.99)), 'mean_inference_communication_cycles': float(np.mean([r['communication_cycles'] for r in rows])), 'observed_source_peak_entries_per_core': source_peak.tolist(), 'observed_destination_peak_entries_per_core': dest_peak.tolist(), 'deadline_violations': 0, 'reservation_collisions': 0, 'events_checked': total_events, 'deliveries_checked': total_deliveries, 'neural_replay_verified_samples': validated, 'deployed_accuracy': correct / validated if validated else None, 'export_reference_accuracy': manifest['export_reference_accuracy'], 'samples': len(rows), 'per_inference': rows}

def save_result(out, result):
    slug = result['method'].lower().replace(' ', '_').replace('/', '_')
    write_json(Path(out) / (slug + '.json'), result)

def self_test():
    adj = np.ones((10, 10), bool)
    np.fill_diagonal(adj, False)
    m = np.zeros(10, int)
    d = np.full(10, 5)
    cal = calendar(m, adj, d, 1)
    assert cal['omega'] == 2
    for s in range(10):
        for g in range(20):
            f = g + (cal['phase'][s] - g) % 5
            assert g <= f and f + 1 <= g + 5
    rng = np.random.default_rng(91)
    for _ in range(100):
        gen = rng.integers(0, 20, 10)
        used = set()
        for s, g in enumerate(gen):
            f = int(g + (cal['phase'][s] - g) % 5)
            key = (f, int(cal['subslot'][s]))
            assert key not in used
            used.add(key)
    assert calendar(m, adj, np.ones(10, int), 1)['omega'] == 10
    assert calendar(m, adj, d, 1, True)['omega'] == 10
    from bundle_io import run_reservoir
    model = {'rec': np.array([[0, 2], [0, 0]], np.float32), 'delays': np.array([3, 2]), 'beta': np.float32(0.9), 'threshold': np.float32(1)}
    g, _, H, _, _ = run_reservoir(np.array([[2, 0]], np.float32), model)
    assert g.tolist() == [0, 3] and H == 4
    model['rec'][:] = 0
    g, _, _, _, _ = run_reservoir(np.full((20, 2), 2, np.float32), model)
    assert g.tolist() == [0, 0]

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workload')
    p.add_argument('--out')
    p.add_argument('--grid', type=int, default=4)
    p.add_argument('--capacity', type=int)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--iterations', type=int, default=2000)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--route-attempts', type=int, default=100)
    p.add_argument('--route-depth', type=int)
    p.add_argument('--route-file')
    p.add_argument('--flit-bits', type=int, default=32)
    p.add_argument('--verify-neural', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test:
        self_test()
        return
    if not a.workload or not a.out:
        p.error('--workload and --out required')
    if a.grid < 1 or a.iterations < 0 or (not 0 <= a.alpha <= 1):
        p.error('Invalid grid/iterations/alpha')
    bundle = load_bundle(a.workload)
    if a.verify_neural:
        require_replay_traces(bundle)
    root, manifest, model, events, traces = bundle
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    route = json.loads(Path(a.route_file).read_text()) if a.route_file else build_routes(a.grid, a.seed, a.route_attempts, a.route_depth)
    if route['grid'] != a.grid:
        raise ValueError('Route/grid mismatch')
    validate_routes(route)
    write_json(out / 'route.json', route)
    n = manifest['neurons']
    capacity = a.capacity or max(1, 2 * math.ceil(n / (a.grid * a.grid)))
    bound = manifest['observation_steps_max'] + n * int(model['delays'].max()) + 2
    mapping, info = optimize_mapping(model['rec'] != 0, model['delays'], route, capacity, bound, a.flit_bits, a.seed, a.iterations, a.alpha)
    info['routing_seconds'] = route['routing_seconds']
    info['total_seconds'] = info['routing_seconds'] + info['mapping_seconds']
    for immediate, name in [(False, 'DARSA'), (True, 'DARSA_immediate')]:
        result = evaluate(bundle, mapping, route, capacity, name, immediate=immediate, flit_bits=a.flit_bits, compile_info=info, verify_neural=a.verify_neural)
        result['provenance'] = 'proposed static periodic DARSA; deterministic balanced NI calendar'
        result['compilation_reused'] = immediate
        save_result(out, result)
    write_json(out / 'experiment.json', {'bundle_id': manifest['bundle_id'], 'workload': str(root.resolve()), 'grid': a.grid, 'capacity': capacity, 'flit_bits': a.flit_bits, 'route_file': 'route.json', 'seed': a.seed, 'mapping': mapping.tolist()})
if __name__ == '__main__':
    main()
