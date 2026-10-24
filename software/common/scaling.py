"""Shared fixed-route DARSA evaluation for workload, mesh, and utilization studies."""
import argparse, collections, csv, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
SCHEMA = 'chronoc-scaling'

def bits(n):
    return max(1, int(math.ceil(math.log2(max(2, int(n))))))

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

def memory_model(mapping, adj, delays, cal, route, max_steps, flit_bits=32, fabric='tdm', router_depth=16):
    n = len(mapping)
    pes = int(route['grid']) ** 2
    idbits = bits(n)
    tagbits = bits(max_steps + 1)
    event_bits = idbits + tagbits + 1
    packet_bits = event_bits + bits(pes)
    if packet_bits > flit_bits:
        raise ValueError(f'{flit_bits}-bit flit too short; need {packet_bits} including destination and timing metadata')
    sources = np.flatnonzero(adj.any(axis=1))
    dest_caps = np.zeros(pes, int)
    src_caps = np.zeros(pes, int)
    packets = np.zeros(pes, int)
    for s in sources:
        src_caps[mapping[s]] += 1
        targets = np.unique(mapping[np.flatnonzero(adj[s])])
        dest_caps[targets] += 1
        packets[mapping[s]] += int(np.sum(targets != mapping[s]))
    source = int(src_caps.sum()) * event_bits
    destination = int(dest_caps.sum()) * event_bits
    ni = 2 * pes * flit_bits
    calendar_bits = 0
    if fabric == 'tdm':
        for v in range(pes):
            calendar_bits += int(cal['periods'][v] * cal['counts'][v]) * (idbits + 1)
            calendar_bits += bits(cal['periods'][v]) + bits(cal['omega'])
        network = 2 * cal['omega'] * len(physical_links(route['grid'])) * flit_bits
        queue = 0
        network_model = 'two omega-word slot banks per receiving channel (provisioning assumption; not original RTL PPA)'
    else:
        network = 5 * pes * router_depth * flit_bits
        queue = int(packets.sum()) * flit_bits
        network_model = 'five finite output queues per router; router_depth packets per output'
    out = {'event_entry_bits': event_bits, 'packet_required_bits': packet_bits, 'flit_bits': flit_bits, 'source_retention_bits': source, 'destination_retention_bits': destination, 'calendar_bits': calendar_bits, 'ni_register_bits': ni, 'source_packet_queue_bits': queue, 'network_storage_bits': network, 'source_capacity_per_core': src_caps.tolist(), 'destination_capacity_per_core': dest_caps.tolist(), 'source_packet_capacity_per_core': packets.tolist(), 'network_storage_model': network_model, 'scope': 'communication storage; excludes neuron/synaptic parameters, external input/readout, and combinational logic'}
    out['total_provisioned_bits'] = sum((out[x] for x in ['source_retention_bits', 'destination_retention_bits', 'calendar_bits', 'ni_register_bits', 'source_packet_queue_bits', 'network_storage_bits']))
    return out

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

def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = ['schema', 'study', 'run', 'delay_profile', 'pattern', 'neurons', 'grid', 'frame_cycles', 'total_provisioned_bits', 'routing_seconds', 'mapping_search_seconds', 'state_verified', 'deadline_violations']
    if rows and rows[0].get('study') == 'utilization':
        fields = ['schema', 'study', 'run', 'delay_profile', 'pattern', 'max_sources_setting', 'structural_utilization', 'state_verified', 'deadline_violations']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

def ints(s):
    return [int(x) for x in s.split(',')]

def profiles(s):
    result = []
    for pair in s.split(','):
        lo, hi = map(int, pair.split(':'))
        if not 1 <= lo <= hi:
            raise ValueError('Require 1 <= minimum <= maximum delay')
        result.append((lo, hi))
    return result

def graph(n, max_sources, seed):
    if n < 2 or max_sources < 1:
        raise ValueError('Require n>=2 and max_sources>=1')
    rng = np.random.default_rng(seed)
    a = np.zeros((n, n), bool)
    for d in range(n):
        pool = np.delete(np.arange(n), d)
        rng.shuffle(pool)
        degree = max(1, int(math.ceil(rng.random() * min(max_sources, n - 1))))
        a[pool[:degree], d] = True
    return a

def delays_for(n, lo, hi, seed):
    u = np.random.default_rng(seed).random(n)
    return lo + np.floor(u * (hi - lo + 1)).astype(int)

def reference_mapping(n, pes):
    return np.arange(n, dtype=int) % pes

def cost_context(adj, delays, route, steps, flit, alpha):
    from fractions import Fraction
    initial = reference_mapping(len(adj), route['grid'] ** 2)
    c = calendar(initial, adj, delays, route['grid'] ** 2)
    mem = memory_model(initial, adj, delays, c, route, steps, flit)
    L0 = int(route['T'] * c['omega'])
    B0 = int(mem['total_provisioned_bits'])
    a = Fraction(str(alpha)).limit_denominator(10000)
    if not 0 <= a <= 1:
        raise ValueError('alpha must be between 0 and 1')
    return {'L0': L0, 'B0': B0, 'weight_L': a.numerator * B0, 'weight_B': (a.denominator - a.numerator) * L0, 'normalizer': a.denominator * L0 * B0, 'alpha': alpha}

def mapping_cost(m, adj, delays, route, steps, flit, ctx):
    c = calendar(m, adj, delays, route['grid'] ** 2)
    b = memory_model(m, adj, delays, c, route, steps, flit)['total_provisioned_bits']
    return int(ctx['weight_L'] * route['T'] * c['omega'] + ctx['weight_B'] * b)

def search_mapping(adj, delays, route, cap, steps, flit, ctx, seed, iterations, budget, method='DARSA'):
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = len(adj)
    pes = route['grid'] ** 2
    m = reference_mapping(n, pes)
    validate_mapping(m, n, pes, cap)
    value = mapping_cost(m, adj, delays, route, steps, flit, ctx)
    best = m.copy()
    bv = value
    done = 0
    for i in range(iterations):
        if time.perf_counter() - start >= budget:
            break
        if method == 'Random':
            candidate = rng.permutation(reference_mapping(n, pes))
        else:
            candidate = m.copy()
            s = int(rng.integers(n))
            if rng.random() < 0.5:
                t = int(rng.integers(n))
                candidate[s], candidate[t] = (candidate[t], candidate[s])
            else:
                target = int(rng.integers(pes))
                if np.count_nonzero(candidate == target) >= cap:
                    continue
                candidate[s] = target
        nv = mapping_cost(candidate, adj, delays, route, steps, flit, ctx)
        done += 1
        temperature = 0.15 * (0.001 / 0.15) ** (i / max(1, iterations - 1)) * ctx['normalizer']
        if method == 'Random' or nv <= value or rng.random() < math.exp(min(0, (value - nv) / max(temperature, 1e-12))):
            m, value = (candidate, nv)
        if nv < bv:
            best, bv = (candidate.copy(), nv)
    return (best, {'mapping_search_seconds': time.perf_counter() - start, 'evaluations': done, 'objective_integer': bv, 'objective_normalized': bv / ctx['normalizer'], 'status': 'FEASIBLE', 'bound_integer': '', 'relative_gap': '', 'budget_seconds': budget})

def firing_cases(n, cal, mapping, seed, window, activity):
    rng = np.random.default_rng(seed)
    g = rng.integers(window, size=n)
    sparse = g.copy()
    sparse[rng.random(n) >= activity] = -1
    after = np.maximum(cal['phase'], 0) + 1
    return {'simultaneous': np.zeros(n, int), 'staggered': g, 'silent': sparse, 'after_opportunity': after, 'all_silent': np.full(n, -1, int)}

def structural_counts(adj, mapping, route, active):
    pes = route['grid'] ** 2
    useful = total = 0
    for s in active:
        v = int(mapping[s])
        edges = [tuple(e) for e in route['trees'][str(v)]]
        parent = {(e[2], e[3]): e for e in edges}
        needed = set()
        for d in np.unique(mapping[np.flatnonzero(adj[s])]):
            if d == v:
                continue
            node = (int(d), int(route['arrivals'][v][d]))
            while node in parent:
                e = parent[node]
                needed.add(e)
                node = (e[0], e[1])
        useful += len(needed)
        total += len(edges)
    return (useful, total)

def replay(adj, delays, mapping, route, cal, g, window, activity_label):
    n = len(adj)
    pes = route['grid'] ** 2
    sources = np.flatnonzero(adj.any(axis=1))
    active = sources[g[sources] >= 0]
    last = max([int(g[s] + delays[s]) for s in active] + [window - 1])
    H = last + 1
    ref = np.zeros((H, n), np.int64)
    got = np.zeros_like(ref)
    used = set()
    ledger = set()
    si = []
    di = []
    count = 0
    for s in active:
        gen = int(g[s])
        due = gen + int(delays[s])
        v = int(mapping[s])
        board = gen + (int(cal['phase'][s]) - gen) % int(cal['periods'][v])
        sub = int(cal['subslot'][s])
        key = (v, board, sub)
        if key in used:
            raise AssertionError('Injection collision')
        used.add(key)
        if not gen <= board < due or not sub < cal['omega']:
            raise AssertionError('Illegal boarding/deadline')
        si.append((v, gen, board + 1))
        for d in np.flatnonzero(adj[s]):
            ref[due, d] += 1
        for core in np.unique(mapping[np.flatnonzero(adj[s])]):
            within = int(route['arrivals'][v][core]) * cal['omega'] + sub
            if within >= route['T'] * cal['omega']:
                raise AssertionError('Route exceeds frame')
            event = (int(s), int(core))
            if event in ledger:
                raise AssertionError('Duplicate delivery')
            ledger.add(event)
            di.append((int(core), board + 1, due))
            for d in np.flatnonzero(adj[s] & (mapping == core)):
                got[due, d] += 1
                count += 1
    if not np.array_equal(ref, got):
        raise AssertionError('Consumed events mismatch')
    r = np.zeros(n, np.int64)
    d = r.copy()
    for t in range(H):
        r = 3 * r // 4 + ref[t]
        d = 3 * d // 4 + got[t]
        if not np.array_equal(r, d):
            raise AssertionError('Synaptic state mismatch')
    reserved = sum((int(cal['counts'][v]) * H for v in range(pes)))
    assigned = sum(((H - 1 - int(cal['phase'][s])) // int(cal['periods'][mapping[s]]) + 1 for s in sources if cal['phase'][s] < H))
    useful, total = structural_counts(adj, mapping, route, active)
    sp = interval_peaks(si, pes)
    dp = interval_peaks(di, pes)
    return {'pattern': activity_label, 'inference_frames': H, 'inference_cycles': H * route['T'] * cal['omega'], 'tail_frames': max(0, H - window), 'emitted_events': len(active), 'delivered_core_events': len(ledger), 'consumed_synaptic_events': count, 'source_peak_entries': max(sp, default=0), 'destination_peak_entries': max(dp, default=0), 'sum_source_peak_entries': sum(sp), 'sum_destination_peak_entries': sum(dp), 'reserved_opportunities': reserved, 'assigned_opportunities': assigned, 'idle_reservations': reserved - len(active), 'temporal_occupancy': len(active) / reserved if reserved else 0.0, 'physical_injection_opportunities': H * pes * cal['omega'], 'physical_injection_occupancy': len(active) / (H * pes * cal['omega']), 'useful_tree_traversals': useful, 'total_tree_traversals': total, 'structural_utilization': useful / total if total else '', 'deadline_violations': 0, 'state_verified': True, 'spike_count_verified': bool(np.all(g >= -1))}

def evaluate_case(adj, delays, mapping, route, cap, args, metadata, info):
    validate_mapping(mapping, len(adj), route['grid'] ** 2, cap)
    start = time.perf_counter()
    cal = calendar(mapping, adj, delays, route['grid'] ** 2)
    calendar_seconds = time.perf_counter() - start
    steps = args.window + 2 * int(delays.max()) + 2
    mem = memory_model(mapping, adj, delays, cal, route, steps, args.flit_bits)
    base = {'schema': SCHEMA, **metadata, 'grid': route['grid'], 'neurons': len(adj), 'sources': int(adj.any(axis=1).sum()), 'connections': int(adj.sum()), 'capacity': cap, 'T': route['T'], 'omega': cal['omega'], 'frame_cycles': route['T'] * cal['omega'], 'periods': json.dumps(cal['periods'].tolist()), 'counts': json.dumps(cal['counts'].tolist()), 'route_id': route['route_id'], 'routing_seconds': route['routing_seconds'], **info, 'final_calendar_seconds': calendar_seconds, 'deployment_seconds': info['mapping_search_seconds'] + calendar_seconds, **{k: v for k, v in mem.items() if isinstance(v, (int, float, str))}}
    rows = []
    for label, g in firing_cases(len(adj), cal, mapping, args.seed + metadata['run'], args.window, args.activity).items():
        r = replay(adj, delays, mapping, route, cal, g, args.window, label)
        if r['sum_source_peak_entries'] * mem['event_entry_bits'] > mem['source_retention_bits']:
            raise AssertionError('Source capacity')
        if r['sum_destination_peak_entries'] * mem['event_entry_bits'] > mem['destination_retention_bits']:
            raise AssertionError('Destination capacity')
        rows.append({**base, **r})
    return (rows, {'metadata': metadata, 'mapping': mapping.tolist(), 'delays': delays.tolist(), 'phase': cal['phase'].tolist(), 'subslot': cal['subslot'].tolist(), 'periods': cal['periods'].tolist(), 'memory': mem, 'compilation': info})

def common_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--out', required=True)
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--iterations', type=int, default=500)
    p.add_argument('--budget', type=float, default=120.0)
    p.add_argument('--route-attempts', type=int, default=50)
    p.add_argument('--delay-profiles', type=profiles, default=profiles('1:1,5:5,1:20,5:20'))
    p.add_argument('--max-sources', type=int, default=20)
    p.add_argument('--window', type=int, default=32)
    p.add_argument('--activity', type=float, default=0.5)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--flit-bits', type=int, default=32)
    return p

def validate_args(a):
    if a.runs < 1 or a.iterations < 0 or a.budget <= 0 or (a.route_attempts < 1) or (a.window < 1) or (not 0 <= a.activity <= 1):
        raise ValueError('Invalid experiment arguments')

def route_for(grid, run, args):
    route = build_routes(grid, args.seed + run, args.route_attempts)
    return route

def run_mapping(adj, delays, route, args, run):
    cap = max(1, 2 * math.ceil(len(adj) / route['grid'] ** 2))
    steps = args.window + 2 * int(delays.max()) + 2
    ctx = cost_context(adj, delays, route, steps, args.flit_bits, args.alpha)
    mapping, info = search_mapping(adj, delays, route, cap, steps, args.flit_bits, ctx, args.seed + run, args.iterations, args.budget)
    return (cap, mapping, {**info, **ctx})

def self_test():
    a = np.ones((10, 10), bool)
    np.fill_diagonal(a, False)
    m = np.zeros(10, int)
    for delay, expected in [(5, 2), (1, 10)]:
        c = calendar(m, a, np.full(10, delay), 1)
        assert c['omega'] == expected
        for g in range(30):
            f = g + (c['phase'] - g) % delay
            assert np.all(f + 1 <= g + delay)
    return [{'neurons_on_core': 10, 'delay': 5, 'period': 5, 'omega': 2}, {'neurons_on_core': 10, 'delay': 1, 'period': 1, 'omega': 10}]

def main():
    p = common_parser(__doc__)
    p.add_argument('--neurons', type=ints, default=ints('16,32,64,128,256,512,1024'))
    a = p.parse_args()
    validate_args(a)
    rows = []
    self_test()
    for run in range(a.runs):
        route = route_for(4, run, a)
        for n in a.neurons:
            adj = graph(n, a.max_sources, a.seed + run)
            for lo, hi in a.delay_profiles:
                delays = delays_for(n, lo, hi, a.seed + 1000 + run)
                cap, m, info = run_mapping(adj, delays, route, a, run)
                meta = {'study': 'logical', 'run': run, 'delay_profile': f'{lo}:{hi}', 'method': 'DARSA'}
                r, detail = evaluate_case(adj, delays, m, route, cap, a, meta, info)
                rows += r
                write_csv(Path(a.out) / 'metrics.csv', rows)
if __name__ == '__main__':
    main()
