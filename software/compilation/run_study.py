"""Compare joint CP-SAT, exact-route CP-SAT, heuristic-route CP-SAT, and DARSA pipelines."""
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

def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = ['study_schema', 'run', 'delay_profile', 'pattern', 'method', 'neurons', 'grid', 'status', 'frame_cycles', 'total_provisioned_bits', 'total_compilation_seconds', 'mapping_search_seconds', 'routing_seconds', 'state_verified', 'deadline_violations', 'objective_normalized']
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
    p.add_argument('--runs', type=int, default=1)
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
    write_json(Path(args.out) / f'route_grid{grid}_run{run}.json', route)
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
import multiprocessing as mp
import os

def atomic_checkpoint(path, data):
    temp = str(path) + '.tmp'
    write_json(temp, data)
    os.replace(temp, path)

def exact_objective(reader, expression):
    value = int(reader.Value(expression))
    reported = float(reader.ObjectiveValue())
    bound = float(reader.BestObjectiveBound())
    if bound > value + 1e-05:
        raise AssertionError(f'Invalid bound: {bound} > evaluated objective {value}')
    return (value, reported, bound)

def cp_worker(payload, checkpoint):
    start = time.perf_counter()
    try:
        from ortools.sat.python import cp_model
        adj, delays, route, cap, steps, flit, ctx, budget, seed = payload
        n = len(adj)
        pes = route['grid'] ** 2
        src = list(map(int, np.flatnonzero(adj.any(axis=1))))
        model = cp_model.CpModel()
        x = [[model.NewBoolVar(f'x{i}_{v}') for v in range(pes)] for i in range(n)]
        for row in x:
            model.AddExactlyOne(row)
        for v in range(pes):
            model.Add(sum((x[i][v] for i in range(n))) <= cap)
        model.Add(x[0][0] == 1)
        maxd = int(delays.max())
        mvars = []
        cvars = []
        calvars = []
        idb = bits(n)
        eventb = idb + bits(steps + 1) + 1
        for v in range(pes):
            m = model.NewIntVar(0, cap, f'm{v}')
            model.Add(m == sum((x[s][v] for s in src)))
            mvars.append(m)
            ds = []
            for s in src:
                d = model.NewIntVar(1, maxd + 1, f'd{s}_{v}')
                model.Add(d == int(delays[s])).OnlyEnforceIf(x[s][v])
                model.Add(d == maxd + 1).OnlyEnforceIf(x[s][v].Not())
                ds.append(d)
            minimum = model.NewIntVar(1, maxd + 1, f'min{v}')
            model.AddMinEquality(minimum, ds)
            empty = model.NewBoolVar(f'empty{v}')
            model.Add(m == 0).OnlyEnforceIf(empty)
            model.Add(m >= 1).OnlyEnforceIf(empty.Not())
            period = model.NewIntVar(1, maxd, f'p{v}')
            model.Add(period == 1).OnlyEnforceIf(empty)
            model.Add(period == minimum).OnlyEnforceIf(empty.Not())
            c = model.NewIntVar(0, cap, f'c{v}')
            cvars.append(c)
            cb = model.NewIntVar(0, (cap + maxd) * (idb + 1) + bits(maxd), f'cal{v}')
            calvars.append(cb)
            table = [(0, 1, 0, bits(1))]
            for mm in range(1, cap + 1):
                for pp in sorted(set(map(int, delays[src]))):
                    cc = math.ceil(mm / pp)
                    table.append((mm, pp, cc, pp * cc * (idb + 1) + bits(pp)))
            model.AddAllowedAssignments([m, period, c, cb], table)
        omega = model.NewIntVar(1, cap, 'omega')
        model.AddMaxEquality(omega, cvars + [1])
        ob = model.NewIntVar(1, bits(cap), 'omega_bits')
        model.AddAllowedAssignments([omega, ob], [(i, bits(i)) for i in range(1, cap + 1)])
        dest = []
        for s in src:
            targets = list(map(int, np.flatnonzero(adj[s])))
            for v in range(pes):
                y = model.NewBoolVar(f'hold{s}_{v}')
                model.AddMaxEquality(y, [x[d][v] for d in targets])
                dest.append(y)
        memory = len(src) * eventb + sum(dest) * eventb + 2 * pes * flit + sum(calvars) + pes * ob + 2 * len(physical_links(route['grid'])) * flit * omega
        objective = ctx['weight_L'] * route['T'] * omega + ctx['weight_B'] * memory
        model.Minimize(objective)
        build = time.perf_counter() - start

        class Checkpoint(cp_model.CpSolverSolutionCallback):

            def on_solution_callback(self):
                mapping = [next((v for v in range(pes) if self.Value(x[i][v]))) for i in range(n)]
                value, reported, bound = exact_objective(self, objective)
                data = {'mapping': mapping, 'status': 'FEASIBLE', 'objective_integer': value, 'solver_reported_objective': reported, 'bound_integer': bound, 'relative_gap': max(0.0, (value - bound) / max(1, abs(value))), 'mapping_search_seconds': time.perf_counter() - start, 'model_build_seconds': build, 'solver_seconds': max(0, time.perf_counter() - start - build)}
                atomic_checkpoint(checkpoint, data)
        remaining = budget - build
        if remaining <= 0:
            atomic_checkpoint(checkpoint, {'status': 'TIMEOUT_BUILD', 'mapping_search_seconds': build, 'model_build_seconds': build})
            return
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = remaining
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = seed
        status = solver.Solve(model, Checkpoint())
        elapsed = time.perf_counter() - start
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            mapping = [next((v for v in range(pes) if solver.Value(x[i][v]))) for i in range(n)]
            value, reported, bound = exact_objective(solver, objective)
            result = {'mapping': mapping, 'status': solver.StatusName(status), 'objective_integer': value, 'solver_reported_objective': reported, 'bound_integer': bound, 'relative_gap': max(0.0, (value - bound) / max(1, abs(value)))}
        else:
            result = {'status': solver.StatusName(status), 'bound_integer': float(solver.BestObjectiveBound())}
        result.update(mapping_search_seconds=elapsed, model_build_seconds=build, solver_seconds=elapsed - build)
        atomic_checkpoint(checkpoint, result)
    except Exception as exc:
        atomic_checkpoint(checkpoint, {'status': 'ERROR', 'error': repr(exc), 'mapping_search_seconds': time.perf_counter() - start})

def cp_search(payload, path, grace):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    start = time.perf_counter()
    p = mp.get_context('spawn').Process(target=cp_worker, args=(payload, str(path)))
    p.start()
    p.join(payload[-2] + grace)
    timed = p.is_alive()
    if timed:
        p.terminate()
        p.join()
    result = json.loads(path.read_text()) if path.exists() else {'status': 'TIMEOUT_NO_INCUMBENT' if timed else 'WORKER_FAILED'}
    result['worker_wall_seconds'] = time.perf_counter() - start
    result.setdefault('mapping_search_seconds', result['worker_wall_seconds'])
    if timed:
        result['incumbent_found_seconds'] = result.get('mapping_search_seconds', '')
        result['mapping_search_seconds'] = result['worker_wall_seconds']
        if 'mapping' in result:
            result['status'] = 'FEASIBLE_TIMEOUT'
    return result
STUDY_SCHEMA = 'chronoc-compilation'
PIPELINES = ['Joint-CP', 'ExactRoute-CP', 'HeuristicRoute-CP', 'HeuristicRoute-DARSA']
FEASIBLE = {'FEASIBLE', 'OPTIMAL', 'FEASIBLE_TIMEOUT'}
PATTERNS = ['simultaneous', 'staggered', 'silent', 'after_opportunity', 'all_silent']

def add_route_constraints(model, grid, horizon):
    pes = grid * grid
    if grid < 2:
        raise ValueError('Exact routing requires --grid >= 2')
    links = physical_links(grid)
    edgevars = {}
    injections = {}
    arrivals = {}
    resources = collections.defaultdict(list)
    for s in range(pes):
        incoming = collections.defaultdict(list)
        outgoing = collections.defaultdict(list)
        sr, sc = divmod(s, grid)
        for u, v in links:
            if u < pes and u != s:
                continue
            if v == s:
                continue
            if u < pes:
                earliest = 0
            else:
                ur, uc = divmod(u - pes, grid)
                earliest = 1 + abs(sr - ur) + abs(sc - uc)
            for t in range(earliest, horizon - 1):
                e = (u, t, v, t + 1)
                b = model.NewBoolVar(f'e{s}_{u}_{v}_{t}')
                edgevars[s, e] = b
                resources[e].append(b)
                outgoing[u, t].append(b)
                incoming[v, t + 1].append(b)
        root = [(t, bs[0]) for (u, t), bs in outgoing.items() if u == s]
        model.Add(sum((b for _, b in root)) == 1)
        injections[s] = sum((t * b for t, b in root))
        nodes = set(incoming) | set(outgoing)
        for u, t in sorted(nodes):
            if u < pes:
                continue
            ins = incoming.get((u, t), [])
            outs = outgoing.get((u, t), [])
            model.Add(sum(ins) <= 1)
            for b in outs:
                model.Add(b <= sum(ins))
            model.Add(sum(outs) >= sum(ins))
        for d in range(pes):
            if d == s:
                continue
            reaches = [(t, b) for (u, t), bs in incoming.items() if u == d for b in bs]
            model.Add(sum((b for _, b in reaches)) == 1)
            a = model.NewIntVar(1, horizon - 1, f'arrival{s}_{d}')
            model.Add(a == sum((t * b for t, b in reaches)))
            arrivals[s, d] = a
    for bs in resources.values():
        model.Add(sum(bs) <= 1)
    depth = model.NewIntVar(1, horizon, 'schedule_depth')
    model.AddMaxEquality(depth, [a + 1 for a in arrivals.values()])
    return (edgevars, injections, arrivals, depth)

def extract_route(reader, grid, horizon, routevars):
    edges, starts, arrivals, depth = routevars
    pes = grid * grid
    trees = {str(s): [] for s in range(pes)}
    for (s, e), b in edges.items():
        if reader.Value(b):
            trees[str(s)].append(list(e))
    start = [int(reader.Value(starts[s])) for s in range(pes)]
    matrix = [[start[s] if s == d else int(reader.Value(arrivals[s, d])) for d in range(pes)] for s in range(pes)]
    route = {'grid': grid, 'T': int(reader.Value(depth)), 'starts': start, 'arrivals': matrix, 'trees': trees, 'search_depth': horizon, 'routing_algorithm': 'CP-SAT space-time arborescences'}
    route['route_id'] = hashlib.sha256(json.dumps(route, sort_keys=True).encode()).hexdigest()
    validate_routes(route)
    return route

def route_joint_worker(payload, checkpoint):
    if payload['mode'] == 'application_joint':
        return application_joint_worker(payload, checkpoint)
    started = time.perf_counter()
    build = None
    try:
        from ortools.sat.python import cp_model
        mode = payload['mode']
        grid = payload['grid']
        horizon = payload['horizon']
        model = cp_model.CpModel()
        rv = add_route_constraints(model, grid, horizon)
        objective = rv[3]
        model.Minimize(objective)
        build = time.perf_counter() - started

        def snapshot(reader, status):
            route = extract_route(reader, grid, horizon, rv)
            value, reported, bound = exact_objective(reader, objective)
            result = {'status': status, 'route': route, 'objective_integer': value, 'bound_integer': bound, 'solver_reported_objective': reported, 'relative_gap': max(0.0, (value - bound) / max(1, abs(value))), 'stage_seconds': time.perf_counter() - started, 'model_build_seconds': build, 'solver_seconds': time.perf_counter() - started - build, 'objective_scope': 'schedule_depth'}
            if value != route['T']:
                raise AssertionError('Routing objective mismatch')
            return result

        class Incumbent(cp_model.CpSolverSolutionCallback):

            def on_solution_callback(self):
                atomic_checkpoint(checkpoint, snapshot(self, 'FEASIBLE'))
        remaining = payload['budget'] - build
        if remaining <= 0:
            atomic_checkpoint(checkpoint, {'status': 'TIMEOUT_BUILD', 'stage_seconds': build, 'model_build_seconds': build})
            return
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = remaining
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = payload['seed']
        status = solver.Solve(model, Incumbent())
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            result = snapshot(solver, solver.StatusName(status))
        else:
            result = {'status': solver.StatusName(status), 'bound_integer': float(solver.BestObjectiveBound()), 'stage_seconds': time.perf_counter() - started, 'model_build_seconds': build, 'solver_seconds': time.perf_counter() - started - build}
            if status == cp_model.MODEL_INVALID:
                result['error'] = model.Validate()
        atomic_checkpoint(checkpoint, result)
    except Exception as exc:
        atomic_checkpoint(checkpoint, {'status': 'ERROR', 'error': repr(exc), 'stage_seconds': time.perf_counter() - started, 'model_build_seconds': build})

def bounded_route_joint(payload, path, grace):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    start = time.perf_counter()
    process = mp.get_context('spawn').Process(target=route_joint_worker, args=(payload, str(path)))
    process.start()
    process.join(payload['budget'] + grace)
    timed = process.is_alive()
    if timed:
        process.terminate()
        process.join()
    result = json.loads(path.read_text()) if path.exists() else {'status': 'TIMEOUT_NO_INCUMBENT' if timed else 'WORKER_FAILED'}
    result['worker_wall_seconds'] = time.perf_counter() - start
    result.setdefault('stage_seconds', result['worker_wall_seconds'])
    if timed:
        result['incumbent_found_seconds'] = result.get('stage_seconds', '')
        result['stage_seconds'] = result['worker_wall_seconds']
        if 'route' in result:
            result['status'] = 'FEASIBLE_TIMEOUT'
    return result

def cp_stage_fields(result, prefix):
    return {prefix + '_' + key: result.get(key, '') for key in ['status', 'bound_integer', 'relative_gap', 'model_build_seconds', 'solver_seconds', 'error']}

def validate_exact_result(result):
    if result.get('status') in {'ERROR', 'MODEL_INVALID', 'WORKER_FAILED'}:
        raise RuntimeError('Exact stage failed technically, not a timeout: ' + json.dumps(result))
    if 'route' in result:
        if result['route'].get('kind') == 'application_joint':
            validate_application_route(result['route'])
        else:
            validate_routes(result['route'])

def add_application_joint_model(model, adj, grid, cap, horizon):
    pes = grid * grid
    n = len(adj)
    sources = list(map(int, np.flatnonzero(adj.any(axis=1))))
    x = [[model.NewBoolVar(f'jx{i}_{v}') for v in range(pes)] for i in range(n)]
    for row in x:
        model.AddExactlyOne(row)
    for v in range(pes):
        model.Add(sum((x[i][v] for i in range(n))) <= cap)
    rowvars = []
    colvars = []
    for i in range(n):
        r = model.NewIntVar(0, grid - 1, f'row{i}')
        c = model.NewIntVar(0, grid - 1, f'col{i}')
        model.Add(r == sum((v // grid * x[i][v] for v in range(pes))))
        model.Add(c == sum((v % grid * x[i][v] for v in range(pes))))
        rowvars.append(r)
        colvars.append(c)
    differences = []
    for s, d in zip(*np.nonzero(adj)):
        dr = model.NewIntVar(0, grid - 1, f'dr{s}_{d}')
        dc = model.NewIntVar(0, grid - 1, f'dc{s}_{d}')
        model.AddAbsEquality(dr, rowvars[s] - rowvars[d])
        model.AddAbsEquality(dc, colvars[s] - colvars[d])
        differences.extend([dr, dc])
    distance = sum(differences)
    router_links = [(u - pes, v - pes) for u, v in physical_links(grid) if u >= pes and v >= pes]
    resources = collections.defaultdict(list)
    arcs = {}
    roots = {}
    remote = {}
    times = {}
    starts = {}
    active = {}
    arrivals = []

    def reserve(link, start, present, name):
        resources[link].append(model.NewOptionalIntervalVar(start, 1, start + 1, present, name))
    for s in sources:
        targets = list(map(int, np.flatnonzero(adj[s])))
        rms = []
        for v in range(pes):
            has = model.NewBoolVar(f'has{s}_{v}')
            model.AddMaxEquality(has, [x[d][v] for d in targets])
            b = model.NewBoolVar(f'remote{s}_{v}')
            model.Add(b <= has)
            model.Add(b + x[s][v] <= 1)
            model.Add(b >= has - x[s][v])
            remote[s, v] = b
            rms.append(b)
        act = model.NewBoolVar(f'active{s}')
        model.AddMaxEquality(act, rms)
        active[s] = act
        inj = model.NewIntVar(0, horizon - 3, f'inject{s}')
        starts[s] = inj
        model.Add(inj == 0).OnlyEnforceIf(act.Not())
        visit = {}
        parent = collections.defaultdict(list)
        children = collections.defaultdict(list)
        for v in range(pes):
            root = model.NewBoolVar(f'root{s}_{v}')
            model.Add(root <= x[s][v])
            model.Add(root <= act)
            model.Add(root >= x[s][v] + act - 1)
            roots[s, v] = root
            visited = model.NewBoolVar(f'visit{s}_{v}')
            visit[v] = visited
            model.Add(visited <= act)
            model.Add(visited >= remote[s, v])
            t = model.NewIntVar(1, horizon - 2, f'time{s}_{v}')
            times[s, v] = t
            model.Add(t == 1).OnlyEnforceIf(visited.Not())
            model.Add(t == inj + 1).OnlyEnforceIf(root)
            reserve((v, pes + v), inj, root, f'injlink{s}_{v}')
            reserve((pes + v, v), t, remote[s, v], f'eject{s}_{v}')
            arrival = model.NewIntVar(0, horizon - 1, f'receive{s}_{v}')
            model.Add(arrival == t + 1).OnlyEnforceIf(remote[s, v])
            model.Add(arrival == 0).OnlyEnforceIf(remote[s, v].Not())
            arrivals.append(arrival)
        for u, v in router_links:
            b = model.NewBoolVar(f'arc{s}_{u}_{v}')
            arcs[s, u, v] = b
            parent[v].append(b)
            children[u].append(b)
            model.Add(b <= visit[u])
            model.Add(b <= visit[v])
            model.Add(times[s, v] == times[s, u] + 1).OnlyEnforceIf(b)
            reserve((pes + u, pes + v), times[s, u], b, f'hop{s}_{u}_{v}')
        for v in range(pes):
            model.Add(sum(parent[v]) == visit[v] - roots[s, v])
            model.Add(sum(children[v]) + remote[s, v] >= visit[v])
    for intervals in resources.values():
        model.AddNoOverlap(intervals)
    frame = model.NewIntVar(1, horizon, 'joint_frame_clocks')
    model.AddMaxEquality(frame, [a + 1 for a in arrivals] + [1])
    hops = sum(arcs.values())
    hop_bound = len(sources) * (pes - 1)
    hop_weight = horizon + 1
    distance_weight = (hop_bound + 1) * hop_weight
    distance_bound = int(adj.sum()) * 2 * (grid - 1)
    upper = distance_bound * distance_weight + hop_bound * hop_weight + horizon
    if upper >= 2 ** 60:
        raise ValueError('Joint objective too large; reduce instance/horizon')
    objective = distance_weight * distance + hop_weight * hops + frame
    model.Minimize(objective)
    return {'x': x, 'sources': sources, 'arcs': arcs, 'roots': roots, 'remote': remote, 'times': times, 'starts': starts, 'active': active, 'frame': frame, 'distance': distance, 'hops': hops, 'objective': objective, 'distance_weight': distance_weight, 'hop_weight': hop_weight}

def joint_locality_metrics(adj, mapping, grid):
    total = local = remote = 0
    for s, d in zip(*np.nonzero(adj)):
        u, v = (int(mapping[s]), int(mapping[d]))
        ur, uc = divmod(u, grid)
        vr, vc = divmod(v, grid)
        total += abs(ur - vr) + abs(uc - vc)
        if u == v:
            local += 1
        else:
            remote += 1
    return {'connection_distance': int(total), 'local_connections': local, 'remote_connections': remote}

def extract_application_joint(reader, variables, adj, grid, cap, horizon):
    pes = grid * grid
    n = len(adj)
    mapping = np.array([next((v for v in range(pes) if reader.Value(variables['x'][i][v]))) for i in range(n)], int)
    validate_mapping(mapping, n, pes, cap)
    trees = {}
    injections = {}
    arrivals = {}
    targets = {}
    for s in variables['sources']:
        sourcecore = int(mapping[s])
        destcores = sorted(set(map(int, mapping[np.flatnonzero(adj[s])])) - {sourcecore})
        targets[str(s)] = destcores
        tree = []
        arrival = {}
        if destcores:
            inj = int(reader.Value(variables['starts'][s]))
            injections[str(s)] = inj
            tree.append([sourcecore, inj, pes + sourcecore, inj + 1])
            for pu, pv in physical_links(grid):
                if pu < pes or pv < pes:
                    continue
                u, v = (pu - pes, pv - pes)
                if reader.Value(variables['arcs'][s, u, v]):
                    t = int(reader.Value(variables['times'][s, u]))
                    tree.append([pes + u, t, pes + v, t + 1])
            for v in destcores:
                t = int(reader.Value(variables['times'][s, v]))
                tree.append([pes + v, t, v, t + 1])
                arrival[str(v)] = t + 1
        else:
            injections[str(s)] = -1
        trees[str(s)] = tree
        arrivals[str(s)] = arrival
    route = {'kind': 'application_joint', 'grid': grid, 'T': int(reader.Value(variables['frame'])), 'horizon': horizon, 'mapping': mapping.tolist(), 'targets': targets, 'trees': trees, 'injections': injections, 'arrivals': arrivals, 'reservation_period': 1, 'omega': 1}
    route['route_id'] = hashlib.sha256(json.dumps(route, sort_keys=True).encode()).hexdigest()
    validate_application_route(route, adj, mapping)
    return (mapping, route)

def validate_application_route(route, adj=None, mapping=None):
    grid = route['grid']
    pes = grid * grid
    T = route['T']
    links = set(physical_links(grid))
    used = set()
    mapping = np.asarray(route['mapping'] if mapping is None else mapping, int)
    for key, raw in route['trees'].items():
        s = int(key)
        core = int(mapping[s])
        target = set(route['targets'][key])
        if adj is not None:
            actual = set(map(int, mapping[np.flatnonzero(adj[s])])) - {core}
            if actual != target:
                raise AssertionError('Joint targets differ from workload')
        edges = [tuple(e) for e in raw]
        if not target:
            if edges or route['injections'][key] != -1:
                raise AssertionError('Local source injected')
            continue
        root = (core, route['injections'][key])
        nodes = {root}
        pending = edges.copy()
        reached = []
        outgoing = collections.Counter()
        if len(edges) != len(set(edges)) or used.intersection(edges):
            raise AssertionError('Joint link/clock collision')
        used.update(edges)
        while pending:
            later = []
            progress = False
            for u, t, v, t2 in pending:
                if (u, t) not in nodes:
                    later.append((u, t, v, t2))
                    continue
                if (u, v) not in links or t2 != t + 1 or (not 0 <= t < t2 < T):
                    raise AssertionError('Joint invalid edge/timing')
                if u < pes and (u, t) != root:
                    raise AssertionError('Endpoint transit')
                if (v, t2) in nodes:
                    raise AssertionError('Joint reconvergence')
                outgoing[u, t] += 1
                nodes.add((v, t2))
                progress = True
                if v < pes:
                    reached.append(v)
                    if route['arrivals'][key].get(str(v)) != t2:
                        raise AssertionError('Joint arrival mismatch')
            if not progress:
                raise AssertionError('Joint disconnected branch')
            pending = later
        if set(reached) != target or len(reached) != len(target):
            raise AssertionError('Joint missing/extra/duplicate target')
        if outgoing[root] != 1:
            raise AssertionError('Joint duplicate injection')
        for u, t in nodes:
            if u >= pes and (not outgoing[u, t]):
                raise AssertionError('Unneeded dangling branch')
        routers = [u for u, t in nodes if u >= pes]
        if len(routers) != len(set(routers)):
            raise AssertionError('Joint tree revisits router')
    return True

def application_joint_worker(payload, checkpoint):
    start = time.perf_counter()
    build = None
    try:
        from ortools.sat.python import cp_model
        adj = payload['adj']
        grid = payload['grid']
        cap = payload['cap']
        horizon = payload['horizon']
        model = cp_model.CpModel()
        variables = add_application_joint_model(model, adj, grid, cap, horizon)
        build = time.perf_counter() - start

        def snapshot(reader, status):
            mapping, route = extract_application_joint(reader, variables, adj, grid, cap, horizon)
            value, reported, bound = exact_objective(reader, variables['objective'])
            loc = joint_locality_metrics(adj, mapping, grid)
            hops = sum((1 for es in route['trees'].values() for u, t, v, t2 in es if u >= grid * grid and v >= grid * grid))
            actual = variables['distance_weight'] * loc['connection_distance'] + variables['hop_weight'] * hops + route['T']
            if value != actual:
                raise AssertionError('Joint locality objective mismatch')
            return {'status': status, 'route': route, 'mapping': mapping.tolist(), 'objective_integer': value, 'solver_reported_objective': reported, 'bound_integer': bound, 'relative_gap': max(0.0, (value - bound) / max(1, abs(value))), 'objective_scope': 'lexicographic connection distance / multicast hops / frame clocks', 'joint_distance_weight': variables['distance_weight'], 'joint_hop_weight': variables['hop_weight'], 'multicast_router_hops': hops, **loc, 'stage_seconds': time.perf_counter() - start, 'model_build_seconds': build, 'solver_seconds': time.perf_counter() - start - build}

        class Checkpoint(cp_model.CpSolverSolutionCallback):

            def on_solution_callback(self):
                atomic_checkpoint(checkpoint, snapshot(self, 'FEASIBLE'))
        remaining = payload['budget'] - build
        if remaining <= 0:
            atomic_checkpoint(checkpoint, {'status': 'TIMEOUT_BUILD', 'stage_seconds': build, 'model_build_seconds': build})
            return
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = remaining
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = payload['seed']
        status = solver.Solve(model, Checkpoint())
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            result = snapshot(solver, solver.StatusName(status))
        else:
            result = {'status': solver.StatusName(status), 'bound_integer': float(solver.BestObjectiveBound()), 'stage_seconds': time.perf_counter() - start, 'model_build_seconds': build, 'solver_seconds': time.perf_counter() - start - build}
            if status == cp_model.MODEL_INVALID:
                result['error'] = model.Validate()
        atomic_checkpoint(checkpoint, result)
    except Exception as exc:
        atomic_checkpoint(checkpoint, {'status': 'ERROR', 'error': repr(exc), 'stage_seconds': time.perf_counter() - start, 'model_build_seconds': build})

def application_memory(adj, mapping, route, steps, flit):
    pes = route['grid'] ** 2
    n = len(adj)
    T = route['T']
    c = calendar(mapping, adj, np.ones(n, int), pes)
    c['omega'] = 1
    mem = memory_model(mapping, adj, np.ones(n, int), c, route, steps, flit)
    old = mem['calendar_bits']
    mem['calendar_bits'] = pes * (T * (bits(n) + 1) + bits(T))
    mem['total_provisioned_bits'] += mem['calendar_bits'] - old
    mem['network_storage_model'] = 'two one-word banks per receiving channel; unit-slot application schedule'
    mem['calendar_storage_model'] = 'dense source-id/valid injection table per core per physical slot, plus slot counter'
    return mem

def replay_application(adj, delays, mapping, route, g, window, label):
    n = len(adj)
    pes = route['grid'] ** 2
    sources = np.flatnonzero(adj.any(axis=1))
    active = sources[g[sources] >= 0]
    H = max([int(g[s] + delays[s]) for s in active] + [window - 1]) + 1
    reference = np.zeros((H, n), np.int64)
    got = np.zeros_like(reference)
    source_intervals = []
    destination_intervals = []
    ledger = set()
    injections = set()
    network_events = 0
    traversals = 0
    for s in active:
        gen = int(g[s])
        due = gen + int(delays[s])
        src = int(mapping[s])
        key = str(s)
        source_intervals.append((src, gen, gen + 1))
        if route['targets'][key]:
            slot = route['injections'][key]
            reservation = (src, gen, slot)
            if reservation in injections:
                raise AssertionError('Joint injection collision')
            injections.add(reservation)
            network_events += 1
            traversals += len(route['trees'][key])
        for d in np.flatnonzero(adj[s]):
            reference[due, d] += 1
        for v in np.unique(mapping[np.flatnonzero(adj[s])]):
            if v != src and str(int(v)) not in route['arrivals'][key]:
                raise AssertionError('Joint missing delivery')
            if v != src and (not route['arrivals'][key][str(int(v))] < route['T']):
                raise AssertionError('Joint late frame arrival')
            if gen + 1 > due:
                raise AssertionError('Joint logical deadline')
            ev = (int(s), int(v))
            if ev in ledger:
                raise AssertionError('Joint duplicate core event')
            ledger.add(ev)
            destination_intervals.append((int(v), gen + 1, due))
            for d in np.flatnonzero(adj[s] & (mapping == v)):
                got[due, d] += 1
    if not np.array_equal(reference, got):
        raise AssertionError('Joint event mismatch')
    a = np.zeros(n, np.int64)
    b = a.copy()
    for t in range(H):
        a = 3 * a // 4 + reference[t]
        b = 3 * b // 4 + got[t]
        if not np.array_equal(a, b):
            raise AssertionError('Joint synaptic-state mismatch')
    remote_sources = sum((bool(v) for v in route['targets'].values()))
    reserved = H * remote_sources
    sp = interval_peaks(source_intervals, pes)
    dp = interval_peaks(destination_intervals, pes)
    return {'pattern': label, 'inference_frames': H, 'inference_cycles': H * route['T'], 'tail_frames': max(0, H - window), 'emitted_events': len(active), 'network_injected_events': network_events, 'delivered_core_events': len(ledger), 'consumed_synaptic_events': int(got.sum()), 'source_peak_entries': max(sp, default=0), 'destination_peak_entries': max(dp, default=0), 'sum_source_peak_entries': sum(sp), 'sum_destination_peak_entries': sum(dp), 'reserved_opportunities': reserved, 'assigned_opportunities': reserved, 'idle_reservations': reserved - network_events, 'temporal_occupancy': network_events / reserved if reserved else 0.0, 'physical_injection_opportunities': H * pes * route['T'], 'physical_injection_occupancy': network_events / (H * pes * route['T']), 'useful_tree_traversals': traversals, 'total_tree_traversals': traversals, 'structural_utilization': 1.0 if traversals else '', 'deadline_violations': 0, 'state_verified': True, 'spike_count_verified': bool(np.all(g >= -1))}

def evaluate_application_case(adj, delays, mapping, route, cap, args, metadata, info):
    validate_mapping(mapping, len(adj), route['grid'] ** 2, cap)
    validate_application_route(route, adj, mapping)
    started = time.perf_counter()
    slots = [route['injections'].get(str(s), -1) for s in range(len(adj))]
    final_seconds = time.perf_counter() - started
    steps = args.window + 2 * int(delays.max()) + 2
    mem = application_memory(adj, mapping, route, steps, args.flit_bits)
    resource_cost = info['weight_L'] * route['T'] + info['weight_B'] * mem['total_provisioned_bits']
    base = {'schema': SCHEMA, **metadata, 'grid': route['grid'], 'neurons': len(adj), 'sources': int(adj.any(axis=1).sum()), 'connections': int(adj.sum()), 'capacity': cap, 'T': route['T'], 'omega': 1, 'frame_cycles': route['T'], 'periods': json.dumps([1] * route['grid'] ** 2), 'route_id': route['route_id'], **info, 'objective_normalized': resource_cost / info['normalizer'], 'resource_score_integer': resource_cost, 'objective_normalized_scope': 'common resource score, not optimized by application Joint-CP', 'calendar_model': 'source-specific immediate TDM reservations, P=1', 'final_calendar_seconds': final_seconds, **{k: v for k, v in mem.items() if isinstance(v, (int, float, str))}}
    rows = []
    dummy = {'phase': np.zeros(len(adj), int)}
    for label, g in firing_cases(len(adj), dummy, mapping, args.seed + metadata['run'], args.window, args.activity).items():
        result = replay_application(adj, delays, mapping, route, g, args.window, label)
        if result['sum_source_peak_entries'] * mem['event_entry_bits'] > mem['source_retention_bits']:
            raise AssertionError('Joint source capacity')
        if result['sum_destination_peak_entries'] * mem['event_entry_bits'] > mem['destination_retention_bits']:
            raise AssertionError('Joint destination capacity')
        rows.append({**base, **result})
    return (rows, {'metadata': metadata, 'mapping': mapping.tolist(), 'delays': delays.tolist(), 'injection_slots': slots, 'memory': mem, 'compilation': info, 'route': route})

def run_experiment():
    p = common_parser(__doc__)
    p.add_argument('--neurons', type=ints, default=ints('16,32,64,128,256,512,1024'))
    p.add_argument('--grid', type=int, default=4)
    p.add_argument('--route-depth', type=int, default=None, help='Common finite time-expanded horizon for all routing methods')
    p.add_argument('--joint-horizon', type=int, default=None, help='Application joint frame horizon; default S*(PEs+1)+1')
    p.add_argument('--route-budget', type=float, default=None)
    p.add_argument('--joint-budget', type=float, default=None)
    p.add_argument('--grace', type=float, default=3.0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--pipelines', default=','.join(PIPELINES), help='Comma-separated pipeline IDs, default all four')
    a = p.parse_args()
    validate_args(a)
    if a.grid < 2 or min(a.neurons) < 2 or a.grace < 0:
        raise ValueError('Require grid>=2, neurons>=2, grace>=0')
    a.route_budget = a.budget if a.route_budget is None else a.route_budget
    a.joint_budget = a.route_budget + a.budget if a.joint_budget is None else a.joint_budget
    a.route_depth = a.route_depth or max(32, 4 * a.grid * a.grid + 4 * a.grid)
    if a.joint_horizon is not None and a.joint_horizon < 4:
        raise ValueError('joint-horizon must be >=4')
    selected = a.pipelines.split(',')
    if not set(selected) <= set(PIPELINES) or len(selected) != len(set(selected)):
        raise ValueError('Invalid/duplicate pipeline IDs')
    if min(a.route_budget, a.joint_budget) <= 0 or a.route_depth < 4:
        raise ValueError('Invalid budget/horizon')
    try:
        import ortools
    except ImportError:
        raise SystemExit('Install dependencies: pip install numpy ortools')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(json.dumps({**vars(a), 'revision': STUDY_SCHEMA, 'ortools_version': ortools.__version__, 'routing_model': 'fixed-grid broadcast and application-specific multicast pipelines', 'joint_objective': 'lexicographic connection distance, multicast router hops, frame clocks', 'joint_calendar': 'source-specific immediate reservations P=1; no common core injection slot', 'joint_comparison': 'different formulation/objective; common resource score is evaluation-only', 'calendar_model': 'deterministic common-period calendars', 'normalization': 'same heuristic-route reference L0/B0 across all four pipelines', 'joint_budget_policy': 'route_budget + mapping_budget by default', 'heuristic_routing': 'randomized breadth-first routing'}))
    cp = out / 'config.json'
    csvpath = out / 'metrics.csv'
    if cp.exists():
        if not a.resume:
            raise ValueError('Use a new output directory or --resume')
        old = json.loads(cp.read_text())
        if {k: v for k, v in old.items() if k not in ('out', 'resume')} != {k: v for k, v in config.items() if k not in ('out', 'resume')}:
            raise ValueError('Resume configuration differs')
    elif csvpath.exists():
        raise ValueError('CSV without matching configuration')
    write_json(cp, config)
    rows = list(csv.DictReader(csvpath.open())) if a.resume and csvpath.exists() else []
    self_test()
    for run in range(a.runs):
        hpath = out / f'heuristic_route_run{run}.json'
        if a.resume and hpath.exists():
            heur = json.loads(hpath.read_text())
            validate_routes(heur)
        else:
            heur = build_routes(a.grid, a.seed + run, a.route_attempts, a.route_depth)
            write_json(hpath, heur)
        exact = None
        if 'ExactRoute-CP' in selected:
            ep = out / f'exact_route_stage_run{run}.json'
            if a.resume and ep.exists():
                exact = json.loads(ep.read_text())
            else:
                exact = bounded_route_joint({'mode': 'route', 'grid': a.grid, 'horizon': a.route_depth, 'budget': a.route_budget, 'seed': a.seed + run}, out / 'checkpoints' / f'route_r{run}.json', a.grace)
                write_json(ep, exact)
            validate_exact_result(exact)
        for n in a.neurons:
            adj = graph(n, a.max_sources, a.seed + run)
            cap = 2 * math.ceil(n / (a.grid * a.grid))
            for lo, hi in a.delay_profiles:
                delays = delays_for(n, lo, hi, a.seed + 1000 + run)
                steps = a.window + 2 * int(delays.max()) + 2
                ctx = cost_context(adj, delays, heur, steps, a.flit_bits, a.alpha)
                for method in selected:
                    key = f'n{n}_d{lo}-{hi}_r{run}_{method}'
                    previous = [r for r in rows if r.get('case_id') == key]
                    if a.resume and len(previous) == len(PATTERNS) and ({r['pattern'] for r in previous} == set(PATTERNS)):
                        continue
                    rows = [r for r in rows if r.get('case_id') != key]
                    meta = {'study': 'compilation', 'study_schema': STUDY_SCHEMA, 'case_id': key, 'run': run, 'delay_profile': f'{lo}:{hi}', 'method': method, 'routing_horizon': a.route_depth}
                    m = None
                    route = None
                    route_seconds = None
                    joint_seconds = None
                    map_seconds = None
                    fields = {}
                    info = {}
                    route_status = 'NOT_SEPARATE'
                    if method == 'Joint-CP':
                        meta['routing_horizon'] = a.joint_horizon or int(adj.any(axis=1).sum()) * (a.grid * a.grid + 1) + 1
                        result = bounded_route_joint({'mode': 'application_joint', 'grid': a.grid, 'horizon': a.joint_horizon or int(adj.any(axis=1).sum()) * (a.grid * a.grid + 1) + 1, 'budget': a.joint_budget, 'seed': a.seed + run, 'adj': adj, 'delays': delays, 'cap': cap, 'steps': steps, 'flit': a.flit_bits, 'ctx': ctx}, out / 'checkpoints' / (key + '.json'), a.grace)
                        validate_exact_result(result)
                        route = result.pop('route', None)
                        m = np.asarray(result.pop('mapping'), int) if 'mapping' in result else None
                        joint_seconds = result['stage_seconds']
                        info = result
                        fields.update(cp_stage_fields(result, 'joint'))
                        total = joint_seconds
                        deployment = joint_seconds
                    else:
                        if method == 'ExactRoute-CP':
                            route = exact.get('route')
                            route_seconds = exact['stage_seconds']
                            route_status = exact['status']
                            fields.update(cp_stage_fields(exact, 'route'))
                        else:
                            route = heur
                            route_seconds = heur['routing_seconds']
                            route_status = 'FEASIBLE_HEURISTIC'
                        if route is None:
                            info = {'status': 'NO_ROUTE_' + route_status}
                            map_seconds = None
                            fields['mapping_status'] = 'NOT_RUN_NO_ROUTE'
                        elif method == 'HeuristicRoute-DARSA':
                            m, info = search_mapping(adj, delays, route, cap, steps, a.flit_bits, ctx, a.seed + run, a.iterations, a.budget, 'DARSA')
                            map_seconds = info['mapping_search_seconds']
                            fields['mapping_status'] = info['status']
                        else:
                            info = cp_search((adj, delays, route, cap, steps, a.flit_bits, ctx, a.budget, a.seed + run), out / 'checkpoints' / (key + '.json'), a.grace)
                            validate_exact_result(info)
                            fields.update(cp_stage_fields(info, 'mapping'))
                            m = np.asarray(info.pop('mapping'), int) if 'mapping' in info else None
                            map_seconds = info['mapping_search_seconds']
                        total = route_seconds + (map_seconds or 0.0)
                        deployment = map_seconds
                    fields.update(route_status=route_status, routing_seconds=route_seconds, mapping_search_seconds=map_seconds, joint_search_seconds=joint_seconds, total_compilation_seconds=total, deployment_seconds=deployment, mapping_budget_seconds=a.budget, route_budget_seconds=a.route_budget, joint_budget_seconds=a.joint_budget, optimality_scope='application locality objective within finite horizon' if method == 'Joint-CP' else 'placement conditional on selected routes')
                    info.update(ctx)
                    if m is not None:
                        route = dict(route)
                        route['routing_seconds'] = route_seconds if route_seconds is not None else 0.0
                        if method == 'Joint-CP':
                            evaluated, detail = evaluate_application_case(adj, delays, m, route, cap, a, meta, info)
                        else:
                            actual = mapping_cost(m, adj, delays, route, steps, a.flit_bits, ctx)
                            if actual != info['objective_integer']:
                                raise AssertionError(f'{key}: solver/evaluator objective mismatch')
                            info['objective_normalized'] = actual / ctx['normalizer']
                            info['objective_scope'] = 'normalized frame / memory resource objective'
                            info['mapping_search_seconds'] = map_seconds if map_seconds is not None else 0.0
                            evaluated, detail = evaluate_case(adj, delays, m, route, cap, a, meta, info)
                            locality = joint_locality_metrics(adj, m, a.grid)
                            for row in evaluated:
                                row.update(locality)
                                row['objective_normalized_scope'] = 'common resource score, optimized by fixed-substrate mappers'
                        for r in evaluated:
                            r.update(fields)
                            r['total_compilation_seconds'] = total + r['final_calendar_seconds']
                            if deployment is not None:
                                r['deployment_seconds'] = deployment + r['final_calendar_seconds']
                        rows.extend(evaluated)
                        detail['compilation'].update(fields)
                        detail['route'] = route
                    else:
                        for pattern in PATTERNS:
                            rows.append({'schema': SCHEMA, **meta, 'neurons': n, 'grid': a.grid, 'T': route['T'] if route else '', 'pattern': pattern, 'frame_cycles': '', 'total_provisioned_bits': '', 'objective_normalized': '', 'omega': '', 'final_calendar_seconds': '', **info, **fields})
                    write_csv(csvpath, rows)
def main():
    import sys
    import tempfile
    import shutil
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument('--out', required=True)
    cli.add_argument('--resume', action='store_true')
    if '--help' in sys.argv or '-h' in sys.argv:
        run_experiment()
        return
    output, remaining = cli.parse_known_args()
    if output.resume:
        cli.error('Resume is unavailable for temporary solver runs')
    destination = Path(output.out)
    destination.mkdir(parents=True, exist_ok=True)
    original = sys.argv[:]
    with tempfile.TemporaryDirectory(prefix='chronoc_compilation_') as temporary:
        try:
            sys.argv = [original[0], '--out', temporary, *remaining]
            run_experiment()
        finally:
            sys.argv = original
            result = Path(temporary) / 'metrics.csv'
            if result.exists():
                shutil.copyfile(result, destination / 'metrics.csv')
if __name__ == '__main__':
    main()
