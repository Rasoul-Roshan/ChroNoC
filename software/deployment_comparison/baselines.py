"""Evaluate adapted deployment baselines on the common TDM communication model."""
import argparse, heapq, json, math, random, time
from pathlib import Path
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
from types import SimpleNamespace
import numpy as np
from bundle_io import require_replay_traces, load_bundle
from darsa import make_workload, validate_mapping, evaluate, save_result
PROVENANCE = {}
PROVENANCE['sneap'] = {'method': 'sneap', 'status': 'adapted baseline'}

def _sneap_namespace():

    def set_global_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)


    class WorkloadGenerator:

        def __init__(self, num_pes=16, num_agents=64):
            self.num_pes = num_pes
            self.num_agents = num_agents

        def generate_workload(self):
            workload = {'agents': []}
            num_shared_sources = max(self.num_pes, self.num_agents * 2, 20)
            shared_source_pool = [f'src_{i}' for i in range(num_shared_sources)]
            for agent_id in range(self.num_agents):
                dest = f'dest_a{agent_id}'
                num_sources = random.randint(1, 20)
                sources = random.sample(shared_source_pool, num_sources)
                workload['agents'].append({'agent_id': f'a_{agent_id}', 'destination': dest, 'sources': sources})
            return workload

    @dataclass
    class WeightedGraph:
        nodes: list
        node_weight: dict
        adj: dict

    def build_task_graph(workload):
        tasks = set()
        edge_weights = defaultdict(int)
        for agent in workload['agents']:
            d = agent['destination']
            tasks.add(d)
            for s in agent['sources']:
                tasks.add(s)
                a, b = sorted((s, d))
                if a != b:
                    edge_weights[a, b] += 1
        tasks = sorted(tasks)
        adj = {t: {} for t in tasks}
        for (a, b), w in edge_weights.items():
            adj[a][b] = adj[a].get(b, 0) + w
            adj[b][a] = adj[b].get(a, 0) + w
        return WeightedGraph(tasks, {t: 1 for t in tasks}, adj)

    def graph_cut_weight(graph: WeightedGraph, labels):
        cut = 0
        seen = set()
        for u in graph.nodes:
            for v, w in graph.adj[u].items():
                key = tuple(sorted((u, v), key=str))
                if key in seen:
                    continue
                seen.add(key)
                if labels[u] != labels[v]:
                    cut += w
        return float(cut)

    def partition_communication_matrix(graph: WeightedGraph, labels, k):
        C = np.zeros((k, k), dtype=np.float64)
        seen = set()
        for u in graph.nodes:
            for v, w in graph.adj[u].items():
                key = tuple(sorted((u, v), key=str))
                if key in seen:
                    continue
                seen.add(key)
                a, b = (labels[u], labels[v])
                if a != b:
                    C[a, b] += w
                    C[b, a] += w
        return C


    @dataclass
    class CoarsenStep:
        fine_graph: WeightedGraph
        fine_to_coarse: dict

    class SneapMultilevelPartitioner:

        def __init__(self, core_capacity, num_parts, refinement_patience=10, seed=42):
            self.capacity = int(core_capacity)
            self.k = int(num_parts)
            self.patience = int(refinement_patience)
            self.rng = random.Random(seed)
            self.stage_times = {'partition_coarsen_time': 0.0, 'partition_initial_time': 0.0, 'partition_uncoarsen_refine_time': 0.0}

        def _coarsen_once(self, graph: WeightedGraph):
            order = list(graph.nodes)
            self.rng.shuffle(order)
            unmatched = set(graph.nodes)
            groups = []
            for u in order:
                if u not in unmatched:
                    continue
                candidates = []
                for v, ew in graph.adj[u].items():
                    if v in unmatched and v != u:
                        if graph.node_weight[u] + graph.node_weight[v] <= self.capacity:
                            candidates.append((ew, str(v), v))
                if candidates:
                    candidates.sort(reverse=True)
                    v = candidates[0][2]
                    groups.append([u, v])
                    unmatched.remove(u)
                    unmatched.remove(v)
                else:
                    groups.append([u])
                    unmatched.remove(u)
            fine_to_coarse = {}
            coarse_nodes = []
            coarse_weight = {}
            for cid, members in enumerate(groups):
                coarse_nodes.append(cid)
                coarse_weight[cid] = sum((graph.node_weight[m] for m in members))
                for m in members:
                    fine_to_coarse[m] = cid
            coarse_adj = {c: {} for c in coarse_nodes}
            seen = set()
            for u in graph.nodes:
                for v, w in graph.adj[u].items():
                    key = tuple(sorted((str(u), str(v))))
                    if key in seen:
                        continue
                    seen.add(key)
                    cu, cv = (fine_to_coarse[u], fine_to_coarse[v])
                    if cu != cv:
                        coarse_adj[cu][cv] = coarse_adj[cu].get(cv, 0) + w
                        coarse_adj[cv][cu] = coarse_adj[cv].get(cu, 0) + w
            return (WeightedGraph(coarse_nodes, coarse_weight, coarse_adj), fine_to_coarse)

        def _initial_partition(self, graph: WeightedGraph):
            if len(graph.nodes) < self.k:
                raise ValueError('Coarsest graph has fewer nodes than requested partitions')
            labels = {}
            loads = [0] * self.k
            unassigned = set(graph.nodes)
            for p in range(self.k):
                candidates = [n for n in unassigned if graph.node_weight[n] <= self.capacity]
                if not candidates:
                    raise RuntimeError('Unable to seed all SNEAP partitions under capacity')
                seed = self.rng.choice(candidates)
                labels[seed] = p
                loads[p] += graph.node_weight[seed]
                unassigned.remove(seed)
            progress = True
            while unassigned and progress:
                progress = False
                for p in range(self.k):
                    best = None
                    for n in unassigned:
                        if loads[p] + graph.node_weight[n] > self.capacity:
                            continue
                        conn = sum((w for nbr, w in graph.adj[n].items() if nbr in labels and labels[nbr] == p))
                        if conn > 0:
                            cand = (conn, -graph.node_weight[n], str(n), n)
                            if best is None or cand > best:
                                best = cand
                    if best is not None:
                        n = best[-1]
                        labels[n] = p
                        loads[p] += graph.node_weight[n]
                        unassigned.remove(n)
                        progress = True
                        if not unassigned:
                            break
            while unassigned:
                n = next(iter(unassigned))
                options = []
                for p in range(self.k):
                    if loads[p] + graph.node_weight[n] <= self.capacity:
                        conn = sum((w for nbr, w in graph.adj[n].items() if nbr in labels and labels[nbr] == p))
                        options.append((conn, -loads[p], -p, p))
                if not options:
                    raise RuntimeError('SNEAP initial partition cannot satisfy core capacity')
                p = max(options)[-1]
                labels[n] = p
                loads[p] += graph.node_weight[n]
                unassigned.remove(n)
            return labels

        def _refine(self, graph: WeightedGraph, labels):
            labels = dict(labels)
            loads = [0] * self.k
            for n in graph.nodes:
                loads[labels[n]] += graph.node_weight[n]
            best_labels = dict(labels)
            best_cut = graph_cut_weight(graph, labels)
            non_improve = 0
            max_moves = max(1, 4 * len(graph.nodes))
            for _ in range(max_moves):
                heap = []
                for v in graph.nodes:
                    cur = labels[v]
                    internal = 0.0
                    external_by_part = defaultdict(float)
                    for u, w in graph.adj[v].items():
                        pu = labels[u]
                        if pu == cur:
                            internal += w
                        else:
                            external_by_part[pu] += w
                    external_sum = sum(external_by_part.values())
                    if not external_by_part or external_sum < internal:
                        continue
                    target, best_ed = max(external_by_part.items(), key=lambda kv: kv[1])
                    if loads[target] + graph.node_weight[v] > self.capacity:
                        continue
                    gain = best_ed - internal
                    heapq.heappush(heap, (-gain, str(v), v, target))
                if not heap:
                    break
                _neg_gain, _sv, v, target = heapq.heappop(heap)
                old = labels[v]
                labels[v] = target
                loads[old] -= graph.node_weight[v]
                loads[target] += graph.node_weight[v]
                cut = graph_cut_weight(graph, labels)
                if cut < best_cut:
                    best_cut = cut
                    best_labels = dict(labels)
                    non_improve = 0
                else:
                    non_improve += 1
                    if non_improve >= self.patience:
                        labels = best_labels
                        break
            return dict(best_labels)

        def partition(self, graph: WeightedGraph):
            hierarchy = []
            current = graph
            target_nodes = max(self.k, 2 * self.k)
            t0 = time.perf_counter()
            while len(current.nodes) > target_nodes:
                coarse, mapping = self._coarsen_once(current)
                if len(coarse.nodes) >= len(current.nodes) or len(coarse.nodes) < self.k:
                    break
                hierarchy.append(CoarsenStep(current, mapping))
                current = coarse
            self.stage_times['partition_coarsen_time'] = time.perf_counter() - t0
            t0 = time.perf_counter()
            while True:
                try:
                    labels = self._initial_partition(current)
                    break
                except RuntimeError:
                    if not hierarchy:
                        raise
                    step = hierarchy.pop()
                    current = step.fine_graph
            self.stage_times['partition_initial_time'] = time.perf_counter() - t0
            t0 = time.perf_counter()
            labels = self._refine(current, labels)
            for step in reversed(hierarchy):
                fine_labels = {n: labels[step.fine_to_coarse[n]] for n in step.fine_graph.nodes}
                labels = self._refine(step.fine_graph, fine_labels)
                current = step.fine_graph
            self.stage_times['partition_uncoarsen_refine_time'] = time.perf_counter() - t0
            return labels

    class SneapSAMapper:

        def __init__(self, grid_dim, iterations=20000, T0=1.0, Tmin=0.001, seed=42):
            self.N = int(grid_dim)
            self.num_pes = self.N * self.N
            self.iterations = int(iterations)
            self.T0 = float(T0)
            self.Tmin = float(Tmin)
            self.rng = random.Random(seed)

        def _coord(self, pe):
            return divmod(int(pe), self.N)

        def _prepare_cost(self, C):
            C = np.asarray(C, dtype=np.float64)
            a, b = np.nonzero(np.triu(C, k=1) > 0)
            w = C[a, b].astype(np.float64, copy=False)
            denom = float(w.sum())
            return (a.astype(np.int32), b.astype(np.int32), w, denom)

        def _cost_prepared(self, assignment, edge_a, edge_b, edge_w, denom):
            if denom <= 0.0 or edge_w.size == 0:
                return 0.0
            a = np.asarray(assignment, dtype=np.int32)
            rows = a // self.N
            cols = a % self.N
            dist = np.abs(rows[edge_a] - rows[edge_b]) + np.abs(cols[edge_a] - cols[edge_b])
            return float(np.dot(edge_w, dist) / denom)

        def map(self, C):
            k = C.shape[0]
            if k > self.num_pes:
                raise ValueError('More partitions than PEs')
            edge_a, edge_b, edge_w, denom = self._prepare_cost(C)
            current = self.rng.sample(range(self.num_pes), k)
            current_cost = self._cost_prepared(current, edge_a, edge_b, edge_w, denom)
            best = list(current)
            best_cost = current_cost
            for it in range(max(1, self.iterations)):
                frac = it / max(1, self.iterations - 1)
                T = self.T0 * (self.Tmin / self.T0) ** frac
                cand = list(current)
                if k == self.num_pes or self.rng.random() < 0.8:
                    i, j = self.rng.sample(range(k), 2)
                    cand[i], cand[j] = (cand[j], cand[i])
                else:
                    i = self.rng.randrange(k)
                    unused = list(set(range(self.num_pes)) - set(cand))
                    if not unused:
                        continue
                    cand[i] = self.rng.choice(unused)
                cand_cost = self._cost_prepared(cand, edge_a, edge_b, edge_w, denom)
                delta = cand_cost - current_cost
                if delta <= 0 or self.rng.random() < math.exp(-delta / max(T, 1e-12)):
                    current, current_cost = (cand, cand_cost)
                    if current_cost < best_cost:
                        best, best_cost = (list(current), current_cost)
            return ({p: int(best[p]) for p in range(k)}, float(best_cost))

    def final_task_mapping(part_labels, part_to_pe):
        return {task: int(part_to_pe[p]) for task, p in part_labels.items()}
    return SimpleNamespace(**locals())
PROVENANCE['spinemap'] = {'method': 'spinemap', 'status': 'adapted baseline'}

def _spinemap_namespace():

    def set_global_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)


    class WorkloadGenerator:

        def __init__(self, num_pes=16, num_agents=64):
            self.num_pes = num_pes
            self.num_agents = num_agents

        def generate_workload(self):
            workload = {'agents': []}
            num_shared_sources = max(self.num_pes, self.num_agents * 2, 20)
            shared_source_pool = [f'src_{i}' for i in range(num_shared_sources)]
            for agent_id in range(self.num_agents):
                dest = f'dest_a{agent_id}'
                num_sources = random.randint(1, 20)
                sources = random.sample(shared_source_pool, num_sources)
                workload['agents'].append({'agent_id': f'a_{agent_id}', 'destination': dest, 'sources': sources})
            return workload

    @dataclass
    class WeightedGraph:
        nodes: list
        adj: dict

    def build_task_graph(workload):
        tasks = set()
        edge_weights = defaultdict(int)
        for agent in workload['agents']:
            d = agent['destination']
            tasks.add(d)
            for s in agent['sources']:
                tasks.add(s)
                a, b = sorted((s, d))
                if a != b:
                    edge_weights[a, b] += 1
        tasks = sorted(tasks)
        adj = {t: {} for t in tasks}
        for (a, b), w in edge_weights.items():
            adj[a][b] = adj[a].get(b, 0) + w
            adj[b][a] = adj[b].get(a, 0) + w
        return WeightedGraph(tasks, adj)

    def graph_cut_weight(graph: WeightedGraph, labels):
        cut = 0.0
        seen = set()
        for u in graph.nodes:
            for v, w in graph.adj[u].items():
                key = tuple(sorted((u, v)))
                if key in seen:
                    continue
                seen.add(key)
                if labels[u] != labels[v]:
                    cut += w
        return float(cut)

    def partition_communication_matrix(graph: WeightedGraph, labels, k):
        C = np.zeros((k, k), dtype=np.float64)
        seen = set()
        for u in graph.nodes:
            for v, w in graph.adj[u].items():
                key = tuple(sorted((u, v)))
                if key in seen:
                    continue
                seen.add(key)
                a, b = (labels[u], labels[v])
                if a != b:
                    C[a, b] += w
                    C[b, a] += w
        return C


    class SpiNeCluster:

        def __init__(self, num_clusters, capacity, seed=42, pair_passes=1):
            self.k = int(num_clusters)
            self.capacity = int(capacity)
            self.rng = random.Random(seed)
            self.pair_passes = int(pair_passes)
            self.stage_times = {'partition_initialization_time': 0.0, 'partition_pairwise_refinement_time': 0.0}

        def _initial_even_partition(self, graph):
            tasks = list(graph.nodes)
            self.rng.shuffle(tasks)
            labels = {}
            loads = [0] * self.k
            members = [set() for _ in range(self.k)]
            for t in tasks:
                candidates = [p for p in range(self.k) if loads[p] < self.capacity]
                if not candidates:
                    raise RuntimeError('SpiNeCluster cannot satisfy cluster capacity')
                min_load = min((loads[p] for p in candidates))
                least = [p for p in candidates if loads[p] == min_load]
                q = self.rng.choice(least)
                labels[t] = q
                loads[q] += 1
                members[q].add(t)
            return (labels, loads, members)

        @staticmethod
        def _move_delta(graph, labels, node, target):
            old = labels[node]
            if old == target:
                return 0.0
            delta = 0.0
            for v, w in graph.adj[node].items():
                lv = labels[v]
                if lv == old:
                    delta += w
                elif lv == target:
                    delta -= w
            return float(delta)

        @classmethod
        def _swap_delta(cls, graph, labels, u, v):
            a, b = (labels[u], labels[v])
            if a == b:
                return 0.0
            d = cls._move_delta(graph, labels, u, b) + cls._move_delta(graph, labels, v, a)
            wuv = graph.adj[u].get(v, 0.0)
            if wuv:
                d += 2.0 * wuv
            return float(d)

        def _improve_pair(self, graph, labels, loads, members, a, b):
            selected = set()
            improved_any = False
            while True:
                A = [n for n in sorted(members[a]) if n not in selected]
                B = [n for n in sorted(members[b]) if n not in selected]
                if not A and (not B):
                    break
                best = None
                if loads[b] < self.capacity:
                    for u in A:
                        d = self._move_delta(graph, labels, u, b)
                        if best is None or d < best[0]:
                            best = (d, 'move_ab', u, None)
                if loads[a] < self.capacity:
                    for v in B:
                        d = self._move_delta(graph, labels, v, a)
                        if best is None or d < best[0]:
                            best = (d, 'move_ba', v, None)
                for u in A:
                    for v in B:
                        d = self._swap_delta(graph, labels, u, v)
                        if best is None or d < best[0]:
                            best = (d, 'swap', u, v)
                if best is None or best[0] >= -1e-12:
                    break
                _delta, kind, u, v = best
                if kind == 'move_ab':
                    labels[u] = b
                    members[a].remove(u)
                    members[b].add(u)
                    loads[a] -= 1
                    loads[b] += 1
                    selected.add(u)
                elif kind == 'move_ba':
                    labels[u] = a
                    members[b].remove(u)
                    members[a].add(u)
                    loads[b] -= 1
                    loads[a] += 1
                    selected.add(u)
                else:
                    labels[u], labels[v] = (b, a)
                    members[a].remove(u)
                    members[b].remove(v)
                    members[a].add(v)
                    members[b].add(u)
                    selected.add(u)
                    selected.add(v)
                improved_any = True
            return improved_any

        def partition(self, graph):
            t0 = time.perf_counter()
            labels, loads, members = self._initial_even_partition(graph)
            self.stage_times['partition_initialization_time'] = time.perf_counter() - t0
            t0 = time.perf_counter()
            for _pass in range(max(1, self.pair_passes)):
                any_change = False
                for a, b in combinations(range(self.k), 2):
                    any_change |= self._improve_pair(graph, labels, loads, members, a, b)
                if not any_change:
                    break
            self.stage_times['partition_pairwise_refinement_time'] = time.perf_counter() - t0
            return labels

    class SpiNePlacerPSO:

        def __init__(self, workload, graph, part_labels, grid_dim, particles=20, iterations=40, phi1=1.0, phi2=1.0, seed=43):
            self.workload = workload
            self.graph = graph
            self.part_labels = part_labels
            self.N = int(grid_dim)
            self.num_pes = self.N * self.N
            self.np = int(particles)
            self.iterations = int(iterations)
            self.phi1 = float(phi1)
            self.phi2 = float(phi2)
            self.rng = np.random.default_rng(seed)
            self.py_rng = random.Random(seed)
            self.k = max(part_labels.values()) + 1 if part_labels else 0
            if self.k > self.num_pes:
                raise ValueError('SpiNeMap requires at most one cluster per PE')
            self.fitness_evaluations = 0
            self.fitness_calls = 0
            self.fitness_cache = {}
            self.C = partition_communication_matrix(graph, part_labels, self.k)

        def _onehot_from_assignment(self, assignment):
            X = np.zeros((self.k, self.num_pes), dtype=np.float64)
            for c, pe in enumerate(assignment):
                X[c, int(pe)] = 1.0
            return X

        def _assignment_from_onehot(self, X):
            return [int(np.argmax(X[c])) for c in range(self.k)]

        def _random_particle(self):
            assignment = self.py_rng.sample(range(self.num_pes), self.k)
            return self._onehot_from_assignment(assignment)

        def _repair_binary(self, binary, probs):
            score = probs + binary.astype(np.float64)
            score = score + self.rng.random(score.shape) * 1e-09
            order = np.argsort(score, axis=None)[::-1]
            used_cluster = np.zeros(self.k, dtype=bool)
            used_pe = np.zeros(self.num_pes, dtype=bool)
            assignment = np.full(self.k, -1, dtype=np.int64)
            remaining = self.k
            for flat in order:
                c, pe = divmod(int(flat), self.num_pes)
                if used_cluster[c] or used_pe[pe]:
                    continue
                assignment[c] = pe
                used_cluster[c] = True
                used_pe[pe] = True
                remaining -= 1
                if remaining == 0:
                    break
            if remaining:
                unused = list(np.flatnonzero(~used_pe))
                for c in np.flatnonzero(~used_cluster):
                    assignment[int(c)] = int(unused.pop())
            return self._onehot_from_assignment(assignment)

        def _task_mapping(self, X):
            assignment = self._assignment_from_onehot(X)
            return {task: assignment[p] for task, p in self.part_labels.items()}

        def _analytical_avg_hop(self, X):
            assignment = self._assignment_from_onehot(X)
            numerator = 0.0
            denominator = 0.0
            for a in range(self.k):
                ra, ca = divmod(assignment[a], self.N)
                for b in range(a + 1, self.k):
                    w = self.C[a, b]
                    if w <= 0:
                        continue
                    rb, cb = divmod(assignment[b], self.N)
                    numerator += w * (abs(ra - rb) + abs(ca - cb))
                    denominator += w
            return numerator / denominator if denominator else 0.0

        def _fitness(self, X):
            self.fitness_calls += 1
            assignment = tuple(self._assignment_from_onehot(X))
            key = assignment
            if key in self.fitness_cache:
                return self.fitness_cache[key]
            self.fitness_evaluations += 1
            value = self._analytical_avg_hop(X)
            self.fitness_cache[key] = float(value)
            return float(value)

        def run(self):
            X = [self._random_particle() for _ in range(self.np)]
            V = [np.zeros((self.k, self.num_pes), dtype=np.float64) for _ in range(self.np)]
            costs = [self._fitness(x) for x in X]
            P = [x.copy() for x in X]
            Pcost = list(costs)
            g_idx = int(np.argmin(Pcost))
            G = P[g_idx].copy()
            Gcost = float(Pcost[g_idx])
            for _it in range(self.iterations):
                for i in range(self.np):
                    V[i] = V[i] + self.phi1 * (P[i] - X[i]) + self.phi2 * (G - X[i])
                    V[i] = np.clip(V[i], -8.0, 8.0)
                    probs = 1.0 / (1.0 + np.exp(-V[i]))
                    sampled = self.rng.random(probs.shape) < probs
                    Xnew = self._repair_binary(sampled, probs)
                    cnew = self._fitness(Xnew)
                    X[i] = Xnew
                    costs[i] = cnew
                    if cnew < Pcost[i]:
                        P[i] = Xnew.copy()
                        Pcost[i] = cnew
                        if cnew < Gcost:
                            G = Xnew.copy()
                            Gcost = float(cnew)
            return (G, Gcost)

    def final_task_mapping(part_labels, Xbest):
        assignment = [int(np.argmax(Xbest[c])) for c in range(Xbest.shape[0])]
        return {task: assignment[p] for task, p in part_labels.items()}
    return SimpleNamespace(**locals())
PROVENANCE['neumap'] = {'method': 'neumap', 'status': 'adapted baseline'}

def _neumap_namespace():

    def set_global_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)


    class WorkloadGenerator:

        def __init__(self, num_pes=16, num_agents=64):
            self.num_pes = int(num_pes)
            self.num_agents = int(num_agents)

        def generate_workload(self):
            workload = {'agents': []}
            num_shared_sources = max(self.num_pes, self.num_agents * 2, 20)
            shared_source_pool = [f'src_{i}' for i in range(num_shared_sources)]
            for agent_id in range(self.num_agents):
                dest = f'dest_a{agent_id}'
                num_sources = random.randint(1, 20)
                sources = random.sample(shared_source_pool, num_sources)
                workload['agents'].append({'agent_id': f'a_{agent_id}', 'destination': dest, 'sources': sources})
            return workload

    @dataclass
    class WeightedGraph:
        nodes: list
        node_weight: dict
        adj: dict

    @dataclass
    class NeuMapProfile:
        graph: WeightedGraph
        directed_edges: dict
        layers: list
        source_rates: dict

    def build_neumap_profile(workload):
        source_order = []
        dest_order = []
        seen_sources = set()
        seen_dests = set()
        directed = defaultdict(float)
        for agent in workload['agents']:
            d = agent['destination']
            if d not in seen_dests:
                seen_dests.add(d)
                dest_order.append(d)
            for s in agent['sources']:
                if s not in seen_sources:
                    seen_sources.add(s)
                    source_order.append(s)
                if s != d:
                    directed[s, d] += 1.0
        nodes = source_order + [d for d in dest_order if d not in seen_sources]
        node_weight = {n: 1 for n in nodes}
        adj = {n: {} for n in nodes}
        for (u, v), w in directed.items():
            adj[u][v] = adj[u].get(v, 0.0) + w
            adj[v][u] = adj[v].get(u, 0.0) + w
        layers = [list(nodes)] if seen_sources & seen_dests else [list(source_order), list(dest_order)]
        source_rates = {s: 1.0 for s in source_order}
        return NeuMapProfile(graph=WeightedGraph(nodes, node_weight, adj), directed_edges=dict(directed), layers=layers, source_rates=source_rates)

    def graph_cut_weight(graph: WeightedGraph, labels):
        cut = 0.0
        seen = set()
        for u in graph.nodes:
            for v, w in graph.adj[u].items():
                key = tuple(sorted((str(u), str(v))))
                if key in seen:
                    continue
                seen.add(key)
                if labels[u] != labels[v]:
                    cut += float(w)
        return float(cut)

    def induced_subgraph(graph: WeightedGraph, nodes):
        nodes = list(nodes)
        node_set = set(nodes)
        adj = {u: {} for u in nodes}
        for u in nodes:
            for v, w in graph.adj[u].items():
                if v in node_set:
                    adj[u][v] = float(w)
        return WeightedGraph(nodes, {u: graph.node_weight[u] for u in nodes}, adj)

    def directed_cluster_matrix(profile: NeuMapProfile, labels, k):
        C = np.zeros((k, k), dtype=np.float64)
        for (u, v), w in profile.directed_edges.items():
            a, b = (int(labels[u]), int(labels[v]))
            if a != b:
                C[a, b] += float(w)
        return C

    @dataclass
    class CoarsenStep:
        fine_graph: WeightedGraph
        fine_to_coarse: dict

    class MultiLevelPartitioner:

        def __init__(self, core_capacity, num_parts, refinement_patience=10, seed=42):
            self.capacity = int(core_capacity)
            self.k = int(num_parts)
            self.patience = int(refinement_patience)
            self.rng = random.Random(seed)
            self.stage_times = {'coarsen': 0.0, 'initial': 0.0, 'refine': 0.0}

        def _coarsen_once(self, graph: WeightedGraph):
            order = list(graph.nodes)
            self.rng.shuffle(order)
            unmatched = set(graph.nodes)
            groups = []
            for u in order:
                if u not in unmatched:
                    continue
                candidates = []
                for v, ew in graph.adj[u].items():
                    if v in unmatched and v != u:
                        if graph.node_weight[u] + graph.node_weight[v] <= self.capacity:
                            candidates.append((float(ew), str(v), v))
                if candidates:
                    candidates.sort(reverse=True)
                    v = candidates[0][2]
                    groups.append([u, v])
                    unmatched.remove(u)
                    unmatched.remove(v)
                else:
                    groups.append([u])
                    unmatched.remove(u)
            fine_to_coarse = {}
            coarse_nodes = []
            coarse_weight = {}
            for cid, members in enumerate(groups):
                coarse_nodes.append(cid)
                coarse_weight[cid] = sum((graph.node_weight[m] for m in members))
                for m in members:
                    fine_to_coarse[m] = cid
            coarse_adj = {c: {} for c in coarse_nodes}
            seen = set()
            for u in graph.nodes:
                for v, w in graph.adj[u].items():
                    key = tuple(sorted((str(u), str(v))))
                    if key in seen:
                        continue
                    seen.add(key)
                    cu, cv = (fine_to_coarse[u], fine_to_coarse[v])
                    if cu != cv:
                        coarse_adj[cu][cv] = coarse_adj[cu].get(cv, 0.0) + float(w)
                        coarse_adj[cv][cu] = coarse_adj[cv].get(cu, 0.0) + float(w)
            return (WeightedGraph(coarse_nodes, coarse_weight, coarse_adj), fine_to_coarse)

        def _initial_partition_reference(self, graph: WeightedGraph):
            if self.k <= 0:
                raise ValueError('num_parts must be positive')
            if len(graph.nodes) < self.k:
                raise ValueError('Coarsest graph has fewer nodes than requested partitions')
            labels = {}
            loads = [0] * self.k
            unassigned = set(graph.nodes)
            for p in range(self.k):
                candidates = [n for n in unassigned if graph.node_weight[n] <= self.capacity]
                if not candidates:
                    raise RuntimeError('Unable to seed all multilevel partitions under capacity')
                self.rng.shuffle(candidates)
                seed = max(candidates, key=lambda n: sum(graph.adj[n].values()))
                labels[seed] = p
                loads[p] += graph.node_weight[seed]
                unassigned.remove(seed)
            while unassigned:
                best_global = None
                for n in list(unassigned):
                    for p in range(self.k):
                        if loads[p] + graph.node_weight[n] > self.capacity:
                            continue
                        conn = sum((float(w) for nbr, w in graph.adj[n].items() if nbr in labels and labels[nbr] == p))
                        cand = (conn, -loads[p], -p, str(n), n, p)
                        if best_global is None or cand > best_global:
                            best_global = cand
                if best_global is None:
                    raise RuntimeError('Initial partition cannot satisfy core capacity')
                n, p = (best_global[-2], best_global[-1])
                labels[n] = p
                loads[p] += graph.node_weight[n]
                unassigned.remove(n)
            return labels

        def _refine_reference(self, graph: WeightedGraph, labels):
            labels = dict(labels)
            loads = [0] * self.k
            counts = [0] * self.k
            for n in graph.nodes:
                loads[labels[n]] += graph.node_weight[n]
                counts[labels[n]] += 1
            best_labels = dict(labels)
            best_cut = graph_cut_weight(graph, labels)
            no_improve = 0
            max_moves = max(1, 4 * len(graph.nodes))
            for _ in range(max_moves):
                heap = []
                for v in graph.nodes:
                    cur = labels[v]
                    if counts[cur] <= 1:
                        continue
                    internal = 0.0
                    external = defaultdict(float)
                    for u, w in graph.adj[v].items():
                        pu = labels[u]
                        if pu == cur:
                            internal += float(w)
                        else:
                            external[pu] += float(w)
                    for target, ext in external.items():
                        if loads[target] + graph.node_weight[v] > self.capacity:
                            continue
                        gain = ext - internal
                        heapq.heappush(heap, (-gain, str(v), v, target))
                if not heap:
                    break
                _neg_gain, _sv, v, target = heapq.heappop(heap)
                old = labels[v]
                labels[v] = target
                loads[old] -= graph.node_weight[v]
                loads[target] += graph.node_weight[v]
                counts[old] -= 1
                counts[target] += 1
                cut = graph_cut_weight(graph, labels)
                if cut < best_cut:
                    best_cut = cut
                    best_labels = dict(labels)
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        break
            return dict(best_labels)

        def _initial_partition_fast(self, graph: WeightedGraph):
            if self.k <= 0:
                raise ValueError('num_parts must be positive')
            if len(graph.nodes) < self.k:
                raise ValueError('Coarsest graph has fewer nodes than requested partitions')
            nodes = list(graph.nodes)
            nodes.sort(key=lambda n: (sum(graph.adj[n].values()), -graph.node_weight[n], str(n)), reverse=True)
            labels = {}
            loads = [0] * self.k
            counts = [0] * self.k
            remaining = []
            p = 0
            for n in nodes:
                if p < self.k and graph.node_weight[n] <= self.capacity:
                    labels[n] = p
                    loads[p] += graph.node_weight[n]
                    counts[p] += 1
                    p += 1
                else:
                    remaining.append(n)
            if p < self.k:
                raise RuntimeError('Unable to seed all multilevel partitions under capacity')
            for n in remaining:
                w_n = graph.node_weight[n]
                conn = defaultdict(float)
                for nbr, ew in graph.adj[n].items():
                    if nbr in labels:
                        conn[labels[nbr]] += float(ew)
                feasible = [q for q in range(self.k) if loads[q] + w_n <= self.capacity]
                if not feasible:
                    raise RuntimeError('Initial partition cannot satisfy core capacity')
                best = max(feasible, key=lambda q: (conn.get(q, 0.0), -loads[q], -q))
                labels[n] = best
                loads[best] += w_n
                counts[best] += 1
            return labels

        def _refine_fast(self, graph: WeightedGraph, labels):
            labels = dict(labels)
            loads = [0] * self.k
            counts = [0] * self.k
            for n in graph.nodes:
                loads[labels[n]] += graph.node_weight[n]
                counts[labels[n]] += 1
            max_passes = max(1, self.patience)
            for _ in range(max_passes):
                improved = False
                order = sorted(graph.nodes, key=lambda n: sum(graph.adj[n].values()), reverse=True)
                for v in order:
                    cur = labels[v]
                    if counts[cur] <= 1:
                        continue
                    internal = 0.0
                    external = defaultdict(float)
                    for u, ew in graph.adj[v].items():
                        pu = labels[u]
                        if pu == cur:
                            internal += float(ew)
                        else:
                            external[pu] += float(ew)
                    if not external:
                        continue
                    candidates = [(ext - internal, ext, -loads[target], -target, target) for target, ext in external.items() if loads[target] + graph.node_weight[v] <= self.capacity]
                    if not candidates:
                        continue
                    gain, _ext, _negload, _negt, target = max(candidates)
                    if gain <= 1e-12:
                        continue
                    labels[v] = target
                    loads[cur] -= graph.node_weight[v]
                    loads[target] += graph.node_weight[v]
                    counts[cur] -= 1
                    counts[target] += 1
                    improved = True
                if not improved:
                    break
            return labels

        def _initial_partition(self, graph: WeightedGraph):
            if self.k <= 32:
                return self._initial_partition_reference(graph)
            return self._initial_partition_fast(graph)

        def _refine(self, graph: WeightedGraph, labels):
            if self.k <= 32:
                return self._refine_reference(graph, labels)
            return self._refine_fast(graph, labels)

        def partition(self, graph: WeightedGraph):
            if self.k * self.capacity < sum(graph.node_weight.values()):
                raise ValueError('Insufficient partition capacity')
            if self.k > len(graph.nodes):
                raise ValueError('Cannot make more non-empty partitions than nodes')
            hierarchy = []
            current = graph
            target_nodes = max(self.k, 2 * self.k)
            t0 = time.perf_counter()
            while len(current.nodes) > target_nodes:
                coarse, mapping = self._coarsen_once(current)
                if len(coarse.nodes) >= len(current.nodes) or len(coarse.nodes) < self.k:
                    break
                hierarchy.append(CoarsenStep(current, mapping))
                current = coarse
            self.stage_times['coarsen'] = time.perf_counter() - t0
            t0 = time.perf_counter()
            try:
                labels = self._initial_partition(current)
            except RuntimeError:
                hierarchy = []
                current = graph
                labels = self._initial_partition(current)
            self.stage_times['initial'] = time.perf_counter() - t0
            t0 = time.perf_counter()
            labels = self._refine(current, labels)
            for step in reversed(hierarchy):
                fine_labels = {n: labels[step.fine_to_coarse[n]] for n in step.fine_graph.nodes}
                labels = self._refine(step.fine_graph, fine_labels)
                current = step.fine_graph
            self.stage_times['refine'] = time.perf_counter() - t0
            return labels

    def allocate_controlled_parts(subnetworks, total_parts, capacity):
        sizes = [len(x) for x in subnetworks]
        mins = [max(1, math.ceil(n / capacity)) if n > 0 else 0 for n in sizes]
        maxs = list(sizes)
        if sum(mins) > total_parts or sum(maxs) < total_parts:
            return None
        parts = list(mins)
        while sum(parts) < total_parts:
            candidates = [i for i in range(len(parts)) if parts[i] < maxs[i]]
            if not candidates:
                return None
            i = max(candidates, key=lambda j: sizes[j] / max(parts[j], 1))
            parts[i] += 1
        return parts

    class NeuMapLocalPartitioner:

        def __init__(self, profile, capacity, total_controlled_parts=None, refinement_patience=10, seed=42):
            self.profile = profile
            self.capacity = int(capacity)
            self.total_controlled_parts = None if total_controlled_parts is None else int(total_controlled_parts)
            self.refinement_patience = int(refinement_patience)
            self.seed = int(seed)
            self.stats = {'candidate_count': 0, 'multilevel_total_time': 0.0, 'scheme_scoring_time': 0.0, 'coarsen_time': 0.0, 'initial_time': 0.0, 'refine_time': 0.0}

        def _candidate_subnetworks(self, size):
            layers = self.profile.layers
            return [[n for layer in layers[i:i + size] for n in layer] for i in range(0, len(layers), size) if any(layers[i:i + size])]

        def run(self):
            L = len(self.profile.layers)
            best = None
            candidate_records = []
            for size in range(1, L + 1):
                subnetworks = self._candidate_subnetworks(size)
                if self.total_controlled_parts is None:
                    parts = [max(1, math.ceil(len(nodes) / self.capacity)) for nodes in subnetworks]
                else:
                    parts = allocate_controlled_parts(subnetworks, self.total_controlled_parts, self.capacity)
                    if parts is None:
                        continue
                if any((p > len(nodes) for p, nodes in zip(parts, subnetworks))):
                    continue
                if sum(parts) * self.capacity < len(self.profile.graph.nodes):
                    continue
                combined = {}
                offset = 0
                candidate_ml_time = 0.0
                for si, (nodes, k_sub) in enumerate(zip(subnetworks, parts)):
                    sg = induced_subgraph(self.profile.graph, nodes)
                    ml = MultiLevelPartitioner(core_capacity=self.capacity, num_parts=k_sub, refinement_patience=self.refinement_patience, seed=self.seed + 1000 * size + si)
                    t0 = time.perf_counter()
                    local_labels = ml.partition(sg)
                    candidate_ml_time += time.perf_counter() - t0
                    self.stats['coarsen_time'] += ml.stage_times['coarsen']
                    self.stats['initial_time'] += ml.stage_times['initial']
                    self.stats['refine_time'] += ml.stage_times['refine']
                    for n, p in local_labels.items():
                        combined[n] = offset + int(p)
                    offset += int(k_sub)
                self.stats['multilevel_total_time'] += candidate_ml_time
                t0 = time.perf_counter()
                cut = graph_cut_weight(self.profile.graph, combined)
                self.stats['scheme_scoring_time'] += time.perf_counter() - t0
                k_total = offset
                self.stats['candidate_count'] += 1
                record = {'subnetwork_size': size, 'num_subnetworks': len(subnetworks), 'parts_per_subnetwork': parts, 'clusters': k_total, 'cut': float(cut), 'labels': combined}
                candidate_records.append(record)
                key = (float(cut), int(k_total), -int(size))
                if best is None or key < best[0]:
                    best = (key, record)
            if best is None:
                raise RuntimeError('No capacity-feasible NeuMap local-partition candidate')
            return (best[1], candidate_records)

    class NeuMapTabuMapper:

        def __init__(self, grid_dim, num_clusters, iterations=1000, tabu_tenure=10, max_neighbors=128, stagnation=250, max_evaluations=40000, seed=42):
            self.N = int(grid_dim)
            self.num_pes = self.N * self.N
            self.k = int(num_clusters)
            self.iterations = int(iterations)
            self.tabu_tenure = int(tabu_tenure)
            self.max_neighbors = int(max_neighbors)
            self.stagnation = int(stagnation)
            self.max_evaluations = int(max_evaluations)
            self.rng = random.Random(seed)
            self.evaluations = 0
            self.region_side = int(math.ceil(math.sqrt(self.k)))
            if self.region_side > self.N:
                raise ValueError(f'NeuMap search region {self.region_side}x{self.region_side} does not fit inside {self.N}x{self.N} physical mesh')
            self.candidate_cores = [r * self.N + c for r in range(self.region_side) for c in range(self.region_side)]
            if len(self.candidate_cores) < self.k:
                raise RuntimeError('Reduced search region cannot accommodate all clusters')
            coords = np.asarray([divmod(pe, self.N) for pe in range(self.num_pes)], dtype=np.int16)
            self._dist = (np.abs(coords[:, None, 0] - coords[None, :, 0]) + np.abs(coords[:, None, 1] - coords[None, :, 1])).astype(np.float64)
            link_id = {}
            lid = 0
            for r in range(self.N):
                for c in range(self.N):
                    u = r * self.N + c
                    if c + 1 < self.N:
                        v = r * self.N + c + 1
                        link_id[u, v] = lid
                        lid += 1
                        link_id[v, u] = lid
                        lid += 1
                    if r + 1 < self.N:
                        v = (r + 1) * self.N + c
                        link_id[u, v] = lid
                        lid += 1
                        link_id[v, u] = lid
                        lid += 1
            self._num_links = lid
            self._max_hops = max(1, 2 * (self.N - 1))
            self._route_ids = np.full((self.num_pes, self.num_pes, self._max_hops), -1, dtype=np.int32)
            for src in range(self.num_pes):
                for dst in range(self.num_pes):
                    if src == dst:
                        continue
                    ids = [link_id[x] for x in self._xy_links(src, dst)]
                    self._route_ids[src, dst, :len(ids)] = ids
            self._cached_C_id = None
            self._edge_src = self._edge_dst = self._edge_w = None
            self._total_weight = 0.0

        def _coord(self, pe):
            return divmod(int(pe), self.N)

        def _xy_links(self, src, dst):
            if src == dst:
                return []
            r, c = self._coord(src)
            rd, cd = self._coord(dst)
            links = []
            while c != cd:
                nc = c + (1 if cd > c else -1)
                u = r * self.N + c
                v = r * self.N + nc
                links.append((u, v))
                c = nc
            while r != rd:
                nr = r + (1 if rd > r else -1)
                u = r * self.N + c
                v = nr * self.N + c
                links.append((u, v))
                r = nr
            return links

        def evaluate(self, C, assignment):
            self.evaluations += 1
            Cn = np.asarray(C, dtype=np.float64)
            total_weight = float(Cn.sum())
            if total_weight <= 0.0:
                return (0.0, 0.0)
            a = np.asarray(assignment, dtype=np.int32)
            T = np.zeros((self.num_pes, self.num_pes), dtype=np.float64)
            T[np.ix_(a, a)] = Cn
            avg_hop = float(np.sum(T * self._dist) / total_weight)
            N = self.N
            T4 = T.reshape(N, N, N, N)
            A = T4.sum(axis=2)
            src_cum = A.cumsum(axis=1)
            dst_cum = A.cumsum(axis=2)
            both = src_cum.cumsum(axis=2)
            idx = np.arange(max(0, N - 1))
            if idx.size:
                left_src_all_dst = src_cum.sum(axis=2)[:, idx]
                all_src_dest_left = dst_cum.sum(axis=1)[:, idx]
                both_h = both[:, idx, idx]
                east = left_src_all_dst - both_h
                west = all_src_dest_left - both_h
                max_h = max(float(east.max(initial=0.0)), float(west.max(initial=0.0)))
            else:
                max_h = 0.0
            B = T4.sum(axis=1)
            src_r_cum = B.cumsum(axis=0)
            dst_r_cum = B.cumsum(axis=1)
            both_v_all = src_r_cum.cumsum(axis=1)
            if idx.size:
                top_src_all_dst = src_r_cum.sum(axis=1)[idx, :]
                all_src_dest_top = dst_r_cum.sum(axis=0)[idx, :]
                both_v = both_v_all[idx, idx, :]
                down = top_src_all_dst - both_v
                up = all_src_dest_top - both_v
                max_v = max(float(down.max(initial=0.0)), float(up.max(initial=0.0)))
            else:
                max_v = 0.0
            return (max(max_h, max_v), avg_hop)

        @staticmethod
        def _better(a, b):
            return tuple(a) < tuple(b)

        def _all_moves(self, assignment):
            moves = []
            for i in range(self.k):
                for j in range(i + 1, self.k):
                    moves.append(('swap', i, j))
            used = set(assignment)
            unused = [c for c in self.candidate_cores if c not in used]
            for i in range(self.k):
                for c in unused:
                    moves.append(('move', i, c))
            return moves

        def _apply_move(self, assignment, move):
            cand = list(assignment)
            if move[0] == 'swap':
                _, i, j = move
                cand[i], cand[j] = (cand[j], cand[i])
            else:
                _, i, c = move
                cand[i] = c
            return cand

        @staticmethod
        def _tabu_key(assignment, move):
            if move[0] == 'swap':
                _, i, j = move
                return ('swap', min(i, j), max(i, j))
            _, i, new_core = move
            old_core = assignment[i]
            return ('move', i, min(old_core, new_core), max(old_core, new_core))

        def map(self, C):
            current = self.rng.sample(self.candidate_cores, self.k)
            current_obj = self.evaluate(C, current)
            best = list(current)
            best_obj = current_obj
            tabu_until = {}
            no_improve = 0
            executed_iterations = 0
            for it in range(max(1, self.iterations)):
                if self.max_evaluations > 0 and self.evaluations >= self.max_evaluations:
                    break
                executed_iterations = it + 1
                used = set(current)
                unused = [c for c in self.candidate_cores if c not in used]
                total_swaps = self.k * (self.k - 1) // 2
                if self.max_neighbors > 0 and (not unused) and (total_swaps > self.max_neighbors * 2):
                    chosen = set()
                    while len(chosen) < min(self.max_neighbors, total_swaps):
                        i, j = self.rng.sample(range(self.k), 2)
                        if i > j:
                            i, j = (j, i)
                        chosen.add(('swap', i, j))
                    moves = list(chosen)
                else:
                    moves = self._all_moves(current)
                    if self.max_neighbors > 0 and len(moves) > self.max_neighbors:
                        moves = self.rng.sample(moves, self.max_neighbors)
                if self.max_evaluations > 0:
                    remaining_budget = self.max_evaluations - self.evaluations
                    if remaining_budget <= 0:
                        break
                    if len(moves) > remaining_budget:
                        moves = moves[:remaining_budget]
                best_candidate = None
                best_candidate_obj = None
                best_move = None
                best_key = None
                for move in moves:
                    key = self._tabu_key(current, move)
                    cand = self._apply_move(current, move)
                    obj = self.evaluate(C, cand)
                    is_tabu = tabu_until.get(key, -1) > it
                    aspiration = self._better(obj, best_obj)
                    if is_tabu and (not aspiration):
                        continue
                    if best_candidate is None or self._better(obj, best_candidate_obj):
                        best_candidate = cand
                        best_candidate_obj = obj
                        best_move = move
                        best_key = key
                if best_candidate is None:
                    break
                current = best_candidate
                current_obj = best_candidate_obj
                tabu_until[best_key] = it + max(1, self.tabu_tenure)
                if self._better(current_obj, best_obj):
                    best = list(current)
                    best_obj = current_obj
                    no_improve = 0
                else:
                    no_improve += 1
                    if self.stagnation > 0 and no_improve >= self.stagnation:
                        break
            return {'cluster_to_pe': {i: int(best[i]) for i in range(self.k)}, 'max_link_load_objective': float(best_obj[0]), 'avg_hop_objective': float(best_obj[1]), 'region_side': self.region_side, 'candidate_cores': list(self.candidate_cores), 'evaluations': int(self.evaluations), 'iterations_executed': int(executed_iterations)}


    def final_task_mapping(labels, cluster_to_pe):
        return {task: int(cluster_to_pe[int(cluster)]) for task, cluster in labels.items()}
    return SimpleNamespace(**locals())
PROVENANCE['jin'] = {'method': 'jin', 'status': 'adapted baseline'}

def _jin_namespace():

    def set_global_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)

    class WorkloadGenerator:

        def __init__(self, num_pes=16, num_agents=64):
            self.num_pes = int(num_pes)
            self.num_agents = int(num_agents)

        def generate_workload(self):
            workload = {'agents': []}
            num_shared_sources = max(self.num_pes, self.num_agents * 2, 20)
            shared_source_pool = [f'src_{i}' for i in range(num_shared_sources)]
            for agent_id in range(self.num_agents):
                dest = f'dest_a{agent_id}'
                num_sources = random.randint(1, 20)
                sources = random.sample(shared_source_pool, num_sources)
                workload['agents'].append({'agent_id': f'a_{agent_id}', 'destination': dest, 'sources': sources})
            return workload

    def natural_task_key(name):
        s = str(name)
        head = s.rstrip('0123456789')
        tail = s[len(head):]
        return (head, int(tail) if tail.isdigit() else -1, s)


    @dataclass
    class Profile:
        tasks: list
        sources: list
        destinations: list
        directed_edges: dict

    def build_profile(workload):
        sources, destinations = ([], [])
        seen_s, seen_d = (set(), set())
        directed = defaultdict(float)
        for agent in workload['agents']:
            d = str(agent['destination'])
            if d not in seen_d:
                seen_d.add(d)
                destinations.append(d)
            for s in agent['sources']:
                s = str(s)
                if s not in seen_s:
                    seen_s.add(s)
                    sources.append(s)
                if s != d:
                    directed[s, d] += 1.0
        sources = sorted(sources, key=natural_task_key)
        destinations = sorted(destinations, key=natural_task_key)
        tasks = sources + [d for d in destinations if d not in seen_s]
        return Profile(tasks, sources, destinations, dict(directed))

    def balanced_sizes(n_items: int, k: int):
        if k <= 0:
            raise ValueError('number of clusters must be positive')
        if n_items < k:
            k = n_items
        q, r = divmod(n_items, k)
        return [q + (1 if i < r else 0) for i in range(k)]

    def controlled_partition(profile: Profile, num_clusters: int, capacity: int):
        order = list(profile.sources) + list(profile.destinations)
        if len(order) != len(profile.tasks):
            order = list(profile.tasks)
        sizes = balanced_sizes(len(order), int(num_clusters))
        if sizes and max(sizes) > int(capacity):
            raise ValueError(f'controlled {num_clusters}-cluster partition needs at least {max(sizes)} tasks/cluster, but capacity={capacity}')
        labels = {}
        p = 0
        for cid, sz in enumerate(sizes):
            for t in order[p:p + sz]:
                labels[t] = cid
            p += sz
        return (labels, len(sizes), 'controlled-balanced-sequential-PCN')

    def paper_family_min_partition(profile: Profile, capacity: int, max_pes: int):
        if capacity <= 0:
            raise ValueError('capacity must be positive')
        order = list(profile.tasks)
        labels = {}
        cid = 0
        used = 0
        for t in order:
            if used >= capacity:
                cid += 1
                used = 0
            if cid >= max_pes:
                raise ValueError(f'workload requires more than {max_pes} clusters at capacity={capacity}')
            labels[t] = cid
            used += 1
        k = 0 if not order else cid + 1
        return (labels, k, 'paper-family-sequential-fill-to-capacity')

    def build_pcn(profile: Profile, task_to_cluster: dict, num_real_clusters: int):
        weights = defaultdict(float)
        indeg = [0] * num_real_clusters
        out = [set() for _ in range(num_real_clusters)]
        for (u, v), w in profile.directed_edges.items():
            cu = int(task_to_cluster[u])
            cv = int(task_to_cluster[v])
            if cu == cv:
                continue
            weights[cu, cv] += float(w)
        for (u, v), w in weights.items():
            if w > 0 and v not in out[u]:
                out[u].add(v)
                indeg[v] += 1
        return (dict(weights), indeg, out)

    def topological_order(num_clusters: int, indeg, out):
        indeg = list(indeg)
        heap = [i for i in range(num_clusters) if indeg[i] == 0]
        heapq.heapify(heap)
        order = []
        while heap:
            u = heapq.heappop(heap)
            order.append(u)
            for v in sorted(out[u]):
                indeg[v] -= 1
                if indeg[v] == 0:
                    heapq.heappush(heap, v)
        if len(order) != num_clusters:
            remain = [i for i in range(num_clusters) if i not in set(order)]
            order.extend(sorted(remain))
            return (order, False)
        return (order, True)

    def cell_neighbors(cell, cells_set):
        r, c = cell
        for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if q in cells_set:
                yield q

    def nearest_cell(cells, hint):
        hr, hc = hint
        return min(cells, key=lambda x: (abs(x[0] - hr) + abs(x[1] - hc), x[0], x[1]))

    def boundary_cross_pairs(A, B):
        Bset = set(B)
        pairs = []
        for a in A:
            ar, ac = a
            for b in ((ar - 1, ac), (ar + 1, ac), (ar, ac - 1), (ar, ac + 1)):
                if b in Bset:
                    pairs.append((a, b))
        return pairs

    def graph_distances(cells, source):
        S = set(cells)
        src = nearest_cell(cells, source)
        d = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in cell_neighbors(u, S):
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        return d

    def _best_straight_split(cells, start_cell, end_cell):
        cells = list(cells)
        rows = sorted({r for r, _ in cells})
        cols = sorted({c for _, c in cells})
        candidates = []
        for k in rows[:-1]:
            A = [x for x in cells if x[0] <= k]
            B = [x for x in cells if x[0] > k]
            if not A or not B:
                continue
            if (start_cell in A) == (end_cell in A):
                continue
            cross = boundary_cross_pairs(A, B)
            if not cross:
                continue
            score = (len(cross), abs(len(A) - len(B)), 0, k)
            candidates.append((score, A, B))
        for k in cols[:-1]:
            A = [x for x in cells if x[1] <= k]
            B = [x for x in cells if x[1] > k]
            if not A or not B:
                continue
            if (start_cell in A) == (end_cell in A):
                continue
            cross = boundary_cross_pairs(A, B)
            if not cross:
                continue
            score = (len(cross), abs(len(A) - len(B)), 1, k)
            candidates.append((score, A, B))
        if not candidates:
            return None
        candidates.sort(key=lambda z: z[0])
        _score, A, B = candidates[0]
        if start_cell not in A:
            A, B = (B, A)
        return (A, B)

    def _bfs_split(cells, start_cell, end_cell):
        ds = graph_distances(cells, start_cell)
        de = graph_distances(cells, end_cell)
        A, B, ties = ([], [], [])
        inf = 10 ** 9
        for x in cells:
            a = ds.get(x, inf)
            b = de.get(x, inf)
            if a < b:
                A.append(x)
            elif b < a:
                B.append(x)
            else:
                ties.append(x)
        cr = sum((r for r, _ in cells)) / len(cells)
        cc = sum((c for _, c in cells)) / len(cells)
        ties.sort(key=lambda x: (abs(x[0] - cr) + abs(x[1] - cc), x[0], x[1]))
        for x in ties:
            if len(A) <= len(B):
                A.append(x)
            else:
                B.append(x)
        if start_cell not in A:
            if start_cell in B:
                B.remove(start_cell)
            A.append(start_cell)
        if end_cell not in B:
            if end_cell in A:
                A.remove(end_cell)
            B.append(end_cell)
        if not A or not B:
            ordered = sorted(cells, key=lambda x: (abs(x[0] - start_cell[0]) + abs(x[1] - start_cell[1]), x[0], x[1]))
            m = max(1, min(len(ordered) - 1, len(ordered) // 2))
            A, B = (ordered[:m], ordered[m:])
            if start_cell not in A:
                if start_cell in B:
                    B.remove(start_cell)
                A[0] = start_cell
            if end_cell not in B:
                if end_cell in A:
                    A.remove(end_cell)
                B.append(end_cell)
        return (A, B)

    def _choose_bridge(A, B):
        pairs = boundary_cross_pairs(A, B)
        if not pairs:
            return min(((a, b) for a in A for b in B), key=lambda p: (abs(p[0][0] - p[1][0]) + abs(p[0][1] - p[1][1]), p[0], p[1]))
        cr = sum((r for r, _ in A + B)) / (len(A) + len(B))
        cc = sum((c for _, c in A + B)) / (len(A) + len(B))
        return min(pairs, key=lambda p: (abs((p[0][0] + p[1][0]) / 2 - cr) + abs((p[0][1] + p[1][1]) / 2 - cc), p[0], p[1]))

    def _hilbert_d2xy(n, d):
        x = y = 0
        t = int(d)
        s = 1
        while s < n:
            rx = 1 & t // 2
            ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = (y, x)
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        return (x, y)

    def construct_alp_order(cells, start_hint, end_hint):
        cells = list(dict.fromkeys(cells))
        if not cells:
            return []
        rows = sorted({r for r, _ in cells})
        cols = sorted({c for _, c in cells})
        full_square = len(rows) == len(cols) and len(cells) == len(rows) * len(cols) and (rows == list(range(min(rows), min(rows) + len(rows)))) and (cols == list(range(min(cols), min(cols) + len(cols))))
        n = len(rows) if full_square else 0
        power2 = n > 0 and n & n - 1 == 0
        if full_square and power2 and (start_hint == (min(rows), min(cols))) and (end_hint == (min(rows), max(cols))):
            r0, c0 = (min(rows), min(cols))
            return [(r0 + y, c0 + x) for d in range(n * n) for x, y in [_hilbert_d2xy(n, d)]]

        def rec(subcells, sh, eh):
            subcells = list(subcells)
            if len(subcells) <= 1:
                return subcells
            s = nearest_cell(subcells, sh)
            e = nearest_cell(subcells, eh)
            if s == e:
                e = max(subcells, key=lambda x: (abs(x[0] - s[0]) + abs(x[1] - s[1]), x[0], x[1]))
            split = _best_straight_split(subcells, s, e)
            if split is None:
                A, B = _bfs_split(subcells, s, e)
            else:
                A, B = split
            if not A or not B or len(A) == len(subcells) or (len(B) == len(subcells)):
                ordered = sorted(subcells)
                m = len(ordered) // 2
                A, B = (ordered[:m], ordered[m:])
                if s not in A:
                    A, B = (B, A)
            a_bridge, b_bridge = _choose_bridge(A, B)
            left = rec(A, s, a_bridge)
            right = rec(B, b_bridge, e)
            return left + right
        order = rec(cells, start_hint, end_hint)
        if len(order) != len(cells) or len(set(order)) != len(cells):
            raise RuntimeError('ALP reproduction failed to generate a bijection')
        return order

    def corner_hint(name, N):
        d = {'tl': (0, 0), 'tr': (0, N - 1), 'bl': (N - 1, 0), 'br': (N - 1, N - 1)}
        if name not in d:
            raise ValueError(f'unknown corner {name}')
        return d[name]

    def coord(pe, N):
        return divmod(int(pe), int(N))

    def manhattan_pe(a, b, N):
        ar, ac = coord(a, N)
        br, bc = coord(b, N)
        return abs(ar - br) + abs(ac - bc)

    def placement_energy(cluster_to_pe, weights, N, Er=1.0, Ew=0.1):
        total = 0.0
        for (u, v), w in weights.items():
            h = manhattan_pe(cluster_to_pe[u], cluster_to_pe[v], N)
            total += float(w) * ((h + 1) * Er + h * Ew)
        return float(total)

    def placement_latency_proxy(cluster_to_pe, weights, N, Lr=1.0, Lw=0.01):
        if not weights:
            return 0.0
        return float(max(((manhattan_pe(cluster_to_pe[u], cluster_to_pe[v], N) + 1) * Lr + manhattan_pe(cluster_to_pe[u], cluster_to_pe[v], N) * Lw for (u, v), w in weights.items() if w > 0)))

    def weighted_avg_hop(cluster_to_pe, weights, N):
        den = sum((float(w) for w in weights.values()))
        if den <= 0:
            return 0.0
        num = sum((float(w) * manhattan_pe(cluster_to_pe[u], cluster_to_pe[v], N) for (u, v), w in weights.items()))
        return float(num / den)

    def total_spike_travel_distance(cluster_to_pe, weights, N):
        return float(sum((float(w) * manhattan_pe(cluster_to_pe[u], cluster_to_pe[v], N) for (u, v), w in weights.items())))

    def adjacent_pe_pairs(N):
        pairs = []
        for r in range(N):
            for c in range(N):
                p = r * N + c
                if c + 1 < N:
                    pairs.append((p, p + 1))
                if r + 1 < N:
                    pairs.append((p, p + N))
        return pairs

    def fd_refine(initial_cluster_to_pe, weights, N, num_pes, lam=0.3, Er=1.0, Ew=0.1, max_iterations=1000):
        cluster_to_pe = dict(initial_cluster_to_pe)
        pe_to_cluster = {pe: None for pe in range(num_pes)}
        for c, pe in cluster_to_pe.items():
            pe_to_cluster[int(pe)] = int(c)

        def current_energy():
            return placement_energy(cluster_to_pe, weights, N, Er, Ew)

        def swap_delta(p, q):
            cp = pe_to_cluster[p]
            cq = pe_to_cluster[q]
            if cp is None and cq is None:
                return 0.0
            before = current_energy()
            pe_to_cluster[p], pe_to_cluster[q] = (cq, cp)
            if cp is not None:
                cluster_to_pe[cp] = q
            if cq is not None:
                cluster_to_pe[cq] = p
            after = current_energy()
            pe_to_cluster[p], pe_to_cluster[q] = (cp, cq)
            if cp is not None:
                cluster_to_pe[cp] = p
            if cq is not None:
                cluster_to_pe[cq] = q
            return float(before - after)
        pairs = adjacent_pe_pairs(N)
        iterations = 0
        accepted = 0
        tensions_evaluated = 0
        energy_trace = [current_energy()]
        while iterations < max_iterations:
            queue = []
            for p, q in pairs:
                t = swap_delta(p, q)
                tensions_evaluated += 1
                if t > 1e-12:
                    queue.append((t, p, q))
            if not queue:
                break
            queue.sort(key=lambda z: (-z[0], z[1], z[2]))
            take = max(1, int(math.ceil(float(lam) * len(queue))))
            did = 0
            for _old_t, p, q in queue[:take]:
                t = swap_delta(p, q)
                tensions_evaluated += 1
                if t <= 1e-12:
                    continue
                cp = pe_to_cluster[p]
                cq = pe_to_cluster[q]
                pe_to_cluster[p], pe_to_cluster[q] = (cq, cp)
                if cp is not None:
                    cluster_to_pe[cp] = q
                if cq is not None:
                    cluster_to_pe[cq] = p
                accepted += 1
                did += 1
            iterations += 1
            energy_trace.append(current_energy())
            if did == 0:
                break
        return (cluster_to_pe, {'iterations': int(iterations), 'accepted_swaps': int(accepted), 'tension_evaluations': int(tensions_evaluated), 'energy_initial': float(energy_trace[0]), 'energy_final': float(energy_trace[-1]), 'converged': bool(iterations < max_iterations)})

    def xy_path(src, dst, N):
        sr, sc = coord(src, N)
        dr, dc = coord(dst, N)
        path = [src]
        r, c = (sr, sc)
        while c != dc:
            c += 1 if dc > c else -1
            path.append(r * N + c)
        while r != dr:
            r += 1 if dr > r else -1
            path.append(r * N + c)
        return path


    def task_mapping(task_to_cluster, cluster_to_pe):
        return {t: int(cluster_to_pe[int(c)]) for t, c in task_to_cluster.items()}
    return SimpleNamespace(**locals())
PROVENANCE['xiao'] = {'method': 'xiao', 'status': 'adapted baseline'}

def _xiao_namespace():

    def set_global_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)

    class WorkloadGenerator:

        def __init__(self, num_pes=16, num_agents=64):
            self.num_pes = int(num_pes)
            self.num_agents = int(num_agents)

        def generate_workload(self):
            workload = {'agents': []}
            num_shared_sources = max(self.num_pes, self.num_agents * 2, 20)
            shared_source_pool = [f'src_{i}' for i in range(num_shared_sources)]
            for agent_id in range(self.num_agents):
                dest = f'dest_a{agent_id}'
                num_sources = random.randint(1, 20)
                sources = random.sample(shared_source_pool, num_sources)
                workload['agents'].append({'agent_id': f'a_{agent_id}', 'destination': dest, 'sources': sources})
            return workload

    def natural_task_key(name):
        s = str(name)
        head = s.rstrip('0123456789')
        tail = s[len(head):]
        return (head, int(tail) if tail.isdigit() else -1, s)


    @dataclass
    class Profile:
        tasks: list
        sources: list
        destinations: list
        directed_edges: dict
        presynaptic: dict
        postsynaptic: dict
        layer: dict

    def build_profile(workload):
        sources = []
        destinations = []
        seen_s = set()
        seen_d = set()
        directed = defaultdict(float)
        presyn = defaultdict(set)
        postsyn = defaultdict(set)
        for agent in workload['agents']:
            d = agent['destination']
            if d not in seen_d:
                seen_d.add(d)
                destinations.append(d)
            for s in agent['sources']:
                if s not in seen_s:
                    seen_s.add(s)
                    sources.append(s)
                if s == d:
                    continue
                directed[s, d] += 1.0
                presyn[d].add(s)
                postsyn[s].add(d)
        sources = sorted(sources, key=natural_task_key)
        destinations = sorted(destinations, key=natural_task_key)
        tasks = sources + [d for d in destinations if d not in seen_s]
        layer = {t: 0 for t in sources}
        layer.update({d: 1 for d in destinations})
        if seen_s & seen_d:
            layer = {t: 0 for t in tasks}
        for t in tasks:
            presyn.setdefault(t, set())
            postsyn.setdefault(t, set())
        return Profile(tasks=tasks, sources=sources, destinations=destinations, directed_edges=dict(directed), presynaptic={k: set(v) for k, v in presyn.items()}, postsynaptic={k: set(v) for k, v in postsyn.items()}, layer=layer)

    def linkage_topological_order(profile: Profile, tasks_subset=None):
        allowed = set(profile.tasks if tasks_subset is None else tasks_subset)
        sources = [s for s in profile.sources if s in allowed]
        dests = [d for d in profile.destinations if d in allowed]
        remaining_pre = {d: set((x for x in profile.presynaptic[d] if x in allowed)) for d in dests}
        waiting_by_source = defaultdict(list)
        for d, ps in remaining_pre.items():
            for s in ps:
                waiting_by_source[s].append(d)
        order = []
        visited = set()
        emitted_dest = set()
        ready0 = sorted([d for d in dests if not remaining_pre[d]], key=natural_task_key)
        for d in ready0:
            order.append(d)
            visited.add(d)
            emitted_dest.add(d)
        for s in sorted(sources, key=natural_task_key):
            if s in visited:
                continue
            order.append(s)
            visited.add(s)
            newly_ready = []
            for d in waiting_by_source.get(s, []):
                remaining_pre[d].discard(s)
                if not remaining_pre[d] and d not in emitted_dest and (d not in visited):
                    newly_ready.append(d)
            newly_ready.sort(key=lambda d: (-len(profile.presynaptic[d]), natural_task_key(d)))
            for d in newly_ready:
                order.append(d)
                visited.add(d)
                emitted_dest.add(d)
        leftovers = sorted(allowed - visited, key=natural_task_key)
        order.extend(leftovers)
        return order

    def split_if_needed(profile: Profile, num_pes: int, max_tasks_per_pe: int):
        hw_capacity = int(num_pes) * int(max_tasks_per_pe)
        if len(profile.tasks) <= hw_capacity:
            return ([list(profile.tasks)], 'no-split-workload-fits')
        order = linkage_topological_order(profile)
        subnets = [order[i:i + hw_capacity] for i in range(0, len(order), hw_capacity)]
        return (subnets, 'linkage-topological-acyclic-proxy')

    def balanced_sizes(n, k, max_cap):
        if k <= 0:
            raise ValueError('k must be positive')
        if n > k * max_cap:
            raise ValueError(f'cannot place {n} tasks into {k} clusters of cap {max_cap}')
        base, rem = divmod(n, k)
        sizes = [base + (1 if i < rem else 0) for i in range(k)]
        if max(sizes, default=0) > max_cap:
            raise ValueError('balanced cluster size exceeds hardware capacity')
        return sizes

    def allocate_layer_cluster_counts(n_src, n_dst, k, max_cap):
        if n_src == 0:
            return (0, k)
        if n_dst == 0:
            return (k, 0)
        min_src = math.ceil(n_src / max_cap)
        min_dst = math.ceil(n_dst / max_cap)
        if min_src + min_dst > k:
            return None
        extra = k - min_src - min_dst
        src_extra = round(extra * n_src / max(1, n_src + n_dst))
        src_extra = max(0, min(extra, src_extra))
        return (min_src + src_extra, min_dst + (extra - src_extra))

    def das_of_tasks(tasks, profile: Profile):
        if not tasks:
            return 0.0
        synapses = sum((len(profile.presynaptic[t]) for t in tasks))
        unique_axons = set()
        for t in tasks:
            unique_axons.update(profile.presynaptic[t])
        if not unique_axons:
            return 0.0
        return float(synapses) / float(len(unique_axons))

    def greedy_das_groups(tasks, k, max_cap, profile: Profile):
        tasks = list(tasks)
        sizes = balanced_sizes(len(tasks), k, max_cap)
        unassigned = set(tasks)
        clusters = []
        for target_size in sizes:
            if target_size == 0:
                clusters.append([])
                continue
            seed = min(unassigned, key=lambda t: (-len(profile.presynaptic[t]), natural_task_key(t)))
            group = [seed]
            unassigned.remove(seed)
            while len(group) < target_size:
                old_das = das_of_tasks(group, profile)

                def cand_key(t):
                    new_das = das_of_tasks(group + [t], profile)
                    overlap = len(set().union(*(profile.presynaptic[g] for g in group)) & profile.presynaptic[t])
                    return (-(new_das - old_das), -overlap, natural_task_key(t))
                nxt = min(unassigned, key=cand_key)
                group.append(nxt)
                unassigned.remove(nxt)
            clusters.append(group)
        return clusters

    def channel_partition_proxy(profile: Profile, k: int, max_cap: int):
        if set(profile.sources) & set(profile.destinations):
            groups = greedy_das_groups(profile.tasks, k, max_cap, profile)
            return ({t: ci for ci, group in enumerate(groups) for t in group}, 'channel-single-recurrent-population-DAS-proxy')
        alloc = allocate_layer_cluster_counts(len(profile.sources), len(profile.destinations), k, max_cap)
        if alloc is None:
            return (link_partition_proxy(profile, k, max_cap), 'channel-fallback-to-link-capacity')
        ks, kd = alloc
        clusters = []
        if ks:
            src_sizes = balanced_sizes(len(profile.sources), ks, max_cap)
            src_order = sorted(profile.sources, key=natural_task_key)
            pos = 0
            for sz in src_sizes:
                clusters.append(src_order[pos:pos + sz])
                pos += sz
        if kd:
            clusters.extend(greedy_das_groups(profile.destinations, kd, max_cap, profile))
        labels = {}
        for ci, group in enumerate(clusters):
            for t in group:
                labels[t] = ci
        return (labels, 'layer-preserving-DAS-sharing-proxy')

    def link_partition_proxy(profile: Profile, k: int, max_cap: int):
        order = linkage_topological_order(profile)
        sizes = balanced_sizes(len(order), k, max_cap)
        labels = {}
        pos = 0
        for ci, sz in enumerate(sizes):
            for t in order[pos:pos + sz]:
                labels[t] = ci
            pos += sz
        return labels

    def partition_profile(profile: Profile, k: int, max_cap: int, strategy: str):
        if strategy == 'channel':
            labels, note = channel_partition_proxy(profile, k, max_cap)
        elif strategy == 'link':
            labels = link_partition_proxy(profile, k, max_cap)
            note = 'linkage-recurrent-order-proxy' if set(profile.sources) & set(profile.destinations) else 'linkage-topological-order-proxy'
        else:
            raise ValueError(strategy)
        counts = [0] * k
        for c in labels.values():
            counts[int(c)] += 1
        if max(counts, default=0) > max_cap:
            raise RuntimeError('partition exceeded PE task capacity')
        if len(labels) != len(profile.tasks):
            missing = set(profile.tasks) - set(labels)
            raise RuntimeError(f'partition omitted tasks: {sorted(missing)}')
        return (labels, note, counts)

    def partition_das(profile: Profile, labels, k):
        vals = []
        for c in range(k):
            tasks = [t for t in profile.tasks if labels[t] == c]
            vals.append(das_of_tasks(tasks, profile))
        return (float(np.mean(vals)) if vals else 0.0, float(max(vals) if vals else 0.0))

    def cluster_communication(profile: Profile, labels, k):
        C = np.zeros((k, k), dtype=np.float64)
        for (u, v), w in profile.directed_edges.items():
            a = int(labels[u])
            b = int(labels[v])
            if a != b:
                C[a, b] += float(w)
        return C

    def coord(pe, N):
        return divmod(int(pe), int(N))

    def xy_path(src, dst, N):
        if src == dst:
            return [int(src)]
        r, c = coord(src, N)
        rd, cd = coord(dst, N)
        path = [int(src)]
        while c != cd:
            c += 1 if cd > c else -1
            path.append(r * N + c)
        while r != rd:
            r += 1 if rd > r else -1
            path.append(r * N + c)
        return path

    class MappingObjective:

        def __init__(self, C, grid_dim):
            self.C = np.asarray(C, dtype=np.float64)
            self.k = self.C.shape[0]
            self.N = int(grid_dim)
            self.num_pes = self.N * self.N
            self.evaluations = 0

        def evaluate(self, full_permutation):
            self.evaluations += 1
            assigned = list(full_permutation[:self.k])
            total_w = float(self.C.sum())
            hop_sum = 0.0
            router_load = np.zeros(self.num_pes, dtype=np.float64)
            for i in range(self.k):
                pi = int(assigned[i])
                ri, ci = coord(pi, self.N)
                for j in range(self.k):
                    w = float(self.C[i, j])
                    if w <= 0:
                        continue
                    pj = int(assigned[j])
                    rj, cj = coord(pj, self.N)
                    hop_sum += w * (abs(ri - rj) + abs(ci - cj))
                    for router in xy_path(pi, pj, self.N):
                        router_load[router] += w
            ave_hop = hop_sum / total_w if total_w > 0 else 0.0
            max_cong = float(router_load.max()) if router_load.size else 0.0
            return (float(ave_hop), float(max_cong))

    @dataclass
    class Individual:
        code: tuple
        obj: tuple = None
        rank: int = 10 ** 9
        crowding: float = 0.0

    def dominates(a, b):
        return all((x <= y for x, y in zip(a, b))) and any((x < y for x, y in zip(a, b)))

    def fast_nondominated_sort(pop):
        n = len(pop)
        dominated_sets = [[] for _ in range(n)]
        domination_count = [0] * n
        fronts = [[]]
        for p in range(n):
            for q in range(p + 1, n):
                if dominates(pop[p].obj, pop[q].obj):
                    dominated_sets[p].append(q)
                    domination_count[q] += 1
                elif dominates(pop[q].obj, pop[p].obj):
                    dominated_sets[q].append(p)
                    domination_count[p] += 1
        for i in range(n):
            if domination_count[i] == 0:
                pop[i].rank = 0
                fronts[0].append(i)
        f = 0
        while f < len(fronts) and fronts[f]:
            nxt = []
            for p in fronts[f]:
                for q in dominated_sets[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        pop[q].rank = f + 1
                        nxt.append(q)
            if nxt:
                fronts.append(nxt)
            f += 1
        return fronts

    def assign_crowding(pop, front):
        if not front:
            return
        for idx in front:
            pop[idx].crowding = 0.0
        if len(front) <= 2:
            for idx in front:
                pop[idx].crowding = float('inf')
            return
        mcount = len(pop[front[0]].obj)
        for m in range(mcount):
            ordered = sorted(front, key=lambda i: pop[i].obj[m])
            pop[ordered[0]].crowding = float('inf')
            pop[ordered[-1]].crowding = float('inf')
            fmin = pop[ordered[0]].obj[m]
            fmax = pop[ordered[-1]].obj[m]
            if fmax <= fmin:
                continue
            for pos in range(1, len(ordered) - 1):
                idx = ordered[pos]
                if math.isinf(pop[idx].crowding):
                    continue
                prevv = pop[ordered[pos - 1]].obj[m]
                nextv = pop[ordered[pos + 1]].obj[m]
                pop[idx].crowding += (nextv - prevv) / (fmax - fmin)

    def rank_and_crowding(pop):
        fronts = fast_nondominated_sort(pop)
        for f in fronts:
            assign_crowding(pop, f)
        return fronts

    def tournament(pop):
        if len(pop) == 1:
            return pop[0]
        a, b = random.sample(pop, 2)
        if a.rank != b.rank:
            return a if a.rank < b.rank else b
        if a.crowding != b.crowding:
            return a if a.crowding > b.crowding else b
        return a if random.random() < 0.5 else b

    def ordered_crossover(p1, p2, rate):
        n = len(p1)
        if n < 2 or random.random() >= rate:
            return (tuple(p1), tuple(p2))
        a, b = sorted(random.sample(range(n), 2))

        def ox(x, y):
            child = [None] * n
            child[a:b + 1] = x[a:b + 1]
            used = set(child[a:b + 1])
            fill = [v for v in y if v not in used]
            positions = list(range(b + 1, n)) + list(range(0, a))
            for pos, val in zip(positions, fill):
                child[pos] = val
            return tuple(child)
        return (ox(p1, p2), ox(p2, p1))

    def swap_mutation(code, rate):
        out = list(code)
        if len(out) >= 2 and random.random() < rate:
            i, j = random.sample(range(len(out)), 2)
            out[i], out[j] = (out[j], out[i])
        return tuple(out)

    class PaperNSGA2:

        def __init__(self, objective, pop_size=100, generations=100, crossover_rate=0.9, mutation_rate=0.1):
            self.objective = objective
            self.pop_size = int(pop_size)
            self.generations = int(generations)
            self.crossover_rate = float(crossover_rate)
            self.mutation_rate = float(mutation_rate)
            self.eval_count = 0

        def evaluate(self, ind):
            if ind.obj is None:
                ind.obj = self.objective.evaluate(ind.code)
                self.eval_count += 1
            return ind

        def init_population(self):
            n = self.objective.num_pes
            base = list(range(n))
            pop = []
            for _ in range(self.pop_size):
                code = tuple(random.sample(base, n))
                pop.append(self.evaluate(Individual(code)))
            return pop

        def make_offspring(self, pop):
            rank_and_crowding(pop)
            Q = []
            while len(Q) < self.pop_size:
                p1 = tournament(pop)
                p2 = tournament(pop)
                c1, c2 = ordered_crossover(p1.code, p2.code, self.crossover_rate)
                c1 = swap_mutation(c1, self.mutation_rate)
                c2 = swap_mutation(c2, self.mutation_rate)
                Q.append(self.evaluate(Individual(c1)))
                if len(Q) < self.pop_size:
                    Q.append(self.evaluate(Individual(c2)))
            return Q

        def environmental_select(self, combined):
            fronts = rank_and_crowding(combined)
            new_pop = []
            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend((combined[i] for i in front))
                else:
                    remaining = self.pop_size - len(new_pop)
                    ordered = sorted((combined[i] for i in front), key=lambda ind: ind.crowding, reverse=True)
                    new_pop.extend(ordered[:remaining])
                    break
            return new_pop

        def run(self):
            P = self.init_population()
            rank_and_crowding(P)
            Q = self.make_offspring(P)
            for _ in range(max(0, self.generations)):
                R = P + Q
                P = self.environmental_select(R)
                Q = self.make_offspring(P)
            fronts = rank_and_crowding(P)
            pareto = [P[i] for i in fronts[0]] if fronts and fronts[0] else []
            return (P, pareto)

    def select_paper_solution(pareto):
        if not pareto:
            raise RuntimeError('NSGA-II returned no Pareto solutions')
        max_ave = max((float(ind.obj[0]) for ind in pareto))
        max_cong = max((float(ind.obj[1]) for ind in pareto))
        max_ave = max(max_ave, 1e-12)
        max_cong = max(max_cong, 1e-12)

        def score(ind):
            ave, cong = ind.obj
            return float(cong) / max_cong + float(ave) / max_ave
        best = min(pareto, key=lambda ind: (score(ind), ind.obj[1], ind.obj[0], ind.code))
        return (best, score(best))


    def labels_to_task_mapping(labels, cluster_to_pe):
        return {t: int(cluster_to_pe[int(c)]) for t, c in labels.items()}

    def _minimum_capacity_for_fixed_grid(profile: Profile, num_pes: int, strict_layer_separation: bool=False):
        if num_pes <= 0:
            raise ValueError('num_pes must be positive')
        if not strict_layer_separation or set(profile.sources) & set(profile.destinations):
            return int(math.ceil(len(profile.tasks) / num_pes))
        sizes = [len(profile.sources), len(profile.destinations)]
        lo, hi = (1, max(sizes) if sizes else 1)
        while lo < hi:
            mid = (lo + hi) // 2
            need = sum((math.ceil(n / mid) for n in sizes if n))
            if need <= num_pes:
                hi = mid
            else:
                lo = mid + 1
        return lo
    return SimpleNamespace(**locals())

def mapped_tasks(method, workload, grid, capacity, seed, quick=False):
    ns = globals()['_' + method + '_namespace']()
    random.seed(seed)
    np.random.seed(seed)
    n = len(workload['agents'])
    pes = grid * grid
    k = min(n, pes)
    budget = 20 if quick else 1000
    if n == 1 or grid == 1 or (not any((x['sources'] for x in workload['agents']))):
        return ({x['destination']: i % pes for i, x in enumerate(workload['agents'])}, {'degenerate_balanced_placement': True, 'mapping_seconds': 0.0})
    notes = {}
    start = time.perf_counter()
    if method == 'sneap':
        graph = ns.build_task_graph(workload)
        part = ns.SneapMultilevelPartitioner(core_capacity=capacity, num_parts=k, refinement_patience=2 if quick else 10, seed=seed)
        labels = part.partition(graph)
        C = ns.partition_communication_matrix(graph, labels, k)
        placement, _ = ns.SneapSAMapper(grid, iterations=budget, seed=seed + 1).map(C)
        mapping = ns.final_task_mapping(labels, placement)
    elif method == 'spinemap':
        graph = ns.build_task_graph(workload)
        labels = ns.SpiNeCluster(num_clusters=k, capacity=capacity, seed=seed, pair_passes=2 if quick else 10).partition(graph)
        placer = ns.SpiNePlacerPSO(workload, graph, labels, grid, particles=4 if quick else 20, iterations=3 if quick else 40, seed=seed + 1)
        best, _ = placer.run()
        mapping = ns.final_task_mapping(labels, best)
        notes['fitness'] = 'average-hop placement objective'
    elif method == 'neumap':
        profile = ns.build_neumap_profile(workload)
        part = ns.NeuMapLocalPartitioner(profile, capacity, total_controlled_parts=k, refinement_patience=2 if quick else 10, seed=seed)
        chosen, _ = part.run()
        labels = chosen['labels']
        k = int(chosen['clusters'])
        C = ns.directed_cluster_matrix(profile, labels, k)
        placer = ns.NeuMapTabuMapper(grid, k, iterations=budget, max_neighbors=16 if quick else 128, stagnation=10 if quick else 250, max_evaluations=200 if quick else 40000, seed=seed + 1)
        result = placer.map(C)
        mapping = ns.final_task_mapping(labels, result['cluster_to_pe'])
    elif method == 'jin':
        profile = ns.build_profile(workload)
        labels, k, note = ns.controlled_partition(profile, k, capacity)
        weights, indeg, out = ns.build_pcn(profile, labels, k)
        order, acyclic = ns.topological_order(k, indeg, out)
        cells = [(r, c) for r in range(grid) for c in range(grid)]
        curve = ns.construct_alp_order(cells, ns.corner_hint('tl', grid), ns.corner_hint('br', grid))
        placement = {int(cid): int(curve[i][0] * grid + curve[i][1]) for i, cid in enumerate(order)}
        placement, _ = ns.fd_refine(placement, weights, grid, pes, max_iterations=budget)
        mapping = ns.task_mapping(labels, placement)
        notes.update(partition=note, pcn_acyclic=bool(acyclic))
    else:
        profile = ns.build_profile(workload)
        labels, note, _ = ns.partition_profile(profile, k, capacity, 'link')
        C = ns.cluster_communication(profile, labels, k)
        objective = ns.MappingObjective(C, grid)
        optimizer = ns.PaperNSGA2(objective, pop_size=8 if quick else 100, generations=3 if quick else 100)
        _, pareto = optimizer.run()
        best, _ = ns.select_paper_solution(pareto)
        placement = {i: int(best.code[i]) for i in range(k)}
        mapping = ns.labels_to_task_mapping(labels, placement)
        notes['partition'] = note
    notes['mapping_seconds'] = time.perf_counter() - start
    return (mapping, notes)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workload', required=True)
    p.add_argument('--results', required=True)
    p.add_argument('--methods', default='jin,xiao,neumap,spinemap,sneap')
    p.add_argument('--fabric', choices=['tdm'], default='tdm', help='TDM evaluation (the only supported fabric)')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--verify-neural', action=argparse.BooleanOptionalAction, default=False)
    a = p.parse_args()
    out = Path(a.results)
    config = json.loads((out / 'experiment.json').read_text())
    route = json.loads((out / config['route_file']).read_text())
    bundle = load_bundle(a.workload)
    if a.verify_neural:
        require_replay_traces(bundle)
    manifest = bundle[1]
    if config['bundle_id'] != manifest['bundle_id']:
        raise ValueError('Counterparts must use the same bundle as DARSA')
    workload = make_workload(bundle[2])
    for method in a.methods.split(','):
        if method not in PROVENANCE:
            raise ValueError('Unknown method: ' + method)
        mapping, notes = mapped_tasks(method, workload, config['grid'], config['capacity'], a.seed, a.quick)
        ordered = np.array([mapping[f'neuron_{i:06d}'] for i in range(manifest['neurons'])], dtype=int)
        validate_mapping(ordered, manifest['neurons'], config['grid'] ** 2, config['capacity'])
        seconds = notes.get('mapping_seconds', 0.0)
        ci = {'mapping_seconds': seconds, 'routing_seconds': route['routing_seconds'], 'total_seconds': seconds + route['routing_seconds'], 'routing_cost_reused_from_shared_substrate': True}
        result = evaluate(bundle, ordered, route, config['capacity'], method + '_tdm', immediate=True, flit_bits=config['flit_bits'], compile_info=ci, verify_neural=a.verify_neural)
        result['provenance'] = PROVENANCE[method]
        result['mapper_details'] = notes
        result['experiment_budget'] = {'quick': a.quick, 'seed': a.seed}
        save_result(out, result)
if __name__ == '__main__':
    main()
