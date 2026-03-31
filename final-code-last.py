import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
from math import gcd
from functools import reduce
from fractions import Fraction
import matplotlib.pyplot as plt
import json
import argparse
import time

import numpy as np


# ------------------ Data model ------------------

@dataclass
class Node:
    id: int
    period: int
    alpha_base: Optional[int] = None
    # incoming list of (source_id, edge_id)
    sigma: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class Edge:
    id: int
    source: int
    target: int
    A_p: int
    U_p: int
    W_p: int


@dataclass
class MimosNetwork:
    nodes: Dict[int, Node] = field(default_factory=dict)
    edges: Dict[int, Edge] = field(default_factory=dict)

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge):
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError("Add nodes before edges.")
        if edge.id in self.edges:
            raise ValueError(f"Duplicate edge id {edge.id}.")
        if edge.W_p <= 0:
            raise ValueError(f"Edge {edge.id} has W_p={edge.W_p}; this implementation requires W_p > 0.")
        if edge.U_p < 0 or edge.A_p < 0:
            raise ValueError(f"Edge {edge.id} has negative parameters.")
        self.edges[edge.id] = edge
        self.nodes[edge.target].sigma.append((edge.source, edge.id))

    @classmethod
    def from_json_dict(cls, data: dict) -> "MimosNetwork":
        net = cls()

        for nd in data.get("nodes", []):
            node = Node(
                id=int(nd["id"]),
                period=int(nd["period"]),
                alpha_base=nd.get("alpha_base", None),
            )
            net.add_node(node)

        for ed in data.get("edges", []):
            edge = Edge(
                id=int(ed["id"]),
                source=int(ed["source"]),
                target=int(ed["target"]),
                A_p=int(ed["A_p"]),
                U_p=int(ed["U_p"]),
                W_p=int(ed["W_p"]),
            )
            net.add_edge(edge)

        return net

    @classmethod
    def from_json_file(cls, filepath: str) -> "MimosNetwork":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_json_dict(data)

    def __str__(self):
        return f"MimosNetwork(nodes={list(self.nodes.keys())}, edges={list(self.edges.keys())})"

    # ------------ Graph helpers ------------

    def _adj_out(self) -> Dict[int, List[int]]:
        adj = {j: [] for j in self.nodes}
        for e in self.edges.values():
            adj[e.source].append(e.target)
        return adj

    def _adj_in(self) -> Dict[int, List[int]]:
        adjT = {j: [] for j in self.nodes}
        for e in self.edges.values():
            adjT[e.target].append(e.source)
        return adjT

    def incoming_edges(self, j: int):
        for (_src, eid) in self.nodes[j].sigma:
            yield self.edges[eid]

    def outgoing_edges(self, i: int):
        for e in self.edges.values():
            if e.source == i:
                yield e

    def _incoming_edge_ids_all(self, j: int) -> List[int]:
        return [eid for (_src, eid) in self.nodes[j].sigma]

    def _outgoing_edge_ids_all(self, i: int) -> List[int]:
        return [e.id for e in self.edges.values() if e.source == i]

    # ------------ Kosaraju SCC (topo order, iterative) ------------

    def sccs_in_topo_order(self) -> List[List[int]]:
        adj = self._adj_out()
        adjT = self._adj_in()

        visited = {v: False for v in self.nodes}
        finish_stack: List[int] = []

        for start in self.nodes:
            if visited[start]:
                continue

            stack: List[Tuple[int, int]] = [(start, 0)]
            visited[start] = True

            while stack:
                u, idx = stack[-1]
                nbrs = adj[u]

                if idx < len(nbrs):
                    v = nbrs[idx]
                    stack[-1] = (u, idx + 1)
                    if not visited[v]:
                        visited[v] = True
                        stack.append((v, 0))
                else:
                    finish_stack.append(u)
                    stack.pop()

        visited = {v: False for v in self.nodes}
        sccs: List[List[int]] = []

        while finish_stack:
            start = finish_stack.pop()
            if visited[start]:
                continue

            comp: List[int] = []
            stack = [start]
            visited[start] = True

            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adjT[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)

            sccs.append(comp)

        return sccs

    def scc_index_map(self) -> Tuple[List[List[int]], Dict[int, int]]:
        sccs = self.sccs_in_topo_order()
        idx: Dict[int, int] = {}
        for k, comp in enumerate(sccs):
            for v in comp:
                idx[v] = k
        return sccs, idx

    def is_trivial_scc(self, comp: List[int]) -> bool:
        if len(comp) != 1:
            return False
        u = comp[0]
        for e in self.edges.values():
            if e.source == u and e.target == u:
                return False
        return True

    # ------------ ILP feasibility for SCC inequalities ------------

    def _scc_ilp_feasible(self, comp: List[int]) -> bool:
        try:
            import pulp
        except Exception as e:
            raise RuntimeError("PuLP is required for blocked-node ILP: pip install pulp") from e

        comp_set = set(comp)

        if self.is_trivial_scc(comp):
            return False

        prob = pulp.LpProblem("scc_blocked", pulp.LpMinimize)

        incoming_internal: Dict[int, List[Edge]] = {j: [] for j in comp}
        y: Dict[int, pulp.LpVariable] = {}

        for e in self.edges.values():
            if e.source in comp_set and e.target in comp_set:
                incoming_internal[e.target].append(e)
                y[e.id] = pulp.LpVariable(f"y_{e.id}", lowBound=0, upBound=1, cat=pulp.LpBinary)

        for j in comp:
            if not incoming_internal[j]:
                return False

        lb = {
            j: min(e.A_p // e.W_p for e in incoming_internal[j])
            for j in comp
        }

        x = {
            j: pulp.LpVariable(f"x_{j}", lowBound=lb[j], cat=pulp.LpInteger)
            for j in comp
        }

        for j in comp:
            prob += pulp.lpSum(y[e.id] for e in incoming_internal[j]) == 1

        M = 10 ** 6
        for j in comp:
            for e in incoming_internal[j]:
                prob += (
                    e.A_p + e.U_p * x[e.source]
                    <= e.W_p * x[j] + (e.W_p - 1) + M * (1 - y[e.id])
                )

        prob += 0

        ilp_t0 = time.perf_counter()
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        ilp_t1 = time.perf_counter()
        print(f"ILP time: {ilp_t1 - ilp_t0:.6f} s")

        return pulp.LpStatus[prob.status] == "Optimal"

    # ------------ Blocked-node detection ------------

    def x0(self, j: int) -> int:
        incoming = list(self.incoming_edges(j))
        if not incoming:
            return 1
        m = min(e.A_p // e.W_p for e in incoming)
        return min(1, m)

    def _trivial_scc_blocked(self, comp: List[int], blocked: Set[int]) -> bool:
        if not self.is_trivial_scc(comp):
            return False

        j = comp[0]

        external_preds = set()
        has_blocked_external_pred = False

        for e in self.edges.values():
            if e.target == j and e.source != j:
                external_preds.add(e.source)
                if e.source in blocked:
                    has_blocked_external_pred = True

        if has_blocked_external_pred:
            return True

        if not external_preds and self.x0(j) == 0:
            return True

        return False

    def detect_blocked_nodes(self) -> Set[int]:
        blocked: Set[int] = set()
        sccs_ordered = self.sccs_in_topo_order()

        def has_incoming_from_blocked(comp: List[int]) -> bool:
            comp_set = set(comp)
            for e in self.edges.values():
                if e.target in comp_set and e.source not in comp_set and e.source in blocked:
                    return True
            return False

        for comp in sccs_ordered:
            if self.is_trivial_scc(comp):
                if self._trivial_scc_blocked(comp, blocked):
                    blocked.update(comp)
                continue

            if has_incoming_from_blocked(comp):
                blocked.update(comp)
            else:
                if self._scc_ilp_feasible(comp):
                    blocked.update(comp)

        return blocked

    def detect_deadlock(self) -> Set[int]:
        return self.detect_blocked_nodes()

    # ---------- Cycle / SCC helpers ----------

    def _pair_to_eid(self) -> Dict[Tuple[int, int], int]:
        m: Dict[Tuple[int, int], int] = {}
        for e in self.edges.values():
            m[(e.source, e.target)] = e.id
        return m

    def _cycles_in_component(self, comp: List[int]) -> List[List[int]]:
        comp_set = set(comp)
        adj: Dict[int, List[int]] = {u: [] for u in comp}
        for e in self.edges.values():
            if e.source in comp_set and e.target in comp_set:
                adj[e.source].append(e.target)

        cycles: List[List[int]] = []

        for s in sorted(comp):
            stack = [s]
            onpath = {s}

            def dfs(u: int):
                for v in adj[u]:
                    if v < s:
                        continue
                    if v == s:
                        if len(stack) >= 2:
                            cycles.append(stack[:] + [s])
                        continue
                    if v not in onpath:
                        onpath.add(v)
                        stack.append(v)
                        dfs(v)
                        stack.pop()
                        onpath.remove(v)

            dfs(s)

        return cycles

    def detect_buffer_overflow_loops(self, restrict_to_nodes: Optional[Set[int]] = None) -> List[Dict]:
        results: List[Dict] = []
        pair2eid = self._pair_to_eid()

        for comp in self.sccs_in_topo_order():
            if restrict_to_nodes is not None and not set(comp).issubset(restrict_to_nodes):
                continue

            for cyc in self._cycles_in_component(comp):
                eids: List[int] = []
                ok = True
                for a, b in zip(cyc[:-1], cyc[1:]):
                    eid = pair2eid.get((a, b))
                    if eid is None:
                        ok = False
                        break
                    eids.append(eid)
                if not ok:
                    continue

                num, den = 1, 1
                for eid in eids:
                    e = self.edges[eid]
                    num *= e.U_p
                    den *= e.W_p

                if Fraction(num, den) > 1:
                    results.append({
                        "scc": list(comp),
                        "cycle_nodes": cyc,
                        "edge_ids": eids,
                        "g_num": num,
                        "g_den": den,
                        "g": float(Fraction(num, den)),
                    })
        return results

    # ---------------- Simulation utilities ----------------

    @staticmethod
    def _lcm(a: int, b: int) -> int:
        return a * b // gcd(a, b) if a and b else 0

    @staticmethod
    def _lcm_list(vals: List[int]) -> int:
        return reduce(MimosNetwork._lcm, vals, 1)

    @staticmethod
    def _last_boundary_leq(t: int, P: int):
        if t < 0:
            return None
        return (t // P) * P

    def hyperperiod(self, node_subset: Optional[Set[int]] = None) -> int:
        if node_subset is None:
            vals = [n.period for n in self.nodes.values()]
        else:
            vals = [self.nodes[j].period for j in node_subset]
        return self._lcm_list(vals)

    def _get_x(self, x_hist: Dict[int, Dict[int, int]], j: int, t: int) -> int:
        if t < 0:
            return 0
        Pj = self.nodes[j].period
        tb = self._last_boundary_leq(t, Pj)
        return x_hist[j].get(tb, 0)

    def _queue_value_from_x(self, x_hist: Dict[int, Dict[int, int]], eid: int, t: int) -> int:
        e = self.edges[eid]
        Pi = self.nodes[e.source].period
        xi_term = self._get_x(x_hist, e.source, t - Pi)
        xj_term = self._get_x(x_hist, e.target, t)
        return e.A_p + e.U_p * xi_term - e.W_p * xj_term

    def _event_times_up_to(self, H: int, max_hyper: int, allowed_nodes: Optional[Set[int]] = None) -> List[int]:
        active_nodes = sorted(self.nodes.keys()) if allowed_nodes is None else sorted(allowed_nodes)
        times = {0}
        horizon = H * max_hyper
        for j in active_nodes:
            Pj = self.nodes[j].period
            t = 0
            while t <= horizon:
                times.add(t)
                t += Pj
        return sorted(times)

    # ---------------- Effective period computation ----------------

    def isolated_scc_effective_periods(self, comp: List[int], max_hyper: int = 200) -> Dict[int, float]:
        comp_set = set(comp)
        internal_edges = {
            e.id for e in self.edges.values()
            if e.source in comp_set and e.target in comp_set
        }

        H_i = self.hyperperiod(comp_set)

        q: Dict[int, int] = {eid: self.edges[eid].A_p for eid in internal_edges}
        firings: Dict[int, int] = {j: 0 for j in comp}
        pending_writes: Dict[int, List[Tuple[int, int]]] = {}

        def incoming_internal(j: int) -> List[int]:
            return [eid for eid in self._incoming_edge_ids_all(j) if eid in internal_edges]

        def outgoing_internal(j: int) -> List[int]:
            return [eid for eid in self._outgoing_edge_ids_all(j) if eid in internal_edges]

        def schedule_outputs(j: int, t_release: int):
            complete_t = t_release + self.nodes[j].period
            if complete_t not in pending_writes:
                pending_writes[complete_t] = []
            for eid in outgoing_internal(j):
                pending_writes[complete_t].append((eid, self.edges[eid].U_p))

        def sampled_state() -> Tuple[int, ...]:
            return tuple(q[eid] for eid in sorted(internal_edges))

        sampled_seen: Dict[Tuple[int, ...], Tuple[int, Dict[int, int]]] = {}

        initial_fired_nodes: List[int] = []
        for j in sorted(comp):
            incoming = incoming_internal(j)
            if not incoming:
                initial_fired_nodes.append(j)
            else:
                if all(q[eid] >= self.edges[eid].W_p for eid in incoming):
                    initial_fired_nodes.append(j)

        for j in initial_fired_nodes:
            for eid in incoming_internal(j):
                q[eid] -= self.edges[eid].W_p

        for j in initial_fired_nodes:
            firings[j] += 1
            schedule_outputs(j, 0)

        state0 = sampled_state()
        sampled_seen[state0] = (0, dict(firings))

        event_times = self._event_times_up_to(H_i, max_hyper, comp_set)

        for t in event_times[1:]:
            for eid, amount in pending_writes.get(t, []):
                q[eid] += amount

            released = [j for j in sorted(comp) if t % self.nodes[j].period == 0]

            will_fire: List[int] = []
            for j in released:
                incoming = incoming_internal(j)
                if not incoming:
                    will_fire.append(j)
                else:
                    if all(q[eid] >= self.edges[eid].W_p for eid in incoming):
                        will_fire.append(j)

            for j in will_fire:
                for eid in incoming_internal(j):
                    q[eid] -= self.edges[eid].W_p

            for j in will_fire:
                firings[j] += 1
                schedule_outputs(j, t)

            if t % H_i == 0:
                m = t // H_i
                s = sampled_state()

                if s in sampled_seen:
                    m1, firings_m1 = sampled_seen[s]
                    m2 = m
                    T_hat = (m2 - m1) * H_i

                    peff: Dict[int, float] = {}
                    for j in comp:
                        activations = firings[j] - firings_m1[j]
                        peff[j] = float("inf") if activations == 0 else T_hat / activations
                    return peff

                sampled_seen[s] = (m, dict(firings))

        raise RuntimeError(f"Could not find repeated sampled SCC state within {max_hyper} hyperperiods.")

    def actual_effective_periods(
        self,
        blocked: Set[int],
        max_hyper_scc: int = 200,
    ) -> Dict[int, float]:
        sccs, scc_idx = self.scc_index_map()
        nonblocked_nodes = set(self.nodes.keys()) - blocked

        intrinsic: Dict[int, float] = {}
        actual: Dict[int, float] = {}

        for comp in sccs:
            comp_set = set(comp)
            if not comp_set.issubset(nonblocked_nodes):
                continue

            if self.is_trivial_scc(comp):
                j = comp[0]
                intrinsic[j] = float(self.nodes[j].period)
            else:
                intrinsic.update(self.isolated_scc_effective_periods(comp, max_hyper=max_hyper_scc))

            for j in comp:
                ext_candidates = []
                for e in self.incoming_edges(j):
                    if e.source in blocked:
                        continue
                    if scc_idx[e.source] != scc_idx[j]:
                        if e.source not in actual:
                            raise RuntimeError(
                                f"Missing P_eff for predecessor node {e.source} during topological propagation."
                            )
                        ext_candidates.append((e.W_p / e.U_p) * actual[e.source])

                actual[j] = max(intrinsic[j], max(ext_candidates)) if ext_candidates else intrinsic[j]

        return actual

    # ---------------- Unboundedness detection ----------------
    def detect_unbounded_queue_growth(
            self,
            blocked: Optional[Set[int]] = None,
            max_hyper_scc: int = 200,
    ) -> Dict:
        if blocked is None:
            blocked = self.detect_blocked_nodes()

        nonblocked_nodes = set(self.nodes.keys()) - blocked

        # 1) Internal SCC overflow loops among nonblocked SCCs
        overflow_loops = self.detect_buffer_overflow_loops(restrict_to_nodes=nonblocked_nodes)
        if overflow_loops:
            return {
                "status": "unbounded",
                "reason": "cycle_gain",
                "details": overflow_loops,
            }

        # 2) Effective periods on nonblocked part
        P_eff = self.actual_effective_periods(blocked, max_hyper_scc=max_hyper_scc)

        sccs, scc_idx = self.scc_index_map()
        drifts = []
        positive_drift_edges = []
        blocked_target_edges = []

        for e in self.edges.values():
            src_blocked = e.source in blocked
            tgt_blocked = e.target in blocked

            # Case A:
            # live source -> blocked target
            if (not src_blocked) and tgt_blocked:
                info = {
                    "edge_id": e.id,
                    "source": e.source,
                    "target": e.target,
                    "delta": float("inf"),
                    "reason_detail": (
                        "source is live, target is blocked; "
                        "P_eff(target)=inf so target consumption rate is zero"
                    ),
                }
                drifts.append(info)
                blocked_target_edges.append(info)
                continue

            # blocked source cannot sustain positive long-run production
            if src_blocked:
                continue

            # same-SCC edges are handled by cycle-gain analysis above
            if scc_idx[e.source] == scc_idx[e.target]:
                continue

            P_src = P_eff[e.source]
            P_tgt = P_eff.get(e.target, float("inf"))

            if math.isinf(P_tgt):
                delta = e.U_p / P_src
            else:
                delta = e.U_p / P_src - e.W_p / P_tgt

            info = {
                "edge_id": e.id,
                "source": e.source,
                "target": e.target,
                "delta": delta,
            }
            drifts.append(info)

            if delta > 0:
                positive_drift_edges.append(info)

        if blocked_target_edges:
            return {
                "status": "unbounded",
                "reason": "blocked_target_positive_input",
                "details": {
                    "edges": blocked_target_edges,
                    "P_eff": P_eff,
                    "drifts": drifts,
                },
            }

        if positive_drift_edges:
            return {
                "status": "unbounded",
                "reason": "positive_drift",
                "details": {
                    "edges": positive_drift_edges,
                    "P_eff": P_eff,
                    "drifts": drifts,
                },
            }

        return {
            "status": "no_unboundedness_detected",
            "P_eff": P_eff,
            "drifts": drifts,
        }
    # ---------------- Exact queue-bound computation ----------------

    def exact_queue_bounds_for_bounded_execution(
        self,
        max_hyper: int = 500,
        verbose: bool = False,
        store_history: bool = True,
    ) -> Dict:
        """
        Exact bounded analysis using the same simultaneous-event semantics as the
        event-driven simulator:
          1) apply writes visible at time t,
          2) evaluate eligibility of all released nodes at time t,
          3) consume for all nodes that fire at time t,
          4) schedule their future writes.
        """
        H = self.hyperperiod()

        q: Dict[int, int] = {eid: self.edges[eid].A_p for eid in self.edges}
        firings: Dict[int, int] = {j: 0 for j in self.nodes}
        pending_writes: Dict[int, List[Tuple[int, int]]] = {}

        sampled_states: Dict[Tuple[int, ...], int] = {}
        M_p: Dict[int, int] = {eid: q[eid] for eid in self.edges}

        if store_history:
            q_hist: Optional[Dict[int, Dict[int, int]]] = {eid: {0: q[eid]} for eid in self.edges}
            fire_hist: Optional[Dict[int, Dict[int, int]]] = {j: {0: 0} for j in self.nodes}
            alpha_hist: Optional[Dict[int, Dict[int, float]]] = {j: {} for j in self.nodes}
        else:
            q_hist = None
            fire_hist = None
            alpha_hist = None

        def record_time(t: int):
            for eid in self.edges:
                if q[eid] > M_p[eid]:
                    M_p[eid] = q[eid]

            if not store_history:
                return

            for eid in self.edges:
                q_hist[eid][t] = q[eid]
            for j in self.nodes:
                fire_hist[j][t] = firings[j]
                base = self.nodes[j].alpha_base or self.nodes[j].period
                alpha_hist[j][t] = firings[j] / max(1, (t // base) + 1)

        def sampled_state() -> Tuple[int, ...]:
            return tuple(q[eid] for eid in sorted(self.edges.keys()))

        def schedule_outputs(j: int, t_release: int):
            complete_t = t_release + self.nodes[j].period
            if complete_t not in pending_writes:
                pending_writes[complete_t] = []
            for eid in self._outgoing_edge_ids_all(j):
                pending_writes[complete_t].append((eid, self.edges[eid].U_p))

        initial_fired_nodes: List[int] = []
        for j in sorted(self.nodes.keys()):
            incoming = self._incoming_edge_ids_all(j)
            if not incoming:
                initial_fired_nodes.append(j)
            else:
                if all(q[eid] >= self.edges[eid].W_p for eid in incoming):
                    initial_fired_nodes.append(j)

        for j in initial_fired_nodes:
            for eid in self._incoming_edge_ids_all(j):
                q[eid] -= self.edges[eid].W_p

        for j in initial_fired_nodes:
            firings[j] += 1
            schedule_outputs(j, 0)

        record_time(0)

        state0 = sampled_state()
        sampled_states[state0] = 0
        if verbose:
            print(f"[m=0] q(0) = {state0}")

        event_times = self._event_times_up_to(H, max_hyper, set(self.nodes.keys()))

        for t in event_times[1:]:
            visible_writes = pending_writes.get(t, [])
            for eid, amount in visible_writes:
                q[eid] += amount

            released = [j for j in sorted(self.nodes.keys()) if t % self.nodes[j].period == 0]

            will_fire: List[int] = []
            for j in released:
                incoming = self._incoming_edge_ids_all(j)
                if not incoming:
                    will_fire.append(j)
                else:
                    if all(q[eid] >= self.edges[eid].W_p for eid in incoming):
                        will_fire.append(j)

            for j in will_fire:
                for eid in self._incoming_edge_ids_all(j):
                    q[eid] -= self.edges[eid].W_p

            for j in will_fire:
                firings[j] += 1
                schedule_outputs(j, t)

            record_time(t)

            if t % H == 0:
                m = t // H
                state = sampled_state()

                if verbose:
                    print(f"[m={m}] q({t}) = {state}")

                if state in sampled_states:
                    m1 = sampled_states[state]
                    m2 = m
                    t0 = m1 * H
                    T_sys = (m2 - m1) * H

                    result = {
                        "status": "bounded",
                        "H": H,
                        "m1": m1,
                        "m2": m2,
                        "t0": t0,
                        "T_sys": T_sys,
                        "M_p": M_p,
                    }
                    if store_history:
                        result["q_hist"] = q_hist
                        result["fire_hist"] = fire_hist
                        result["alpha_hist"] = alpha_hist
                    return result

                sampled_states[state] = m

        result = {
            "status": "max_iter_reached",
            "H": H,
            "M_p": M_p,
        }
        if store_history:
            result["q_hist"] = q_hist
            result["fire_hist"] = fire_hist
            result["alpha_hist"] = alpha_hist
        return result

    # ---------------- Full analysis pipeline ----------------

    def analyze_execution(
        self,
        max_hyper_bounded: int = 500,
        max_hyper_scc: int = 200,
        verbose: bool = False,
        store_history: bool = True,
    ) -> Dict:
        blocked = self.detect_blocked_nodes()

        result: Dict = {
            "blocked_nodes": blocked,
            "global_deadlock": (len(blocked) == len(self.nodes)),
        }

        if result["global_deadlock"]:
            result["status"] = "deadlock"
            return result

        ub_t0 = time.perf_counter()
        ub = self.detect_unbounded_queue_growth(
            blocked=blocked,
            max_hyper_scc=max_hyper_scc,
        )
        ub_t1 = time.perf_counter()
        print(f"unboundedness time: {ub_t1 - ub_t0:.6f} s")
        result["unboundedness"] = ub

        if ub["status"] == "unbounded":
            result["status"] = "unbounded"
            return result

        b_t0 = time.perf_counter()
        bounded = self.exact_queue_bounds_for_bounded_execution(
            max_hyper=max_hyper_bounded,
            verbose=verbose,
            store_history=store_history,
        )
        b_t1 = time.perf_counter()
        print(f"exact queue bounds time: {b_t1 - b_t0:.6f} s")

        result["bounded_analysis"] = bounded
        result["status"] = bounded["status"]
        return result

    # ---------------- Independent event-driven simulator ----------------

    def simulate_event_driven(
        self,
        max_hyper: int = 200,
        stop_on_repeat: bool = True,
        stop_on_deadlock: bool = True,
        allowed_nodes: Optional[Set[int]] = None,
        allowed_edges: Optional[Set[int]] = None,
        verbose: bool = False,
        store_history: bool = False,
    ) -> Dict:
        active_nodes = set(self.nodes.keys()) if allowed_nodes is None else set(allowed_nodes)
        active_edges = set(self.edges.keys()) if allowed_edges is None else set(allowed_edges)

        H = self.hyperperiod(active_nodes)

        q: Dict[int, int] = {eid: self.edges[eid].A_p for eid in active_edges}
        firings: Dict[int, int] = {j: 0 for j in active_nodes}
        pending_writes: Dict[int, List[Tuple[int, int]]] = {}

        sampled_seen: Dict[Tuple[int, ...], int] = {}
        M_p: Dict[int, int] = {eid: q[eid] for eid in active_edges}

        if store_history:
            q_hist: Optional[Dict[int, Dict[int, int]]] = {eid: {0: q[eid]} for eid in active_edges}
            fire_hist: Optional[Dict[int, Dict[int, int]]] = {j: {0: 0} for j in active_nodes}
            alpha_hist: Optional[Dict[int, Dict[int, float]]] = {j: {} for j in active_nodes}
        else:
            q_hist = None
            fire_hist = None
            alpha_hist = None

        def record_time(t: int):
            for eid in active_edges:
                if q[eid] > M_p[eid]:
                    M_p[eid] = q[eid]

            if not store_history:
                return

            for eid in active_edges:
                q_hist[eid][t] = q[eid]
            for j in active_nodes:
                fire_hist[j][t] = firings[j]
                base = self.nodes[j].alpha_base or self.nodes[j].period
                alpha_hist[j][t] = firings[j] / max(1, (t // base) + 1)

        def sampled_state() -> Tuple[int, ...]:
            return tuple(q[eid] for eid in sorted(active_edges))

        def schedule_outputs(j: int, t_release: int):
            complete_t = t_release + self.nodes[j].period
            if complete_t not in pending_writes:
                pending_writes[complete_t] = []
            for eid in self._outgoing_edge_ids_all(j):
                if eid in active_edges:
                    pending_writes[complete_t].append((eid, self.edges[eid].U_p))

        def is_deadlocked_after_time(t: int) -> bool:
            released = [j for j in sorted(active_nodes) if t % self.nodes[j].period == 0]
            for j in released:
                incoming = [eid for eid in self._incoming_edge_ids_all(j) if eid in active_edges]
                if not incoming:
                    return False
                if all(q[eid] >= self.edges[eid].W_p for eid in incoming):
                    return False
            return True

        initial_fired_nodes: List[int] = []
        for j in sorted(active_nodes):
            incoming = [eid for eid in self._incoming_edge_ids_all(j) if eid in active_edges]
            if not incoming:
                initial_fired_nodes.append(j)
            else:
                if all(q[eid] >= self.edges[eid].W_p for eid in incoming):
                    initial_fired_nodes.append(j)

        for j in initial_fired_nodes:
            for eid in [eid for eid in self._incoming_edge_ids_all(j) if eid in active_edges]:
                q[eid] -= self.edges[eid].W_p

        for j in initial_fired_nodes:
            firings[j] += 1
            schedule_outputs(j, 0)

        record_time(0)

        state0 = sampled_state()
        sampled_seen[state0] = 0
        if verbose:
            print(f"[sim] m=0, t=0, sampled_state={state0}, initial_fired={initial_fired_nodes}")

        event_times = self._event_times_up_to(H, max_hyper, active_nodes)

        for t in event_times[1:]:
            visible_writes = pending_writes.get(t, [])
            for eid, amount in visible_writes:
                q[eid] += amount

            released = [j for j in sorted(active_nodes) if t % self.nodes[j].period == 0]

            will_fire: List[int] = []
            for j in released:
                incoming = [eid for eid in self._incoming_edge_ids_all(j) if eid in active_edges]
                if not incoming:
                    will_fire.append(j)
                else:
                    if all(q[eid] >= self.edges[eid].W_p for eid in incoming):
                        will_fire.append(j)

            for j in will_fire:
                for eid in [eid for eid in self._incoming_edge_ids_all(j) if eid in active_edges]:
                    q[eid] -= self.edges[eid].W_p

            for j in will_fire:
                firings[j] += 1
                schedule_outputs(j, t)

            record_time(t)

            if verbose:
                print(f"[sim] t={t}, writes={visible_writes}, released={released}, fired={will_fire}")

            if t % H == 0:
                m = t // H
                state = sampled_state()
                if verbose:
                    print(f"[sim] m={m}, t={t}, sampled_state={state}")

                if stop_on_repeat:
                    if state in sampled_seen:
                        m1 = sampled_seen[state]
                        result = {
                            "status": "repeat_detected",
                            "H": H,
                            "m1": m1,
                            "m2": m,
                            "t0": m1 * H,
                            "T_sys": (m - m1) * H,
                            "M_p": M_p,
                            "final_queues": dict(q),
                            "final_firings": dict(firings),
                        }
                        if store_history:
                            result["q_hist"] = q_hist
                            result["fire_hist"] = fire_hist
                            result["alpha_hist"] = alpha_hist
                        return result
                    sampled_seen[state] = m

                if stop_on_deadlock and is_deadlocked_after_time(t):
                    result = {
                        "status": "deadlock_detected",
                        "H": H,
                        "t_deadlock": t,
                        "M_p": M_p,
                        "final_queues": dict(q),
                        "final_firings": dict(firings),
                    }
                    if store_history:
                        result["q_hist"] = q_hist
                        result["fire_hist"] = fire_hist
                        result["alpha_hist"] = alpha_hist
                    return result

        result = {
            "status": "max_iter_reached",
            "H": H,
            "M_p": M_p,
            "final_queues": dict(q),
            "final_firings": dict(firings),
        }
        if store_history:
            result["q_hist"] = q_hist
            result["fire_hist"] = fire_hist
            result["alpha_hist"] = alpha_hist
        return result


# ---------------- Plotting helpers ----------------

def _add_transient_steady_markers(ax, t0: Optional[int] = None, T_sys: Optional[int] = None):
    if t0 is None:
        return
    ax.axvline(t0, linestyle="-.", linewidth=1.5, label="transient end")
    if T_sys is not None:
        ax.axvline(t0 + T_sys, linestyle="--", linewidth=1.5, label="one steady-state period")
        ax.axvspan(t0, t0 + T_sys, alpha=0.15)


def plot_queues_separate(
    q_hist: Dict[int, Dict[int, int]],
    edges: Dict[int, Edge],
    title_prefix: str = "Queue",
    t0: Optional[int] = None,
    T_sys: Optional[int] = None,
):
    for eid, hist in q_hist.items():
        if not hist:
            continue
        times = sorted(hist.keys())
        vals = [hist[t] for t in times]
        e = edges[eid]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.step(times, vals, where="post")
        _add_transient_steady_markers(ax, t0=t0, T_sys=T_sys)
        ax.set_xlabel("time", fontsize=18)
        ax.set_ylabel(rf"$Q_{{{eid}}}(t)$", fontsize=18)
        #ax.set_title(f"Queue size for edge {eid}", fontsize=18)
        ax.grid(True, which="both", linestyle=":")
        ax.tick_params(axis="both", labelsize=16)
        ymax = max(vals)
        ax.set_yticks(range(0, int(math.ceil(ymax)) + 3, 1))
        ax.set_ylim(0, int(math.ceil(ymax)) + 3)
        ax.legend()
        fig.tight_layout()
        plt.savefig(f"queue_{eid}.pdf")
        plt.show()


def plot_alpha_separate(
    alpha_hist: Dict[int, Dict[int, float]],
    title_prefix: str = "Activation ratio",
    t0: Optional[int] = None,
    T_sys: Optional[int] = None,
):
    for nid, hist in alpha_hist.items():
        if not hist:
            continue
        times = sorted(hist.keys())
        vals = [hist[t] for t in times]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.step(times, vals, where="post")
        _add_transient_steady_markers(ax, t0=t0, T_sys=T_sys)
        ax.set_xlabel("time", fontsize=18)
        ax.set_ylabel(rf"$\alpha({nid},t)$", fontsize=18)
        #ax.set_title(f"Activation ratio for node {nid}", fontsize=18)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, which="both", linestyle=":")
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.01, 0.1))
        ax.legend()
        fig.tight_layout()
        plt.savefig(f"alpha_{nid}.pdf")
        plt.show()


# ---------------- Comparison helper ----------------

def compare_analysis_with_event_simulation(net: MimosNetwork, max_hyper: int = 200, verbose: bool = True):
    print("Running analysis...")
    t0 = time.perf_counter()
    analysis = net.analyze_execution(
        max_hyper_bounded=max_hyper,
        verbose=verbose,
        store_history=False,
    )
    t1 = time.perf_counter()
    analysis_time = t1 - t0

    print("\nRunning independent event-driven simulation...")
    t2 = time.perf_counter()
    sim = net.simulate_event_driven(
        max_hyper=max_hyper,
        stop_on_deadlock=False,
        verbose=verbose,
        store_history=False,
    )
    t3 = time.perf_counter()
    simulation_time = t3 - t2

    print("\n--- Comparison ---")
    print("Analysis status:", analysis["status"])
    print("Simulation status:", sim["status"])
    print(f"Analysis execution time:   {analysis_time:.6f} s")
    print(f"Simulation execution time: {simulation_time:.6f} s")

    if analysis["status"] == "bounded" and sim["status"] == "repeat_detected":
        b = analysis["bounded_analysis"]

        print("Analysis H       =", b["H"])
        print("Simulation H     =", sim["H"])
        print("Analysis t0      =", b["t0"])
        print("Simulation t0    =", sim["t0"])
        print("Analysis T_sys   =", b["T_sys"])
        print("Simulation T_sys =", sim["T_sys"])
        print("Analysis M_p     =", b["M_p"])
        print("Simulation M_p   =", sim["M_p"])

    elif sim["status"] == "deadlock_detected":
        print("Simulation deadlock time:", sim["t_deadlock"])

    return {
        "analysis": analysis,
        "simulation": sim,
        "analysis_time_sec": analysis_time,
        "simulation_time_sec": simulation_time,
    }


# ------------------ JSON runner ------------------

def run_from_json(filepath: str, verbose: bool = True, run_sim: bool = True, plot_analysis: bool = True):
    net = MimosNetwork.from_json_file(filepath)

    print(net)
    print("SCCs (topo):", net.sccs_in_topo_order())

    analysis_store_history = plot_analysis

    t0 = time.perf_counter()
    res = net.analyze_execution(
        verbose=verbose,
        store_history=analysis_store_history,
    )
    t1 = time.perf_counter()

    print("\nFinal analysis status:", res["status"])
    print("Blocked nodes:", sorted(res["blocked_nodes"]))
    print("Global deadlock:", res["global_deadlock"])
    print(f"Analysis execution time: {t1 - t0:.6f} s")

    if res["status"] == "unbounded":
        print("Unboundedness reason:", res["unboundedness"]["reason"])
        print("Details:", res["unboundedness"]["details"])

    elif res["status"] == "bounded":
        b = res["bounded_analysis"]
        print("H =", b["H"])
        print("t0 =", b["t0"])
        print("T_sys =", b["T_sys"])
        print("Exact queue bounds M_p =", b["M_p"])

        if plot_analysis:
            plot_queues_separate(
                b["q_hist"],
                net.edges,
                title_prefix="Analysis queue",
                t0=b["t0"],
                T_sys=b["T_sys"],
            )
            plot_alpha_separate(
                b["alpha_hist"],
                title_prefix="Analysis alpha",
                t0=b["t0"],
                T_sys=b["T_sys"],
            )

    elif res["status"] == "deadlock":
        print("The network is globally deadlocked.")

    if run_sim:
        print("\n--- Running independent event-driven simulator ---")
        t2 = time.perf_counter()
        sim = net.simulate_event_driven(
            max_hyper=200,
            stop_on_deadlock=False,
            verbose=verbose,
            store_history=False,
        )
        t3 = time.perf_counter()

        print("Simulation status:", sim["status"])
        print(f"Simulation execution time: {t3 - t2:.6f} s")

        if sim["status"] == "repeat_detected":
            print("Simulation H =", sim["H"])
            print("Simulation t0 =", sim["t0"])
            print("Simulation Tsys =", sim["T_sys"])
            print("Simulation queue bounds M_p =", sim["M_p"])
        elif sim["status"] == "deadlock_detected":
            print("Simulation deadlock detected at t =", sim["t_deadlock"])
        else:
            print("Simulation reached max iterations without repeat/deadlock.")


# ------------------ Examples ------------------

def example1(verbose: bool = True, run_sim: bool = True, plot_analysis: bool = True):
    net = MimosNetwork()

    net.add_node(Node(id=0, period=3))
    net.add_node(Node(id=1, period=5))
    net.add_node(Node(id=2, period=7))

    # Matches the paper example:
    # n1 -> n2 : (U,W,A) = (3,5,14)
    # n2 -> n3 : (U,W,A) = (5,7,0)
    # n3 -> n1 : (U,W,A) = (7,3,0)
    net.add_edge(Edge(id=0, source=0, target=1, U_p=3, W_p=5, A_p=14))
    net.add_edge(Edge(id=1, source=1, target=2, U_p=5, W_p=7, A_p=0))
    net.add_edge(Edge(id=2, source=2, target=0, U_p=7, W_p=3, A_p=0))

    print(net)
    print("SCCs (topo):", net.sccs_in_topo_order())

    analysis_store_history = plot_analysis

    res = net.analyze_execution(
        verbose=verbose,
        store_history=analysis_store_history,
    )

    print("\nFinal status:", res["status"])
    print("Blocked nodes:", res["blocked_nodes"])
    print("Global deadlock:", res["global_deadlock"])

    if res["status"] == "unbounded":
        print("Unboundedness reason:", res["unboundedness"]["reason"])
        print("Details:", res["unboundedness"]["details"])

    elif res["status"] == "bounded":
        b = res["bounded_analysis"]
        print("H =", b["H"])
        print("t0 =", b["t0"])
        print("T_sys =", b["T_sys"])
        print("Exact queue bounds M_p =", b["M_p"])
        if plot_analysis:
            plot_queues_separate(
                b["q_hist"],
                net.edges,
                title_prefix="Analysis queue",
                t0=b["t0"],
                T_sys=b["T_sys"],
            )
            plot_alpha_separate(
                b["alpha_hist"],
                title_prefix="Analysis alpha",
                t0=b["t0"],
                T_sys=b["T_sys"],
            )

    elif res["status"] == "deadlock":
        print("The network is globally deadlocked.")

    if run_sim:
        print("\nNow compare with event-driven simulation:")
        compare_analysis_with_event_simulation(net, max_hyper=200, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a benchmark JSON file.")
    parser.add_argument("--json", type=str, help="Path to benchmark.json")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose analysis output")
    parser.add_argument("--no-sim", action="store_true", help="Disable standalone event-driven simulation")
    parser.add_argument("--no-plot", action="store_true", help="Disable plots for analysis approach")
    args = parser.parse_args()

    if args.json:
        run_from_json(
            args.json,
            verbose=not args.quiet,
            run_sim=not args.no_sim,
            plot_analysis=not args.no_plot,
        )
    else:
        example1(
            verbose=not args.quiet,
            run_sim=not args.no_sim,
            plot_analysis=not args.no_plot,
        )