from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
from math import gcd
from functools import reduce
from fractions import Fraction
import matplotlib.pyplot as plt
import json
import argparse
import time


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

    def x0(self, j: int) -> int:
        incoming = list(self.incoming_edges(j))
        if not incoming:
            return 1

        m = min(e.A_p // e.W_p for e in incoming)
        return min(1, m)

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

    def _update_at_boundary(
        self,
        x_hist: Dict[int, Dict[int, int]],
        j: int,
        t: int,
        allowed_nodes: Optional[Set[int]] = None,
        allowed_edges: Optional[Set[int]] = None,
    ):
        Pj = self.nodes[j].period

        if allowed_nodes is not None and j not in allowed_nodes:
            return

        if t == 0:
            x_hist[j][0] = self.x0(j)
            return

        prev_xj = self._get_x(x_hist, j, t - Pj)
        pacing_term = prev_xj + 1

        sigma_edges = []
        for e in self.incoming_edges(j):
            if allowed_edges is not None and e.id not in allowed_edges:
                continue
            if allowed_nodes is not None and (e.source not in allowed_nodes or e.target not in allowed_nodes):
                continue
            sigma_edges.append(e)

        if len(sigma_edges) == 0:
            val = prev_xj
        else:
            token_terms = []
            for e in sigma_edges:
                Pi = self.nodes[e.source].period
                xi_term = self._get_x(x_hist, e.source, t - Pi)
                token_terms.append((e.A_p + e.U_p * xi_term) // e.W_p)
            token_term = min(token_terms)
            val = min(pacing_term, token_term)

        x_hist[j][t] = max(x_hist[j].get(t, 0), val)

    def alpha_at(self, x_hist: Dict[int, Dict[int, int]], i: int, t: int) -> float:
        base = self.nodes[i].alpha_base or self.nodes[i].period
        denom = max(1, (t // base) + 1)
        return self._get_x(x_hist, i, t) / denom

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

    # ---------------- Phase 1: structural overflow using Bellman-Ford ----------------

    def _scc_has_gain_gt_one_bellman_ford(self, comp: List[int]) -> Tuple[bool, Optional[Dict]]:
        """
        Internal SCC overflow test:
            dist[j] >= dist[i] * (U_p / W_p)

        Initialize all nodes in the SCC with 1.
        If an update still happens on the n-th round, then some cycle has gain > 1.
        """
        comp_set = set(comp)
        internal_edges = [
            e for e in self.edges.values()
            if e.source in comp_set and e.target in comp_set
        ]

        if not internal_edges:
            return False, None

        n = len(comp)
        dist: Dict[int, Fraction] = {j: Fraction(1, 1) for j in comp}

        for it in range(n):
            updated = False
            last_edge = None

            for e in internal_edges:
                cand = dist[e.source] * Fraction(e.U_p, e.W_p)
                if cand > dist[e.target]:
                    dist[e.target] = cand
                    updated = True
                    last_edge = e

                    if it == n - 1:
                        return True, {
                            "scc": list(comp),
                            "witness_edge_id": last_edge.id,
                            "source": last_edge.source,
                            "target": last_edge.target,
                            "reason": "gain_gt_one_cycle",
                        }

            if not updated:
                return False, None

        return False, None

    def detect_internal_scc_overflow(self) -> List[Dict]:
        results: List[Dict] = []
        for comp in self.sccs_in_topo_order():
            has_overflow, details = self._scc_has_gain_gt_one_bellman_ford(comp)
            if has_overflow:
                results.append(details)
        return results

    # ---------------- Phase 2: compute intrinsic P_eff of each SCC ----------------

    def isolated_scc_steady_state(self, comp: List[int], max_hyper: int = 200) -> Dict:
        """
        Analyze one SCC in isolation, without storing full history.

        Returns:
          - H_i: SCC hyperperiod
          - T_hat: steady-state period
          - firings_delta[j]: firings of node j over one steady-state period
          - P_eff[j]
          - deadlocked: True iff all firings over one steady-state period are zero
        """
        comp_set = set(comp)

        internal_edges = {
            e.id for e in self.edges.values()
            if e.source in comp_set and e.target in comp_set
        }

        if len(comp) == 1 and len(internal_edges) == 0:
            j = comp[0]
            return {
                "H_i": self.nodes[j].period,
                "m1": 0,
                "m2": 1,
                "T_hat": self.nodes[j].period,
                "firings_delta": {j: 1},
                "P_eff": {j: float(self.nodes[j].period)},
                "deadlocked": False,
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
            pending_writes.setdefault(complete_t, [])
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

        sampled_seen[sampled_state()] = (0, dict(firings))

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

                    firings_delta = {
                        j: firings[j] - firings_m1[j]
                        for j in comp
                    }

                    P_eff = {}
                    for j in comp:
                        if firings_delta[j] == 0:
                            P_eff[j] = float("inf")
                        else:
                            P_eff[j] = T_hat / firings_delta[j]

                    deadlocked = all(firings_delta[j] == 0 for j in comp)

                    return {
                        "H_i": H_i,
                        "m1": m1,
                        "m2": m2,
                        "T_hat": T_hat,
                        "firings_delta": firings_delta,
                        "P_eff": P_eff,
                        "deadlocked": deadlocked,
                    }

                sampled_seen[s] = (m, dict(firings))

        raise RuntimeError(
            f"Could not find repeated sampled SCC state within {max_hyper} hyperperiods for SCC {comp}."
        )

    def compute_actual_peff(
        self,
        overflowing_scc_sets: Optional[List[Set[int]]] = None,
        max_hyper_scc: int = 200,
    ) -> Dict:
        """
        New phase after structural overflow:
          - compute intrinsic SCC P_eff from isolated steady state
          - then propagate actual P_eff topologically
          - blocked SCCs are those with P_eff = inf
        """
        sccs, scc_idx = self.scc_index_map()

        if overflowing_scc_sets is None:
            overflow_loops = self.detect_internal_scc_overflow()
            overflowing_scc_sets = [set(item["scc"]) for item in overflow_loops]

        intrinsic_P_eff: Dict[int, float] = {}
        actual_P_eff: Dict[int, float] = {}
        blocked_nodes: Set[int] = set()
        blocked_sccs: List[List[int]] = []
        scc_summaries: Dict[int, Dict] = {}

        def has_incoming_from_blocked(comp: List[int]) -> bool:
            comp_set = set(comp)
            for e in self.edges.values():
                if e.target in comp_set and e.source not in comp_set and e.source in blocked_nodes:
                    return True
            return False

        for k, comp in enumerate(sccs):
            comp_set = set(comp)

            # overflowing SCCs should have been handled earlier
            if any(comp_set == ov for ov in overflowing_scc_sets):
                scc_summaries[k] = {
                    "comp": comp,
                    "overflow": True,
                }
                continue

            # propagate blockedness from predecessors, as in your theory
            if has_incoming_from_blocked(comp):
                blocked_nodes.update(comp)
                blocked_sccs.append(comp)
                for j in comp:
                    intrinsic_P_eff[j] = float("inf")
                    actual_P_eff[j] = float("inf")
                scc_summaries[k] = {
                    "comp": comp,
                    "overflow": False,
                    "blocked": True,
                    "reason": "propagated_from_blocked_predecessor",
                }
                continue

            # intrinsic SCC analysis
            if self.is_trivial_scc(comp):
                j = comp[0]
                intrinsic_P_eff[j] = float(self.nodes[j].period)
                actual_P_eff[j] = intrinsic_P_eff[j]
                scc_summaries[k] = {
                    "comp": comp,
                    "overflow": False,
                    "blocked": False,
                    "T_hat": self.nodes[j].period,
                    "firings_delta": {j: 1},
                    "P_eff": {j: intrinsic_P_eff[j]},
                }
            else:
                info = self.isolated_scc_steady_state(comp, max_hyper=max_hyper_scc)
                for j in comp:
                    intrinsic_P_eff[j] = info["P_eff"][j]
                scc_summaries[k] = {
                    "comp": comp,
                    "overflow": False,
                    **info,
                }

                # blocked SCC: P_eff = inf for all nodes
                if info["deadlocked"]:
                    blocked_nodes.update(comp)
                    blocked_sccs.append(comp)
                    for j in comp:
                        actual_P_eff[j] = float("inf")
                    continue

                # live SCC: propagate actual P_eff
                for j in comp:
                    ext_candidates = []
                    for e in self.incoming_edges(j):
                        if scc_idx[e.source] != scc_idx[j]:
                            peff_src = actual_P_eff[e.source]
                            if peff_src != float("inf"):
                                ext_candidates.append((e.W_p / e.U_p) * peff_src)

                    if ext_candidates:
                        actual_P_eff[j] = max(intrinsic_P_eff[j], max(ext_candidates))
                    else:
                        actual_P_eff[j] = intrinsic_P_eff[j]

                continue

            # trivial SCC case with possible external constraints
            for j in comp:
                ext_candidates = []
                for e in self.incoming_edges(j):
                    if scc_idx[e.source] != scc_idx[j]:
                        peff_src = actual_P_eff[e.source]
                        if peff_src != float("inf"):
                            ext_candidates.append((e.W_p / e.U_p) * peff_src)

                if ext_candidates:
                    actual_P_eff[j] = max(intrinsic_P_eff[j], max(ext_candidates))
                else:
                    actual_P_eff[j] = intrinsic_P_eff[j]

        return {
            "intrinsic_P_eff": intrinsic_P_eff,
            "actual_P_eff": actual_P_eff,
            "blocked_nodes": blocked_nodes,
            "blocked_sccs": blocked_sccs,
            "scc_summaries": scc_summaries,
        }

    # ---------------- Phase 3: inter-SCC overflow using P_eff ----------------

    def detect_inter_scc_overflow_from_peff(
        self,
        actual_P_eff: Dict[int, float],
        blocked_nodes: Set[int],
    ) -> Dict:
        sccs, scc_idx = self.scc_index_map()
        drifts = []

        for e in self.edges.values():
            if e.source in blocked_nodes or e.target in blocked_nodes:
                continue
            if scc_idx[e.source] == scc_idx[e.target]:
                continue

            peff_src = actual_P_eff[e.source]
            peff_tgt = actual_P_eff[e.target]

            prod_rate = 0.0 if peff_src == float("inf") else e.U_p / peff_src
            cons_rate = 0.0 if peff_tgt == float("inf") else e.W_p / peff_tgt
            delta = prod_rate - cons_rate

            info = {
                "edge_id": e.id,
                "source": e.source,
                "target": e.target,
                "delta": delta,
                "prod_rate": prod_rate,
                "cons_rate": cons_rate,
            }
            drifts.append(info)

            if delta > 0:
                return {
                    "status": "unbounded",
                    "reason": "positive_drift",
                    "details": info,
                    "drifts": drifts,
                }

        return {
            "status": "no_unboundedness_detected",
            "drifts": drifts,
        }

    # ---------------- Exact queue-bound computation ----------------

    def exact_queue_bounds_for_bounded_execution(
        self,
        max_hyper: int = 500,
        verbose: bool = False,
        store_history: bool = True,
    ) -> Dict:
        H = self.hyperperiod()

        if store_history:
            x_hist: Dict[int, Dict[int, int]] = {j: {0: self.x0(j)} for j in self.nodes}
            q_hist: Dict[int, Dict[int, int]] = {eid: {} for eid in self.edges}
            alpha_hist: Dict[int, Dict[int, float]] = {j: {} for j in self.nodes}

            sampled_states: Dict[Tuple[int, ...], int] = {}
            M_p: Dict[int, int] = {eid: float("-inf") for eid in self.edges}

            def queue_value_at(eid: int, t: int) -> int:
                return self._queue_value_from_x(x_hist, eid, t)

            def sampled_state_at(t: int) -> Tuple[int, ...]:
                return tuple(queue_value_at(eid, t) for eid in sorted(self.edges.keys()))

            def update_time_record(t: int):
                for eid in self.edges:
                    qv = queue_value_at(eid, t)
                    if qv > M_p[eid]:
                        M_p[eid] = qv
                    q_hist[eid][t] = qv

                for j, node in self.nodes.items():
                    base = node.alpha_base or node.period
                    denom = max(1, (t // base) + 1)
                    alpha_hist[j][t] = self._get_x(x_hist, j, t) / denom

            update_time_record(0)

            events: List[Tuple[int, int]] = []
            horizon = max_hyper * H
            for j in self.nodes:
                Pj = self.nodes[j].period
                t = Pj
                while t <= horizon:
                    events.append((t, j))
                    t += Pj
            events.sort()
            event_idx = 0

            for m in range(max_hyper + 1):
                t_boundary = m * H

                while event_idx < len(events) and events[event_idx][0] <= t_boundary:
                    t, j = events[event_idx]
                    self._update_at_boundary(x_hist, j, t)
                    update_time_record(t)
                    event_idx += 1

                state = sampled_state_at(t_boundary)

                if verbose:
                    print(f"[m={m}] q({t_boundary}) = {state}")

                if state in sampled_states:
                    m1 = sampled_states[state]
                    m2 = m
                    t0 = m1 * H
                    T_sys = (m2 - m1) * H

                    return {
                        "status": "bounded",
                        "H": H,
                        "m1": m1,
                        "m2": m2,
                        "t0": t0,
                        "T_sys": T_sys,
                        "M_p": M_p,
                        "x_hist": x_hist,
                        "q_hist": q_hist,
                        "alpha_hist": alpha_hist,
                    }

                sampled_states[state] = m

            return {
                "status": "max_iter_reached",
                "H": H,
                "M_p": M_p,
                "x_hist": x_hist,
                "q_hist": q_hist,
                "alpha_hist": alpha_hist,
            }

        q: Dict[int, int] = {eid: self.edges[eid].A_p for eid in self.edges}
        firings: Dict[int, int] = {j: 0 for j in self.nodes}
        pending_writes: Dict[int, List[Tuple[int, int]]] = {}
        sampled_states: Dict[Tuple[int, ...], int] = {}
        M_p: Dict[int, int] = {eid: q[eid] for eid in self.edges}

        def sampled_state() -> Tuple[int, ...]:
            return tuple(q[eid] for eid in sorted(self.edges.keys()))

        def schedule_outputs(j: int, t_release: int):
            complete_t = t_release + self.nodes[j].period
            if complete_t not in pending_writes:
                pending_writes[complete_t] = []
            for eid in self._outgoing_edge_ids_all(j):
                pending_writes[complete_t].append((eid, self.edges[eid].U_p))

        def record_post_event_state():
            for eid in self.edges:
                if q[eid] > M_p[eid]:
                    M_p[eid] = q[eid]

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

        record_post_event_state()

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

            record_post_event_state()

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
                    return {
                        "status": "bounded",
                        "H": H,
                        "m1": m1,
                        "m2": m2,
                        "t0": t0,
                        "T_sys": T_sys,
                        "M_p": M_p,
                    }

                sampled_states[state] = m

        return {
            "status": "max_iter_reached",
            "H": H,
            "M_p": M_p,
        }

    # ---------------- Full analysis pipeline (new order) ----------------

    def analyze_execution(
        self,
        max_hyper_bounded: int = 500,
        max_hyper_scc: int = 200,
        verbose: bool = False,
        store_history: bool = True,
    ) -> Dict:
        result: Dict = {}

        # Phase 1: structural overflow inside SCCs
        overflow_loops = self.detect_internal_scc_overflow()
        result["internal_overflow_loops"] = overflow_loops

        if overflow_loops:
            result["status"] = "unbounded"
            result["reason"] = "internal_scc_overflow"
            result["unboundedness"] = {
                "status": "unbounded",
                "reason": "gain_gt_one",
                "details": overflow_loops,
            }
            result["blocked_nodes"] = set()
            result["global_deadlock"] = False
            return result

        overflowing_scc_sets = [set(item["scc"]) for item in overflow_loops]

        # Phase 2: compute P_eff of SCCs
        peff_info = self.compute_actual_peff(
            overflowing_scc_sets=overflowing_scc_sets,
            max_hyper_scc=max_hyper_scc,
        )

        actual_P_eff = peff_info["actual_P_eff"]
        blocked_nodes = peff_info["blocked_nodes"]

        result["actual_P_eff"] = actual_P_eff
        result["intrinsic_P_eff"] = peff_info["intrinsic_P_eff"]
        result["blocked_nodes"] = blocked_nodes
        result["blocked_sccs"] = peff_info["blocked_sccs"]
        result["scc_summaries"] = peff_info["scc_summaries"]
        result["global_deadlock"] = (len(blocked_nodes) == len(self.nodes))

        # Phase 3: blocked SCCs are those with P_eff = inf
        if result["global_deadlock"]:
            result["status"] = "deadlock"
            return result

        # Phase 4: inter-SCC overflow using P_eff
        ub = self.detect_inter_scc_overflow_from_peff(
            actual_P_eff=actual_P_eff,
            blocked_nodes=blocked_nodes,
        )
        result["inter_scc_unboundedness"] = ub

        if ub["status"] == "unbounded":
            result["status"] = "unbounded"
            result["reason"] = "inter_scc_overflow"
            return result

        # Phase 5: exact queue size
        bounded = self.exact_queue_bounds_for_bounded_execution(
            max_hyper=max_hyper_bounded,
            verbose=verbose,
            store_history=store_history,
        )
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
                print(
                    f"[sim] t={t}, writes={visible_writes}, "
                    f"released={released}, fired={will_fire}"
                )

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

def plot_queues_separate(q_hist: Dict[int, Dict[int, int]], edges: Dict[int, Edge], title_prefix: str = "Queue"):
    for eid, hist in q_hist.items():
        if not hist:
            continue
        times = sorted(hist.keys())
        vals = [hist[t] for t in times]
        e = edges[eid]
        plt.figure(figsize=(8, 4))
        plt.step(times, vals, where="post")
        plt.xlabel("time")
        plt.ylabel(f"q_e{eid} ({e.source}->{e.target})")
        plt.title(f"{title_prefix}: edge {eid} ({e.source}->{e.target})")
        plt.grid(True, which="both", linestyle=":")
        plt.tight_layout()
        plt.show()


def plot_alpha_separate(alpha_hist: Dict[int, Dict[int, float]], title_prefix: str = "Alpha"):
    for nid, hist in alpha_hist.items():
        if not hist:
            continue
        times = sorted(hist.keys())
        vals = [hist[t] for t in times]
        plt.figure(figsize=(8, 4))
        plt.step(times, vals, where="post")
        plt.xlabel("time")
        plt.ylabel(f"alpha_{nid}(t)")
        plt.title(f"{title_prefix} for node {nid}")
        plt.grid(True, which="both", linestyle=":")
        plt.tight_layout()
        plt.show()


# ---------------- Comparison helper ----------------

def compare_analysis_with_event_simulation(
    net: MimosNetwork,
    max_hyper: int = 200,
    verbose: bool = True,
):
    print("Running analysis...")
    t0 = time.perf_counter()
    analysis = net.analyze_execution(
        max_hyper_bounded=max_hyper,
        max_hyper_scc=max_hyper,
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

    if "blocked_nodes" in analysis:
        print("Blocked nodes:", sorted(analysis["blocked_nodes"]))

    if "actual_P_eff" in analysis:
        print("Actual P_eff:", analysis["actual_P_eff"])

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

    analysis_store_history = not run_sim

    t0 = time.perf_counter()
    res = net.analyze_execution(
        verbose=verbose,
        store_history=analysis_store_history,
    )
    t1 = time.perf_counter()

    print("\nFinal analysis status:", res["status"])
    print("Blocked nodes:", sorted(res.get("blocked_nodes", set())))
    print("Global deadlock:", res.get("global_deadlock", False))
    print(f"Analysis execution time: {t1 - t0:.6f} s")

    if res["status"] == "unbounded":
        if res.get("reason") == "internal_scc_overflow":
            print("Unboundedness reason: internal SCC overflow")
            print("Details:", res["unboundedness"]["details"])
        elif res.get("reason") == "inter_scc_overflow":
            print("Unboundedness reason: inter-SCC positive drift")
            print("Details:", res["inter_scc_unboundedness"]["details"])

    elif res["status"] == "bounded":
        b = res["bounded_analysis"]
        print("H =", b["H"])
        print("t0 =", b["t0"])
        print("T_sys =", b["T_sys"])
        print("Exact queue bounds M_p =", b["M_p"])
        print("Actual P_eff =", res["actual_P_eff"])

        if plot_analysis and not run_sim:
            plot_queues_separate(b["q_hist"], net.edges, title_prefix="Analysis queue")
            plot_alpha_separate(b["alpha_hist"], title_prefix="Analysis alpha")

    elif res["status"] == "deadlock":
        print("The network is globally deadlocked.")
        print("Actual P_eff =", res["actual_P_eff"])

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

    net.add_edge(Edge(id=0, source=0, target=1, U_p=3, W_p=5, A_p=10))
    net.add_edge(Edge(id=1, source=1, target=2, U_p=5, W_p=7, A_p=0))
    net.add_edge(Edge(id=2, source=2, target=0, U_p=7, W_p=3, A_p=8))

    print(net)
    print("SCCs (topo):", net.sccs_in_topo_order())

    analysis_store_history = not run_sim

    res = net.analyze_execution(
        verbose=verbose,
        store_history=analysis_store_history,
    )

    print("\nFinal status:", res["status"])
    print("Blocked nodes:", res.get("blocked_nodes", set()))
    print("Global deadlock:", res.get("global_deadlock", False))

    if res["status"] == "unbounded":
        if res.get("reason") == "internal_scc_overflow":
            print("Unboundedness reason: internal SCC overflow")
            print("Details:", res["unboundedness"]["details"])
        elif res.get("reason") == "inter_scc_overflow":
            print("Unboundedness reason: inter-SCC positive drift")
            print("Details:", res["inter_scc_unboundedness"]["details"])

    elif res["status"] == "bounded":
        b = res["bounded_analysis"]
        print("H =", b["H"])
        print("t0 =", b["t0"])
        print("T_sys =", b["T_sys"])
        print("Exact queue bounds M_p =", b["M_p"])
        print("Actual P_eff =", res["actual_P_eff"])
        if plot_analysis and not run_sim:
            plot_queues_separate(b["q_hist"], net.edges, title_prefix="Analysis queue")
            plot_alpha_separate(b["alpha_hist"], title_prefix="Analysis alpha")

    elif res["status"] == "deadlock":
        print("The network is globally deadlocked.")
        print("Actual P_eff =", res["actual_P_eff"])

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