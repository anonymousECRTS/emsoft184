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

    # ------------ Kosaraju SCC (topo order) ------------

    def sccs_in_topo_order(self) -> List[List[int]]:
        adj = self._adj_out()
        adjT = self._adj_in()
        visited = {v: False for v in self.nodes}
        finish_stack: List[int] = []

        def dfs1(u: int):
            visited[u] = True
            for v in adj[u]:
                if not visited[v]:
                    dfs1(v)
            finish_stack.append(u)

        for u in self.nodes:
            if not visited[u]:
                dfs1(u)

        visited = {v: False for v in self.nodes}
        sccs: List[List[int]] = []

        def dfs2(u: int, comp: List[int]):
            visited[u] = True
            comp.append(u)
            for v in adjT[u]:
                if not visited[v]:
                    dfs2(v, comp)

        while finish_stack:
            u = finish_stack.pop()
            if not visited[u]:
                comp: List[int] = []
                dfs2(u, comp)
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
        """
        Decide whether the SCC inequality system is feasible:
          x(j) >= min_{internal incoming (i,p)} ceil((A_p - W_p + 1 + U_p x(i))/W_p)

        Using:
          floor((A_p + U_p x(i))/W_p) <= x(j)
          <=> A_p + U_p x(i) <= W_p x(j) + (W_p - 1)
        """
        try:
            import pulp
        except Exception as e:
            raise RuntimeError("PuLP is required for blocked-node ILP: pip install pulp") from e

        comp_set = set(comp)

        if self.is_trivial_scc(comp):
            return False

        prob = pulp.LpProblem("scc_blocked", pulp.LpMinimize)

        x = {j: pulp.LpVariable(f"x_{j}", lowBound=0, cat=pulp.LpInteger) for j in comp}

        incoming_internal: Dict[int, List[Edge]] = {j: [] for j in comp}
        y: Dict[int, pulp.LpVariable] = {}

        for e in self.edges.values():
            if e.source in comp_set and e.target in comp_set:
                incoming_internal[e.target].append(e)
                y[e.id] = pulp.LpVariable(f"y_{e.id}", lowBound=0, upBound=1, cat=pulp.LpBinary)

        for j in comp:
            if not incoming_internal[j]:
                return False
            prob += pulp.lpSum(y[e.id] for e in incoming_internal[j]) >= 1

        M = 10 ** 6
        for j in comp:
            for e in incoming_internal[j]:
                prob += (
                    e.A_p + e.U_p * x[e.source]
                    <= e.W_p * x[j] + (e.W_p - 1) + M * (1 - y[e.id])
                )

        prob += pulp.lpSum(x[j] for j in comp)

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        return pulp.LpStatus[prob.status] == "Optimal"

    # ------------ Blocked-node detection ------------

    def x0(self, j: int) -> int:
        """
        Initial completed firings at t = 0.

        Semantics:
        - Nodes with no incoming edges start with one initial completed firing.
        - Nodes with incoming edges may fire initially only if enough
          initial tokens are available on every required incoming queue.
        """
        incoming = list(self.incoming_edges(j))
        if not incoming:
            return 1

        m = min(e.A_p // e.W_p for e in incoming)
        return min(1, m)

    def _trivial_scc_blocked(self, comp: List[int], blocked: Set[int]) -> bool:
        """
        A trivial SCC (single node without self-loop) is blocked if:
        1) it has an incoming edge from an already-blocked predecessor, or
        2) it has no external predecessor at all and cannot fire initially.
        """
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

    # ---------------- Simulation utilities for analysis approach ----------------

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

    def _compute_queues(
        self,
        x_hist: Dict[int, Dict[int, int]],
        t: int,
        q_hist: Dict[int, Dict[int, int]],
        allowed_edges: Optional[Set[int]] = None,
    ):
        for e in self.edges.values():
            if allowed_edges is not None and e.id not in allowed_edges:
                continue
            Pi = self.nodes[e.source].period
            xi_term = self._get_x(x_hist, e.source, t - Pi)
            xj_term = self._get_x(x_hist, e.target, t)
            q_val = e.A_p + e.U_p * xi_term - e.W_p * xj_term
            q_hist[e.id][t] = q_val

    def _compute_alphas(
        self,
        x_hist: Dict[int, Dict[int, int]],
        t: int,
        alpha_hist: Dict[int, Dict[int, float]],
        allowed_nodes: Optional[Set[int]] = None,
    ):
        for j, node in self.nodes.items():
            if allowed_nodes is not None and j not in allowed_nodes:
                continue
            base = node.alpha_base or node.period
            denom = max(1, (t // base) + 1)
            alpha_val = self._get_x(x_hist, j, t) / denom
            alpha_hist[j][t] = alpha_val

    def simulate_one_hyperperiod(
        self,
        x_hist: Dict[int, Dict[int, int]],
        q_hist: Dict[int, Dict[int, int]],
        alpha_hist: Dict[int, Dict[int, float]],
        t0: int,
        H: int,
        allowed_nodes: Optional[Set[int]] = None,
        allowed_edges: Optional[Set[int]] = None,
    ):
        self._compute_queues(x_hist, t0, q_hist, allowed_edges=allowed_edges)
        self._compute_alphas(x_hist, t0, alpha_hist, allowed_nodes=allowed_nodes)

        active_nodes = list(self.nodes.keys()) if allowed_nodes is None else sorted(allowed_nodes)

        events: List[Tuple[int, int]] = []
        for j in active_nodes:
            Pj = self.nodes[j].period
            t = ((t0 // Pj) + 1) * Pj
            while t <= t0 + H:
                events.append((t, j))
                t += Pj
        events.sort()

        for t, j in events:
            self._update_at_boundary(
                x_hist,
                j,
                t,
                allowed_nodes=allowed_nodes,
                allowed_edges=allowed_edges,
            )
            self._compute_queues(x_hist, t, q_hist, allowed_edges=allowed_edges)
            self._compute_alphas(x_hist, t, alpha_hist, allowed_nodes=allowed_nodes)

    def alpha_at(self, x_hist: Dict[int, Dict[int, int]], i: int, t: int) -> float:
        base = self.nodes[i].alpha_base or self.nodes[i].period
        denom = max(1, (t // base) + 1)
        return self._get_x(x_hist, i, t) / denom

    # ---------------- Effective period computation ----------------

    def isolated_scc_effective_periods(self, comp: List[int], max_hyper: int = 200) -> Dict[int, float]:
        comp_set = set(comp)
        internal_edges = {
            e.id for e in self.edges.values()
            if e.source in comp_set and e.target in comp_set
        }

        H_i = self.hyperperiod(comp_set)

        x_hist: Dict[int, Dict[int, int]] = {j: {0: self.x0(j)} for j in comp}
        q_hist: Dict[int, Dict[int, int]] = {eid: {} for eid in internal_edges}
        alpha_hist: Dict[int, Dict[int, float]] = {j: {} for j in comp}

        sampled_states: Dict[Tuple[int, ...], int] = {}

        def state_at_boundary(t: int) -> Tuple[int, ...]:
            vals = []
            for eid in sorted(internal_edges):
                e = self.edges[eid]
                Pi = self.nodes[e.source].period
                xi_term = self._get_x(x_hist, e.source, t - Pi)
                xj_term = self._get_x(x_hist, e.target, t)
                vals.append(e.A_p + e.U_p * xi_term - e.W_p * xj_term)
            return tuple(vals)

        for m in range(max_hyper + 1):
            t = m * H_i
            s = state_at_boundary(t)
            if s in sampled_states:
                m1 = sampled_states[s]
                m2 = m
                T_hat = (m2 - m1) * H_i
                t_start = m1 * H_i

                peff: Dict[int, float] = {}
                for j in comp:
                    activations = 0
                    period_j = self.nodes[j].period
                    for tb in sorted(x_hist[j].keys()):
                        if t_start < tb <= t_start + T_hat:
                            prev_t = tb - period_j
                            prev = self._get_x(x_hist, j, prev_t)
                            curr = x_hist[j][tb]
                            activations += max(0, curr - prev)
                    peff[j] = float("inf") if activations == 0 else T_hat / activations
                return peff

            sampled_states[s] = m

            if m < max_hyper:
                self.simulate_one_hyperperiod(
                    x_hist, q_hist, alpha_hist, t, H_i,
                    allowed_nodes=comp_set,
                    allowed_edges=internal_edges,
                )

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

        overflow_loops = self.detect_buffer_overflow_loops(restrict_to_nodes=nonblocked_nodes)
        if overflow_loops:
            return {
                "status": "unbounded",
                "reason": "cycle_gain",
                "details": overflow_loops,
            }

        P_eff = self.actual_effective_periods(blocked, max_hyper_scc=max_hyper_scc)

        sccs, scc_idx = self.scc_index_map()
        drifts = []

        for e in self.edges.values():
            if e.source in blocked or e.target in blocked:
                continue
            if scc_idx[e.source] == scc_idx[e.target]:
                continue

            delta = e.U_p / P_eff[e.source] - e.W_p / P_eff[e.target]
            info = {
                "edge_id": e.id,
                "source": e.source,
                "target": e.target,
                "delta": delta,
            }
            drifts.append(info)

            if delta > 0:
                return {
                    "status": "unbounded",
                    "reason": "positive_drift",
                    "details": {
                        **info,
                        "P_eff": P_eff,
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
    ) -> Dict:
        H = self.hyperperiod()

        x_hist: Dict[int, Dict[int, int]] = {j: {0: self.x0(j)} for j in self.nodes}
        q_hist: Dict[int, Dict[int, int]] = {eid: {} for eid in self.edges}
        alpha_hist: Dict[int, Dict[int, float]] = {j: {} for j in self.nodes}

        sampled_states: Dict[Tuple[int, ...], int] = {}

        def sampled_state_at(t: int) -> Tuple[int, ...]:
            vals = []
            for eid in sorted(self.edges.keys()):
                e = self.edges[eid]
                Pi = self.nodes[e.source].period
                xi_term = self._get_x(x_hist, e.source, t - Pi)
                xj_term = self._get_x(x_hist, e.target, t)
                vals.append(e.A_p + e.U_p * xi_term - e.W_p * xj_term)
            return tuple(vals)

        for m in range(max_hyper + 1):
            t = m * H
            state = sampled_state_at(t)

            if verbose:
                print(f"[m={m}] q({t}) = {state}")

            if state in sampled_states:
                m1 = sampled_states[state]
                m2 = m
                t0 = m1 * H
                T_sys = (m2 - m1) * H

                M_p: Dict[int, int] = {}
                for eid in self.edges:
                    vals = [qv for tt, qv in q_hist[eid].items() if 0 <= tt < t0 + T_sys]
                    if not vals:
                        idx = list(sorted(self.edges.keys())).index(eid)
                        vals = [state[idx]]
                    M_p[eid] = max(vals)

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

            if m < max_hyper:
                self.simulate_one_hyperperiod(x_hist, q_hist, alpha_hist, t, H)

        return {
            "status": "max_iter_reached",
            "H": H,
            "x_hist": x_hist,
            "q_hist": q_hist,
            "alpha_hist": alpha_hist,
        }

    # ---------------- Alpha simulation ----------------

    def simulate_until_converged_alpha(
        self,
        eps: float = 1e-9,
        max_hyper: int = 200,
        verbose: bool = False,
    ):
        blocked = self.detect_blocked_nodes()
        if blocked:
            return {"status": "deadlock", "blocked": blocked}

        overflow_loops = self.detect_buffer_overflow_loops()
        if verbose:
            if overflow_loops:
                print("Loops with g > 1:")
                for k, info in enumerate(overflow_loops, 1):
                    cyc = " -> ".join(map(str, info["cycle_nodes"]))
                    eids = ", ".join(f"e{eid}" for eid in info["edge_ids"])
                    print(f"  [{k}] {cyc} via [{eids}]   g = {info['g_num']}/{info['g_den']} = {info['g']:.6f}")
            else:
                print("No loops with g > 1 found in SCCs.")

        x_hist: Dict[int, Dict[int, int]] = {j: {0: self.x0(j)} for j in self.nodes}
        q_hist: Dict[int, Dict[int, int]] = {eid: {} for eid in self.edges}
        alpha_hist: Dict[int, Dict[int, float]] = {j: {} for j in self.nodes}

        H = self.hyperperiod()
        alphas_at_H: List[Dict[int, float]] = []

        for k in range(1, max_hyper + 1):
            t0, t1 = (k - 1) * H, k * H
            self.simulate_one_hyperperiod(x_hist, q_hist, alpha_hist, t0, H)

            alpha_k = {j: self.alpha_at(x_hist, j, t1) for j in self.nodes}
            alphas_at_H.append(alpha_k)
            if verbose:
                print(f"[k={k}] alpha@{t1} =", alpha_k)

            if k >= 2:
                prev = alphas_at_H[-2]
                diff = max(abs(alpha_k[j] - prev[j]) for j in self.nodes)
                if verbose:
                    print(f"   max|Δalpha| = {diff:.3e}")
                if diff < eps:
                    return {
                        "status": "converged",
                        "hyperperiods": k,
                        "H": H,
                        "alphas_at_H": alphas_at_H,
                        "alpha_hist": alpha_hist,
                        "x_hist": x_hist,
                        "q_hist": q_hist,
                        "eps": eps,
                        "max_diff": diff,
                        "overflow_loops": overflow_loops,
                    }

        max_diff = None
        if len(alphas_at_H) > 1:
            prev, last = alphas_at_H[-2], alphas_at_H[-1]
            max_diff = max(abs(last[j] - prev[j]) for j in self.nodes)

        return {
            "status": "max_iter_reached",
            "hyperperiods": max_hyper,
            "H": H,
            "alphas_at_H": alphas_at_H,
            "alpha_hist": alpha_hist,
            "x_hist": x_hist,
            "q_hist": q_hist,
            "eps": eps,
            "max_diff": max_diff,
            "overflow_loops": overflow_loops,
        }

    # ---------------- Full analysis pipeline ----------------

    def analyze_execution(
        self,
        max_hyper_bounded: int = 500,
        max_hyper_scc: int = 200,
        verbose: bool = False,
    ) -> Dict:
        t0 = time.perf_counter()
        blocked = self.detect_blocked_nodes()
        t1 = time.perf_counter()
        print(f"ILP time: {t1 - t0:.6f} s")

        result: Dict = {
            "blocked_nodes": blocked,
            "global_deadlock": (len(blocked) == len(self.nodes)),
        }

        if result["global_deadlock"]:
            result["status"] = "deadlock"
            return result

        t0 = time.perf_counter()
        ub = self.detect_unbounded_queue_growth(
            blocked=blocked,
            max_hyper_scc=max_hyper_scc,
        )
        result["unboundedness"] = ub
        t1 = time.perf_counter()
        print(f"undoundedness time: {t1 - t0:.6f} s")

        if ub["status"] == "unbounded":
            result["status"] = "unbounded"
            return result

        t0 = time.perf_counter()
        bounded = self.exact_queue_bounds_for_bounded_execution(
            max_hyper=max_hyper_bounded,
            verbose=verbose,
        )
        t1 = time.perf_counter()
        print(f"exact queue bounds time: {t1 - t0:.6f} s")
        result["bounded_analysis"] = bounded
        result["status"] = bounded["status"]
        return result

    # ---------------- Independent event-driven simulator ----------------

    def _incoming_edge_ids(self, j: int, allowed_edges: Optional[Set[int]] = None) -> List[int]:
        eids = []
        for _src, eid in self.nodes[j].sigma:
            if allowed_edges is None or eid in allowed_edges:
                eids.append(eid)
        return eids

    def _outgoing_edge_ids(self, i: int, allowed_edges: Optional[Set[int]] = None) -> List[int]:
        eids = []
        for e in self.edges.values():
            if e.source == i and (allowed_edges is None or e.id in allowed_edges):
                eids.append(e.id)
        return eids

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

    def simulate_event_driven(
        self,
        max_hyper: int = 200,
        stop_on_repeat: bool = True,
        stop_on_deadlock: bool = True,
        allowed_nodes: Optional[Set[int]] = None,
        allowed_edges: Optional[Set[int]] = None,
        verbose: bool = False,
    ) -> Dict:
        """
        Independent event-driven simulator with write-before-read semantics.

        Semantics:
        - A firing started/released at time t consumes its inputs at t.
        - Its outputs become visible at time t + P_j.
        - If writes and reads coincide at time t, writes are applied first.
        """
        active_nodes = set(self.nodes.keys()) if allowed_nodes is None else set(allowed_nodes)
        active_edges = set(self.edges.keys()) if allowed_edges is None else set(allowed_edges)

        H = self.hyperperiod(active_nodes)

        q: Dict[int, int] = {eid: self.edges[eid].A_p for eid in active_edges}
        firings: Dict[int, int] = {j: 0 for j in active_nodes}
        pending_writes: Dict[int, List[Tuple[int, int]]] = {}

        q_hist: Dict[int, Dict[int, int]] = {eid: {0: q[eid]} for eid in active_edges}
        fire_hist: Dict[int, Dict[int, int]] = {j: {0: 0} for j in active_nodes}
        alpha_hist: Dict[int, Dict[int, float]] = {j: {} for j in active_nodes}
        sampled_seen: Dict[Tuple[int, ...], int] = {}

        def record_time(t: int):
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
            for eid in self._outgoing_edge_ids(j, active_edges):
                pending_writes[complete_t].append((eid, self.edges[eid].U_p))

        def is_deadlocked_after_time(t: int) -> bool:
            released = [j for j in sorted(active_nodes) if t % self.nodes[j].period == 0]
            for j in released:
                incoming = self._incoming_edge_ids(j, active_edges)
                if not incoming:
                    return False
                if all(q[eid] >= self.edges[eid].W_p for eid in incoming):
                    return False
            return True

        initial_fired_nodes: List[int] = []
        for j in sorted(active_nodes):
            incoming = self._incoming_edge_ids(j, active_edges)
            if not incoming:
                initial_fired_nodes.append(j)
            else:
                if all(q[eid] >= self.edges[eid].W_p for eid in incoming):
                    initial_fired_nodes.append(j)

        for j in initial_fired_nodes:
            for eid in self._incoming_edge_ids(j, active_edges):
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
                incoming = self._incoming_edge_ids(j, active_edges)
                if not incoming:
                    will_fire.append(j)
                else:
                    if all(q[eid] >= self.edges[eid].W_p for eid in incoming):
                        will_fire.append(j)

            for j in will_fire:
                for eid in self._incoming_edge_ids(j, active_edges):
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
                        m2 = m
                        return {
                            "status": "repeat_detected",
                            "H": H,
                            "m1": m1,
                            "m2": m2,
                            "t0": m1 * H,
                            "T_sys": (m2 - m1) * H,
                            "q_hist": q_hist,
                            "fire_hist": fire_hist,
                            "alpha_hist": alpha_hist,
                            "final_queues": dict(q),
                            "final_firings": dict(firings),
                        }
                    sampled_seen[state] = m

                if stop_on_deadlock and is_deadlocked_after_time(t):
                    return {
                        "status": "deadlock_detected",
                        "H": H,
                        "t_deadlock": t,
                        "q_hist": q_hist,
                        "fire_hist": fire_hist,
                        "alpha_hist": alpha_hist,
                        "final_queues": dict(q),
                        "final_firings": dict(firings),
                    }

        return {
            "status": "max_iter_reached",
            "H": H,
            "q_hist": q_hist,
            "fire_hist": fire_hist,
            "alpha_hist": alpha_hist,
            "final_queues": dict(q),
            "final_firings": dict(firings),
        }

    def queue_bounds_from_event_simulation(self, sim_result: Dict) -> Dict[int, int]:
        if sim_result["status"] != "repeat_detected":
            raise ValueError("repeat_detected result required to compute exact bounds from simulation.")

        q_hist = sim_result["q_hist"]
        t0 = sim_result["t0"]
        T_sys = sim_result["T_sys"]

        M_p: Dict[int, int] = {}
        for eid, hist in q_hist.items():
            vals = [qv for tt, qv in hist.items() if 0 <= tt < t0 + T_sys]
            M_p[eid] = max(vals) if vals else 0
        return M_p


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


def plot_alpha_separate(alpha_hist: Dict[int, Dict[int, float]], title_prefix: str = "Alpha over time"):
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


# ---------------- Timing / comparison helper ----------------

def compare_analysis_with_event_simulation(net: MimosNetwork, max_hyper: int = 200, verbose: bool = True):
    print("Running analysis...")
    t0 = time.perf_counter()
    analysis = net.analyze_execution(max_hyper_bounded=max_hyper, verbose=verbose)
    t1 = time.perf_counter()
    analysis_time = t1 - t0

    print("\nRunning independent event-driven simulation...")
    t2 = time.perf_counter()
    sim = net.simulate_event_driven(max_hyper=max_hyper, verbose=verbose)
    t3 = time.perf_counter()
    simulation_time = t3 - t2

    print("\n--- Comparison ---")
    print("Analysis status:", analysis["status"])
    print("Simulation status:", sim["status"])
    print(f"Analysis execution time:   {analysis_time:.6f} s")
    print(f"Simulation execution time: {simulation_time:.6f} s")

    if analysis["status"] == "bounded" and sim["status"] == "repeat_detected":
        b = analysis["bounded_analysis"]
        sim_bounds = net.queue_bounds_from_event_simulation(sim)

        print("Analysis H       =", b["H"])
        print("Simulation H     =", sim["H"])
        print("Analysis t0      =", b["t0"])
        print("Simulation t0    =", sim["t0"])
        print("Analysis T_sys   =", b["T_sys"])
        print("Simulation T_sys =", sim["T_sys"])
        print("Analysis M_p     =", b["M_p"])
        print("Simulation M_p   =", sim_bounds)

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

    t0 = time.perf_counter()
    res = net.analyze_execution(verbose=verbose)
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
            plot_queues_separate(b["q_hist"], net.edges, title_prefix="Analysis queue")
            plot_alpha_separate(b["alpha_hist"], title_prefix="Analysis alpha")

    elif res["status"] == "deadlock":
        print("The network is globally deadlocked.")

    if run_sim:
        print("\n--- Running independent event-driven simulator ---")
        t2 = time.perf_counter()
        sim = net.simulate_event_driven(max_hyper=200, verbose=verbose)
        t3 = time.perf_counter()

        print("Simulation status:", sim["status"])
        print(f"Simulation execution time: {t3 - t2:.6f} s")

        if sim["status"] == "repeat_detected":
            print("Simulation H =", sim["H"])
            print("Simulation t0 =", sim["t0"])
            print("Simulation T_sys =", sim["T_sys"])
            sim_bounds = net.queue_bounds_from_event_simulation(sim)
            print("Simulation queue bounds M_p =", sim_bounds)

        elif sim["status"] == "deadlock_detected":
            print("Simulation deadlock detected at t =", sim["t_deadlock"])

        else:
            print("Simulation reached max iterations without repeat/deadlock.")


# ------------------ Examples ------------------

def example():
    net = MimosNetwork()

    net.add_node(Node(id=0, period=3))
    net.add_node(Node(id=1, period=5))
    net.add_node(Node(id=2, period=7))
    net.add_node(Node(id=3, period=3))
    net.add_node(Node(id=4, period=7))

    net.add_edge(Edge(id=0, source=0, target=1, U_p=3, W_p=5, A_p=0))
    net.add_edge(Edge(id=1, source=1, target=2, U_p=5, W_p=7, A_p=0))
    net.add_edge(Edge(id=2, source=2, target=0, U_p=7, W_p=3, A_p=13))

    net.add_edge(Edge(id=3, source=3, target=1, U_p=3, W_p=5, A_p=0))
    net.add_edge(Edge(id=4, source=1, target=4, U_p=5, W_p=7, A_p=0))
    net.add_edge(Edge(id=6, source=4, target=3, U_p=7, W_p=3, A_p=16))

    print(net)
    print("SCCs (topo):", net.sccs_in_topo_order())

    t0 = time.perf_counter()
    res = net.analyze_execution(verbose=True)
    t1 = time.perf_counter()

    print("Status:", res["status"])
    print("Blocked nodes:", res["blocked_nodes"])
    print(f"Analysis execution time: {t1 - t0:.6f} s")

    if res["status"] == "bounded":
        b = res["bounded_analysis"]
        print("H =", b["H"])
        print("t0 =", b["t0"])
        print("T_sys =", b["T_sys"])
        print("M_p =", b["M_p"])
        plot_queues_separate(b["q_hist"], net.edges, title_prefix="Analysis queue")
        plot_alpha_separate(b["alpha_hist"], title_prefix="Analysis alpha")

    print("\nNow compare with event-driven simulation:")
    compare_analysis_with_event_simulation(net, max_hyper=200, verbose=True)


def example1():
    net = MimosNetwork()

    net.add_node(Node(id=0, period=3))
    net.add_node(Node(id=1, period=5))
    net.add_node(Node(id=2, period=7))

    net.add_edge(Edge(id=0, source=0, target=1, U_p=3, W_p=5, A_p=10))
    net.add_edge(Edge(id=1, source=1, target=2, U_p=5, W_p=7, A_p=0))
    net.add_edge(Edge(id=2, source=2, target=0, U_p=7, W_p=3, A_p=8))

    print(net)
    print("SCCs (topo):", net.sccs_in_topo_order())

    t0 = time.perf_counter()
    res = net.analyze_execution(verbose=True)
    t1 = time.perf_counter()

    print("\nFinal status:", res["status"])
    print("Blocked nodes:", res["blocked_nodes"])
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
        plot_queues_separate(b["q_hist"], net.edges, title_prefix="Analysis queue")
        plot_alpha_separate(b["alpha_hist"], title_prefix="Analysis alpha")

    print("\nNow compare with event-driven simulation:")
    compare_analysis_with_event_simulation(net, max_hyper=200, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a MIMOS benchmark JSON file.")
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
        example1()