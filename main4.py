from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
from math import gcd
from functools import reduce
from fractions import Fraction
import matplotlib.pyplot as plt

# ------------------ Data model ------------------

@dataclass
class Node:
    id: int
    period: int
    # Optional: base used in alpha denominator (p_i). If None, defaults to period.
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
        self.edges[edge.id] = edge
        # register incoming for the target
        self.nodes[edge.target].sigma.append((edge.source, edge.id))

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
        return sccs  # topo order

    # ------------ ILP feasibility for SCC inequalities ------------

    def _scc_ilp_feasible(self, comp: List[int]) -> bool:
        """
        Deadlock certificate for one SCC:
          ∃ nonnegative integers x(j) such that for every node j in the SCC,
          at least one internal incoming edge p:i->j satisfies
             floor((A_p + U_p * x(i)) / W_p) <= x(j)
          ⇔  A_p + U_p * x(i) <= W_p * x(j) + (W_p - 1).

        We encode the OR per node with binaries: one y_e per internal edge.
        """
        try:
            import pulp
        except Exception as e:
            raise RuntimeError("PuLP is required for deadlock check: pip install pulp") from e

        comp_set = set(comp)
        prob = pulp.LpProblem("scc_deadlock_or", pulp.LpMinimize)

        # Integer counters for nodes in this SCC
        x = {j: pulp.LpVariable(f"x_{j}", lowBound=0, cat=pulp.LpInteger) for j in comp}

        # Collect internal incoming edges per node, create selector binaries
        incoming_internal: Dict[int, List[Edge]] = {j: [] for j in comp}
        y: Dict[int, pulp.LpVariable] = {}
        for e in self.edges.values():
            if e.source in comp_set and e.target in comp_set:
                incoming_internal[e.target].append(e)
                y[e.id] = pulp.LpVariable(f"y_{e.id}", lowBound=0, upBound=1, cat=pulp.LpBinary)

        # If any node has no internal incoming edge, this SCC can't be internally deadlocked
        # (it could still be blocked via an upstream blocked node — handled in detect_deadlock()).
        for j, es in incoming_internal.items():
            if not es:
                return False

        # Each node must select at least one internal incoming edge that blocks it
        for j, es in incoming_internal.items():
            prob += pulp.lpSum(y[e.id] for e in es) >= 1

        # Activate blocking inequality only for selected edges (Big-M deactivation otherwise)
        # floor((A + U x_i)/W) <= x_j  <=>  A + U x_i <= W x_j + (W - 1)
        M = 10 ** 6  # generic big-M; tighten if you add variable bounds
        for j, es in incoming_internal.items():
            for e in es:
                prob += (
                        e.A_p + e.U_p * x[e.source]
                        <= e.W_p * x[j] + (e.W_p - 1) + M * (1 - y[e.id])
                )

        # Small objective to keep values bounded
        prob += pulp.lpSum(x[j] for j in comp)

        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        return pulp.LpStatus[prob.status] == "Optimal"

    # ------------ Deadlock detection ------------

    def detect_deadlock(self) -> Set[int]:
        blocked: Set[int] = set()
        sccs_ordered = self.sccs_in_topo_order()

        def depends_on_blocked(comp: List[int]) -> bool:
            comp_set = set(comp)
            for e in self.edges.values():
                if e.target in comp_set and e.source not in comp_set and e.source in blocked:
                    return True
            return False

        for comp in sccs_ordered:
            if depends_on_blocked(comp):
                blocked.update(comp)
                continue
            if self._scc_ilp_feasible(comp):
                blocked.update(comp)
        return blocked

    # ---------- Buffer overflow in SCC cycles ----------

    def _pair_to_eid(self) -> Dict[Tuple[int, int], int]:
        """Map (u,v) -> edge id for a unique directed edge u->v."""
        m: Dict[Tuple[int, int], int] = {}
        for e in self.edges.values():
            key = (e.source, e.target)
            m[key] = e.id
        return m

    def _cycles_in_component(self, comp: List[int]) -> List[List[int]]:
        """
        Enumerate simple directed cycles as node lists [n0, n1, ..., nk-1, n0]
        within the given SCC 'comp'. Uses a DFS 'min-start' trick to avoid duplicates.
        """
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
                    # ensure each cycle is reported once (start at min node)
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

    def detect_buffer_overflow_loops(self) -> List[Dict]:
        """
        Find loops inside SCCs with g = prod(U_p)/prod(W_p) > 1.
        Returns a list of dicts:
          {
            'scc': [nodes...],
            'cycle_nodes': [n0, n1, ..., n0],
            'edge_ids': [e0, e1, ..., ek-1],
            'g_num': int, 'g_den': int, 'g': float
          }
        """
        results: List[Dict] = []
        pair2eid = self._pair_to_eid()

        for comp in self.sccs_in_topo_order():
            for cyc in self._cycles_in_component(comp):
                # Build edge list along the cycle
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

                # Compute g exactly as a rational
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
                        "g": float(num/den),
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

    def incoming_edges(self, j: int):
        for (_src, eid) in self.nodes[j].sigma:
            yield self.edges[eid]

    # x(j,0) base case (your spec)
    def x0(self, j: int) -> int:
        if len(self.nodes[j].sigma) == 0:
            return 1
        m = min(e.A_p // e.W_p for e in self.incoming_edges(j))
        return min(1, m)

    # hyperperiod
    def hyperperiod(self) -> int:
        return self._lcm_list([n.period for n in self.nodes.values()])

    # sample-and-hold x(j,t) per your "otherwise" clause
    def _get_x(self, x_hist: Dict[int, Dict[int, int]], j: int, t: int) -> int:
        if t < 0:
            return 0
        Pj = self.nodes[j].period
        tb = self._last_boundary_leq(t, Pj)
        return x_hist[j].get(tb, 0)

    # update node j at boundary time t (t % P_j == 0) using your recurrence
    def _update_at_boundary(self, x_hist: Dict[int, Dict[int, int]], j: int, t: int):
        Pj = self.nodes[j].period

        if t == 0:
            x_hist[j][0] = self.x0(j)
            return

        prev_xj = self._get_x(x_hist, j, t - Pj)
        pacing_term = prev_xj + 1

        if len(self.nodes[j].sigma) == 0:
            val = pacing_term
        else:
            # token term: min_{(i,p) in Sigma_j} floor( (A_p + U_p * x(i, t - P_i)) / W_p )
            token_terms = []
            for e in self.incoming_edges(j):
                Pi = self.nodes[e.source].period
                xi_term = self._get_x(x_hist, e.source, t - Pi)  # NOTE: uses t - P_i
                token_terms.append((e.A_p + e.U_p * xi_term) // e.W_p)
            token_term = min(token_terms)
            val = min(pacing_term, token_term)

        x_hist[j][t] = max(x_hist[j].get(t, 0), val)

    # compute all queues at time t:
    # q_p(t) = A_p + U_p * x(i, t - P_i) - W_p * x(j, t)
    def _compute_queues(self, x_hist: Dict[int, Dict[int, int]], t: int,
                        q_hist: Dict[int, Dict[int, int]]):
        for e in self.edges.values():
            Pi = self.nodes[e.source].period
            xi_term = self._get_x(x_hist, e.source, t - Pi)
            xj_term = self._get_x(x_hist, e.target, t)
            q_val = e.A_p + e.U_p * xi_term - e.W_p * xj_term
            q_hist[e.id][t] = q_val

    # compute alphas for all nodes at time t and record them
    def _compute_alphas(self, x_hist: Dict[int, Dict[int, int]], t: int,
                        alpha_hist: Dict[int, Dict[int, float]]):
        for j, node in self.nodes.items():
            base = node.alpha_base or node.period
            denom = max(1, (t // base)+1)
            alpha_val = self._get_x(x_hist, j, t) / denom
            alpha_hist[j][t] = alpha_val

    # simulate one hyperperiod (t0 -> t0+H], updating x, q, alpha at every boundary event
    def simulate_one_hyperperiod(self, x_hist: Dict[int, Dict[int, int]],
                                 q_hist: Dict[int, Dict[int, int]],
                                 alpha_hist: Dict[int, Dict[int, float]],
                                 t0: int, H: int):
        # record at t0
        self._compute_queues(x_hist, t0, q_hist)
        self._compute_alphas(x_hist, t0, alpha_hist)

        # build all boundary events in (t0, t0+H]
        events: List[Tuple[int, int]] = []
        for j, node in self.nodes.items():
            Pj = node.period
            t = ((t0 // Pj) + 1) * Pj  # next multiple of Pj > t0
            while t <= t0 + H:
                events.append((t, j))
                t += Pj
        events.sort()

        for t, j in events:
            self._update_at_boundary(x_hist, j, t)
            self._compute_queues(x_hist, t, q_hist)
            self._compute_alphas(x_hist, t, alpha_hist)

    # alpha(i,t) at a specific time (used for convergence at kH)
    def alpha_at(self, x_hist: Dict[int, Dict[int, int]], i: int, t: int) -> float:
        base = self.nodes[i].alpha_base or self.nodes[i].period
        denom = max(1, t // base)  # avoid divide-by-zero when t < base
        return self._get_x(x_hist, i, t) / denom

    # main loop: deadlock -> buffer overflow -> simulate hyperperiods -> alpha convergence
    def simulate_until_converged_alpha(self, eps: float = 1e-9, max_hyper: int = 200,
                                       verbose: bool = False):
        # 0) deadlock first
        blocked = self.detect_deadlock()
        if blocked:
            return {"status": "deadlock", "blocked": blocked}

        # 0.5) buffer-overflow check (SCC loops)
        overflow_loops = self.detect_buffer_overflow_loops()
        if verbose:
            if overflow_loops:
                print("Loops with g > 1 (potential buffer overflow):")
                for k, info in enumerate(overflow_loops, 1):
                    cyc = " -> ".join(map(str, info["cycle_nodes"]))
                    eids = ", ".join(f"e{eid}" for eid in info["edge_ids"])
                    print(f"  [{k}] {cyc} via [{eids}]   g = {info['g_num']}/{info['g_den']} = {info['g']:.6f}")
            else:
                print("No loops with g > 1 found in SCCs.")

        # 1) init histories
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

        # not converged within max_hyper
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

# ---------------- Plotting helpers ----------------

def plot_queues_separate(q_hist: Dict[int, Dict[int, int]], edges: Dict[int, Edge], title_prefix: str = "Queue"):
    """One figure per edge queue."""
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

def plot_alpha_separate(alpha_hist: Dict[int, Dict[int, float]]):
    """One figure per node's alpha(t)."""
    for nid, hist in alpha_hist.items():
        if not hist:
            continue
        times = sorted(hist.keys())
        vals = [hist[t] for t in times]
        plt.figure(figsize=(8, 4))
        plt.step(times, vals, where="post")
        plt.xlabel("time")
        plt.ylabel(f"alpha_{nid}(t)")
        plt.title(f"Alpha over time for node {nid}")
        plt.grid(True, which="both", linestyle=":")
        plt.tight_layout()
        plt.show()

# ------------------ Example ------------------

def example():
    net = MimosNetwork()
    # nodes: id, period, (optional alpha_base p_i)
    net.add_node(Node(id=0, period=3))        # alpha_base defaults to 3
    net.add_node(Node(id=1, period=5))
    net.add_node(Node(id=2, period=7))

    net.add_node(Node(id=3, period=3))
    net.add_node(Node(id=4, period=7))

    # edges: id, source, target, A_p, U_p, W_p
    net.add_edge(Edge(id=0, source=0, target=1, U_p=3, W_p=5, A_p=0))
    net.add_edge(Edge(id=1, source=1, target=2, U_p=5, W_p=7, A_p=0))
    net.add_edge(Edge(id=2, source=2, target=0, U_p=7, W_p=3, A_p=13))

    net.add_edge(Edge(id=3, source=3, target=1, U_p=3, W_p=5, A_p=0))
    net.add_edge(Edge(id=4, source=1, target=4, U_p=5, W_p=7, A_p=0))
    net.add_edge(Edge(id=6, source=4, target=3, U_p=7, W_p=3, A_p=16))

    print(net)
    print("SCCs (topo):", net.sccs_in_topo_order())

    res = net.simulate_until_converged_alpha(eps=1e-3, max_hyper=200, verbose=True)
    print("Status:", res["status"])
    if res["status"] != "deadlock":
        print("H =", res["H"])
        print("hyperperiods simulated:", res["hyperperiods"])
        print("last alpha@kH:", res["alphas_at_H"][-1])

        # separate figures
        plot_queues_separate(res["q_hist"], net.edges)
        plot_alpha_separate(res["alpha_hist"])

def example1():
    net = MimosNetwork()
    # nodes: id, period, (optional alpha_base p_i)
    net.add_node(Node(id=0, period=3))        # alpha_base defaults to 3
    net.add_node(Node(id=1, period=5))
    net.add_node(Node(id=2, period=7))

    # edges: id, source, target, A_p, U_p, W_p
    net.add_edge(Edge(id=0, source=0, target=1, U_p=3, W_p=5, A_p=8))
    net.add_edge(Edge(id=1, source=1, target=2, U_p=5, W_p=7, A_p=0))
    net.add_edge(Edge(id=2, source=2, target=0, U_p=7, W_p=3, A_p=8))

    print(net)
    print("SCCs (topo):", net.sccs_in_topo_order())

    res = net.simulate_until_converged_alpha(eps=1e-3, max_hyper=200, verbose=True)
    print("Status:", res["status"])
    if res["status"] != "deadlock":
        print("H =", res["H"])
        print("hyperperiods simulated:", res["hyperperiods"])
        print("last alpha@kH:", res["alphas_at_H"][-1])

        # separate figures
        plot_queues_separate(res["q_hist"], net.edges)
        plot_alpha_separate(res["alpha_hist"])

if __name__ == "__main__":
    # example()     # g = 1 cycles (likely prints "No loops with g > 1")
    example1()      # has a loop with g = 140/105 > 1
