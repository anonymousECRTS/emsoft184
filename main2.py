from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from math import gcd
from functools import reduce
from fractions import Fraction
import bisect

# ------------------ helpers ------------------

def lcm(a: int, b: int) -> int:
    return abs(a*b) // gcd(a, b) if a and b else 0

def lcm_many(vals: List[int]) -> int:
    if not vals:
        return 0
    return reduce(lcm, vals)

# ------------------ data model ------------------

@dataclass
class Node:
    id: int
    period: int
    sigma: List[Tuple[int, int]] = field(default_factory=list)  # incoming (source_id, edge_id)

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
        self.edges[edge.id] = edge
        self.nodes[edge.target].sigma.append((edge.source, edge.id))

    def __str__(self):
        return f"MimosNetwork(nodes={list(self.nodes.keys())}, edges={list(self.edges.keys())})"

    # ---------- repetition vector (SDF balance U*q_i = W*q_j) ----------

    def repetition_vector(self) -> Optional[Dict[int, int]]:
        q_rat: Dict[int, Fraction] = {}
        seen = set()
        for start in self.nodes:
            if start in seen:
                continue
            q_rat[start] = Fraction(1, 1)
            stack = [start]
            seen.add(start)
            while stack:
                u = stack.pop()
                # outgoing: q_v = (U/W) q_u
                for e in self.edges.values():
                    if e.source == u:
                        v = e.target
                        implied = q_rat[u] * Fraction(e.U_p, e.W_p)
                        if v not in q_rat:
                            q_rat[v] = implied
                            stack.append(v)
                            seen.add(v)
                        elif q_rat[v] != implied:
                            return None
                # incoming: q_i = (W/U) q_u
                for e in self.edges.values():
                    if e.target == u:
                        i = e.source
                        implied = q_rat[u] * Fraction(e.W_p, e.U_p)
                        if i not in q_rat:
                            q_rat[i] = implied
                            stack.append(i)
                            seen.add(i)
                        elif q_rat[i] != implied:
                            return None
        # scale to integers
        den_lcm = 1
        for r in q_rat.values():
            den_lcm = lcm(den_lcm, r.denominator)
        q_int = {j: int(q_rat[j] * den_lcm) for j in q_rat}
        g = 0
        for v in q_int.values():
            g = gcd(g, v)
        if g > 1:
            q_int = {j: v // g for j, v in q_int.items()}
        return q_int

    # ---------- simulation with x(i, t - P_i) semantics ----------

    def hyperperiod(self) -> int:
        return lcm_many([n.period for n in self.nodes.values()])

    def rep_hyperperiod(self) -> Optional[int]:
        q = self.repetition_vector()
        if q is None:
            return None
        return lcm_many([q[j]*self.nodes[j].period for j in self.nodes])

    def simulate(self, T_end: int, record_series: bool = True):
        """
        Discrete-time simulation that matches:

          if t % P_j == 0:
             x(j,t) = min( x(j,t-P_j)+1 ,
                           min_{(i,p) in Σ_j} floor((A_p + U_p * x(i, t - P_i)) / W_p) )

        Queue length:
             q(p,t) = A_p + U_p * x(i, t - P_i) - W_p * x(j, t)

        x is piecewise-constant between node period boundaries.
        """
        # per-node histories at boundaries: times[j] sorted, vals[j] aligned
        times: Dict[int, List[int]] = {j: [0] for j in self.nodes}
        vals: Dict[int, List[int]]  = {j: [] for j in self.nodes}

        # x(j,0)
        for j, node in self.nodes.items():
            if not node.sigma:
                x0 = 1
            else:
                min_init = min((self.edges[eid].A_p // self.edges[eid].W_p) for (_, eid) in node.sigma)
                x0 = min(1, min_init)
            vals[j].append(x0)

        # fast lookup: x(j, τ) = value at floor(τ / P_j) * P_j
        def x_at(j: int, t: int) -> int:
            if t < 0:
                return 0
            Pj = self.nodes[j].period
            snap = (t // Pj) * Pj
            idx = bisect.bisect_right(times[j], snap) - 1
            return vals[j][idx]

        # queue series (optional)
        series_q: Dict[int, List[Tuple[int, int]]] = {e.id: [] for e in self.edges.values()}
        def q_at_edge(e: Edge, t: int) -> int:
            return e.A_p + e.U_p * x_at(e.source, t - self.nodes[e.source].period) - e.W_p * x_at(e.target, t)

        # initial queues
        if record_series:
            for e in self.edges.values():
                series_q[e.id].append((0, q_at_edge(e, 0)))

        # simulate
        for t in range(1, T_end + 1):
            # update nodes at their own period boundaries
            for j, node in self.nodes.items():
                if t % node.period == 0:
                    if not node.sigma:
                        new_x = (t // node.period) + 1
                    else:
                        # periodic opportunity = prev + 1
                        opp = x_at(j, t - node.period) + 1
                        # token bound uses PRODUCER history at (t - P_i)
                        token_bound = min(
                            (self.edges[eid].A_p + self.edges[eid].U_p * x_at(src, t - self.nodes[src].period)) //
                            self.edges[eid].W_p
                            for (src, eid) in node.sigma
                        )
                        new_x = min(opp, token_bound)
                    times[j].append(t)
                    vals[j].append(new_x)

            # queues
            if record_series:
                for e in self.edges.values():
                    series_q[e.id].append((t, q_at_edge(e, t)))

        # summaries
        H = self.hyperperiod()
        x_final = {j: vals[j][-1] for j in self.nodes}
        max_queue = {e.id: max(q for _, q in series_q[e.id]) for e in self.edges.values()} if record_series else {}
        last_queue = {e.id: series_q[e.id][-1][1] for e in self.edges.values()} if record_series else {}

        # activation rates & effective periods at T_end
        alpha, P_eff = {}, {}
        for j, node in self.nodes.items():
            opportunities = T_end // node.period  # ⌊t/P_j⌋
            if opportunities > 0:
                alpha[j] = x_final[j] / opportunities
                P_eff[j] = node.period / alpha[j] if alpha[j] > 0 else float("inf")
            else:
                alpha[j] = float("nan")
                P_eff[j] = float("inf")

        # simple drift flag (compare last two hyperperiods if available)
        drift = {e.id: False for e in self.edges.values()}
        if record_series and H > 0 and T_end >= 2 * H:
            for e in self.edges.values():
                s = series_q[e.id]
                seg1 = [q for (tt, q) in s if T_end - 2*H < tt <= T_end - H]
                seg2 = [q for (tt, q) in s if T_end -   H < tt <= T_end]
                if seg1 and seg2:
                    drift[e.id] = (sum(seg2)/len(seg2)) > (sum(seg1)/len(seg1)) + 1e-9

        return {
            "H": H,
            "x_final": x_final,
            "alpha": alpha,
            "P_eff": P_eff,
            "series_q": series_q if record_series else None,
            "max_queue": max_queue,
            "last_queue": last_queue,
            "drift": drift,
        }

# ------------------ example ------------------

def mimos_sim1():
    net = MimosNetwork()
    net.add_node(Node(id=0, period=3))
    net.add_node(Node(id=1, period=5))
    net.add_node(Node(id=2, period=7))

    # edge 2 has A_p = 14 per your test
    net.add_edge(Edge(id=0, source=0, target=1, U_p=3, W_p=5, A_p=0))
    net.add_edge(Edge(id=1, source=1, target=2, U_p=5, W_p=7, A_p=0))
    net.add_edge(Edge(id=2, source=2, target=0, U_p=7, W_p=3, A_p=14))

    H = net.hyperperiod()           # 105 for this example
    T_end = 200 * H                   # simulate two hyperperiods

    res = net.simulate(T_end=T_end, record_series=True)

    print("Hyperperiod:", res["H"])
    print("x_final:", res["x_final"])
    print("Activation rates alpha:", {k: round(v, 4) for k, v in res["alpha"].items()})
    print("Effective periods P_eff:", {k: round(v, 4) if v != float('inf') else v for k, v in res["P_eff"].items()})
    print("Max queue per edge:", res["max_queue"])
    print("Last queue per edge:", res["last_queue"])
    print("Growing drift:", res["drift"])

if __name__ == "__main__":
    mimos_sim1()
