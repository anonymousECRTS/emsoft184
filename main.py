from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set

# ------------------ Data model ------------------

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
        # register incoming for the target
        self.nodes[edge.target].sigma.append((edge.source, edge.id))

    def __str__(self):
        return f"MimosNetwork(nodes={list(self.nodes.keys())}, edges={list(self.edges.keys())})"

    # ------------ Graph helpers (adjacency of G and G^T) ------------

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

    # ------------ CLRS / Kosaraju SCC (returns SCCs in topo order) ------------

    def sccs_in_topo_order(self) -> List[List[int]]:
        """
        Kosaraju/CLRS SCC:
        1) DFS on G to get finishing order
        2) DFS on G^T in decreasing finishing time → SCCs
        The order of discovered SCCs is a topological order of the SCC-DAG.
        """
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

        # pop gives decreasing finish time automatically
        while finish_stack:
            u = finish_stack.pop()
            if not visited[u]:
                comp: List[int] = []
                dfs2(u, comp)
                sccs.append(comp)

        # In CLRS, this list is already a topological order (sources first)
        return sccs

    # ------------ ILP feasibility for SCC inequalities ------------

    def _scc_ilp_feasible(self, comp: List[int]) -> bool:
        """
        Feasibility of: W_p * x(j) >= A_p + U_p * x(i)   for all internal edges i->j in comp,
        with x(j) ∈ Z_{>=0}.  If feasible → SCC is blocked (per your theorem).
        """
        try:
            import pulp
        except Exception as e:
            raise RuntimeError("PuLP is required: pip install pulp") from e

        prob = pulp.LpProblem("scc_block_check", pulp.LpMinimize)
        x = {j: pulp.LpVariable(f"x_{j}", lowBound=0, cat=pulp.LpInteger) for j in comp}

        comp_set = set(comp)
        # constraints for each internal edge
        for e in self.edges.values():
            if e.source in comp_set and e.target in comp_set:
                prob += e.W_p * x[e.target] >= e.A_p + e.U_p * x[e.source] + 1 - e.W_p

        # any simple objective (not needed for feasibility) — minimize sum for determinism
        prob += pulp.lpSum(x[j] for j in comp)

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        return pulp.LpStatus[prob.status] == "Optimal"

    # ------------ Deadlock detection (CLRS SCC + ILP) ------------

    def detect_deadlock(self) -> Set[int]:
        """
        Returns the set of blocked node ids.
        Algorithm:
          - Get SCCs in topological order via Kosaraju/CLRS.
          - For each SCC in order:
              (i) if it has an incoming edge from any already-blocked node → mark entire SCC blocked
             (ii) else, solve ILP on internal edges; if feasible → mark SCC blocked
        """
        blocked: Set[int] = set()
        sccs_ordered = self.sccs_in_topo_order()

        def depends_on_blocked(comp: List[int]) -> bool:
            comp_set = set(comp)
            # any incoming edge from a blocked node outside the component?
            for e in self.edges.values():
                if e.target in comp_set and e.source not in comp_set:
                    if e.source in blocked:
                        return True
            return False

        for comp in sccs_ordered:
            if depends_on_blocked(comp):
                blocked.update(comp)
                continue
            # Check internal inequality system by ILP
            if self._scc_ilp_feasible(comp):
                blocked.update(comp)

        return blocked


def mimos_sim1():
    # Create network
    net = MimosNetwork()

    # Add nodes
    net.add_node(Node(id=0, period=3))
    net.add_node(Node(id=1, period=5))
    net.add_node(Node(id=2, period=7))

    # Add edges
    net.add_edge(Edge(id=0, source=0, target=1, U_p=3, W_p=5, A_p=0))
    net.add_edge(Edge(id=1, source=1, target=2, U_p=5, W_p=7, A_p=0))
    net.add_edge(Edge(id=2, source=2, target=0, U_p=7, W_p=3, A_p=14))

    print(net)
    # MimosNetwork(nodes=[0, 1, 2], edges=[0, 1, 2])

    # Check target sigma list
    print(net.nodes[0].sigma)  # [(2, 2)] because edge 2 is incoming to node 0
    blocked = net.detect_deadlock()
    print("Blocked nodes:", blocked)


if __name__ == "__main__":
    print("salam")
    mimos_sim1()