import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class Node:
    id: int
    period: int
    alpha_base: Optional[int] = None
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
    def from_json_file(cls, filepath: str) -> "MimosNetwork":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        net = cls()
        for nd in data.get("nodes", []):
            net.add_node(Node(
                id=int(nd["id"]),
                period=int(nd["period"]),
                alpha_base=nd.get("alpha_base", None),
            ))

        for ed in data.get("edges", []):
            net.add_edge(Edge(
                id=int(ed["id"]),
                source=int(ed["source"]),
                target=int(ed["target"]),
                A_p=int(ed["A_p"]),
                U_p=int(ed["U_p"]),
                W_p=int(ed["W_p"]),
            ))
        return net

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

            sccs.append(sorted(comp))

        return sccs

    def is_trivial_scc(self, comp: List[int]) -> bool:
        if len(comp) != 1:
            return False
        u = comp[0]
        for e in self.edges.values():
            if e.source == u and e.target == u:
                return False
        return True

    def internal_incoming_edges(self, comp: List[int]) -> Dict[int, List[Edge]]:
        comp_set = set(comp)
        incoming: Dict[int, List[Edge]] = {j: [] for j in comp}
        for e in self.edges.values():
            if e.source in comp_set and e.target in comp_set:
                incoming[e.target].append(e)
        for j in comp:
            incoming[j].sort(key=lambda ee: ee.id)
        return incoming

    @staticmethod
    def rhs_value(e: Edge, x_src: int) -> int:
        num = e.A_p - e.W_p + 1 + e.U_p * x_src
        return (num + e.W_p - 1) // e.W_p  # ceil(num / W_p)

    def solve_by_formula_iteration(
        self,
        comp: List[int],
        max_iter: int = 100,
        verbose: bool = False,
    ) -> Dict:
        """
        Start with x^(0)=0 and iterate exactly:

            x_j^(k+1) = min_{internal incoming e=(i->j)}
                        ceil((A_p - W_p + 1 + U_p x_i^(k)) / W_p)
        """
        incoming = self.internal_incoming_edges(comp)

        for j in comp:
            if not incoming[j]:
                return {
                    "status": "infeasible_structure",
                    "iteration": 0,
                    "x": {},
                    "sum_x": None,
                    "time_sec": 0.0,
                    "reason": f"node {j} has no internal incoming edge",
                }

        x_prev = {j: 0 for j in comp}

        t0 = time.perf_counter()
        for it in range(1, max_iter + 1):
            x_curr: Dict[int, int] = {}

            for j in comp:
                x_curr[j] = min(
                    self.rhs_value(e, x_prev[e.source])
                    for e in incoming[j]
                )

            if verbose:
                print(f"iter={it}, sum(x)={sum(x_curr.values())}")

            if x_curr == x_prev:
                t1 = time.perf_counter()
                return {
                    "status": "converged",
                    "iteration": it,
                    "x": x_curr,
                    "sum_x": sum(x_curr.values()),
                    "time_sec": t1 - t0,
                }

            x_prev = x_curr

        t1 = time.perf_counter()
        return {
            "status": "max_iter_reached",
            "iteration": max_iter,
            "x": x_prev,
            "sum_x": sum(x_prev.values()),
            "time_sec": t1 - t0,
        }

    def solve_by_pulp(
        self,
        comp: List[int],
        msg: bool = False,
        objective_mode: str = "min_sum",
    ) -> Dict:
        try:
            import pulp
        except Exception as e:
            raise RuntimeError("PuLP is required: pip install pulp") from e

        if self.is_trivial_scc(comp):
            return {
                "solver": "pulp",
                "status": "trivial_scc",
                "feasible": False,
                "objective": None,
                "x": {},
                "time_sec": 0.0,
            }

        incoming = self.internal_incoming_edges(comp)

        for j in comp:
            if not incoming[j]:
                return {
                    "solver": "pulp",
                    "status": "missing_internal_incoming",
                    "feasible": False,
                    "objective": None,
                    "x": {},
                    "time_sec": 0.0,
                }

        prob = pulp.LpProblem("scc_witness", pulp.LpMinimize)

        lb = {
            j: min(e.A_p // e.W_p for e in incoming[j])
            for j in comp
        }

        x = {
            j: pulp.LpVariable(f"x_{j}", lowBound=lb[j], cat=pulp.LpInteger)
            for j in comp
        }

        y = {}
        for j in comp:
            for e in incoming[j]:
                y[e.id] = pulp.LpVariable(
                    f"y_{e.id}",
                    lowBound=0,
                    upBound=1,
                    cat=pulp.LpBinary
                )

        for j in comp:
            prob += pulp.lpSum(y[e.id] for e in incoming[j]) == 1

        M = 10**6
        for j in comp:
            for e in incoming[j]:
                prob += (
                    e.A_p + e.U_p * x[e.source]
                    <= e.W_p * x[j] + (e.W_p - 1) + M * (1 - y[e.id])
                )

        if objective_mode == "min_sum":
            prob += pulp.lpSum(x[j] for j in comp)
        elif objective_mode == "feasibility":
            prob += 0
        else:
            raise ValueError("objective_mode must be 'min_sum' or 'feasibility'")

        t0 = time.perf_counter()
        status = prob.solve(pulp.PULP_CBC_CMD(msg=msg))
        t1 = time.perf_counter()

        status_str = pulp.LpStatus[status]
        feasible = (status_str == "Optimal")

        x_sol = {}
        obj = None
        if feasible:
            x_sol = {j: int(round(pulp.value(x[j]))) for j in comp}
            if objective_mode == "min_sum":
                obj_val = pulp.value(prob.objective)
                obj = None if obj_val is None else int(round(obj_val))

        return {
            "solver": "pulp",
            "status": status_str,
            "feasible": feasible,
            "objective": obj,
            "x": x_sol,
            "time_sec": t1 - t0,
        }

    def solve_by_z3(
        self,
        comp: List[int],
        objective_mode: str = "min_sum",
    ) -> Dict:
        try:
            import z3
        except Exception as e:
            raise RuntimeError("z3-solver is required: pip install z3-solver") from e

        if self.is_trivial_scc(comp):
            return {
                "solver": "z3",
                "status": "trivial_scc",
                "feasible": False,
                "objective": None,
                "x": {},
                "time_sec": 0.0,
            }

        incoming = self.internal_incoming_edges(comp)

        for j in comp:
            if not incoming[j]:
                return {
                    "solver": "z3",
                    "status": "missing_internal_incoming",
                    "feasible": False,
                    "objective": None,
                    "x": {},
                    "time_sec": 0.0,
                }

        lb = {
            j: min(e.A_p // e.W_p for e in incoming[j])
            for j in comp
        }

        x = {j: z3.Int(f"x_{j}") for j in comp}
        y = {}
        for j in comp:
            for e in incoming[j]:
                y[e.id] = z3.Bool(f"y_{e.id}")

        M = 10**6

        if objective_mode == "min_sum":
            solver = z3.Optimize()
        elif objective_mode == "feasibility":
            solver = z3.Solver()
        else:
            raise ValueError("objective_mode must be 'min_sum' or 'feasibility'")

        for j in comp:
            solver.add(x[j] >= lb[j])

        for j in comp:
            bools = [y[e.id] for e in incoming[j]]
            solver.add(z3.PbEq([(b, 1) for b in bools], 1))

        for j in comp:
            for e in incoming[j]:
                solver.add(
                    z3.Implies(
                        y[e.id],
                        e.A_p + e.U_p * x[e.source] <= e.W_p * x[j] + (e.W_p - 1)
                    )
                )

        obj_handle = None
        if objective_mode == "min_sum":
            obj_handle = solver.minimize(z3.Sum([x[j] for j in comp]))

        t0 = time.perf_counter()
        result = solver.check()
        t1 = time.perf_counter()

        feasible = (result == z3.sat)
        x_sol = {}
        obj = None

        if feasible:
            model = solver.model()
            x_sol = {j: model.eval(x[j], model_completion=True).as_long() for j in comp}
            if objective_mode == "min_sum":
                obj = sum(x_sol.values())

        return {
            "solver": "z3",
            "status": str(result),
            "feasible": feasible,
            "objective": obj,
            "x": x_sol,
            "time_sec": t1 - t0,
        }

    def solve_by_solver(
        self,
        comp: List[int],
        solver_name: str = "pulp",
        objective_mode: str = "min_sum",
        solver_msg: bool = False,
    ) -> Dict:
        if solver_name == "pulp":
            return self.solve_by_pulp(comp, msg=solver_msg, objective_mode=objective_mode)
        elif solver_name == "z3":
            return self.solve_by_z3(comp, objective_mode=objective_mode)
        else:
            raise ValueError("solver_name must be 'pulp' or 'z3'")


def compare_formula_and_solver(
    json_path: str,
    scc_index: int = 0,
    max_iter: int = 100,
    verbose: bool = False,
    solver_name: str = "pulp",
    solver_msg: bool = False,
    objective_mode: str = "min_sum",
):
    net = MimosNetwork.from_json_file(json_path)
    sccs = net.sccs_in_topo_order()

    if scc_index < 0 or scc_index >= len(sccs):
        raise ValueError(f"scc-index out of range: 0..{len(sccs)-1}")

    comp = sccs[scc_index]

    fp_res = net.solve_by_formula_iteration(
        comp=comp,
        max_iter=max_iter,
        verbose=verbose,
    )

    solver_res = net.solve_by_solver(
        comp=comp,
        solver_name=solver_name,
        objective_mode=objective_mode,
        solver_msg=solver_msg,
    )

    print(f"Benchmark:                 {json_path}")
    print(f"SCC index:                 {scc_index}")
    print(f"SCC size:                  {len(comp)}")
    print(f"Formula iteration status:  {fp_res['status']}")
    print(f"Formula iterations:        {fp_res['iteration']}")
    print(f"Formula sum x:             {fp_res['sum_x']}")
    print(f"Formula time:              {fp_res['time_sec']:.6f} s")
    print(f"Solver:                    {solver_name}")
    print(f"Solver status:             {solver_res['status']}")
    print(f"Solver feasible:           {solver_res['feasible']}")
    print(f"Objective mode:            {objective_mode}")
    print(f"Solver objective:          {solver_res['objective']}")
    print(f"Solver time:               {solver_res['time_sec']:.6f} s")

    if objective_mode == "min_sum":
        same_x = False
        same_sum = False

        if fp_res["status"] == "converged" and solver_res["feasible"]:
            same_x = (fp_res["x"] == solver_res["x"])
            same_sum = (fp_res["sum_x"] == solver_res["objective"])

        print(f"Same x vector:             {same_x}")
        print(f"Same sum/objective:        {same_sum}")

        if fp_res["status"] == "converged" and solver_res["feasible"] and not same_x:
            mismatches = [
                (j, fp_res["x"][j], solver_res["x"][j])
                for j in comp
                if fp_res["x"][j] != solver_res["x"][j]
            ]
            print(f"Mismatched nodes:          {len(mismatches)}")
            for j, xv_fp, xv_solver in mismatches[:20]:
                print(f"  node {j}: formula={xv_fp}, solver={xv_solver}")
            if len(mismatches) > 20:
                print("  ...")
    else:
        print("Same x vector:             N/A in feasibility mode")
        print("Same sum/objective:        N/A in feasibility mode")

    return {
        "formula": fp_res,
        "solver": solver_res,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run formula iteration and either PuLP or Z3 together on one SCC."
    )
    parser.add_argument("--json", required=True, help="Path to benchmark.json")
    parser.add_argument("--scc-index", type=int, default=0, help="SCC index in topological order")
    parser.add_argument("--max-iter", type=int, default=100, help="Max iterations for formula iteration")
    parser.add_argument("--verbose", action="store_true", help="Show formula iteration sums")
    parser.add_argument("--solver-msg", action="store_true", help="Show CBC log for PuLP")
    parser.add_argument("--solver", choices=["pulp", "z3"], default="pulp", help="Choose solver backend")
    parser.add_argument(
        "--objective-mode",
        choices=["min_sum", "feasibility"],
        default="min_sum",
        help="Solver objective: minimize sum(x) or pure feasibility",
    )
    args = parser.parse_args()

    compare_formula_and_solver(
        json_path=args.json,
        scc_index=args.scc_index,
        max_iter=args.max_iter,
        verbose=args.verbose,
        solver_name=args.solver,
        solver_msg=args.solver_msg,
        objective_mode=args.objective_mode,
    )


if __name__ == "__main__":
    main()