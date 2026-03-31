from __future__ import annotations

import os
import json
import random
import argparse
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


# ----------------------------
# Data model
# ----------------------------

@dataclass
class Node:
    id: int
    period: int


@dataclass
class Edge:
    id: int
    source: int
    target: int
    A_p: int
    U_p: int
    W_p: int


@dataclass
class BenchmarkMetadata:
    benchmark_id: int
    num_nodes: int
    num_sccs: int
    sccs: List[List[int]]
    scc_dag_edges: List[Tuple[int, int]]

    # intrinsic reason only: "initial", "structural", or "none"
    scc_deadlock_type: Dict[int, str]

    # final reason after propagation:
    # "initial", "structural", "reachable_from_blocked", or "unblocked"
    scc_final_reason: Dict[int, str]

    blocked_sccs: List[int]
    unblocked_sccs: List[int]

    blocked_nodes: List[int]
    global_deadlock: bool


@dataclass
class BenchmarkInstance:
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    metadata: Optional[BenchmarkMetadata] = None


# ----------------------------
# Generator
# ----------------------------

class DeadlockBenchmarkGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.edge_id = 0

    def _next_edge_id(self) -> int:
        eid = self.edge_id
        self.edge_id += 1
        return eid

    def _random_period(self, period_range: Tuple[int, int]) -> int:
        lo, hi = period_range
        return self.rng.randint(lo, hi)

    def _partition_nodes_into_sccs(self, num_nodes: int, num_sccs: int) -> List[List[int]]:
        if not (1 <= num_sccs <= num_nodes):
            raise ValueError("num_sccs must satisfy 1 <= num_sccs <= num_nodes")

        nodes = list(range(num_nodes))
        self.rng.shuffle(nodes)

        sizes = [1] * num_sccs
        for _ in range(num_nodes - num_sccs):
            sizes[self.rng.randrange(num_sccs)] += 1

        sccs = []
        idx = 0
        for s in sizes:
            sccs.append(sorted(nodes[idx:idx + s]))
            idx += s
        return sccs

    def _choose_seed_deadlock_type(self) -> str:
        return self.rng.choice(["structural", "initial"])

    def _choose_seed_deadlock_sccs(
        self,
        num_sccs: int,
        extra_deadlock_prob: float = 0.3,
    ) -> List[int]:
        """
        Choose intrinsically deadlocked SCCs.
        At least one SCC is guaranteed to be deadlocked.
        """
        if num_sccs <= 0:
            return []

        first = self.rng.randrange(num_sccs)
        seeds = {first}

        for scc_idx in range(num_sccs):
            if scc_idx == first:
                continue
            if self.rng.random() < extra_deadlock_prob:
                seeds.add(scc_idx)

        return sorted(seeds)

    def _generate_random_scc_tree(self, num_sccs: int) -> List[Tuple[int, int]]:
        """
        Generate exactly num_sccs - 1 SCC-level edges to connect all SCCs.
        Then orient each edge from smaller SCC index to larger SCC index to form a DAG.
        """
        if num_sccs <= 0:
            raise ValueError("num_sccs must be positive")
        if num_sccs == 1:
            return []

        remaining = list(range(num_sccs))
        self.rng.shuffle(remaining)

        connected = {remaining.pop()}
        tree_edges_undirected: List[Tuple[int, int]] = []

        while remaining:
            a = self.rng.choice(list(connected))
            b = self.rng.choice(remaining)
            tree_edges_undirected.append((a, b))
            connected.add(b)
            remaining.remove(b)

        dag_edges: List[Tuple[int, int]] = []
        for a, b in tree_edges_undirected:
            dag_edges.append((a, b) if a < b else (b, a))

        return dag_edges

    def _compute_blocked_sccs(
        self,
        num_sccs: int,
        scc_dag_edges: List[Tuple[int, int]],
        seed_deadlocked_sccs: List[int],
    ) -> List[int]:
        adj = {i: [] for i in range(num_sccs)}
        for u, v in scc_dag_edges:
            adj[u].append(v)

        blocked = set(seed_deadlocked_sccs)
        stack = list(seed_deadlocked_sccs)

        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in blocked:
                    blocked.add(v)
                    stack.append(v)

        return sorted(blocked)

    def _compute_scc_final_reason(
        self,
        num_sccs: int,
        seed_deadlocked_sccs: List[int],
        blocked_sccs: List[int],
        scc_deadlock_type: Dict[int, str],
    ) -> Dict[int, str]:
        final_reason: Dict[int, str] = {}
        seed_set = set(seed_deadlocked_sccs)
        blocked_set = set(blocked_sccs)

        for i in range(num_sccs):
            intrinsic = scc_deadlock_type[i]
            if i in seed_set:
                final_reason[i] = intrinsic
            elif i in blocked_set:
                final_reason[i] = "reachable_from_blocked"
            else:
                final_reason[i] = "unblocked"

        return final_reason

    def _make_cycle_edges_for_scc(self, scc_nodes: List[int], deadlock_type: str) -> List[Edge]:
        """
        For SCC size >= 2, create one internal cycle.
        For singleton SCC, do NOT force a self-loop.

        deadlock_type:
          - "structural": intrinsically blocked
          - "initial": intrinsically blocked
          - "none": not intrinsically blocked
        """
        edges: List[Edge] = []

        if len(scc_nodes) <= 1:
            return edges

        cyc = scc_nodes[:] + [scc_nodes[0]]
        cycle_pairs = list(zip(cyc[:-1], cyc[1:]))

        if deadlock_type == "structural":
            for u, v in cycle_pairs:
                W_p = self.rng.randint(2, 6)
                U_p = self.rng.randint(1, W_p - 1)
                A_p = self.rng.randint(0, 2 * W_p)
                edges.append(
                    Edge(
                        id=self._next_edge_id(),
                        source=u,
                        target=v,
                        A_p=A_p,
                        U_p=U_p,
                        W_p=W_p,
                    )
                )

        elif deadlock_type == "initial":
            tmp = []
            total_threshold = 0

            for u, v in cycle_pairs:
                W_p = self.rng.randint(2, 6)
                U_p = self.rng.randint(1, W_p)
                tmp.append((u, v, U_p, W_p))
                total_threshold += (W_p - 1)

            total_A = self.rng.randint(0, max(0, total_threshold - 1))
            A_values = [0] * len(tmp)
            for _ in range(total_A):
                A_values[self.rng.randrange(len(tmp))] += 1

            for (u, v, U_p, W_p), A_p in zip(tmp, A_values):
                edges.append(
                    Edge(
                        id=self._next_edge_id(),
                        source=u,
                        target=v,
                        A_p=A_p,
                        U_p=U_p,
                        W_p=W_p,
                    )
                )

        elif deadlock_type == "none":
            for u, v in cycle_pairs:
                W_p = self.rng.randint(2, 6)
                U_p = self.rng.randint(1, W_p)
                A_p = self.rng.randint(W_p, 2 * W_p + 3)
                edges.append(
                    Edge(
                        id=self._next_edge_id(),
                        source=u,
                        target=v,
                        A_p=A_p,
                        U_p=U_p,
                        W_p=W_p,
                    )
                )
        else:
            raise ValueError(f"Unknown deadlock_type: {deadlock_type}")

        return edges

    def _add_extra_internal_edges(
        self,
        scc_nodes: List[int],
        existing_edges: List[Edge],
        extra_edge_prob: float = 0.20,
    ) -> List[Edge]:
        edges = existing_edges[:]
        existing_pairs = {(e.source, e.target) for e in existing_edges}

        for u in scc_nodes:
            for v in scc_nodes:
                if u == v:
                    continue
                if (u, v) in existing_pairs:
                    continue
                if self.rng.random() < extra_edge_prob:
                    W_p = self.rng.randint(2, 6)
                    U_p = self.rng.randint(1, W_p)
                    A_p = self.rng.randint(0, 2 * W_p)
                    edges.append(
                        Edge(
                            id=self._next_edge_id(),
                            source=u,
                            target=v,
                            A_p=A_p,
                            U_p=U_p,
                            W_p=W_p,
                        )
                    )
                    existing_pairs.add((u, v))
        return edges

    def _add_inter_scc_edges_from_dag(
        self,
        sccs: List[List[int]],
        scc_dag_edges: List[Tuple[int, int]],
        existing_edges: List[Edge],
    ) -> List[Edge]:
        edges = existing_edges[:]
        existing_pairs = {(e.source, e.target) for e in existing_edges}

        for scc_u, scc_v in scc_dag_edges:
            src_nodes = sccs[scc_u]
            dst_nodes = sccs[scc_v]

            attempts = 0
            while True:
                attempts += 1
                if attempts > 100:
                    raise RuntimeError(
                        f"Could not find a fresh inter-SCC edge from SCC {scc_u} to SCC {scc_v}"
                    )

                u = self.rng.choice(src_nodes)
                v = self.rng.choice(dst_nodes)

                if (u, v) in existing_pairs:
                    continue

                W_p = self.rng.randint(2, 6)
                U_p = self.rng.randint(1, W_p)
                A_p = self.rng.randint(0, 2 * W_p)

                edges.append(
                    Edge(
                        id=self._next_edge_id(),
                        source=u,
                        target=v,
                        A_p=A_p,
                        U_p=U_p,
                        W_p=W_p,
                    )
                )
                existing_pairs.add((u, v))
                break

        return edges

    def generate(
        self,
        benchmark_id: int,
        num_nodes: int,
        period_range: Tuple[int, int] = (2, 10),
        num_sccs: Optional[int] = None,
        extra_internal_edge_prob: float = 0.15,
        inter_scc_edge_prob: float = 0.25,
        extra_deadlock_prob: float = 0.3,
    ) -> BenchmarkInstance:
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")

        self.edge_id = 0

        if num_sccs is None:
            num_sccs = self.rng.randint(1, num_nodes)

        sccs = self._partition_nodes_into_sccs(num_nodes, num_sccs)
        scc_dag_edges = self._generate_random_scc_tree(num_sccs=num_sccs)

        nodes = [Node(id=i, period=self._random_period(period_range)) for i in range(num_nodes)]
        edges: List[Edge] = []

        seed_deadlocked_sccs = self._choose_seed_deadlock_sccs(
            num_sccs=num_sccs,
            extra_deadlock_prob=extra_deadlock_prob,
        )

        blocked_sccs = self._compute_blocked_sccs(
            num_sccs=num_sccs,
            scc_dag_edges=scc_dag_edges,
            seed_deadlocked_sccs=seed_deadlocked_sccs,
        )

        scc_deadlock_type: Dict[int, str] = {}

        for scc_idx, comp in enumerate(sccs):
            if scc_idx in seed_deadlocked_sccs:
                deadlock_type = self._choose_seed_deadlock_type()
            else:
                deadlock_type = "none"

            scc_deadlock_type[scc_idx] = deadlock_type

            cyc_edges = self._make_cycle_edges_for_scc(comp, deadlock_type)
            cyc_edges = self._add_extra_internal_edges(
                comp,
                cyc_edges,
                extra_edge_prob=extra_internal_edge_prob,
            )
            edges.extend(cyc_edges)

        edges = self._add_inter_scc_edges_from_dag(
            sccs=sccs,
            scc_dag_edges=scc_dag_edges,
            existing_edges=edges,
        )

        scc_final_reason = self._compute_scc_final_reason(
            num_sccs=num_sccs,
            seed_deadlocked_sccs=seed_deadlocked_sccs,
            blocked_sccs=blocked_sccs,
            scc_deadlock_type=scc_deadlock_type,
        )

        blocked_nodes = sorted(
            nid
            for scc_idx in blocked_sccs
            for nid in sccs[scc_idx]
        )

        unblocked_sccs = sorted(set(range(num_sccs)) - set(blocked_sccs))

        metadata = BenchmarkMetadata(
            benchmark_id=benchmark_id,
            num_nodes=num_nodes,
            num_sccs=len(sccs),
            sccs=sccs,
            scc_dag_edges=scc_dag_edges,
            scc_deadlock_type=scc_deadlock_type,
            scc_final_reason=scc_final_reason,
            blocked_sccs=blocked_sccs,
            unblocked_sccs=unblocked_sccs,
            blocked_nodes=blocked_nodes,
            global_deadlock=(len(blocked_nodes) == num_nodes),
        )

        return BenchmarkInstance(nodes=nodes, edges=edges, metadata=metadata)


# ----------------------------
# Save / summary / drawing
# ----------------------------

def save_benchmark(instance: BenchmarkInstance, folderpath: str):
    os.makedirs(folderpath, exist_ok=True)

    payload = {
        "nodes": [asdict(n) for n in instance.nodes],
        "edges": [asdict(e) for e in instance.edges],
        "metadata": asdict(instance.metadata) if instance.metadata else None,
    }

    json_path = os.path.join(folderpath, "benchmark.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_path = os.path.join(folderpath, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(build_summary_text(instance))


def build_summary_text(instance: BenchmarkInstance) -> str:
    md = instance.metadata
    if md is None:
        return "No metadata available."

    scc_final_reason = {int(k): v for k, v in md.scc_final_reason.items()}

    node_final_reason: Dict[int, str] = {}
    for i, comp in enumerate(md.sccs):
        reason = scc_final_reason[i]
        for nid in comp:
            node_final_reason[nid] = reason

    nodes_by_reason = {
        "initial": sorted([nid for nid, r in node_final_reason.items() if r == "initial"]),
        "structural": sorted([nid for nid, r in node_final_reason.items() if r == "structural"]),
        "reachable_from_blocked": sorted(
            [nid for nid, r in node_final_reason.items() if r == "reachable_from_blocked"]
        ),
        "unblocked": sorted([nid for nid, r in node_final_reason.items() if r == "unblocked"]),
    }

    lines = []
    lines.append("=== Benchmark Summary ===")
    lines.append(f"Benchmark ID: {md.benchmark_id}")
    lines.append(f"Nodes: {len(instance.nodes)}")
    lines.append(f"Edges: {len(instance.edges)}")
    lines.append(f"SCCs: {md.num_sccs}")
    lines.append(f"SCC-DAG edges: {md.scc_dag_edges}")
    lines.append(f"Blocked SCCs: {md.blocked_sccs}")
    lines.append(f"Unblocked SCCs: {md.unblocked_sccs}")
    lines.append("")

    lines.append("SCC final status:")
    for i, comp in enumerate(md.sccs):
        intrinsic = md.scc_deadlock_type[i] if isinstance(list(md.scc_deadlock_type.keys())[0], int) else md.scc_deadlock_type[str(i)]
        lines.append(
            f"  SCC {i}: nodes={comp}, intrinsic_type={intrinsic}, final_reason={scc_final_reason[i]}"
        )

    lines.append("")
    lines.append("Blocked nodes by reason:")
    lines.append(f"  initial: {nodes_by_reason['initial']}")
    lines.append(f"  structural: {nodes_by_reason['structural']}")
    lines.append(f"  reachable_from_blocked: {nodes_by_reason['reachable_from_blocked']}")
    lines.append(f"  unblocked: {nodes_by_reason['unblocked']}")
    lines.append("")
    lines.append(f"All blocked nodes: {md.blocked_nodes}")
    lines.append(f"Global deadlock: {md.global_deadlock}")

    return "\n".join(lines)


def print_benchmark_summary(instance: BenchmarkInstance):
    print(build_summary_text(instance))


def draw_benchmark(instance, image_path: str):
    sccs = instance.metadata.sccs
    if not sccs:
        raise ValueError("No SCC information available for drawing.")

    node_to_scc: Dict[int, int] = {}
    for scc_idx, comp in enumerate(sccs):
        for nid in comp:
            node_to_scc[nid] = scc_idx

    node_pos: Dict[int, Tuple[float, float]] = {}
    x_gap = 3.0
    y_gap = 3.0
    max_width = max(len(comp) for comp in sccs)

    for row_idx, comp in enumerate(sccs):
        y = -row_idx * y_gap
        row_shift = (max_width - len(comp)) * 1.0
        for col_idx, nid in enumerate(comp):
            x = row_shift + col_idx * x_gap
            node_pos[nid] = (x, y)

    fig_w = max(12, 2.8 * max_width + 8)
    fig_h = max(6, 2.5 * len(sccs) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    box_w = 1.55
    box_h = 0.95

    def rect_bounds(nid: int):
        x, y = node_pos[nid]
        left = x - box_w / 2
        right = x + box_w / 2
        bottom = y - box_h / 2
        top = y + box_h / 2
        return left, right, bottom, top

    def draw_polyline(points, lw=1.0):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, color="black", linewidth=lw)

    def draw_arrow(start, end, lw=1.0):
        arr = FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=10,
            linewidth=lw,
            color="black",
            shrinkA=0,
            shrinkB=0,
        )
        ax.add_patch(arr)

    def label_at(x: float, y: float, text: str):
        ax.text(
            x,
            y,
            text,
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.2)
        )

    internal_forward = []
    internal_backward = []
    inter_scc = []

    for e in instance.edges:
        scc_u = node_to_scc[e.source]
        scc_v = node_to_scc[e.target]
        sx, sy = node_pos[e.source]
        tx, ty = node_pos[e.target]

        if scc_u == scc_v:
            if abs(sy - ty) < 1e-9 and sx < tx:
                internal_forward.append(e)
            else:
                internal_backward.append(e)
        else:
            inter_scc.append(e)

    side_edges: Dict[Tuple[int, str], List[int]] = {}

    def add_side_edge(nid: int, side: str, eid: int):
        side_edges.setdefault((nid, side), []).append(eid)

    for e in internal_forward:
        add_side_edge(e.source, "right", e.id)
        add_side_edge(e.target, "left", e.id)

    for e in internal_backward:
        add_side_edge(e.source, "top", e.id)
        add_side_edge(e.target, "top", e.id)

    for e in inter_scc:
        add_side_edge(e.source, "bottom", e.id)
        add_side_edge(e.target, "top", e.id)

    for key in side_edges:
        side_edges[key].sort()

    def side_port(nid: int, side: str, eid: int) -> Tuple[float, float]:
        left, right, bottom, top = rect_bounds(nid)
        lst = side_edges.get((nid, side), [])
        if eid not in lst:
            raise ValueError(f"Edge {eid} not assigned to side {side} of node {nid}")

        idx = lst.index(eid)
        n = len(lst)
        frac = (idx + 1) / (n + 1)

        if side == "top":
            return (left + frac * (right - left), top)
        if side == "bottom":
            return (left + frac * (right - left), bottom)
        if side == "left":
            return (left, bottom + frac * (top - bottom))
        if side == "right":
            return (right, bottom + frac * (top - bottom))
        raise ValueError(f"Unknown side: {side}")

    for n in instance.nodes:
        x, y = node_pos[n.id]
        rect = Rectangle(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            fill=False,
            edgecolor="black",
            linewidth=1.2
        )
        ax.add_patch(rect)
        ax.text(x, y + 0.16, f"n{n.id}", ha="center", va="center", fontsize=12)
        ax.text(x, y - 0.18, f"P={n.period}", ha="center", va="center", fontsize=10)

    for e in internal_forward:
        p1 = side_port(e.source, "right", e.id)
        p2 = side_port(e.target, "left", e.id)
        draw_arrow(p1, p2)
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2 + 0.18
        label_at(mx, my, rf"$({e.U_p},{e.W_p})$")

    per_row_count: Dict[int, int] = {}
    for e in internal_backward:
        row = node_to_scc[e.source]
        k = per_row_count.get(row, 0)
        per_row_count[row] = k + 1

        p0 = side_port(e.source, "top", e.id)
        p3 = side_port(e.target, "top", e.id)

        y_corr = max(p0[1], p3[1]) + 0.9 + 0.55 * k

        p1 = (p0[0], y_corr)
        p2 = (p3[0], y_corr)

        draw_polyline([p0, p1, p2])
        draw_arrow(p2, p3)
        label_at((p1[0] + p2[0]) / 2, y_corr + 0.18, rf"$({e.U_p},{e.W_p})$")

    sorted_inter = sorted(
        inter_scc,
        key=lambda e: (node_to_scc[e.source], node_to_scc[e.target], e.source, e.target)
    )

    right_margin_start = max(x for x, _ in node_pos.values()) + 2.0

    for idx, e in enumerate(sorted_inter):
        src = e.source
        dst = e.target

        p0 = side_port(src, "bottom", e.id)
        p5 = side_port(dst, "top", e.id)

        x_corr = right_margin_start + idx * 0.9
        y_leave = p0[1] - 0.40 - 0.06 * idx
        y_above_target = p5[1] + 0.55 + 0.15 * idx

        p1 = (p0[0], y_leave)
        p2 = (x_corr, y_leave)
        p3 = (x_corr, y_above_target)
        p4 = (p5[0], y_above_target)

        draw_polyline([p0, p1, p2, p3, p4])
        draw_arrow(p4, p5)
        label_at(x_corr, (y_leave + y_above_target) / 2, rf"$({e.U_p},{e.W_p})$")

    xs = [p[0] for p in node_pos.values()]
    ys = [p[1] for p in node_pos.values()]
    ax.set_xlim(min(xs) - 2.5, right_margin_start + max(1, len(inter_scc)) * 0.9 + 2.0)
    ax.set_ylim(min(ys) - 2.5, max(ys) + 3.0)

    plt.tight_layout()
    plt.savefig(image_path, dpi=250, bbox_inches="tight")
    plt.close()


# ----------------------------
# Main
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate deadlock benchmark networks.")
    parser.add_argument("--num-benchmarks", type=int, required=True, help="Number of benchmarks to generate.")
    parser.add_argument("--dest", type=str, required=True, help="Destination folder.")
    parser.add_argument("--num-nodes", type=int, required=True, help="Number of nodes in each benchmark.")
    parser.add_argument("--period-min", type=int, default=2, help="Minimum node period.")
    parser.add_argument("--period-max", type=int, default=10, help="Maximum node period.")
    parser.add_argument("--num-sccs", type=int, default=None, help="Number of SCCs. If omitted, chosen randomly for each benchmark.")
    parser.add_argument("--extra-internal-edge-prob", type=float, default=0.15, help="Probability of extra internal SCC edges.")
    parser.add_argument("--inter-scc-edge-prob", type=float, default=0.25, help="Unused now; kept for compatibility.")
    parser.add_argument("--extra-deadlock-prob", type=float, default=0.3, help="For SCCs other than the first guaranteed deadlocked one, probability of also being intrinsically deadlocked.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_benchmarks <= 0:
        raise ValueError("--num-benchmarks must be positive")
    if args.num_nodes <= 0:
        raise ValueError("--num-nodes must be positive")
    if args.period_min > args.period_max:
        raise ValueError("--period-min must be <= --period-max")
    if args.num_sccs is not None and not (1 <= args.num_sccs <= args.num_nodes):
        raise ValueError("--num-sccs must satisfy 1 <= num_sccs <= num_nodes")
    if not (0.0 <= args.extra_deadlock_prob <= 1.0):
        raise ValueError("--extra-deadlock-prob must be in [0, 1]")

    os.makedirs(args.dest, exist_ok=True)

    generator = DeadlockBenchmarkGenerator(seed=args.seed)

    for bench_id in range(args.num_benchmarks):
        instance = generator.generate(
            benchmark_id=bench_id,
            num_nodes=args.num_nodes,
            period_range=(args.period_min, args.period_max),
            num_sccs=args.num_sccs,
            extra_internal_edge_prob=args.extra_internal_edge_prob,
            inter_scc_edge_prob=args.inter_scc_edge_prob,
            extra_deadlock_prob=args.extra_deadlock_prob,
        )

        bench_folder = os.path.join(args.dest, f"benchmark_{bench_id:04d}")
        os.makedirs(bench_folder, exist_ok=True)

        save_benchmark(instance, bench_folder)
        draw_benchmark(instance, os.path.join(bench_folder, "graph.png"))

        print(f"Generated benchmark {bench_id} in {bench_folder}")


if __name__ == "__main__":
    main()