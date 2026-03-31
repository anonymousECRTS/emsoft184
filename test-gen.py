from __future__ import annotations

import os
import json
import random
import argparse
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional
from math import gcd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import sys
sys.setrecursionlimit(2000)


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
    num_edges: int
    num_sccs: int
    sccs: List[List[int]]
    scc_dag_edges: List[Tuple[int, int]]
    generation_mode: str


@dataclass
class BenchmarkInstance:
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    metadata: Optional[BenchmarkMetadata] = None


# ----------------------------
# Generator
# ----------------------------

class RandomBenchmarkGenerator:
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

    def _generate_random_scc_tree(self, num_sccs: int) -> List[Tuple[int, int]]:
        if num_sccs <= 0:
            raise ValueError("num_sccs must be positive")
        if num_sccs == 1:
            return []

        remaining = list(range(num_sccs))
        self.rng.shuffle(remaining)

        connected = {remaining.pop()}
        undirected: List[Tuple[int, int]] = []

        while remaining:
            a = self.rng.choice(list(connected))
            b = self.rng.choice(remaining)
            undirected.append((a, b))
            connected.add(b)
            remaining.remove(b)

        dag_edges: List[Tuple[int, int]] = []
        for a, b in undirected:
            dag_edges.append((a, b) if a < b else (b, a))
        return dag_edges

    def _random_edge_params(
        self,
        w_range: Tuple[int, int] = (2, 6),
        a_factor: int = 2,
    ) -> Tuple[int, int, int]:
        W_p = self.rng.randint(*w_range)
        U_p = self.rng.randint(1, W_p)
        A_p = self.rng.randint(0, a_factor * W_p)
        return A_p, U_p, W_p

    def _ratio_from_potentials(self, r_src: int, r_dst: int, scale: Optional[int] = None) -> Tuple[int, int]:
        g = gcd(r_src, r_dst)
        base_u = r_dst // g
        base_w = r_src // g
        k = self.rng.randint(1, 3) if scale is None else scale
        return k * base_u, k * base_w

    def _ap_deadlock_free(self, W_p: int) -> int:
        return (W_p - 1) + self.rng.randint(1, 4)

    def _make_base_cycle_for_scc(
        self,
        scc_nodes: List[int],
        enforce_loop_deadlock_free: bool = False,
        w_range: Tuple[int, int] = (2, 6),
        a_factor: int = 2,
    ) -> List[Edge]:
        """
        For SCC size >= 2, create one internal directed cycle so the component is strongly connected.
        For singleton SCC, create no self-loop by default.

        If enforce_loop_deadlock_free=True:
          - assign node potentials r_j
          - set U_p/W_p = r_dst/r_src on every internal edge
          - set A_p > W_p - 1 on every internal edge
        This guarantees every loop in the SCC has g = 1.
        """
        edges: List[Edge] = []

        if len(scc_nodes) <= 1:
            return edges

        cyc = scc_nodes[:] + [scc_nodes[0]]

        if enforce_loop_deadlock_free:
            potentials = {u: self.rng.randint(1, 8) for u in scc_nodes}

            for u, v in zip(cyc[:-1], cyc[1:]):
                U_p, W_p = self._ratio_from_potentials(potentials[u], potentials[v])
                A_p = self._ap_deadlock_free(W_p)
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
            for u, v in zip(cyc[:-1], cyc[1:]):
                A_p, U_p, W_p = self._random_edge_params(
                    w_range=w_range,
                    a_factor=a_factor,
                )
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
        return edges

    def _add_extra_internal_edges(
        self,
        scc_nodes: List[int],
        existing_edges: List[Edge],
        extra_edge_prob: float = 0.15,
        enforce_loop_deadlock_free: bool = False,
        w_range: Tuple[int, int] = (2, 6),
        a_factor: int = 2,
    ) -> List[Edge]:
        """
        If enforce_loop_deadlock_free=True, do NOT add random extra internal edges.
        Reason: arbitrary extra edges create new loops, and direct construction of
        g=1 for all loops requires a globally consistent ratio assignment across all edges.
        The base cycle already forms a strongly connected SCC and safely satisfies the rule.
        """
        if enforce_loop_deadlock_free:
            return existing_edges[:]

        edges = existing_edges[:]
        existing_pairs = {(e.source, e.target) for e in existing_edges}

        for u in scc_nodes:
            for v in scc_nodes:
                if u == v:
                    continue
                if (u, v) in existing_pairs:
                    continue
                if self.rng.random() < extra_edge_prob:
                    A_p, U_p, W_p = self._random_edge_params(
                        w_range=w_range,
                        a_factor=a_factor,
                    )
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
        extra_inter_scc_edge_prob: float = 0.0,
        w_range: Tuple[int, int] = (2, 6),
        a_factor: int = 2,
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

                A_p, U_p, W_p = self._random_edge_params(
                    w_range=w_range,
                    a_factor=a_factor,
                )
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

        num_sccs = len(sccs)
        for su in range(num_sccs):
            for sv in range(su + 1, num_sccs):
                if (su, sv) in scc_dag_edges:
                    continue
                if self.rng.random() >= extra_inter_scc_edge_prob:
                    continue

                src_nodes = sccs[su]
                dst_nodes = sccs[sv]

                attempts = 0
                while True:
                    attempts += 1
                    if attempts > 50:
                        break

                    u = self.rng.choice(src_nodes)
                    v = self.rng.choice(dst_nodes)
                    if (u, v) in existing_pairs:
                        continue

                    A_p, U_p, W_p = self._random_edge_params(
                        w_range=w_range,
                        a_factor=a_factor,
                    )
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
        extra_inter_scc_edge_prob: float = 0.0,
        enforce_loop_deadlock_free: bool = False,
        w_range: Tuple[int, int] = (2, 6),
        a_factor: int = 2,
    ) -> BenchmarkInstance:
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")

        self.edge_id = 0

        if num_sccs is None:
            num_sccs = self.rng.randint(1, num_nodes)

        sccs = self._partition_nodes_into_sccs(num_nodes, num_sccs)
        scc_dag_edges = self._generate_random_scc_tree(num_sccs=num_sccs)

        nodes = [
            Node(id=i, period=self._random_period(period_range))
            for i in range(num_nodes)
        ]

        edges: List[Edge] = []

        for comp in sccs:
            internal_edges = self._make_base_cycle_for_scc(
                comp,
                enforce_loop_deadlock_free=enforce_loop_deadlock_free,
                w_range=w_range,
                a_factor=a_factor,
            )
            internal_edges = self._add_extra_internal_edges(
                comp,
                internal_edges,
                extra_edge_prob=extra_internal_edge_prob,
                enforce_loop_deadlock_free=enforce_loop_deadlock_free,
                w_range=w_range,
                a_factor=a_factor,
            )
            edges.extend(internal_edges)

        edges = self._add_inter_scc_edges_from_dag(
            sccs=sccs,
            scc_dag_edges=scc_dag_edges,
            existing_edges=edges,
            extra_inter_scc_edge_prob=extra_inter_scc_edge_prob,
            w_range=w_range,
            a_factor=a_factor,
        )

        metadata = BenchmarkMetadata(
            benchmark_id=benchmark_id,
            num_nodes=num_nodes,
            num_edges=len(edges),
            num_sccs=len(sccs),
            sccs=sccs,
            scc_dag_edges=scc_dag_edges,
            generation_mode=("loop_deadlock_free" if enforce_loop_deadlock_free else "neutral_random"),
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

    lines = []
    lines.append("=== Benchmark Summary ===")
    lines.append(f"Benchmark ID: {md.benchmark_id}")
    lines.append(f"Generation mode: {md.generation_mode}")
    lines.append(f"Nodes: {len(instance.nodes)}")
    lines.append(f"Edges: {len(instance.edges)}")
    lines.append(f"SCCs: {md.num_sccs}")
    lines.append(f"SCC-DAG edges: {md.scc_dag_edges}")
    lines.append("SCC decomposition:")

    for i, comp in enumerate(md.sccs):
        lines.append(f"  SCC {i}: nodes={comp}")

    return "\n".join(lines)


def draw_benchmark(instance: BenchmarkInstance, image_path: str, dpi: int = 120):
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

    fig_w = min(max(12, 2.8 * max_width + 8), 40)
    fig_h = min(max(6, 2.5 * len(sccs) + 3), 20)

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
            x, y, text, fontsize=10, ha="center", va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.2)
        )

    internal_forward, internal_backward, inter_scc = [], [], []

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
        rect = Rectangle((x - box_w / 2, y - box_h / 2), box_w, box_h,
                         fill=False, edgecolor="black", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, y + 0.16, f"n{n.id}", ha="center", va="center", fontsize=12)
        ax.text(x, y - 0.18, f"P={n.period}", ha="center", va="center", fontsize=10)

    for e in internal_forward:
        p1 = side_port(e.source, "right", e.id)
        p2 = side_port(e.target, "left", e.id)
        draw_arrow(p1, p2)
        label_at((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.18, rf"$({e.U_p},{e.W_p})$")

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
        src, dst = e.source, e.target
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

    plt.savefig(image_path, dpi=dpi, bbox_inches="tight")
    plt.close()


# ----------------------------
# Main
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate benchmark networks.")
    parser.add_argument("--num-benchmarks", type=int, required=True)
    parser.add_argument("--dest", type=str, required=True)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--period-min", type=int, default=2)
    parser.add_argument("--period-max", type=int, default=10)
    parser.add_argument("--num-sccs", type=int, default=None)

    parser.add_argument("--w-min", type=int, default=2)
    parser.add_argument("--w-max", type=int, default=6)
    parser.add_argument("--a-factor", type=int, default=2,
                        help="Initial tokens are sampled as A_p in [0, a_factor * W_p].")

    parser.add_argument("--extra-internal-edge-prob", type=float, default=0.15)
    parser.add_argument("--extra-inter-scc-edge-prob", type=float, default=0.0)
    parser.add_argument(
        "--enforce-loop-deadlock-free",
        action="store_true",
        help="Construct each SCC so all loops have g=1 and A_p > W_p-1 edgewise.",
    )
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--no-draw", action="store_true", help="Do not render graph.png files.")
    parser.add_argument("--draw-first-k", type=int, default=None,
                        help="Draw only the first k benchmarks. Others are saved without images.")
    parser.add_argument("--draw-dpi", type=int, default=120,
                        help="DPI for saved images. Lower values use less memory.")
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
    if not (0.0 <= args.extra_internal_edge_prob <= 1.0):
        raise ValueError("--extra-internal-edge-prob must be in [0, 1]")
    if not (0.0 <= args.extra_inter_scc_edge_prob <= 1.0):
        raise ValueError("--extra-inter-scc-edge-prob must be in [0, 1]")
    if args.draw_dpi <= 0:
        raise ValueError("--draw-dpi must be positive")
    if args.draw_first_k is not None and args.draw_first_k < 0:
        raise ValueError("--draw-first-k must be nonnegative")
    if args.w_min > args.w_max:
        raise ValueError("--w-min must be <= --w-max")
    if args.a_factor < 0:
        raise ValueError("--a-factor must be nonnegative")

    os.makedirs(args.dest, exist_ok=True)
    generator = RandomBenchmarkGenerator(seed=args.seed)

    for bench_id in range(args.num_benchmarks):
        instance = generator.generate(
            benchmark_id=bench_id,
            num_nodes=args.num_nodes,
            period_range=(args.period_min, args.period_max),
            num_sccs=args.num_sccs,
            extra_internal_edge_prob=args.extra_internal_edge_prob,
            extra_inter_scc_edge_prob=args.extra_inter_scc_edge_prob,
            enforce_loop_deadlock_free=args.enforce_loop_deadlock_free,
            w_range=(args.w_min, args.w_max),
            a_factor=args.a_factor,
        )

        bench_folder = os.path.join(args.dest, f"benchmark_{bench_id:04d}")
        os.makedirs(bench_folder, exist_ok=True)

        save_benchmark(instance, bench_folder)

        should_draw = not args.no_draw
        if args.draw_first_k is not None:
            should_draw = should_draw and (bench_id < args.draw_first_k)

        if should_draw:
            draw_benchmark(
                instance,
                os.path.join(bench_folder, "graph.png"),
                dpi=args.draw_dpi,
            )

        print(f"Generated benchmark {bench_id} in {bench_folder}")


if __name__ == "__main__":
    main()