 python test-gen.py   --num-benchmarks 10   --dest ./deadlock_suite   --num-nodes 12   --period-min 2   --period-max 10   --seed 42

python final-code.py --json ./deadlock_suite/benchmark_0000/benchmark.json

python test-gen.py --num-benchmarks 5 --dest out --num-nodes 12 --num-sccs 1 --extra-deadlock-prob 0.3 --seed 42

python generator.py --num-benchmarks 10 --dest out --num-nodes 20 --num-sccs 5 --enforce-loop-deadlock-free --seed 42

python test-gen.py --num-benchmarks 10 --dest out --num-nodes 100 --num-sccs 1 --seed 42 --a-factor 3 --no-draw --extra-internal-edge-prob 0.0

python test-gen.py --num-benchmarks 50 --dest scc-500/1 --num-nodes 500  --num-sccs 1 --seed 42 --a-factor 3 --no-draw --extra-inter-scc-edge-prob 0.1 --extra-internal-edge-prob 0.02

 python test-gen.py --num-benchmarks 50 --dest scc-500-n/incons/1 --num-nodes 500  --num-sccs 1 --seed 42 --a-factor 2 --no-draw --extra-inter-scc-edge-prob 0.1 --extra-internal-edge-prob 0.02 --enforce-loop-deadlock-free
