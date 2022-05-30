#!/bin/bash
#$ -pe smp 32
#$ -l h_vmem=8G
#$ -l h_rt=240:0:0
#$ -cwd
#$ -j y
#$ -m bea

# Run cell-line experiment with 5 timepoints (samples).
# Execute script from the ecdna-evo folder to be able to use cargo

function remove_dot {
  echo "$1" | tr -d "."
}

RUNS=1000
CELLS=1000000000
SAMPLE=1000

# ground truth
f=2.8
g=20

# ranges for abc
F1=1
F2=3
D1=0
D2=0

RESULTS=$HOME/ecdna-evo/results
name=f$(remove_dot $f)g$g
EXPERIMENT=culture5timepoints/${name}
echo $name

# simulate
cargo run -q --release --bin ecdna -- simulate \
  --cells ${CELLS} \
  --runs 1 \
  --rho1 $f \
  --distribution ${RESULTS}/distributions/${g}copies.json \
  --dynamics nplus \
  --samples ${SAMPLE} ${SAMPLE} ${SAMPLE} ${SAMPLE} ${SAMPLE} \
  --sizes ${CELLS} ${CELLS} ${CELLS} ${CELLS} ${CELLS} \
  --culture \
	--patient ${EXPERIMENT} \
	--savedir ${TMPDIR}

if [[ ! "$?" -eq 0 ]]; then
	echo "Error while generating the ground truth data in ${TMPDIR} $(date +"%T")"
	exit 1
fi

# preprocess
for id in {0..4}; do
  cargo run -q --release --bin preprocess -- \
		${EXPERIMENT} sample${id} ${CELLS} \
		--distribution ${TMPDIR}/results/${EXPERIMENT}/${SAMPLE}samples${CELLS}cells/${id}/ecdna/0.json
done
if [[ ! "$?" -eq 0 ]]; then
	echo "Error while preprocessing the ground truth data ${TMPDIR} $(date +"%T")"
	exit 1
fi

# abc
cargo run -q --release --bin ecdna -- abc \
  -r 10000 --culture \
  --rho1-range 1 3 --rho2-range 1 \
  --delta1-range 0 --delta2-range 0 \
  --copies-range 1 30 \
	--patient ${TMPDIR}/results/preprocessed/${EXPERIMENT}.json \
	--savedir ${TMPDIR}

if [[ ! "$?" -eq 0 ]]; then
  echo "Error while running ABC $(date +"%T")"
	exit 1
else
	echo "End simulations $(date +"%T")"
fi

echo "Copying file back home $(date +"%T")"
# Copy only changed files back
rsync -rauL --ignore-existing $TMPDIR/results/ $RESULTS/
if [[ ! "$?" -eq 0 ]]; then
  echo "Error while copying output back home $(date +"%T")"
	exit 1
fi

echo "End abc $(date +"%T")"
