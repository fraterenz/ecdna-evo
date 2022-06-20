#!/bin/bash
# Synthetic data abc inference with the whole population (no subsampling).

function remove_dot {
  echo "$1" | tr -d "."
}

RUNS=10000
CELLS=1000000
PATH2ECDNA=$HOME/ecdna-evo
RESULTS=${PATH2ECDNA}/results
SAMPLE=1000000
id=0

source ~/venvs/ecdna-evo/bin/activate

# ranges for abc
F1=1
F2=3
D1=0
D2=0

# ground truth
for f in 1 1.9 2.8; do
  for g in 10; do
    name=f$(remove_dot $f)g$g
    EXPERIMENT=whole_population/${name}
    PATH2EXPERIMENT=${RESULTS}/${EXPERIMENT}/${SAMPLE}samples${CELLS}cells

    echo $name

    # simulate
    ${HOME}/ecdna-v0.3.2-x86_64-unknown-linux-gnu/ecdna simulate \
      --cells ${CELLS} \
      --runs 1 \
      --rho1 $f \
      --distribution ${RESULTS}/distributions/${g}copies.json \
      --dynamics nplus nminus mean \
    	--patient ${EXPERIMENT} \
    	--savedir ${PATH2ECDNA}

    # plot
    python3 -m ecdna-plot.dynamics \
      --nplus ${PATH2EXPERIMENT}/nplus.tar.gz \
      --nminus ${PATH2EXPERIMENT}/nminus.tar.gz \
      --mean ${PATH2EXPERIMENT}/mean_dynamics.tar.gz \
      --save ${PATH2EXPERIMENT}

    if [[ ! "$?" -eq 0 ]]; then
    	echo "Error while generating the ground truth data $(date +"%T")"
    	exit 1
    fi

    # preprocess
    ${HOME}/ecdna-v0.3.2-x86_64-unknown-linux-gnu/ecdna preprocess ${EXPERIMENT} sample${id} ${CELLS} \
      --distribution ${PATH2EXPERIMENT}/${id}/ecdna/0.json

    if [[ ! "$?" -eq 0 ]]; then
    	echo "Error while preprocessing the ground truth data $(date +"%T")"
    	exit 1
    fi

    # abc
    ${HOME}/ecdna-v0.3.2-x86_64-unknown-linux-gnu/ecdna abc \
      -r ${RUNS} --culture \
      --rho1-range 1 3 --rho2-range 1 \
      --delta1-range 0 --delta2-range 0 \
      --copies-range 1 30 \
    	--patient ${RESULTS}/preprocessed/${EXPERIMENT}.json \

    if [[ ! "$?" -eq 0 ]]; then
      echo "Error while running ABC $(date +"%T")"
    	exit 1
    else
    	echo "End simulations for ${name} $(date +"%T")"
    fi

    echo "End abc $(date +"%T")"
  done
done
