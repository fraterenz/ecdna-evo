#!/bin/bash

function remove_dot {
  echo "$1" | tr -d "."
}



for f in 1 1.9; do
  for g in 1 5 20; do
    name=f$(remove_dot $f)g$g
    echo $name

    # simulate
    cargo run -q --release --bin ecdna -- simulate \
      --cells 1000000 \
      --runs 2 \
      --rho1 $f \
      --distribution results/culture5timepoints/${g}copies.json \
      --dynamics nplus nminus \
      --samples 1000 1000 1000 1000 1000 1000 1000 1000 1000 \
      --sizes 1000000 1000000 1000000 1000000 1000000 1000000 1000000 1000000 1000000 \
      --culture \
      --patient culture5timepoints/$name

    # preprocess
    for id in 0 1 2 3 4 5 6 7 8; do
      cargo run -q --release --bin preprocess -- \
        culture5timepoints/${name} sample${id} 1000000 \
        --distribution results/culture5timepoints/${name}/1000samples1000000cells/${id}/ecdna/0.json
    done

    # abc
    cargo run -q --release --bin ecdna -- abc \
      -r 10000 --culture \
      --rho1-range 1 3 --rho2-range 1 \
      --delta1-range 0 --delta2-range 0 \
      --copies-range 1 30 \
      --patient results/preprocessed/culture5timepoints/${name}.json


  done
done

