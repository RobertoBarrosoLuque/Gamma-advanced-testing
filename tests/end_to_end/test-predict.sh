#!/bin/bash
BASE_DIR=$(git rev-parse --show-toplevel)

cd $BASE_DIR && python -m src.main --predict --input="$(< data/sample.json)"