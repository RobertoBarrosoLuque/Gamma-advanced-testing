#!/bin/bash
BASE_DIR=$(git rev-parse --show-toplevel)
echo $BASE_DIR

cd $BASE_DIR && python -m src.main --train && python -m src.main --train -hp
