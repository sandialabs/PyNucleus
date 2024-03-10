#!/bin/bash
set -e

mkdir -p ~/examples
mkdir -p ~/drivers
cp -r --update=none /pynucleus/examples/* ~/examples
cp -r --update=none /pynucleus/drivers/* ~/drivers

jupyter notebook --port=8889 --no-browser --allow-root --ip=0.0.0.0 \
        --NotebookApp.token='' --NotebookApp.password='' \
        --notebook-dir=/root  > /dev/null 2>&1 &

exec "$@"
