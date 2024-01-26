#!/bin/bash

config="dev"

if [ -n "$1" ]; then
    param1="$1"
fi

cp ./configs/"${config}.json" ./config.json
cp ./configs/"${config}.json" ./mamo-front/src/config.json

python servers/serve_dcm.py &
python servers/serve_rest.py