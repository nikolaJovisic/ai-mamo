#!/bin/bash

# Default to 'local' if no flag is passed
ENV="local"

# Check for --<env> flag
if [[ "$1" =~ ^--(.+)$ ]]; then
  ENV="${BASH_REMATCH[1]}"
fi

CONFIG_SOURCE="config.${ENV}.json"
CONFIG_DEST="servers/config.json"

if [[ ! -f "$CONFIG_SOURCE" ]]; then
  echo "Error: Config file '$CONFIG_SOURCE' does not exist."
  exit 1
fi

echo "Using config: $CONFIG_SOURCE"

cp "$CONFIG_SOURCE" "$CONFIG_DEST"

external_port=$(jq '.external_port' $CONFIG_DEST)
internal_port=$(jq '.internal_port' $CONFIG_DEST)

docker build . -t aimamo

docker kill aim
docker rm aim
docker run --name aim -p ${external_port}:${internal_port} aimamo 
