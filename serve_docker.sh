#!/bin/bash

config_file="servers/config.json"

external_port=$(jq '.external_port' $config_file)
internal_port=$(jq '.internal_port' $config_file)

docker build . -t aimamo

docker kill aim
docker rm aim
docker run --name aim -p ${external_port}:${internal_port} aimamo
