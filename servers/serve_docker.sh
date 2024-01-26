#!/bin/bash

config="prod"
config_file="configs/${config}.json"

rest_port=$(jq '.rest_port' $config_file)
dcm_port=$(jq '.dcm_port' $config_file)
rest_port_mapping=$(jq '.rest_port_mapping' $config_file)
dcm_port_mapping=$(jq '.dcm_port_mapping' $config_file)

docker build . -t aimamo

docker kill aim
docker rm aim
docker run --name aim -p ${rest_port}:${rest_port_inside} -p ${dcm_port}:${dcm_port_inside} aimamo
