#!/bin/bash
echo "--- CONFIGURANDO AMBIENTE DE EXECUÇÃO ---"

echo "--> Carregando ambiente ROS 2 Humble"
source /opt/ros/humble/setup.bash

echo "--> Carregando ambiente virtual"
source "/home/lygia/action_detection/src/venv/bin/activate"

echo "--- AMBIENTE PRONTO! ---"