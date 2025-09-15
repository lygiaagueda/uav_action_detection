# uav_action_detection

## Detecção de ação
Este pacote tem como objetivo detectar ações realizadas por um humano, que podem ser utilizadas para controlar um drone.

As ações previstas podem ser uma das seguintes classes:
- hover
- land
- all_clear
- have_command
- landing_direction
- move_ahead
- move_downward
- move_to_left
- move_to_right
- move_upward
- not_clear
- slow_down
- wave_off

Mais detalhes sobre cada classe podem ser encontrados em [UAV-GESTURE: A Dataset for UAV Control and Gesture Recognition](https://asankagp.github.io/uavgesture/).

Além disso, o pacote pode retornar Pose Incompleta quando nem todos os pontos do corpo humano de interesse forem detectados.

## Instalação de bibliotecas
Instale o opencv-bridge

```
sudo apt-get install ros-humble-cv-bridge
```

## Preparação do ambiente Python

Para executar o pacote, é necessário ter instalado o ROS 2 Humble e criar um ambiente virtual Python:

```
cd ~/uav_action_detection/src
python3 -m venv venv
```

Em seguida, ative o ambiente virtual e instale as bibliotecas necessárias:

```
source venv/bin/activate
pip install torch opencv-python mediapipe
pip install --upgrade pip setuptools
```

Após instalar as bibliotecas, ajuste os caminhos nos arquivos `setup_env.sh` e `src/scripts/detector_node`. Com isso, o ambiente estará pronto para execução.


## Execução do pacote
Primeiro, configure os paths:

```
cd ~/uav_action_detection/
source setup_env.sh
```

Depois, compile o pacote:

```
colcon build -- packages-select detector
```

E execute utilizando o launch:

```
ros2 launch detector detector.launch.py
```

