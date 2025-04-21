# Traffic Light Control using Reinforcement Learning

This project implements an intelligent traffic light control system using Deep Reinforcement Learning (Deep Q-Learning) with the SUMO traffic simulator.

## Project Overview

The project uses a Deep Q-Network (DQN) agent to learn optimal traffic light control strategies. It includes both static and reinforcement learning-based implementations for traffic signal control.

## Features

- Deep Q-Learning based traffic light control
- Integration with SUMO traffic simulator
- Real-time traffic state monitoring
- Configurable traffic scenarios
- Both static and dynamic traffic control implementations

## Prerequisites

- Python 3.7+
- SUMO (Simulation of Urban MObility)
- TensorFlow
- Keras
- NumPy
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone "https://github.com/Melsa16/Traffic-Light-Control-using-Reinforcement-Learning.git"
cd Traffic-Light-Control-using-Reinforcement-Learning
```

2. Install SUMO:
   - Follow the installation instructions at [SUMO Documentation](https://sumo.dlr.de/docs/Downloads.php)

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `n500/` - Main implementation directory
  - `traffic_light_control.py` - Core implementation of the DQN agent and traffic control
  - `static_waiting_time.py` - Static traffic control implementation
  - Configuration files for SUMO simulation

## Usage

1. Configure your traffic scenario in the SUMO configuration files
2. Run the reinforcement learning training:
```bash
python n500/traffic_light_control.py
```

3. For static traffic control:
```bash
python n500/static_waiting_time.py
```

## Results

The system has been trained on various traffic scenarios and shows improved traffic flow compared to traditional static traffic light control.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
