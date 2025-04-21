'''
Traffic Light Control using Deep Q-Learning
This implementation uses a DQN agent to optimize traffic light control at intersections.
'''

from __future__ import absolute_import
from __future__ import print_function
from sumolib import checkBinary

import os
import sys
import optparse
import subprocess
import random
import traci
import numpy as np
import keras
import h5py
from collections import deque
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model


class TrafficLightAgent:
    def __init__(self):
        self.discount_factor = 0.95   # discount rate
        self.exploration_rate = 0.1  # exploration rate
        self.learning_rate = 0.0002
        self.experience_buffer = deque(maxlen=200)
        self.model = self._build_model()
        self.action_space = 2

    def _build_model(self):
        # Neural Network for Deep Q-Learning
        # Input 1: Position matrix
        position_input = Input(shape=(12, 12, 1))
        pos_conv1 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(position_input)
        pos_conv2 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(pos_conv1)
        pos_flat = Flatten()(pos_conv2)

        # Input 2: Velocity matrix
        velocity_input = Input(shape=(12, 12, 1))
        vel_conv1 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(velocity_input)
        vel_conv2 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(vel_conv1)
        vel_flat = Flatten()(vel_conv2)

        # Input 3: Traffic light state
        light_input = Input(shape=(2, 1))
        light_flat = Flatten()(light_input)

        # Combine all inputs
        combined = keras.layers.concatenate([pos_flat, vel_flat, light_flat])
        hidden1 = Dense(128, activation='relu')(combined)
        hidden2 = Dense(64, activation='relu')(hidden1)
        output = Dense(2, activation='linear')(hidden2)

        # Create and compile model
        model = Model(inputs=[position_input, velocity_input, light_input], outputs=[output])
        model.compile(optimizer=keras.optimizers.RMSprop(
            learning_rate=self.learning_rate), loss='mse')

        return model

    def store_experience(self, state, action, reward, next_state, done):
        self.experience_buffer.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_space)
        action_values = self.model.predict(state)
        return np.argmax(action_values[0])

    def train_on_batch(self, batch_size):
        minibatch = random.sample(self.experience_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.discount_factor *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def load_weights(self, name):
        self.model.load_weights(name)

    def save_weights(self, name):
        self.model.save_weights(name)


class TrafficSimulation:
    def __init__(self):
        # Import SUMO tools
        try:
            sys.path.append(os.path.join(os.path.dirname(
                __file__), '..', '..', '..', '..', "tools"))
            sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
                os.path.dirname(__file__), "..", "..", "..")), "tools"))
            from sumolib import checkBinary
        except ImportError:
            sys.exit(
                "Please set the SUMO_HOME environment variable to the root directory of your SUMO installation")

    def generate_traffic_routes(self):
        random.seed(42)  # For reproducibility
        simulation_steps = 500  # Number of simulation steps
        
        # Traffic generation probabilities
        prob_horizontal = 1. / 7
        prob_vertical = 1. / 11
        prob_right_turn = 1. / 30
        prob_left_turn = 1. / 25
        
        with open("traffic_routes.xml", "w") as routes:
            print('''<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="SUMO_DEFAULT_TYPE" accel="0.8" decel="4.5" sigma="0" length="5" minGap="2" maxSpeed="70"/>
    <route id="always_right" edges="1fi 1si 4o 4fi 4si 2o 2fi 2si 3o 3fi 3si 1o 1fi"/>
    <route id="always_left" edges="3fi 3si 2o 2fi 2si 4o 4fi 4si 1o 1fi 1si 3o 3fi"/>
    <route id="horizontal" edges="2fi 2si 1o 1fi 1si 2o 2fi"/>
    <route id="vertical" edges="3fi 3si 4o 4fi 4si 3o 3fi"/>

    ''', file=routes)
            vehicle_count = 0
            last_vehicle_time = 0
            for step in range(simulation_steps):
                if random.uniform(0, 1) < prob_horizontal:
                    print('    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="horizontal" depart="%i" />' % (
                        vehicle_count, step), file=routes)
                    vehicle_count += 1
                    last_vehicle_time = step
                if random.uniform(0, 1) < prob_vertical:
                    print('    <vehicle id="left_%i" type="SUMO_DEFAULT_TYPE" route="vertical" depart="%i" />' % (
                        vehicle_count, step), file=routes)
                    vehicle_count += 1
                    last_vehicle_time = step
                if random.uniform(0, 1) < prob_left_turn:
                    print('    <vehicle id="down_%i" type="SUMO_DEFAULT_TYPE" route="always_left" depart="%i" color="1,0,0"/>' % (
                        vehicle_count, step), file=routes)
                    vehicle_count += 1
                    last_vehicle_time = step
                if random.uniform(0, 1) < prob_right_turn:
                    print('    <vehicle id="down_%i" type="SUMO_DEFAULT_TYPE" route="always_right" depart="%i" color="1,0,0"/>' % (
                        vehicle_count, step), file=routes)
                    vehicle_count += 1
                    last_vehicle_time = step
            print("</routes>", file=routes)

    def get_simulation_options(self):
        optParser = optparse.OptionParser()
        optParser.add_option("--nogui", action="store_true",
                             default=False, help="Run the commandline version of SUMO")
        options, args = optParser.parse_args()
        return options

    def get_current_state(self):
        # Initialize matrices for position and velocity
        position_matrix = []
        velocity_matrix = []

        # Simulation parameters
        cell_length = 7
        offset_distance = 11
        max_speed = 14

        # Get junction position
        junction_position_x = traci.junction.getPosition('0')[0]
        junction_position_y = traci.junction.getPosition('0')[1]
        
        # Get vehicles on each road
        vehicles_road1 = traci.edge.getLastStepVehicleIDs('1si')
        vehicles_road2 = traci.edge.getLastStepVehicleIDs('2si')
        vehicles_road3 = traci.edge.getLastStepVehicleIDs('3si')
        vehicles_road4 = traci.edge.getLastStepVehicleIDs('4si')
        
        # Initialize matrices with zeros
        for i in range(12):
            position_matrix.append([])
            velocity_matrix.append([])
            for j in range(12):
                position_matrix[i].append(0)
                velocity_matrix[i].append(0)

        # Process vehicles on road 1
        for vehicle in vehicles_road1:
            cell_index = int(
                abs((junction_position_x - traci.vehicle.getPosition(vehicle)[0] - offset_distance)) / cell_length)
            if(cell_index < 12):
                position_matrix[2 - traci.vehicle.getLaneIndex(vehicle)][11 - cell_index] = 1
                velocity_matrix[2 - traci.vehicle.getLaneIndex(
                    vehicle)][11 - cell_index] = traci.vehicle.getSpeed(vehicle) / max_speed

        # Process vehicles on road 2
        for vehicle in vehicles_road2:
            cell_index = int(
                abs((junction_position_x - traci.vehicle.getPosition(vehicle)[0] + offset_distance)) / cell_length)
            if(cell_index < 12):
                position_matrix[3 + traci.vehicle.getLaneIndex(vehicle)][cell_index] = 1
                velocity_matrix[3 + traci.vehicle.getLaneIndex(
                    vehicle)][cell_index] = traci.vehicle.getSpeed(vehicle) / max_speed

        # Process vehicles on road 3
        for vehicle in vehicles_road3:
            cell_index = int(
                abs((junction_position_y - traci.vehicle.getPosition(vehicle)[1] - offset_distance)) / cell_length)
            if(cell_index < 12):
                position_matrix[6 + 2 -
                               traci.vehicle.getLaneIndex(vehicle)][11 - cell_index] = 1
                velocity_matrix[6 + 2 - traci.vehicle.getLaneIndex(
                    vehicle)][11 - cell_index] = traci.vehicle.getSpeed(vehicle) / max_speed

        # Process vehicles on road 4
        for vehicle in vehicles_road4:
            cell_index = int(
                abs((junction_position_y - traci.vehicle.getPosition(vehicle)[1] + offset_distance)) / cell_length)
            if(cell_index < 12):
                position_matrix[9 + traci.vehicle.getLaneIndex(vehicle)][cell_index] = 1
                velocity_matrix[9 + traci.vehicle.getLaneIndex(
                    vehicle)][cell_index] = traci.vehicle.getSpeed(vehicle) / max_speed

        # Get traffic light state
        traffic_light_state = []
        if(traci.trafficlight.getPhase('0') == 4):
            traffic_light_state = [1, 0]
        else:
            traffic_light_state = [0, 1]

        # Reshape matrices for neural network input
        position = np.array(position_matrix)
        position = position.reshape(1, 12, 12, 1)

        velocity = np.array(velocity_matrix)
        velocity = velocity.reshape(1, 12, 12, 1)

        lights = np.array(traffic_light_state)
        lights = lights.reshape(1, 2, 1)

        return [position, velocity, lights]


if __name__ == '__main__':
    # Initialize simulation
    traffic_sim = TrafficSimulation()
    
    # Get simulation options
    options = traffic_sim.get_simulation_options()

    # Set SUMO binary based on options
    if options.nogui:
        sumo_binary = checkBinary('sumo')
    else:
        sumo_binary = checkBinary('sumo-gui')
    
    # Generate traffic routes
    traffic_sim.generate_traffic_routes()

    # Training parameters
    num_episodes = 10
    batch_size = 8

    # Initialize agent
    agent = TrafficLightAgent()
    
    # Try to load pre-trained weights
    try:
        agent.load_weights('Models/reinf_traf_control.h5')
        print("Loaded pre-trained model")
    except:
        print('No pre-trained models found, starting from scratch')

    # Training loop
    for episode in range(num_episodes):
        # Initialize episode variables
        log_file = open('log.txt', 'a')
        step_count = 0
        total_waiting_time = 0
        reward_horizontal = 0
        reward_vertical = 0
        total_reward = reward_horizontal - reward_vertical
        simulation_step = 0
        selected_action = 0

        # Start SUMO simulation
        traci.start([sumo_binary, "-c", "traffic_intersection.sumocfg", '--start'])
        traci.trafficlight.setPhase("0", 0)
        traci.trafficlight.setPhaseDuration("0", 200)
        
        # Main simulation loop
        while traci.simulation.getMinExpectedNumber() > 0 and simulation_step < 600:
            traci.simulationStep()
            current_state = traffic_sim.get_current_state()
            selected_action = agent.select_action(current_state)
            light_state = current_state[2]

            # Handle action 0 (horizontal green)
            if(selected_action == 0 and light_state[0][0][0] == 0):
                # Transition Phase
                for i in range(6):
                    simulation_step += 1
                    traci.trafficlight.setPhase('0', 1)
                    total_waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()
                for i in range(10):
                    simulation_step += 1
                    traci.trafficlight.setPhase('0', 2)
                    total_waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()
                for i in range(6):
                    simulation_step += 1
                    traci.trafficlight.setPhase('0', 3)
                    total_waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()

                # Action Execution
                reward_horizontal = traci.edge.getLastStepVehicleNumber(
                    '1si') + traci.edge.getLastStepVehicleNumber('2si')
                reward_vertical = traci.edge.getLastStepHaltingNumber(
                    '3si') + traci.edge.getLastStepHaltingNumber('4si')
                for i in range(10):
                    simulation_step += 1
                    traci.trafficlight.setPhase('0', 4)
                    reward_horizontal += traci.edge.getLastStepVehicleNumber(
                        '1si') + traci.edge.getLastStepVehicleNumber('2si')
                    reward_vertical += traci.edge.getLastStepHaltingNumber(
                        '3si') + traci.edge.getLastStepHaltingNumber('4si')
                    total_waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()

            # Handle action 0 when already in horizontal green phase
            if(selected_action == 0 and light_state[0][0][0] == 1):
                # Action Execution, no state change
                reward_horizontal = traci.edge.getLastStepVehicleNumber(
                    '1si') + traci.edge.getLastStepVehicleNumber('2si')
                reward_vertical = traci.edge.getLastStepHaltingNumber(
                    '3si') + traci.edge.getLastStepHaltingNumber('4si')
                for i in range(10):
                    simulation_step += 1
                    traci.trafficlight.setPhase('0', 4)
                    reward_horizontal += traci.edge.getLastStepVehicleNumber(
                        '1si') + traci.edge.getLastStepVehicleNumber('2si')
                    reward_vertical += traci.edge.getLastStepHaltingNumber(
                        '3si') + traci.edge.getLastStepHaltingNumber('4si')
                    total_waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()

            # Handle action 1 (vertical green) when in horizontal phase
            if(selected_action == 1 and light_state[0][0][0] == 0):
                # Action Execution, no state change
                reward_horizontal = traci.edge.getLastStepVehicleNumber(
                    '4si') + traci.edge.getLastStepVehicleNumber('3si')
                reward_vertical = traci.edge.getLastStepHaltingNumber(
                    '2si') + traci.edge.getLastStepHaltingNumber('1si')
                for i in range(10):
                    simulation_step += 1
                    reward_horizontal += traci.edge.getLastStepVehicleNumber(
                        '4si') + traci.edge.getLastStepVehicleNumber('3si')
                    reward_vertical += traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('1si')
                    traci.trafficlight.setPhase('0', 0)
                    total_waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()

            # Handle action 1 when already in vertical green phase
            if(selected_action == 1 and light_state[0][0][0] == 1):
                for i in range(6):
                    simulation_step += 1
                    traci.trafficlight.setPhase('0', 5)
                    total_waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()
                for i in range(10):
                    simulation_step += 1
                    traci.trafficlight.setPhase('0', 6)
                    total_waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()
                for i in range(6):
                    simulation_step += 1
                    traci.trafficlight.setPhase('0', 7)
                    total_waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()

                reward_horizontal = traci.edge.getLastStepVehicleNumber(
                    '4si') + traci.edge.getLastStepVehicleNumber('3si')
                reward_vertical = traci.edge.getLastStepHaltingNumber(
                    '2si') + traci.edge.getLastStepHaltingNumber('1si')
                for i in range(10):
                    simulation_step += 1
                    traci.trafficlight.setPhase('0', 0)
                    reward_horizontal += traci.edge.getLastStepVehicleNumber(
                        '4si') + traci.edge.getLastStepVehicleNumber('3si')
                    reward_vertical += traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('1si')
                    total_waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()

            # Get next state and calculate reward
            next_state = traffic_sim.get_current_state()
            episode_reward = reward_horizontal - reward_vertical
            agent.store_experience(current_state, selected_action, episode_reward, next_state, False)
            
            # Train the agent if enough experiences are collected
            if(len(agent.experience_buffer) > batch_size):
                agent.train_on_batch(batch_size)

        # End of episode processing
        last_experience = agent.experience_buffer[-1]
        del agent.experience_buffer[-1]
        agent.experience_buffer.append((last_experience[0], last_experience[1], episode_reward, last_experience[3], True))
        
        # Log episode results
        log_file.write('episode - ' + str(episode) + ', total waiting time - ' +
                  str(total_waiting_time) + ', static waiting time - 60433 \n')
        log_file.close()
        print('episode - ' + str(episode) + ' total waiting time - ' + str(total_waiting_time))
        
        # Save model weights
        agent.save_weights('reinf_traf_control_' + str(episode) + '.weights.h5')
        traci.close(wait=False)

sys.stdout.flush()
