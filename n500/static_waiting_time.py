'''
Static Traffic Light Control Implementation
This script implements a static traffic light control strategy for comparison with the RL-based approach.
'''

import traci
import random
import os
import sys
import optparse
from sumolib import checkBinary

def calculate_total_waiting_time():
    """
    Calculate the total waiting time for all vehicles in the simulation.
    
    Returns:
        float: Total waiting time across all vehicles
    """
    total_waiting_time = 0
    vehicle_ids = traci.vehicle.getIDList()
    
    for vehicle_id in vehicle_ids:
        waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
        total_waiting_time += waiting_time
        
    return total_waiting_time

def run_static_control_simulation(max_steps, use_random=False, output_file=None):
    """
    Run a simulation with static traffic light control.
    
    Args:
        max_steps (int): Maximum number of simulation steps
        use_random (bool): Whether to use random traffic light phases
        output_file (str): Path to output file for logging results
        
    Returns:
        float: Total waiting time across the simulation
    """
    # Configure SUMO command
    sumo_cmd = ["sumo", "-c", "traffic_intersection.sumocfg", "-r", "traffic_routes.xml"]
    
    # Start SUMO simulation
    traci.start(sumo_cmd)
    
    # Initialize simulation variables
    step = 0
    total_waiting_time = 0
    
    # Define traffic light phases
    traffic_light_phases = [
        "GGggrrrrGGggrrrr",  # Phase 0: Green for horizontal direction
        "yyggrrrryyggrrrr",  # Phase 1: Yellow transition
        "rrGGrrrrrrGGrrrr",  # Phase 2: Green for vertical direction
        "rryyrrrrrryyrrrr",  # Phase 3: Yellow transition
        "rrrrGGggrrrrGGgg",  # Phase 4: Green for diagonal direction
        "rrrryyggrrrryygg",  # Phase 5: Yellow transition
        "rrrrrrGGrrrrrrGG",  # Phase 6: Green for other diagonal
        "rrrrrryyrrrrrryy"   # Phase 7: Yellow transition
    ]

    # Main simulation loop
    while step < max_steps:
        # Advance simulation by one step
        traci.simulationStep()

        # Set traffic light phase
        if use_random:
            # Randomly select traffic light phase
            current_phase = random.choice(traffic_light_phases)
            traci.trafficlight.setRedYellowGreenState("0", current_phase)
        else:
            # Use a fixed round-robin sequence
            current_phase = traffic_light_phases[step % len(traffic_light_phases)]
            traci.trafficlight.setRedYellowGreenState("0", current_phase)

        # Calculate waiting time at this step
        step_waiting_time = calculate_total_waiting_time()
        total_waiting_time += step_waiting_time

        step += 1
    
    # Close the SUMO simulation
    traci.close()

    # Prepare output message
    output_message = f"Total static waiting time: {total_waiting_time} over {max_steps} steps"
    print(output_message)

    # Save output to file if specified
    if output_file:
        with open(output_file, 'a') as f:
            f.write(output_message + '\n')

    return total_waiting_time

def get_simulation_options():
    """
    Parse command line options for the simulation.
    
    Returns:
        optparse.Values: Parsed command line options
    """
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                         default=False, help="Run the commandline version of SUMO")
    options, args = opt_parser.parse_args()
    return options

if __name__ == "__main__":
    # Get simulation options
    options = get_simulation_options()
    
    # Set SUMO binary based on options
    if options.nogui:
        sumo_binary = checkBinary('sumo')
    else:
        sumo_binary = checkBinary('sumo-gui')
    
    # Define simulation parameters
    max_simulation_steps = 500
    output_file_path = "static_waiting_time_output.txt"
    
    # Run simulation with random traffic light phases
    print("Running simulation with random traffic light phases...")
    random_waiting_time = run_static_control_simulation(
        max_simulation_steps, 
        use_random=True, 
        output_file=output_file_path
    )
    
    # Run simulation with fixed traffic light sequence
    print("\nRunning simulation with fixed traffic light sequence...")
    fixed_waiting_time = run_static_control_simulation(
        max_simulation_steps, 
        use_random=False, 
        output_file=output_file_path
    )
    
    # Print comparison
    print(f"\nComparison:")
    print(f"Random control waiting time: {random_waiting_time}")
    print(f"Fixed control waiting time: {fixed_waiting_time}")
    print(f"Difference: {abs(random_waiting_time - fixed_waiting_time)}")
