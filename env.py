import math
import time 
import traci
import torch
import random
import gymnasium as gym
import numpy as np
from utils import convert_demand_to_scale_factor, scale_demand
from sim_config import get_direction_lookup, get_related_lanes_edges, get_intersection_phase_groups

class ControlEnv(gym.Env):
    """
    Parallelizable environment, includes features:  
    - Scaling the demand.
    - Tracking and occupancy map.
    """

    def __init__(self, control_args, worker_id=None):
        super().__init__()
        self.worker_id = worker_id
        self.vehicle_input_trips = control_args['vehicle_input_trips']
        self.vehicle_output_trips = control_args['vehicle_output_trips']
        self.pedestrian_input_trips = control_args['pedestrian_input_trips']
        self.pedestrian_output_trips = control_args['pedestrian_output_trips']
        self.manual_demand_veh = control_args['manual_demand_veh']
        self.manual_demand_ped = control_args['manual_demand_ped']
        self.demand_scale_min = control_args['demand_scale_min']
        self.demand_scale_max = control_args['demand_scale_max']
        self.step_length = control_args['step_length'] # SUMO allows to specify how long (in real world time) should each step be.
        self.action_duration = control_args['action_duration']
        self.max_timesteps = control_args['max_timesteps']
        self.use_gui = control_args['gui']
        self.auto_start = control_args['auto_start']
        self.sumo_running = False
        self.step_count = 0

        # Modify file paths to include the unique suffix. Each worker has their own environment and hence their own copy of the trips file.
        self.unique_suffix = f"_{worker_id}" if worker_id is not None else ""
        self.vehicle_output_trips = self.vehicle_output_trips.replace('.xml', f'{self.unique_suffix}.xml')
        self.pedestrian_output_trips = self.pedestrian_output_trips.replace('.xml', f'{self.unique_suffix}.xml')
        
        if self.manual_demand_veh is not None :
            scaling = convert_demand_to_scale_factor(self.manual_demand_veh, "vehicle", self.vehicle_input_trips) # Convert the demand to scaling factor first
            scale_demand(self.vehicle_input_trips, self.vehicle_output_trips, scaling, demand_type="vehicle")

        if self.manual_demand_ped is not None:
            scaling = convert_demand_to_scale_factor(self.manual_demand_ped, "pedestrian", self.pedestrian_input_trips)
            scale_demand(self.pedestrian_input_trips, self.pedestrian_output_trips, scaling, demand_type="pedestrian")

        self.tl_ids = ['cluster_172228464_482708521_9687148201_9687148202_#5more' # Intersection
                       '9727816850', # Mid block TL + crosswalks from left to right
                       '9727816623',
                       '9740157155',
                       'cluster_9740157181_9740483933',
                       '9740157194',
                       '9740157209',
                       '9740484527'] 
        
        self.previous_action = None
        # Number of simulation steps that should occur for each action. 
        self.steps_per_action = int(self.action_duration / self.step_length) # This is also one of the dimensions of the size of the observation buffer
        print(f"Steps per action: {self.steps_per_action}")

        self.current_action_step = 0 # To track where we are within the curret action's duration
        self.tl_phase_groups, self.crosswalk_phase_groups = get_intersection_phase_groups(self.action_duration)
        self.direction_and_edges  = get_direction_lookup()
        self.tl_lane_dict = get_related_lanes_edges()
        self.tl_pedestrian_status = {} # For pedestrians related to crosswalks attached to TLS.

        self.current_tl_phase_group = None
        self.current_crosswalk_actions = None
        self.current_tl_state_index = None 
        self.corrected_occupancy_map = None

        # Create a reverse lookup dict
        self.edge_to_direction = {}
        for direction, edges in self.direction_and_edges.items():
            for edge in edges:
                self.edge_to_direction[edge] = direction

        self.pressure_dict = {tl_id: {'vehicle': {}, 'pedestrian': {}} for tl_id in self.tl_ids}
        self.directions = ['north', 'east', 'south', 'west']
        self.turns = ['straight', 'right', 'left']

    def _get_vehicle_direction(self, signal_state):
        # Define signal bits for left and right blinkers
        VEH_SIGNAL_BLINKER_RIGHT = 0b1  # Bit 0
        VEH_SIGNAL_BLINKER_LEFT = 0b10  # Bit 1

        # Check if left blinker or right blinker is on
        left_blinker = bool(signal_state & VEH_SIGNAL_BLINKER_LEFT)
        right_blinker = bool(signal_state & VEH_SIGNAL_BLINKER_RIGHT)

        if left_blinker and not right_blinker:
            return "left"
        elif right_blinker and not left_blinker:
            return "right"
        else:
            # This covers cases where both blinkers are on (emergency) or off
            return "center"

    def _step_operations(self, occupancy_map, print_map=False, cutoff_distance=100):
        """
        Requires occupancy map as input. The changes made here should be reflected in the next time step's occupancy map.
        Some corrections have to be done every step.
        1. Update the pedestrian status when they cross: For each traffic light, check the outgoing pedestrians.
        If a pedestrian is in the outgoing area and hasn't been marked as 'crossed', update their status to 'crossed' in the self.tl_pedestrian_status dictionary.
        2. In case the same lanes are used for L, R, S turns (in case of vehicles and incoming). The straight lane will have repeated entries, remove them.  
        3. Vehicles are only included in the occupancy map if they are close to a given distance. In both incoming and outgoing directions.
        """
        # Handle outgoing pedestrians
        for tl_id in self.tl_ids:
            for _, persons in occupancy_map[tl_id]['pedestrian']['outgoing'].items():
                for person in persons:
                    if person not in self.tl_pedestrian_status or self.tl_pedestrian_status[person] != 'crossed':
                        # If the pedestrian crossed once, consider them as crossed (assume they wont cross twice, there is no way to know this without looking into their route, which is not practical.) 
                        self.tl_pedestrian_status[person] = 'crossed'

        # Handle special case for incoming vehicles
        for tl_id in self.tl_ids:
            for lane_group, vehicles in occupancy_map[tl_id]['vehicle']['incoming'].items():
                if lane_group not in ['south-straight', 'west-straight', 'east-straight', 'north-straight']:
                    ew_ns_direction = lane_group.split('-')[0]
                    straight_lane_group = f"{ew_ns_direction}-straight"
                    
                    # If this vehicle (which is in a non-straight lane) is also in the straight lane, remove it from the straight lane.
                    for vehicle in vehicles:
                        if vehicle in occupancy_map[tl_id]['vehicle']['incoming'][straight_lane_group]:
                            occupancy_map[tl_id]['vehicle']['incoming'][straight_lane_group].remove(vehicle) # Remove from the straight lane group
        
        # Handle vehicles
        for direction in ['incoming', 'outgoing']:
            for lane_group, vehicles in occupancy_map[tl_id]['vehicle'][direction].items():
                vehicles_to_remove = []
                for vehicle in vehicles:
                    distance = self._get_vehicle_distance_to_junction(tl_id, vehicle)
                    if distance > cutoff_distance:
                        vehicles_to_remove.append(vehicle)
                    
                # Remove vehicles outside the cutoff distance
                for vehicle in vehicles_to_remove:
                    occupancy_map[tl_id]['vehicle'][direction][lane_group].remove(vehicle)

        if print_map: # Corrected map
            print("\nOccupancy Map:")
            for id, data in occupancy_map.items():
                if id == 'crosswalks':
                    print(f"\nCrosswalks:")
                    for crosswalk_id, crosswalk_data in data.items():
                        print(f"    Crosswalk: {crosswalk_id}")
                        for direction, count in crosswalk_data.items():
                            print(f"        {direction.capitalize()}: {count}")
                else:
                    print(f"\nTraffic Light: {id}")
                    for agent_type in ["vehicle", "pedestrian"]:
                        print(f"  {agent_type.capitalize()}s:")
                        for direction in occupancy_map[id][agent_type].keys():
                            print(f"    {direction.capitalize()}:")
                            for lane_group, ids in data[agent_type][direction].items():
                                print(f"      {lane_group}: {len(ids)} [{', '.join(ids)}]")
                                if agent_type == "vehicle":
                                    for idx in ids:
                                        distance = self._get_vehicle_distance_to_junction(id, idx)
                                        print(f"        {idx}: {distance:.2f}m")   
            
        return occupancy_map

    def _get_vehicle_distance_to_junction(self, junction_id, vehicle_id):
        """
        Calculate the distance between a vehicle and a specific junction.

        :param junction_id: ID of the junction
        :param vehicle_id: ID of the vehicle
        :return: Distance between the vehicle and the junction in meters
        """
        try:
            # Get the x, y coordinates of the junction
            junction_pos = traci.junction.getPosition(junction_id)

            # Get the x, y coordinates of the vehicle
            vehicle_pos = traci.vehicle.getPosition(vehicle_id)

            # Calculate the Euclidean distance
            distance = math.sqrt(
                (junction_pos[0] - vehicle_pos[0])**2 + 
                (junction_pos[1] - vehicle_pos[1])**2
            )

            return distance

        except traci.TraCIException as e:
            print(f"Error calculating distance: {e}")
            return None
    
    def _update_pressure_dict(self, corrected_occupancy_map):
        """
        Update the data structure that holds info about pressure in outgoing directions.
        For both vehicles and pedestrians.

        For crosswalks, If the pedestrians are being rerouted, that means there is pressure that is not being addressed.
        Pressure = incoming (upside + downside) - outgoing (inside)
        However, if rerouted, then Pressure = -ve (rerouted)
        """

        for tl_id in self.tl_ids:
            #### VEHICLES ####

            # Initialize pressure and calculate for each direction
            vehicle_pressure = {d: 0 for d in self.directions}

            for outgoing_direction in self.directions:
                # Calculate incoming traffic towards this direction
                incoming = 0
                if outgoing_direction == 'north': # These four are outgoing directions
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['south-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['east-right'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['west-left'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['south-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['east-right'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['west-left'])

                elif outgoing_direction == 'south': # These four are outgoing directions
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['north-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['east-left'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['west-right'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['north-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['east-left'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['west-right'])

                elif outgoing_direction == 'east': # These four are outgoing directions
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['west-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['north-left'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['south-right'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['west-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['north-left'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['south-right'])

                elif outgoing_direction == 'west': # These four are outgoing directions
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['east-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['north-right'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['south-left'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['east-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['north-right'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['south-left'])
                
                # Calculate actual outgoing traffic
                outgoing = len(corrected_occupancy_map[tl_id]['vehicle']['outgoing'][outgoing_direction])
                
                # Calculate pressure
                vehicle_pressure[outgoing_direction] = incoming - outgoing
                self.pressure_dict[tl_id]['vehicle'][outgoing_direction] = vehicle_pressure[outgoing_direction]

            #### PEDESTRIANS (In this TL crossings) ####
            pedestrian_pressure = {d: 0 for d in self.directions}

            for outgoing_direction in self.directions:
                # Calculate incoming pedestrians towards this direction
                incoming = len(corrected_occupancy_map[tl_id]['pedestrian']['incoming'][outgoing_direction])
                
                # Calculate actual outgoing pedestrians
                outgoing = len(corrected_occupancy_map[tl_id]['pedestrian']['outgoing'][outgoing_direction])
                
                # Calculate pressure
                pedestrian_pressure[outgoing_direction] = incoming - outgoing
                self.pressure_dict[tl_id]['pedestrian'][outgoing_direction] = pedestrian_pressure[outgoing_direction]

        #### CROSSWALKS ####  
        for crosswalk_id in self.controlled_crosswalk_masked_ids:
            if crosswalk_id != ':9687187501_c0': # Special case crosswalk
                incoming = len(corrected_occupancy_map['crosswalks'][crosswalk_id]['upside']) + len(corrected_occupancy_map['crosswalks'][crosswalk_id]['downside'])
                outgoing = len(corrected_occupancy_map['crosswalks'][crosswalk_id]['inside'])
                rerouted = len(corrected_occupancy_map['crosswalks'][crosswalk_id]['rerouted'])

                if rerouted > 0:
                    self.pressure_dict['crosswalks'][crosswalk_id] = -rerouted
                else:
                    self.pressure_dict['crosswalks'][crosswalk_id] = incoming - outgoing

    def _get_occupancy_map(self, ):
        """
        Occupancy map = dict with vehicle and pedestrian information in the neighbourhood of all TLs
        If the same lane is used for multiple turns (straight, right, left), the indicator light (turns ON about 100m as a vehicle approaches the junction) of vehicle is used to determine the actual direction. 
        """

        occupancy_map = {}
        for tl_id in self.tl_ids:
            occupancy_map[tl_id] = {
                "vehicle": {
                    "incoming": {},
                    "inside": {}, # Inside the junction (since mid-block TLs are small, may not apply to them)
                    "outgoing": {}
                },
                "pedestrian": {
                    "incoming": {},
                    "outgoing": {}
                }
            }
            








            for agent_type in ["vehicle", "pedestrian"]:
                for direction in  occupancy_map[tl_id][agent_type].keys():
                    for lane_group, lane_list in lanes[agent_type][direction].items():
                        occupancy_map[tl_id][agent_type][direction][lane_group] = []
                        for lane in lane_list:
                            if agent_type == "vehicle":

                                if lane != '-1':  # Skip lanes that are common for all directions
                                    ids = traci.lane.getLastStepVehicleIDs(lane) if "edge" not in lane else traci.edge.getLastStepVehicleIDs(lane.split('.')[1]) # Its actually an edge in the else case.
                                    occupancy_map[tl_id][agent_type][direction][lane_group].extend(ids)
                                else: 
                                    # If there are multiple -1s, this case can occur multiple times.
                                    # In this case, look at the indicator light of the vehicle to get the direction.
                                    # Get the EW-NS direction and the current turn direction, then all vehicles in the straight lane group
                                    ew_ns_direction = lane_group.split('-')[0]
                                    turn_direction = lane_group.split('-')[1]

                                    straight_lane_group = f"{ew_ns_direction}-straight"

                                    # TODO: If there are multiple straight lanes where vehicles that want to go left or right also exist, then need to account for that
                                    straight_lane = lanes[agent_type][direction][straight_lane_group][0] 
                                    existing_ids = traci.lane.getLastStepVehicleIDs(straight_lane) if "edge" not in lane else traci.edge.getLastStepVehicleIDs(lane.split('.')[1])  # Its actually an edge in the else case.

                                    #print(f"Straight lane: {straight_lane}, Existing ids: {existing_ids}")

                                    if len(existing_ids)>0:
                                        #print(f"Vehicle exists")
                                        new_ids = []
                                        for veh_id in existing_ids:
                                            signal_state = traci.vehicle.getSignals(veh_id)
                                            veh_direction = self._get_vehicle_direction(signal_state)
                                            # print(f"Vehicle: {veh_id}, Signal: {signal_state}, Direction: {veh_direction}")
                                            if veh_direction == turn_direction:
                                                new_ids.append(veh_id)

                                        occupancy_map[tl_id][agent_type][direction][lane_group].extend(new_ids) 

                            else:  # pedestrian
                                if lane.startswith(':'):  # Check if it's an internal lane
                                    # Doing it the computationally expensive way
                                    # Get all persons in the simulation
                                    all_persons = traci.person.getIDList()
                                    # Filter persons on this junction
                                    for person in all_persons:
                                        if traci.person.getRoadID(person) == lane:
                                            # If not crossed yet, add to incoming 
                                            
                                            if direction == "incoming":
                                                if person not in self.tl_pedestrian_status or self.tl_pedestrian_status[person] != 'crossed': 
                                                    occupancy_map[tl_id][agent_type][direction][lane_group].append(person)
                                            else: 
                                                # Add to outgoing, just being inside the crossing is enough.
                                                occupancy_map[tl_id][agent_type][direction][lane_group].append(person)

                                else: 
                                    print("Only implemented to work with JunctionDomain. Not implemented yet for external lanes or edges")

        # For the crosswalks related components
        # For each crosswalk, get the occupancy in the upside, downside, inside, and rerouted.
        # Do we get the pedestrian ids or do we just get the count? # For now, let's get the ids.
        # If they are being-rerouted, they should not be a part of upside, downside, and inside. 
        occupancy_map['crosswalks'] = {}
        for crosswalk_id in self.controlled_crosswalk_masked_ids:
            occupancy_map['crosswalks'][crosswalk_id] = {
                "upside": [],
                "downside": [],
                "inside": [],
                "rerouted": [] }

            # These already contain internal edges
            vicinity_walking_edges = self.controlled_crosswalks[crosswalk_id]['vicinity_walking_edges']
        
            # For upside and downside
            for edge in vicinity_walking_edges:
                pedestrians = traci.edge.getLastStepPersonIDs(edge)
                direction = self.edge_to_direction[edge]
                occupancy_map['crosswalks'][crosswalk_id][direction].extend(pedestrians) # Incase ids are wanted, use the list instead of the length

            # For inside, use the crosswalk id itself.
            pedestrians = traci.edge.getLastStepPersonIDs(crosswalk_id)
            occupancy_map['crosswalks'][crosswalk_id]['inside'].extend(pedestrians)

            # Add re-routed pedestrians
            # If this crosswalk happens to be disabled, then add the upside and downside values to get the rerouted value. Setting upside, downside to 0.
            # Inside may contain pessengers that are in the process of crossing when the new decision is made. Not setting that to 0.
            if crosswalk_id in self.crosswalks_to_disable:
                occupancy_map['crosswalks'][crosswalk_id]['rerouted'] = occupancy_map['crosswalks'][crosswalk_id]['upside'] + occupancy_map['crosswalks'][crosswalk_id]['downside']
                occupancy_map['crosswalks'][crosswalk_id]['upside'] = []
                occupancy_map['crosswalks'][crosswalk_id]['downside'] = []

        # Special case: This id 'ids': [':9687187500_c0', ':9687187501_c0'] represents a single disjoint crosswalk. Do not repeat the counts twice.
        # Just use the 9687187500_c0 once and skip the second part of the special case crosswalk (only for upside and downside) because it would have been counted in the first part.
        # This is the special case crosswalk # Since this is updating for inside (it is not affected by re-routing)
        # Also not affected by other possible problems beacuse of the order [500 comes before 501 inthe crosswalks list]
        pedestrians = traci.edge.getLastStepPersonIDs(crosswalk_id)
        occupancy_map['crosswalks'][':9687187500_c0']['inside'].extend(occupancy_map['crosswalks'][':9687187501_c0']['inside']) # Use ':9687187500_c0'
        del occupancy_map['crosswalks'][':9687187501_c0'] # Delete the second part of the special case crosswalk

        # for crosswalk_id in self.controlled_crosswalk_masked_ids:
        #     if crosswalk_id != ':9687187501_c0':
        #         print(f"\nStep:{self.step_count}\n\nCrosswalk: {crosswalk_id}\nUpside: {occupancy_map['crosswalks'][crosswalk_id]['upside']}\nDownside: {occupancy_map['crosswalks'][crosswalk_id]['downside']}\nInside: {occupancy_map['crosswalks'][crosswalk_id]['inside']}\nRerouted: {occupancy_map['crosswalks'][crosswalk_id]['rerouted']}")
        return occupancy_map
    
    @property
    def action_space(self):
        """
        The control action is represented as a 9-bit string for each traffic light:

        - First two bits: Intersection signal (4 mutually exclusive configurations)
            00 = allow vehicular traffic through North-South only
            01 = allow vehicular traffic through East-West only
            10 = allow dedicated left turns through N-E and S-W only
            11 = disallow vehicular traffic in all directions

        - Next seven bits: Mid-block crosswalk signals (each bit is Bernoulli)
            1  = allow pedestrians to cross at a given mid-block segment
            0  = disallow pedestrians at that segment
        
        Returns:
            gym.spaces.MultiDiscrete: A 9-dimensional space of binary actions.
        """
        action_space = []
        # First 2 bits: intersection signal (4 configurations)
        action_space.extend([2, 2])
        # Next 7 bits: each mid-block crosswalk (on/off)
        action_space.extend([2] * 7)

        return gym.spaces.MultiDiscrete(action_space)

    @property
    def observation_space(self):
        """
        Observation is composed of: 
        - Current state information (9 bit string which is repeated steps_per_action times)
        - For Intersection and Mid-block: Advanced Traffic State (ATS)

        Observation space defined per action step (i.e. accumulation over a number of action duration steps)
        Intersection (40 x steps_per_action)
        * Vehicles
        - For each direction + turn ():
            - Incoming
            - In
        - Pedestrians

        Each midblock ()
        - 
        """
        # The observation is the entire observation buffer
        return gym.spaces.Box(
            low=0, 
            high=1, 
            shape=(self.steps_per_action, 40),
            dtype=np.float32
        )

    def step(self, action):
        """
        """
        if not self.sumo_running:
            raise Exception("Environment is not running. Call reset() to start the environment.")
        
        reward = 0
        done = False
        observation_buffer = []
        for _ in range(self.steps_per_action): # Run simulation steps for the duration of the action
            
            # Apply action needs to happen every timestep (TODO: return information useful for reward calculation)
            self._apply_action(action, self.current_action_step, self.previous_action)
            
            traci.simulationStep() # Step length is the simulation time that elapses when each time this is called.

            self.step_count += 1
            # Increment the current action step (goes from 0 to steps_per_action-1)
            self.current_action_step = (self.current_action_step + 1) % self.steps_per_action 

            # TODO: For the time being. Modify it later.
            obs = np.random.rand(40)
            observation_buffer.append(obs)

            #obs = self._get_observation(print_map=False)
            #print(f"\nObservation: {obs}")
            
            # TODO: For the time being. Modify it later.
            #self._update_pressure_dict(self.corrected_occupancy_map)

            # Accumulate reward
            # TODO: For the time being. Modify it later.
            #reward += self._get_reward(current_tl_action)
            reward += 0 

        # Check if episode is done (outside the for loop, otherwise it would create a broken observation)
        if self._check_done():
            done = True

        observation = np.asarray(observation_buffer, dtype=np.float32) 
        return observation, reward, done, False, {} # info is empty
        
    def _get_observation(self, print_map=False):
        """
        This is per step observation.
        About including previous action in the observation:
            - Each action persists for a number of timesteps and the observation is collected at each timestep.
            - Therefore, the previous action is the same for a number of timesteps. This adds too much extra computational overhead for the model. 
            - It would have been fine if model was MLP (we can just attach the previous action at the end)
            But for CNN, it breaks the grid structure.
        Pressure itself is not a part of the observation. It is only used for reward calculation.

        TODO: Some parts of the observation dont need normalization (current phase information).
        """
        
        # Get the occupancy map and print it
        occupancy_map = self._get_occupancy_map()
        self.corrected_occupancy_map = self._step_operations(occupancy_map, print_map=print_map, cutoff_distance=100)
        
        observation = []
        tl_id = self.tl_ids[0]
        
        #### Current phase group info (This changes even within the action timesteps) ####
        current_tl_info = []
        current_tl_info.append(self.current_tl_phase_group/4) # 0, 1, 2, 3 to 0, 0.25, 0.5, 0.75 
        current_tl_info.append(self.current_tl_state_index/2) # For 0, 1, 2, 3, its always 0 but for 4 and 5, it varies in 0, 1, 2; convert that to 0, 0.5, 1

        observation.extend(current_tl_info)
        observation.extend([float(x) for x in self.current_crosswalk_actions]) # 0 and 1 to 0.0 and 1.0
        
        #### VEHICLES INFO ####
        # Incoming
        for outgoing_direction in self.directions:
            for turn in self.turns:
                incoming = len(self.corrected_occupancy_map[tl_id]['vehicle']['incoming'][f"{outgoing_direction}-{turn}"])
                observation.append(incoming)

        # Inside
        for outgoing_direction in self.directions:
            for turn in self.turns:
                inside = len(self.corrected_occupancy_map[tl_id]['vehicle']['inside'][f"{outgoing_direction}-{turn}"])
                observation.append(inside)

        # Outgoing
        for outgoing_direction in self.directions:
            outgoing = len(self.corrected_occupancy_map[tl_id]['vehicle']['outgoing'][outgoing_direction])
            observation.append(outgoing)
            
        #### PEDESTRIANS INFO ####
        # Incoming
        for outgoing_direction in self.directions:
            incoming = len(self.corrected_occupancy_map[tl_id]['pedestrian']['incoming'][outgoing_direction])
            observation.append(incoming)

        # Outgoing
        for outgoing_direction in self.directions:
            outgoing = len(self.corrected_occupancy_map[tl_id]['pedestrian']['outgoing'][outgoing_direction])
            observation.append(outgoing)

        observation = np.asarray(observation, dtype=np.float32)
        return observation

    def _apply_action(self, action, current_action_step, previous_action=None):
        """
        apply_action is the enforcement of the chosen action (9-bit string) to the traffic lights and crosswalks, and will be called every step.
        previous_action will be None in reset.

        The function internally checks if a switch in vehicle directions that turn green.
        If a switch is detected, a 4-second mandatory yellow phase is enforced to the direction/ light that is turning red, before starting the new phase in the other light.
        """
        # Use previous action to detect signal switching
        if previous_action is None: # First action 
            previous_action = action  # Assume that there was no switch

        # Intersection action: integer in [0..3]
        current_intersection_action = action[0:2].item()
        previous_intersection_action = previous_action[0:2].item()

        # for crosswalk signals as an example
        current_mid_block_action = action[2:]  
        previous_mid_block_action = previous_action[2:]

        # Detect a switch in all the components (the vehicle part at the intersection considered as one component) i.e., total 8 components
        switches = [] # contains boolean values (0 = No switch, 1 = Switch)
        if previous_intersection_action == 11: # If the last phase was all red, assume no switch (no intermediate yellow required)
            switch_in_intersection = 0
        else:
            if current_intersection_action != previous_intersection_action:
                switch_in_intersection = 1
            else:
                switch_in_intersection = 0

        switches.append(switch_in_intersection)
        # For midblock, there is a switch if the previous action is not the same as the current action
        switch_in_midblock = (current_mid_block_action != previous_mid_block_action)
        switches.append(switch_in_midblock)

        # Mandatory yellow enforcement
        # if any(switches):
        #     # Insert a 4-second yellow transition between switching states
        #     tl_state = self._get_tl_switch_state(east_to_north_switch, north_to_east_switch, current_action_step)

        # # For the signalized crosswalk control. Append ArBCrD at the end of the tl state string.
        # self.current_crosswalk_actions = str(action[1].item()) + str(action[2].item()) # two binary actions 0, 1
        # crosswalk_state = self.crosswalk_phase_groups[self.current_crosswalk_actions]
        
        # # Construct the crosswalk state string from the dict values
        # crosswalk_state_str = (crosswalk_state['A'] + crosswalk_state['B'] + 'r' + crosswalk_state['C'] + 'r' + crosswalk_state['D'])
        
        # state = tl_state + crosswalk_state_str
        # #print(f"\nState: {state}\n")
        # traci.trafficlight.setRedYellowGreenState(self.tl_ids[0], state)

        self.previous_action = action

    def _get_reward(self, current_tl_action):
        """ 
        * Reward is based on the alleviation of pressure (penalize high pressure)
        * Intersection:
            - Vehicle pressure = Incoming - Outgoing 
            - Pedestrian pressure = Incoming (upside + downside) - Outgoing (inside crosswalk)
        * Mid-block:
            - Same as intersection
        Other components: 
            - Penalize frequent changes of action/ switching

        # TODO: 
        0. Get the lambda values from wandb config.
        1. Alternatively Maximum wait aggregated queue (mwaq) can also be used for reward. 
            mwaq = (sum of queue lengths of all directions) x maximum waiting time among all 
            Can be used for both vehicle and pedestrian. Penalize high mwaq.
        """

        reward = 0
        lambda1, lambda2, lambda3 = -0.33, -0.33, -0.33

        # Intersection
        vehicle_pressure = 0
        for tl_id in self.tl_ids:
            for direction in self.directions:
                vehicle_pressure += self.pressure_dict[tl_id]['vehicle'][direction]

        # Crosswalk Signal Control
        pedestrian_pressure = 0
        for tl_id in self.tl_ids:
            for direction in self.directions:
                pedestrian_pressure += self.pressure_dict[tl_id]['pedestrian'][direction]

        #### MWAQ based ####
        # TODO:: Implement this and add to sweep.

        # Crosswalk control
        crosswalks_pressure = 0
        controlled_crosswalk_pressures = []
        for crosswalk_id in self.controlled_crosswalk_masked_ids:
            if crosswalk_id != '9687187501_c0':
                controlled_crosswalk_pressures.append(self.pressure_dict['crosswalks'][crosswalk_id]) 

        # Only collect the positive pressure values. A negative value means re-routed i.e., the pressure was discarded.
        crosswalks_pressure = sum(pressure for pressure in controlled_crosswalk_pressures if pressure > 0)
                
        reward = lambda1*vehicle_pressure + lambda2*pedestrian_pressure + lambda3*crosswalks_pressure


        # Corridor
        # Crosswalk control

        # Other general components
        # Frequency penalty
        if self.previous_action is not None and current_tl_action != self.previous_action:
            reward -= 0.5  # Penalty for changing tl actions. Since this is per step reward. Action change is reflected multiplied by action steps.
        
        # Re-route penalty (because any re-route increases travel time). Only collect the negative pressure values
        reroute_pressure = sum(pressure for pressure in controlled_crosswalk_pressures if pressure < 0) # This is already negative.
        reward -= reroute_pressure

        #print(f"\nStep Reward: {reward}")
        return reward

    def _check_done(self):
        """
        TODO: What more conditions can be added here?
        - Gridlock? Jam? of vehicles or pedestrians? Crashes?
        """
        return self.step_count >= self.max_timesteps

    def reset(self):
        """
        """
        super().reset()
        if self.sumo_running:
            time.sleep(5) # Wait until the process really finishes 
            traci.close(False) #https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html
        
        # Automatically scale demand (separately for pedestrian and vehicle)
        scale_factor_vehicle = random.uniform(self.demand_scale_min, self.demand_scale_max)
        scale_factor_pedestrian = random.uniform(self.demand_scale_min, self.demand_scale_max)

        scale_demand(self.vehicle_input_trips, self.vehicle_output_trips, scale_factor_vehicle, demand_type="vehicle")
        scale_demand(self.pedestrian_input_trips, self.pedestrian_output_trips, scale_factor_pedestrian, demand_type="pedestrian")

        if self.auto_start:
            sumo_cmd = ["sumo-gui" if self.use_gui else "sumo", 
                        "--verbose",
                        "--start" , 
                        "--quit-on-end", 
                        "-c", "./SUMO_files/Craver_traffic_lights.sumocfg", 
                        "--step-length", str(self.step_length),
                        "--route-files", f"{self.vehicle_output_trips},{self.pedestrian_output_trips}"
                        ]
                        
        else:
            sumo_cmd = ["sumo-gui" if self.use_gui else "sumo", 
                        "--verbose",
                        "--quit-on-end", 
                        "-c", "./SUMO_files/Craver_traffic_lights.sumocfg", 
                        "--step-length", str(self.step_length),
                        "--route-files", f"{self.vehicle_output_trips},{self.pedestrian_output_trips}"
                        ]
                        
        max_retries = 3
        try:
            for attempt in range(max_retries):
                try:
                    traci.start(sumo_cmd)
                    break
                except traci.exceptions.FatalTraCIError:
                    if attempt < max_retries - 1:
                        print(f"TraCI connection failed. Retrying... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(5)
                    else:
                        print(f"Failed to start TraCI after {max_retries} attempts.")
                        raise
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            raise

        self.sumo_running = True
        self.step_count = 0 # This counts the timesteps in an episode. Needs reset.
        self.current_action_step = 0
        self.tl_lane_dict = get_tl_related_lanes()

        # Randomly initialize the actions (current tl phase group and combined binary action for crosswalks) 
        self.current_tl_phase_group = random.choice([0, 1, 2, 3]) # not including [4, 5] from the list
        self.current_crosswalk_actions = str(random.randint(0, 1)) + str(random.randint(0, 1))

        action_list = [int(x) for x in str(self.current_tl_phase_group) + self.current_crosswalk_actions]
        initial_action = torch.tensor(action_list, dtype=torch.long)
        print(f"\nInitial action: {initial_action}\n")

        # Initialize the observation buffer
        observation_buffer = []
        for step in range(self.steps_per_action):
            # Apply the current phase group using _apply_action
            self._apply_action(initial_action, step, None)
            traci.simulationStep()

            # TODO: For the time being. Modify it later.
            obs = np.random.rand(40)
            observation_buffer.append(obs)

            #obs = self._get_observation()
            
        observation = np.asarray(observation_buffer, dtype=np.float32)
        #print(f"\nObservation (in reset): {observation.shape}")
        return observation, {} # nfo is empty

    def close(self):
        if self.sumo_running:
            traci.close(False) #https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html
            self.sumo_running = False