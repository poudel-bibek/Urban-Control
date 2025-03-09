import math
import time 
import traci
import torch
import random
import gymnasium as gym
import numpy as np
from utils import scale_demand
from simulation.sim_setup import get_direction_lookup, get_related_lanes_edges, get_intersection_phase_groups

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
        self.warmup_steps = control_args['warmup_steps'] # is a list of two values
        self.sumo_running = False
        self.step_count = 0
        
        # Modify file paths to include the unique suffix. Each worker has their own environment and hence their own copy of the trips file.
        self.unique_suffix = f"_{worker_id}" if worker_id is not None else ""
        self.total_unique_ids_veh = []
        self.total_unique_ids_ped = []
        # dictionaries to store the last waiting time for each entity
        self.prev_vehicle_waiting_time = {}
        self.prev_ped_waiting_time = {}

        self.vehicle_output_trips = self.vehicle_output_trips.replace('.xml', f'{self.unique_suffix}.xml')

        self.pedestrian_output_trips = self.pedestrian_output_trips.replace('.xml', f'{self.unique_suffix}.xml')
        self.tl_ids = ['cluster_172228464_482708521_9687148201_9687148202_#5more', # Intersection
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
        self.per_timestep_state_dim = control_args['per_timestep_state_dim']
        print(f"Steps per action: {self.steps_per_action}")

        self.current_action_step = 0 # To track where we are within the curret action's duration
        self.int_tl_phase_groups, self.int_crosswalk_phase_groups = get_intersection_phase_groups()
        self.direction_and_edges  = get_direction_lookup()
        self.tl_lane_dict = get_related_lanes_edges()
        self.tl_pedestrian_status = {} # For pedestrians related to crosswalks attached to TLS.
        self.corrected_occupancy_map = None

        # Create a reverse lookup dict
        self.edge_to_direction = {}
        for direction, edges in self.direction_and_edges.items():
            for edge in edges:
                self.edge_to_direction[edge] = direction

        self.midblock_phase_groups = {0: "rrG", 1: "GGr", 2: "yyr", 3: "rrr"}
        self.directions = ["north", "east", "south", "west"]
        self.turns = ["straight", "right", "left"]
        self.direction_turn_intersection_incoming = [f"{direction}-{turn}" for direction in self.directions for turn in self.turns]
        self.direction_turn_intersection_inside = [f"{direction}-{turn}" for direction in self.directions for turn in ["straight", "left"]] # Exclude right
        self.direction_turn_midblock = ["west-straight", "east-straight"]

        # Normalization parameters (100m cutoff distance with 5m veh and 2.5m gap = approx 13 vehicles)
        self.max_pressure_vehicle = 15
        self.max_pressure_pedestrian = 25 # arbitrary
        self.max_switches = len(self.tl_ids)
        
        # For safety conflict tracking (only in unsignalized)
        self.recorded_conflicts = set()  # Set of unique conflict identifiers to prevent double-counting
        self.total_conflicts = 0         # Running total of unique vehicle-pedestrian conflicts
        self.total_switches = 0 # Only applicable for RL. 

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
            return "center" # cases where both blinkers are on (emergency) or off

    def _step_operations(self, occupancy_map, print_map=False, cutoff_distance=100):
        """
        Some corrections have to be done to the occupancy map every step.
        1. Pedestrians wanting to cross, and pedestrians already crossed both are going to be in the same area in simulation. This mechanism helps distinguish between them. 
        - If a pedestrian is inside (present in outgoing of occupancy map) the crosswalk and hasn't been marked as 'crossed', mark them.
        - Later during tracking, only include unmarked pedestrians in incoming.
        - This mechanism fails when a pedestrian wants to cross twice (however, the probability of this happening is low, assume they wont cross twice).
        2. At the intersection, in some directions (incoming), same lanes are used for L, R, S turns (in case of vehicles and incoming). The straight lane will have repeated entries of those vehicles, remove them.  
        3. Vehicles are only included in the occupancy map if they are within a given distance (due to the length of some edges, far away vehicles may also be present). Only for incoming and outgoing.
        """
        # 1.
        for tl_id in self.tl_ids:
            if tl_id == 'cluster_172228464_482708521_9687148201_9687148202_#5more': # Intersection has 4 outgoing directions

                for direction in self.directions:
                    for area in ["main", "vicinity"]:
                        for ped_id in occupancy_map[tl_id]["pedestrian"]["outgoing"][direction][area]:
                            if self._check_crossing_status(ped_id):
                                self.tl_pedestrian_status[ped_id] = 'crossed'

            else: # Midblock has one direction (north). Also midblocks have no vicinity edges.
                for ped_id in occupancy_map[tl_id]["pedestrian"]["outgoing"]["north"]["main"]:
                    if self._check_crossing_status(ped_id):
                        self.tl_pedestrian_status[ped_id] = 'crossed'

        # 2.
        for direction_turn in self.direction_turn_intersection_incoming:
            if direction_turn not in ['south-straight', 'west-straight', 'east-straight', 'north-straight']:
                direction, _ = direction_turn.split('-')
                straight_lane = f"{direction}-straight"
                straight_lane_vehicles = occupancy_map['cluster_172228464_482708521_9687148201_9687148202_#5more']['vehicle']['incoming'][straight_lane]
                
                # If this vehicles in a non-straight lane are also in the straight lane, remove them from the straight lane.
                for veh_id in occupancy_map['cluster_172228464_482708521_9687148201_9687148202_#5more']['vehicle']['incoming'][direction_turn]:
                    if veh_id in straight_lane_vehicles:
                        occupancy_map['cluster_172228464_482708521_9687148201_9687148202_#5more']['vehicle']['incoming'][straight_lane].remove(veh_id)

        # 3.
        for tl_id in self.tl_ids:
            if tl_id == 'cluster_172228464_482708521_9687148201_9687148202_#5more': # Intersection has 8 directions for incoming and 4 for outgoing
                # Incoming
                for direction_turn in self.direction_turn_intersection_incoming:
                    for veh_id in occupancy_map[tl_id]["vehicle"]["incoming"][direction_turn]:
                        distance = self._get_vehicle_distance_to_junction(tl_id, veh_id)
                        if distance > cutoff_distance:
                            occupancy_map[tl_id]["vehicle"]["incoming"][direction_turn].remove(veh_id)

                # Outgoing
                for direction in self.directions:
                    for veh_id in occupancy_map[tl_id]["vehicle"]["outgoing"][direction]:
                        distance = self._get_vehicle_distance_to_junction(tl_id, veh_id)
                        if distance > cutoff_distance:
                            occupancy_map[tl_id]["vehicle"]["outgoing"][direction].remove(veh_id)

            else: # Midblock has two directions (west-straight, east-straight) 
                for group in ["incoming", "outgoing"]:
                    for direction_turn in self.direction_turn_midblock:
                        for veh_id in occupancy_map[tl_id]["vehicle"][group][direction_turn]:
                            distance = self._get_vehicle_distance_to_junction(tl_id, veh_id)
                            if distance > cutoff_distance:
                                occupancy_map[tl_id]["vehicle"][group][direction_turn].remove(veh_id)

        if print_map: 
            print("\nOccupancy Map:")
            for tl_id, tl_data in occupancy_map.items():
                print(f"\nTraffic Light: {tl_id}")
                for type in ["vehicle", "pedestrian"]:
                    print(f"  {type.capitalize()}:")

                    for group in tl_data[type].keys():
                        print(f"    {group.capitalize()}:")
                        for direction, ids in tl_data[type][group].items():
                            if type == "vehicle":
                                print(f"      {direction.capitalize()}: {len(ids)} [{', '.join(ids)}]")
                                for idx in ids:
                                    distance = self._get_vehicle_distance_to_junction(tl_id, idx)
                                    print(f"        {idx}: {distance:.2f}m")   
                            else: 
                                for area in ids.keys():
                                    print(f"      {direction.capitalize(), area}: {len(ids[area])} [{', '.join(ids[area])}]")

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
            vehicle_pos = traci.vehicle.getPosition(vehicle_id)
            # Euclidean distance
            distance = math.sqrt(
                (junction_pos[0] - vehicle_pos[0])**2 + 
                (junction_pos[1] - vehicle_pos[1])**2
            )
            return distance

        except traci.TraCIException as e:
            print(f"Error calculating distance: {e}")
            return None
    
    def _get_pressure_dict(self, occupancy_map, print_dict=False):
        """
        Update the data structure that holds info about pressure (in each outgoing direction).
        For both vehicles and pedestrians:
            - Pressure = incoming - outgoing
        """
        pressure_dict = {tl_id: {"vehicle": {}, "pedestrian": {}} for tl_id in self.tl_ids}
        # Intersection
        # 1. Vehicles
        veh_pressure = {d: 0 for d in self.directions}
        for outgoing_direction in self.directions:
            incoming = 0 
            if outgoing_direction == 'north':
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["south-straight"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["east-right"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["west-left"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["inside"]["south-straight"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["inside"]["west-left"])

            elif outgoing_direction == 'south':
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["north-straight"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["east-left"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["west-right"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["inside"]["north-straight"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["inside"]["east-left"])

            elif outgoing_direction == 'east':
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["west-straight"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["north-left"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["south-right"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["inside"]["west-straight"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["inside"]["north-left"])

            elif outgoing_direction == 'west':
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["east-straight"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["north-right"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["south-left"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["inside"]["east-straight"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["inside"]["south-left"])

            outgoing = len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["outgoing"][outgoing_direction])
            veh_pressure[outgoing_direction] = incoming - outgoing
            pressure_dict["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"][outgoing_direction] = veh_pressure[outgoing_direction]

        # 2. Pedestrians 
        # Incmoing (0.7*main + 0.3*vicinity)
        ped_pressure = {d: 0 for d in self.directions}
        for outgoing_direction in self.directions:
            incoming = 0.7 * len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["pedestrian"]["incoming"][outgoing_direction]["main"]) + \
                0.3 * len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["pedestrian"]["incoming"][outgoing_direction]["vicinity"])
            outgoing = len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["pedestrian"]["outgoing"][outgoing_direction]["main"]) # vicinity does not exist
            ped_pressure[outgoing_direction] = incoming - outgoing
            pressure_dict["cluster_172228464_482708521_9687148201_9687148202_#5more"]["pedestrian"][outgoing_direction] = ped_pressure[outgoing_direction]

        # Midblock
        for tl_id in self.tl_ids[1:]:
            # 1. Vehicles (have two directions)
            veh_pressure = {d: 0 for d in self.direction_turn_midblock}
            for direction in self.direction_turn_midblock:
                incoming = len(occupancy_map[tl_id]["vehicle"]["incoming"][direction])
                outgoing = len(occupancy_map[tl_id]["vehicle"]["outgoing"][direction])
                veh_pressure[direction] = incoming - outgoing
                pressure_dict[tl_id]["vehicle"][direction] = veh_pressure[direction]

            # 2. Pedestrians (have one direction)
            # Vicinity does not exist
            ped_pressure = {"north": 0}
            incoming = len(occupancy_map[tl_id]["pedestrian"]["incoming"]["north"]["main"])
            outgoing = len(occupancy_map[tl_id]["pedestrian"]["outgoing"]["north"]["main"])
            ped_pressure["north"] = incoming - outgoing
            pressure_dict[tl_id]["pedestrian"]["north"] = ped_pressure["north"]

        if print_dict:
            print("\nPressure Dictionary:")
            print("===================")
            print(f"\nIntersection:")
            print(f"Vehicles: {pressure_dict['cluster_172228464_482708521_9687148201_9687148202_#5more']['vehicle']}")
            print(f"Pedestrians: {pressure_dict['cluster_172228464_482708521_9687148201_9687148202_#5more']['pedestrian']}")
            print("\nMidblock TLs:")
            for tl_id in self.tl_ids[1:]:
                print(f"{tl_id}:")
                print(f"  Vehicles: {pressure_dict[tl_id]['vehicle']}")
                print(f"  Pedestrians: {{'north': {pressure_dict[tl_id]['pedestrian']['north']}}}")

        return pressure_dict
    
    def _get_occupancy_map(self, ):
        """
        Occupancy map = dict with vehicle and pedestrian information in the neighbourhood of all TLs
        If the same lane is used for multiple turns (straight, right, left), the indicator light (turns ON about 100m as a vehicle approaches the junction) of vehicle is used to determine the actual direction. 
        occupancy_map stores the ids.
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
                    "incoming": {"north": {}},
                    "outgoing": {"north": {}}
                }
            }
            
        for tl_id in self.tl_ids:
            for type in ["vehicle", "pedestrian"]:

                if type == "vehicle": # Vehicles 
                    # At the intersection, they have 8 directions for incoming + inside and 4 directions for outgoing
                    if tl_id == 'cluster_172228464_482708521_9687148201_9687148202_#5more':
                        for group_key, direction_turns_value in [("incoming", self.direction_turn_intersection_incoming), ("inside", self.direction_turn_intersection_inside)]:
                            for direction_turn in direction_turns_value:
                                occupancy_map[tl_id][type][group_key][direction_turn] = []
                                lanes_edges = self.tl_lane_dict[tl_id][type][group_key][direction_turn]

                                for lane_edge in lanes_edges:
                                    if lane_edge != '-1':  # Skip lanes that are common for all directions
                                        veh_ids = traci.lane.getLastStepVehicleIDs(lane_edge) if "edge" not in lane_edge else traci.edge.getLastStepVehicleIDs(lane_edge.split('.')[1]) 
                                        occupancy_map[tl_id][type][group_key][direction_turn].extend(veh_ids)
                                    
                                    else:
                                        # Look at the indicator light of the vehicle to get the actual direction.
                                        # Get the direction, turn. Then get all vehicles in the straight lane 
                                        # If there are multiple -1s, this case can occur multiple times.
                                        direction, turn = direction_turn.split('-')
                                        straight_lane_edge = self.tl_lane_dict[tl_id][type][group_key][f"{direction}-straight"][0] # Use 0 because in the current implementation, there is only one straight lane per direction
                                        veh_ids = traci.lane.getLastStepVehicleIDs(straight_lane_edge) if "edge" not in straight_lane_edge else traci.edge.getLastStepVehicleIDs(straight_lane_edge.split('.')[1]) 

                                        if len(veh_ids) > 0:
                                            new_ids = []
                                            for veh_id in veh_ids:
                                                veh_direction = self._get_vehicle_direction(traci.vehicle.getSignals(veh_id))
                                                if veh_direction == turn:
                                                    new_ids.append(veh_id)

                                            occupancy_map[tl_id][type][group_key][direction_turn].extend(new_ids)

                        # outgoing
                        for direction in self.directions:
                            occupancy_map[tl_id][type]["outgoing"][direction] = []
                            lanes_edges = self.tl_lane_dict[tl_id][type]["outgoing"][direction]
                            for lane_edge in lanes_edges:
                                veh_ids = traci.lane.getLastStepVehicleIDs(lane_edge) if "edge" not in lane_edge else traci.edge.getLastStepVehicleIDs(lane_edge.split('.')[1]) 
                                occupancy_map[tl_id][type]["outgoing"][direction].extend(veh_ids)

                    else:  # For midblock TLs, they have 2 directions each for incoming, inside, and outgoing
                        for group in ["incoming", "outgoing"]: 
                            for direction_turn in self.direction_turn_midblock:
                                occupancy_map[tl_id][type][group][direction_turn] = []
                                lanes_edges = self.tl_lane_dict[tl_id][type][group][direction_turn]
                                for lane_edge in lanes_edges:
                                    veh_ids = traci.lane.getLastStepVehicleIDs(lane_edge) if "edge" not in lane_edge else traci.edge.getLastStepVehicleIDs(lane_edge.split('.')[1]) 
                                    occupancy_map[tl_id][type][group][direction_turn].extend(veh_ids)

                        # Inside has to be done in a computationally expensive way (because its an interal edge not detected either by edge.getLastStepVehicleIDs or lane.getLastStepVehicleIDs). 
                        # Get all vehicles in the simulation and compare their road id with the lanes_edges.
                        for direction_turn in self.direction_turn_midblock:
                            occupancy_map[tl_id][type]["inside"][direction_turn] = []
                            lanes_edges = self.tl_lane_dict[tl_id][type]["inside"][direction_turn]
                            for veh_id in traci.vehicle.getIDList():
                                if traci.vehicle.getRoadID(veh_id) in lanes_edges:
                                    occupancy_map[tl_id][type]["inside"][direction_turn].append(veh_id)

                else: # Pedestrians: lanes or edges further divided into main and vicinity, each with incoming and outgoing
                    for group in ["incoming", "outgoing"]:
                    
                        # Intersection has four directions.
                        if tl_id == 'cluster_172228464_482708521_9687148201_9687148202_#5more':
                            for direction in self.directions:
                                occupancy_map[tl_id][type][group][direction] = {}
                                for area in ["main", "vicinity"]:
                                    occupancy_map[tl_id][type][group][direction][area] = []
                                    
                                    lanes_edges = self.tl_lane_dict[tl_id][type][group][direction][area]
                                    for lane_edge in lanes_edges:
                                        if lane_edge.startswith(':'): # Check if it's an internal lane. If yes, do it the computationally expensive way. Get all persons in the simulation
                                            
                                            for ped_id in traci.person.getIDList():
                                                if traci.person.getRoadID(ped_id) == lane_edge:
                                                    if group == "incoming":
                                                        if self._check_crossing_status(ped_id):
                                                            occupancy_map[tl_id][type][group][direction][area].append(ped_id)
                                                    else:
                                                        occupancy_map[tl_id][type][group][direction][area].append(ped_id)
                                        else:
                                            for ped_id in traci.edge.getLastStepPersonIDs(lane_edge):
                                                if group == "incoming":
                                                    if self._check_crossing_status(ped_id):
                                                        occupancy_map[tl_id][type][group][direction][area].append(ped_id)
                                                else:
                                                    occupancy_map[tl_id][type][group][direction][area].append(ped_id)

                        else: # Midblock has one direction (north). Also midblocks have no vicinity edges.
                            occupancy_map[tl_id][type][group]["north"]["main"] = []
                            lanes_edges = self.tl_lane_dict[tl_id][type][group]["north"]["main"]

                            for lane_edge in lanes_edges:
                                if lane_edge.startswith(':'): # Check if it's an internal lane. If yes, do it the computationally expensive way. Get all persons in the simulation
                                    
                                    for ped_id in traci.person.getIDList():
                                        if traci.person.getRoadID(ped_id) == lane_edge:
                                            
                                            if group == "incoming":
                                                if self._check_crossing_status(ped_id):
                                                    occupancy_map[tl_id][type][group]["north"]["main"].append(ped_id)
                                            else:
                                                occupancy_map[tl_id][type][group]["north"]["main"].append(ped_id)
                                else:

                                    for ped_id in traci.edge.getLastStepPersonIDs(lane_edge):
                                        if group == "incoming":
                                            if self._check_crossing_status(ped_id):
                                                occupancy_map[tl_id][type][group]["north"]["main"].append(ped_id)
                                        else:
                                            occupancy_map[tl_id][type][group]["north"]["main"].append(ped_id)

        return occupancy_map
    
    def _check_crossing_status(self, pedestrian_id):
        """
        There is a 1 timestep delay in marking pedestrians as crossed.
        In the first timestep, the pedestrian will not be in the tl_pedestrian_status dictionary.
        """
        return True if pedestrian_id not in self.tl_pedestrian_status or self.tl_pedestrian_status[pedestrian_id] != "crossed" else False

    @property
    def action_space(self):
        """
        * Simple action space
        The control action is represented as a 9-bit string for each traffic light:
        - First digit: Intersection signal (4 mutually exclusive configurations)
            0 = allow vehicular traffic through East-West only
            1 = allow vehicular traffic through North-South only
            2 = allow dedicated left turns through N-E and S-W only
            3 = disallow vehicular traffic in all directions
        - Next seven bits: Mid-block crosswalk signals (each bit is Bernoulli)
            1  = allow vehicles to cross (pedestrians red) at a given mid-block segment
            0  = disallow vehicles to cross (pedestrians green) at that segment
        """
        # Simple action
        action_space = [4] + [2] * 7  # First value is 0-3 for intersection, remaining 7 are binary
        return gym.spaces.MultiDiscrete(action_space)

    @property
    def observation_space(self):
        """
        * Observation space is defined per action step (i.e. accumulated over action duration)
        """
        return gym.spaces.Box(low=0, high=1, shape=(self.steps_per_action, self.per_timestep_state_dim), dtype=np.float32)

    def train_step(self, action):
        """
        If Tl = True, operating in eval mode with TL.
        """
        # if not self.sumo_running:
        #     raise Exception("Environment is not running. Call reset() to start the environment.")
        if self.previous_action is None:
            self.previous_action = action

        reward = 0
        done = False
        observation_buffer = []
        action = np.array(action)
        # switch detection does not need to be done every timestep. 
        switch_state, _ = self._detect_switch(action, self.previous_action)
 
        for i in range(self.steps_per_action): # Run simulation steps for the duration of the action
            # Apply action is called every timestep (return information useful for reward calculation)
            current_phase = self._apply_action(action, i, switch_state)
            traci.simulationStep() # Step length is the simulation time that elapses when each time this is called.
            self.step_count += 1
            obs = self._get_observation(current_phase)
            observation_buffer.append(obs)
            
        # outside the loop
        # Do before reward calculation
        # pressure_dict = self._get_pressure_dict(self.corrected_occupancy_map)
        # Reward outside the loop (only once per duration)
        reward = self._get_reward(self.corrected_occupancy_map, switch_state, pressure_dict = None) # pressure dict used only in pressure-based reward
        self.previous_action = action

        # Check if episode is done (outside the for loop, otherwise it would create a broken observation)
        if self._check_done():
            done = True

        observation = np.asarray(observation_buffer, dtype=np.float32) 
        #print(f"\nObservation shape: {observation.shape}, type: {type(observation)}, value: {observation}")
        #visualize_observation(observation)
        return observation, reward, done, False, {} # info is empty
    
    def eval_step(self, action, tl= False, unsignalized= False):
        """
        Step during evaluation.
        """
        if not self.sumo_running:
            raise Exception("Environment is not running. Call reset() to start the environment.")
        if self.previous_action is None:
            self.previous_action = action

        reward = 0
        done = False
        observation_buffer = []
        action = np.array(action)
        
        # switch detection does not need to be done every timestep. 
        switch_state, full_switch_state = self._detect_switch(action, self.previous_action)
        
        #TODO: Placing an if-else here (at almost the innermost loop) is very inefficient.
        if tl: 

            if unsignalized: # Set all Midblock as green.
                for i in range(self.steps_per_action): # Run simulation steps for the duration of the action
                    current_phase = [1]*len(self.tl_ids) # random phase
                    # Midblock
                    for j in range(1, len(self.tl_ids)):
                        tl_id = self.tl_ids[j]
                        mb_phase_string = "GgggggGG" if tl_id == "cluster_9740157181_9740483933" else "GGG"
                        traci.trafficlight.setRedYellowGreenState(tl_id, mb_phase_string)
                    traci.simulationStep() # Step length is the simulation time that elapses when each time this is called.
                    self.step_count += 1
                    obs = self._get_observation(current_phase)
                    observation_buffer.append(obs)

                    # Count near-conflicts after each step (after each simulation step has been applied)
                    self.total_conflicts += self._count_near_conflicts(self.corrected_occupancy_map)

            else: # Apply action for TL (For evaluation)
                for i in range(self.steps_per_action): # Run simulation steps for the duration of the action
                    current_phase = [1]*len(self.tl_ids) # random phase
                    traci.simulationStep() # Step length is the simulation time that elapses when each time this is called.
                    self.step_count += 1
                    obs = self._get_observation(current_phase)
                    observation_buffer.append(obs)
        else:
            self.total_switches += sum(full_switch_state)
            # Only apply the action from policy if not TL.   
            for i in range(self.steps_per_action): # Run simulation steps for the duration of the action
                # Apply action is called every timestep (return information useful for reward calculation)
                current_phase = self._apply_action(action, i, switch_state)
                traci.simulationStep() # Step length is the simulation time that elapses when each time this is called.
                self.step_count += 1
                obs = self._get_observation(current_phase)
                observation_buffer.append(obs)
        
        # Outside the loop, before reward calculation
        # pressure_dict = self._get_pressure_dict(self.corrected_occupancy_map)
        # Reward outside the loop (only once per duration)
        reward = self._get_reward(self.corrected_occupancy_map, switch_state, pressure_dict = None) # pressure dict used only in pressure-based reward
        self.previous_action = action

        # Check if episode is done (outside the for loop, otherwise it would create a broken observation)
        if self._check_done():
            done = True

        observation = np.asarray(observation_buffer, dtype=np.float32) 
        
        #print(f"\nObservation shape: {observation.shape}, type: {type(observation)}, value: {observation}")
        #visualize_observation(observation)
        # print(f"\nRecorded conflicts: {self.recorded_conflicts}\n")
        #print(f"Total switches: {self.total_switches}")
        return observation, reward, done, False, {} # info is empty

    def _detect_switch(self, current_action, previous_action):
        """
        Detect a switch in all the components (the vehicle part at the intersection considered as one component) i.e., total 8 components
        For Intersection:
            - A switch is detected if action changes from: 
                - 0 to 1: E-W green to N-S green
                - 1 to 0: N-S green to E-W green
                - 2 to 0: N-E + S-W green (Dedicated left) to E-W green
        For Midblock: 
            - Action = 1 means vehicle green (pedestrian red)
            - Action = 0 means vehicle red (pedestrian green)
            - We only enfore a mandatory yellow phase if there is a 1 to 0 transition i.e., only consider this as a switch.
        """
        current_action = ''.join(map(str, current_action))
        previous_action = ''.join(map(str, previous_action))
        current_intersection_action = current_action[0:1]
        current_mid_block_action = current_action[1:]  
        previous_intersection_action = previous_action[0:1]
        previous_mid_block_action = previous_action[1:]
        
        # print(f"Previous actions: Intersection: {previous_intersection_action}, Midblock: {previous_mid_block_action}")
        # print(f"Current actions: Intersection: {current_intersection_action}, Midblock: {current_mid_block_action}")
        
        switch_state = []
        intersection_switch = [int(c1 == '0' and c2 == '1') or int(c1 == '1' and c2 == '0') or int(c1 == '2' and c2 == '0') for c1, c2 in zip(previous_intersection_action, current_intersection_action)]
        switch_state.extend(intersection_switch)
        midblock_switch = [int(c1 == '1' and c2 == '0') for c1, c2 in zip(previous_mid_block_action, current_mid_block_action)] # only detect 1->0 transitions
        switch_state.extend(midblock_switch)
        # print(f"Switch state: {switch_state}")

        # For the plot, also detect all the switches.
        full_switch_state = []
        for i in range(len(current_action)):
            full_switch_state.append(int(current_action[i] != previous_action[i]))
        
        # print(f"Full switch state: {full_switch_state}\n")

        return switch_state, full_switch_state

    def _get_observation(self, current_phase, print_map=False):
        """
        * Per step observation size = 96
        * Composed of: 
            - Current phase information (8 elements that transitions throughout the action duration)
            - Intersection (32 elements):
                - Vehicles: 
                    - Incoming (12 directions)
                    - Inside (8 directions)
                    - Outgoing (4 directions)
                - Pedestrians
                    - Incoming (4 directions)
                    - Outgoing (4 directions)
            - 7 Midblocks (7x8 = 56 elements):
                - Vehicles: 
                    - Incoming (2 directions)
                    - Inside (2 directions)
                    - Outgoing (2 directions)
                - Pedestrians
                    - Incoming (1 direction)
                    - Outgoing (1 direction)
            - 8 + 32 + 56 = 96

        * Each action persists for a number of timesteps (transitioning though phases); observation is collected at each timestep.
        * Pressure itself is not a part of the observation (only used for reward calculation).
        * Add full info about one TL at a time (preserve locality).
        * Spatial locality: TLs are arranged from left to right. Information about incoming, inside, and outgoing is kept next to each other.
        * Temporal locality: Observations are stacked verticalled one timestep to the next.
        * Normalize everything with a max normalizer.

        * Main and vicinity are given equal weightage (1). A single person in observation areas should represent same val.
        """
        
        self.corrected_occupancy_map = self._step_operations(self._get_occupancy_map(), print_map=print_map, cutoff_distance=100)
        observation = list(current_phase) # 8 elements

        # Intersection
        # - vehicles
        int_incoming = [] # 12 directions
        for direction_turn in self.direction_turn_intersection_incoming:
            int_incoming.append(len(self.corrected_occupancy_map['cluster_172228464_482708521_9687148201_9687148202_#5more']["vehicle"]["incoming"][direction_turn]))
        observation.extend(int_incoming)

        int_inside = [] # 8 directions
        for direction_turn in self.direction_turn_intersection_inside:
            int_inside.append(len(self.corrected_occupancy_map['cluster_172228464_482708521_9687148201_9687148202_#5more']["vehicle"]["inside"][direction_turn]))
        observation.extend(int_inside)
  
        int_outgoing = [] # 4 directions
        for direction in self.directions:
            int_outgoing.append(len(self.corrected_occupancy_map['cluster_172228464_482708521_9687148201_9687148202_#5more']["vehicle"]["outgoing"][direction]))
        observation.extend(int_outgoing)

        # - pedestrians
        int_incoming_ped = [] # 4 directions. This contains both main and vicinity areas. Sum (70% = main, 30% = vicinity)
        for direction in self.directions:
            main = len(self.corrected_occupancy_map['cluster_172228464_482708521_9687148201_9687148202_#5more']["pedestrian"]["incoming"][direction]["main"])
            vicinity = len(self.corrected_occupancy_map['cluster_172228464_482708521_9687148201_9687148202_#5more']["pedestrian"]["incoming"][direction]["vicinity"])
            int_incoming_ped.append(main + vicinity)
        observation.extend(int_incoming_ped)

        int_outgoing_ped = [] # 4 directions
        for direction in self.directions:
            int_outgoing_ped.append(len(self.corrected_occupancy_map['cluster_172228464_482708521_9687148201_9687148202_#5more']["pedestrian"]["outgoing"][direction]["main"]))
        observation.extend(int_outgoing_ped)

        # Midblock (7 TLs)
        for tl_id in self.tl_ids[1:]:
            # - vehicles
            mb_incoming = [] # 2 directions
            for direction_turn in self.direction_turn_midblock:
                mb_incoming.append(len(self.corrected_occupancy_map[tl_id]["vehicle"]["incoming"][direction_turn]))
            observation.extend(mb_incoming)

            mb_inside = [] # 2 directions
            for direction_turn in self.direction_turn_midblock:
                mb_inside.append(len(self.corrected_occupancy_map[tl_id]["vehicle"]["inside"][direction_turn]))
            observation.extend(mb_inside)

            mb_outgoing = [] # 2 directions
            for direction in self.direction_turn_midblock:
                mb_outgoing.append(len(self.corrected_occupancy_map[tl_id]["vehicle"]["outgoing"][direction]))
            observation.extend(mb_outgoing)

            # - pedestrians
            # Incoming, 1 direction
            observation.append(len(self.corrected_occupancy_map[tl_id]["pedestrian"]["incoming"]["north"]["main"]))

            # Outgoing, 1 direction
            observation.append(len(self.corrected_occupancy_map[tl_id]["pedestrian"]["outgoing"]["north"]["main"]))
        # print(f"\nObservation before normalization: {observation}")

        # Normalize with running mean and std
        observation = np.asarray(observation, dtype=np.float32)
        #print(f"\nObservation: shape: {observation.shape}") 
        return observation

    def _apply_action(self, action, current_action_step, switch_state):
        """
        apply_action is the enforcement of the chosen action (9-bit string) to the traffic lights and crosswalks at every step.
        Based on the switch_state, a 4-second mandatory yellow phase is enforced to the direction/ light that is turning red, before starting the new phase in the other light.
        For Intersection: 

        For each Midblock:
        - Action = 0: rrG: Red for both vehicle directions, green for pedestrian crosswalk
        - Action = 1: GGr: Green for the 2 vehicle directions, red for pedestrian crosswalk
        - If switch state is 1 (that means current action is 0):
             - Action 2: 4 timesteps of yyr: Yellow for the 2 vehicle directions, red for pedestrian crosswalk
             - Action 3: 1 timestep of rrr (all-red)
             - Action 0: 5 timestep of rrG

        * It does not matter what phases are specified in the Tlogic in net file, we override it from here.
        """
        #print(f"Action: {action}, switch_state: {switch_state}, type: {type(switch_state)}")
        current_phase = []

        # Intersection
        int_action = action[0]
        int_switch_state = switch_state[0]
        if int_switch_state == 0:
            int_vehicle_phase_group = int_action
            vehicle_phase_string = self.int_tl_phase_groups[int_vehicle_phase_group]
        else:
            # If switch_state is 1 and current_action is 1, select phase group 4
            # If switch_state is 1 and current_action is 0, select phase group 5
            if int_action == 1:
                int_vehicle_phase_group = 4
                vehicle_phase_string = self.int_tl_phase_groups[int_vehicle_phase_group][current_action_step]
            elif int_action == 0:
                int_vehicle_phase_group = 5
                vehicle_phase_string = self.int_tl_phase_groups[int_vehicle_phase_group][current_action_step]
            else: # coming from dedicated left turn (keep as is)
                int_vehicle_phase_group = int_action
                vehicle_phase_string = self.int_tl_phase_groups[int_vehicle_phase_group]

        current_phase.append(int_vehicle_phase_group)
        pedestrian_phase_abcd = self.int_crosswalk_phase_groups[int_action]
        # ArBCrD
        pedestrian_phase_string = pedestrian_phase_abcd['A'] + 'r' + pedestrian_phase_abcd['B'] + pedestrian_phase_abcd['C'] + 'r' + pedestrian_phase_abcd['D']
        int_state = vehicle_phase_string + pedestrian_phase_string
        traci.trafficlight.setRedYellowGreenState(self.tl_ids[0], int_state)
        
        # Midblock
        for i in range(1, len(self.tl_ids)):
            tl_id = self.tl_ids[i]
            mb_switch_state = switch_state[i]

            if mb_switch_state == 1:
                if current_action_step < 4:
                    mb_action = 2
                elif current_action_step == 4:
                    mb_action = 3
                elif current_action_step >= 5:
                    mb_action = 0
            else:
                mb_action = action[i]
            
            current_phase.append(mb_action)
            mb_phase_string = self.midblock_phase_groups[mb_action]

            if tl_id == "cluster_9740157181_9740483933": # For this TL, add 4 small 'g's after the first letter
                mb_phase_string = mb_phase_string[0] + "g"*5 + mb_phase_string[1:]

            #print(f"Setting phase: {mb_phase_string} for {tl_id}")
            traci.trafficlight.setRedYellowGreenState(tl_id, mb_phase_string)

        return current_phase
    
    def _get_reward(self, corrected_occupancy_map, switch_state, pressure_dict):
        """ 
        wrapper
        """
        return self._get_mwaq_reward_exponential(corrected_occupancy_map, switch_state, print_reward=False)

    def _get_mwaq_reward(self, corrected_occupancy_map, switch_state, print_reward=False):
        """
        * Maximum wait aggregated queue (MWAQ)
        * MWAQ = For each TL, For both veh and ped: [sum of queue lengths of all incoming directions x maximum waiting time among all]
        * Penalize high MWAQ, i.e. make it negative.
        * Individually normalize MWAQ for each TL.
        Other components (Optional): 
            - Penalize frequent changes of action based on switch_state 
        Vehicles:
        - getWaitingTime: The waiting time (in seconds) of a vehicle spent with speed below 0.1 m/s. Reset to 0 every time it moves.
        - getAccumulatedWaitingTime: The accumulated waiting time (in seconds) of a vehicle over a certain time interval (interval length is set with --waiting-time-memory)
        Pedestrians:
        - getWaitingTime: The waiting time (in seconds) of a pedestrian spent with speed below 0.1 m/s. Reset to 0 every time it moves.

        Normalizers:
        - A fixed normalizer multiplied by the number of incoming directions

        # TODO: Should we consider vicinity for pedestrians as well?
        """
        MWAQ_VEH_NORMALIZER = 100
        MWAQ_PED_NORMALIZER = 100
        VEH_THRESHOLD_SPEED = 0.2 # m/s
        PED_THRESHOLD_SPEED = 0.5 # m/s

        # Intersection 
        # Vehicle
        int_veh_mwaq = 0
        # queue length only starts counting if the vehicles are below 0.1 m/s
        # set thsi to 0.5 so that it does not have a zero value if vehicles  moving but very slow (upto the threshold speed)
        max_wait_time_veh_int = 0.5 
        veh_queue_length = 0

        for direction_turn in self.direction_turn_intersection_incoming:
            int_vehicles = corrected_occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"][direction_turn]
            for veh_id in int_vehicles:
                # queue length
                if traci.vehicle.getSpeed(veh_id) < VEH_THRESHOLD_SPEED:
                    veh_queue_length += 1
                # wait time
                wait_time = traci.vehicle.getWaitingTime(veh_id)
                if wait_time > max_wait_time_veh_int:
                    max_wait_time_veh_int = wait_time

        int_veh_mwaq = veh_queue_length * max_wait_time_veh_int
        norm_int_veh_mwaq = int_veh_mwaq / (MWAQ_VEH_NORMALIZER * len(self.direction_turn_intersection_incoming))

        # Pedestrian
        int_ped_mwaq = 0
        max_wait_time_ped_int = 0.5
        ped_queue_length = 0
        for direction in self.directions:
            int_pedestrians = corrected_occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["pedestrian"]["incoming"][direction]["main"]
            for ped_id in int_pedestrians:
                # queue length
                if traci.person.getSpeed(ped_id) < PED_THRESHOLD_SPEED:
                    ped_queue_length += 1
                # wait time
                wait_time = traci.person.getWaitingTime(ped_id)
                if wait_time > max_wait_time_ped_int:

                    max_wait_time_ped_int = wait_time
        int_ped_mwaq = ped_queue_length * max_wait_time_ped_int
        norm_int_ped_mwaq = int_ped_mwaq / (MWAQ_PED_NORMALIZER * len(self.directions))

        # Midblock
        # Vehicle
        norm_mb_veh_mwaq = 0
        norm_mb_veh_mwaq_per_tl = {}
        for tl_id in self.tl_ids[1:]:
            tl_veh_mwaq = 0
            max_wait_time_veh_mb = 0.5
            veh_queue_length = 0
            for direction in self.direction_turn_midblock:
                mb_vehicles = corrected_occupancy_map[tl_id]["vehicle"]["incoming"][direction]

                # queue length
                for veh_id in mb_vehicles:
                    if traci.vehicle.getSpeed(veh_id) < VEH_THRESHOLD_SPEED:
                        veh_queue_length += 1

                    # wait time
                    wait_time = traci.vehicle.getWaitingTime(veh_id)
                    if wait_time > max_wait_time_veh_mb:
                        max_wait_time_veh_mb = wait_time

            tl_veh_mwaq = veh_queue_length * max_wait_time_veh_mb
            norm_tl_veh_mwaq = tl_veh_mwaq / (MWAQ_VEH_NORMALIZER * len(self.direction_turn_midblock))
            norm_mb_veh_mwaq_per_tl[tl_id] = norm_tl_veh_mwaq
            norm_mb_veh_mwaq += norm_tl_veh_mwaq
        norm_mb_veh_mwaq = norm_mb_veh_mwaq / len(self.tl_ids[1:])


        # Pedestrian    
        norm_mb_ped_mwaq = 0
        norm_mb_ped_mwaq_per_tl = {}
        for tl_id in self.tl_ids[1:]:
            tl_ped_mwaq = 0
            max_wait_time_ped_mb = 0.5
            ped_queue_length = 0
            mb_pedestrians = corrected_occupancy_map[tl_id]["pedestrian"]["incoming"]["north"]["main"] # only one direction # 
            for ped_id in mb_pedestrians:
                # queue length
                if traci.person.getSpeed(ped_id) < PED_THRESHOLD_SPEED:
                    ped_queue_length += 1
                # wait time
                wait_time = traci.person.getWaitingTime(ped_id)
                if wait_time > max_wait_time_ped_mb:

                    max_wait_time_ped_mb = wait_time

            tl_ped_mwaq = ped_queue_length * max_wait_time_ped_mb
            norm_tl_ped_mwaq = tl_ped_mwaq / MWAQ_PED_NORMALIZER
            norm_mb_ped_mwaq_per_tl[tl_id] = norm_tl_ped_mwaq
            norm_mb_ped_mwaq += norm_tl_ped_mwaq
        norm_mb_ped_mwaq = norm_mb_ped_mwaq / len(self.tl_ids[1:])  

        # Frequency of switch state changes
        # norm_switch_penalty = sum(switch_state) / len(self.tl_ids)
        
        reward = -1 * (norm_int_veh_mwaq + norm_int_ped_mwaq + norm_mb_veh_mwaq + norm_mb_ped_mwaq)

        if print_reward:
            print(f"Intersection Reward Components:\n"
                  f"\tVehicle MWAQ: {norm_int_veh_mwaq}\n"
                  f"\tPedestrian MWAQ: {norm_int_ped_mwaq}")

            for tl_id in self.tl_ids[1:]:
                print(f"Midblock TL {tl_id} Reward Components:\n"
                      f"\tVehicle MWAQ: {norm_mb_veh_mwaq_per_tl[tl_id]}\n"
                      f"\tPedestrian MWAQ: {norm_mb_ped_mwaq_per_tl[tl_id]}")
            # print(f"Switch penalty: {norm_switch_penalty}")
            print(f"Total Reward: {reward}\n\n")

        return reward

    def _get_mwaq_reward_exponential(self, corrected_occupancy_map, switch_state, print_reward=False):
        """
        - Exponentially Increasing Normalized Maximum wait aggregated queue (EI-MWAQ) 
        * MWAQ = For each TL, For both veh and ped: [sum of queue lengths of all incoming directions x maximum waiting time among all]
        * Penalize high MWAQ, i.e. make it negative.
        * Individually normalize MWAQ for each TL.
        Other components (Optional): 
            - Penalize frequent changes of action based on switch_state 

        Vehicles:
        - getWaitingTime: The waiting time (in seconds) of a vehicle spent with speed below 0.1 m/s. Reset to 0 every time it moves.
        - getAccumulatedWaitingTime: The accumulated waiting time (in seconds) of a vehicle over a certain time interval (interval length is set with --waiting-time-memory)
        Pedestrians:
        - getWaitingTime: The waiting time (in seconds) of a pedestrian spent with speed below 0.1 m/s. Reset to 0 every time it moves.

        Normalizers:
        - A fixed normalizer multiplied by the number of incoming directions

        # TODO: Should we consider vicinity for pedestrians as well?
        """
        MWAQ_VEH_NORMALIZER = 8.0
        MWAQ_PED_NORMALIZER = 10.0
        VEH_THRESHOLD_SPEED = 0.2 # m/s
        PED_THRESHOLD_SPEED = 0.5 # m/s # 0.1 is the threshold in SUMO by default (i.e. wait time is counted when speed is below 0.1 m/s)

        # Intersection 
        # Vehicle
        int_veh_mwaq = 0
        max_wait_time_veh_int = 0.5
        veh_queue_length = 0

        for direction_turn in self.direction_turn_intersection_incoming:
            int_vehicles = corrected_occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"][direction_turn]
            for veh_id in int_vehicles:
                # queue length
                if traci.vehicle.getSpeed(veh_id) < VEH_THRESHOLD_SPEED:
                    veh_queue_length += 1
                # wait time
                wait_time = traci.vehicle.getWaitingTime(veh_id)
                if wait_time > max_wait_time_veh_int:
                    max_wait_time_veh_int = wait_time

        int_veh_mwaq = veh_queue_length * max_wait_time_veh_int
        # In practice, using len(self.direction_turn_intersection_incoming) (with value 12) in the division, is too harsh, and will lead to pedestrians getting extreme favor.
        # Only 7 incoming directions are physically present.
        norm_int_veh_mwaq = int_veh_mwaq / (MWAQ_VEH_NORMALIZER * len(self.directions))

        # Pedestrian
        int_ped_mwaq = 0
        max_wait_time_ped_int = 0.5
        ped_queue_length = 0
        for direction in self.directions:
            int_pedestrians = corrected_occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["pedestrian"]["incoming"][direction]["main"]
            for ped_id in int_pedestrians:
                # queue length
                if traci.person.getSpeed(ped_id) < PED_THRESHOLD_SPEED:
                    ped_queue_length += 1
                # wait time
                wait_time = traci.person.getWaitingTime(ped_id)
                if wait_time > max_wait_time_ped_int:

                    max_wait_time_ped_int = wait_time
        int_ped_mwaq = ped_queue_length * max_wait_time_ped_int
        norm_int_ped_mwaq = int_ped_mwaq / (MWAQ_PED_NORMALIZER * len(self.directions))

        # Midblock
        # Vehicle
        norm_mb_veh_mwaq_per_tl = {}
        for tl_id in self.tl_ids[1:]:
            tl_veh_mwaq = 0
            max_wait_time_veh_mb = 0.5
            veh_queue_length = 0
            for direction in self.direction_turn_midblock:
                mb_vehicles = corrected_occupancy_map[tl_id]["vehicle"]["incoming"][direction]

                # queue length
                for veh_id in mb_vehicles:
                    if traci.vehicle.getSpeed(veh_id) < VEH_THRESHOLD_SPEED:
                        veh_queue_length += 1

                    # wait time
                    wait_time = traci.vehicle.getWaitingTime(veh_id)
                    if wait_time > max_wait_time_veh_mb:
                        max_wait_time_veh_mb = wait_time

            tl_veh_mwaq = veh_queue_length * max_wait_time_veh_mb
            norm_tl_veh_mwaq = tl_veh_mwaq / (MWAQ_VEH_NORMALIZER * len(self.direction_turn_midblock))
            norm_mb_veh_mwaq_per_tl[tl_id] = norm_tl_veh_mwaq

        # Pedestrian    
        norm_mb_ped_mwaq_per_tl = {}
        for tl_id in self.tl_ids[1:]:
            tl_ped_mwaq = 0
            max_wait_time_ped_mb = 0.5
            ped_queue_length = 0
            mb_pedestrians = corrected_occupancy_map[tl_id]["pedestrian"]["incoming"]["north"]["main"] # only one direction # 
            for ped_id in mb_pedestrians:
                # queue length
                if traci.person.getSpeed(ped_id) < PED_THRESHOLD_SPEED:
                    ped_queue_length += 1
                # wait time
                wait_time = traci.person.getWaitingTime(ped_id)
                if wait_time > max_wait_time_ped_mb:
                    max_wait_time_ped_mb = wait_time

            tl_ped_mwaq = ped_queue_length * max_wait_time_ped_mb
            norm_tl_ped_mwaq = tl_ped_mwaq / MWAQ_PED_NORMALIZER
            norm_mb_ped_mwaq_per_tl[tl_id] = norm_tl_ped_mwaq

        # Frequency of switch state changes
        # norm_switch_penalty = sum(switch_state) / len(self.tl_ids)
        
        # Final reward calculation
        # For the intersection, exponent the current normalized value (for both veh and ped)
        final_int_veh = np.exp(norm_int_veh_mwaq) # Alpha value of 0.5 to reduce sensitivity 
        final_int_ped = np.exp(norm_int_ped_mwaq) 

        # For the midblock, compute an L2 norm over the vector of normalized values for each TL, then exponentiate (for both veh and ped)
        norm_mb_veh_l2 = np.linalg.norm(np.array(list(norm_mb_veh_mwaq_per_tl.values())))
        norm_mb_ped_l2 = np.linalg.norm(np.array(list(norm_mb_ped_mwaq_per_tl.values())))

        # TODO: For Midblock: Can further, effectively "normalize" the L2 aggregation. By dividing ny sqrt(N). Reduce the effect of outliers.
        # Perhaps preserving outliers is beneficial.
        final_mb_veh = np.exp(norm_mb_veh_l2)
        final_mb_ped = np.exp(norm_mb_ped_l2)

        # Final reward is the negative sum of the four exponentiated values
        reward = -1 * (final_int_veh + final_int_ped + final_mb_veh + final_mb_ped)

        # Clip the reward (In an appropriately chosen range) before returning. 
        clipped_reward = np.clip(reward, -100000, 100000)

        if print_reward:
            print(f"Intersection Reward Components:\n"
                f"\tVehicle MWAQ: {norm_int_veh_mwaq} (exp: {final_int_veh})\n"
                f"\tPedestrian MWAQ: {norm_int_ped_mwaq} (exp: {final_int_ped})")

            for tl_id in self.tl_ids[1:]:
                print(f"Midblock TL {tl_id} Reward Components:\n"
                    f"\tVehicle MWAQ: {norm_mb_veh_mwaq_per_tl[tl_id]}\n"
                    f"\tPedestrian MWAQ: {norm_mb_ped_mwaq_per_tl[tl_id]}")
            print(f"Midblock Overall L2 Norms:\n"
                f"\tVehicle L2 Norm: {norm_mb_veh_l2} (exp: {final_mb_veh})\n"
                f"\tPedestrian L2 Norm: {norm_mb_ped_l2} (exp: {final_mb_ped})")
            # print(f"Switch penalty: {norm_switch_penalty}")
            print(f"Total Reward: {reward}\n\n")
            print(f"Clipped Reward: {clipped_reward}\n\n")

        return clipped_reward

    def _count_near_conflicts(self, corrected_occupancy_map, threshold_speed=1.0, distance_threshold=5.0):
        """
        Count potential vehicle-pedestrian conflicts at signalized/ unsignalized crosswalks.
        Only meaningful at high volumes. 

        A "near-conflict" represents a safety-critical event where a vehicle must yield
        to a pedestrian at a crosswalk. Each unique vehicle-pedestrian interaction is
        counted only once, even if it spans multiple timesteps.

        Features:
        - Tracks unique conflicts using vehicle+pedestrian+location identifiers
        - Only counts each unique interaction once during the simulation
        - Records yielding behavior as a proxy for potential safety incidents
        - Focuses specifically on mid-block crosswalks where conflicts are most likely

        Parameters:
        - threshold_speed: Speed below which a vehicle is considered to be yielding (m/s)
        - distance_threshold: Maximum distance between vehicle and pedestrian to qualify as conflict (m)

        Returns:
        - Number of new unique conflicts detected in the current timestep
        """
        new_conflicts = 0

        # For each midblock location
        for tl_id in self.tl_ids[1:]:  # Skip intersection, focus on midblocks
            # Get pedestrians at this crosswalk (incoming or outgoing. Most likely outgoing)
            pedestrians = corrected_occupancy_map[tl_id]["pedestrian"]["incoming"]["north"]["main"] + corrected_occupancy_map[tl_id]["pedestrian"]["outgoing"]["north"]["main"]

            if not pedestrians:  # Skip if no pedestrians present
                continue

            # Check vehicles approaching this location
            for direction in self.direction_turn_midblock:
                vehicles = corrected_occupancy_map[tl_id]["vehicle"]["incoming"][direction]

                for veh_id in vehicles:
                    # Current vehicle speed
                    current_speed = traci.vehicle.getSpeed(veh_id)

                    # If vehicle speed is still high
                    if current_speed > threshold_speed:
                        veh_pos = traci.vehicle.getPosition(veh_id)

                        # Check proximity to pedestrians
                        for ped_id in pedestrians:
                            ped_pos = traci.person.getPosition(ped_id)
                            distance = math.sqrt((veh_pos[0]-ped_pos[0])**2 +
                                            (veh_pos[1]-ped_pos[1])**2)

                            if distance < distance_threshold:
                                # Create a unique conflict identifier
                                conflict_id = f"{veh_id}_{ped_id}_{tl_id}"

                                # If this is a new conflict we haven't seen before
                                if conflict_id not in self.recorded_conflicts:
                                    self.recorded_conflicts.add(conflict_id)
                                    new_conflicts += 1

        return new_conflicts

    def _check_done(self):
        """
        TODO: What more conditions can be added here?
        - Gridlock? Jam? of vehicles or pedestrians? Crashes?
        """
        return self.step_count >= self.max_timesteps

    def reset(self, tl= False):
        """
        """
        if self.sumo_running:
            time.sleep(5) # Wait until the process really finishes 
            traci.close(False) #https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html
        
        if self.manual_demand_veh is not None :
            #scaling = convert_demand_to_scale_factor(self.manual_demand_veh, "vehicle", self.vehicle_input_trips) # Convert the demand to scaling factor first
            scale_demand(self.vehicle_input_trips, self.vehicle_output_trips, self.manual_demand_veh, demand_type="vehicle") # directly scaling factor given
        else: 
            # Automatically scale demand 
            scale_factor_vehicle = random.uniform(self.demand_scale_min, self.demand_scale_max)
            scale_demand(self.vehicle_input_trips, self.vehicle_output_trips, scale_factor_vehicle, demand_type="vehicle")

        if self.manual_demand_ped is not None:
            # scaling = convert_demand_to_scale_factor(self.manual_demand_ped, "pedestrian", self.pedestrian_input_trips)
            scale_demand(self.pedestrian_input_trips, self.pedestrian_output_trips, self.manual_demand_ped, demand_type="pedestrian") # directly scaling factor given
        else: 
            scale_factor_pedestrian = random.uniform(self.demand_scale_min, self.demand_scale_max)
            scale_demand(self.pedestrian_input_trips, self.pedestrian_output_trips, scale_factor_pedestrian, demand_type="pedestrian")

        if self.auto_start:
            sumo_cmd = ["sumo-gui" if self.use_gui else "sumo", 
                        "--verbose",
                        "--start" , 
                        "--quit-on-end", 
                        "-c", "./simulation/Craver_traffic_lights.sumocfg", 
                        "--step-length", str(self.step_length),
                        "--route-files", f"{self.vehicle_output_trips},{self.pedestrian_output_trips}"]
        else:
            sumo_cmd = ["sumo-gui" if self.use_gui else "sumo", 
                        "--verbose",
                        "--quit-on-end", 
                        "-c", "./simulation/Craver_traffic_lights.sumocfg", 
                        "--step-length", str(self.step_length),
                        "--route-files", f"{self.vehicle_output_trips},{self.pedestrian_output_trips}"]
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
        self.tl_lane_dict = get_related_lanes_edges()

        # Warmup period
        # How many actions to take during warmup
        warmup = random.randint(self.warmup_steps[0], self.warmup_steps[1])
        num_actions_warmup = warmup // self.steps_per_action
        #print(f"Number of actions during warmup: {num_actions_warmup}")
        observation_buffer = []
        for i in range(num_actions_warmup):
            # Randomly sample actions (1 digit for intersection, 7 bits for crosswalks)
            action = np.concatenate([np.random.randint(4, size=1), np.random.randint(2, size=7)]).astype(np.int32)
            #print(f"\nWarmup action {i}: {action}\n")
            if i==0:
                prev_action = action
            switch_state, _ = self._detect_switch(action, prev_action)

            # TODO: Inefficient do it this way.
            if tl:
                for j in range(self.steps_per_action):
                    current_phase = [1]*len(self.tl_ids) # random phase
                    traci.simulationStep() 
                    obs = self._get_observation(current_phase)
                    _ = self._get_pressure_dict(self.corrected_occupancy_map)
                    observation_buffer.append(obs)
                    # No reward calculation
                    # self.step_count += 1 # We are not counting the warmup steps in the total simulation steps
            else: 
                for j in range(self.steps_per_action):
                    current_phase = self._apply_action(action, j, switch_state)
                    traci.simulationStep() 
                    obs = self._get_observation(current_phase)
                    _ = self._get_pressure_dict(self.corrected_occupancy_map)
                    observation_buffer.append(obs)
                    # No reward calculation
                    # self.step_count += 1 # We are not counting the warmup steps in the total simulation steps

        print(f"\n{warmup} steps of warmup ended.\n")
        observation_buffer = observation_buffer[-self.steps_per_action:] # Only keep the observation of thelast action
        observation = np.asarray(observation_buffer, dtype=np.float32)
        #print(f"\nObservation (in reset): {observation.shape}")

        # reset the waiting times
        self.prev_vehicle_waiting_time = {}
        self.prev_ped_waiting_time = {}

        self.total_conflicts = 0 # Running total of unique vehicle-pedestrian conflicts
        self.total_switches = 0 # Only applicable for RL.
        
        return observation, {} # info is empty

    def close(self):
        if self.sumo_running:
            traci.close(False) #https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html
            self.sumo_running = False

    # Eval specific methods
    def get_vehicle_waiting_time(self,):
        """
        The cumulative incremental wait time for all vehicles.
        Instead of summing the cumulative waiting time each step, we add only the difference since the last call.
        """
        incremental_wait = 0.0
        current_vehicle_ids = traci.vehicle.getIDList()
        for veh_id in current_vehicle_ids:
            current_wait = traci.vehicle.getWaitingTime(veh_id)
            last_wait = self.prev_vehicle_waiting_time.get(veh_id, 0.0)
            diff = current_wait - last_wait
            # In case the waiting time was reset (e.g. when a vehicle moves) and diff becomes negative, use current_wait.
            if diff < 0:
                diff = current_wait
            incremental_wait += diff
            self.prev_vehicle_waiting_time[veh_id] = current_wait
            if veh_id not in self.total_unique_ids_veh:
                self.total_unique_ids_veh.append(veh_id)
        return incremental_wait

    def get_pedestrian_waiting_time(self):
        """
        The cumulative incremental wait time for all pedestrians.
        Only the increase in waiting time since the last call is accumulated.
        """
        incremental_wait = 0.0
        current_ped_ids = traci.person.getIDList()
        for ped_id in current_ped_ids:
            current_wait = traci.person.getWaitingTime(ped_id)
            last_wait = self.prev_ped_waiting_time.get(ped_id, 0.0)
            diff = current_wait - last_wait
            if diff < 0:
                diff = current_wait
            incremental_wait += diff
            self.prev_ped_waiting_time[ped_id] = current_wait
            if ped_id not in self.total_unique_ids_ped:
                self.total_unique_ids_ped.append(ped_id)
        return incremental_wait
    
    def total_unique_ids(self):
        """
        So far in the simulation, how many total unique ids were seen. 
        """
        return len(self.total_unique_ids_veh),  len(self.total_unique_ids_ped)