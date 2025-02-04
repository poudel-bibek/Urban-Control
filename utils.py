import os
import xml
import time
import json
import logging
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib  
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LinearLocator
import xml.etree.ElementTree as ET

def convert_demand_to_scale_factor(demand, demand_type, input_file):
    """
    Convert the demand to a scaling factor number.
    For vehicles: (veh/hr) that want to enter the network
    For pedestrians: (ped/hr) that want to enter the network
    """

    if demand <= 0:
        raise ValueError("Demand must be a positive number")
    
    if demand_type not in ['vehicle', 'pedestrian']:
        raise ValueError("Demand type must be either 'vehicle' or 'pedestrian'")
    
    # Calculate the original demand from the input file
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    if demand_type == 'vehicle':
        original_demand = len(root.findall("trip"))
    else:  # pedestrian
        original_demand = len(root.findall(".//person"))
    
    if original_demand == 0:
        raise ValueError(f"No {demand_type} demand found in the input file")
    
    # Calculate the time span of the original demand
    if demand_type == 'vehicle':
        elements = root.findall("trip")
    else:
        elements = root.findall(".//person")
    
    # Find the start and end time of the demand
    start_time = min(float(elem.get('depart')) for elem in elements)
    end_time = max(float(elem.get('depart')) for elem in elements)
    time_span = (end_time - start_time) / 3600  # Convert to hours
    
    # Calculate the original demand per hour
    original_demand_per_hour = original_demand / time_span if time_span > 0 else 0
    print(f"\nOriginal {demand_type} demand per hour: {original_demand_per_hour:.2f}")

    if original_demand_per_hour == 0:
        raise ValueError(f"Cannot calculate original {demand_type} demand per hour")
    
    # Calculate the scale factor
    scale_factor = demand / original_demand_per_hour
    
    return scale_factor

def scale_demand(input_file, output_file, scale_factor, demand_type):
    """
    This function was causing some errors, so there is a new version as well.
    """
    # Parse the XML file
    tree = ET.parse(input_file)
    root = tree.getroot()

    if demand_type == "vehicle":
        # Vehicle demand
        trips = root.findall("trip")
        for trip in trips:
            current_depart = float(trip.get('depart'))
            new_depart = current_depart / scale_factor
            trip.set('depart', f"{new_depart:.2f}")

        original_trip_count = len(trips)
        for i in range(1, int(scale_factor)):
            for trip in trips[:original_trip_count]:
                new_trip = ET.Element('trip')
                for attr, value in trip.attrib.items():
                    if attr == 'id':
                        new_trip.set(attr, f"{value}_{i}")
                    elif attr == 'depart':
                        new_depart = float(value) + (3600 * i / scale_factor)
                        new_trip.set(attr, f"{new_depart:.2f}")
                    else:
                        new_trip.set(attr, value)
                root.append(new_trip)

    elif demand_type == "pedestrian":
        # Pedestrian demand
        persons = root.findall(".//person")
        for person in persons:
            current_depart = float(person.get('depart'))
            new_depart = current_depart / scale_factor
            person.set('depart', f"{new_depart:.2f}")

        original_person_count = len(persons)
        for i in range(1, int(scale_factor)):
            for person in persons[:original_person_count]:
                new_person = ET.Element('person')
                for attr, value in person.attrib.items():
                    if attr == 'id':
                        new_person.set(attr, f"{value}_{i}")
                    elif attr == 'depart':
                        new_depart = float(value) + (3600 * i / scale_factor)
                        new_person.set(attr, f"{new_depart:.2f}")
                    else:
                        new_person.set(attr, value)
                
                # Copy all child elements (like <walk>)
                for child in person:
                    new_child = ET.SubElement(new_person, child.tag, child.attrib)
                    # Ensure 'from' attribute is present for walk elements
                    if child.tag == 'walk' and 'from' not in child.attrib:
                        # If 'from' is missing, use the first edge in the route
                        edges = child.get('edges', '').split()
                        if edges:
                            new_child.set('from', edges[0])
                        else:
                            logging.warning(f"Walk element for person {new_person.get('id')} is missing both 'from' and 'edges' attributes.")
                
                # Find the correct parent to append the new person
                parent = root.find(".//routes")
                if parent is None:
                    parent = root
                parent.append(new_person)

    else:
        print("Invalid demand type. Please specify 'vehicle' or 'pedestrian'.")
        return

    # Convert to string
    xml_str = ET.tostring(root, encoding='unicode')
   
    # Pretty print the XML string
    dom = xml.dom.minidom.parseString(xml_str)
    pretty_xml_str = dom.toprettyxml(indent="    ")
   
    # Remove extra newlines between elements
    pretty_xml_str = '\n'.join([line for line in pretty_xml_str.split('\n') if line.strip()])
    
    # If there are folders in the path that dont exist, create them
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write the formatted XML to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pretty_xml_str)
    
    print(f"{demand_type.capitalize()} demand scaled by factor {scale_factor}.") # Output written to {output_file}")
    
    # Wait for the file writing operations to finish (it could be large)
    time.sleep(2)
    
def visualize_observation(observation):
    """
    Visualize each timestep observation as image (save as png).
    Observation shape is (96, action_timesteps) containing normalized values between 0-1 representing:
    - Intersection vehicle counts (incoming, inside, outgoing) for each direction
    - Intersection pedestrian counts (incoming, outgoing) for each direction  
    - Midblock vehicle counts (incoming, inside, outgoing) for each direction
    - Midblock pedestrian counts (incoming, outgoing) for each direction
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    n_timesteps = observation.shape[0] 
    n_features = observation.shape[1]
    
    im = ax.imshow(observation, cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title('Observation')
    ax.set_xlabel('Features')
    ax.set_ylabel('Timesteps')
    
    # Add grid lines at feature boundaries
    ax.set_xticks(np.arange(-.5, n_features, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_timesteps , 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    
    plt.colorbar(im, location='right', shrink=0.25, aspect=10)
    plt.savefig('observation.png', bbox_inches='tight', dpi=200)
    plt.close()

def plot_consolidated_results(result_json_path, in_range_demand_scales, out_of_range_demand_scales):
    """
    Plot the consolidated results of the evaluation.
    Split the plot into two subplots:
    x-axis = various demand scales and demand values (use the scale and multiply the original demand to get demand values at each scale)
    - original_vehicle_demand = 201.54 veh/hr
    - original_pedestrian_demand = 2222.80 ped/hr
    y-axis = Average wait time of either vehicles or pedestrians
    Draw vertical dotted lines to separate in_range_demand_scales and out_of_range_demand scales
    """

    with open(result_json_path, 'r') as f:
        results = json.load(f)

    scales, veh_avg, ped_avg = [], [], []  
    for scale_str, runs in results.items():  
        scale = float(scale_str)  
        scales.append(scale)  
        veh_vals = []  
        ped_vals = []  
        for run in runs.values():  
            veh_vals.append(run["veh_avg_waiting_time"])  
            ped_vals.append(run["ped_avg_waiting_time"])  
        veh_avg.append(np.mean(veh_vals))  
        ped_avg.append(np.mean(ped_vals))  

    scales, veh_avg, ped_avg = [], [], []  
    for scale_str, runs in results.items():  
        scale = float(scale_str)  
        scales.append(scale)  
        veh_vals = []  
        ped_vals = []  
        for run in runs.values():  
            veh_vals.append(run["veh_avg_waiting_time"])  
            ped_vals.append(run["ped_avg_waiting_time"])  
        veh_avg.append(np.mean(veh_vals))  
        ped_avg.append(np.mean(ped_vals))  

    # Convert lists to numpy arrays and sort by scale.  
    scales = np.array(scales)  
    veh_avg = np.array(veh_avg)  
    ped_avg = np.array(ped_avg)  

    sort_idx = np.argsort(scales)  
    sorted_scales = scales[sort_idx]  
    veh_avg = veh_avg[sort_idx]  
    ped_avg = ped_avg[sort_idx]  

    # Original demand values.  
    original_vehicle_demand = 201.54    # veh/hr  
    original_pedestrian_demand = 2222.80  # ped/hr  

    # Compute actual demand values.  
    veh_x = sorted_scales * original_vehicle_demand  
    ped_x = sorted_scales * original_pedestrian_demand  

    # Use a clean, publication-style theme.  
    sns.set_theme(style="whitegrid", context="talk")  

    # Create figure and subplots.  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))  

    # Plot data.  
    sns.lineplot(x=veh_x, y=veh_avg, marker='o', color='tab:blue',  
                 ax=ax1, label="RL (Ours)")  
    sns.lineplot(x=ped_x, y=ped_avg, marker='o', color='tab:green',  
                 ax=ax2, label="RL (Ours)")  

    # Set subplot titles and axis labels.  
    ax1.set_title("Vehicle", fontweight="bold")  
    ax2.set_title("Pedestrian", fontweight="bold")  
    ax1.set_xlabel("Vehicle Demand (veh/hr)")  
    ax1.set_ylabel("Average Waiting Time (s)")  
    ax2.set_xlabel("Pedestrian Demand (ped/hr)")  
    ax2.set_ylabel("Average Waiting Time (s)")  

    # Add a margin of 5% to the x-axis limits so data are not flush.  
    veh_margin = 0.05 * (veh_x.max() - veh_x.min())  
    ped_margin = 0.05 * (ped_x.max() - ped_x.min())  
    ax1.set_xlim(veh_x.min() - veh_margin, veh_x.max() + veh_margin)  
    ax2.set_xlim(ped_x.min() - ped_margin, ped_x.max() + ped_margin)  

    # Format y-axis ticks to one decimal place.  
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))  
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))  

    # Reduce the number of x-ticks (every other tick).  
    veh_ticks = veh_x[::2]  
    ped_ticks = ped_x[::2]  
    ax1.set_xticks(veh_ticks)  
    ax2.set_xticks(ped_ticks)  

    # Create custom x tick labels with two lines: demand value and scale factor.  
    veh_xtick_labels = [f"{int(round(val))}"   # \n{scale:.2f}x
                        for val, scale in zip(veh_x[::2], sorted_scales[::2])]  
    ped_xtick_labels = [f"{int(round(val))}"  #\n{scale:.2f}x
                        for val, scale in zip(ped_x[::2], sorted_scales[::2])]  
    ax1.set_xticklabels(veh_xtick_labels)  
    ax2.set_xticklabels(ped_xtick_labels)  

    # Use dotted grid lines.  
    ax1.grid(True, linestyle=':')  
    ax2.grid(True, linestyle=':')  

    # Determine the valid (non-shaded) regions.  
    valid_min_scale = min(out_of_range_demand_scales)  
    valid_max_scale = max(out_of_range_demand_scales)  
    veh_valid_min = valid_min_scale * original_vehicle_demand  
    veh_valid_max = valid_max_scale * original_vehicle_demand  
    ped_valid_min = valid_min_scale * original_pedestrian_demand  
    ped_valid_max = valid_max_scale * original_pedestrian_demand  

    # Retrieve current (margined) x-axis limits and add shading.  
    veh_xlim = ax1.get_xlim()  
    ped_xlim = ax2.get_xlim()  
    ax1.axvspan(veh_xlim[0], veh_valid_min, facecolor='grey', alpha=0.2)  
    ax1.axvspan(veh_valid_max, veh_xlim[1], facecolor='grey', alpha=0.2)  
    ax2.axvspan(ped_xlim[0], ped_valid_min, facecolor='grey', alpha=0.2)  
    ax2.axvspan(ped_valid_max, ped_xlim[1], facecolor='grey', alpha=0.2)  

    # Place legends at the top right.  
    ax1.legend(loc='upper right')  
    ax2.legend(loc='upper right')  

    # Increase spacing between subplots and add outer margins.  
    plt.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.1, wspace=0.5)  

    # --- Adjust y-axis so that both subplots have the same number of tick labels ---  
    # First, extend each axis's top by one tick spacing.  
    for ax in [ax1, ax2]:  
        ymin, ymax = ax.get_ylim()  
        yticks = ax.get_yticks()  
        if len(yticks) > 1:  
            spacing = yticks[-1] - yticks[-2]  
        else:  
            spacing = (ymax - ymin) / 5  # fallback spacing  
        ax.set_ylim(ymin, ymax + spacing)  

    # Then force both subplots to use the same number of y-ticks.  
    # We choose the maximum count from the two as the fixed number.  
    n_ticks = max(len(ax1.get_yticks()), len(ax2.get_yticks()))  
    ax1.yaxis.set_major_locator(LinearLocator(numticks=n_ticks))  
    ax2.yaxis.set_major_locator(LinearLocator(numticks=n_ticks))  
    # Reapply the formatter after setting the locator  
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))  
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))  
    # --- End y-axis adjustment ---  

    plt.tight_layout()  
    plt.savefig("consolidated_results.png", dpi=300)  
    plt.show()

# # Can be used separately.
# in_range_demand_scales = [0.25, 0.5, 0.75, 3.75, 4.0, 4.25]
# out_of_range_demand_scales = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]
# result_json_path = './results/eval_results_Feb03_08-42-43.json'
# plot_consolidated_results(result_json_path, in_range_demand_scales, out_of_range_demand_scales)