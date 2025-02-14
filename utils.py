import os
import xml
import time
import json
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
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

def truncate_colormap(cmap, minval=0.5, maxval=0.8, n=100):
    """
    Create a truncated colormap from an existing colormap.
    """
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_gradient_line(ax, x, y, cmap_name, label, lw, zorder):
    """
    Helper function to plot a gradient line.
    It creates line segments from the x and y coordinates, colors them
    using a truncated colormap (with a narrow color range so that the gradient
    is only a slight hint), and adds markers on top.
    """
    x = np.array(x)
    y = np.array(y)
    # Create line segments so that each segment can be colored individually.
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Use a truncated colormap over a narrow range so that the gradient is subtle.
    cmap_original = plt.get_cmap(cmap_name)
    cmap = truncate_colormap(cmap_original, 0.5, 0.8)
    norm = plt.Normalize(x.min(), x.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=lw, zorder=zorder)
    # Solid line (no dash).
    lc.set_array(x)
    ax.add_collection(lc)

    # Add markers at the data points.
    ax.scatter(x, y, c=x, cmap=cmap, norm=norm, zorder=zorder+1, edgecolor='k', s=50)

    # Create a dummy Line2D object for the legend that includes the line and
    # marker with a black border on the marker.
    mid_val = (x.min() + x.max()) / 2
    color = cmap(norm(mid_val))
    handle = Line2D([0], [0], color=color, lw=lw, marker='o', markersize=6,
                    markerfacecolor=color, markeredgecolor='k', label=label)
    return handle

def plot_consolidated_results(*result_json_paths, in_range_demand_scales):
    """
    Plot consolidated results for both TL and PPO.
    There are two subplots:
      • Left: Vehicle average waiting times
      • Right: Pedestrian average waiting times
    For each subplot, the x-axis shows the actual demand (scale × original demand)
    and the y-axis shows the average waiting time.
    Dotted shading indicates regions outside the valid (in-range) demand area.
    """
    # Original demand values.
    original_vehicle_demand = 201.54    # veh/hr
    original_pedestrian_demand = 2222.80  # ped/hr
    default_cmaps = ['Blues', 'Oranges', 'Greens', 'Purples', 'Reds', 'Greys']

    num_methods = len(result_json_paths)
    if num_methods == 3:
        labels = ['Unsignalized', 'TL', 'RL (Ours)']
    else:
        labels = [f"Method {i+1}" for i in range(num_methods)]

    results = []
    for json_path in result_json_paths:
        # get_averages() should return: scales, veh_avg, ped_avg
        scales, veh_avg, ped_avg = get_averages(json_path)
        results.append({'scales': scales, 'veh_avg': veh_avg, 'ped_avg': ped_avg})
    
    # Use the scales from the first JSON as a reference for ticks.
    sorted_scales = results[0]['scales']
    
    # Compute actual demand values for each result.
    for res in results:
        res['veh_x'] = res['scales'] * original_vehicle_demand
        res['ped_x'] = res['scales'] * original_pedestrian_demand

    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Plot each method’s data with its designated colormap.
    handles_vehicle = []
    handles_ped = []
    for i, res in enumerate(results):
        cmap = default_cmaps[i] if i < len(default_cmaps) else None  # Use None if we run out of defaults.
        handle_v = plot_gradient_line(ax1, res['veh_x'], res['veh_avg'], cmap_name=cmap,
                                      label=labels[i], lw=2, zorder=2)
        handle_p = plot_gradient_line(ax2, res['ped_x'], res['ped_avg'], cmap_name=cmap,
                                      label=labels[i], lw=2, zorder=2)
        handles_vehicle.append(handle_v)
        handles_ped.append(handle_p)

    # Set subplot titles and axis labels.
    ax1.set_title("Vehicle", fontweight="bold")
    ax2.set_title("Pedestrian", fontweight="bold")
    ax1.set_xlabel("Vehicle demand (veh/hr)")
    ax1.set_ylabel("Average waiting time (s)")
    ax2.set_xlabel("Pedestrian demand (ped/hr)")
    ax2.set_ylabel("Average waiting time (s)")

    # Determine the overall x-axis limits across all methods.
    veh_x_min = min(res['veh_x'].min() for res in results)
    veh_x_max = max(res['veh_x'].max() for res in results)
    ped_x_min = min(res['ped_x'].min() for res in results)
    ped_x_max = max(res['ped_x'].max() for res in results)
    veh_margin = 0.05 * (veh_x_max - veh_x_min)
    ped_margin = 0.05 * (ped_x_max - ped_x_min)
    veh_xlim = (veh_x_min - veh_margin, veh_x_max + veh_margin)
    ped_xlim = (ped_x_min - ped_margin, ped_x_max + ped_margin)
    ax1.set_xlim(veh_xlim)
    ax2.set_xlim(ped_xlim)

    # Format y-axis ticks to one decimal place.
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))

    # Set x-ticks (using the reference scales for spacing).
    veh_ticks = np.linspace(veh_xlim[0], veh_xlim[1], len(sorted_scales))[::2]
    ped_ticks = np.linspace(ped_xlim[0], ped_xlim[1], len(sorted_scales))[::2]
    ax1.set_xticks(veh_ticks)
    ax2.set_xticks(ped_ticks)
    veh_xtick_labels = [f"{int(round(val))}" for val in veh_ticks]
    ped_xtick_labels = [f"{int(round(val))}" for val in ped_ticks]
    ax1.set_xticklabels(veh_xtick_labels)
    ax2.set_xticklabels(ped_xtick_labels)

    # Set grid lines with short dashes.
    ax1.grid(True, linestyle=(0, (3, 3)), linewidth=0.85)
    ax2.grid(True, linestyle=(0, (3, 3)), linewidth=0.85)

    # Determine the valid (non-shaded) demand region.
    valid_min_scale = min(in_range_demand_scales)
    valid_max_scale = max(in_range_demand_scales)
    veh_valid_min = valid_min_scale * original_vehicle_demand
    veh_valid_max = valid_max_scale * original_vehicle_demand
    ped_valid_min = valid_min_scale * original_pedestrian_demand
    ped_valid_max = valid_max_scale * original_pedestrian_demand

    # Shade the areas outside the valid demand limits.
    ax1.axvspan(veh_xlim[0], veh_valid_min, facecolor='grey', alpha=0.2, zorder=0)
    ax1.axvspan(veh_valid_max, veh_xlim[1], facecolor='grey', alpha=0.2, zorder=0)
    ax2.axvspan(ped_xlim[0], ped_valid_min, facecolor='grey', alpha=0.2, zorder=0)
    ax2.axvspan(ped_valid_max, ped_xlim[1], facecolor='grey', alpha=0.2, zorder=0)

    # Create legends and bring them above the shaded areas.
    leg1 = ax1.legend(handles=handles_vehicle, loc='upper right')
    leg2 = ax2.legend(handles=handles_ped, loc='upper right')
    leg1.set_zorder(10)
    leg2.set_zorder(10)

    # Remove the top and right spines for a cleaner look.
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Adjust the y-axes so that both subplots use the same number of ticks.
    for ax in [ax1, ax2]:
        ymin, ymax = ax.get_ylim()
        yticks = ax.get_yticks()
        if len(yticks) > 1:
            spacing = yticks[-1] - yticks[-2]
        else:
            spacing = (ymax - ymin) / 5
        ax.set_ylim(ymin, ymax + spacing)
    n_ticks = max(len(ax1.get_yticks()), len(ax2.get_yticks()))
    ax1.yaxis.set_major_locator(LinearLocator(numticks=n_ticks))
    ax2.yaxis.set_major_locator(LinearLocator(numticks=n_ticks))
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))

    plt.tight_layout()
    plt.savefig("consolidated_results.pdf", dpi=300)
    plt.show()

def get_averages(result_json_path):
    """
    Helper function that reads a JSON file with results and returns the scales and the
    average waiting times for vehicles and pedestrians.
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

    scales = np.array(scales)
    veh_avg = np.array(veh_avg)
    ped_avg = np.array(ped_avg)

    sort_idx = np.argsort(scales)
    sorted_scales = scales[sort_idx]
    veh_avg = veh_avg[sort_idx]
    ped_avg = ped_avg[sort_idx]

    return sorted_scales, veh_avg, ped_avg

def plot_sampled_actions(actions_json):
    """
    Plot the sampled actions for PPO.
    """
    pass

# def plot_consolidated_results(tl_result_json_path, ppo_result_json_path, in_range_demand_scales, out_of_range_demand_scales):
#     """
    
#     """
#     # Get TL results.
#     tl_scales, tl_veh_avg, tl_ped_avg = get_averages(tl_result_json_path)
#     # Get PPO results.
#     ppo_scales, ppo_veh_avg, ppo_ped_avg = get_averages(ppo_result_json_path)

#     # If the evaluation scales differ, decide which array to use.
#     if not np.allclose(tl_scales, ppo_scales):
#         print("Warning: TL and PPO scales differ. Using TL scales for the x-axis.")
#     sorted_scales = tl_scales

#     # Original demand values.
#     original_vehicle_demand = 201.54    # veh/hr
#     original_pedestrian_demand = 2222.80  # ped/hr

#     # Compute actual demand values.
#     tl_veh_x = sorted_scales * original_vehicle_demand
#     tl_ped_x = sorted_scales * original_pedestrian_demand
#     ppo_veh_x = ppo_scales * original_vehicle_demand
#     ppo_ped_x = ppo_scales * original_pedestrian_demand

#     # Set a clean, publication-style theme with a white background.
#     sns.set_theme(style="whitegrid", context="talk")

#     # Create the figure and subplots.
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

#     # Plot the vehicles subplot with gradient solid lines.
#     handle_tl = plot_gradient_line(ax1, tl_veh_x, tl_veh_avg, cmap_name='Blues',
#                                    label="TL", lw=2, zorder=2)
#     handle_rl = plot_gradient_line(ax1, ppo_veh_x, ppo_veh_avg, cmap_name='Oranges',
#                                    label="RL (Ours)", lw=2, zorder=2)

#     # Plot the pedestrians subplot with gradient solid lines.
#     handle_tl_ped = plot_gradient_line(ax2, tl_ped_x, tl_ped_avg, cmap_name='Blues',
#                                        label="TL", lw=2, zorder=2)
#     handle_rl_ped = plot_gradient_line(ax2, ppo_ped_x, ppo_ped_avg, cmap_name='Oranges',
#                                        label="RL (Ours)", lw=2, zorder=2)

#     # Set subplot titles and axis labels.
#     ax1.set_title("Vehicle", fontweight="bold")
#     ax2.set_title("Pedestrian", fontweight="bold")
#     ax1.set_xlabel("Vehicle demand (veh/hr)")
#     ax1.set_ylabel("Average waiting time (s)")
#     ax2.set_xlabel("Pedestrian demand (ped/hr)")
#     ax2.set_ylabel("Average waiting time (s)")

#     # Compute a 5% margin for the x-axis limits.
#     veh_x_min = min(ppo_veh_x.min(), tl_veh_x.min())
#     veh_x_max = max(ppo_veh_x.max(), tl_veh_x.max())
#     ped_x_min = min(ppo_ped_x.min(), tl_ped_x.min())
#     ped_x_max = max(ppo_ped_x.max(), tl_ped_x.max())
#     veh_margin = 0.05 * (veh_x_max - veh_x_min)
#     ped_margin = 0.05 * (ped_x_max - ped_x_min)
#     veh_xlim = (veh_x_min - veh_margin, veh_x_max + veh_margin)
#     ped_xlim = (ped_x_min - ped_margin, ped_x_max + ped_margin)
#     ax1.set_xlim(veh_xlim)
#     ax2.set_xlim(ped_xlim)

#     # Format y-axis ticks to one decimal place.
#     ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
#     ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))

#     # Reduce the number of x-ticks and customize tick labels.
#     veh_ticks = np.linspace(veh_xlim[0], veh_xlim[1], len(sorted_scales))[::2]
#     ped_ticks = np.linspace(ped_xlim[0], ped_xlim[1], len(sorted_scales))[::2]
#     ax1.set_xticks(veh_ticks)
#     ax2.set_xticks(ped_ticks)
#     veh_xtick_labels = [f"{int(round(val))}" for val in veh_ticks]
#     ped_xtick_labels = [f"{int(round(val))}" for val in ped_ticks]
#     ax1.set_xticklabels(veh_xtick_labels)
#     ax2.set_xticklabels(ped_xtick_labels)

#     # Set grid lines with shorter dashes.
#     ax1.grid(True, linestyle=(0, (3, 3)), linewidth=0.85)
#     ax2.grid(True, linestyle=(0, (3, 3)), linewidth=0.85)

#     # Determine the valid (non-shaded) demand region using the in_range_demand_scales.
#     valid_min_scale = min(in_range_demand_scales)
#     valid_max_scale = max(in_range_demand_scales)
#     veh_valid_min = valid_min_scale * original_vehicle_demand
#     veh_valid_max = valid_max_scale * original_vehicle_demand
#     ped_valid_min = valid_min_scale * original_pedestrian_demand
#     ped_valid_max = valid_max_scale * original_pedestrian_demand

#     # Shade the areas outside the valid demand limits.
#     ax1.axvspan(veh_xlim[0], veh_valid_min, facecolor='grey', alpha=0.2, zorder=0)
#     ax1.axvspan(veh_valid_max, veh_xlim[1], facecolor='grey', alpha=0.2, zorder=0)
#     ax2.axvspan(ped_xlim[0], ped_valid_min, facecolor='grey', alpha=0.2, zorder=0)
#     ax2.axvspan(ped_valid_max, ped_xlim[1], facecolor='grey', alpha=0.2, zorder=0)

#     # Place legends in the upper right corners and lift them above shading.
#     leg1 = ax1.legend(handles=[handle_tl, handle_rl], loc='upper right')
#     leg2 = ax2.legend(handles=[handle_tl_ped, handle_rl_ped], loc='upper right')
#     leg1.set_zorder(10)
#     leg2.set_zorder(10)

#     # Remove the top and right spines for a cleaner look.
#     for ax in [ax1, ax2]:
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         # Optionally, you can also adjust the bottom and left spines if needed.

#     # Adjust y-axes so that both subplots use the same number of ticks.
#     for ax in [ax1, ax2]:
#         ymin, ymax = ax.get_ylim()
#         yticks = ax.get_yticks()
#         if len(yticks) > 1:
#             spacing = yticks[-1] - yticks[-2]
#         else:
#             spacing = (ymax - ymin) / 5
#         ax.set_ylim(ymin, ymax + spacing)
#     n_ticks = max(len(ax1.get_yticks()), len(ax2.get_yticks()))
#     ax1.yaxis.set_major_locator(LinearLocator(numticks=n_ticks))
#     ax2.yaxis.set_major_locator(LinearLocator(numticks=n_ticks))
#     ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
#     ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))

#     plt.tight_layout()
#     plt.savefig("consolidated_results.pdf", dpi=300)
#     plt.show()