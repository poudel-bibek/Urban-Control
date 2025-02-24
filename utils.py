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
from matplotlib.gridspec import GridSpec

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

def plot_gradient_line(ax, x, y, std=None, cmap_name='Blues', label='', lw=2, zorder=2):
    """
    Helper function to plot a gradient line with optional standard deviation shading.
    """
    x = np.array(x)
    y = np.array(y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Use a truncated colormap over a narrow range for subtle gradient
    cmap_original = plt.get_cmap(cmap_name)
    cmap = truncate_colormap(cmap_original, 0.5, 0.8)
    norm = plt.Normalize(x.min(), x.max())

    # If std is provided, add shaded region for standard deviation
    if std is not None:
        # Create gradient colors for the fill region
        colors = cmap(norm(x))
        # Add alpha channel to make fill slightly transparent
        colors = np.array([(*c[:-1], 0.3) for c in colors])
        
        # Plot the shaded region
        ax.fill_between(x, y - std, y + std, 
                       color=colors,
                       zorder=zorder)

    # Plot the line with gradient
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=lw, zorder=zorder+1)
    lc.set_array(x)
    ax.add_collection(lc)

    # Add markers
    ax.scatter(x, y, c=x, cmap=cmap, norm=norm, zorder=zorder+2, edgecolor='k', s=50)

    # Create dummy line for legend
    mid_val = (x.min() + x.max()) / 2
    color = cmap(norm(mid_val))
    handle = Line2D([0], [0], color=color, lw=lw, marker='o', markersize=6,
                    markerfacecolor=color, markeredgecolor='k', label=label)
    return handle

def plot_individual_results(*result_json_paths, in_range_demand_scales, total=False, show_scales=False):
    """
    Plot consolidated results with standard deviation shading.
    Parameters:
        *result_json_paths: Paths to result JSON files
        in_range_demand_scales: List of scales considered in-range
        total: If True, plot total wait times instead of averages
        show_scales: If True, show scale labels (e.g. 0.5x) below demand values
    """
    # Original demand values.
    original_vehicle_demand = 201.54    # veh/hr
    original_pedestrian_demand = 2222.80  # ped/hr
    default_cmaps = ['Blues', 'Oranges', 'Greens', 'Purples', 'Reds', 'Greys']

    num_methods = len(result_json_paths)
    if num_methods == 3:
        labels = ['Unsignalized', 'TL', 'RL (Ours)']
    else:
        labels = ['TL', 'RL (Ours)']

    results = []
    for json_path in result_json_paths:
        scales, veh_mean, ped_mean, veh_std, ped_std = get_averages(json_path, total=total)
        results.append({
            'scales': scales, 
            'veh_mean': veh_mean, 
            'ped_mean': ped_mean,
            'veh_std': veh_std,
            'ped_std': ped_std
        })
    
    # Compute actual demand values
    for res in results:
        res['veh_x'] = res['scales'] * original_vehicle_demand
        res['ped_x'] = res['scales'] * original_pedestrian_demand

    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Plot each method's data with gradient lines and standard deviation shading
    handles_vehicle = []
    handles_ped = []
    for i, res in enumerate(results):
        cmap = default_cmaps[i] if i < len(default_cmaps) else None
        
        handle_v = plot_gradient_line(ax1, res['veh_x'], res['veh_mean'], 
                                    std=res['veh_std'], cmap_name=cmap,
                                    label=labels[i], lw=2, zorder=2)
        handle_p = plot_gradient_line(ax2, res['ped_x'], res['ped_mean'], 
                                    std=res['ped_std'], cmap_name=cmap,
                                    label=labels[i], lw=2, zorder=2)
        
        handles_vehicle.append(handle_v)
        handles_ped.append(handle_p)

    # Set base font size
    fs = 19
    
    # Set subplot titles and axis labels
    time_type_veh = "Total wait time" if total else "Average wait time per vehicle"
    time_type_ped = "Total wait time" if total else "Average wait time per pedestrian"
    ax1.set_title("Vehicle", fontweight="bold", fontsize=fs)
    ax2.set_title("Pedestrian", fontweight="bold", fontsize=fs)
    ax1.set_xlabel("Demand (veh/hr)", fontsize=fs)
    ax2.set_xlabel("Demand (ped/hr)", fontsize=fs)
    ax1.set_ylabel(f"{time_type_veh} (s)", fontsize=fs)
    ax2.set_ylabel(f"{time_type_ped} (s)", fontsize=fs)

    # Set tick label sizes
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax2.tick_params(axis='both', which='major', labelsize=fs-2)

    # Format y-axis values
    def format_avg_ticks(x, _):
        return f"{x:.1f}"
    
    def format_total_ticks(x, _):
        return f"{(x/1000):.1f}"
    
    # Apply formatters
    ax1.yaxis.set_major_formatter(FuncFormatter(format_total_ticks))
    ax2.yaxis.set_major_formatter(FuncFormatter(format_total_ticks))
    
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

    # Set x-ticks using the actual scale points from the data
    all_scales = np.unique(np.concatenate([res['scales'] for res in results]))  # Get unique scales from results
    veh_ticks = [scale * original_vehicle_demand for scale in all_scales]
    ped_ticks = [scale * original_pedestrian_demand for scale in all_scales]
    
    # Use every other tick but include the last one
    veh_ticks = veh_ticks[::2]
    ped_ticks = ped_ticks[::2]
    if veh_ticks[-1] != all_scales[-1] * original_vehicle_demand:
        veh_ticks = np.append(veh_ticks, all_scales[-1] * original_vehicle_demand)
    if ped_ticks[-1] != all_scales[-1] * original_pedestrian_demand:
        ped_ticks = np.append(ped_ticks, all_scales[-1] * original_pedestrian_demand)
    
    # Get corresponding scales for labels
    scales_for_labels = list(all_scales[::2])
    if scales_for_labels[-1] != all_scales[-1]:
        scales_for_labels.append(all_scales[-1])
    
    ax1.set_xticks(veh_ticks)
    ax2.set_xticks(ped_ticks)
    
    # Create main tick labels (demand values)
    veh_xtick_labels = [f"{int(round(val))}" for val in veh_ticks]
    ped_xtick_labels = [f"{int(round(val))}" for val in ped_ticks]
    
    if show_scales:
        # Create scale labels using the actual scales - simplified format
        veh_scales = [f"{scale:g}x" for scale in scales_for_labels]  # :g removes unnecessary zeros
        ped_scales = [f"{scale:g}x" for scale in scales_for_labels]  # :g removes unnecessary zeros
        
        ax1.set_xticklabels([f"{val}\n{scale}" for val, scale in zip(veh_xtick_labels, veh_scales)])
        ax2.set_xticklabels([f"{val}\n{scale}" for val, scale in zip(ped_xtick_labels, ped_scales)])
    else:
        ax1.set_xticklabels(veh_xtick_labels)
        ax2.set_xticklabels(ped_xtick_labels)
    
    # Make the tick labels slightly smaller
    for ax in [ax1, ax2]:
        for tick in ax.get_xticklabels():
            tick.set_fontsize(fs-4)

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

    # Create a single legend at the bottom
    handles = handles_vehicle  # We can use either handles_vehicle or handles_ped since they're the same
    leg = fig.legend(handles=handles, 
                    loc='center', 
                    bbox_to_anchor=(0.5, 0.02),
                    ncol=len(handles),  # Place handles horizontally
                    frameon=True,  # Add the frame
                    framealpha=1.0,  # Solid background
                    edgecolor='gray',  # Gray edge color (more subtle)
                    fancybox=True,  # Rounded corners
                    shadow=False,  # Add shadow
                    bbox_transform=fig.transFigure,
                    fontsize=fs-4)  # Legend text slightly smaller than titles

    # Adjust the layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # Ensure same number of y-ticks for both subplots
    ax1_yticks = ax1.get_yticks()
    ax2_yticks = ax2.get_yticks()
    n_ticks = max(len(ax1_yticks), len(ax2_yticks))
    
    # Get y-limits for both plots
    ax1_ymin, ax1_ymax = ax1.get_ylim()
    ax2_ymin, ax2_ymax = ax2.get_ylim()
    
    # Set same number of ticks for both plots
    ax1.yaxis.set_major_locator(LinearLocator(n_ticks))
    ax2.yaxis.set_major_locator(LinearLocator(n_ticks))
    
    # Reset the limits to avoid auto-adjustment
    ax1.set_ylim(ax1_ymin, ax1_ymax)
    ax2.set_ylim(ax2_ymin, ax2_ymax)

    # Save with bbox_inches to ensure legend is included in saved file
    plt.savefig(f"./results/individual_results_{'total' if total else 'avg'}.pdf", 
                dpi=300, 
                bbox_inches='tight')
    plt.show()

def get_averages(result_json_path, total=False):
    """
    Helper function that reads a JSON file with results and returns the scales,
    means and standard deviations for vehicles and pedestrians.
    """
    with open(result_json_path, 'r') as f:
        results = json.load(f)

    scales, veh_mean, ped_mean = [], [], []
    veh_std, ped_std = [], []
    
    for scale_str, runs in results.items():
        scale = float(scale_str)
        scales.append(scale)
        veh_vals = []
        ped_vals = []
        
        for run in runs.values():
            if total:
                veh_vals.append(run["total_veh_waiting_time"])
                ped_vals.append(run["total_ped_waiting_time"])
            else:
                veh_vals.append(run["veh_avg_waiting_time"])
                ped_vals.append(run["ped_avg_waiting_time"])
                
        veh_mean.append(np.mean(veh_vals))
        ped_mean.append(np.mean(ped_vals))
        veh_std.append(np.std(veh_vals))
        ped_std.append(np.std(ped_vals))

    # Convert to numpy arrays and sort by scale
    scales = np.array(scales)
    sort_idx = np.argsort(scales)
    
    return (scales[sort_idx], 
            np.array(veh_mean)[sort_idx], 
            np.array(ped_mean)[sort_idx],
            np.array(veh_std)[sort_idx],
            np.array(ped_std)[sort_idx])

def count_consecutive_ones_filtered(actions):
    """
    Helper function to count consecutive occurrences of 1's in the mid-block actions list.
    The first action (corresponding to intersection) in each list is ignored.
    """
    counts = []
    count = 0
    for action in actions[1:]:  # Ignore the first action
        if action == 1:
            count += 1
        else:
            if count > 0:
                counts.append(count)
            count = 0
    if count > 0:
        counts.append(count)  # Add the last streak if it ends at the last element
    return counts

def plot_avg_consecutive_ones(file_path):
    """
    Plots the average sum of consecutive occurrences of '1's per training iteration.
    The first action in each list is ignored.
    
    Parameters:
        data (dict): Dictionary where keys are iterations and values are lists of action sequences.
    """
    with open(file_path, "r") as file:
        data = json.load(file)

    # Compute the average sum of consecutive 1's per iteration
    avg_consecutive_ones_per_iteration = []
    iterations = []

    for iteration, actions_list in data.items():
        iteration = int(iteration)  # Convert iteration key to integer
        consecutive_ones = [count_consecutive_ones_filtered(action_list) for action_list in actions_list]
        avg_consecutive_ones = np.mean([sum(seq) for seq in consecutive_ones if seq]) if consecutive_ones else 0
        iterations.append(iteration)
        avg_consecutive_ones_per_iteration.append(avg_consecutive_ones)

    # Sort by iteration
    iterations, avg_consecutive_ones_per_iteration = zip(*sorted(zip(iterations, avg_consecutive_ones_per_iteration)))
    iterations = np.array(iterations)
    avg_consecutive_ones_per_iteration = np.array(avg_consecutive_ones_per_iteration)
    
    # Set style and create figure
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot with green gradient
    handle = plot_gradient_line(ax, iterations, avg_consecutive_ones_per_iteration, 
                              cmap_name='Greens', label="Training Progress", lw=3, zorder=2)
    
    # Set font size
    fs = 28  # Increased base font size
    
    # Set labels with increased font size
    ax.set_xlabel("Training Iteration", fontsize=fs)
    ax.set_ylabel("Avg. Sum of Consecutive 1's", fontsize=fs)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.grid(True, linestyle=(0, (3, 3)), linewidth=0.85)
    
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=fs-4)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("./results/sampled_actions.pdf", dpi=300)
    plt.show()

def plot_consolidated_results(*json_paths, in_range_demand_scales, show_scales=True):
    """
    Plot consolidated results from multiple JSON files into a single figure with 4 subplots.
    """
    # Original demand values
    original_vehicle_demand = 201.54    # veh/hr
    original_pedestrian_demand = 2222.80  # ped/hr
    default_cmaps = ['Blues', 'Oranges', 'Greens', 'Purples', 'Reds', 'Greys']
    
    # Set style
    sns.set_theme(style="whitegrid", context="talk")
    
    # Set up the figure with a 2x2 grid
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 2, figure=fig)
    
    # Create subplots with shared x-axes
    ax_ped_avg = fig.add_subplot(gs[0, 0])
    ax_ped_total = fig.add_subplot(gs[1, 0], sharex=ax_ped_avg)
    ax_veh_avg = fig.add_subplot(gs[0, 1])
    ax_veh_total = fig.add_subplot(gs[1, 1], sharex=ax_veh_avg)
    
    # Store legend handles and labels
    legend_handles = []
    legend_labels = []
    
    # Calculate valid demand ranges
    valid_min_scale = min(in_range_demand_scales)
    valid_max_scale = max(in_range_demand_scales)
    veh_valid_min = valid_min_scale * original_vehicle_demand
    veh_valid_max = valid_max_scale * original_vehicle_demand
    ped_valid_min = valid_min_scale * original_pedestrian_demand
    ped_valid_max = valid_max_scale * original_pedestrian_demand
    
    # Set labels dictionary for methods
    if len(json_paths) == 3:
        labels = ['Unsignalized', 'TL', 'RL (Ours)']
    else:
        labels = ['TL', 'RL (Ours)']
    
    # First get the data range to set proper limits
    all_ped_demands = []
    all_veh_demands = []
    for json_path in json_paths:
        scales, _, _, _, _ = get_averages(json_path, total=False)
        all_ped_demands.extend(scales * original_pedestrian_demand)
        all_veh_demands.extend(scales * original_vehicle_demand)
    
    # Calculate the limits with symmetrical margins (like in plot_individual_results)
    ped_min, ped_max = min(all_ped_demands), max(all_ped_demands)
    veh_min, veh_max = min(all_veh_demands), max(all_veh_demands)
    ped_margin = 0.05 * (ped_max - ped_min)  # 5% margin on both sides
    veh_margin = 0.05 * (veh_max - veh_min)  # 5% margin on both sides
    
    # Set the limits for all plots with symmetrical margins
    for ax in [ax_ped_avg, ax_ped_total]:
        ax.set_xlim(ped_min - ped_margin, ped_max + ped_margin)
    for ax in [ax_veh_avg, ax_veh_total]:
        ax.set_xlim(veh_min - veh_margin, veh_max + veh_margin)
    
    # Now add the shading using the full plot limits
    for ax in [ax_ped_avg, ax_ped_total]:
        xlim = ax.get_xlim()
        ax.axvspan(xlim[0], ped_valid_min, facecolor='grey', alpha=0.25, zorder=-1)
        ax.axvspan(ped_valid_max, xlim[1], facecolor='grey', alpha=0.25, zorder=-1)
    
    for ax in [ax_veh_avg, ax_veh_total]:
        xlim = ax.get_xlim()
        ax.axvspan(xlim[0], veh_valid_min, facecolor='grey', alpha=0.25, zorder=-1)
        ax.axvspan(veh_valid_max, xlim[1], facecolor='grey', alpha=0.25, zorder=-1)
    
    for idx, json_path in enumerate(json_paths):
        # Get data using the helper function
        scales, veh_mean, ped_mean, veh_std, ped_std = get_averages(json_path, total=False)
        _, veh_total, ped_total, veh_total_std, ped_total_std = get_averages(json_path, total=True)
        
        # Use labels instead of raw method name
        method_name = labels[idx]
        
        # Convert scales to actual demands
        veh_demands = scales * original_vehicle_demand
        ped_demands = scales * original_pedestrian_demand
        
        # Get color map for this method
        cmap = default_cmaps[idx] if idx < len(default_cmaps) else None
        
        # Plot pedestrian data with gradient lines
        handle_ped_avg = plot_gradient_line(ax_ped_avg, ped_demands, ped_mean, 
                                          std=ped_std, cmap_name=cmap,
                                          label=method_name, lw=2, zorder=2)
        
        handle_ped_total = plot_gradient_line(ax_ped_total, ped_demands, ped_total, 
                                            std=ped_total_std, cmap_name=cmap,
                                            label=method_name, lw=2, zorder=2)
        
        # Plot vehicle data with gradient lines
        handle_veh_avg = plot_gradient_line(ax_veh_avg, veh_demands, veh_mean,
                                          std=veh_std, cmap_name=cmap,
                                          label=method_name, lw=2, zorder=2)
        
        handle_veh_total = plot_gradient_line(ax_veh_total, veh_demands, veh_total,
                                            std=veh_total_std, cmap_name=cmap,
                                            label=method_name, lw=2, zorder=2)
        
        # Add to legend handles and labels (remove the if idx == 0 condition)
        legend_handles.append(handle_ped_avg)
        legend_labels.append(method_name)
    
    # Set grid style with short dashes
    for ax in [ax_ped_avg, ax_ped_total, ax_veh_avg, ax_veh_total]:
        ax.grid(True, linestyle=(0, (3, 3)), linewidth=0.85)
    
    fs = 18  # Base font size
    
    # Set titles
    ax_ped_avg.set_title('Pedestrian', fontweight='bold', fontsize=fs)
    ax_veh_avg.set_title('Vehicle', fontweight='bold', fontsize=fs)
    
    # Set y-axis labels with consistent alignment
    fig.text(0.039, 0.74, 'Average Wait Time (s)', va='center', rotation='vertical', fontsize=fs-2)
    fig.text(0.039, 0.32, 'Total Wait Time (×10³ s)', va='center', rotation='vertical', fontsize=fs-2)
    
    # Right side (Vehicle)
    fig.text(0.50, 0.74, 'Average Wait Time (s)', va='center', rotation='vertical', fontsize=fs-2)
    fig.text(0.50, 0.32, 'Total Wait Time (×10³ s)', va='center', rotation='vertical', fontsize=fs-2)
    
    # Set x-labels
    ax_ped_avg.set_xlabel('')  # Remove label from top plots
    ax_veh_avg.set_xlabel('')
    if show_scales:
        ax_ped_total.set_xlabel('Demand Scale', fontsize=fs-2)
        ax_veh_total.set_xlabel('Demand Scale', fontsize=fs-2)
    else:
        ax_ped_total.set_xlabel('Demand (ped/hr)', fontsize=fs-2)
        ax_veh_total.set_xlabel('Demand (veh/hr)', fontsize=fs-2)
    
    # Set tick sizes for all axes
    for ax in [ax_ped_avg, ax_ped_total, ax_veh_avg, ax_veh_total]:
        ax.tick_params(axis='both', which='major', labelsize=fs-2)
    
    # Set consistent x-ticks for all subplots
    all_scales = np.unique(scales)  # Get unique scales
    veh_ticks = [scale * original_vehicle_demand for scale in all_scales]
    ped_ticks = [scale * original_pedestrian_demand for scale in all_scales]
    
    # Use every other tick but include the last one
    veh_ticks = veh_ticks[::2]
    ped_ticks = ped_ticks[::2]
    if veh_ticks[-1] != all_scales[-1] * original_vehicle_demand:
        veh_ticks = np.append(veh_ticks, all_scales[-1] * original_vehicle_demand)
    if ped_ticks[-1] != all_scales[-1] * original_pedestrian_demand:
        ped_ticks = np.append(ped_ticks, all_scales[-1] * original_pedestrian_demand)
    
    # Get corresponding scales for labels
    scales_for_labels = list(all_scales[::2])
    if scales_for_labels[-1] != all_scales[-1]:
        scales_for_labels.append(all_scales[-1])
    
    # Set ticks for all subplots
    ax_ped_avg.set_xticks(ped_ticks)
    ax_ped_total.set_xticks(ped_ticks)
    ax_veh_avg.set_xticks(veh_ticks)
    ax_veh_total.set_xticks(veh_ticks)
    
    # Create main tick labels (demand values)
    veh_xtick_labels = [f"{int(round(val))}" for val in veh_ticks]
    ped_xtick_labels = [f"{int(round(val))}" for val in ped_ticks]
    
    # Format y-axis values
    def format_avg_ticks(x, _):
        return f"{x:.1f}"
    
    def format_total_ticks(x, _):
        return f"{(x/1000):.1f}"
    
    # Apply formatters
    ax_ped_avg.yaxis.set_major_formatter(FuncFormatter(format_avg_ticks))
    ax_veh_avg.yaxis.set_major_formatter(FuncFormatter(format_avg_ticks))
    ax_ped_total.yaxis.set_major_formatter(FuncFormatter(format_total_ticks))
    ax_veh_total.yaxis.set_major_formatter(FuncFormatter(format_total_ticks))
    
    # Ensure same number of ticks for all plots
    n_ticks = 7  # Set desired number of ticks
    
    # Get current limits for each plot
    ped_avg_ymin, ped_avg_ymax = ax_ped_avg.get_ylim()
    veh_avg_ymin, veh_avg_ymax = ax_veh_avg.get_ylim()
    ped_total_ymin, ped_total_ymax = ax_ped_total.get_ylim()
    veh_total_ymin, veh_total_ymax = ax_veh_total.get_ylim()
    
    # Set ticks for each plot
    ax_ped_avg.yaxis.set_major_locator(plt.LinearLocator(n_ticks))
    ax_veh_avg.yaxis.set_major_locator(plt.LinearLocator(n_ticks))
    ax_ped_total.yaxis.set_major_locator(plt.LinearLocator(n_ticks))
    ax_veh_total.yaxis.set_major_locator(plt.LinearLocator(n_ticks))
    
    # Reset the limits to avoid auto-adjustment
    ax_ped_avg.set_ylim(ped_avg_ymin, ped_avg_ymax)
    ax_veh_avg.set_ylim(veh_avg_ymin, veh_avg_ymax)
    ax_ped_total.set_ylim(ped_total_ymin, ped_total_ymax)
    ax_veh_total.set_ylim(veh_total_ymin, veh_total_ymax)
    
    # Remove x-ticks and labels from top plots
    ax_ped_avg.set_xticklabels([])
    ax_veh_avg.set_xticklabels([])
    
    # Set ticks for bottom plots only
    if show_scales:
        scale_labels = [f"{scale:g}x" for scale in scales_for_labels]
        scale_labels[-1] = ''
        ax_ped_total.set_xticklabels(scale_labels, fontsize=fs-2)
        ax_veh_total.set_xticklabels(scale_labels, fontsize=fs-2)
    else:
        ped_xtick_labels[-1] = ''
        veh_xtick_labels[-1] = ''
        ax_ped_total.set_xticklabels(ped_xtick_labels, fontsize=fs-2)
        ax_veh_total.set_xticklabels(veh_xtick_labels, fontsize=fs-2)
    
    # Legend with consistent font size
    leg = fig.legend(legend_handles, legend_labels, 
                    loc='center', 
                    bbox_to_anchor=(0.5, 0.04),
                    ncol=len(legend_handles),
                    frameon=True,
                    framealpha=1.0,
                    edgecolor='gray',
                    fancybox=True,
                    shadow=False,
                    bbox_transform=fig.transFigure,
                    fontsize=fs-2)
    
    # Adjust subplot spacing
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, wspace=0.25)
    plt.savefig("./results/consolidated_results.pdf", bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


###### CONSOLIDATED PLOT ######
# unsignalized_results_path = "./results/results_unsignalized.json"
# tl_results_path = "./results/results_tl.json"
# ppo_results_path = "./results/eval_ppo_akash_best.json"

# plot_consolidated_results(unsignalized_results_path, 
#                          tl_results_path, 
#                          ppo_results_path,
#                          in_range_demand_scales=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25])

######  Plot samples 1's ###### 
# sampled_actions_file_path = "./saved_models/Feb23_11-20-53/sampled_actions.json"
# plot_avg_consecutive_ones(sampled_actions_file_path)


###### SEPARATE PLOTS FOR AVERAGE AND TOTAL ######
# unsignalized_results_path = "./results/eval_Feb17_08-17-07/eval_Feb16_13-09-44_unsignalized.json"
# tl_results_path = "./results/eval_Feb18_16-13-39/eval_Feb17_17-36-27_tl.json"
# ppo_results_path = "./results/eval_Feb18_16-13-39/eval_Feb17_17-36-27_ppo.json"

# plot_individual_results(tl_results_path,
#                             ppo_results_path,
#                             in_range_demand_scales = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25])

# plot_individual_results(tl_results_path,
#                             ppo_results_path,
#                             in_range_demand_scales = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25],
#                             total=True)
