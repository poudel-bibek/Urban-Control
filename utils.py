import os
import xml
import time
import json
import logging
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, LinearLocator
from matplotlib.ticker import MaxNLocator, MultipleLocator
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
    Helper function to count consecutive occurrences of 1's in the action list.
    The first action (corresponding to intersection) is ignored.
    Returns a list where each element is the length of a consecutive sequence of 1's.

    Example:
    [0, 1, 1, 0, 1, 0, 0, 1, 1, 1] → [2, 1, 3]
    """
    if not actions or len(actions) <= 1:
        return []

    counts = []
    count = 0

    # Start from the second action (index 1)
    for action in actions[1:]:
        if action == 1:
            count += 1
        else:
            if count > 0:
                counts.append(count)
                count = 0

    # Don't forget to add the last sequence if it ends with 1's
    if count > 0:
        counts.append(count)

    return counts

def plot_avg_consecutive_ones(file_path, output_path="./results/sampled_actions_retro.pdf"):
    """
    Creates a clean, professional plot of the average sum of consecutive occurrences of '1's
    per training iteration with a vibrant appearance.

    Parameters:
        file_path (str): Path to the JSON file containing the data.
        output_path (str): Path to save the output PDF file.
    """

    # Load data
    with open(file_path, "r") as file:
        data = json.load(file)

    # Compute the average sum of consecutive 1's per iteration
    avg_consecutive_ones_per_iteration = []
    iterations = []

    for iteration, actions_list in data.items():
        iteration = int(iteration)  # Convert iteration key to integer
        consecutive_ones = [count_consecutive_ones_filtered(action_list) for action_list in actions_list]

        # Calculate the sum of consecutive 1's for each sample, then average across samples
        sums_of_consecutive_ones = [sum(seq) for seq in consecutive_ones if seq]
        avg_consecutive_ones = np.mean(sums_of_consecutive_ones) if sums_of_consecutive_ones else 0

        iterations.append(iteration)
        avg_consecutive_ones_per_iteration.append(avg_consecutive_ones)

    # Sort by iteration
    iterations, avg_consecutive_ones_per_iteration = zip(*sorted(zip(iterations, avg_consecutive_ones_per_iteration)))
    iterations = np.array(iterations)
    avg_consecutive_ones_per_iteration = np.array(avg_consecutive_ones_per_iteration)

    # Set base font size
    fs = 24  # Base font size - adjust this to change all font sizes proportionally

    # Set up the figure with a clean style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.major.size'] = 0
    plt.rcParams['ytick.major.size'] = 0

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')

    # Set background color
    ax.set_facecolor('white')

    # Calculate y-axis limits with some padding
    y_min = min(avg_consecutive_ones_per_iteration) * 0.9
    y_max = max(avg_consecutive_ones_per_iteration) * 1.1

    # Calculate x-axis limits with added margins
    x_min = min(iterations) - (max(iterations) - min(iterations)) * 0.05  # 5% margin on left
    x_max = max(iterations) + (max(iterations) - min(iterations)) * 0.05  # 5% margin on right

    # Set axis limits
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    # Format y-axis with one decimal place
    def format_with_decimals(x, pos):
        return f'{x:.1f}'

    ax.yaxis.set_major_formatter(FuncFormatter(format_with_decimals))

    # Add light grid lines with slightly more visibility
    ax.grid(True, linestyle='-', alpha=0.15, color='#333333')
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Use a more vibrant blue for the data points
    VIBRANT_BLUE = '#2E5EAA'  # More vibrant blue for data points

    # Create scatter plot with more vibrant, semi-transparent circles
    scatter = ax.scatter(iterations, avg_consecutive_ones_per_iteration,
                        s=110, edgecolors=VIBRANT_BLUE, facecolors='none',
                        linewidth=2.0, alpha=0.75, zorder=3)

    # Fit a trend line
    z = np.polyfit(iterations, avg_consecutive_ones_per_iteration, 1)
    p = np.poly1d(z)

    # Create x values for the trend line (only within the data range)
    x_trend = np.linspace(min(iterations), max(iterations), 100)
    y_trend = p(x_trend)

    # Use a very dark blue color for the trend line - almost navy blue
    VERY_DARK_BLUE = '#0A2472'  # Very dark blue/navy color

    # Plot the trend line as a solid, very dark line
    trend_line = ax.plot(x_trend, y_trend, color=VERY_DARK_BLUE, linewidth=4.0, zorder=4)

    # Set labels with increased font size and more vibrant color
    LABEL_COLOR = '#1A1A1A'  # Slightly lighter than pure black for better contrast
    ax.set_xlabel('Training Iteration', fontsize=fs*1.2, labelpad=10, color=LABEL_COLOR)
    ax.set_ylabel('# of Synchronized Green Signals', fontsize=fs*1.2, labelpad=10, color=LABEL_COLOR)

    # Line for trend line - use the very dark blue color
    trend_line_handle = mlines.Line2D([], [], color=VERY_DARK_BLUE, linewidth=4.0,
                                     label='Trend Line')

    # Add the legend with the proper handles
    ax.legend(handles=[trend_line_handle],
             loc='upper right', frameon=True, framealpha=0.9,
             edgecolor='#CCCCCC', fontsize=fs)

    # Add padding between y-axis and tick labels
    ax.tick_params(axis='y', pad=8)  # Add padding between y-axis and y-tick labels

    # Customize tick parameters with larger font size and more vibrant color
    ax.tick_params(axis='both', colors=LABEL_COLOR, labelsize=fs)

    # Add a subtle border around the plot with slightly more visible color
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#AAAAAA')  # Slightly darker border
        ax.spines[spine].set_linewidth(1.2)  # Slightly thicker border

    # Add more padding around the entire plot
    plt.tight_layout(pad=2.0)

    # Save with extra padding
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.show()

    print(f"Plot saved to {output_path}")

def plot_main_results(*json_paths, in_range_demand_scales, show_scales=True):
    """
    Plot consolidated results from multiple JSON files into a single figure with 4 subplots.
    """
    # Original demand values
    original_vehicle_demand = 201.54    # veh/hr
    original_pedestrian_demand = 2222.80  # ped/hr

    # Custom color map assignment - TL orange, unsignalized blue, RL green
    custom_cmaps = ['Oranges', 'Blues', 'Greens']

    # Set style
    sns.set_theme(style="whitegrid", context="talk")

    # Set up the figure with a 2x2 grid
    fig = plt.figure(figsize=(16, 7))
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

    # Set labels dictionary for methods and reorder TL to be first
    if len(json_paths) == 3:
        # Get the indices of each method
        tl_idx = [i for i, path in enumerate(json_paths) if 'tl' in path.lower()][0]
        us_idx = [i for i, path in enumerate(json_paths) if 'unsignalized' in path.lower()][0]
        rl_idx = [i for i, path in enumerate(json_paths) if 'ppo' in path.lower()][0]

        # Reorder json_paths to have TL first
        json_paths = list(json_paths)
        json_paths = [json_paths[tl_idx], json_paths[us_idx], json_paths[rl_idx]]
        labels = ['Signalized', 'Unsignalized', 'RL (Ours)']
    else:
        # Get the indices of each method
        tl_idx = [i for i, path in enumerate(json_paths) if 'tl' in path.lower()][0]
        rl_idx = [i for i, path in enumerate(json_paths) if 'ppo' in path.lower()][0]

        # Reorder json_paths to have TL first
        json_paths = list(json_paths)
        json_paths = [json_paths[tl_idx], json_paths[rl_idx]]
        labels = ['Signalized', 'RL (Ours)']

    # First get the data range to set proper limits
    all_ped_demands = []
    all_veh_demands = []
    for json_path in json_paths:
        scales, _, _, _, _ = get_averages(json_path, total=False)
        all_ped_demands.extend(scales * original_pedestrian_demand)
        all_veh_demands.extend(scales * original_vehicle_demand)

    # Calculate the limits with symmetrical margins 
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

        # Get color map for this method - use custom color assignment
        cmap = custom_cmaps[idx] if idx < len(custom_cmaps) else None

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
    fig.text(0.044, 0.74, 'Average Wait Time (s)', va='center', rotation='vertical', fontsize=fs-2)
    fig.text(0.044, 0.32, 'Total Wait Time (×10³ s)', va='center', rotation='vertical', fontsize=fs-2)

    # Right side (Vehicle)
    fig.text(0.505, 0.74, 'Average Wait Time (s)', va='center', rotation='vertical', fontsize=fs-2)
    fig.text(0.505, 0.32, 'Total Wait Time (×10³ s)', va='center', rotation='vertical', fontsize=fs-2)

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

    # MODIFICATION: Remove the last scale (2.75x) from the scales list
    all_scales = all_scales[:-1]

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

    # Set ticks for bottom plots only
    ax_ped_total.set_xticks(ped_ticks)
    ax_veh_total.set_xticks(veh_ticks)

    # Completely hide x-ticks for top plots
    ax_ped_avg.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_veh_avg.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

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

    # Get data ranges for each plot
    ped_avg_data_min = min([min(ped_mean - ped_std) for _, _, ped_mean, _, ped_std in
                          [get_averages(path, total=False) for path in json_paths]])
    ped_avg_data_max = max([max(ped_mean + ped_std) for _, _, ped_mean, _, ped_std in
                          [get_averages(path, total=False) for path in json_paths]])

    veh_avg_data_min = min([min(veh_mean - veh_std) for _, veh_mean, _, veh_std, _ in
                          [get_averages(path, total=False) for path in json_paths]])
    veh_avg_data_max = max([max(veh_mean + veh_std) for _, veh_mean, _, veh_std, _ in
                          [get_averages(path, total=False) for path in json_paths]])

    ped_total_data_min = min([min(ped_total - ped_total_std) for _, _, ped_total, _, ped_total_std in
                            [get_averages(path, total=True) for path in json_paths]])
    ped_total_data_max = max([max(ped_total + ped_total_std) for _, _, ped_total, _, ped_total_std in
                            [get_averages(path, total=True) for path in json_paths]])

    veh_total_data_min = min([min(veh_total - veh_total_std) for _, veh_total, _, veh_total_std, _ in
                            [get_averages(path, total=True) for path in json_paths]])
    veh_total_data_max = max([max(veh_total + veh_total_std) for _, veh_total, _, veh_total_std, _ in
                            [get_averages(path, total=True) for path in json_paths]])

    # Ensure minimum values allow for lower data points (fixing crop issue at 0.5x)
    ped_avg_data_min = max(0.5, ped_avg_data_min)  # Start at 0.5 for top plots
    veh_avg_data_min = max(5.0, veh_avg_data_min)  # Start at 5.0 for top plots
    ped_total_data_min = min(0.0, ped_total_data_min)  # Allow negative values for padding
    veh_total_data_min = min(0.0, veh_total_data_min)  # Allow negative values for padding

    # Set custom y-ticks with more space between plots
    # For top plots, provide more headroom (fixing the orange line crop issue)
    ped_avg_yticks = np.linspace(ped_avg_data_min, ped_avg_data_max * 1.05, 5)
    veh_avg_yticks = np.linspace(veh_avg_data_min, veh_avg_data_max * 1.05, 5)

    # For bottom plots, ensure full visibility
    ped_total_yticks = np.linspace(ped_total_data_min, ped_total_data_max * 1.05, 5)
    veh_total_yticks = np.linspace(veh_total_data_min, veh_total_data_max * 1.05, 5)

    # Set the custom ticks
    ax_ped_avg.set_yticks(ped_avg_yticks)
    ax_veh_avg.set_yticks(veh_avg_yticks)
    ax_ped_total.set_yticks(ped_total_yticks)
    ax_veh_total.set_yticks(veh_total_yticks)

    # Set the y-limits with additional padding
    ax_ped_avg.set_ylim(ped_avg_yticks[0], ped_avg_yticks[-1] )
    ax_veh_avg.set_ylim(veh_avg_yticks[0], veh_avg_yticks[-1] )
    ax_ped_total.set_ylim(ped_total_yticks[0], ped_total_yticks[-1] * 1.1)
    ax_veh_total.set_ylim(veh_total_yticks[0], veh_total_yticks[-1] * 1.1)

    # Set ticks for bottom plots only
    if show_scales:
        scale_labels = [f"{scale:g}x" for scale in scales_for_labels]
        ax_ped_total.set_xticklabels(scale_labels, fontsize=fs-2)
        ax_veh_total.set_xticklabels(scale_labels, fontsize=fs-2)
    else:
        ax_ped_total.set_xticklabels(ped_xtick_labels, fontsize=fs-2)
        ax_veh_total.set_xticklabels(veh_xtick_labels, fontsize=fs-2)

    # Legend with consistent font size
    leg = fig.legend(legend_handles, legend_labels,
                    loc='center',
                    bbox_to_anchor=(0.525, 0.04),
                    ncol=len(legend_handles),
                    frameon=True,
                    framealpha=1.0,
                    edgecolor='gray',
                    fancybox=True,
                    shadow=False,
                    bbox_transform=fig.transFigure,
                    fontsize=fs-4)

    # Adjust subplot spacing
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, wspace=0.20, hspace=0.14)
    plt.savefig("./results/consolidated_results.pdf", bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_consolidated_insights(sampled_actions_file_path, conflict_json_file_path, switching_freq_data_path):
    """
    Creates a consolidated figure with three subplots:
    1. Left: Bar chart of mean conflicts across demand scales with error bars
    2. Middle: Plot of average consecutive ones over training iterations
    3. Right: TL as horizontal line and RL as histogram for switching frequency (TL switching frequency is obtained analytically as 54 for 600 timestep horizon)

    Parameters:
    - sampled_actions_file_path: Path to JSON file containing action data
    - conflict_json_file_path: Path to JSON file containing conflict data
    - switching_freq_data: Dictionary containing switching frequency data (optional)
    """
    # Function to process data from json
    def process_json_data(json_data, key):
        # Extract data by demand scale
        data = {}
        for demand_scale, runs in json_data.items():
            values = [run_data[key] for run_index, run_data in runs.items()]
            data[float(demand_scale)] = {
                "mean": np.mean(values),
                "std": np.std(values)
            }
        return data

    # Load conflict data
    with open(conflict_json_file_path, 'r') as f:
        conflict_json_data = json.load(f)

    # Process conflict data
    processed_conflict_data = process_json_data(conflict_json_data, "total_conflicts")

    # Set base font size
    fs = 23

    # Set consistent number of y-ticks for all subplots
    n_ticks = 5  # Define the number of y-ticks to use across all subplots

    # Set up the figure with a 1x3 grid
    fig = plt.figure(figsize=(24, 6.2))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1.2, 1])

    # Create subplots
    ax_near_accidents = fig.add_subplot(gs[0, 0])
    ax_consecutive_ones = fig.add_subplot(gs[0, 1])
    ax_switching_freq = fig.add_subplot(gs[0, 2])

    # Set style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0

    # Define colors - updated middle plot colors
    BRIGHT_BLUE = '#0078D7'  # New bright blue for middle plot trend line
    VIBRANT_BLUE = '#2E5EAA'  # Keep for scatter points
    SALMON = '#E29587'  # Subtle salmon for TL/Unsignalized
    SEA_GREEN = '#85B79D'  # Subtle sea green for RL

    # ========== LEFT SUBPLOT: Conflict events across demand scales ==========
    # Filter demand scales to only include the specified levels
    selected_demand_scales = [0.5, 1.0, 1.5, 2.0, 2.5]
    filtered_scales = [scale for scale in selected_demand_scales if scale in processed_conflict_data]

    conflict_means = [processed_conflict_data[scale]["mean"] for scale in filtered_scales]
    conflict_stds = [processed_conflict_data[scale]["std"] for scale in filtered_scales]

    # Even more subtle gradient - using shades of orange/coral with less intensity
    colors = [
        '#FDE5D2',  # Very pale orange for 0.5x
        '#FDCBAD',  # Lighter orange for 1.0x
        '#FCB08A',  # Light salmon for 1.5x
        '#FC9774',  # Salmon for 2.0x
        '#FB7D5B'   # Darker salmon for 2.5x
    ]

    # Make sure we have enough colors
    if len(colors) < len(filtered_scales):
        colors = colors * (len(filtered_scales) // len(colors) + 1)
    colors = colors[:len(filtered_scales)]

    # Create bar positions
    x_positions = np.arange(len(filtered_scales))
    width = 0.5

    # Create bar chart with MORE PROMINENT error bars
    bars = ax_near_accidents.bar(x_positions, conflict_means, width, color=colors,
                               edgecolor='#333333', linewidth=1.0,
                               yerr=conflict_stds, capsize=8, error_kw={'elinewidth': 2.5, 'ecolor': '#333333', 'capthick': 2.5})

    # Add data labels to the left of the top of each bar
    # for i, bar in enumerate(bars):
    #     height = bar.get_height() + 9
    #     # Position text to the left of the bar top
    #     ax_near_accidents.text(bar.get_x() + 0.25*width, height,
    #                          f'{int(conflict_means[i])}', ha='right', va='center',
    #                          fontsize=fs-4)

    labelsize = fs-4
    # Set x-ticks at the bar positions with the appropriate labels
    ax_near_accidents.set_xticks(x_positions)
    ax_near_accidents.set_xticklabels([f'{scale}x' for scale in filtered_scales], fontsize=labelsize)

    # Styling
    ax_near_accidents.set_ylabel('# of Conflicts in Unsignalized', fontsize=fs)  # Updated label
    ax_near_accidents.set_xlabel('Demand Scale', fontsize=fs)
    ax_near_accidents.tick_params(axis='both', labelsize=labelsize)

    # Set y-limit with headroom for labels and error bars
    ax_near_accidents.set_ylim(0, max(conflict_means + np.array(conflict_stds)) * 1.1)  # More headroom for labels

    # Make grid match middle plot (light lines behind data)
    ax_near_accidents.grid(True, linestyle='-', alpha=0.15, color='#333333')
    ax_near_accidents.set_axisbelow(True)

    # Remove top and right spines to match middle plot
    ax_near_accidents.spines['top'].set_visible(False)
    ax_near_accidents.spines['right'].set_visible(False)

    # Set consistent y-ticks
    ax_near_accidents.yaxis.set_major_locator(MaxNLocator(n_ticks))

    # ========== MIDDLE SUBPLOT: Average consecutive ones plot ==========
    # Load data
    with open(sampled_actions_file_path, "r") as file:
        data = json.load(file)

    # Compute the average sum of consecutive 1's per iteration
    avg_consecutive_ones_per_iteration = []
    iterations = []

    for iteration, actions_list in data.items():
        iteration = int(iteration)
        consecutive_ones = [count_consecutive_ones_filtered(action_list) for action_list in actions_list]
        sums_of_consecutive_ones = [sum(seq) for seq in consecutive_ones if seq]
        avg_consecutive_ones = np.mean(sums_of_consecutive_ones) if sums_of_consecutive_ones else 0
        iterations.append(iteration)
        avg_consecutive_ones_per_iteration.append(avg_consecutive_ones)

    # Sort by iteration
    iterations, avg_consecutive_ones_per_iteration = zip(*sorted(zip(iterations, avg_consecutive_ones_per_iteration)))
    iterations = np.array(iterations)
    avg_consecutive_ones_per_iteration = np.array(avg_consecutive_ones_per_iteration)

    # Set background color
    ax_consecutive_ones.set_facecolor('white')

    # Calculate y-axis limits with padding
    y_min = 3.3  # Set explicitly to 3.2 to match the lowest data point
    y_max = 4.1  # Set explicitly to 4.1 to provide headroom for highest points

    # Calculate x-axis limits with margins
    x_min = min(iterations) - (max(iterations) - min(iterations)) * 0.05
    x_max = max(iterations) + (max(iterations) - min(iterations)) * 0.05

    # Set axis limits
    ax_consecutive_ones.set_ylim(y_min, y_max)
    ax_consecutive_ones.set_xlim(x_min, x_max)

    # Format y-axis with one decimal place
    ax_consecutive_ones.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}'))

    # Add light grid lines
    ax_consecutive_ones.grid(True, linestyle='-', alpha=0.15, color='#333333')
    ax_consecutive_ones.set_axisbelow(True)

    # Remove top and right spines
    ax_consecutive_ones.spines['top'].set_visible(False)
    ax_consecutive_ones.spines['right'].set_visible(False)

    # Create scatter plot - KEEPING ORIGINAL COLORS
    scatter = ax_consecutive_ones.scatter(iterations, avg_consecutive_ones_per_iteration,
                                        s=110, edgecolors=VIBRANT_BLUE, facecolors='none',
                                        linewidth=2.0, alpha=0.75, zorder=3)

    # Fit a trend line
    z = np.polyfit(iterations, avg_consecutive_ones_per_iteration, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(iterations), max(iterations), 100)
    y_trend = p(x_trend)

    # Calculate 95% confidence interval
    n = len(iterations)
    x_mean = np.mean(iterations)
    y_mean = np.mean(avg_consecutive_ones_per_iteration)

    # Sum of squares
    ss_xx = np.sum((iterations - x_mean)**2)
    ss_xy = np.sum((iterations - x_mean) * (avg_consecutive_ones_per_iteration - y_mean))
    ss_yy = np.sum((avg_consecutive_ones_per_iteration - y_mean)**2)

    # Regression slope and intercept
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    # Standard error of estimate
    y_hat = slope * iterations + intercept
    se = np.sqrt(np.sum((avg_consecutive_ones_per_iteration - y_hat)**2) / (n - 2))

    # Confidence interval
    alpha = 0.05  # 95% confidence interval
    t_val = stats.t.ppf(1 - alpha/2, n - 2)

    # Calculate confidence bands
    x_eval = x_trend
    ci = t_val * se * np.sqrt(1/n + (x_eval - x_mean)**2 / ss_xx)
    y_upper = y_trend + ci
    y_lower = y_trend - ci

    # Plot the trend line with new bright blue color
    trend_line = ax_consecutive_ones.plot(x_trend, y_trend, color=BRIGHT_BLUE, linewidth=4.0, zorder=4, label='Trend Line')

    # Add confidence interval with shading
    confidence_interval = ax_consecutive_ones.fill_between(x_trend, y_lower, y_upper,
                                                         color=BRIGHT_BLUE, alpha=0.2,
                                                         zorder=2, label='95% Confidence Interval')

    # Set labels
    ax_consecutive_ones.set_xlabel('Training Episode', fontsize=fs)
    ax_consecutive_ones.set_ylabel('Synchronized Green Signals', fontsize=fs)

    # Create legend with both trend line and confidence interval
    trend_line_handle = mlines.Line2D([], [], color=BRIGHT_BLUE, linewidth=4.0,
                                    label='Trend Line')
    ci_handle = mpatches.Patch(facecolor=BRIGHT_BLUE, alpha=0.2,
                              label='95% Confidence Interval')

    ax_consecutive_ones.legend(handles=[trend_line_handle, ci_handle],
                            loc='upper right', frameon=True, framealpha=0.9,
                            edgecolor='#CCCCCC', fontsize=fs-4)

    # Tick parameters
    ax_consecutive_ones.tick_params(axis='both', labelsize=labelsize)

    # Set consistent y-ticks with fixed 0.1 interval to ensure we have 3.6 tick
    ax_consecutive_ones.yaxis.set_major_locator(MultipleLocator(0.2))

    # ========== RIGHT SUBPLOT: Switching frequency with TL as horizontal line ==========

    # Load frequency data
    with open(switching_freq_data_path, 'r') as f:
        frequency_json_data = json.load(f)

    # Process frequency data
    processed_frequency_data = process_json_data(frequency_json_data, "total_switches")

    frequency_demands = [0.5, 1.0, 1.5, 2.0, 2.5]
    filtered_demands = [demand for demand in frequency_demands if demand in processed_frequency_data]

    frequency_means = [processed_frequency_data[demand]["mean"] for demand in filtered_demands]
    frequency_stds = [processed_frequency_data[demand]["std"] for demand in filtered_demands]

    # Create placeholder data with TL having same value across demand scales
    tl_value = 54  # Same value for all demand scales

    # Get x positions for grouped bars
    x = np.arange(len(filtered_demands))
    width = 0.5  # Width of bars - keep the same

    # Create subtle gradient for RL bars
    rl_colors = [
        '#CFEAD6',  # Lower level lighter green
        '#A8D5BA',  # Lightest sea green
        '#8CCB9B',  # Light sea green
        '#73C17E',  # Medium sea green
        '#5AB663'   # Deeper sea green
    ]

    # Ensure we have enough colors
    if len(rl_colors) < len(filtered_demands):
        rl_colors = rl_colors * (len(filtered_demands) // len(rl_colors) + 1)
    rl_colors = rl_colors[:len(filtered_demands)]

    # Set up the plot with a discontinuous y-axis
    ax_switching_freq.set_facecolor('white')

    # Function to transform values to the broken y-axis scale
    def transform_y(y):
        # Map values to a discontinuous scale:
        # 0-54 maps to 0-0.2 (bottom 20% of plot)
        # 260-320 maps to 0.3-1.0 (top 70% of plot)
        if y <= 54:
            return y / 54 * 0.2
        else:
            return 0.3 + (y - 260) / (320 - 260) * 0.7

    # Plot the bars with standard deviations
    for i, (mean, std) in enumerate(zip(frequency_means, frequency_stds)):
        # Calculate bar height in the transformed space
        bar_height = transform_y(mean) - transform_y(0)

        # Draw the bar
        bar = ax_switching_freq.bar(x[i], bar_height, width=width,
                                   bottom=transform_y(0),
                                   color=rl_colors[i],
                                   edgecolor='#333333',
                                   linewidth=1.0)

        # Add error bars
        # Calculate the std dev in the transformed space
        yerr = transform_y(mean + std) - transform_y(mean)

        # Draw error bar
        ax_switching_freq.errorbar(x[i], transform_y(mean), yerr=yerr,
                                  fmt='none', ecolor='#333333', capsize=8,
                                  elinewidth=2.5, capthick=2.5)

    # Add the TL horizontal line
    tl_line = ax_switching_freq.axhline(y=transform_y(tl_value), color=SALMON, linewidth=3, linestyle='-', zorder=5)

    # Get the y-axis line width to match the break marks to it
    axis_line_width = ax_switching_freq.spines['left'].get_linewidth()

    # Create break marks for the y-axis
    # Position of the break in the transformed scale
    break_pos = 0.31  # middle of the gap between 0.2 and 0.3

    # Draw break marks on the left y-axis only
    # Increased spacing between diagonal lines
    gap = 0.020  # Increased gap between the diagonal lines
    d = 0.03    # Size of the diagonal lines

    # First create a white rectangle to "erase" part of the axis
    # This ensures the break appears as a true gap in the axis
    rect_height = gap * 1.5  # Height of white rectangle
    rect_width = d * 2.0     # Width of white rectangle

    # Draw white background rectangle to create a clean break
    white_patch = plt.Rectangle((-rect_width/2, break_pos-rect_height/2), rect_width, rect_height,
                              facecolor='white', edgecolor='none', transform=ax_switching_freq.transAxes,
                              clip_on=False, zorder=10)
    ax_switching_freq.add_patch(white_patch)

    # Then draw the diagonal lines centered on the axis
    # Make sure line width matches the axis line width
    kwargs = dict(transform=ax_switching_freq.transAxes, color='black',
                 clip_on=False, linewidth=axis_line_width, zorder=11)

    # Upper diagonal line
    ax_switching_freq.plot([-d/2, d/2], [break_pos+gap/2, break_pos+gap/2 + d], **kwargs)

    # Lower diagonal line
    ax_switching_freq.plot([-d/2, d/2], [break_pos-gap/2, break_pos-gap/2 + d], **kwargs)

    # Set the y-ticks at the actual data values
    yticks = [0, tl_value, 275, 300]
    yticklabels = [str(int(y)) for y in yticks]

    ax_switching_freq.set_yticks([transform_y(y) for y in yticks])
    ax_switching_freq.set_yticklabels(yticklabels, fontsize=labelsize)

    # Create legend handles
    tl_handle = mlines.Line2D([], [], color=SALMON, linewidth=3, linestyle='-', label='Signalized')
    rl_handle = mpatches.Patch(facecolor=rl_colors[1], edgecolor='#333333', linewidth=1.0, label='RL (Ours)')

    # Styling
    ax_switching_freq.set_ylabel('Switching Frequency', fontsize=fs)
    ax_switching_freq.set_xlabel('Demand Scale', fontsize=fs)
    ax_switching_freq.set_xticks(x)

    # Format x-ticks to show demand scale
    demand_labels = [f"{d}x" for d in filtered_demands]
    ax_switching_freq.set_xticklabels(demand_labels, fontsize=labelsize)

    ax_switching_freq.tick_params(axis='both', labelsize=labelsize)

    # Make grid match middle plot (light lines behind data)
    ax_switching_freq.grid(True, linestyle='-', alpha=0.15, color='#333333')
    ax_switching_freq.set_axisbelow(True)

    # Remove top and right spines to match middle plot
    ax_switching_freq.spines['top'].set_visible(False)
    ax_switching_freq.spines['right'].set_visible(False)

    # Set uniform margins in right subplot
    # Calculate the margin to add on each side (half the width of a bar)
    margin = 0.7
    # Set the x-limits to create uniform margins
    ax_switching_freq.set_xlim(-margin, len(filtered_demands) - 1 + margin)

    # Set y-limits for the plot
    ax_switching_freq.set_ylim(0, 1.05)  # Provide headroom for the legend

    # Add legend in the top right corner
    ax_switching_freq.legend(handles=[tl_handle, rl_handle], fontsize=fs-4, loc='upper right',
                           bbox_to_anchor=(1.0, 1.01))

    # ========== Add (a), (b), (c) labels centered below each subplot ==========
    # Get the exact position of each subplot after tight_layout
    bbox1 = ax_near_accidents.get_position()
    bbox2 = ax_consecutive_ones.get_position()
    bbox3 = ax_switching_freq.get_position()

    # Calculate the center x-coordinate for each subplot
    x1 = bbox1.x0 + bbox1.width/2
    x2 = bbox2.x0 + bbox2.width/2
    x3 = bbox3.x0 + bbox3.width/2

    # Define y position - a tiny bit lower than before
    label_y = -0.08  # Moved down slightly from -0.01 to -0.03

    # Add the labels at the exact centers
    fig.text(x1, label_y, '(a)', ha='center', va='center', fontsize=fs, fontweight='bold')
    fig.text(x2, label_y, '(b)', ha='center', va='center', fontsize=fs, fontweight='bold')
    fig.text(x3, label_y, '(c)', ha='center', va='center', fontsize=fs, fontweight='bold')

    # ========== Figure-level adjustments ==========
    plt.subplots_adjust(wspace=0.23, bottom=0.1)  # Adjusted bottom margin to make room for labels

    # Save figure
    plt.savefig("./results/consolidated_insights.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)

    plt.show()
    return fig

# Usage
####### CONSOLIDATED 3 SUBPLOTS ######
# sampled_actions_file_path = "./results/sampled_actions.json"
# unsignalized_conflicts_file_path = "./results/eval_unsignalized_conflicts.json"
# switching_freq_data_path = "./results/switching_freq_data.json"
# plot_consolidated_insights(sampled_actions_file_path, unsignalized_conflicts_file_path, switching_freq_data_path) # Other values are manually input inside the function.


# ###### MAIN RESULTS PLOT ######
# unsignalized_results_path = "./results/eval_unsignalized.json"
# tl_results_path = "./results/eval_tl.json"
# ppo_results_path = "./results/eval_ppo.json"

# plot_main_results(unsignalized_results_path, 
#                          tl_results_path, 
#                          ppo_results_path,
#                          in_range_demand_scales=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25])

######  Just plot sampled 1's ###### 
# sampled_actions_file_path = "./saved_models/Feb24_13-54-26/sampled_actions.json"
# plot_avg_consecutive_ones(sampled_actions_file_path)
