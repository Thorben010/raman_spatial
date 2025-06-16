# Spatial Raman Analysis - Cleaned Version
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import binned_statistic_2d
from sklearn.decomposition import PCA
import libpysal as lp
import esda
import argparse
import ast
import os
import datetime
import logging
import seaborn as sns

# Configuration (will be handled by argparse)
# file_path = '/Users/thorbenp/PycharmProjects/spatial_raman/data/steffen/mapping Ta50 20250213--Scan LA Step--007--Spec.Data 1.txt'

# Data reading and preprocessing functions
def read_data_to_df(file_path):
    """Function to read data from file and convert it to a DataFrame"""
    logging.info(f"  Reading data from: {file_path}")
    data_rows = []
    with open(file_path, 'r') as file:
        for line in file:
            row = line.strip().split(',')
            data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    logging.info(f"  Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def normalize_per_column(df):
    """Normalize each column"""
    logging.info("  Normalizing data per column...")
    normalized_df = df.apply(lambda col: (col - col.min()) / (col.max() - col.min()), axis=0)
    logging.info("  Normalization complete")
    return normalized_df

def savgol_(df):
    """Apply Savitzky-Golay filter to DataFrame"""
    logging.info("  Applying Savitzky-Golay filter...")
    filtered_df = df.copy()
    for col in filtered_df.columns:
        filtered_df[col] = savgol_filter(filtered_df[col], 6, 2)
    logging.info("  Savitzky-Golay filtering complete")
    return filtered_df

def filter_large_deviations(original_df, smoothed_df, threshold=.5):
    """Filter values where deviation is more than the threshold"""
    logging.info(f"  Filtering large deviations with threshold {threshold}...")
    deviation = original_df - smoothed_df
    allowed_deviation = smoothed_df * threshold
    filtered_df = original_df.where(deviation.abs() <= allowed_deviation)
    logging.info("  Deviation filtering complete")
    return filtered_df

def smooth_spectra(df):
    """Function to smooth the spectra in the DataFrame"""
    logging.info("  Smoothing spectra with Savitzky-Golay filter...")
    smoothed_df = df.copy()
    for col in smoothed_df.columns:
        smoothed_df[col] = savgol_filter(smoothed_df[col], window_length=5, polyorder=2)
    logging.info("  Spectra smoothing complete")
    return smoothed_df

def detect_peaks(df, peak_width=15):
    """Detect peaks in the mean spectrum"""
    logging.info(f"  Detecting peaks with width parameter {peak_width}...")
    mean_spectrum = df.mean(axis=1)
    peaks, _ = find_peaks(mean_spectrum, distance=peak_width)
    
    peak_ranges = []
    for peak in peaks:
        start = df.index[peak] - peak_width
        end = df.index[peak] + peak_width
        peak_ranges.append((start, end))
    
    logging.info(f"  Found {len(peak_ranges)} peaks")
    return peak_ranges

def filter_frequencies(df, frequency_range):
    """Filters the DataFrame to include only the frequencies within the given range"""
    start_freq, end_freq = frequency_range
    filtered = df[(df.index >= start_freq) & (df.index <= end_freq)]
    return filtered

def normalize_and_pca(df):
    """Normalizes the area under the curve of each spectrum to 1 and performs PCA to reduce data to 1D space"""
    df_normalized = df
    
    # Perform PCA
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(df_normalized.T)
    
    df_normalized.plot(legend=None)
    
    return pd.DataFrame(data=principal_components, columns=['PC1'])

def plot_peak_shifts(df):
    """Plot peak shifts and return results"""
    max_peak_wavenumbers = []
    peak_intensity = []
    integration = []

    for column in df.columns:
        spectrum = df[column].values
        peak_indices = np.argmax(spectrum)
        peak_value = df.index[peak_indices]

        peak_intensity.append(np.max(spectrum))
        max_peak_wavenumbers.append(peak_value)
        integration.append(df[column].values.sum())

    result_df = pd.DataFrame({'Max Peak Wavenumber': max_peak_wavenumbers})
    result_df_int = pd.DataFrame({'Max Peak Intensity': peak_intensity})
    integration_df_int = pd.DataFrame({'Intensity Integration': integration})
    return result_df, result_df_int, integration_df_int

def plot_map_heatmap(pca_df, positions, value, ax, label_cbar):
    """Plot heatmap visualization of spatial data"""
    spatial_coords = positions
    intensity_values = pca_df.values.flatten()

    # Ensure we have the same number of points
    if len(intensity_values) != len(spatial_coords):
        logging.warning(f"Dimension mismatch: {len(intensity_values)} intensity values vs {len(spatial_coords)} positions")
        # Truncate to the smaller size
        min_size = min(len(intensity_values), len(spatial_coords))
        intensity_values = intensity_values[:min_size]
        spatial_coords = spatial_coords[:min_size]

    x_coords = spatial_coords[:, 0]
    y_coords = spatial_coords[:, 1]

    # Use a smaller number of bins to avoid empty bins
    n_bins = min(20, int(np.sqrt(len(x_coords))))
    heatmap, xedges, yedges, _ = binned_statistic_2d(x_coords, y_coords, intensity_values, 
                                                    statistic='mean', bins=n_bins)
    X, Y = np.meshgrid(xedges, yedges)
    heatmap_plot = ax.pcolormesh(X, Y, heatmap.T, cmap='viridis', shading='auto')

    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    cbar = plt.colorbar(heatmap_plot, ax=ax, label=label_cbar)

    ax.set_xlabel('X Space')
    ax.set_ylabel('Y Space')
    ax.set_title(value)

    return heatmap_plot, cbar

def plot_map_scatter(pca_df, positions, value, ax, label_cbar):
    """Plot scatter visualization of spatial data"""
    spatial_coords = positions
    intensity_values = pca_df.values.flatten()

    x_coords = spatial_coords[:, 0]
    y_coords = spatial_coords[:, 1]

    scatter = ax.scatter(x_coords, y_coords, c=intensity_values, cmap='viridis', s=100)
    cbar = plt.colorbar(scatter, ax=ax, label=label_cbar)

    ax.set_xlabel('X Space')
    ax.set_ylabel('Y Space')
    ax.set_title(value)
    ax.grid(True)

    return scatter, cbar

def plot_map_scientific(pca_df, positions, value, label_cbar, run_log_dir):
    """Create scientific style individual plots"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    mpl.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14

    spatial_coords = positions
    intensity_values = pca_df.values.flatten()

    x_coords = spatial_coords[:, 0]
    y_coords = spatial_coords[:, 1]

    scatter = ax.scatter(x_coords, y_coords, c=intensity_values, cmap='CMRmap', s=80, edgecolor='k')

    cbar = plt.colorbar(scatter, ax=ax, label=label_cbar)
    cbar.ax.set_ylabel(label_cbar, rotation=270, labelpad=15)

    ax.set_xlabel('X Space')
    ax.set_ylabel('Y Space')
    ax.set_title(value)
    ax.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f'{run_log_dir}/peak_wavenumber_map_{value}.png', dpi=300)
    plt.close(fig)

def plot_means_and_std(df):
    """Plot mean spectrum with standard deviation"""
    mean_spectrum = df.mean(axis=1)
    std_deviation = df.std(axis=1)
    wavenumbers = df.index

    plt.plot(wavenumbers, mean_spectrum, label='Mean Spectrum', color='blue')
    plt.fill_between(wavenumbers, mean_spectrum - std_deviation, mean_spectrum + std_deviation, color='gray', alpha=0.3)

    plt.xlabel('Wavenumber')
    plt.ylabel('Intensity')
    plt.title('Mean Spectrum with Standard Deviation Band')
    plt.show()

def main(args):
    """Main function to run the spatial raman analysis pipeline"""
    # Create log directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = 'log'
    run_log_dir = os.path.join(log_dir, timestamp)
    os.makedirs(run_log_dir, exist_ok=True)

    # Setup logging
    log_file_path = os.path.join(run_log_dir, 'run.log')
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file_path),
                            logging.StreamHandler()
                        ])

    # Main data processing
    logging.info("="*60)
    logging.info("SPATIAL RAMAN ANALYSIS PIPELINE")
    logging.info("="*60)

    logging.info("\n1. LOADING AND PREPROCESSING DATA...")
    df = read_data_to_df(args.file_path)

    logging.info("  Converting data types...")
    df = df.iloc[1:]
    df = df.apply(pd.to_numeric, errors='coerce')
    logging.info("  Data type conversion complete")

    logging.info("\n2. CREATING SPATIAL GRID...")
    # Create spatial grid
    num_columns = len(df.columns)-1  # Subtract 1 for wavenumber column
    grid_size = int(np.sqrt(num_columns))
    logging.info(f'  Number of data points: {num_columns}')
    logging.info(f'  Grid size: {grid_size}x{grid_size}')
    
    # Create a proper grid
    x = np.linspace(0, grid_size, num=grid_size)
    y = np.linspace(0, grid_size, num=grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Flatten the grid and take only the number of points we need
    positions = np.vstack((X.flatten()[:num_columns], Y.flatten()[:num_columns])).T
    logging.info(f"  Created {len(positions)} spatial positions")

    logging.info("\n3. FILTERING AND INDEXING DATA...")
    # Set index and filter data
    df.index = df.iloc[:, 0]
    df = df.loc[:, 1:]
    original_size = len(df)
    df = df[(df.index >= 100) & (df.index <= 800)]
    logging.info(f"  Filtered wavenumber range: {len(df)} points (from {original_size})")

    logging.info("\n4. NORMALIZING DATA...")
    normalized_df = normalize_per_column(df.loc[:, :])

    logging.info("\n5. REMOVING OUTLIERS...")
    logging.info("  Calculating z-scores for outlier detection...")
    z_scores = np.abs((normalized_df - normalized_df.mean()) / normalized_df.std())
    threshold = args.outlier_std_dev
    logging.info(f"  Using outlier removal threshold: {threshold} standard deviations")
    outlier_indices = np.where(z_scores > threshold)

    df_clean = normalized_df.copy()
    df_clean.iloc[outlier_indices] = np.nan

    nan_count_before = df_clean.isna().sum().sum()
    logging.info(f"  Found {nan_count_before} outlier values")
    logging.info("  Interpolating outlier values...")
    df_clean = df_clean.interpolate()
    nan_count_after = df_clean.isna().sum().sum()

    logging.info(f"Number of NaN values before interpolation: {nan_count_before}")
    logging.info(f"Number of NaN values after interpolation: {nan_count_after}")

    logging.info("\n6. PLOTTING CLEANED SPECTRA...")
    # Plot all spectra
    df_clean.plot(legend=False)
    plt.savefig(f'{run_log_dir}/all_spectra.png')
    logging.info(f"  Saved: {run_log_dir}/all_spectra.png")

    logging.info("\n7. APPLYING SMOOTHING...")
    # Apply smoothing
    smoothed_df = smooth_spectra(df_clean)
    plt.figure()
    smoothed_df.plot(legend=False)
    plt.title("Smoothed Spectra")
    plt.savefig(f'{run_log_dir}/smoothed_spectra.png')
    logging.info(f"  Saved: {run_log_dir}/smoothed_spectra.png")

    logging.info("\n8. DEFINING PEAK RANGES...")
    peak_ranges = args.peak_ranges
    logging.info(f"  Using {len(peak_ranges)} predefined peak ranges from arguments")

    # Auto-detect peaks (option)
    logging.info("  Auto-detecting peaks...")
    detected_peak_ranges = detect_peaks(df, peak_width=15)
    logging.info(f"Detected peak ranges: {detected_peak_ranges}")

    logging.info("\n9. PLOTTING PEAK RANGES ON NORMALIZED MEAN SPECTRUM...")
    # Plot mean spectrum with peak ranges highlighted
    plt.figure()
    normalized_df.mean(axis=1).plot()
    colors = ['gray', 'darkblue', 'green', 'purple']
    for i, entry in enumerate(peak_ranges):
        plt.axvspan(entry[0], entry[1], color=colors[i % len(colors)], alpha=0.3)
    plt.title("Mean of Normalized Spectra with Peak Ranges")
    plt.savefig(f"{run_log_dir}/mean_normalized.png", dpi=600)
    logging.info(f"  Saved: {run_log_dir}/mean_normalized.png")

    logging.info("\n10. GENERATING SPATIAL GRID MAPS...")
    # Generate wavenumber maps
    num_plots = len(peak_ranges)
    grid_size = int(np.ceil(np.sqrt(num_plots)))
    logging.info(f"  Creating {num_plots} maps in {grid_size}x{grid_size} grid for each metric")

    # Figure for Max Wavenumber
    fig_wavenumber, axes_wavenumber = plt.subplots(grid_size, grid_size, figsize=(19, 15))
    axes_wavenumber = axes_wavenumber.flatten()
    fig_wavenumber.suptitle('Max Wavenumber', fontsize=16)

    # Figure for Max Intensity
    fig_intensity, axes_intensity = plt.subplots(grid_size, grid_size, figsize=(19, 15))
    axes_intensity = axes_intensity.flatten()
    fig_intensity.suptitle('Max Intensity', fontsize=16)

    # Figure for Integrated Intensity
    fig_integration, axes_integration = plt.subplots(grid_size, grid_size, figsize=(19, 15))
    axes_integration = axes_integration.flatten()
    fig_integration.suptitle('Integrated Intensity', fontsize=16)

    for index, value in enumerate(peak_ranges):
        logging.info(f"  Processing peak range {index+1}/{len(peak_ranges)}: {value}")
        df_filtered = filter_frequencies(df_clean, value)
        logging.info(f"    Filtered data shape: {df_filtered.shape}")
        
        logging.info("    Performing PCA analysis...")
        pca_df = normalize_and_pca(df_filtered)
        
        logging.info("    Calculating peak shifts...")
        max_wavenumber, max_intensity, df_integration = plot_peak_shifts(df_filtered)
        
        # Plot for Max Wavenumber
        value_label_wavenumber = f'{np.mean(max_wavenumber.mean()).round(1)} cm-1'
        logging.info(f"    Creating heatmap for Max Wavenumber ({value_label_wavenumber})...")
        plot_map_heatmap(max_wavenumber, positions, value_label_wavenumber, axes_wavenumber[index], 'Max Wavenumber (cm-1)')

        # Plot for Max Intensity
        value_label_intensity = f'{value[0]}-{value[1]} cm-1'
        logging.info(f"    Creating heatmap for Max Intensity...")
        plot_map_heatmap(max_intensity, positions, value_label_intensity, axes_intensity[index], 'Max Intensity (a.u.)')
        
        # Plot for Integrated Intensity
        logging.info(f"    Creating heatmap for Integrated Intensity...")
        plot_map_heatmap(df_integration, positions, value_label_intensity, axes_integration[index], 'Integrated Intensity (a.u.)')

    # Clean up and save plots
    figs = {
        'wavenumber': (fig_wavenumber, axes_wavenumber),
        'intensity': (fig_intensity, axes_intensity),
        'integration': (fig_integration, axes_integration)
    }

    for name, (fig, axes) in figs.items():
        logging.info(f"  Cleaning up and saving {name} grid...")
        for i in range(num_plots, len(axes)):
            fig.delaxes(axes[i])
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
        fig.savefig(f'{run_log_dir}/spatial_grid_{name}.png', dpi=600)
        logging.info(f"  Saved: {run_log_dir}/spatial_grid_{name}.png")
        plt.close(fig)

    logging.info("\n" + "="*60)
    logging.info("ANALYSIS COMPLETE!")
    logging.info("="*60)
    logging.info("\nGenerated files in: " + run_log_dir)
    logging.info("  - run.log")
    logging.info("  - all_spectra.png")
    logging.info("  - smoothed_spectra.png")
    logging.info("  - mean_normalized.png")
    logging.info("  - spatial_grid_wavenumber.png")
    logging.info("  - spatial_grid_intensity.png")
    logging.info("  - spatial_grid_integration.png")
    logging.info("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatial Raman Analysis Pipeline")
    parser.add_argument('--file_path', type=str, 
                        default='data/new_data.txt',
                        help='Path to the spectral data file.')
    parser.add_argument('--peak_ranges', type=str,
                        default="[(500.236, 530.236), (620.236, 670), (720.236, 750.236)]",
                        help="List of tuples representing peak ranges, e.g., '[(100, 150), (200, 250)]'")
    parser.add_argument('--outlier_std_dev', type=float, default=3.0,
                        help='Standard deviation threshold for outlier removal.')
    
    args = parser.parse_args()
    
    # Safely evaluate the peak_ranges string
    try:
        args.peak_ranges = ast.literal_eval(args.peak_ranges)
    except (ValueError, SyntaxError):
        logging.error("Error: Invalid format for peak_ranges. Please use a list of tuples format, e.g., '[(100, 150), (200, 250)]'")
        exit()

    main(args)



