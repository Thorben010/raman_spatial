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

# Configuration (will be handled by argparse)
# file_path = '/Users/thorbenp/PycharmProjects/spatial_raman/data/steffen/mapping Ta50 20250213--Scan LA Step--007--Spec.Data 1.txt'

# Data reading and preprocessing functions
def read_data_to_df(file_path):
    """Function to read data from file and convert it to a DataFrame"""
    print(f"  Reading data from: {file_path}")
    data_rows = []
    with open(file_path, 'r') as file:
        for line in file:
            row = line.strip().split(',')
            data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    print(f"  Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def normalize_per_column(df):
    """Normalize each column"""
    print("  Normalizing data per column...")
    normalized_df = df.apply(lambda col: (col - col.min()) / (col.max() - col.min()), axis=0)
    print("  Normalization complete")
    return normalized_df

def savgol_(df):
    """Apply Savitzky-Golay filter to DataFrame"""
    print("  Applying Savitzky-Golay filter...")
    filtered_df = df.copy()
    for col in filtered_df.columns:
        filtered_df[col] = savgol_filter(filtered_df[col], 6, 2)
    print("  Savitzky-Golay filtering complete")
    return filtered_df

def filter_large_deviations(original_df, smoothed_df, threshold=.5):
    """Filter values where deviation is more than the threshold"""
    print(f"  Filtering large deviations with threshold {threshold}...")
    deviation = original_df - smoothed_df
    allowed_deviation = smoothed_df * threshold
    filtered_df = original_df.where(deviation.abs() <= allowed_deviation)
    print("  Deviation filtering complete")
    return filtered_df

def smooth_spectra(df):
    """Function to smooth the spectra in the DataFrame"""
    print("  Smoothing spectra with Savitzky-Golay filter...")
    smoothed_df = df.copy()
    for col in smoothed_df.columns:
        smoothed_df[col] = savgol_filter(smoothed_df[col], window_length=5, polyorder=2)
    print("  Spectra smoothing complete")
    return smoothed_df

def detect_peaks(df, peak_width=15):
    """Detect peaks in the mean spectrum"""
    print(f"  Detecting peaks with width parameter {peak_width}...")
    mean_spectrum = df.mean(axis=1)
    peaks, _ = find_peaks(mean_spectrum, distance=peak_width)
    
    peak_ranges = []
    for peak in peaks:
        start = df.index[peak] - peak_width
        end = df.index[peak] + peak_width
        peak_ranges.append((start, end))
    
    print(f"  Found {len(peak_ranges)} peaks")
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

    x_coords = spatial_coords[:, 0]
    y_coords = spatial_coords[:, 1]

    heatmap, xedges, yedges, _ = binned_statistic_2d(x_coords, y_coords, intensity_values, statistic='mean', bins=50)
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

def plot_map_scientific(pca_df, positions, value, label_cbar):
    """Create scientific style individual plots"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    mpl.style.use('seaborn-whitegrid')
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
    plt.savefig(f'peak_wavenumber_map_{value}.png', dpi=300)
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
    # Main data processing
    print("="*60)
    print("SPATIAL RAMAN ANALYSIS PIPELINE")
    print("="*60)

    print("\n1. LOADING AND PREPROCESSING DATA...")
    df = read_data_to_df(args.file_path)

    print("  Converting data types...")
    df = df.iloc[1:]
    df = df.apply(pd.to_numeric, errors='coerce')
    print("  Data type conversion complete")

    print("\n2. CREATING SPATIAL GRID...")
    # Create spatial grid
    grid_size = int(np.sqrt(len(df.columns)-1))
    print(f'  Grid size: {grid_size}x{grid_size}')
    vector_grid_2d = np.linspace(0, grid_size, num=grid_size)
    x, y = np.meshgrid(vector_grid_2d, vector_grid_2d)
    positions = np.vstack((x.flatten(), y.flatten())).T
    print(f"  Created {len(positions)} spatial positions")

    print("\n3. FILTERING AND INDEXING DATA...")
    # Set index and filter data
    df.index = df.iloc[:, 0]
    df = df.loc[:, 1:] 
    original_size = len(df)
    df = df[(df.index >= 100) & (df.index <= 800)]
    print(f"  Filtered wavenumber range: {len(df)} points (from {original_size})")

    print("\n4. NORMALIZING DATA...")
    normalized_df = normalize_per_column(df.loc[:, :])

    print("\n5. REMOVING OUTLIERS...")
    print("  Calculating z-scores for outlier detection...")
    z_scores = np.abs((normalized_df - normalized_df.mean()) / normalized_df.std())
    threshold = 3
    outlier_indices = np.where(z_scores > threshold)

    df_clean = normalized_df.copy()
    df_clean.iloc[outlier_indices] = np.nan

    nan_count_before = df_clean.isna().sum().sum()
    print(f"  Found {nan_count_before} outlier values")
    print("  Interpolating outlier values...")
    df_clean = df_clean.interpolate()
    nan_count_after = df_clean.isna().sum().sum()

    print("Number of NaN values before interpolation:", nan_count_before)
    print("Number of NaN values after interpolation:", nan_count_after)

    print("\n6. PLOTTING CLEANED SPECTRA...")
    # Plot all spectra
    df_clean.plot(legend=False)
    plt.savefig('all_spectra.png')
    plt.show()
    print("  Saved: all_spectra.png")

    print("\n7. APPLYING SMOOTHING...")
    # Apply smoothing
    smoothed_df = smooth_spectra(df_clean)
    smoothed_df.plot(legend=False)

    print("\n8. GENERATING MEAN SPECTRUM...")
    # Plot mean spectrum
    df.mean(axis=1).plot()
    plt.xlabel('Wavenumber')
    plt.ylabel('Intensity')
    plt.savefig('mean_spectrum.png', dpi=600)
    plt.show()
    print("  Saved: mean_spectrum.png")

    print("\n9. DEFINING PEAK RANGES...")
    peak_ranges = args.peak_ranges
    print(f"  Using {len(peak_ranges)} predefined peak ranges from arguments")

    # Auto-detect peaks (option)
    print("  Auto-detecting peaks...")
    detected_peak_ranges = detect_peaks(df, peak_width=15)
    print("Detected peak ranges:", detected_peak_ranges)

    print("\n10. PLOTTING PEAK RANGES ON MEAN SPECTRUM...")
    # Plot mean spectrum with peak ranges highlighted
    normalized_df.mean(axis=1).plot()
    colors = ['gray', 'darkblue', 'green', 'purple']
    for i, entry in enumerate(peak_ranges):
        plt.axvspan(entry[0], entry[1], color=colors[i % len(colors)], alpha=0.3)
    plt.savefig("mean.png", dpi=600)
    print("  Saved: mean.png")

    print("\n11. GENERATING WAVENUMBER MAPS...")
    # Generate wavenumber maps
    num_plots = len(peak_ranges)
    grid_size = int(np.ceil(np.sqrt(num_plots)))
    print(f"  Creating {num_plots} wavenumber maps in {grid_size}x{grid_size} grid")

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(19, 15))
    axes = axes.flatten()
    fig.suptitle('', fontsize=16)

    df_int = pd.DataFrame()
    means = []
    for index, value in enumerate(peak_ranges):
        print(f"  Processing peak range {index+1}/{len(peak_ranges)}: {value}")
        df_filtered = filter_frequencies(df_clean, value)
        print(f"    Filtered data shape: {df_filtered.shape}")
        
        print("    Performing PCA analysis...")
        pca_df = normalize_and_pca(df_filtered)
        
        print("    Calculating peak shifts...")
        max_wavenumber, max_intensity, df_integration = plot_peak_shifts(df_filtered)
        
        value_label = f'{np.mean(max_wavenumber.mean()).round(1)}'
        means.append(value_label)
        
        print(f"    Creating heatmap for {value_label}...")
        scatter, cbar = plot_map_heatmap(max_wavenumber, positions, value_label, axes[index], 'Max Wavenumber')

    # Remove empty subplots
    print("  Removing empty subplots...")
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    fig.savefig('Siqi_8_11_24_wavenumber.png', dpi=600)
    print("  Saved: Siqi_8_11_24_wavenumber.png")

    print("\n12. GENERATING INTENSITY MAPS...")
    # Generate intensity maps
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    fig.suptitle('Intensity', fontsize=16)

    for index, value in enumerate(peak_ranges):
        print(f"  Processing intensity map {index+1}/{len(peak_ranges)}: {value}")
        df_filtered = filter_frequencies(df_clean, value)
        
        print("    Performing PCA analysis...")
        pca_df = normalize_and_pca(df_filtered)
        
        print("    Calculating peak shifts...")
        max_wavenumber, max_intensity = plot_peak_shifts(df_filtered)
        
        print("    Creating scatter plot...")
        scatter, cbar = plot_map_scatter(max_intensity, positions, value, axes[index], 'Intensity')

    print("  Removing empty subplots...")
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('peak_wavenumber_maps.png')
    print("  Saved: peak_wavenumber_maps.png")

    print("\n13. GENERATING INDIVIDUAL SCIENTIFIC STYLE PLOTS...")
    # Generate individual scientific style plots
    for index, value in enumerate(peak_ranges):
        print(f"  Creating scientific plot {index+1}/{len(peak_ranges)}: {value}")
        df_filtered = filter_frequencies(df_clean, value)
        
        print("    Performing PCA analysis...")
        pca_df = normalize_and_pca(df_filtered)
        
        print("    Calculating peak shifts...")
        max_wavenumber, max_intensity = plot_peak_shifts(df_filtered)
        
        print(f"    Saving individual plot: peak_wavenumber_map_{value}.png")
        plot_map_scientific(max_intensity, positions, value, 'Intensity')

    print("\n14. PERFORMING SPATIAL CORRELATION ANALYSIS...")
    # Spatial correlation analysis using Moran's I
    print("  Initializing Moran's I calculation...")
    i_values = []
    p_values = []

    total_iterations = len(df)
    print(f"  Processing {total_iterations} wavenumber points...")

    for i in range(len(df)):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    Progress: {i+1}/{total_iterations} ({((i+1)/total_iterations)*100:.1f}%)")
        
        change_values = normalized_df.iloc[i, 1:].values
        spatial_coords = positions

        knn = 8
        spatial_w = lp.weights.KNN.from_array(spatial_coords, k=knn)
        change_values_std = (change_values - np.mean(change_values)) / np.std(change_values)
        moran_change = esda.Moran(change_values_std, spatial_w)

        p_values.append(moran_change.p_sim)
        if moran_change.p_sim <= 0.01:
            i_values.append(moran_change.I)
        else:
            i_values.append(0)

    print("  Spatial correlation analysis complete")
    df['p_values'] = p_values
    df['i_values'] = i_values

    print("\n15. FINAL DATA NORMALIZATION...")
    # Normalize data for final analysis
    df_old = df
    df = (df - df.min()) / (df.max() - df.min())
    df['Wavelength'] = df_old.iloc[:, 0]
    df['i_values'] = df_old['i_values']
    df['p_values'] = df_old['p_values']
    print("  Final normalization complete")

    print("\n16. PLOTTING STATISTICAL SIGNIFICANCE ANALYSIS...")
    # Plot statistical significance analysis
    x_values = df['Wavelength']
    i_values = df['i_values']
    means = df.iloc[:, 2:].mean(axis=1)

    plt.plot(x_values, means)
    plt.scatter(x_values, means, c=i_values, cmap='viridis', label='i_value', s=50)

    lower_bound = means - i_values / 2
    upper_bound = means + i_values / 2
    plt.fill_between(x_values, lower_bound, upper_bound, alpha=0.3)

    cbar = plt.colorbar(label='i_value')
    plt.xlabel('Wavelength')
    plt.ylabel('Mean Intensity')
    plt.title('Intensity Changes By Statistical Significance')
    plt.grid(True)
    plt.savefig('output.png', dpi=300)
    plt.show()
    print("  Saved: statistical significance plot")

    print("\n17. FINDING MAXIMUM STATISTICAL SIGNIFICANCE...")
    # Find and plot maximum statistical significance
    max_val = np.where(df['i_values'] == df['i_values'].max())[0][0]
    print(f"  Maximum Moran's I value found at index: {max_val}")
    df['i_values'].plot()

    print("\n18. PLOTTING SPATIAL DISTRIBUTION AT MAXIMUM SIGNIFICANCE...")
    # Plot spatial distribution at maximum significance
    spatial_coords = positions
    intensity_values = df.iloc[max_val, 1:-3]

    x_coords = spatial_coords[:, 0]
    y_coords = spatial_coords[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(x_coords, y_coords, c=intensity_values, cmap='viridis', s=100)
    plt.colorbar(label='Intensity')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Variations in Intensity by Color')
    plt.grid(True)
    plt.savefig('output.png', dpi=300)
    plt.show()
    print("  Saved: spatial distribution plot")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - all_spectra.png")
    print("  - mean_spectrum.png") 
    print("  - mean.png")
    print("  - Siqi_8_11_24_wavenumber.png")
    print("  - peak_wavenumber_maps.png")
    print("  - Individual peak maps: peak_wavenumber_map_*.png")
    print("  - output.png (statistical significance & spatial distribution)")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatial Raman Analysis Pipeline")
    parser.add_argument('--file_path', type=str, 
                        default='/Users/thorbenp/PycharmProjects/spatial_raman/data/steffen/mapping Ta50 20250213--Scan LA Step--007--Spec.Data 1.txt',
                        help='Path to the spectral data file.')
    parser.add_argument('--peak_ranges', type=str,
                        default="[(500.236, 530.236), (620.236, 670), (720.236, 750.236)]",
                        help="List of tuples representing peak ranges, e.g., '[(100, 150), (200, 250)]'")
    
    args = parser.parse_args()
    
    # Safely evaluate the peak_ranges string
    try:
        args.peak_ranges = ast.literal_eval(args.peak_ranges)
    except (ValueError, SyntaxError):
        print("Error: Invalid format for peak_ranges. Please use a list of tuples format, e.g., '[(100, 150), (200, 250)]'")
        exit()

    main(args)



