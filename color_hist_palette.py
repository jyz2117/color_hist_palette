import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
import glob
import os
from scipy.ndimage import gaussian_filter1d

from config import N_COLORS, GROUP_FACTOR, MIN_DISTANCE, LUMINANCE_BIAS, NUM_BINS, LUMINANCE_CURVE, SMOOTHING_SIGMA, BACKGROUND


def color_distance(hsv1, hsv2):
    
    dh = min(abs(hsv1[0] - hsv2[0]), 1.0 - abs(hsv1[0] - hsv2[0]))
    ds = hsv1[1] - hsv2[1]
    dv = hsv1[2] - hsv2[2]
    return np.sqrt(dh**2 + ds**2 + dv**2)

def extract_varied_colors(img_array, n_colors=N_COLORS, group_factor=GROUP_FACTOR, min_distance=MIN_DISTANCE, luminance_bias=LUMINANCE_BIAS):
    
    hsv_array = rgb_to_hsv(img_array)
    pixels = hsv_array.reshape(-1, 3)
    total_pixels = len(pixels)
    
    quantized_pixels = np.round(pixels * group_factor) / group_factor
    quantized_pixels = np.clip(quantized_pixels, 0.0, 1.0)
    quantized_pixels[:, 0][quantized_pixels[:, 0] == 1.0] = 0.0
    
    unique_colors, counts = np.unique(quantized_pixels, axis=0, return_counts=True)
    
    unique_rgb = hsv_to_rgb(unique_colors.reshape(1, -1, 3))[0]
    luminances = (0.2126 * unique_rgb[:, 0] + 
                  0.7152 * unique_rgb[:, 1] + 
                  0.0722 * unique_rgb[:, 2])
    
    scores = counts * np.power(luminances + 0.05, luminance_bias)
    sorted_indices = np.argsort(scores)[::-1]
    
    sorted_colors = unique_colors[sorted_indices]
    sorted_counts = counts[sorted_indices] 
    
    final_colors_hsv = []
    final_proportions = []
    
    for color, count in zip(sorted_colors, sorted_counts):
        if len(final_colors_hsv) == 0:
            final_colors_hsv.append(color)
            final_proportions.append(count / total_pixels)
            continue
            
        is_too_similar = False
        for accepted_color in final_colors_hsv:
            dist = color_distance(color, accepted_color)
            if dist < min_distance:
                is_too_similar = True
                break 
                
        if not is_too_similar:
            final_colors_hsv.append(color)
            final_proportions.append(count / total_pixels)
            
        if len(final_colors_hsv) == n_colors:
            break
            
    # MATPLOT PLOTTING USE RGB
    final_colors_rgb = hsv_to_rgb(np.array(final_colors_hsv).reshape(1, -1, 3))[0]
    return final_colors_rgb, final_proportions

# MAIN

def generate_donut_color_wheel_png(image_path, num_bins=120):
    
    filename = os.path.basename(image_path)
    base_name, _ = os.path.splitext(filename)
    output_filename = f"wheel_{base_name}.png"
    
    print(f"Processing: {filename} -> {output_filename}")
    
    # LOAD IMAGE
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img) / 255.0

    # COLOE PALETTE
    palette_rgb, proportions = extract_varied_colors(
        img_array, 
        n_colors=N_COLORS, 
        group_factor=GROUP_FACTOR, 
        min_distance=MIN_DISTANCE, 
        luminance_bias=LUMINANCE_BIAS
    )

    # COLOR HIST
    hsv_img = rgb_to_hsv(img_array)
    hues = hsv_img[:, :, 0].flatten()        
    saturations = hsv_img[:, :, 1].flatten() 
    
    luminances = (0.2126 * img_array[:, :, 0] + 
                  0.7152 * img_array[:, :, 1] + 
                  0.0722 * img_array[:, :, 2]).flatten()
                  
    luminance_curve = LUMINANCE_CURVE
    luminances_curved = np.power(luminances, luminance_curve)
    combined_weights = saturations * luminances_curved
    
    hist_raw, bin_edges = np.histogram(hues, bins=num_bins, range=(0.0, 1.0), weights=combined_weights)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    theta = bin_centers * 2 * np.pi

    smoothing_sigma = SMOOTHING_SIGMA
    hist = gaussian_filter1d(hist_raw, sigma=smoothing_sigma, mode='wrap')
    
    # HISTOGRAM INNER AND OUTER RADIUS
    R_inner = 5.6
    R_outer = 6.4
    max_line_extension = 3.6 
    
    max_val = np.max(hist) if np.max(hist) > 0 else 1
    hist_normalized = (hist / max_val) * max_line_extension + 0.4 
    hist_plot = R_outer + hist_normalized
    
    theta_closed = np.append(theta, theta[0])
    hist_plot_closed = np.append(hist_plot, hist_plot[0])

    # GRAPHING SETUP
    dpi = 100
    fig = plt.figure(figsize=(512/dpi, 600/dpi), dpi=dpi)
    bg_color = BACKGROUND
    fig.patch.set_facecolor(bg_color)
    
    split_y = 0.80 
    
    box_height = split_y 
    box_bottom = 0.0
    box_width = (box_height * 600) / 512 
    box_left = (1.0 - box_width) / 2 
    
    ax = fig.add_axes([box_left, box_bottom, box_width, box_height], projection='polar')
    ax.set_facecolor(bg_color)
    ax.set_yticklabels([])
    ax.set_xticklabels([]) 
    ax.spines['polar'].set_visible(False)
    ax.grid(False) 
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1) 
    ax.set_ylim(0, 10.0) # PLOT_LIMIT

    
    theta_bg = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    width_bg = (2 * np.pi) / 360
    bars_bg = ax.bar(theta_bg, np.full(360, R_outer - R_inner), width=width_bg, bottom=R_inner, zorder=1)

    for i, bar in enumerate(bars_bg):
        hue_val = i / 360.0 
        hsv_color = np.array([[[hue_val, 1.0, 1.0]]])
        rgb_color = hsv_to_rgb(hsv_color)[0, 0]
        bar.set_facecolor(rgb_color)
        bar.set_edgecolor(rgb_color)

    # HIST LINE
    ax.plot(theta_closed, hist_plot_closed, color='white', linewidth=2.0, zorder=3, clip_on=False)
    ax.fill_between(theta_closed, R_outer, hist_plot_closed, color='white', alpha=0.2, zorder=2, clip_on=False)
    
    # Thumbnail
    aspect_ratio = img.width / img.height
    thumb_height = 1.0 - split_y
    thumb_width = thumb_height * (600 / 512) * aspect_ratio
    
    ax_img = fig.add_axes([0.0, split_y, thumb_width, thumb_height], zorder=5)
    ax_img.imshow(img)
    ax_img.axis('off') 

    # Title
    title_x = thumb_width + 0.02
    title_y = split_y + ((1.0 - split_y) / 2.0)
    title_text = f"Saturation & Curved Luminance:\n{filename}"
    fig.text(title_x, title_y, title_text, color='white', fontsize=11, fontweight='bold', ha='left', va='center', zorder=5)

    # RENDER PALETTE
    palette_ax = fig.add_axes([0.02, split_y - 0.14, 0.40, 0.12], zorder=6)
    palette_ax.axis('off')
    
    n_extracted = len(palette_rgb)
    palette_ax.set_xlim(0, n_extracted)
    palette_ax.set_ylim(-1, 1)
    
    for i, (color, prop) in enumerate(zip(palette_rgb, proportions)):
        palette_ax.add_patch(plt.Rectangle((i, 0), 0.9, 5, facecolor=color, edgecolor='none'))
        palette_ax.text(i + 0.45, -0.4, f"{prop*100:.1f}%", color='white', fontsize=8, ha='center', va='top')

    # Export
    plt.savefig(output_filename, facecolor=bg_color, dpi=dpi)
    plt.close(fig)

if __name__ == "__main__":
    search_patterns = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    image_files = []
    
    for pattern in search_patterns:
        image_files.extend(glob.glob(pattern))
    
    image_files = list(set(image_files))
    image_files = [f for f in image_files if not f.startswith("wheel_")]
    
    if not image_files:
        print("No source PNG or JPG files found in the current directory.")
    else:
        print(f"Found {len(image_files)} unique image file(s). Generating outputs...")
        for file in image_files:
            generate_donut_color_wheel_png(file, num_bins=NUM_BINS)
        print("Done!")