import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from matplotlib.colors import LogNorm
from PIL import Image

import plotly.graph_objs as go
import plotly.io as pio

def preprocess_data(I_peaks, az_deg, el_deg, time_peaks, userInput):
    threshold = int(userInput["Threshold_dB"].iloc[0])
    I_abs = I_peaks.copy()
    I_max = np.max(I_abs) + 1e-30
    I_dB = 20 * np.log10(I_abs / I_max)

    mask = I_dB >= threshold
    I_f = I_abs[mask]
    az_f = az_deg[mask]
    el_f = el_deg[mask]
    t_f = time_peaks[mask]
    I_dB_f = I_dB[mask]

    desired_min = 0.05
    desired_max = 1.0
    scale = desired_max - desired_min
    I_plot = ((I_dB_f - threshold) / -threshold) * scale + desired_min

    y = I_plot * np.cos(np.deg2rad(el_f)) * np.sin(np.deg2rad(az_f))
    x = I_plot * np.cos(np.deg2rad(el_f)) * np.cos(np.deg2rad(az_f))
    z = I_plot * np.sin(np.deg2rad(el_f))

    t_safe = t_f.copy()
    t_safe[t_safe <= 0] = np.min(t_safe[t_safe > 0])

    return x, y, z, I_f, I_dB_f, t_f, t_safe, I_plot

def plot_3d_data(ax, x, y, z, t_safe, scatter_kwargs=None, line_kwargs=None):
    if scatter_kwargs is None:
        scatter_kwargs = dict(marker=".", s=20, cmap=plt.plasma(), alpha=0.6)
    if line_kwargs is None:
        line_kwargs = dict(linewidth=2, alpha=0.5)

    norm = LogNorm(vmin=np.min(t_safe), vmax=np.max(t_safe))
    cmap = plt.cm.viridis
    colors = cmap(norm(t_safe))
    sc = ax.scatter(x, y, z, c=t_safe, **scatter_kwargs)

    step = max(1, len(x) // 100)
    for j in range(0, len(x), step):
        ax.plot([0, x[j]], [0, y[j]], [0, z[j]], color=colors[j], **line_kwargs)
    return sc

def top_view_overlay(x, y, z, I_f, I_dB_f, I_plot, t_f, t_safe):
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    sc = plot_3d_data(
        ax, x, y, z, t_safe,
        scatter_kwargs=dict(marker=".", linewidths=4, cmap=plt.plasma(), alpha=0.5),
        line_kwargs=dict(linewidth=4, alpha=0.7)
    )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=90, azim=-90)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_visible(False)
        axis._axinfo["grid"]['color'] = (0, 0, 0, 0)
        axis.line.set_color((0, 0, 0, 0))

    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=20, pad=0.05)
    cbar.set_label("Time [ms]")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.grid(False)
    ax.set_box_aspect([1, 1, 1])

    return fig, ax

def save_floorplan(x, y, z, I_f, I_dB_f, I_plot, t_f, t_safe, userInput):
    # Generate figure.
    fig, ax = top_view_overlay(x, y, z, I_f, I_dB_f, I_plot, t_f, t_safe)

    overlay_name = "top_view_overlay.png"
    plt.savefig(overlay_name, dpi=600, bbox_inches='tight', transparent=True)

    floorplan_path = userInput["FloorPlan"].iloc[0]
    floorplan_img = Image.open(floorplan_path).convert("RGBA")
    overlay_img = Image.open(overlay_name).convert("RGBA")

    dim_x = float(userInput["Dim_X"].iloc[0])
    dim_y = float(userInput["Dim_Y"].iloc[0])
    pos_x = float(userInput["Pos_X"].iloc[0])
    pos_y = float(userInput["Pos_Y"].iloc[0])

    overlay_scale = 0.75
    new_w = int(floorplan_img.width * overlay_scale)
    new_h = int(overlay_img.height * (new_w / overlay_img.width))
    overlay_img = overlay_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    pos_x_pix = int((pos_x / dim_x) * floorplan_img.width) - new_w // 2
    pos_y_pix = floorplan_img.height - int((pos_y / dim_y) * floorplan_img.height) - new_h // 2

    combined_img = floorplan_img.copy()
    combined_img.paste(overlay_img, (pos_x_pix, pos_y_pix), overlay_img)
    combined_img.save("floorplan_with_overlay.png")
    return fig, ax

def custom_views(x, y, z, I_f, I_dB_f, t_f, I_plot, t_safe):
    fig_size = (8, 8)

    def setup_axis(ax):
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xticks([-0.5, 0.5])
        ax.set_xticklabels(['Back (X-)', 'Front (X+)'])
        ax.set_yticks([-0.5, 0.5])
        ax.set_yticklabels(['Right (Y-)', 'Left (Y+)'])
        ax.set_zticks([-0.5, 0.5])
        ax.set_zticklabels(['Down (Z-)', 'Up (Z+)'])
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.set_visible(True)
            axis.pane.set_edgecolor('lightgray')
            axis.line.set_color('black')
            axis._axinfo["grid"]['color'] = (0, 0, 0, 0.2)

    views = [
        (45, -45, "isometric"),
        (90, -90, "top"),
        (0, 0, "front"),
        (0, -90, "side")
    ]

    for elev, azim, title in views:
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d')

        norm = LogNorm(vmin=np.min(t_safe), vmax=np.max(t_safe))
        cmap = plt.cm.viridis
        colors = cmap(norm(t_safe))
        sc = ax.scatter(x, y, z, c=t_safe, marker=".", s=20, cmap=plt.plasma(), alpha=0.6)

        step = max(1, len(x) // 100)
        for j in range(0, len(x), step):
            ax.plot([0, x[j]], [0, y[j]], [0, z[j]], color=colors[j], linewidth=2, alpha=0.5)

        setup_axis(ax)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title.capitalize())

        filename = f"view_{title}.png"
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        plt.close(fig)

def interactive_3d(fig, ax, x, y, z, I_f, I_dB_f, t_f, I_plot, t_safe):
    # Isometric
    ax.view_init(elev=45, azim=45)

    # Axes
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_xticklabels(['Back (X-)', '', 'Front (X+)'])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_yticklabels(['Right (Y-)', '', 'Left (Y+)'])
    ax.set_zticks([-0.5, 0, 0.5])
    ax.set_zticklabels(['Down (Z-)', '', 'Up (Z+)'])

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_visible(True)
        axis.pane.set_edgecolor('lightgray')
        axis.line.set_color('black')
        axis._axinfo['grid']['color'] = (0, 0, 0, 0.2)

    ax.grid(True)
    ax.set_box_aspect([1,1,1])

    # Add cursor interactivity
    sc = ax.collections[0]

    def on_add(sel):
        idx = sel.index
        elev_ = np.degrees(np.arcsin(z[idx] / (I_plot[idx] + 1e-30)))
        azim_ = np.degrees(np.arctan2(y[idx], x[idx]))
        sel.annotation.set_text(
            f"Intensity [dB]: {I_dB_f[idx]:.1f} dB\n"
            f"Time: {t_f[idx]:.1f} ms\n"
            f"Elevation: {elev_:.1f}°\n"
            f"Azimuth: {azim_:.1f}°"
        )

    cursor = mplcursors.cursor(sc, hover=True)
    cursor.connect("add", on_add)
    plt.show()


def plot_intensity_3d(I_peaks, az_deg, el_deg, time_peaks, t, userInput):
    # Transform Intensity, azimuth, elevation, time and threshold to an interactive figure.
    x, y, z, I_f, I_dB_f, t_f, t_safe, I_plot = preprocess_data(I_peaks, az_deg, el_deg, time_peaks, userInput)
    # Save floorplan with top view as overlay.
    fig, ax = save_floorplan(x, y, z, I_f, I_dB_f, I_plot, t_f, t_safe, userInput)
    # Generate interactive objects and .png views.
    custom_views(x, y, z, I_f, I_dB_f, t_f, I_plot, t_safe)
    # Interactive object:
    interactive_3d(fig, ax, x, y, z, I_f, I_dB_f, t_f, I_plot, t_safe)
        
        
        
        
