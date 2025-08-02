# Standard library imports
from PIL import Image

# Third-party numerical/scientific imports
import numpy as np

# Matplotlib imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import proj3d
import mplcursors

# Plotly imports
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots



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
    cmap = plt.cm.plasma  
    colors = cmap(norm(t_safe))
    sc = ax.scatter(x, y, z, c=t_safe, norm=norm, **scatter_kwargs)

    step = max(1, len(x) // 100)
    for j in range(0, len(x), step):
        color_hex = mcolors.to_hex(colors[j])
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
    cbar.set_ticks([np.min(t_safe), np.max(t_safe)])
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    cbar.ax.tick_params(labelsize=14) 
    
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

    # Tomar overlay_scale del userInput
    overlay_scale = float(userInput["overlay_scale"].iloc[0])

    new_w = int(floorplan_img.width * overlay_scale)
    new_h = int(overlay_img.height * (new_w / overlay_img.width))
    overlay_img = overlay_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    overlay_rotation = float(userInput["overlay_rotation"].iloc[0])
    overlay_img = overlay_img.rotate(overlay_rotation, expand=True, resample=Image.Resampling.BICUBIC)
    new_w, new_h = overlay_img.size

    pos_x_pix = int((pos_x / dim_x) * floorplan_img.width) - new_w // 2
    pos_y_pix = floorplan_img.height - int((pos_y / dim_y) * floorplan_img.height) - new_h // 2

    combined_img = floorplan_img.copy()
    combined_img.paste(overlay_img, (pos_x_pix, pos_y_pix), overlay_img)
    combined_img.save("floorplan_with_overlay.png")
    return fig, ax

def interactive_3d(fig, ax, x, y, z, I_f, I_dB_f, t_f, I_plot, t_safe):
    """
    Creates an interactive 3D visualization of acoustic data with:
    - 3D scatter plot showing sound direction and intensity
    - 2D time-intensity plot below
    - Interactive view rotation button
    - Hover tooltips and click annotations
    
    Parameters:
        fig: matplotlib figure object
        ax: matplotlib axes object
        x, y, z: Arrays of 3D coordinates
        I_f: Intensity values (linear scale)
        I_dB_f: Intensity values (dB scale)
        t_f: Time values in ms
        I_plot: Intensity values for plotting
        t_safe: Time values for color mapping
    """
    plt.close(fig)
    
    # =============================================
    # FIGURE LAYOUT SETUP
    # =============================================
    # Create figure with custom layout using GridSpec
    # Top row: 3D plot spanning both columns
    # Bottom row: 2D plot (left) and view button (right)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, 
                        height_ratios=[6, 1],  # 6:1 ratio for 3D vs 2D plots
                        width_ratios=[6, 1],   # 6:1 ratio for plots vs button
                        hspace=0.1, wspace=0.1)
    
    # Create plot areas
    ax = fig.add_subplot(gs[0, :], projection='3d')  # Main 3D plot
    ax_2d = fig.add_subplot(gs[1, 0])                # 2D time plot
    ax_buttons = fig.add_subplot(gs[1, 1])           # Button area
    ax_buttons.axis('off')  # Hide axes for button area
    
    # Create view rotation button with custom styling
    btn_view_ax = fig.add_axes([0.82, 0.15, 0.15, 0.07])  # Button position/size
    btn_view = Button(btn_view_ax, 'Change view', 
                     color='#ffffff', hovercolor='#e3ffff')

    # =============================================
    # APPLICATION STATE
    # =============================================
    # Dictionary to store interactive elements state
    state = {
        'views': [  # Predefined camera views
            {'elev': 45, 'azim': 45, 'name': "Isometric"},
            {'elev': 90, 'azim': -90, 'name': "Top"}, 
            {'elev': 0, 'azim': 0, 'name': "Front"},
            {'elev': 0, 'azim': -90, 'name': "Side"}
        ],
        'current_view': 0,  # Current view index
        # Annotation and marker storage
        'left_annotation': None,
        'right_annotation': None,
        'left_marker_3d': None,
        'right_marker_3d': None,
        'left_marker_2d': None,
        'right_marker_2d': None,
        'scatter': None,    # 3D scatter plot object
        'cursor': None     # Interactive cursor object
    }

    # =============================================
    # VIEW CONTROL FUNCTIONS
    # =============================================
    def change_view(event):
        """Cycles through predefined 3D views"""
        state['current_view'] = (state['current_view'] + 1) % len(state['views'])
        view = state['views'][state['current_view']]
        ax.view_init(elev=view['elev'], azim=view['azim'])
        btn_view.label.set_text(f"View: {view['name']}")
        fig.canvas.draw_idle()

    # Connect button to view change function
    btn_view.on_clicked(change_view)

    # =============================================
    # PLOT SETUP FUNCTIONS
    # =============================================
    def setup_plots():
        """
        Initializes or updates both 2D and 3D plots
        Configures axes, labels, grids, and visual elements
        """
        # Clear and setup 2D time plot
        ax_2d.clear()
        ax_2d.plot(t_f, I_dB_f, 'b-', alpha=1, linewidth=0.8)
        ax_2d.set_xlabel('Time', fontsize=12)
        ax_2d.set_ylabel('Intensity: ', fontsize=12)
        ax_2d.grid(True, alpha=0.3)
        
        # Clear and setup 3D plot
        ax.clear()
        view = state['views'][state['current_view']]
        ax.view_init(elev=view['elev'], azim=view['azim'])
        
        # Set 3D plot limits and labels
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_xticklabels(['Back (X-)', '', 'Front (X+)'], fontsize=12)
        ax.set_yticks([-0.5, 0, 0.5])
        ax.set_yticklabels(['Right (Y-)', '', 'Left (Y+)'], fontsize=12)
        ax.set_zticks([-0.5, 0, 0.5])
        ax.set_zticklabels(['Down (Z-)', '', 'Up (Z+)'], fontsize=12)

        # Style 3D axes
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.set_visible(True)
            axis.pane.set_edgecolor('lightgray')
            axis.line.set_color('black')
            axis._axinfo['grid'].update({'color': (0, 0, 0, 0.2)})
        ax.set_box_aspect([1, 1, 1])
        
        # Create 3D scatter plot with direction vectors
        norm = LogNorm(vmin=np.min(t_safe), vmax=np.max(t_safe))
        cmap = plt.cm.plasma
        colors = cmap(norm(t_safe))
        state['scatter'] = ax.scatter(x, y, z, c=t_safe, marker=".", s=20,
                                    cmap=cmap, alpha=1, norm=norm)
        
        # Draw direction vectors (subsampled for clarity)
        step = max(1, len(x) // 100)
        for j in range(0, len(x), step):
            ax.plot([0, x[j]], [0, y[j]], [0, z[j]], 
                   color=colors[j], linewidth=2.75, alpha=0.7)
        
        # Restore annotations if they exist
        if state['left_annotation']:
            recreate_annotation('left', (-1.25, 0.60), '#24e5ff')
        if state['right_annotation']:
            recreate_annotation('right', (-1.25, 0.40), '#1ce61f')

    def recreate_annotation(side, pos, color):
        """
        Recreates an annotation after view changes
        Args:
            side: 'left' or 'right' annotation
            pos: (x,y) position in axes coordinates
            color: Annotation color
        """
        i = state[f'{side}_annotation'].ind
        state[f'{side}_annotation'] = ax.annotate(
            f"Intensity [dB]: {I_dB_f[i]:.1f} dB\n"
            f"Time: {t_f[i]:.1f} ms\n"
            f"Elevation: {np.degrees(np.arcsin(z[i]/(I_plot[i]+1e-30))):.1f}°\n"
            f"Azimuth: {np.degrees(np.arctan2(y[i], x[i])):.1f}°",
            xy=pos,
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white',
                     edgecolor=color, linewidth=3),
            fontsize=11,
            color='black'
        )
        state[f'{side}_annotation'].ind = i
        
        # Add 3D marker at annotated point
        state[f'{side}_marker_3d'] = ax.scatter(
            [x[i]], [y[i]], [z[i]],
            color=color,
            edgecolor='black',
            linewidth=1,
            s=150,
            alpha=1,
            zorder=10
        )

    # =============================================
    # INTERACTIVITY FUNCTIONS
    # =============================================
    def reconnect_interactivity():
        """Reconnects interactive elements after plot updates"""
        if state['scatter']:
            # Remove old cursor if exists
            if state['cursor']:
                state['cursor'].remove()
            
            # Create new cursor and connect handlers
            state['cursor'] = mplcursors.cursor(state['scatter'], hover=True)
            state['cursor'].connect("add", on_hover)
            
            # Reconnect click handler
            if hasattr(fig, '_button_press_cid'):
                fig.canvas.mpl_disconnect(fig._button_press_cid)
            fig._button_press_cid = fig.canvas.mpl_connect('button_press_event', onclick)

    def on_hover(sel):
        """Shows tooltip when hovering over points"""
        i = sel.index
        text = (
            f"Intensity [dB]: {I_dB_f[i]:.1f} dB\n"
            f"Time: {t_f[i]:.1f} ms\n"
            f"Elevation: {np.degrees(np.arcsin(z[i]/(I_plot[i]+1e-30))):.1f}°\n"
            f"Azimuth: {np.degrees(np.arctan2(y[i], x[i])):.1f}°"
        )
        sel.annotation.set_text(text)
        sel.annotation.get_bbox_patch().set(facecolor='white', edgecolor='black', alpha=1.0)
        sel.annotation.set_fontsize(10)
        sel.annotation.xyann = (-0.10, 0.06)

    def onclick(event):
        """Handles mouse clicks to create persistent annotations"""
        if event.inaxes != ax or not state['scatter'].contains(event)[0]:
            return
            
        i = state['scatter'].contains(event)[1]["ind"][0]
        
        # Determine if left or right click
        if event.button == 1:  # Left click
            color = '#24e5ff'
            pos = (-0.30, 0.60)
            annotation_store = 'left_annotation'
            marker_3d_store = 'left_marker_3d'
            marker_2d_store = 'left_marker_2d'
        else:  # Right click
            color = '#1ce61f'
            pos = (-0.30, 0.40)
            annotation_store = 'right_annotation'
            marker_3d_store = 'right_marker_3d'
            marker_2d_store = 'right_marker_2d'
        
        # Clear previous elements
        for element in [annotation_store, marker_3d_store, marker_2d_store]:
            if state[element]:
                state[element].remove()
        
        # Create new annotation
        state[annotation_store] = ax.annotate(
            f"Intensity [dB]: {I_dB_f[i]:.1f} dB\n"
            f"Time: {t_f[i]:.1f} ms\n"
            f"Elevation: {np.degrees(np.arcsin(z[i]/(I_plot[i]+1e-30))):.1f}°\n"
            f"Azimuth: {np.degrees(np.arctan2(y[i], x[i])):.1f}°",
            xy=pos,
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white',
                     edgecolor=color, linewidth=3),
            fontsize=11,
            color='black'
        )
        state[annotation_store].ind = i  # Store point index
        
        # Add 3D marker
        state[marker_3d_store] = ax.scatter(
            [x[i]], [y[i]], [z[i]],
            color=color,
            edgecolor='black',
            linewidth=1,
            s=150,
            alpha=1,
            zorder=10
        )
        
        # Add 2D time marker
        state[marker_2d_store] = ax_2d.axvline(
            t_f[i],
            color=color,
            linestyle='--',
            linewidth=2,
            alpha=1,
            label=f'{t_f[i]:.1f} ms'
        )
        
        ax_2d.legend(loc='upper right')
        fig.canvas.draw_idle()

    # =============================================
    # INITIALIZATION
    # =============================================
    setup_plots()  # Create initial plots
    
    # Set up interactivity
    state['cursor'] = mplcursors.cursor(state['scatter'], hover=True)
    state['cursor'].connect("add", on_hover)
    fig.canvas.mpl_connect('button_press_event', onclick)
    
    plt.tight_layout()
    plt.show()

def interactive_multiview(x, y, z, I_f, I_dB_f, t_f, I_plot, t_safe):
    # Configuración idéntica a tu estilo original
    fig = plt.figure(figsize=(10, 10))
    
    # Definición de vistas (elev, azim, título) como en tu custom_views()
    views = [
        (45, -45, "isometric"),
        (90, -90, "top"),
        (0, 0, "front"),
        (0, -90, "side")
    ]
    
    axes = []
    
    for i, (elev, azim, title) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        axes.append(ax)
        
        # Configuración IDÉNTICA a tu función original
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        
        if title == "top":
            # Configuración específica para vista superior (como en top_view_overlay)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis.pane.set_visible(False)
                axis._axinfo["grid"]['color'] = (0, 0, 0, 0)
                axis.line.set_color((0, 0, 0, 0))
        else:
            # Configuración estándar (como en interactive_3d)
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
        
        # SCATTER y LÍNEAS con tus parámetros originales exactos
        norm = LogNorm(vmin=np.min(t_safe), vmax=np.max(t_safe))
        cmap = plt.cm.plasma
        colors = cmap(norm(t_safe))
        sc = ax.scatter(x, y, z, c=t_safe, marker=".", s=20, cmap=cmap, alpha=0.6)
        
        step = max(1, len(x) // 100)  # Mismo muestreo que usabas
        for j in range(0, len(x), step):
            ax.plot([0, x[j]], [0, y[j]], [0, z[j]], color=colors[j], linewidth=2, alpha=0.5)
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title.capitalize(), fontsize=12)
    
    # TOOLTIPS IDÉNTICOS a tu función original
    cursor = mplcursors.cursor(sc, hover=False)
    current_tooltip = None  # Guardará la referencia al tooltip activo
    @cursor.connect("add")
    def on_add(sel):
        nonlocal current_tooltip
        
        # Si clickeamos el mismo punto
        if current_tooltip is not None and current_tooltip[0] == sel.index:
            current_tooltip[1].remove()  # Eliminamos completamente el tooltip
            current_tooltip = None
        else:
            # Si hay un tooltip previo, lo eliminamos
            if current_tooltip is not None:
                current_tooltip[1].remove()
            
            # Creamos el nuevo tooltip
            sel.annotation.set_text(
                f"Intensity [dB]: {I_dB_f[sel.index]:.1f} dB\n"
                f"Time: {t_f[sel.index]:.1f} ms\n"
                f"Elevation: {np.degrees(np.arcsin(z[sel.index]/(I_plot[sel.index]+1e-30))):.1f}°\n"
                f"Azimuth: {np.degrees(np.arctan2(y[sel.index], x[sel.index])):.1f}°"
            )
            current_tooltip = (sel.index, sel.annotation)  # Guardamos (índice, referencia)
        
        fig.canvas.draw_idle()
    plt.show()

def plot_intensity_3d(I_peaks, az_deg, el_deg, time_peaks, t, userInput):
    # Transform Intensity, azimuth, elevation, time and threshold to an interactive figure.
    x, y, z, I_f, I_dB_f, t_f, t_safe, I_plot = preprocess_data(I_peaks, az_deg, el_deg, time_peaks, userInput)
    # Save floorplan with top view as overlay.
    fig, ax = save_floorplan(x, y, z, I_f, I_dB_f, I_plot, t_f, t_safe, userInput)
    # Generate interactive objects and .png views.
    # custom_views(x, y, z, I_f, I_dB_f, t_f, I_plot, t_safe)
    # Interactive object:
    interactive_3d(fig, ax, x, y, z, I_f, I_dB_f, t_f, I_plot, t_safe)
    # interactive_multiview(x, y, z, I_f, I_dB_f, t_f, I_plot, t_safe)
