"""
VHL Holographic Resonance - Streamlit Web Interface

Interactive web application for exploring holographic resonance patterns
with real-time parameter adjustment and visualization.

Features:
- Sidebar controls for all simulation parameters
- Multiple trajectory types (Fourier presets, custom, RNN-driven)
- Element mapping with VHL orbital parameters
- Real-time 3D visualization using Plotly
- Export capabilities for data and visualizations
- Integration with VHL laws (polarity, nodal spacing, helical structure)

Usage:
    streamlit run vhl_resonance_streamlit.py

Author: Zynerji
Date: 2025-12-15
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from vhl_holographic_resonance import (
    HolographicResonator,
    ParticleAdvector
)
from vhl_vortex_controller import (
    FourierTrajectory,
    VortexRNN,
    MultiVortexChoreographer
)
import torch
from typing import Optional


# Page configuration
st.set_page_config(
    page_title="VHL Holographic Resonance",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00ffff;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #ff00ff;
        margin-top: 1rem;
    }
    .metric-box {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)


# Element data (from VHL framework)
ELEMENT_DATA = {
    'H': {'atomic_number': 1, 'mass': 1.008, 'radius': 0.53, 'polarity': 1},
    'He': {'atomic_number': 2, 'mass': 4.003, 'radius': 0.31, 'polarity': -1},
    'C': {'atomic_number': 6, 'mass': 12.011, 'radius': 0.77, 'polarity': 1},
    'N': {'atomic_number': 7, 'mass': 14.007, 'radius': 0.75, 'polarity': -1},
    'O': {'atomic_number': 8, 'mass': 15.999, 'radius': 0.73, 'polarity': 1},
    'F': {'atomic_number': 9, 'mass': 18.998, 'radius': 0.71, 'polarity': -1},
    'Ne': {'atomic_number': 10, 'mass': 20.180, 'radius': 0.69, 'polarity': 1},
    'Fe': {'atomic_number': 26, 'mass': 55.845, 'radius': 1.26, 'polarity': 1},
    'Cu': {'atomic_number': 29, 'mass': 63.546, 'radius': 1.28, 'polarity': -1},
    'Au': {'atomic_number': 79, 'mass': 196.967, 'radius': 1.44, 'polarity': 1},
}


@st.cache_resource
def initialize_resonator(num_sources, helical_turns, grid_resolution, sphere_radius):
    """Initialize holographic resonator (cached for performance)."""
    resonator = HolographicResonator(
        sphere_radius=sphere_radius,
        grid_resolution=grid_resolution,
        num_sources=num_sources,
        helical_turns=helical_turns
    )
    return resonator


@st.cache_resource
def load_trained_rnn():
    """Load pre-trained RNN model if available."""
    try:
        rnn = VortexRNN(input_size=4, hidden_size=32, num_layers=2)
        # rnn.load_state_dict(torch.load('models/vortex_rnn.pth'))
        # For now, return untrained model
        return rnn
    except Exception as e:
        st.warning(f"Could not load RNN model: {e}")
        return VortexRNN(input_size=4, hidden_size=32, num_layers=2)


def create_isosurface_plot(resonator, num_levels=3):
    """Create Plotly 3D isosurface visualization."""
    # Get intensity grid
    intensity_3d = resonator.get_intensity_3d()

    # Create coordinate grids
    n = resonator.grid_resolution
    r = resonator.sphere_radius
    x = np.linspace(-r, r, n)
    y = np.linspace(-r, r, n)
    z = np.linspace(-r, r, n)

    # Create figure
    fig = go.Figure()

    # Add isosurfaces
    max_intensity = intensity_3d.max()
    levels = np.linspace(0.3 * max_intensity, 0.9 * max_intensity, num_levels)
    colors = ['cyan', 'magenta', 'yellow', 'lime', 'orange']

    for i, level in enumerate(levels):
        fig.add_trace(go.Isosurface(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=intensity_3d.flatten(),
            isomin=level,
            isomax=level,
            surface_count=1,
            opacity=0.6,
            colorscale=[[0, colors[i % len(colors)]], [1, colors[i % len(colors)]]],
            showscale=False,
            name=f'Level {i+1}'
        ))

    # Add boundary sphere (wireframe approximation)
    theta = np.linspace(0, 2*np.pi, 30)
    phi = np.linspace(0, np.pi, 30)
    sphere_x = r * np.outer(np.cos(theta), np.sin(phi))
    sphere_y = r * np.outer(np.sin(theta), np.sin(phi))
    sphere_z = r * np.outer(np.ones(30), np.cos(phi))

    fig.add_trace(go.Surface(
        x=sphere_x, y=sphere_y, z=sphere_z,
        opacity=0.1,
        colorscale='Blues',
        showscale=False,
        name='Boundary'
    ))

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-r, r], title='X'),
            yaxis=dict(range=[-r, r], title='Y'),
            zaxis=dict(range=[-r, r], title='Z'),
            aspectmode='cube',
            bgcolor='black'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='black',
        height=600
    )

    return fig


def create_intensity_slice_plot(resonator, slice_axis='z', slice_position=0.0):
    """Create 2D slice through intensity field."""
    intensity_3d = resonator.get_intensity_3d()
    n = resonator.grid_resolution
    r = resonator.sphere_radius

    # Get slice index
    slice_idx = int((slice_position / r + 1) * n / 2)
    slice_idx = np.clip(slice_idx, 0, n-1)

    # Extract slice
    if slice_axis == 'z':
        slice_data = intensity_3d[:, :, slice_idx]
        xlabel, ylabel = 'X', 'Y'
    elif slice_axis == 'y':
        slice_data = intensity_3d[:, slice_idx, :]
        xlabel, ylabel = 'X', 'Z'
    else:  # x
        slice_data = intensity_3d[slice_idx, :, :]
        xlabel, ylabel = 'Y', 'Z'

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=slice_data,
        colorscale='Hot',
        colorbar=dict(title='Intensity')
    ))

    fig.update_layout(
        title=f'{slice_axis.upper()}-slice at {slice_position:.2f}',
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=400,
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white')
    )

    return fig


def create_vortex_trajectory_plot(choreographer, time_range, num_points=200):
    """Visualize vortex trajectories over time."""
    times = np.linspace(time_range[0], time_range[1], num_points)

    fig = go.Figure()

    for i in range(choreographer.num_vortices):
        if i < len(choreographer.trajectories):
            traj = choreographer.trajectories[i]
            if traj is not None:
                positions = traj.evaluate_batch(times)

                fig.add_trace(go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode='lines',
                    name=f'Vortex {i+1}',
                    line=dict(width=4)
                ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1, 1], title='X'),
            yaxis=dict(range=[-1, 1], title='Y'),
            zaxis=dict(range=[-1, 1], title='Z'),
            aspectmode='cube',
            bgcolor='black'
        ),
        paper_bgcolor='black',
        height=500,
        title='Vortex Trajectories'
    )

    return fig


def main():
    # Header
    st.markdown('<div class="main-header">🌀 VHL Holographic Resonance Simulator</div>',
                unsafe_allow_html=True)

    st.markdown("""
    Interactive visualization of wave interference patterns on a holographic sphere.
    Explore multi-vortex dynamics, Fourier trajectories, and resonant structures.
    """)

    # Sidebar controls
    st.sidebar.title("⚙️ Simulation Parameters")

    # === Basic Parameters ===
    st.sidebar.markdown("### Basic Configuration")

    num_sources = st.sidebar.slider(
        "Number of Wave Sources",
        min_value=50, max_value=2000, value=500, step=50,
        help="More sources = finer lattice resolution"
    )

    helical_turns = st.sidebar.slider(
        "Helical Turns",
        min_value=1, max_value=20, value=5, step=1,
        help="Spiral twist of lattice on sphere"
    )

    grid_resolution = st.sidebar.select_slider(
        "Grid Resolution",
        options=[32, 48, 64, 96, 128],
        value=64,
        help="Higher = more detail but slower computation"
    )

    sphere_radius = st.sidebar.slider(
        "Sphere Radius",
        min_value=0.5, max_value=2.0, value=1.0, step=0.1
    )

    # === Frequency Parameters ===
    st.sidebar.markdown("### Frequency Configuration")

    base_frequency = st.sidebar.slider(
        "Base Frequency",
        min_value=0.5, max_value=5.0, value=1.0, step=0.1,
        help="Fundamental oscillation frequency"
    )

    freq_spread = st.sidebar.slider(
        "Frequency Spread",
        min_value=0.0, max_value=0.5, value=0.1, step=0.05,
        help="Randomness in source frequencies"
    )

    polarity_phase = st.sidebar.checkbox(
        "Polarity Phase Modulation",
        value=True,
        help="Alternate phase based on lattice position"
    )

    # === Element Mapping ===
    st.sidebar.markdown("### Element Mapping")

    use_element = st.sidebar.checkbox("Map to Element Properties", value=False)

    if use_element:
        selected_element = st.sidebar.selectbox(
            "Element",
            options=list(ELEMENT_DATA.keys()),
            index=5  # Default to F
        )

        element_props = ELEMENT_DATA[selected_element]
        st.sidebar.info(f"""
        **{selected_element}** Properties:
        - Atomic #: {element_props['atomic_number']}
        - Mass: {element_props['mass']:.3f} u
        - Radius: {element_props['radius']:.2f} Å
        - Polarity: {element_props['polarity']}
        """)

        # Modify parameters based on element
        base_frequency *= element_props['atomic_number'] / 10.0
        sphere_radius *= element_props['radius']

    # === Vortex Configuration ===
    st.sidebar.markdown("### Vortex Dynamics")

    num_vortices = st.sidebar.slider(
        "Number of Vortices",
        min_value=0, max_value=5, value=2, step=1
    )

    vortex_trajectories = []
    vortex_charges = []

    if num_vortices > 0:
        trajectory_type = st.sidebar.selectbox(
            "Trajectory Type",
            options=['circle', 'figure8', 'star', 'spiral', 'lissajous', 'custom'],
            index=0
        )

        if trajectory_type == 'custom':
            st.sidebar.markdown("**Custom Fourier Coefficients**")
            # Allow user to input custom coefficients (simplified)
            st.sidebar.info("Using default custom trajectory")

        for i in range(num_vortices):
            charge = st.sidebar.selectbox(
                f"Vortex {i+1} Charge",
                options=[-2, -1, 1, 2],
                index=1 if i % 2 == 0 else 0,
                key=f"charge_{i}"
            )
            vortex_charges.append(charge)

    # === Time Evolution ===
    st.sidebar.markdown("### Time Evolution")

    animate = st.sidebar.checkbox("Enable Animation", value=False)

    if animate:
        time = st.sidebar.slider(
            "Time",
            min_value=0.0, max_value=10.0, value=0.0, step=0.1
        )
    else:
        time = 0.0

    # === Visualization Options ===
    st.sidebar.markdown("### Visualization")

    show_isosurfaces = st.sidebar.checkbox("Show Isosurfaces", value=True)
    show_slice = st.sidebar.checkbox("Show 2D Slice", value=False)
    show_trajectories = st.sidebar.checkbox("Show Vortex Trajectories", value=True)

    num_iso_levels = st.sidebar.slider(
        "Isosurface Levels",
        min_value=1, max_value=5, value=3, step=1
    )

    # === Main Content ===

    # Initialize simulation
    with st.spinner("Initializing holographic resonator..."):
        resonator = initialize_resonator(
            num_sources, helical_turns, grid_resolution, sphere_radius
        )

        # Setup sources
        resonator.initialize_sources_from_lattice(
            base_frequency=base_frequency,
            frequency_spread=freq_spread,
            polarity_phase=polarity_phase
        )

        # Setup vortices
        if num_vortices > 0:
            choreographer = MultiVortexChoreographer(num_vortices=num_vortices)

            for i in range(num_vortices):
                angle = 2 * np.pi * i / num_vortices
                radius = 0.3 * sphere_radius

                center = np.array([
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    0.0
                ])

                # Add Fourier-controlled vortex
                choreographer.add_fourier_vortex(
                    preset=trajectory_type,
                    omega=1.0,
                    amplitude=0.3 * sphere_radius,
                    phase_offset=angle
                )

                resonator.add_vortex(
                    center=center,
                    topological_charge=vortex_charges[i]
                )

        # Compute field
        resonator.compute_field(time=time)
        resonator.compute_intensity()

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Wave Sources", f"{num_sources:,}")
    with col2:
        st.metric("Grid Points", f"{grid_resolution**3:,}")
    with col3:
        st.metric("Max Intensity", f"{resonator.intensity.max():.4f}")
    with col4:
        st.metric("Mean Intensity", f"{resonator.intensity.mean():.4f}")

    # Main visualization
    st.markdown('<div class="sub-header">3D Holographic Field</div>', unsafe_allow_html=True)

    if show_isosurfaces:
        with st.spinner("Rendering isosurfaces..."):
            fig_iso = create_isosurface_plot(resonator, num_levels=num_iso_levels)
            st.plotly_chart(fig_iso, use_container_width=True)

    # Additional visualizations
    viz_cols = st.columns(2)

    with viz_cols[0]:
        if show_slice:
            st.markdown("#### 2D Intensity Slice")
            slice_axis = st.selectbox("Slice Axis", ['x', 'y', 'z'], index=2)
            slice_pos = st.slider("Slice Position", -sphere_radius, sphere_radius, 0.0, 0.1)

            fig_slice = create_intensity_slice_plot(resonator, slice_axis, slice_pos)
            st.plotly_chart(fig_slice, use_container_width=True)

    with viz_cols[1]:
        if show_trajectories and num_vortices > 0:
            st.markdown("#### Vortex Trajectories")
            time_range = st.slider("Time Range", 0.0, 20.0, (0.0, 10.0), 0.5)

            fig_traj = create_vortex_trajectory_plot(choreographer, time_range)
            st.plotly_chart(fig_traj, use_container_width=True)

    # Statistics panel
    with st.expander("📊 Detailed Statistics"):
        stats_cols = st.columns(3)

        with stats_cols[0]:
            st.markdown("**Field Statistics**")
            st.write(f"- Min: {resonator.intensity.min():.6f}")
            st.write(f"- Max: {resonator.intensity.max():.6f}")
            st.write(f"- Mean: {resonator.intensity.mean():.6f}")
            st.write(f"- Std Dev: {resonator.intensity.std():.6f}")

        with stats_cols[1]:
            st.markdown("**Configuration**")
            st.write(f"- Sources: {num_sources}")
            st.write(f"- Vortices: {num_vortices}")
            st.write(f"- Grid: {grid_resolution}³")
            st.write(f"- Helical turns: {helical_turns}")

        with stats_cols[2]:
            st.markdown("**Frequency Info**")
            st.write(f"- Base ω: {base_frequency:.2f}")
            st.write(f"- Spread: {freq_spread:.2f}")
            st.write(f"- Wavelength: {2*np.pi/base_frequency:.3f}")

    # Export section
    with st.expander("💾 Export Data"):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Export Field Data (NPZ)"):
                np.savez(
                    'holographic_field.npz',
                    intensity=resonator.intensity,
                    grid_resolution=grid_resolution,
                    sphere_radius=sphere_radius,
                    num_sources=num_sources
                )
                st.success("Saved to holographic_field.npz")

        with col2:
            if st.button("Export Configuration (JSON)"):
                import json
                config = {
                    'num_sources': num_sources,
                    'helical_turns': helical_turns,
                    'grid_resolution': grid_resolution,
                    'sphere_radius': sphere_radius,
                    'base_frequency': base_frequency,
                    'freq_spread': freq_spread,
                    'num_vortices': num_vortices,
                    'vortex_charges': vortex_charges
                }
                with open('config.json', 'w') as f:
                    json.dump(config, f, indent=2)
                st.success("Saved to config.json")

    # Footer
    st.markdown("---")
    st.markdown("""
    **VHL Holographic Resonance Simulator** |
    Based on vibrational helical lattice physics |
    [GitHub](https://github.com/Zynerji/iVHL)
    """)


if __name__ == "__main__":
    main()
