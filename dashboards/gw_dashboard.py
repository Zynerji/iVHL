"""
LIGO-Inspired GW Lattice Visualization Dashboard

Streamlit interface for interactive exploration of GW lattice phenomena:
- Strain waveform plots
- Log-space fractal views
- Lattice overlays on holographic resonance
- Persistence test animations
- Comprehensive metrics dashboard

Integrates all GW analysis modules for unified visualization and exploration.

Author: iVHL Framework (LIGO Integration)
Date: 2025-12-15
"""

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import json

# Import GW modules
from gw_lattice_mode import GWLatticeProbe, GWLatticeConfig, GWDiscoveryMode
from gw_fractal_analysis import FractalHarmonicAnalyzer, FractalAnalysisConfig
from gw_rl_discovery import GWDiscoveryCampaign, GWDiscoveryEnvironment

# Import existing visualization modules if available
try:
    from vhl_resonance_streamlit import render_holographic_resonance
    RESONANCE_VIZ_AVAILABLE = True
except ImportError:
    RESONANCE_VIZ_AVAILABLE = False


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="iVHL GW Lattice Analysis",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Utility Functions
# ============================================================================

def plot_strain_waveform(
    time: np.ndarray,
    strain: np.ndarray,
    title: str = "GW Strain Waveform"
) -> go.Figure:
    """
    Plot strain time-series

    Args:
        time: (T,) time array (seconds)
        strain: (T,) strain values
        title: Plot title

    Returns:
        fig: Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time,
        y=strain,
        mode='lines',
        name='Strain h(t)',
        line=dict(color='cyan', width=1)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Strain h(t)",
        template='plotly_dark',
        height=400
    )

    return fig


def plot_power_spectrum(
    frequencies: np.ndarray,
    power: np.ndarray,
    peaks: Optional[List[Tuple[float, float]]] = None,
    constant_residues: Optional[Dict] = None
) -> go.Figure:
    """
    Plot power spectral density with peaks and constant residues

    Args:
        frequencies: (N,) frequency array (Hz)
        power: (N,) power values
        peaks: List of (frequency, power) peak tuples
        constant_residues: Dict of constant_name → matches

    Returns:
        fig: Plotly figure
    """
    fig = go.Figure()

    # Power spectrum
    fig.add_trace(go.Scatter(
        x=frequencies,
        y=power,
        mode='lines',
        name='Power Spectrum',
        line=dict(color='lightblue', width=1)
    ))

    # Peaks
    if peaks is not None and len(peaks) > 0:
        peak_freqs = [p[0] for p in peaks]
        peak_powers = [p[1] for p in peaks]

        fig.add_trace(go.Scatter(
            x=peak_freqs,
            y=peak_powers,
            mode='markers',
            name='Detected Peaks',
            marker=dict(color='red', size=8, symbol='x')
        ))

    # Constant residues
    if constant_residues is not None:
        for const_name, matches in constant_residues.items():
            if len(matches) > 0:
                freqs = [m['frequency'] for m in matches]
                # Get powers from main spectrum
                powers = [np.interp(f, frequencies, power) for f in freqs]

                fig.add_trace(go.Scatter(
                    x=freqs,
                    y=powers,
                    mode='markers',
                    name=f'Constant: {const_name}',
                    marker=dict(size=10, symbol='star')
                ))

    fig.update_layout(
        title="Power Spectral Density",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power",
        xaxis_type='log',
        yaxis_type='log',
        template='plotly_dark',
        height=500
    )

    return fig


def plot_fractal_dimension(
    box_sizes: List[int],
    counts: List[int],
    fractal_dim: float,
    r_squared: float
) -> go.Figure:
    """
    Plot box-counting fractal dimension analysis

    Args:
        box_sizes: List of box sizes
        counts: Number of occupied boxes at each size
        fractal_dim: Computed fractal dimension
        r_squared: Fit quality

    Returns:
        fig: Plotly figure
    """
    # Log-log plot
    log_eps = np.log(box_sizes)
    log_N = np.log(counts)

    # Fit line
    coeffs = np.polyfit(log_eps, log_N, deg=1)
    fit_line = coeffs[0] * log_eps + coeffs[1]

    fig = go.Figure()

    # Data points
    fig.add_trace(go.Scatter(
        x=box_sizes,
        y=counts,
        mode='markers',
        name='Box Count Data',
        marker=dict(color='cyan', size=8)
    ))

    # Fit line
    fig.add_trace(go.Scatter(
        x=box_sizes,
        y=np.exp(fit_line),
        mode='lines',
        name=f'Fit: D = {fractal_dim:.3f} (R² = {r_squared:.3f})',
        line=dict(color='red', width=2, dash='dash')
    ))

    fig.update_layout(
        title=f"Fractal Dimension Analysis (D = {fractal_dim:.3f})",
        xaxis_title="Box Size ε",
        yaxis_title="Number of Boxes N(ε)",
        xaxis_type='log',
        yaxis_type='log',
        template='plotly_dark',
        height=450
    )

    return fig


def plot_lattice_3d(
    lattice_positions: np.ndarray,
    title: str = "Helical Lattice Structure",
    color_by: Optional[np.ndarray] = None,
    colorbar_title: Optional[str] = None
) -> go.Figure:
    """
    Plot 3D lattice positions

    Args:
        lattice_positions: (N, 3) positions
        title: Plot title
        color_by: (N,) values for color mapping
        colorbar_title: Title for colorbar

    Returns:
        fig: Plotly 3D scatter figure
    """
    x, y, z = lattice_positions[:, 0], lattice_positions[:, 1], lattice_positions[:, 2]

    if color_by is None:
        color_by = np.arange(len(x))
        colorbar_title = "Index"

    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=color_by,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=colorbar_title)
        )
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='cube'
        ),
        template='plotly_dark',
        height=600
    )

    return fig


def plot_persistence_test(
    scramble_strengths: List[float],
    similarities: List[float],
    test_type: str = "Phase Scrambling"
) -> go.Figure:
    """
    Plot lattice persistence under scrambling

    Args:
        scramble_strengths: List of scramble parameters
        similarities: Lattice similarity scores
        test_type: Type of scrambling test

    Returns:
        fig: Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=scramble_strengths,
        y=similarities,
        mode='lines+markers',
        name='Lattice Similarity',
        line=dict(color='lime', width=2),
        marker=dict(size=8)
    ))

    # Threshold line
    fig.add_hline(
        y=0.8,
        line_dash="dash",
        line_color="red",
        annotation_text="Persistence Threshold"
    )

    fig.update_layout(
        title=f"{test_type} Persistence Test",
        xaxis_title="Scramble Strength",
        yaxis_title="Lattice Similarity",
        template='plotly_dark',
        height=400
    )

    return fig


def plot_log_space_histogram(
    bin_centers: np.ndarray,
    counts: np.ndarray,
    power_law_exponent: float,
    r_squared: float
) -> go.Figure:
    """
    Plot log-space histogram with power-law fit

    Args:
        bin_centers: Log bin centers
        counts: Histogram counts
        power_law_exponent: Fitted exponent α
        r_squared: Fit quality

    Returns:
        fig: Plotly figure
    """
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts,
        name='Field Distribution',
        marker=dict(color='skyblue')
    ))

    # Power-law fit
    if not np.isnan(power_law_exponent) and r_squared > 0.5:
        # Reconstruct fit
        log_counts_fit = -power_law_exponent * bin_centers + counts.max()
        counts_fit = np.exp(log_counts_fit)

        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=counts_fit,
            mode='lines',
            name=f'Power Law: α = {power_law_exponent:.2f} (R² = {r_squared:.2f})',
            line=dict(color='red', width=2, dash='dash')
        ))

    fig.update_layout(
        title="Log-Space Field Distribution",
        xaxis_title="log₁₀(Field Intensity)",
        yaxis_title="Count",
        yaxis_type='log',
        template='plotly_dark',
        height=400
    )

    return fig


# ============================================================================
# Main Dashboard
# ============================================================================

def main():
    """Main Streamlit dashboard"""

    st.title("🌌 LIGO-Inspired GW Lattice Analysis")
    st.markdown("**Exploring vibrational lattice structures, fractal harmonics, and attractor dynamics**")

    st.markdown("---")

    # ========================================================================
    # Sidebar Configuration
    # ========================================================================

    st.sidebar.header("⚙️ Configuration")

    # GW parameters
    st.sidebar.subheader("GW Perturbation")
    perturbation_type = st.sidebar.selectbox(
        "Perturbation Type",
        ['inspiral', 'ringdown', 'stochastic', 'constant_lattice'],
        index=3
    )

    gw_amplitude = st.sidebar.number_input(
        "GW Amplitude",
        min_value=1e-23,
        max_value=1e-19,
        value=1e-21,
        format="%.2e"
    )

    gw_frequency = st.sidebar.number_input(
        "GW Frequency (Hz)",
        min_value=10.0,
        max_value=1000.0,
        value=100.0
    )

    # Lattice parameters
    st.sidebar.subheader("Lattice Structure")
    num_lattice_nodes = st.sidebar.slider(
        "Lattice Nodes",
        min_value=50,
        max_value=1000,
        value=200,
        step=50
    )

    helical_turns = st.sidebar.slider(
        "Helical Turns",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5
    )

    sphere_radius = st.sidebar.slider(
        "Sphere Radius",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1
    )

    # Simulation parameters
    st.sidebar.subheader("Simulation")
    duration = st.sidebar.slider(
        "Duration (s)",
        min_value=0.5,
        max_value=4.0,
        value=2.0,
        step=0.5
    )

    run_scrambling = st.sidebar.checkbox(
        "Run Scrambling Tests",
        value=True
    )

    run_tolerance = st.sidebar.checkbox(
        "Run Tolerance Tests",
        value=False
    )

    # ========================================================================
    # Run Simulation
    # ========================================================================

    if st.sidebar.button("🚀 Run GW Simulation", type="primary"):
        # Create config
        gw_config = GWLatticeConfig(
            perturbation_type=perturbation_type,
            gw_amplitude=gw_amplitude,
            gw_frequency=gw_frequency,
            num_lattice_nodes=num_lattice_nodes,
            helical_turns=helical_turns,
            sphere_radius=sphere_radius,
            duration=duration
        )

        fractal_config = FractalAnalysisConfig()

        # Run simulation
        with st.spinner("Running GW lattice simulation..."):
            probe = GWLatticeProbe(gw_config)
            results = probe.run_simulation(
                with_scrambling=run_scrambling,
                with_tolerance_test=run_tolerance
            )

        st.success("Simulation complete!")

        # Store in session state
        st.session_state['results'] = results
        st.session_state['gw_config'] = gw_config
        st.session_state['fractal_config'] = fractal_config

    # ========================================================================
    # Display Results
    # ========================================================================

    if 'results' in st.session_state:
        results = st.session_state['results']
        gw_config = st.session_state['gw_config']
        fractal_config = st.session_state['fractal_config']

        # Tabs for different views
        tabs = st.tabs([
            "📈 Strain Waveforms",
            "🎵 Harmonic Analysis",
            "🔷 Fractal Dimension",
            "🌀 Lattice Visualization",
            "🔬 Persistence Tests",
            "📊 Metrics Dashboard"
        ])

        # ====================================================================
        # Tab 1: Strain Waveforms
        # ====================================================================

        with tabs[0]:
            st.header("Strain Waveforms")

            # Input strain
            strain_input = results['strain_input']
            time = np.arange(len(strain_input)) / gw_config.sampling_rate

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Input Strain (GW Perturbation)")
                fig_input = plot_strain_waveform(
                    time,
                    strain_input,
                    title=f"Input Strain: {gw_config.perturbation_type}"
                )
                st.plotly_chart(fig_input, use_container_width=True)

            with col2:
                st.subheader("Extracted Strain (From Resonant Field)")
                strain_extracted = np.array(results['strain_extracted'])
                time_extracted = np.arange(len(strain_extracted)) / gw_config.sampling_rate * 10  # Subsampled

                fig_extracted = plot_strain_waveform(
                    time_extracted,
                    strain_extracted,
                    title="Extracted Strain from Field"
                )
                st.plotly_chart(fig_extracted, use_container_width=True)

            # Statistics
            st.subheader("Waveform Statistics")
            stat_cols = st.columns(4)

            with stat_cols[0]:
                st.metric("Input Max Strain", f"{np.abs(strain_input).max():.2e}")

            with stat_cols[1]:
                st.metric("Extracted Max Strain", f"{np.abs(strain_extracted).max():.2e}")

            with stat_cols[2]:
                st.metric("Duration", f"{duration:.1f} s")

            with stat_cols[3]:
                st.metric("Sampling Rate", f"{gw_config.sampling_rate:.0f} Hz")

        # ====================================================================
        # Tab 2: Harmonic Analysis
        # ====================================================================

        with tabs[1]:
            st.header("Harmonic Analysis")

            # Run analysis
            analyzer = FractalHarmonicAnalyzer(fractal_config)
            strain_tensor = torch.from_numpy(strain_extracted).float()

            frequencies, power = analyzer.harmonic_detector.compute_power_spectrum(
                strain_tensor,
                gw_config.sampling_rate / 10  # Account for subsampling
            )

            peaks = analyzer.harmonic_detector.detect_peaks(frequencies, power)
            harmonic_series = analyzer.harmonic_detector.detect_harmonic_series(peaks)
            constant_residues = analyzer.harmonic_detector.detect_constant_residues(peaks)

            # Power spectrum plot
            st.subheader("Power Spectral Density")
            fig_psd = plot_power_spectrum(frequencies, power, peaks, constant_residues)
            st.plotly_chart(fig_psd, use_container_width=True)

            # Harmonic series
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Harmonic Series")
                if harmonic_series['fundamental_frequency'] is not None:
                    st.metric(
                        "Fundamental Frequency",
                        f"{harmonic_series['fundamental_frequency']:.2f} Hz"
                    )
                    st.metric(
                        "Harmonics Detected",
                        f"{len(harmonic_series['harmonics_detected'])}/10"
                    )
                    st.metric(
                        "Harmonic Ratio",
                        f"{harmonic_series['harmonic_ratio']:.2%}"
                    )
                else:
                    st.warning("No harmonic series detected")

            with col2:
                st.subheader("Constant Residues")
                if constant_residues:
                    for const_name, matches in constant_residues.items():
                        st.write(f"**{const_name}**: {len(matches)} matches")
                        for match in matches[:3]:  # Show top 3
                            st.write(f"  - f = {match['frequency']:.2f} Hz (dev: {match['deviation']:.2%})")
                else:
                    st.warning("No constant residues detected")

        # ====================================================================
        # Tab 3: Fractal Dimension
        # ====================================================================

        with tabs[2]:
            st.header("Fractal Dimension Analysis")

            # Create synthetic 3D field from history (simplified)
            st.info("Reconstructing 3D field from probe data...")

            field_history_tensors = [torch.from_numpy(f) for f in results['field_history']]
            mean_field = torch.stack(field_history_tensors).mean(dim=0)

            # Expand to 3D grid (simplified)
            grid_size = 32
            field_3d = mean_field.view(1, -1, 1).expand(grid_size, -1, grid_size).reshape(
                grid_size, grid_size, -1
            )[:, :, :grid_size]

            # Analyze
            fractal_results = analyzer.fractal_analyzer.analyze_field(field_3d)

            # Plot for each threshold
            for key, result in list(fractal_results.items())[:2]:  # Show top 2
                st.subheader(f"Threshold: {result['threshold_value']:.3e}")

                box_sizes, counts = analyzer.fractal_analyzer.box_count(
                    field_3d,
                    threshold=result['threshold_value']
                )

                fig_fractal = plot_fractal_dimension(
                    box_sizes,
                    counts,
                    result['fractal_dimension'],
                    result['r_squared']
                )

                st.plotly_chart(fig_fractal, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fractal Dimension", f"{result['fractal_dimension']:.3f}")
                with col2:
                    st.metric("R² Score", f"{result['r_squared']:.3f}")
                with col3:
                    quality = "Excellent" if result['r_squared'] > 0.95 else ("Good" if result['r_squared'] > 0.85 else "Fair")
                    st.metric("Fit Quality", quality)

        # ====================================================================
        # Tab 4: Lattice Visualization
        # ====================================================================

        with tabs[3]:
            st.header("Lattice Visualization")

            # Select timestep
            num_timesteps = len(results['lattice_history'])
            timestep = st.slider(
                "Timestep",
                min_value=0,
                max_value=num_timesteps - 1,
                value=0
            )

            lattice_positions = results['lattice_history'][timestep].numpy()

            # Color by field intensity
            field_values = results['field_history'][timestep].numpy()

            fig_lattice = plot_lattice_3d(
                lattice_positions,
                title=f"Helical Lattice (t = {timestep / 10:.2f} s)",
                color_by=np.linalg.norm(lattice_positions, axis=1),
                colorbar_title="Radial Distance"
            )

            st.plotly_chart(fig_lattice, use_container_width=True)

            # Statistics
            st.subheader("Lattice Statistics")
            stat_cols = st.columns(4)

            with stat_cols[0]:
                st.metric("Nodes", num_lattice_nodes)

            with stat_cols[1]:
                mean_radius = np.linalg.norm(lattice_positions, axis=1).mean()
                st.metric("Mean Radius", f"{mean_radius:.3f}")

            with stat_cols[2]:
                std_radius = np.linalg.norm(lattice_positions, axis=1).std()
                st.metric("Radius Std", f"{std_radius:.3f}")

            with stat_cols[3]:
                mean_field = field_values.mean()
                st.metric("Mean Field", f"{mean_field:.3e}")

        # ====================================================================
        # Tab 5: Persistence Tests
        # ====================================================================

        with tabs[4]:
            st.header("Persistence Tests")

            if run_scrambling and 'scrambling_tests' in results:
                scrambling = results['scrambling_tests']

                col1, col2 = st.columns(2)

                # Phase scrambling
                with col1:
                    if 'phase' in scrambling:
                        phase_data = scrambling['phase']
                        fig_phase = plot_persistence_test(
                            phase_data['strengths'],
                            phase_data['similarities'],
                            test_type="Phase Scrambling"
                        )
                        st.plotly_chart(fig_phase, use_container_width=True)

                # Null scrambling
                with col2:
                    if 'null' in scrambling:
                        null_data = scrambling['null']
                        fig_null = plot_persistence_test(
                            null_data['fractions'],
                            null_data['similarities'],
                            test_type="Null Scrambling"
                        )
                        st.plotly_chart(fig_null, use_container_width=True)

                # Interpretation
                st.subheader("Interpretation")
                st.write("""
                **Persistence Score**: Measures how well the lattice structure survives scrambling.
                - **High persistence** (>0.8): Robust lattice structure, likely physical significance
                - **Medium persistence** (0.5-0.8): Moderate structure, sensitive to perturbations
                - **Low persistence** (<0.5): Weak structure, may be noise-dominated
                """)

            else:
                st.info("Enable 'Run Scrambling Tests' in sidebar to see persistence analysis")

            # Memory persistence
            if 'persistence_metrics' in results and 'memory' in results['persistence_metrics']:
                st.subheader("Memory Field Persistence")
                memory = results['persistence_metrics']['memory']

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Decay Time τ", f"{memory['decay_time']:.3f} s")

                with col2:
                    st.metric("Quality Factor Q", f"{memory['quality_factor']:.2f}")

                with col3:
                    st.metric("Residual Amplitude", f"{memory['residual_amplitude']:.3e}")

                st.write("""
                **Memory decay time** (τ) indicates how long the field "remembers" the perturbation.
                Longer τ suggests more persistent spacetime structure.
                """)

        # ====================================================================
        # Tab 6: Metrics Dashboard
        # ====================================================================

        with tabs[5]:
            st.header("Metrics Dashboard")

            # Run full analysis
            st.subheader("Comprehensive Analysis")

            with st.spinner("Running full fractal/harmonic analysis..."):
                full_analysis = analyzer.analyze_full(
                    field_3d,
                    strain_tensor,
                    gw_config.sampling_rate / 10,
                    lattice_history=[torch.from_numpy(l) for l in results['lattice_history']]
                )

            # Display metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("🔷 Fractal Metrics")
                fractal_dims = [
                    r['fractal_dimension']
                    for r in full_analysis['fractal_dimensions'].values()
                ]
                st.metric("Mean Fractal Dim", f"{np.mean(fractal_dims):.3f}")

                st.metric(
                    "Power-Law Exponent",
                    f"{full_analysis['log_space']['power_law_exponent']:.2f}"
                )

            with col2:
                st.subheader("🎵 Harmonic Metrics")
                st.metric(
                    "Fundamental Freq",
                    f"{full_analysis['harmonic_series']['fundamental_frequency']:.1f} Hz"
                    if full_analysis['harmonic_series']['fundamental_frequency'] else "N/A"
                )

                st.metric(
                    "Constant Residues",
                    len(full_analysis['constant_residues'])
                )

            with col3:
                st.subheader("🌀 Lattice Metrics")
                st.metric(
                    "Nodal Clusters",
                    full_analysis['nodal_clusters']['num_clusters']
                )

                if 'persistence_score' in full_analysis:
                    st.metric(
                        "Persistence Score",
                        f"{full_analysis['persistence_score']:.2%}"
                    )

            # Export results
            st.subheader("Export Results")

            if st.button("💾 Export Analysis to JSON"):
                export_data = {
                    'config': {
                        'perturbation_type': gw_config.perturbation_type,
                        'gw_amplitude': gw_config.gw_amplitude,
                        'gw_frequency': gw_config.gw_frequency,
                        'num_lattice_nodes': gw_config.num_lattice_nodes
                    },
                    'analysis': {
                        k: v for k, v in full_analysis.items()
                        if k not in ['nodal_clusters']  # Skip non-serializable
                    }
                }

                json_str = json.dumps(export_data, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="gw_lattice_analysis.json",
                    mime="application/json"
                )

    else:
        # Welcome message
        st.info("👈 Configure parameters in the sidebar and click **Run GW Simulation** to begin!")

        st.markdown("""
        ## About This Dashboard

        This interactive dashboard explores **LIGO-inspired gravitational wave lattice phenomena** within the iVHL framework:

        ### Features:
        - **GW Waveform Generation**: Inspiral, ringdown, stochastic, and constant-lattice perturbations
        - **Fractal Analysis**: Box-counting dimension, log-space structure, power-law scaling
        - **Harmonic Detection**: FFT peak finding, harmonic series, mathematical constant residues
        - **Lattice Visualization**: 3D helical boundary structure with perturbation evolution
        - **Persistence Tests**: Phase/null scrambling, memory field decay analysis
        - **Metrics Dashboard**: Comprehensive analysis with exportable results

        ### Conceptual Foundation:
        - **Constant Lattice**: LIGO analysis suggests structured residues at constant-related frequencies
        - **Fractal Harmonics**: Self-similar patterns in log-space indicating scale-invariant structure
        - **Attractor Dynamics**: Stable lattice configurations emerge as basins in parameter space
        - **Memory Field**: GW background persists as long-term spacetime memory

        ### Usage:
        1. Select perturbation type (try 'constant_lattice' for LIGO-inspired patterns)
        2. Adjust lattice parameters (nodes, helical turns, radius)
        3. Configure simulation duration and tests
        4. Click **Run GW Simulation**
        5. Explore results across different tabs
        6. Export analysis for further study

        **LIGO Integration**: Bridges cutting-edge GW observations with iVHL's vibrational lattice unification framework.
        """)


if __name__ == "__main__":
    main()
