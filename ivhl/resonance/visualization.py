"""
VHL Holographic Resonance Visualization

Interactive 3D visualization of holographic resonance patterns using PyVista.

Features:
- Real-time volumetric rendering of wave interference
- Multi-level isosurface extraction (folded topologies)
- Animated particle flow through resonant structures
- Interactive camera controls and parameter adjustment
- Time evolution of standing wave patterns
- Multi-vortex dynamics visualization

Usage:
    python vhl_resonance_viz.py [--mode static|animated|interactive]
    python vhl_resonance_viz.py --num-sources 1000 --vortices 2

Author: Zynerji
Date: 2025-12-15
"""

import numpy as np
import pyvista as pv
from vhl_holographic_resonance import (
    HolographicResonator,
    ParticleAdvector
)
import argparse
from typing import Optional
from tqdm import tqdm


class ResonanceVisualizer:
    """
    Interactive PyVista-based visualization of holographic resonance.
    """

    def __init__(self, resonator: HolographicResonator,
                 window_size: tuple = (1200, 900)):
        """
        Initialize visualizer.

        Args:
            resonator: Configured HolographicResonator instance
            window_size: Viewer window dimensions
        """
        self.resonator = resonator
        self.window_size = window_size
        self.plotter = None
        self.particles = None

    def setup_plotter(self, off_screen: bool = False):
        """Initialize PyVista plotter with standard setup."""
        self.plotter = pv.Plotter(window_size=self.window_size, off_screen=off_screen)
        self.plotter.set_background('black')
        self.plotter.add_axes()

    def add_boundary_sphere(self, color='cyan', opacity=0.1, resolution=50):
        """Add wireframe sphere showing holographic boundary."""
        sphere = pv.Sphere(
            radius=self.resonator.sphere_radius,
            theta_resolution=resolution,
            phi_resolution=resolution
        )
        self.plotter.add_mesh(
            sphere,
            color=color,
            style='wireframe',
            opacity=opacity,
            line_width=1,
            name='boundary_sphere'
        )

    def add_lattice_points(self, color='yellow', point_size=5):
        """Add wave source lattice points on boundary."""
        lattice_points = self.resonator.lattice.get_points()
        point_cloud = pv.PolyData(lattice_points)
        self.plotter.add_mesh(
            point_cloud,
            color=color,
            point_size=point_size,
            render_points_as_spheres=True,
            name='lattice_points'
        )

    def add_volume_rendering(self, opacity_function='sigmoid', cmap='hot'):
        """
        Add volumetric rendering of intensity field.

        Args:
            opacity_function: 'sigmoid', 'linear', or custom
            cmap: Colormap name
        """
        # Get intensity as 3D array
        intensity_3d = self.resonator.get_intensity_3d()

        # Create PyVista UniformGrid
        n = self.resonator.grid_resolution
        grid = pv.UniformGrid(
            dimensions=(n, n, n),
            spacing=(
                2 * self.resonator.sphere_radius / n,
                2 * self.resonator.sphere_radius / n,
                2 * self.resonator.sphere_radius / n
            ),
            origin=(
                -self.resonator.sphere_radius,
                -self.resonator.sphere_radius,
                -self.resonator.sphere_radius
            )
        )

        # Add intensity as scalar field
        grid['intensity'] = intensity_3d.ravel(order='F')

        # Opacity mapping
        if opacity_function == 'sigmoid':
            # Sigmoid opacity: emphasizes mid-range values
            max_int = intensity_3d.max()
            opacity = np.array([[0, 0], [0.1 * max_int, 0], [0.5 * max_int, 0.5],
                                [0.9 * max_int, 0.8], [max_int, 1.0]])
        elif opacity_function == 'linear':
            max_int = intensity_3d.max()
            opacity = np.array([[0, 0], [max_int, 1.0]])
        else:
            opacity = opacity_function

        # Add volume
        self.plotter.add_volume(
            grid,
            scalars='intensity',
            cmap=cmap,
            opacity=opacity,
            name='volume'
        )

    def add_isosurfaces(self, levels: list = [0.3, 0.6, 0.9],
                       colors: Optional[list] = None,
                       opacity: float = 0.6):
        """
        Add multiple isosurface levels for folded topology visualization.

        Args:
            levels: List of intensity thresholds (0-1 fraction of max)
            colors: List of colors for each level
            opacity: Surface opacity
        """
        intensity_3d = self.resonator.get_intensity_3d()
        max_intensity = intensity_3d.max()

        # Default colors
        if colors is None:
            colors = ['cyan', 'magenta', 'yellow', 'lime', 'orange']

        isosurfaces_data = self.resonator.extract_isosurfaces(
            [level * max_intensity for level in levels]
        )

        for i, (level, data) in enumerate(isosurfaces_data.items()):
            # Create mesh from marching cubes output
            vertices = data['vertices']
            faces = data['faces']

            # Convert faces to PyVista format
            faces_pv = np.hstack([[3] + list(face) for face in faces])

            mesh = pv.PolyData(vertices, faces_pv)
            mesh = mesh.compute_normals()

            color = colors[i % len(colors)]
            self.plotter.add_mesh(
                mesh,
                color=color,
                opacity=opacity,
                smooth_shading=True,
                name=f'isosurface_{i}'
            )

    def add_particles(self, particle_advector: ParticleAdvector,
                     color='white', point_size=3):
        """
        Add particle point cloud.

        Args:
            particle_advector: ParticleAdvector with particle positions
            color: Particle color
            point_size: Particle size
        """
        self.particles = particle_advector
        point_cloud = pv.PolyData(self.particles.positions)

        self.plotter.add_mesh(
            point_cloud,
            color=color,
            point_size=point_size,
            render_points_as_spheres=True,
            name='particles'
        )

    def update_particles(self, dt: float = 0.01, mode: str = 'gradient'):
        """Update particle positions (for animation)."""
        if self.particles is not None:
            self.particles.advect(self.resonator, dt=dt, mode=mode)

            # Update mesh
            point_cloud = pv.PolyData(self.particles.positions)
            self.plotter.add_mesh(
                point_cloud,
                color='white',
                point_size=3,
                render_points_as_spheres=True,
                name='particles'
            )

    def render_static(self, show_volume: bool = True, show_isosurfaces: bool = True,
                     show_particles: bool = False):
        """
        Render static visualization.

        Args:
            show_volume: Show volumetric rendering
            show_isosurfaces: Show isosurfaces
            show_particles: Show particles
        """
        self.setup_plotter()

        # Add components
        self.add_boundary_sphere()
        self.add_lattice_points()

        if show_volume:
            self.add_volume_rendering(opacity_function='sigmoid', cmap='hot')

        if show_isosurfaces:
            self.add_isosurfaces(levels=[0.3, 0.6, 0.9])

        if show_particles:
            particles = ParticleAdvector(num_particles=2000,
                                        sphere_radius=self.resonator.sphere_radius * 0.8)
            self.add_particles(particles)

        # Camera setup
        self.plotter.camera_position = 'isometric'
        self.plotter.camera.zoom(1.2)

        # Show
        self.plotter.show()

    def render_animated(self, num_frames: int = 100, dt: float = 0.05,
                       fps: int = 20, save_path: Optional[str] = None):
        """
        Render animated time evolution.

        Args:
            num_frames: Number of animation frames
            dt: Time step between frames
            fps: Frames per second
            save_path: If provided, save animation to this path
        """
        self.setup_plotter(off_screen=save_path is not None)

        # Initialize particles
        particles = ParticleAdvector(num_particles=2000,
                                    sphere_radius=self.resonator.sphere_radius * 0.8)

        # Setup scene
        self.add_boundary_sphere()
        self.add_lattice_points()

        # Camera
        self.plotter.camera_position = 'isometric'
        self.plotter.camera.zoom(1.2)

        if save_path:
            self.plotter.open_gif(save_path, fps=fps)

        # Animation loop
        for frame in tqdm(range(num_frames), desc="Rendering animation"):
            time = frame * dt

            # Recompute field
            self.resonator.compute_field(time=time)
            self.resonator.compute_intensity()

            # Update isosurfaces
            self.plotter.clear_actors()  # Clear previous
            self.add_boundary_sphere()
            self.add_lattice_points()
            self.add_isosurfaces(levels=[0.4, 0.7])

            # Update particles
            particles.advect(self.resonator, dt=dt, mode='gradient')
            self.add_particles(particles)

            # Write frame
            if save_path:
                self.plotter.write_frame()
            else:
                self.plotter.render()

        if save_path:
            self.plotter.close()
        else:
            self.plotter.show()

    def render_interactive(self):
        """
        Render with interactive controls (requires PyVista GUI).

        Allows real-time parameter adjustment.
        """
        self.setup_plotter()

        # Add components
        self.add_boundary_sphere()
        self.add_lattice_points()
        self.add_volume_rendering()
        self.add_isosurfaces()

        # Add slider widgets for parameters (optional, requires PyVista >=0.40)
        # self.plotter.add_slider_widget(...)

        # Camera
        self.plotter.camera_position = 'isometric'

        # Show with interaction
        self.plotter.show()


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="VHL Holographic Resonance Visualization"
    )

    parser.add_argument('--mode', type=str, default='static',
                       choices=['static', 'animated', 'interactive'],
                       help='Visualization mode')
    parser.add_argument('--num-sources', type=int, default=500,
                       help='Number of wave sources on boundary')
    parser.add_argument('--helical-turns', type=int, default=5,
                       help='Number of helical turns')
    parser.add_argument('--grid-res', type=int, default=64,
                       help='Grid resolution for field computation')
    parser.add_argument('--vortices', type=int, default=1,
                       help='Number of vortex modes to add')
    parser.add_argument('--base-freq', type=float, default=1.0,
                       help='Base frequency for sources')
    parser.add_argument('--freq-spread', type=float, default=0.1,
                       help='Frequency spread')
    parser.add_argument('--save-gif', type=str, default=None,
                       help='Save animated GIF to path')
    parser.add_argument('--num-frames', type=int, default=100,
                       help='Number of animation frames')

    args = parser.parse_args()

    print("=" * 70)
    print("VHL HOLOGRAPHIC RESONANCE VISUALIZATION")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Sources: {args.num_sources}, Helical turns: {args.helical_turns}")
    print(f"Grid resolution: {args.grid_res}³ = {args.grid_res**3:,} voxels")
    print(f"Vortices: {args.vortices}")
    print()

    # Create resonator
    print("Initializing holographic resonator...")
    resonator = HolographicResonator(
        sphere_radius=1.0,
        grid_resolution=args.grid_res,
        num_sources=args.num_sources,
        helical_turns=args.helical_turns
    )

    # Initialize sources
    print("Setting up wave sources...")
    resonator.initialize_sources_from_lattice(
        base_frequency=args.base_freq,
        frequency_spread=args.freq_spread,
        polarity_phase=True
    )

    # Add vortices
    if args.vortices > 0:
        print(f"Adding {args.vortices} vortex mode(s)...")
        for i in range(args.vortices):
            # Distribute vortices
            angle = 2 * np.pi * i / args.vortices
            radius = 0.3
            center = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0.0
            ])
            topological_charge = (-1)**i  # Alternate charges
            resonator.add_vortex(center=center, topological_charge=topological_charge)

    # Compute initial field
    print("Computing holographic field...")
    resonator.compute_field(time=0.0)
    resonator.compute_intensity()

    max_int = resonator.intensity.max()
    mean_int = resonator.intensity.mean()
    print(f"✓ Field computed: max={max_int:.4f}, mean={mean_int:.4f}")
    print()

    # Create visualizer
    visualizer = ResonanceVisualizer(resonator)

    # Render based on mode
    if args.mode == 'static':
        print("Rendering static visualization...")
        visualizer.render_static(
            show_volume=True,
            show_isosurfaces=True,
            show_particles=False
        )

    elif args.mode == 'animated':
        print(f"Rendering {args.num_frames} frame animation...")
        visualizer.render_animated(
            num_frames=args.num_frames,
            dt=0.05,
            fps=20,
            save_path=args.save_gif
        )
        if args.save_gif:
            print(f"✓ Animation saved to {args.save_gif}")

    elif args.mode == 'interactive':
        print("Launching interactive viewer...")
        visualizer.render_interactive()

    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
