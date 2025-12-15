"""
Server-Side GPU Renderer
=========================

Renders 3D visualizations on server GPU and streams frames to browser.
Uses PyVista with GPU acceleration for high-quality rendering.
"""

import numpy as np
import pyvista as pv
import io
import base64
from PIL import Image
from typing import Dict, Tuple, Optional
import torch


class GPURenderer:
    """
    Server-side 3D renderer using GPU.

    Renders tensor hierarchy visualization and streams JPEG frames
    to browser via WebSocket.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        quality: int = 85,
        device: str = "cuda"
    ):
        self.resolution = resolution
        self.quality = quality
        self.device = device

        # Initialize PyVista plotter (off-screen for server)
        self.plotter = pv.Plotter(off_screen=True, window_size=resolution)
        self.plotter.set_background('black')

        # Camera setup
        self.camera_position = None
        self.rotation_angle = 0.0

        print(f"GPURenderer initialized:")
        print(f"  - Resolution: {resolution}")
        print(f"  - JPEG quality: {quality}")
        print(f"  - Device: {device}")

    def render_tensor_hierarchy(
        self,
        hierarchy,
        current_layer: int = 0,
        show_correlations: bool = True
    ) -> bytes:
        """
        Render tensor hierarchy as 3D visualization.

        Args:
            hierarchy: TensorHierarchy object
            current_layer: Which layer to highlight
            show_correlations: Draw correlation lines

        Returns:
            JPEG image as bytes
        """
        self.plotter.clear()

        # Get layer data
        num_layers = len(hierarchy.layers)

        for layer_idx in range(num_layers):
            tensor = hierarchy.to_numpy(layer_idx)
            bond_dim, h, w = tensor.shape

            # Position layers vertically
            z_offset = layer_idx * 5.0

            # Take first bond dimension slice for visualization
            data_slice = tensor[0, :, :]

            # Create grid points
            x = np.linspace(-w/2, w/2, w)
            y = np.linspace(-h/2, h/2, h)
            xx, yy = np.meshgrid(x, y)
            zz = np.ones_like(xx) * z_offset + data_slice * 0.5  # Height = data

            # Create mesh
            points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
            cloud = pv.PolyData(points)

            # Color by value
            cloud['values'] = data_slice.ravel()

            # Add to scene
            opacity = 1.0 if layer_idx == current_layer else 0.4
            self.plotter.add_mesh(
                cloud,
                scalars='values',
                cmap='viridis',
                opacity=opacity,
                point_size=10 if layer_idx == current_layer else 5,
                render_points_as_spheres=True
            )

            # Add layer label
            center = np.array([0, 0, z_offset])
            self.plotter.add_text(
                f"Layer {layer_idx}",
                position='upper_left' if layer_idx == 0 else None,
                font_size=10,
                color='white'
            )

        # Correlation lines (if enabled)
        if show_correlations and num_layers > 1:
            for i in range(num_layers - 1):
                # Draw line between layer centers
                p1 = np.array([0, 0, i * 5.0])
                p2 = np.array([0, 0, (i+1) * 5.0])

                # Correlation strength determines opacity
                corr = hierarchy.compute_layer_correlation(i, i+1)
                line = pv.Line(p1, p2)
                self.plotter.add_mesh(
                    line,
                    color='cyan',
                    opacity=abs(corr),
                    line_width=3
                )

        # Auto-rotate camera
        self.rotation_angle += 0.5
        radius = 30
        x_cam = radius * np.cos(np.radians(self.rotation_angle))
        y_cam = radius * np.sin(np.radians(self.rotation_angle))
        z_cam = num_layers * 2.5

        self.plotter.camera_position = [
            (x_cam, y_cam, z_cam),  # Camera position
            (0, 0, num_layers * 2.5),  # Focal point
            (0, 0, 1)  # View up
        ]

        # Render to image
        img_array = self.plotter.screenshot(return_img=True)

        # Convert to JPEG bytes
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=self.quality)
        jpeg_bytes = buffer.getvalue()

        return jpeg_bytes

    def render_to_base64(self, hierarchy, **kwargs) -> str:
        """
        Render and return as base64 string for embedding in HTML.
        """
        jpeg_bytes = self.render_tensor_hierarchy(hierarchy, **kwargs)
        b64_string = base64.b64encode(jpeg_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_string}"

    def create_animation_frame(
        self,
        hierarchy,
        step: int,
        total_steps: int
    ) -> bytes:
        """
        Create a single animation frame.

        Args:
            hierarchy: Current state
            step: Current timestep
            total_steps: Total simulation steps

        Returns:
            JPEG frame
        """
        # Determine which layer to highlight based on step
        current_layer = (step * len(hierarchy.layers)) // total_steps

        # Render
        return self.render_tensor_hierarchy(
            hierarchy,
            current_layer=current_layer,
            show_correlations=True
        )

    def cleanup(self):
        """Close plotter and free GPU memory."""
        self.plotter.close()


if __name__ == "__main__":
    # Test renderer
    print("Testing GPU renderer...")

    # Need to import hierarchy for test
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    from ivhl.hierarchical import TensorHierarchy, HierarchyConfig

    config = HierarchyConfig(num_layers=3, base_dimension=16, bond_dimension=4)
    hierarchy = TensorHierarchy(config)

    renderer = GPURenderer(resolution=(800, 600))

    print("Rendering test frame...")
    jpeg_bytes = renderer.render_tensor_hierarchy(hierarchy)

    print(f"Rendered {len(jpeg_bytes)} bytes")
    print("Test complete!")

    renderer.cleanup()
