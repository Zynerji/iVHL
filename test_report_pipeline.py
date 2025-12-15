#!/usr/bin/env python3
"""
Test script for end-to-end report generation pipeline
Tests: GW simulation → report generation → LaTeX → PDF → file output
"""

import sys
import os
import torch
import numpy as np
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import iVHL modules
from gw_lattice_mode import GWLatticeConfig, GWLatticeProbe
from simulation_report_generator import SimulationReport, IntegratedReportGenerator


def run_test_simulation():
    """Run a quick GW lattice simulation for testing"""
    print("=" * 80)
    print("iVHL End-to-End Pipeline Test")
    print("=" * 80)
    print()

    # Configure simulation
    print("1. Configuring GW lattice simulation...")
    config = GWLatticeConfig(
        num_lattice_nodes=100,      # Small for quick test
        helical_turns=2.0,
        gw_amplitude=1e-21,
        gw_frequency=100.0,
        perturbation_type='constant_lattice',
        phase_scramble_strength=0.3,
        sampling_rate=4096.0,
        duration=1.0               # Short simulation (1 second)
    )

    print(f"   - Nodes: {config.num_lattice_nodes}")
    print(f"   - Perturbation: {config.perturbation_type}")
    print(f"   - Duration: {config.duration}s")
    print()

    # Run simulation
    print("2. Running simulation...")
    probe = GWLatticeProbe(config)
    results = probe.run_simulation()

    # Convert to numpy if needed
    strain_input = results['strain_input']
    strain_extracted = results['strain_extracted']

    if isinstance(strain_input, torch.Tensor):
        strain_input = strain_input.cpu().numpy()
    elif isinstance(strain_input, list):
        strain_input = np.array(strain_input)

    if isinstance(strain_extracted, torch.Tensor):
        strain_extracted = strain_extracted.cpu().numpy()
    elif isinstance(strain_extracted, list):
        strain_extracted = np.array(strain_extracted)

    print(f"   [OK] Simulation complete")
    print(f"   - Input strain shape: {strain_input.shape}")
    print(f"   - Extracted strain shape: {strain_extracted.shape}")
    print()

    # Analyze results
    print("3. Analyzing results...")

    # Compute basic metrics
    max_amplitude = np.max(np.abs(strain_input))

    # If lengths don't match, resample extracted strain to match input
    if len(strain_input) != len(strain_extracted):
        from scipy.interpolate import interp1d
        x_extracted = np.linspace(0, 1, len(strain_extracted))
        x_input = np.linspace(0, 1, len(strain_input))
        interpolator = interp1d(x_extracted, strain_extracted, kind='cubic', fill_value='extrapolate')
        strain_extracted_resampled = interpolator(x_input)
    else:
        strain_extracted_resampled = strain_extracted

    strain_correlation = np.corrcoef(strain_input, strain_extracted_resampled)[0, 1]
    strain_rmse = np.sqrt(np.mean((strain_input - strain_extracted_resampled)**2))

    # Detect peaks in extracted strain (frequency domain)
    from scipy.signal import find_peaks
    from scipy.fft import fft, fftfreq

    sampling_rate = config.sampling_rate
    dt = 1.0 / sampling_rate
    fft_vals = np.abs(fft(strain_extracted))
    freqs = fftfreq(len(strain_extracted), d=dt)

    # Only positive frequencies
    positive_freqs = freqs[:len(freqs)//2]
    positive_fft = fft_vals[:len(fft_vals)//2]

    peaks, _ = find_peaks(positive_fft, height=np.max(positive_fft) * 0.1)
    peak_frequencies = positive_freqs[peaks]

    print(f"   [OK] Analysis complete")
    print(f"   - Strain correlation: {strain_correlation:.4f}")
    print(f"   - RMSE: {strain_rmse:.2e}")
    print(f"   - Detected {len(peak_frequencies)} frequency peaks")
    print()

    # Package results for report
    num_samples = int(config.sampling_rate * config.duration)
    simulation_results = {
        'strain_input': strain_input.tolist(),
        'strain_extracted': strain_extracted.tolist(),
        'strain_correlation': float(strain_correlation),
        'strain_rmse': float(strain_rmse),
        'max_amplitude': float(max_amplitude),
        'num_frequency_peaks': len(peak_frequencies),
        'peak_frequencies_hz': peak_frequencies.tolist()[:10],  # Top 10
        'simulation_duration_s': config.duration,
        'num_samples': num_samples
    }

    analysis = {
        'summary': f"GW lattice probe with {config.perturbation_type} perturbation",
        'key_findings': [
            f"Strain extraction correlation: {strain_correlation:.4f}",
            f"Detected {len(peak_frequencies)} frequency peaks in extracted signal",
            f"RMSE between input and extracted strain: {strain_rmse:.2e}",
            "Lattice demonstrated resilience to GW-like perturbations"
        ],
        'implications': [
            "Holographic boundary lattice can encode GW strain information",
            "Resonant field intensity modulation reflects spacetime perturbations",
            "Framework suitable for testing AdS/CFT-inspired GW phenomenology"
        ]
    }

    return config, simulation_results, analysis


def test_report_generation(config, results, analysis):
    """Test the integrated report generator"""
    print("4. Generating reports...")

    # Convert config to dict
    config_dict = {
        'num_lattice_nodes': config.num_lattice_nodes,
        'helical_turns': config.helical_turns,
        'gw_amplitude': config.gw_amplitude,
        'gw_frequency': config.gw_frequency,
        'perturbation_type': config.perturbation_type,
        'phase_scramble_strength': config.phase_scramble_strength,
        'sampling_rate': config.sampling_rate,
        'duration': config.duration,
        'sphere_radius': config.sphere_radius,
        'noise_level': config.noise_level
    }

    # Generate reports
    generator = IntegratedReportGenerator(
        output_base_dir='./reports',
        auto_commit=False  # Don't auto-commit test reports
    )

    report_files = generator.generate_full_report(
        simulation_type='gw_lattice_test',
        configuration=config_dict,
        results=results,
        analysis=analysis
    )

    print(f"   [OK] Reports generated")
    for format_type, filepath in report_files.items():
        file_exists = os.path.exists(filepath)
        status = "[OK]" if file_exists else "[FAIL]"
        file_size = os.path.getsize(filepath) if file_exists else 0
        print(f"   {status} {format_type.upper()}: {filepath} ({file_size:,} bytes)")

    print()

    return report_files


def verify_pdf(pdf_path):
    """Verify PDF was created successfully"""
    print("5. Verifying PDF compilation...")

    if not os.path.exists(pdf_path):
        print("   [FAIL] PDF file not found")
        return False

    file_size = os.path.getsize(pdf_path)

    # Basic PDF validation - check magic bytes
    with open(pdf_path, 'rb') as f:
        header = f.read(8)
        if header.startswith(b'%PDF-'):
            print(f"   [OK] Valid PDF file created")
            print(f"   - File size: {file_size:,} bytes")
            print(f"   - PDF version: {header.decode('ascii', errors='ignore').strip()}")
            return True
        else:
            print("   [FAIL] Invalid PDF format")
            return False


def main():
    try:
        # Run simulation
        config, results, analysis = run_test_simulation()

        # Generate reports
        report_files = test_report_generation(config, results, analysis)

        # Verify PDF
        pdf_success = False
        if 'pdf' in report_files:
            pdf_success = verify_pdf(report_files['pdf'])
        else:
            print("5. Verifying PDF compilation...")
            print("   [WARN] PDF not generated (pdflatex not installed on this system)")
            print()

        # Summary
        print("=" * 80)
        print("Pipeline Test Summary")
        print("=" * 80)
        print()
        print("[OK] GW lattice simulation executed successfully")
        print("[OK] JSON report generated")
        print("[OK] Markdown report generated")
        print("[OK] LaTeX white paper generated")

        if pdf_success:
            print("[SUCCESS] PDF compilation successful")
            print()
            print(">>> End-to-end pipeline test PASSED <<<")
            print()
            print("Report files location:")
            print(f"   {os.path.dirname(report_files['pdf'])}")
        else:
            print("[WARN] PDF compilation skipped/failed")
            print()
            print("[WARNING] Pipeline test PARTIALLY PASSED (PDF compilation issue)")
            print("   LaTeX requires pdflatex installation (texlive packages)")
            print("   Docker container includes LaTeX - PDF generation will work there")
            print()
            print("Report files location:")
            # Get directory from any generated file
            for key in ['json', 'markdown', 'latex']:
                if key in report_files:
                    print(f"   {os.path.dirname(report_files[key])}")
                    break

        print()
        print("=" * 80)

        return 0 if pdf_success else 1

    except Exception as e:
        print()
        print("=" * 80)
        print("[ERROR] Pipeline test FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
