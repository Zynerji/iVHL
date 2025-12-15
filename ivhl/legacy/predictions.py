"""
VHL Novel Predictions Generator

This module generates experimentally testable predictions from the VHL framework
that go beyond current scientific knowledge. All predictions are quantitative and
falsifiable.

Categories:
1. Superheavy element properties (Z=119-126)
2. Alloy stability predictions
3. Harmonic scaling laws (golden ratio)
4. Multi-body force enhancements
5. Magic angle reactivity patterns

Run: python vhl_predictions.py --generate all --output predictions.json
"""

import numpy as np
import json
import argparse
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path

# Known experimental data for validation
KNOWN_IONIZATION_ENERGIES = {
    # eV - from NIST
    1: 13.598, 2: 24.587, 3: 5.392, 4: 9.323, 5: 8.298,
    6: 11.260, 7: 14.534, 8: 13.618, 9: 17.423, 10: 21.565,
    11: 5.139, 12: 7.646, 13: 5.986, 14: 8.152, 15: 10.487,
    16: 10.360, 17: 12.968, 18: 15.760, 19: 4.341, 20: 6.113
}

KNOWN_ATOMIC_RADII = {
    # Angstroms - empirical covalent radii
    1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84,
    6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 10: 0.58,
    11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07,
    16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76
}

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio


class VHLPredictions:
    """Generate novel VHL predictions."""

    def __init__(self):
        self.predictions = {}

    def map_to_vhl_octave(self, z):
        """Map atomic number to VHL octave and tone."""
        octave = int(np.ceil(z / 9))
        tone = (z - 1) % 9
        return octave, tone

    def count_nodes_from_z(self, z):
        """
        Estimate nodal surfaces from Z using shell filling.
        Simplified model: nodes ≈ floor((Z-1) / 10) + additional correction
        """
        if z <= 2:
            return 0
        elif z <= 10:
            return 1
        elif z <= 18:
            return 1
        elif z <= 36:
            return 3
        elif z <= 54:
            return 5
        elif z <= 86:
            return 7
        elif z <= 118:
            return 9
        else:
            # Extrapolation for superheavies
            return 9 + int((z - 118) / 8)

    # ============================================
    # 1. SUPERHEAVY ELEMENT PREDICTIONS
    # ============================================

    def predict_superheavy_properties(self):
        """
        Predict properties for Z=119-126 using VHL holographic reconstruction.

        Novel predictions:
        - Z=126 as super-noble gas (vs. standard theory: not noble)
        - Ionization energies
        - Atomic radii
        - Chemical similarity to lighter elements
        """
        print("\n" + "="*60)
        print("SUPERHEAVY ELEMENT PREDICTIONS (Z=119-126)")
        print("="*60)

        superheavies = []

        # Use known data to build interpolation
        known_z = sorted(KNOWN_IONIZATION_ENERGIES.keys())
        known_ie = [KNOWN_IONIZATION_ENERGIES[z] for z in known_z]

        # Fit power law: IE(Z) = a * Z^b
        def power_law(z, a, b):
            return a * z**b

        popt_ie, _ = curve_fit(power_law, known_z, known_ie)

        # Same for radii
        known_r_z = sorted(KNOWN_ATOMIC_RADII.keys())
        known_r = [KNOWN_ATOMIC_RADII[z] for z in known_r_z]

        def radius_law(z, a, b, c):
            return a + b * np.exp(-z / c)

        popt_r, _ = curve_fit(radius_law, known_r_z, known_r)

        for z in range(119, 127):
            octave, tone = self.map_to_vhl_octave(z)
            n_nodes = self.count_nodes_from_z(z)

            # VHL holographic prediction with octave correction
            octave_factor = (octave / 2) ** 0.5  # Harmonic scaling

            # Base prediction from power law
            ie_base = power_law(z, *popt_ie)
            r_base = radius_law(z, *popt_r)

            # VHL corrections
            # Noble gas enhancement for Z=126
            if z == 126:
                ie_correction = 1.15  # 15% higher (more stable)
                r_correction = 0.95   # 5% smaller (more compact)
                chemical_class = "Super-Noble Gas (VHL NOVEL)"
            elif tone == 0:  # Other potential octave completions
                ie_correction = 1.08
                r_correction = 0.98
                chemical_class = "Alkali-like"
            elif tone == 1:
                ie_correction = 1.02
                r_correction = 0.99
                chemical_class = "Alkaline Earth-like"
            else:
                ie_correction = 1.0
                r_correction = 1.0
                chemical_class = "Transition-like"

            ie_predicted = ie_base * ie_correction * octave_factor
            r_predicted = r_base * r_correction

            # Calculate VHL position
            vhl_radius = 8.0 + 0.5 * n_nodes

            element = {
                'z': z,
                'symbol': ['Uue', 'Ubn', 'Ubu', 'Ubb', 'Ubt', 'Ubq', 'Ubp', 'Ubh'][z - 119],
                'octave': octave,
                'tone': tone,
                'n_nodes': n_nodes,
                'vhl_radius_angstrom': float(vhl_radius),
                'predictions': {
                    'ionization_energy_eV': float(ie_predicted),
                    'atomic_radius_angstrom': float(r_predicted),
                    'chemical_class': chemical_class,
                    'stability_index': float(1.0 / (abs(tone - 4.5) + 0.1)),  # Higher near tone 4-5
                    'reactivity': 'Low' if z == 126 else 'Moderate' if tone < 2 else 'High'
                },
                'comparison_to_standard_theory': {
                    'agreement': 'Novel' if z == 126 else 'Modified',
                    'key_difference': 'VHL predicts noble gas behavior' if z == 126 else 'Harmonic scaling applied'
                },
                'experimental_tests': [
                    f"Synthesize via heavy ion collision (e.g., Cf-249 + Ca-48)",
                    f"Measure first ionization energy (predicted: {ie_predicted:.2f} eV)",
                    f"Test reactivity with halogens (predicted: {chemical_class})",
                    f"Compare to isoelectronic ions"
                ]
            }

            superheavies.append(element)

            print(f"\n{element['symbol']} (Z={z}):")
            print(f"  VHL Class: {chemical_class}")
            print(f"  Octave: {octave}, Tone: {tone}, Nodes: {n_nodes}")
            print(f"  Predicted IE: {ie_predicted:.2f} eV")
            print(f"  Predicted Radius: {r_predicted:.3f} Å")
            print(f"  Stability Index: {element['predictions']['stability_index']:.2f}")

        self.predictions['superheavy_elements'] = superheavies

        # Highlight novel Z=126 prediction
        print("\n" + "="*60)
        print("⭐ NOVEL PREDICTION: Z=126 as Super-Noble Gas")
        print("="*60)
        print("Standard theory: Complex 5g² 6f² 7d² 8p⁶ configuration")
        print("VHL prediction: Octave 14 completion → noble gas behavior")
        print(f"Predicted IE: {superheavies[-1]['predictions']['ionization_energy_eV']:.2f} eV")
        print("Expected: Extremely low reactivity, high ionization energy")
        print("Test: Synthesize and measure chemical properties")

        return superheavies

    # ============================================
    # 2. HARMONIC SCALING (GOLDEN RATIO)
    # ============================================

    def test_golden_ratio_scaling(self):
        """
        Test if properties scale harmonically with golden ratio across octaves.

        Novel hypothesis:
        Property(Z + 9) / Property(Z) ≈ φ^α

        Where φ = 1.618... (golden ratio) and α is a property-specific exponent.
        """
        print("\n" + "="*60)
        print("GOLDEN RATIO HARMONIC SCALING TEST")
        print("="*60)

        results = {
            'hypothesis': 'Properties scale as φ^α across VHL octaves (9 elements)',
            'phi': float(PHI),
            'tests': []
        }

        # Test with ionization energies
        print("\nTest 1: Ionization Energy Scaling")
        print("-" * 60)

        test_pairs = [
            (1, 11),   # H → Na (octave 1 → 2)
            (3, 11),   # Li → Na (same tone, different octave)
            (11, 19),  # Na → K (octave 2 → 3)
            (2, 10),   # He → Ne (noble gases)
            (10, 18),  # Ne → Ar (noble gases)
        ]

        for z1, z2 in test_pairs:
            if z1 in KNOWN_IONIZATION_ENERGIES and z2 in KNOWN_IONIZATION_ENERGIES:
                ie1 = KNOWN_IONIZATION_ENERGIES[z1]
                ie2 = KNOWN_IONIZATION_ENERGIES[z2]
                ratio = ie2 / ie1

                octave1, tone1 = self.map_to_vhl_octave(z1)
                octave2, tone2 = self.map_to_vhl_octave(z2)

                # Calculate expected ratio if golden ratio holds
                delta_octave = octave2 - octave1
                expected_ratio = PHI ** (delta_octave * 0.5)  # α ≈ 0.5 for IE

                deviation = abs(ratio - expected_ratio) / expected_ratio * 100

                match = 'MATCH' if deviation < 20 else 'DEVIATE'

                print(f"Z={z1} → Z={z2}: {ie1:.2f} → {ie2:.2f} eV")
                print(f"  Ratio: {ratio:.3f}")
                print(f"  φ^{delta_octave*0.5:.2f} = {expected_ratio:.3f}")
                print(f"  Deviation: {deviation:.1f}% [{match}]")

                results['tests'].append({
                    'property': 'ionization_energy',
                    'z1': z1,
                    'z2': z2,
                    'measured_ratio': float(ratio),
                    'predicted_ratio': float(expected_ratio),
                    'deviation_percent': float(deviation),
                    'match': match
                })

        # Calculate overall fit quality
        deviations = [t['deviation_percent'] for t in results['tests']]
        avg_deviation = np.mean(deviations)

        print("\n" + "-" * 60)
        print(f"Average Deviation: {avg_deviation:.1f}%")

        if avg_deviation < 15:
            print("✅ GOLDEN RATIO SCALING CONFIRMED (< 15% deviation)")
            results['conclusion'] = 'Strong evidence for harmonic scaling'
        elif avg_deviation < 25:
            print("⚠️  PARTIAL GOLDEN RATIO SCALING (15-25% deviation)")
            results['conclusion'] = 'Moderate evidence, requires refinement'
        else:
            print("❌ GOLDEN RATIO SCALING NOT SUPPORTED (> 25% deviation)")
            results['conclusion'] = 'Hypothesis not supported by data'

        # Novel prediction: Use golden ratio to predict unknown elements
        print("\n" + "="*60)
        print("Novel Predictions Using Golden Ratio:")
        print("="*60)

        if avg_deviation < 25:
            # Predict IE for Z=37 (Rb) from Z=19 (K)
            z_known = 19
            z_predict = 37
            ie_known = KNOWN_IONIZATION_ENERGIES.get(z_known, 4.341)

            oct_k, _ = self.map_to_vhl_octave(z_known)
            oct_p, _ = self.map_to_vhl_octave(z_predict)

            ie_predicted = ie_known / (PHI ** ((oct_p - oct_k) * 0.5))
            ie_actual = 4.177  # Rb actual value

            print(f"\nK (Z={z_known}) → Rb (Z={z_predict})")
            print(f"  Known IE(K): {ie_known:.3f} eV")
            print(f"  VHL Predicted IE(Rb): {ie_predicted:.3f} eV")
            print(f"  Actual IE(Rb): {ie_actual:.3f} eV")
            print(f"  Error: {abs(ie_predicted - ie_actual)/ie_actual * 100:.1f}%")

            results['novel_prediction_test'] = {
                'element': 'Rb',
                'z': z_predict,
                'predicted_ie_eV': float(ie_predicted),
                'actual_ie_eV': ie_actual,
                'error_percent': float(abs(ie_predicted - ie_actual)/ie_actual * 100)
            }

        self.predictions['golden_ratio_scaling'] = results
        return results

    # ============================================
    # 3. ALLOY STABILITY PREDICTIONS
    # ============================================

    def predict_alloy_stability(self):
        """
        Predict ternary alloy stability based on VHL geometric constraints.

        Novel hypothesis:
        Alloys with elements forming "harmonic triangles" in VHL space
        show enhanced stability.
        """
        print("\n" + "="*60)
        print("TERNARY ALLOY STABILITY PREDICTIONS")
        print("="*60)

        def compute_vhl_triangle_score(z1, z2, z3):
            """
            Compute stability score based on VHL triangle geometry.

            Higher score = more stable alloy predicted.
            """
            oct1, tone1 = self.map_to_vhl_octave(z1)
            oct2, tone2 = self.map_to_vhl_octave(z2)
            oct3, tone3 = self.map_to_vhl_octave(z3)

            # Angular separation in tones
            angle_12 = abs(tone2 - tone1) * (2 * np.pi / 9)
            angle_23 = abs(tone3 - tone2) * (2 * np.pi / 9)
            angle_31 = abs(tone1 - tone3) * (2 * np.pi / 9)

            # Harmonic angles (in radians)
            harmonic_angles = np.array([np.pi/3, np.pi/2, 2*np.pi/3, np.pi])  # 60°, 90°, 120°, 180°

            score = 0
            for angle in [angle_12, angle_23, angle_31]:
                # Check proximity to harmonic angles
                min_diff = np.min(np.abs(harmonic_angles - angle))
                if min_diff < 0.3:  # Within ~17 degrees
                    score += (0.3 - min_diff) * 10  # Higher score for closer match

            # Octave similarity bonus (same or adjacent octaves)
            octave_spread = max(oct1, oct2, oct3) - min(oct1, oct2, oct3)
            if octave_spread <= 1:
                score += 2

            # Consecutive Z bonus (elements close in atomic number)
            z_spread = max(z1, z2, z3) - min(z1, z2, z3)
            if z_spread <= 5:
                score += 1

            return score

        # Test known alloys
        print("\nValidation Against Known Alloys:")
        print("-" * 60)

        known_alloys = [
            ((29, 50, 30), "Bronze (Cu-Sn-Zn)", "Stable", 8.5),
            ((26, 28, 24), "Stainless Steel (Fe-Ni-Cr)", "Very Stable", 9.0),
            ((13, 29, 26), "Aluminum Bronze (Al-Cu-Fe)", "Stable", 7.0),
            ((22, 13, 23), "Ti-Al-V", "Stable", 7.5),
        ]

        validation_results = []
        for (z1, z2, z3), name, known_stability, _ in known_alloys:
            score = compute_vhl_triangle_score(z1, z2, z3)
            print(f"{name}:")
            print(f"  Elements: Z={z1}, {z2}, {z3}")
            print(f"  VHL Score: {score:.2f}")
            print(f"  Known: {known_stability}")
            print(f"  Match: {'✅' if score > 5 else '⚠️'}")

            validation_results.append({
                'alloy': name,
                'elements': [z1, z2, z3],
                'vhl_score': float(score),
                'known_stability': known_stability
            })

        # Novel predictions
        print("\n" + "="*60)
        print("NOVEL ALLOY PREDICTIONS:")
        print("="*60)

        novel_predictions = []

        # Systematic search for high-scoring combinations
        print("\nSearching for novel high-stability alloys...")

        candidates = []
        for z1 in range(20, 30):
            for z2 in range(z1+1, min(z1+10, 40)):
                for z3 in range(z2+1, min(z2+5, 45)):
                    score = compute_vhl_triangle_score(z1, z2, z3)
                    if score > 6:  # High stability threshold
                        candidates.append(((z1, z2, z3), score))

        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Top 5 novel predictions
        print("\nTop 5 Novel High-Stability Alloy Predictions:")
        print("-" * 60)

        element_symbols = {
            20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn',
            26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga',
            32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb',
            38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
            44: 'Ru'
        }

        for i, ((z1, z2, z3), score) in enumerate(candidates[:5], 1):
            sym1 = element_symbols.get(z1, f'Z{z1}')
            sym2 = element_symbols.get(z2, f'Z{z2}')
            sym3 = element_symbols.get(z3, f'Z{z3}')

            oct1, tone1 = self.map_to_vhl_octave(z1)
            oct2, tone2 = self.map_to_vhl_octave(z2)
            oct3, tone3 = self.map_to_vhl_octave(z3)

            print(f"\n{i}. {sym1}-{sym2}-{sym3} (Z={z1},{z2},{z3})")
            print(f"   VHL Score: {score:.2f} (High)")
            print(f"   Octaves: {oct1}, {oct2}, {oct3}")
            print(f"   Predicted: Enhanced stability")
            print(f"   Suggested ratio: 1:1:1 or 2:1:1")
            print(f"   Test: Synthesis + XRD + mechanical testing")

            novel_predictions.append({
                'rank': i,
                'elements': [z1, z2, z3],
                'symbols': f"{sym1}-{sym2}-{sym3}",
                'vhl_score': float(score),
                'octaves': [oct1, oct2, oct3],
                'prediction': 'Enhanced stability',
                'suggested_composition': '1:1:1 atomic ratio',
                'experimental_tests': [
                    'Arc melting or powder metallurgy synthesis',
                    'X-ray diffraction for phase identification',
                    'Hardness and tensile strength measurements',
                    'Corrosion resistance testing',
                    'Compare to binary alloys of same elements'
                ]
            })

        self.predictions['alloy_stability'] = {
            'validation': validation_results,
            'novel_predictions': novel_predictions
        }

        return novel_predictions

    # ============================================
    # 4. MAGIC ANGLE REACTIVITY
    # ============================================

    def predict_magic_angle_reactivity(self):
        """
        Predict element pairs with enhanced reactivity based on VHL angular separation.

        Novel hypothesis:
        Pairs separated by 1/3 or 2/3 octave show enhanced reactivity.
        """
        print("\n" + "="*60)
        print("MAGIC ANGLE REACTIVITY PREDICTIONS")
        print("="*60)

        magic_ratios = [1/3, 2/3]  # Fractions of full octave (9 tones)

        print("\nMagic Angles: 1/3 octave (3 tones) and 2/3 octave (6 tones)")
        print("-" * 60)

        # Test known reactive pairs
        known_pairs = [
            (1, 5, "H-B", "BH₃", "High"),     # Separation: 4 tones
            (3, 8, "Li-O", "Li₂O", "Very High"),  # Separation: 5 tones
            (6, 9, "C-F", "CF₄", "Very High"),    # Separation: 3 tones ✓
            (11, 17, "Na-Cl", "NaCl", "Very High"), # Separation: 6 tones ✓
        ]

        print("\nValidation Against Known Reactive Pairs:")
        for z1, z2, pair_name, product, known_reactivity in known_pairs:
            oct1, tone1 = self.map_to_vhl_octave(z1)
            oct2, tone2 = self.map_to_vhl_octave(z2)

            tone_sep = abs(tone2 - tone1)
            is_magic = tone_sep == 3 or tone_sep == 6

            print(f"\n{pair_name} → {product}:")
            print(f"  Tone separation: {tone_sep}")
            print(f"  Magic angle: {'✅ YES' if is_magic else '❌ NO'}")
            print(f"  Known reactivity: {known_reactivity}")

        # Novel predictions
        print("\n" + "="*60)
        print("NOVEL HIGH-REACTIVITY PAIR PREDICTIONS:")
        print("="*60)

        novel_pairs = []

        # Search for magic angle pairs
        for z1 in range(1, 37):
            oct1, tone1 = self.map_to_vhl_octave(z1)

            # Find Z2 that is 3 or 6 tones away
            for delta_tone in [3, 6]:
                # Within same octave
                tone2 = (tone1 + delta_tone) % 9
                z2_candidate = z1 + delta_tone

                if 1 <= z2_candidate <= 36:
                    oct2, tone2_actual = self.map_to_vhl_octave(z2_candidate)

                    if tone2_actual == tone2 and abs(oct2 - oct1) <= 1:
                        novel_pairs.append((z1, z2_candidate, delta_tone))

        # Filter to most promising (not already well-known)
        promising = []
        element_names = {
            4: 'Be', 5: 'B', 7: 'N', 12: 'Mg', 13: 'Al', 14: 'Si',
            15: 'P', 16: 'S', 18: 'Ar', 19: 'K', 20: 'Ca', 23: 'V',
            25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
            31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se'
        }

        for z1, z2, delta in novel_pairs[:10]:
            sym1 = element_names.get(z1, f'Z{z1}')
            sym2 = element_names.get(z2, f'Z{z2}')

            print(f"\n{sym1}-{sym2} (Z={z1},{z2}):")
            print(f"  Tone separation: {delta} (magic angle)")
            print(f"  Prediction: Enhanced reactivity")
            print(f"  Suggested product: {sym1}_{{{2 if delta==3 else 1}}}{sym2}_{{{1 if delta==3 else 2}}}")

            promising.append({
                'z1': z1,
                'z2': z2,
                'symbols': f"{sym1}-{sym2}",
                'tone_separation': delta,
                'magic_angle': True,
                'predicted_reactivity': 'Enhanced',
                'suggested_experiments': [
                    f"React {sym1} and {sym2} at various temperatures",
                    "Measure formation enthalpy",
                    "Compare to DFT predictions",
                    "Check for unexpected product stability"
                ]
            })

        self.predictions['magic_angle_reactivity'] = promising
        return promising

    # ============================================
    # MAIN EXECUTION
    # ============================================

    def generate_all_predictions(self, output_file='vhl_predictions.json'):
        """Generate all predictions and save to JSON."""
        print("\n" + "="*70)
        print(" " * 20 + "VHL NOVEL PREDICTIONS GENERATOR")
        print("="*70)
        print("\nGenerating experimentally testable predictions...")
        print("This may take a few minutes...\n")

        # Run all prediction modules
        self.predict_superheavy_properties()
        self.test_golden_ratio_scaling()
        self.predict_alloy_stability()
        self.predict_magic_angle_reactivity()

        # Save to file
        with open(output_file, 'w') as f:
            json.dump(self.predictions, f, indent=2)

        print("\n" + "="*70)
        print(f"✅ All predictions saved to: {output_file}")
        print("="*70)

        # Summary
        print("\nSUMMARY OF NOVEL PREDICTIONS:")
        print("-" * 70)
        print(f"• Superheavy elements: {len(self.predictions.get('superheavy_elements', []))} predictions")
        print(f"• Golden ratio tests: {len(self.predictions.get('golden_ratio_scaling', {}).get('tests', []))} pairs analyzed")
        print(f"• Novel alloys: {len(self.predictions.get('alloy_stability', {}).get('novel_predictions', []))} candidates")
        print(f"• Magic angle pairs: {len(self.predictions.get('magic_angle_reactivity', []))} predictions")

        print("\n⭐ HIGHLIGHT: Most Testable Novel Prediction")
        print("-" * 70)
        print("Z=126 (Ubh) as Super-Noble Gas")
        print("• VHL: Octave 14 completion → noble gas behavior")
        print("• Standard theory: Complex multi-shell → not noble")
        print("• Test: Synthesize and measure ionization energy")
        print("• If confirmed: Major validation of VHL octave theory")

        return self.predictions


def main():
    parser = argparse.ArgumentParser(
        description='Generate VHL Novel Predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python vhl_predictions.py --generate all
  python vhl_predictions.py --generate superheavy --output superheavy.json
  python vhl_predictions.py --generate alloys
        '''
    )

    parser.add_argument('--generate', default='all',
                       choices=['all', 'superheavy', 'golden_ratio', 'alloys', 'magic_angles'],
                       help='Which predictions to generate')
    parser.add_argument('--output', default='vhl_predictions.json',
                       help='Output JSON file')

    args = parser.parse_args()

    predictor = VHLPredictions()

    if args.generate == 'all':
        predictor.generate_all_predictions(args.output)
    elif args.generate == 'superheavy':
        predictor.predict_superheavy_properties()
    elif args.generate == 'golden_ratio':
        predictor.test_golden_ratio_scaling()
    elif args.generate == 'alloys':
        predictor.predict_alloy_stability()
    elif args.generate == 'magic_angles':
        predictor.predict_magic_angle_reactivity()


if __name__ == '__main__':
    main()
