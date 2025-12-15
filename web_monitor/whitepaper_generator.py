"""Whitepaper Generator - Creates LaTeX PDFs from LLM notes"""
import subprocess
from pathlib import Path

class WhitepaperGenerator:
    def __init__(self, output_dir="docs/whitepapers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self, llm_agent, data: dict, filename="hierarchical_dynamics"):
        """Generate whitepaper PDF using LLM notes."""
        
        # Generate sections using LLM
        abstract = llm_agent.generate_whitepaper_section("abstract", data)
        results = llm_agent.generate_whitepaper_section("results", data)
        discussion = llm_agent.generate_whitepaper_section("discussion", data)
        
        # Assemble LaTeX document
        latex = f"""\documentclass{{article}}
\usepackage{{amsmath}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}

\title{{Hierarchical Information Dynamics: Computational Exploration}}
\author{{iVHL Framework + Qwen2.5-2B AI Assistant}}
\date{{\today}}

\begin{{document}}
\maketitle

{abstract}

\section{{Introduction}}
This whitepaper presents results from a computational simulation exploring
information flow in hierarchical tensor networks. This is a MATHEMATICAL
study, not a physical theory.

{results}

{discussion}

\section{{Conclusion}}
This simulation demonstrated computational patterns in tensor network
compression. All results represent abstract mathematical structures.

\textbf{{Disclaimer}}: This work explores mathematical models only.
No claims are made about physical reality, dark matter, cosmology, or
quantum gravity.

\end{{document}}
"""
        
        # Write LaTeX
        tex_path = self.output_dir / f"{filename}.tex"
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(latex)
        
        # Compile PDF
        try:
            subprocess.run(['pdflatex', '-output-directory', str(self.output_dir), str(tex_path)],
                         check=True, capture_output=True)
            print(f"✅ Whitepaper generated: {self.output_dir}/{filename}.pdf")
            return str(self.output_dir / f"{filename}.pdf")
        except subprocess.CalledProcessError:
            print("⚠️ pdflatex failed, LaTeX source saved")
            return str(tex_path)
