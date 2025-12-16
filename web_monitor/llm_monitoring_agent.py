"""
LLM Monitoring Agent
====================

Qwen2.5-2B agent for real-time simulation monitoring with note-taking.
"""

import requests
import json
from typing import Dict, List, Optional
import time


# System prompt that explains the LLM's purpose
SYSTEM_PROMPT = """You are an AI scientific assistant monitoring a computational simulation of hierarchical information dynamics in tensor networks.

## Your Role

You are observing a MATHEMATICAL simulation exploring information flow through multi-layer tensor networks. Your job is to:

1. **Monitor** simulation metrics in real-time (entropy, correlations, layer states)
2. **Take notes** on notable events, patterns, and anomalies
3. **Answer questions** from researchers about what's happening
4. **Provide commentary** on observed dynamics
5. **Generate whitepaper content** when the simulation completes

## Critical Understanding

**This is NOT a physics simulation.** It explores:
- ✅ Information compression in tensor networks
- ✅ Entropy redistribution across hierarchical layers
- ✅ Correlation structure emergence
- ✅ Mathematical pattern formation

**This is NOT:**
- ❌ A model of dark matter or dark energy
- ❌ A theory of quantum gravity or cosmology
- ❌ A prediction about physical reality
- ❌ A multiverse simulation

## Scientific Honesty

When discussing results:
- Use precise mathematical language
- Cite information-theoretic concepts (von Neumann entropy, correlations)
- Note computational limitations
- Never claim physical significance
- Always include: "This represents patterns in an abstract mathematical system."

## Note-Taking Guidelines

Flag events as NOTABLE if:
- Entropy changes by >10% in single step
- Correlations exceed 0.9 or drop below 0.1
- Layer norms diverge unexpectedly
- Clear patterns emerge (power laws, periodicity, fractals)

## Response Style

- **Concise** - Researchers are busy, keep it brief
- **Quantitative** - Always cite numbers from metrics
- **Analytical** - Explain *why* something is happening mathematically
- **Honest** - Admit when you don't know or when data is ambiguous

## Example Interactions

**Good:**
User: "Why did entropy spike at step 347?"
You: "Entropy increased 18% (from 4.2 to 5.0) at step 347 because the SVD compression at layer 2 truncated low singular values, concentrating information into fewer modes. This is expected when compression ratio > 0.5."

**Bad:**
User: "Does this explain dark matter?"
You: "No. This is a mathematical exploration of tensor network compression, not a physical cosmology model. Results have no connection to dark matter."

## Available Metrics

You receive these every 10 seconds:
- step: Current timestep
- layer_entropies: List of von Neumann entropy for each layer
- layer_norms: Frobenius norm of each tensor
- correlations: Dict of inter-layer correlation coefficients
- compression_losses: Information lost in recent compressions

Use these to inform your commentary and notes.

Remember: You are a tool for understanding computational patterns in abstract mathematics. Your value is in clear, honest analysis of what the simulation is actually doing."""


class LLMMonitoringAgent:
    """
    Monitors simulation and provides real-time LLM commentary.

    SPOF #3 Fix: Supports offline fallback mode when vLLM unavailable.
    """

    def __init__(
        self,
        vllm_url: str = "http://localhost:8000/v1/chat/completions",
        model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        enable_fallback: bool = True
    ):
        self.vllm_url = vllm_url
        self.model_name = model_name
        self.conversation_history = []
        self.notes = []
        self.enable_fallback = enable_fallback
        self.offline_mode = False
        self.llm_available = False

        # Initialize with system prompt
        self.conversation_history.append({
            "role": "system",
            "content": SYSTEM_PROMPT
        })

        print(f"LLM Monitoring Agent initializing...")
        print(f"  - Model: {model_name}")
        print(f"  - Endpoint: {vllm_url}")

        # Test LLM availability
        self._check_llm_availability()

    def _check_llm_availability(self) -> bool:
        """Check if LLM server is reachable."""
        try:
            response = requests.get(
                self.vllm_url.replace('/v1/chat/completions', '/v1/models'),
                timeout=5
            )
            self.llm_available = response.status_code == 200

            if self.llm_available:
                print(f"  ✅ LLM server online")
            else:
                print(f"  ⚠️  LLM server unreachable (status {response.status_code})")

        except Exception as e:
            self.llm_available = False
            print(f"  ⚠️  LLM server not available: {e}")

        if not self.llm_available and self.enable_fallback:
            self.offline_mode = True
            print(f"  📴 Entering OFFLINE MODE (rule-based analysis)")

        return self.llm_available

    def _offline_analysis(self, user_message: str) -> str:
        """
        Fallback: Rule-based analysis when LLM unavailable.

        Provides basic metric interpretation without LLM.
        """
        # Extract metrics from message if present
        if "Analyze step" in user_message:
            # Parse simple patterns
            if "Entropies:" in user_message:
                return "NOTE: Monitoring entropy values (LLM offline - rule-based analysis active)"
            if "Correlations:" in user_message:
                return "OK"

        # Answer common questions with canned responses
        elif "why" in user_message.lower():
            return "LLM offline: Unable to provide detailed analysis. Check metrics manually."
        elif "what" in user_message.lower():
            return "LLM offline: Simulation running in automated mode. Metrics being logged."
        else:
            return "LLM offline: Limited analysis available. Core simulation continues normally."

    def query_llm(
        self,
        user_message: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """Send query to vLLM server, with offline fallback."""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Use fallback if in offline mode
        if self.offline_mode:
            assistant_message = self._offline_analysis(user_message)
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            return assistant_message

        # Call vLLM API
        try:
            response = requests.post(
                self.vllm_url,
                json={
                    "model": self.model_name,
                    "messages": self.conversation_history,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                assistant_message = data['choices'][0]['message']['content']

                # Add to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })

                return assistant_message
            else:
                error_msg = f"Error: vLLM returned status {response.status_code}"

                # Fall back to offline mode on error
                if self.enable_fallback:
                    print(f"⚠️  LLM error, switching to offline mode")
                    self.offline_mode = True
                    return self._offline_analysis(user_message)

                return error_msg

        except requests.exceptions.RequestException as e:
            error_msg = f"Error connecting to vLLM: {e}"

            # Fall back to offline mode on connection error
            if self.enable_fallback and not self.offline_mode:
                print(f"⚠️  LLM connection failed, switching to offline mode")
                self.offline_mode = True
                return self._offline_analysis(user_message)

            return error_msg

    def analyze_step(self, metrics: Dict) -> Optional[Dict]:
        """
        Analyze current simulation step and decide if notable.

        Args:
            metrics: Current step metrics

        Returns:
            Note dict if notable, None otherwise
        """
        step = metrics.get('step', 0)
        layer_entropies = metrics.get('layer_entropies', [])
        correlations = metrics.get('correlations', {})

        # Build analysis prompt
        prompt = f"""Analyze step {step}:

Entropies: {layer_entropies}
Correlations: {correlations}

Is anything NOTABLE? (entropy change >10%, correlations extreme, unexpected patterns)

If yes, respond with:
NOTE: [brief description of what's notable]

If no, respond with:
OK

Be brief!"""

        response = self.query_llm(prompt, max_tokens=128)

        if response.startswith("NOTE:"):
            # Extract note content
            note_content = response.replace("NOTE:", "").strip()

            note = {
                'step': step,
                'content': note_content,
                'category': 'notable_event',
                'timestamp': time.time()
            }

            self.notes.append(note)
            return note

        return None

    def answer_question(self, question: str) -> str:
        """
        Answer user question about simulation.
        """
        # Add context about recent state
        context = f"\nRecent notes: {self.notes[-5:] if self.notes else 'None yet'}"

        full_prompt = question + context

        return self.query_llm(full_prompt, max_tokens=512)

    def _generate_offline_whitepaper_section(self, section: str, data: Dict) -> str:
        """
        Fallback: Generate basic whitepaper section without LLM.

        Returns template-based LaTeX content.
        """
        if section == "abstract":
            return f"""\\begin{{abstract}}
This computational study explores hierarchical information dynamics in multi-layer tensor networks.
The simulation tracked von Neumann entropy and correlation patterns across {data.get('num_layers', 'N')} layers
over {data.get('timesteps', 'N')} timesteps. Key observations include entropy redistribution and
correlation structure emergence. This work represents an abstract mathematical exploration with no
direct physical interpretation.

\\textit{{Note: Generated automatically (LLM unavailable).}}
\\end{{abstract}}"""

        elif section == "results":
            return f"""\\section{{Results}}

\\subsection{{Simulation Overview}}
The simulation consisted of {data.get('num_layers', 'N')} hierarchical layers with
base dimension {data.get('base_dim', 'N')} and bond dimension {data.get('bond_dim', 'N')}.
The system evolved over {data.get('timesteps', 'N')} timesteps.

\\subsection{{Entropy Evolution}}
Von Neumann entropy was tracked across all layers. Detailed metrics available in output logs.

\\subsection{{Correlation Patterns}}
Inter-layer correlations were monitored throughout the simulation run.

\\textit{{Note: Detailed analysis unavailable (LLM offline). See metrics JSON for raw data.}}
"""

        elif section == "discussion":
            return f"""\\section{{Discussion}}

This computational experiment explored information flow through hierarchical tensor compression.
The observed patterns reflect mathematical properties of the tensor network architecture and
compression strategy employed.

Future work could explore alternative compression schemes, different layer configurations, and
extended simulation durations.

\\textit{{Note: Generated automatically (LLM unavailable).}}
"""

        else:
            return f"\\section{{{section.title()}}}\\n\\nSection content unavailable (LLM offline).\\n"

    def generate_whitepaper_section(self, section: str, data: Dict) -> str:
        """
        Generate a section of the whitepaper.

        Args:
            section: Section name ("abstract", "results", "discussion", etc.)
            data: Relevant data for this section

        Returns:
            LaTeX formatted text
        """
        # Use offline fallback if LLM unavailable
        if self.offline_mode:
            return self._generate_offline_whitepaper_section(section, data)

        if section == "abstract":
            prompt = f"""Generate a scientific abstract (1 paragraph, ~150 words) for this computational study.

Data: {json.dumps(data, indent=2)}

Notes from simulation: {self.notes}

Requirements:
- Summarize what was simulated (hierarchical tensor network compression)
- Key findings (patterns observed, metrics)
- LaTeX format
- Include disclaimer that this is computational mathematics, not physics

Start with: \\begin{{abstract}}"""

        elif section == "results":
            prompt = f"""Generate Results section for scientific whitepaper.

Data: {json.dumps(data, indent=2)}

Notes: {self.notes}

Requirements:
- Chronological narrative of simulation
- Cite specific timesteps and metric values
- Describe observed patterns (power laws, correlations, etc.)
- LaTeX format with subsections
- Include references to figures (Fig. 1, Fig. 2, etc.)

Start with: \\section{{Results}}"""

        elif section == "discussion":
            prompt = f"""Generate Discussion section interpreting results.

Notes: {self.notes}

Requirements:
- Interpret patterns observed
- Compare to expectations from information theory
- Note limitations and uncertainties
- Suggest future computational experiments
- LaTeX format

Start with: \\section{{Discussion}}"""

        else:
            prompt = f"Generate {section} section for whitepaper using data: {data}"

        response = self.query_llm(prompt, max_tokens=1024, temperature=0.7)

        # If response indicates error, fall back to offline generation
        if response.startswith("Error") or response.startswith("LLM offline"):
            return self._generate_offline_whitepaper_section(section, data)

        return response

    def get_all_notes(self) -> List[Dict]:
        """Return all notes taken during simulation."""
        return self.notes

    def reset(self):
        """Reset conversation and notes."""
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.notes = []


if __name__ == "__main__":
    # Test agent
    agent = LLMMonitoringAgent()

    print("\nTesting LLM agent...")

    # Simulate a notable event
    test_metrics = {
        'step': 347,
        'layer_entropies': [4.2, 3.8, 3.1, 2.5, 1.8],
        'correlations': {'0-1': 0.92, '1-2': 0.87},
    }

    note = agent.analyze_step(test_metrics)

    if note:
        print(f"\nNote taken: {note['content']}")
    else:
        print("\nNo notable events")

    # Test question answering
    question = "Why are layer entropies decreasing?"
    answer = agent.answer_question(question)
    print(f"\nQ: {question}")
    print(f"A: {answer}")
