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
    """

    def __init__(
        self,
        vllm_url: str = "http://localhost:8000/v1/chat/completions",
        model_name: str = "Qwen/Qwen2.5-2B-Instruct"
    ):
        self.vllm_url = vllm_url
        self.model_name = model_name
        self.conversation_history = []
        self.notes = []

        # Initialize with system prompt
        self.conversation_history.append({
            "role": "system",
            "content": SYSTEM_PROMPT
        })

        print(f"LLM Monitoring Agent initialized:")
        print(f"  - Model: {model_name}")
        print(f"  - Endpoint: {vllm_url}")

    def query_llm(
        self,
        user_message: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """Send query to vLLM server."""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

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
                return f"Error: vLLM returned status {response.status_code}"

        except requests.exceptions.RequestException as e:
            return f"Error connecting to vLLM: {e}"

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

    def generate_whitepaper_section(self, section: str, data: Dict) -> str:
        """
        Generate a section of the whitepaper.

        Args:
            section: Section name ("abstract", "results", "discussion", etc.)
            data: Relevant data for this section

        Returns:
            LaTeX formatted text
        """
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
