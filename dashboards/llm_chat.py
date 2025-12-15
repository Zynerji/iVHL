"""
iVHL LLM Chat Interface
=======================

Interactive chat interface for the embedded LLM agent.
Allows users to:
- Ask questions about simulations
- Modify parameters via natural language
- Request white papers
- Monitor simulation progress

Uses Gradio for web UI with WebSocket support.

Author: iVHL Framework
Date: 2025-12-15
"""

import sys
import os
from pathlib import Path
import asyncio
import json
import logging

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import torch

from ivhl.llm.agent import iVHLAgent, LLMConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimulationManager:
    """
    Mock simulation manager for demonstration
    In production, this would be the actual iVHL simulation
    """

    def __init__(self):
        self.status = "running"
        self.current_timestep = 0
        self.total_timesteps = 1000
        self.progress = 0.0
        self.parameters = {
            "num_lattice_nodes": 1000,
            "gw_amplitude": 1e-21,
            "gw_frequency": 100.0,
            "perturbation_type": "constant_lattice",
            "temperature": 0.7
        }
        self.metrics = {
            "max_field_value": 1.23e-20,
            "correlation": 0.95,
            "nan_fraction": 0.0,
            "fractal_dimension": 2.1,
            "entanglement_entropy": 15.3
        }
        self.metrics_history = {
            "correlation": [0.93 + i*0.001 for i in range(100)]
        }

    def get_metrics(self):
        return self.metrics

    def get_parameters(self):
        return self.parameters

    def set_parameter(self, name, value):
        if name in self.parameters:
            self.parameters[name] = value
        else:
            raise ValueError(f"Unknown parameter: {name}")

    def pause(self):
        self.status = "paused"

    def resume(self):
        self.status = "running"

    def get_metric_history(self, metric_name, last_n=100):
        return self.metrics_history.get(metric_name, [])[-last_n:]

    def generate_report(self):
        return Path("/app/whitepapers/report_latest.pdf")

    def get_parameter_schema(self):
        return [
            {"name": "num_lattice_nodes", "type": "int", "description": "Number of lattice nodes"},
            {"name": "gw_amplitude", "type": "float", "description": "GW strain amplitude"},
            {"name": "gw_frequency", "type": "float", "description": "GW frequency in Hz"},
            {"name": "perturbation_type", "type": "string", "description": "Type of GW perturbation"},
            {"name": "temperature", "type": "float", "description": "RL exploration temperature"}
        ]


class ChatInterface:
    """Gradio chat interface for LLM agent"""

    def __init__(self):
        # Initialize simulation manager
        self.sim_manager = SimulationManager()

        # Initialize LLM agent
        self.config = LLMConfig(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Small, fast
            max_tokens=1024,
            temperature=0.7,
            gpu_memory_utilization=0.15,  # Leave room for simulations
            monitor_interval=30.0
        )

        logger.info("Initializing LLM agent (this may take a minute)...")
        self.agent = iVHLAgent(self.config, self.sim_manager)
        logger.info("LLM agent ready!")

        # Start monitoring in background
        self.monitoring_task = None

    def chat_response(self, message, history):
        """
        Generate chat response

        Args:
            message: User message
            history: Conversation history

        Returns:
            Updated history
        """
        try:
            # Get response from agent
            response = self.agent.chat(message, use_tools=True)

            # Update history
            history.append((message, response))

            return history

        except Exception as e:
            logger.error(f"Chat error: {e}")
            error_msg = f"Error: {str(e)}"
            history.append((message, error_msg))
            return history

    def get_simulation_status(self):
        """Get current simulation status for display"""
        state = self.sim_manager.get_parameters()
        metrics = self.sim_manager.get_metrics()

        status_text = f"""**Simulation Status: {self.sim_manager.status}**

**Parameters:**
- Lattice Nodes: {state['num_lattice_nodes']}
- GW Amplitude: {state['gw_amplitude']:.2e}
- GW Frequency: {state['gw_frequency']} Hz
- Perturbation: {state['perturbation_type']}

**Metrics:**
- Correlation: {metrics['correlation']:.4f}
- Fractal Dimension: {metrics['fractal_dimension']:.2f}
- Entanglement Entropy: {metrics['entanglement_entropy']:.2f}
- Max Field Value: {metrics['max_field_value']:.2e}
"""
        return status_text

    def generate_paper(self):
        """Generate white paper on demand"""
        try:
            simulation_data = {
                "parameters": self.sim_manager.get_parameters(),
                "metrics": self.sim_manager.get_metrics(),
                "status": self.sim_manager.status
            }

            latex_source = self.agent.generate_white_paper_with_llm(simulation_data)

            return f"White paper generated!\n\n```latex\n{latex_source[:500]}...\n```"

        except Exception as e:
            return f"Error generating paper: {e}"

    def build_interface(self):
        """Build Gradio interface"""

        with gr.Blocks(title="iVHL LLM Assistant", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 🤖 iVHL LLM Assistant

            Chat with the embedded AI assistant to:
            - Ask questions about the simulation
            - Modify parameters via natural language
            - Request white paper generation
            - Monitor simulation progress

            **Model**: TinyLlama 1.1B (running on H100 alongside simulations)
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="Chat with iVHL Agent",
                        height=500,
                        show_copy_button=True
                    )

                    with gr.Row():
                        msg = gr.Textbox(
                            label="Message",
                            placeholder="Ask me anything about the simulation...",
                            lines=2,
                            scale=4
                        )
                        send = gr.Button("Send", scale=1, variant="primary")

                    # Example prompts
                    gr.Examples(
                        examples=[
                            "What is the current simulation status?",
                            "Set GW amplitude to 5e-21",
                            "Show me the correlation history",
                            "Generate a white paper for this simulation",
                            "What parameters can I modify?",
                            "Pause the simulation",
                            "What is the fractal dimension telling us?"
                        ],
                        inputs=msg,
                        label="Example Prompts"
                    )

                with gr.Column(scale=1):
                    # Simulation status panel
                    status_display = gr.Markdown(
                        value=self.get_simulation_status(),
                        label="Simulation Status"
                    )

                    refresh_btn = gr.Button("🔄 Refresh Status")

                    gr.Markdown("---")

                    # White paper generation
                    gr.Markdown("### White Paper Generation")
                    paper_btn = gr.Button("📄 Generate White Paper", variant="secondary")
                    paper_output = gr.Textbox(
                        label="Generation Status",
                        lines=10,
                        interactive=False
                    )

            # Event handlers
            send.click(
                fn=self.chat_response,
                inputs=[msg, chatbot],
                outputs=chatbot
            ).then(
                fn=lambda: "",
                outputs=msg
            )

            msg.submit(
                fn=self.chat_response,
                inputs=[msg, chatbot],
                outputs=chatbot
            ).then(
                fn=lambda: "",
                outputs=msg
            )

            refresh_btn.click(
                fn=self.get_simulation_status,
                outputs=status_display
            )

            paper_btn.click(
                fn=self.generate_paper,
                outputs=paper_output
            )

        return demo

    def launch(self, share=False, server_port=7860):
        """Launch the Gradio interface"""
        demo = self.build_interface()

        logger.info(f"Launching chat interface on port {server_port}...")

        demo.launch(
            server_name="0.0.0.0",
            server_port=server_port,
            share=share,
            show_error=True
        )


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="iVHL LLM Chat Interface")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")

    args = parser.parse_args()

    # Create and launch interface
    interface = ChatInterface()
    interface.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
