"""
iVHL LLM Agent - Autonomous Simulation Monitor and Controller
==============================================================

Embeds a small LLM (TinyLlama/Phi-3) into the Docker container to:
- Monitor simulations in real-time
- Answer user questions about simulation state
- Modify simulation parameters via natural language
- Generate white papers autonomously using H100
- Provide interactive chat interface

Models Supported:
-----------------
- TinyLlama-1.1B (fast, lightweight)
- Phi-3-mini-4k (3.8B, more capable)
- Llama-3.2-3B (newer, good performance)
- Mistral-7B-Instruct (quantized 4-bit for quality)

Architecture:
-------------
- vLLM for efficient serving (H100 optimized)
- Function calling for simulation control
- WebSocket for real-time chat
- Continuous monitoring with event triggers
- Autonomous white paper generation

Author: iVHL Framework
Date: 2025-12-15
"""

import os
import sys
import json
import torch
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

# LLM serving
try:
    from vllm import LLM, SamplingParams
    from vllm.entrypoints.openai.api_server import run_server
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM not available - install with: pip install vllm")

# Alternative: llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for embedded LLM"""

    # Model selection
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Default: small and fast
    model_path: Optional[str] = None  # Local path to model weights

    # Quantization (for larger models)
    quantization: Optional[str] = None  # "awq", "gptq", "squeezellm"
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # Generation parameters
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

    # vLLM settings
    tensor_parallel_size: int = 1  # Use 1 GPU for LLM
    gpu_memory_utilization: float = 0.2  # Leave 80% for simulations
    max_model_len: int = 4096

    # Monitoring
    monitor_interval: float = 30.0  # Check simulation every 30s
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'divergence': 1e10,
        'nan_fraction': 0.01,
        'correlation_drop': 0.5
    })

    # White paper generation
    auto_generate_papers: bool = True
    paper_generation_trigger: str = "simulation_complete"  # or "on_request"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SimulationTools:
    """
    Tools/functions that the LLM can call to interact with simulations

    These are exposed as function calls to the LLM via JSON schema
    """

    def __init__(self, simulation_manager):
        self.sim_manager = simulation_manager

    def get_simulation_state(self) -> Dict[str, Any]:
        """Get current simulation state and metrics"""
        return {
            "status": self.sim_manager.status,
            "timestep": self.sim_manager.current_timestep,
            "total_timesteps": self.sim_manager.total_timesteps,
            "progress": self.sim_manager.progress,
            "metrics": self.sim_manager.get_metrics(),
            "parameters": self.sim_manager.get_parameters()
        }

    def set_parameter(self, param_name: str, value: Any) -> Dict[str, str]:
        """Modify a simulation parameter"""
        try:
            self.sim_manager.set_parameter(param_name, value)
            return {"status": "success", "message": f"Set {param_name} to {value}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def pause_simulation(self) -> Dict[str, str]:
        """Pause the running simulation"""
        self.sim_manager.pause()
        return {"status": "success", "message": "Simulation paused"}

    def resume_simulation(self) -> Dict[str, str]:
        """Resume a paused simulation"""
        self.sim_manager.resume()
        return {"status": "success", "message": "Simulation resumed"}

    def get_metrics_history(self, metric_name: str, last_n: int = 100) -> List[float]:
        """Get historical values of a specific metric"""
        return self.sim_manager.get_metric_history(metric_name, last_n)

    def generate_white_paper(self) -> Dict[str, str]:
        """Trigger white paper generation for current simulation"""
        try:
            report_path = self.sim_manager.generate_report()
            return {"status": "success", "report_path": str(report_path)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_available_parameters(self) -> List[Dict[str, Any]]:
        """List all modifiable simulation parameters"""
        return self.sim_manager.get_parameter_schema()


# Function calling schema for LLM
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_simulation_state",
            "description": "Get the current state and metrics of the running simulation",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_parameter",
            "description": "Modify a simulation parameter. Use get_available_parameters first to see options.",
            "parameters": {
                "type": "object",
                "properties": {
                    "param_name": {"type": "string", "description": "Name of parameter to modify"},
                    "value": {"description": "New value for the parameter"}
                },
                "required": ["param_name", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pause_simulation",
            "description": "Pause the currently running simulation",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "resume_simulation",
            "description": "Resume a paused simulation",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_metrics_history",
            "description": "Get historical values of a specific metric",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_name": {"type": "string"},
                    "last_n": {"type": "integer", "default": 100}
                },
                "required": ["metric_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_white_paper",
            "description": "Generate a comprehensive white paper report for the current simulation",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_parameters",
            "description": "List all simulation parameters that can be modified",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]


class iVHLAgent:
    """
    Main LLM agent for iVHL framework

    Provides:
    - Natural language interface to simulations
    - Autonomous monitoring and alerting
    - White paper generation
    - Interactive Q&A
    """

    def __init__(self, config: LLMConfig, simulation_manager=None):
        self.config = config
        self.sim_manager = simulation_manager
        self.tools = SimulationTools(simulation_manager) if simulation_manager else None

        # Initialize LLM
        self.llm = self._initialize_llm()

        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []

        # Monitoring state
        self.monitoring_active = False
        self.last_alert_time = None

        logger.info(f"iVHL Agent initialized with model: {config.model_name}")

    def _initialize_llm(self):
        """Initialize the LLM backend"""

        if VLLM_AVAILABLE and self.config.device == "cuda":
            # Prefer vLLM for GPU (fastest)
            logger.info("Initializing vLLM backend...")
            return LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                quantization=self.config.quantization
            )

        elif LLAMACPP_AVAILABLE:
            # Fallback to llama.cpp (CPU/GPU)
            logger.info("Initializing llama.cpp backend...")
            model_path = self.config.model_path or self._download_gguf_model()
            return Llama(
                model_path=model_path,
                n_ctx=self.config.max_model_len,
                n_gpu_layers=-1 if self.config.device == "cuda" else 0,
                verbose=False
            )

        else:
            raise RuntimeError("No LLM backend available. Install vllm or llama-cpp-python")

    def _download_gguf_model(self) -> str:
        """Download a GGUF model if needed"""
        # Default to TinyLlama GGUF
        model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
        filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

        from huggingface_hub import hf_hub_download

        logger.info(f"Downloading {filename}...")
        model_path = hf_hub_download(repo_id=model_id, filename=filename)
        logger.info(f"Model downloaded to: {model_path}")

        return model_path

    def chat(self, user_message: str, use_tools: bool = True) -> str:
        """
        Chat with the agent

        Args:
            user_message: User's message
            use_tools: Whether to allow function calling

        Returns:
            Agent's response
        """
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})

        # Build prompt with system message
        system_message = self._get_system_prompt()

        # Generate response
        if VLLM_AVAILABLE and isinstance(self.llm, LLM):
            response = self._generate_vllm(system_message, use_tools)
        else:
            response = self._generate_llamacpp(system_message, use_tools)

        # Add to history
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def _get_system_prompt(self) -> str:
        """Get system prompt for the agent"""
        return """You are an AI assistant embedded in the iVHL (Integrated Vibrational Helix Lattice) framework.

Your role:
- Monitor holographic resonance simulations in real-time
- Answer questions about simulation state, parameters, and results
- Modify simulation parameters when requested by the user
- Generate white paper reports autonomously
- Alert users to anomalies or interesting phenomena

Knowledge:
- Group Field Theory (GFT) and quantum gravity
- Tensor networks (MERA) and holographic duality
- LIGO gravitational wave analysis
- Reinforcement learning discovery
- Holographic resonance on spherical boundaries

Available tools:
- get_simulation_state(): Check current simulation status
- set_parameter(name, value): Modify a parameter
- pause_simulation(): Pause the simulation
- resume_simulation(): Resume simulation
- get_metrics_history(metric, n): Get historical metric values
- generate_white_paper(): Create comprehensive report
- get_available_parameters(): List modifiable parameters

Communication style:
- Technical but accessible
- Provide specific metric values
- Suggest parameter adjustments when appropriate
- Be proactive in identifying issues
- Use scientific terminology correctly

When generating white papers:
- Include all relevant equations and physics
- Cite appropriate references (Maldacena, Ryu-Takayanagi, LIGO, etc.)
- Analyze results critically
- Suggest future directions
"""

    def _generate_vllm(self, system_message: str, use_tools: bool) -> str:
        """Generate response using vLLM"""
        # Format messages
        messages = [{"role": "system", "content": system_message}]
        messages.extend(self.conversation_history[-10:])  # Last 10 messages

        # Create prompt (model-specific formatting would go here)
        prompt = self._format_messages_for_model(messages)

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=self.config.max_tokens
        )

        # Generate
        outputs = self.llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()

        # Handle function calls if present
        if use_tools and self.tools:
            response = self._handle_function_calls(response)

        return response

    def _generate_llamacpp(self, system_message: str, use_tools: bool) -> str:
        """Generate response using llama.cpp"""
        # Format messages
        messages = [{"role": "system", "content": system_message}]
        messages.extend(self.conversation_history[-10:])

        # Generate
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p
        )

        response_text = response['choices'][0]['message']['content']

        # Handle function calls if present
        if use_tools and self.tools:
            response_text = self._handle_function_calls(response_text)

        return response_text

    def _format_messages_for_model(self, messages: List[Dict]) -> str:
        """Format messages for specific model"""
        # TinyLlama / Llama format
        prompt = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == "system":
                prompt += f"<|system|>\n{content}</s>\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}</s>\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}</s>\n"

        prompt += "<|assistant|>\n"
        return prompt

    def _handle_function_calls(self, response: str) -> str:
        """Parse and execute function calls from LLM response"""
        # Simple parsing for function calls
        # Format: FUNCTION: function_name(arg1=value1, arg2=value2)

        if "FUNCTION:" not in response:
            return response

        # Extract function call
        lines = response.split('\n')
        result_lines = []

        for line in lines:
            if line.startswith("FUNCTION:"):
                # Parse function call
                func_call = line.replace("FUNCTION:", "").strip()
                result = self._execute_function(func_call)
                result_lines.append(f"RESULT: {json.dumps(result)}")
            else:
                result_lines.append(line)

        return '\n'.join(result_lines)

    def _execute_function(self, func_call: str) -> Any:
        """Execute a function call"""
        try:
            # Parse function name and args (simplified)
            func_name = func_call.split('(')[0]

            # Call the appropriate tool method
            if hasattr(self.tools, func_name):
                method = getattr(self.tools, func_name)
                # Simple execution (full parsing would handle args)
                return method()
            else:
                return {"error": f"Unknown function: {func_name}"}

        except Exception as e:
            return {"error": str(e)}

    async def monitor_simulation(self):
        """Continuously monitor simulation and alert on issues"""
        self.monitoring_active = True

        logger.info("Starting simulation monitoring...")

        while self.monitoring_active:
            try:
                # Get current state
                state = self.tools.get_simulation_state()

                # Check for issues
                alerts = self._check_for_alerts(state)

                if alerts:
                    for alert in alerts:
                        logger.warning(f"ALERT: {alert}")
                        # Could send alert to user via WebSocket

                # Wait before next check
                await asyncio.sleep(self.config.monitor_interval)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5.0)

    def _check_for_alerts(self, state: Dict) -> List[str]:
        """Check simulation state for issues"""
        alerts = []
        metrics = state.get('metrics', {})

        # Check for divergence
        if 'max_field_value' in metrics:
            if metrics['max_field_value'] > self.config.alert_thresholds['divergence']:
                alerts.append("Field values diverging - possible numerical instability")

        # Check for NaNs
        if 'nan_fraction' in metrics:
            if metrics['nan_fraction'] > self.config.alert_thresholds['nan_fraction']:
                alerts.append(f"High NaN fraction: {metrics['nan_fraction']:.2%}")

        # Check correlation drop
        if 'correlation' in metrics:
            if metrics['correlation'] < self.config.alert_thresholds['correlation_drop']:
                alerts.append(f"Correlation dropped to {metrics['correlation']:.3f}")

        return alerts

    def generate_white_paper_with_llm(self, simulation_data: Dict) -> str:
        """
        Generate white paper using the LLM (instead of Claude)

        Args:
            simulation_data: Dictionary with simulation results and analysis

        Returns:
            LaTeX source code for white paper
        """
        # Create a detailed prompt for white paper generation
        prompt = f"""Generate a comprehensive scientific white paper for this iVHL simulation.

Simulation Data:
{json.dumps(simulation_data, indent=2)[:2000]}  # Truncate if too long

Requirements:
1. Title and abstract
2. Introduction to holographic resonance, GFT, and tensor networks
3. Methods section describing simulation setup
4. Results section with key findings
5. Analysis and discussion
6. Conclusions and future work
7. References (Maldacena, Ryu-Takayanagi, LIGO, Oriti, etc.)

Format: Complete LaTeX document suitable for compilation with pdflatex.

Include:
- Proper equation formatting (\\begin{{equation}} ... \\end{{equation}})
- Tables for numerical results
- Citations in \\cite{{}} format
- Professional academic tone

Begin LaTeX document:
"""

        # Generate with increased max_tokens for full paper
        old_max_tokens = self.config.max_tokens
        self.config.max_tokens = 8192  # Allow long generation

        response = self.chat(prompt, use_tools=False)

        self.config.max_tokens = old_max_tokens

        return response

    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.monitoring_active = False
        logger.info("Monitoring stopped")
