"""Benchmarks for evaluating AI agents."""
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel

from lm_evals.core.benchmark import Benchmark, BenchmarkResult
from lm_evals.core.model import BaseModel as LMModel

class AgentSystem(BaseModel):
    """Interface for agent systems being evaluated."""
    
    def plan(self, task: str) -> List[str]:
        """Plan steps to complete the task.
        
        Args:
            task: Task description
            
        Returns:
            List of planned steps
        """
        raise NotImplementedError
        
    def execute(self, task: str, plan: List[str]) -> Dict[str, Any]:
        """Execute planned steps for the task.
        
        Args:
            task: Task description
            plan: List of planned steps
            
        Returns:
            Execution results including success status and outputs
        """
        raise NotImplementedError
        
    def use_tool(self, tool_name: str, **kwargs) -> Any:
        """Use a specific tool.
        
        Args:
            tool_name: Name of tool to use
            **kwargs: Tool-specific parameters
            
        Returns:
            Tool execution results
        """
        raise NotImplementedError

class AgentBenchmark(Benchmark):
    """Base class for agent system benchmarks."""
    
    def __init__(self, agent_system: AgentSystem, **kwargs):
        """Initialize agent benchmark.
        
        Args:
            agent_system: Agent system to evaluate
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.agent = agent_system

class TaskCompletion(AgentBenchmark):
    """Benchmark for measuring task completion success."""
    
    def run(self, model: LMModel) -> BenchmarkResult:
        """Evaluate task completion.
        
        Args:
            model: Language model for scoring task completion
            
        Returns:
            Task completion metrics
        """
        # Example evaluation tasks
        eval_tasks = [
            "Find the latest stock price for Apple Inc.",
            "Summarize the contents of the given PDF file",
            "Book a meeting with John for tomorrow at 2 PM"
        ]
        
        completion_scores = []
        success_rate = 0
        
        for task in eval_tasks:
            # Get agent's plan and execution
            plan = self.agent.plan(task)
            result = self.agent.execute(task, plan)
            
            # Score task completion
            prompt = (
                f"Task: {task}\n"
                f"Plan:\n{chr(10).join(plan)}\n"
                f"Execution Result: {result}\n"
                "On a scale of 0-10:\n"
                "1. How well was the task completed?\n"
                "2. Were all necessary steps executed correctly?\n"
                "3. Was the final outcome satisfactory?\n"
                "Overall score:"
            )
            score = float(model.generate(prompt, max_tokens=2).strip())
            completion_scores.append(score / 10.0)
            
            # Binary success metric
            success_rate += int(result.get("success", False))
            
        return BenchmarkResult(
            metrics={
                "mean_completion_score": np.mean(completion_scores),
                "success_rate": success_rate / len(eval_tasks)
            }
        )
    
    @property
    def name(self) -> str:
        return "task_completion"
    
    @property
    def description(self) -> str:
        return "Measures success rate and quality of task completion"

class ToolUsage(AgentBenchmark):
    """Benchmark for measuring effective tool usage."""
    
    def run(self, model: LMModel) -> BenchmarkResult:
        """Evaluate tool usage effectiveness.
        
        Args:
            model: Language model for scoring tool usage
            
        Returns:
            Tool usage metrics
        """
        # Example tool usage scenarios
        eval_scenarios = [
            {
                "task": "Calculate the square root of 256",
                "required_tool": "calculator",
                "expected_result": 16
            },
            {
                "task": "What's the weather in London?",
                "required_tool": "weather_api",
                "expected_params": {"city": "London"}
            },
            {
                "task": "Translate 'Hello' to French",
                "required_tool": "translator",
                "expected_params": {"text": "Hello", "target_lang": "fr"}
            }
        ]
        
        tool_scores = []
        correct_tool_rate = 0
        
        for scenario in eval_scenarios:
            # Get agent's tool usage
            plan = self.agent.plan(scenario["task"])
            result = self.agent.execute(scenario["task"], plan)
            
            # Check if correct tool was used
            used_tools = result.get("used_tools", [])
            correct_tool = scenario["required_tool"] in used_tools
            correct_tool_rate += int(correct_tool)
            
            # Score tool usage effectiveness
            prompt = (
                f"Task: {scenario['task']}\n"
                f"Required Tool: {scenario['required_tool']}\n"
                f"Tools Used: {', '.join(used_tools)}\n"
                f"Result: {result}\n"
                "On a scale of 0-10:\n"
                "1. Was the correct tool selected?\n"
                "2. Were tool parameters appropriate?\n"
                "3. Was the tool output used effectively?\n"
                "Overall score:"
            )
            score = float(model.generate(prompt, max_tokens=2).strip())
            tool_scores.append(score / 10.0)
            
        return BenchmarkResult(
            metrics={
                "mean_tool_score": np.mean(tool_scores),
                "correct_tool_rate": correct_tool_rate / len(eval_scenarios)
            }
        )
    
    @property
    def name(self) -> str:
        return "tool_usage"
    
    @property
    def description(self) -> str:
        return "Measures effectiveness of tool selection and usage" 
