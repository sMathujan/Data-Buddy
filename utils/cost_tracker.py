"""
Cost Tracking Utility for CrewAI Tabular Agent
Monitors and tracks LLM usage costs for transparency
"""

import time
import tiktoken
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class LLMUsage:
    """Track individual LLM API call usage"""
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    agent_name: str
    task_name: str
    duration: float = 0.0

@dataclass
class SessionCosts:
    """Track costs for an entire session"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_cost: float = 0.0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    api_calls: List[LLMUsage] = field(default_factory=list)
    
    def add_usage(self, usage: LLMUsage):
        """Add a new LLM usage record"""
        self.api_calls.append(usage)
        self.total_cost += usage.cost
        self.total_tokens += usage.total_tokens
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        
    def finalize_session(self):
        """Mark the session as complete"""
        self.end_time = datetime.now()

class CostTracker:
    """Tracks and calculates LLM usage costs"""
    
    # OpenAI pricing (as of 2024) - prices per 1K tokens
    PRICING = {
        'gpt-4o-mini': {
            'prompt': 0.00015,  # $0.15 per 1M tokens
            'completion': 0.0006  # $0.60 per 1M tokens
        },
        'gpt-4o': {
            'prompt': 0.005,  # $5.00 per 1M tokens
            'completion': 0.015  # $15.00 per 1M tokens
        },
        'gpt-4-turbo': {
            'prompt': 0.01,   # $10.00 per 1M tokens
            'completion': 0.03  # $30.00 per 1M tokens
        },
        'gpt-3.5-turbo': {
            'prompt': 0.0005,  # $0.50 per 1M tokens
            'completion': 0.0015  # $1.50 per 1M tokens
        }
    }
    
    def __init__(self):
        self.sessions: Dict[str, SessionCosts] = {}
        self.current_session_id: Optional[str] = None
        
    def start_session(self, session_id: str) -> str:
        """Start a new cost tracking session"""
        self.current_session_id = session_id
        self.sessions[session_id] = SessionCosts(
            session_id=session_id,
            start_time=datetime.now()
        )
        logger.info(f"Started cost tracking session: {session_id}")
        return session_id
        
    def end_session(self, session_id: Optional[str] = None) -> SessionCosts:
        """End a cost tracking session and return summary"""
        session_id = session_id or self.current_session_id
        if session_id and session_id in self.sessions:
            self.sessions[session_id].finalize_session()
            logger.info(f"Ended cost tracking session: {session_id}")
            return self.sessions[session_id]
        return None
        
    def count_tokens(self, text: str, model: str = "gpt-4o-mini") -> int:
        """Count tokens in text using tiktoken"""
        try:
            # Map model names to tiktoken encodings
            encoding_map = {
                'gpt-4o-mini': 'o200k_base',
                'gpt-4o': 'o200k_base',
                'gpt-4-turbo': 'cl100k_base',
                'gpt-3.5-turbo': 'cl100k_base'
            }
            
            encoding_name = encoding_map.get(model, 'cl100k_base')
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}. Using estimate.")
            # Fallback: rough estimate of 4 characters per token
            return len(text) // 4
            
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Calculate cost for token usage"""
        model_key = model.replace('openai/', '').replace('/', '-')
        
        if model_key not in self.PRICING:
            logger.warning(f"Unknown model for pricing: {model_key}. Using gpt-4o-mini pricing.")
            model_key = 'gpt-4o-mini'
            
        pricing = self.PRICING[model_key]
        
        # Convert to cost (pricing is per 1K tokens, so divide by 1000)
        prompt_cost = (prompt_tokens / 1000) * pricing['prompt']
        completion_cost = (completion_tokens / 1000) * pricing['completion']
        
        return prompt_cost + completion_cost
        
    def track_usage(self, 
                   prompt: str,
                   completion: str,
                   model: str,
                   agent_name: str,
                   task_name: str,
                   duration: float = 0.0,
                   session_id: Optional[str] = None) -> LLMUsage:
        """Track LLM usage and calculate costs"""
        
        session_id = session_id or self.current_session_id
        
        # Count tokens
        prompt_tokens = self.count_tokens(prompt, model)
        completion_tokens = self.count_tokens(completion, model)
        total_tokens = prompt_tokens + completion_tokens
        
        # Calculate cost
        cost = self.calculate_cost(prompt_tokens, completion_tokens, model)
        
        # Create usage record
        usage = LLMUsage(
            timestamp=datetime.now(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            agent_name=agent_name,
            task_name=task_name,
            duration=duration
        )
        
        # Add to session if exists
        if session_id and session_id in self.sessions:
            self.sessions[session_id].add_usage(usage)
            
        logger.info(f"Tracked usage: {agent_name}/{task_name} - ${cost:.4f} ({total_tokens} tokens)")
        
        return usage
        
    def get_session_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get cost summary for a session"""
        session_id = session_id or self.current_session_id
        
        if not session_id or session_id not in self.sessions:
            return {"error": "Session not found"}
            
        session = self.sessions[session_id]
        
        # Calculate duration
        end_time = session.end_time or datetime.now()
        duration = (end_time - session.start_time).total_seconds()
        
        # Group by agent
        agent_costs = {}
        for usage in session.api_calls:
            if usage.agent_name not in agent_costs:
                agent_costs[usage.agent_name] = {
                    'cost': 0.0,
                    'tokens': 0,
                    'calls': 0
                }
            agent_costs[usage.agent_name]['cost'] += usage.cost
            agent_costs[usage.agent_name]['tokens'] += usage.total_tokens
            agent_costs[usage.agent_name]['calls'] += 1
            
        return {
            'session_id': session_id,
            'duration_seconds': duration,
            'total_cost': session.total_cost,
            'total_tokens': session.total_tokens,
            'total_prompt_tokens': session.total_prompt_tokens,
            'total_completion_tokens': session.total_completion_tokens,
            'total_api_calls': len(session.api_calls),
            'agent_breakdown': agent_costs,
            'cost_per_token': session.total_cost / session.total_tokens if session.total_tokens > 0 else 0,
            'start_time': session.start_time.isoformat(),
            'end_time': session.end_time.isoformat() if session.end_time else None
        }
        
    def format_cost_summary(self, session_id: Optional[str] = None) -> str:
        """Format cost summary as human-readable text"""
        summary = self.get_session_summary(session_id)
        
        if "error" in summary:
            return summary["error"]
            
        text = f"""
💰 LLM USAGE COST SUMMARY
========================

Session: {summary['session_id']}
Duration: {summary['duration_seconds']:.1f} seconds

📊 TOTALS:
• Total Cost: ${summary['total_cost']:.4f}
• Total Tokens: {summary['total_tokens']:,}
  - Prompt Tokens: {summary['total_prompt_tokens']:,}
  - Completion Tokens: {summary['total_completion_tokens']:,}
• API Calls: {summary['total_api_calls']}
• Cost per Token: ${summary['cost_per_token']:.6f}

🤖 AGENT BREAKDOWN:
"""
        
        for agent_name, stats in summary['agent_breakdown'].items():
            text += f"""• {agent_name}:
  - Cost: ${stats['cost']:.4f}
  - Tokens: {stats['tokens']:,}
  - Calls: {stats['calls']}
"""
        
        return text
        
    def export_session_data(self, session_id: Optional[str] = None) -> str:
        """Export session data as JSON"""
        session_id = session_id or self.current_session_id
        
        if not session_id or session_id not in self.sessions:
            return json.dumps({"error": "Session not found"})
            
        session = self.sessions[session_id]
        
        # Convert to serializable format
        data = {
            'session_id': session.session_id,
            'start_time': session.start_time.isoformat(),
            'end_time': session.end_time.isoformat() if session.end_time else None,
            'total_cost': session.total_cost,
            'total_tokens': session.total_tokens,
            'api_calls': []
        }
        
        for usage in session.api_calls:
            data['api_calls'].append({
                'timestamp': usage.timestamp.isoformat(),
                'model': usage.model,
                'prompt_tokens': usage.prompt_tokens,
                'completion_tokens': usage.completion_tokens,
                'total_tokens': usage.total_tokens,
                'cost': usage.cost,
                'agent_name': usage.agent_name,
                'task_name': usage.task_name,
                'duration': usage.duration
            })
            
        return json.dumps(data, indent=2)


def test_cost_tracker():
    """Test the cost tracking functionality"""
    tracker = CostTracker()
    
    # Start a session
    session_id = tracker.start_session("test_session")
    
    # Simulate some LLM usage
    tracker.track_usage(
        prompt="What is the average salary by department?",
        completion="SELECT department, AVG(salary) FROM employees GROUP BY department;",
        model="gpt-4o-mini",
        agent_name="sql_generator_agent",
        task_name="sql_generation_task",
        duration=2.5
    )
    
    tracker.track_usage(
        prompt="Review this SQL query for safety and correctness...",
        completion="The query looks safe and correct. It uses proper aggregation...",
        model="gpt-4o-mini",
        agent_name="sql_reviewer_agent",
        task_name="sql_review_task",
        duration=1.8
    )
    
    # End session and get summary
    tracker.end_session()
    
    print("Cost Summary:")
    print(tracker.format_cost_summary())
    
    print("\nJSON Export:")
    print(tracker.export_session_data())


if __name__ == "__main__":
    test_cost_tracker()

