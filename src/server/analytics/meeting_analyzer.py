"""
Meeting Analyzer — analyzes session history to generate a debrief report.
Focuses on technical gaps, strengths, and actionable learning points.
"""

import logging
import os
from datetime import datetime
from typing import List
from ..context.schema import ConversationTurn
from ..prompts.templates import MEETING_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)

class MeetingAnalyzer:
    """
    Analyzes conversation history using LLM to generate a performance report.
    """

    def __init__(self, llm_client: "LLMClient"):
        self.llm = llm_client

    async def analyze(self, history: List[ConversationTurn]) -> str:
        """
        Processes history and generates a Markdown report.
        """
        if not history:
            return "没有捕获到有效的对话，无法生成复盘报告。"

        # Format history for analysis
        history_text = []
        for turn in history:
            role = "【候选人/我】" if turn.speaker == "self" else "【面试官/对方】"
            if turn.speaker == "ai":
                role = "【AI 助手】"
            history_text.append(f"{role}: {turn.text}")
        
        full_text = "\n".join(history_text)

        prompt = MEETING_ANALYSIS_PROMPT.format(full_text=full_text)
        logger.info("Generating post-meeting analysis report...")
        report = await self.llm.ask(prompt, "复盘分析请求")
        
        # Save to disk as well
        self._save_report(report)
        return report

    def _save_report(self, content: str):
        """Save report to the reports/ directory."""
        os.makedirs("reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"reports/meeting_report_{timestamp}.md"
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("Report saved to %s", filepath)
        except Exception as e:
            logger.error("Failed to save report: %s", e)
