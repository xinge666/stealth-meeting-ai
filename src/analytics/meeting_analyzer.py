"""
Meeting Analyzer — analyzes session history to generate a debrief report.
Focuses on technical gaps, strengths, and actionable learning points.
"""

import logging
import os
from datetime import datetime
from typing import List
from ..context.schema import ConversationTurn

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

        prompt = f"""
你是一个顶级技术面试官与职业发展教练。请对以下这场“面试/会议”的对话记录进行深度复盘。

[对话全文]:
---
{full_text}
---

请生成一份专业且有温度的复盘报告，包含以下板块：

1. **🏆 闪光点总结**：
   - 候选人在哪些技术点上回答得非常出色？
   - 表达逻辑和专业度如何？

2. **🚩 技术短板识别 (需重点关注)**：
   - 哪些提问候选人回答得不够深入或含糊？
   - 识别出具体的知识盲区（结合对方提问及候选人回答缺失的部分）。

3. **📝 未能完全解答的问题**：
   - 列出面试官提出的核心提问，并标注候选人当时是否给出了满意的答案。

4. **🚀 成长建议与学习路线**：
   - 针对上述短板，给出具体的学习路线图和关键词（如：阅读某某文档、理解某某底层原理）。

5. **👀 辅导提示对比**：
   - AI 助手给出的提示是否被候选人有效利用了？

请直接输出 Markdown 格式的报告，内容要精准、专业。
"""
        logger.info("Generating post-meeting analysis report...")
        report = await self.llm.ask(prompt, "复盘分析请求", stream=False)
        
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
