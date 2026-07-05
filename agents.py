from google import genai
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class BaseAgent:
    def __init__(self, role_prompt, model="gemini-2.5-flash"):
        self.role_prompt = role_prompt
        self.model = model

    def run(self, case_text: str, context: str, history=None, stage=""):
        history = history or []

        history_text = ""
        for h in history:
            history_text += f"{h['role']}: {h['content']}\n\n"

        prompt = f"""
{self.role_prompt}

# 事件
{case_text}

# フェーズ
{stage}

# これまでのやり取り
{history_text}

# 参照すべき法令・判例
{context}
"""

        res = client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return res.text.strip()


# ======================
# 役割定義
# ======================

ProsecutorAgent = BaseAgent(
    "あなたは原告代理人弁護士。原告の請求を最大限法的に正当化する。"
)

DefenseAgent = BaseAgent(
    "あなたは被告代理人弁護士。請求を否定し、責任を回避する論理を構築する。"
)

JudgeAgent = BaseAgent(
    "あなたは裁判官。中立に事実と主張を整理し、争点を確定し、最終的に判決を下す。",
    model="gemini-2.5-pro",
)
