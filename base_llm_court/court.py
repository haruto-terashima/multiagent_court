from agents import ProsecutorAgent, DefenseAgent, JudgeAgent
from google import genai
import os
import time

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class LLMCourt:
    def __init__(self):
        self.history = []

    def run(self, case_text: str):

        print("\n===== CASE START =====\n")

        # =========================
        # ① 原告
        # =========================
        p1 = ProsecutorAgent.run(
            case_text=case_text,
            history=self.history,
            stage="訴状"
        )

        self.history.append({"role": "prosecutor", "content": p1})
        print("【原告①】\n", p1, "\n")

        time.sleep(1.5)

        # =========================
        # ② 被告
        # =========================
        d1 = DefenseAgent.run(
            case_text=case_text,
            history=self.history,
            stage="答弁書"
        )

        self.history.append({"role": "defense", "content": d1})
        print("【被告①】\n", d1, "\n")

        time.sleep(1.5)

        # =========================
        # ③ 争点整理（軽量・単発）
        # =========================
        issue_prompt = f"""
あなたは裁判官です。
以下の事実と主張から法的争点を簡潔に抽出してください。

【事実】
{case_text}

【原告】
{p1}

【被告】
{d1}

出力：
- 争点1
- 争点2
"""

        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=issue_prompt
        )

        issues_text = res.text.strip()
        print("【争点】\n", issues_text, "\n")

        self.history.append({"role": "judge_issues", "content": issues_text})

        time.sleep(1.5)

        # =========================
        # ④ 最終主張
        # =========================
        p2 = ProsecutorAgent.run(
            case_text=case_text,
            history=self.history,
            stage=f"最終準備書面（争点：{issues_text}）"
        )

        self.history.append({"role": "prosecutor_final", "content": p2})

        time.sleep(1.5)

        d2 = DefenseAgent.run(
            case_text=case_text,
            history=self.history,
            stage=f"最終反論（争点：{issues_text}）"
        )

        self.history.append({"role": "defense_final", "content": d2})

        print("【原告②】\n", p2, "\n")
        print("【被告②】\n", d2, "\n")

        time.sleep(1.5)

        # =========================
        # ⑤ 判決（最重要）
        # =========================
        judgment_prompt = f"""
あなたは裁判官です。
以下の情報に基づき、論理的に判決を下してください。

【事実】
{case_text}

【争点】
{issues_text}

【原告最終主張】
{p2}

【被告最終主張】
{d2}

判決理由と結論を明確に述べてください。
"""

        judgment_res = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=judgment_prompt
        )

        judgment = judgment_res.text.strip()

        self.history.append({"role": "judge", "content": judgment})

        print("【判決】\n", judgment)

        print("\n===== CASE END =====\n")

        return {
            "case": case_text,
            "issues": issues_text,
            "judgment": judgment,
            "history": self.history
        }