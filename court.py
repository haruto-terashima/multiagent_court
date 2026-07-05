from __future__ import annotations

import os

from google import genai

from agents import DefenseAgent, JudgeAgent, ProsecutorAgent
from rag import RAGEngine


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class LLMCourt:
    def __init__(self, rag: RAGEngine | None = None):
        self.rag = rag or RAGEngine()
        self.history = []

    def retrieve_context(self, query: str, k: int = 5, source_type: str | None = None) -> str:
        return self.rag.retrieve_text(query=query, k=k, source_type=source_type)

    def run(self, case_text: str):
        print("\n===== CASE START =====\n")

        p_context = self.retrieve_context(
            query=f"原告の請求を基礎づける法律要件、有利な判例、立証責任: {case_text}",
            k=6,
        )
        p1 = ProsecutorAgent.run(
            case_text=case_text,
            context=p_context,
            history=self.history,
            stage="訴状",
        )
        self.history.append({"role": "prosecutor", "content": p1})
        print("【原告①】\n", p1, "\n")

        d_context = self.retrieve_context(
            query=f"被告の抗弁、請求棄却、反論に使える法律要件と判例: {case_text}",
            k=6,
        )
        d1 = DefenseAgent.run(
            case_text=case_text,
            context=d_context,
            history=self.history,
            stage="答弁書",
        )
        self.history.append({"role": "defense", "content": d1})
        print("【被告①】\n", d1, "\n")

        judge_context = self.retrieve_context(
            query=f"民事裁判の争点整理、法律要件、判断枠組み: {case_text}",
            k=8,
        )
        issues_text = self.identify_issues(case_text, judge_context)
        self.history.append({"role": "judge_issues", "content": issues_text})
        print("【争点】\n", issues_text, "\n")

        full_context = self.retrieve_issue_contexts(issues_text)

        p2 = ProsecutorAgent.run(
            case_text=case_text,
            context=full_context,
            history=self.history,
            stage="準備書面（最終主張）",
        )
        self.history.append({"role": "prosecutor_final", "content": p2})

        d2 = DefenseAgent.run(
            case_text=case_text,
            context=full_context,
            history=self.history,
            stage="最終反論",
        )
        self.history.append({"role": "defense_final", "content": d2})

        print("【原告②】\n", p2, "\n")
        print("【被告②】\n", d2, "\n")

        judgment = JudgeAgent.run(
            case_text=case_text,
            context=full_context,
            history=self.history,
            stage="判決",
        )
        self.history.append({"role": "judge", "content": judgment})
        print("【判決】\n", judgment)

        print("\n===== CASE END =====\n")

        return {
            "case": case_text,
            "issues": issues_text,
            "judgment": judgment,
            "history": self.history,
        }

    def identify_issues(self, case_text: str, context: str) -> str:
        history_text = "\n\n".join(f"{h['role']}: {h['content']}" for h in self.history)
        prompt = f"""
あなたは裁判官です。
以下の情報をもとに、民事裁判の法律要件ベースで争点を整理してください。

# 事件
{case_text}

# 原告・被告の主張
{history_text}

# 参照すべき法令・判例
{context}

# 出力
- 争点1（どの法律要件か）
- 争点2（どの法律要件か）
"""
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return res.text.strip()

    def retrieve_issue_contexts(self, issues_text: str) -> str:
        contexts = []

        for line in issues_text.splitlines():
            issue = line.strip().lstrip("-").strip()
            if not issue:
                continue

            docs = self.retrieve_context(
                query=f"争点に関係する法律要件、判例、判断基準: {issue}",
                k=5,
            )
            contexts.append(f"## {issue}\n{docs}")

        return "\n\n".join(contexts)
