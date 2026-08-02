"""
==========================================================
Lightweight Text-to-Pandas Analytics Engine
==========================================================
Loads every CSV/XLSX in data_dir into a DataFrame, and answers
quantitative questions ("average X", "how many rows where Y",
"total Z per month") by asking the LLM to generate a short
pandas snippet, executing it against the loaded frame(s), and
returning the result in a UI-friendly shape (scalar/dataframe).

SECURITY NOTE: this executes LLM-generated Python via exec().
The sandbox below strips dangerous tokens (imports, dunders,
file/network/process access) and restricts builtins, which
covers casual misuse, but exec() of model output is inherently
risk-bearing and should not be exposed to untrusted end users
in a production deployment without a real sandboxing layer
(subprocess with resource limits, a proper code-execution
service, etc). Treat this as a prototype-grade convenience,
not a hardened boundary.
==========================================================
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from logger import logger

_BLOCKED_TOKENS = (
    "import ", "__", "open(", "exec(", "eval(", "os.", "sys.", "subprocess",
    "socket", "requests", "shutil", "pathlib", "input(", "compile(",
)

_CODE_PROMPT = ChatPromptTemplate.from_template(
    "You are a pandas code generator. You are given one or more DataFrames "
    "already loaded into variables (see names below). Write a SHORT pandas "
    "expression/snippet that computes the answer to the user's question and "
    "assigns it to a variable called `result`. Output ONLY python code, no "
    "explanation, no markdown fences, no imports (pandas is already available "
    "as `pd`).\n\n"
    "Available DataFrames:\n{schemas}\n\n"
    "Question: {question}\n\n"
    "Python code:"
)


class AnalyticsEngine:
    def __init__(self, data_dir, llm):
        self.data_dir = Path(data_dir)
        self.llm = llm
        self.dataframes: dict[str, pd.DataFrame] = {}
        self._code_chain = _CODE_PROMPT | self.llm | StrOutputParser()
        self._load()

    def _load(self):
        if not self.data_dir.exists():
            return
        for f in sorted(self.data_dir.glob("*")):
            try:
                if f.suffix.lower() == ".csv":
                    self.dataframes[f.name] = pd.read_csv(f)
                elif f.suffix.lower() in (".xlsx", ".xls"):
                    self.dataframes[f.name] = pd.read_excel(f)
            except Exception as e:
                logger.warning(f"AnalyticsEngine: could not load {f.name}: {e}")
        if self.dataframes:
            logger.info(f"AnalyticsEngine loaded {len(self.dataframes)} tabular file(s): "
                        f"{', '.join(self.dataframes.keys())}")

    def is_analytical(self, question: str) -> bool:
        """No LLM call here on purpose -- the caller (rag_engine's intent
        classifier) already gates entry with a keyword prefilter before
        calling this, so this just needs to confirm we actually have data
        to analyze."""
        return bool(self.dataframes)

    def _schema_summary(self) -> str:
        parts = []
        for name, df in self.dataframes.items():
            cols = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns[:20])
            parts.append(f"- df_{self._safe_var(name)}  (from {name}, {len(df)} rows): {cols}")
        return "\n".join(parts)

    @staticmethod
    def _safe_var(filename: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", Path(filename).stem)

    def run(self, question: str) -> dict:
        if not self.dataframes:
            return {"success": False, "error": "No tabular (CSV/XLSX) files are indexed.", "answer": "I don't have any spreadsheet data loaded to analyze."}

        try:
            code = self._code_chain.invoke({"schemas": self._schema_summary(), "question": question}).strip()
            code = re.sub(r"^```(?:python)?|```$", "", code, flags=re.MULTILINE).strip()
        except Exception as e:
            return {"success": False, "error": f"Code generation failed: {e}", "answer": "I couldn't work out how to analyze that."}

        lowered = code.lower()
        if any(tok in lowered for tok in _BLOCKED_TOKENS):
            logger.error(f"AnalyticsEngine: blocked unsafe generated code:\n{code}")
            return {
                "success": False, "error": "Generated code failed a safety check.",
                "answer": "I couldn't safely run that analysis.", "generated_code": code,
            }

        local_env = {f"df_{self._safe_var(name)}": df for name, df in self.dataframes.items()}
        local_env["pd"] = pd
        safe_builtins = {
            "len": len, "sum": sum, "min": min, "max": max, "round": round,
            "range": range, "sorted": sorted, "list": list, "dict": dict, "str": str,
            "int": int, "float": float, "abs": abs, "enumerate": enumerate,
        }

        try:
            exec(code, {"__builtins__": safe_builtins}, local_env)
            value = local_env.get("result")
        except Exception as e:
            logger.error(f"AnalyticsEngine: execution failed: {e}\nCode was:\n{code}")
            return {
                "success": False, "error": str(e), "answer": "That analysis hit an error while running.",
                "generated_code": code,
            }

        filename = ", ".join(self.dataframes.keys())

        if isinstance(value, pd.DataFrame):
            return {
                "success": True, "result_type": "dataframe", "value": value,
                "filename": filename, "generated_code": code,
                "answer": f"Here's the result ({len(value)} row(s)).",
            }
        if isinstance(value, pd.Series):
            return {
                "success": True, "result_type": "series", "value": value,
                "filename": filename, "generated_code": code,
                "answer": f"Here's the result ({len(value)} row(s)).",
            }
        if value is not None:
            return {
                "success": True, "result_type": "scalar", "value": value,
                "filename": filename, "generated_code": code,
                "answer": f"Result: {value}",
            }

        return {
            "success": False, "error": "No `result` variable was produced.",
            "answer": "I wasn't able to compute a result for that.", "generated_code": code,
        }