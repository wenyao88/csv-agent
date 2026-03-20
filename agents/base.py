from dataclasses import dataclass, field
from typing import Any
import plotly.graph_objects as go


@dataclass
class AgentResult:
    status: str = "success"
    output: str = ""
    data: Any = None
    figs: list = field(default_factory=list)
    error_msg: str = ""

    @property
    def success(self):
        return self.status == "success"

    @property
    def failed(self):
        return self.status == "error"


def make_error(msg: str) -> AgentResult:
    return AgentResult(status="error", error_msg=msg, output=f"执行失败：{msg}")


def make_success(output: str, data=None, figs=None) -> AgentResult:
    return AgentResult(status="success", output=output, data=data, figs=figs or [])