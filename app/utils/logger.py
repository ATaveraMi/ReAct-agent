import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ReactLogger:
    def __init__(self, log_path: str, *, echo_to_stdout: bool = True) -> None:
        self.log_path = log_path
        self.echo_to_stdout = echo_to_stdout
        Path(os.path.dirname(self.log_path)).mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        thought: Optional[str] = None,
        action: Optional[str] = None,
        observation: Optional[str] = None,
        final_answer: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "thought": thought,
            "action": action,
            "observation": observation,
            "final_answer": final_answer,
            "metadata": metadata or {},
        }
        line = json.dumps(entry, ensure_ascii=False)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.echo_to_stdout:
            print(line)
