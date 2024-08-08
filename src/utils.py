from pathlib import Path
import json


def load_json(path: str) -> dict:
    with Path(path).open("r") as handle:
        return json.load(handle)
