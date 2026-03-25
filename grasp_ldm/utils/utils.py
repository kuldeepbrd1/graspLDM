import json


def load_json(path: str) -> dict:
    """Load a JSON file and return its contents as a dict."""
    with open(path, "r") as jf:
        return json.load(jf)
