"""Load prompt files and fill {variables}. str.replace is used (not str.format) because the
prompts contain JSON braces."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_prompt(prompt_file, version="final"):
    text = (ROOT / "prompts" / version / prompt_file).read_text(encoding="utf-8")
    system, user = "", text
    if "### USER" in text:
        head, user = text.split("### USER", 1)
        system = head.replace("### SYSTEM", "").strip()
    return system, user.strip()


def render(template, **variables):
    for key, value in variables.items():
        template = template.replace("{" + key + "}", str(value))
    return template
