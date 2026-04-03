#!/usr/bin/env python3
"""
Run DeLLMa-style decision on your own question.

Reads a question from a .txt file (see HOW_TO_WRITE_QUESTION.md), builds a
prompt, and calls ChatGPT to get a recommended action and explanation.

Usage (from project root):
  python reproduction_visualization/run_my_question.py [question_file.txt]

Default question file: reproduction_visualization/question.txt
Requires: OPENAI_API_KEY set in the environment.
"""

import os
import sys
import re

# Run from project root: add parent so we can import utils and use inference
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.prompt_utils import inference, format_query


def parse_question_file(path: str) -> dict:
    """Parse a question .txt with sections [GOAL], [ACTIONS], [CONTEXT].
    Returns dict with keys 'goal', 'actions' (list of str), 'context'.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    sections = {}
    current = None
    current_lines = []

    for line in text.splitlines():
        if line.strip() == "[GOAL]":
            if current is not None:
                sections[current] = "\n".join(current_lines).strip()
            current = "goal"
            current_lines = []
        elif line.strip() == "[ACTIONS]":
            if current is not None:
                sections[current] = "\n".join(current_lines).strip()
            current = "actions"
            current_lines = []
        elif line.strip() == "[CONTEXT]":
            if current is not None:
                sections[current] = "\n".join(current_lines).strip()
            current = "context"
            current_lines = []
        else:
            current_lines.append(line)

    if current is not None:
        sections[current] = "\n".join(current_lines).strip()

    actions_str = sections.get("actions", "")
    actions = [a.strip() for a in actions_str.split("\n") if a.strip()]

    return {
        "goal": sections.get("goal", ""),
        "actions": actions,
        "context": sections.get("context", ""),
    }


def build_prompt(goal: str, actions: list, context: str) -> str:
    """Build the user prompt for the decision task."""
    lines = []
    if context:
        lines.append("Context:")
        lines.append(context)
        lines.append("")
    lines.append("Goal:")
    lines.append(goal)
    lines.append("")
    lines.append("Actions (choose exactly one):")
    for i, a in enumerate(actions, 1):
        lines.append(f"  {i}. {a}")
    lines.append("")
    lines.append("Which action do you recommend based on the context and goal?")
    return "\n".join(lines)


def main():
    default_question = os.path.join(SCRIPT_DIR, "question.txt")
    question_path = sys.argv[1] if len(sys.argv) > 1 else default_question

    if not os.path.isfile(question_path):
        print(f"Error: Question file not found: {question_path}")
        print("See reproduction_visualization/HOW_TO_WRITE_QUESTION.md for format.")
        sys.exit(1)

    q = parse_question_file(question_path)
    if not q["actions"]:
        print("Error: No actions found. Add at least one line under [ACTIONS].")
        sys.exit(1)

    prompt = build_prompt(q["goal"], q["actions"], q["context"])
    format_instruction = (
        "You should format your response as a JSON object with exactly these keys:\n"
        "- decision: a string that states the recommended action (use the same wording as in the actions list).\n"
        "- explanation: a string that explains your reasoning in detail."
    )
    full_query = format_query(prompt, format_instruction=format_instruction)

    system_content = (
        "You are a helpful decision-making assistant. "
        "Given the user's goal, context, and list of actions, recommend one action and explain your reasoning clearly."
    )

    print("Calling ChatGPT (DeLLMa-style)...")
    response = inference(full_query, system_content=system_content, temperature=0.0)

    print("\n--- Response ---")
    if isinstance(response, dict):
        for k, v in response.items():
            print(f"{k}: {v}")
    else:
        print(response)


if __name__ == "__main__":
    main()
