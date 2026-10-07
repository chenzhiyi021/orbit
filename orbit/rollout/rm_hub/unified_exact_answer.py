# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Exact-answer rewards used by the unified RLVR experiments.

Ported from TRL's ``trl/rewards/unified_rewards.py`` (the M9 verifier GRPO reward)
so Orbit tools can score answers with exactly the same extractor and verifier.

The generic :func:`accuracy_reward` intentionally requires a boxed answer.  The
OpenMathReasoning prompts in our frozen mixture do not all contain that output
instruction, while OpenScienceReasoning prompts do.  This module therefore
uses a conservative three-stage extractor: the last complete ``\\boxed{}``, a
trailing explicit answer phrase, and finally a standalone last line.  It never
searches arbitrary expressions in the full chain of thought.
"""

import importlib.util
import logging
import re
import threading
from collections.abc import Callable
from functools import lru_cache
from typing import Any


def is_math_verify_available() -> bool:
    return importlib.util.find_spec("math_verify") is not None


if is_math_verify_available():
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify


_ANSWER_MARKER_RE = re.compile(r"(?is)(?:the\s+)?(?:final\s+answer|answer)\s*(?:is|=|:)?\s*")
_CHOICE_RE = re.compile(r"^[\s\$\\()\[\]{}]*([A-Za-z])[\s\$\\().,;:\[\]{}]*$")
_LATEX_CHOICE_PREFIX_RE = re.compile(
    r"^[\s\*]*(?:\$+)?\\(?:text|mathrm|mathbf)\s*\{\s*([A-Za-z])(?:\s*[:.)-]|\s*\})",
    re.DOTALL,
)
_PLAIN_CHOICE_PREFIX_RE = re.compile(
    r"^[\s\$\*`_()\[\]{}]*(?:option\s+)?([A-Za-z])(?:\s*[:.)-]|\b)",
    re.IGNORECASE | re.DOTALL,
)
_LATEX_OR_MATH_WORDS = {
    "cos",
    "frac",
    "infinity",
    "infty",
    "ln",
    "log",
    "mathrm",
    "operatorname",
    "pi",
    "pm",
    "sin",
    "sqrt",
    "tan",
    "text",
}


def _completion_content(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)


def _last_complete_boxed(text: str) -> str | None:
    """Return the contents of the last balanced ``\\boxed{...}``/``\\fbox{...}``."""

    candidates: list[str] = []
    for marker in (r"\boxed{", r"\fbox{"):
        start = 0
        while True:
            marker_index = text.find(marker, start)
            if marker_index < 0:
                break
            content_start = marker_index + len(marker)
            depth = 1
            escaped = False
            for index in range(content_start, len(text)):
                char = text[index]
                if char == "\\" and not escaped:
                    escaped = True
                    continue
                if char == "{" and not escaped:
                    depth += 1
                elif char == "}" and not escaped:
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[content_start:index])
                        start = index + 1
                        break
                escaped = False
            else:
                start = content_start
    return candidates[-1].strip() if candidates else None


def _explicit_answer_suffix(text: str) -> str | None:
    matches = list(_ANSWER_MARKER_RE.finditer(text))
    if not matches:
        return None
    suffix = text[matches[-1].end() :].strip()
    return suffix or None


def _last_nonempty_line(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def _standalone_math_line(text: str) -> bool:
    """Reject prose lines so fallback parsing cannot mine numbers from reasoning."""

    if len(text) > 512:
        return False
    words = {word.lower() for word in re.findall(r"[A-Za-z]{2,}", text)}
    return words.issubset(_LATEX_OR_MATH_WORDS)


def _canonical_choice(text: str) -> str | None:
    match = _CHOICE_RE.fullmatch(text.strip())
    return match.group(1).upper() if match else None


def _choice_prefix(text: str) -> str | None:
    """Extract an option prefix from an already trusted answer-only span."""

    for pattern in (_LATEX_CHOICE_PREFIX_RE, _PLAIN_CHOICE_PREFIX_RE):
        match = pattern.search(text.strip())
        if match:
            return match.group(1).upper()
    return None


def _extract_choice(text: str) -> tuple[str | None, str]:
    boxed = _last_complete_boxed(text)
    if boxed is not None:
        choice = _choice_prefix(boxed)
        if choice is not None:
            return choice, "boxed"
    suffix = _explicit_answer_suffix(text)
    if suffix is not None:
        first_line = suffix.splitlines()[0]
        choice = _choice_prefix(first_line)
        if choice is not None:
            return choice, "answer_marker"
    last_line = _last_nonempty_line(text)
    if last_line is not None:
        choice = _canonical_choice(last_line)
        if choice is not None:
            return choice, "last_line"
    return None, "unparseable"


def _math_extraction_configs():
    return [
        LatexExtractionConfig(
            normalization_config=NormalizationConfig(units=True, boxed="last"),
            boxed_match_priority=0,
            try_extract_without_anchor=True,
        ),
        ExprExtractionConfig(try_extract_without_anchor=True),
    ]


def _parse_math_candidate(candidate: str, parsing_timeout: int | None):
    return parse(
        candidate,
        extraction_config=_math_extraction_configs(),
        fallback_mode="first_match",
        extraction_mode="first_match",
        parsing_timeout=parsing_timeout,
    )


@lru_cache(maxsize=131_072)
def _parse_gold_math_cached(solution: str, parsing_timeout: int | None):
    return tuple(parse(solution, parsing_timeout=parsing_timeout))


def _extract_math(text: str, parsing_timeout: int | None) -> tuple[list[Any], str, str]:
    boxed = _last_complete_boxed(text)
    if boxed is not None:
        parsed = _parse_math_candidate(boxed, parsing_timeout)
        if parsed:
            return parsed, "boxed", boxed

    suffix = _explicit_answer_suffix(text)
    if suffix is not None:
        parsed = _parse_math_candidate(suffix, parsing_timeout)
        if parsed:
            return parsed, "answer_marker", suffix

    last_line = _last_nonempty_line(text)
    if last_line is not None and _standalone_math_line(last_line):
        parsed = _parse_math_candidate(last_line, parsing_timeout)
        if parsed:
            return parsed, "last_line", last_line

    return [], "unparseable", ""


def is_parseable_gold_answer(solution: str, answer_type: str) -> bool:
    """Return whether a gold answer can be scored by the exact verifier."""

    if answer_type == "choice":
        return _canonical_choice(solution) is not None
    if answer_type != "math" or not is_math_verify_available():
        return False
    try:
        return bool(_parse_gold_math_cached(solution, 5))
    except Exception:
        return False


def gold_answers_equivalent(first: str, second: str, answer_type: str) -> bool:
    """Return whether two gold strings are equivalent under the training verifier."""

    if answer_type == "choice":
        first_choice = _canonical_choice(first)
        second_choice = _canonical_choice(second)
        return first_choice is not None and first_choice == second_choice
    if answer_type != "math" or not is_math_verify_available():
        return False
    try:
        first_parsed = list(_parse_gold_math_cached(first, 5))
        second_parsed = list(_parse_gold_math_cached(second, 5))
        return bool(first_parsed and second_parsed and verify(first_parsed, second_parsed, timeout_seconds=5))
    except Exception:
        return False


def unified_exact_answer_reward(
    completions: list[Any],
    solution: list[str],
    answer_type: list[str],
    log_extra: Callable[[str, list], None] | None = None,
    log_metric: Callable[[str, float], None] | None = None,
    **kwargs,
) -> list[float | None]:
    """Binary exact-answer reward for symbolic/numeric math and categorical choices.

    A malformed gold answer returns ``None`` so GRPO excludes that row.  A
    malformed model answer returns ``0``.  Dataset construction audits golds,
    so a non-zero gold-skip rate during training is treated as a regression.
    """

    if not is_math_verify_available():
        raise ImportError("Please install the `math_verify` package to use unified_exact_answer_reward")
    if not (len(completions) == len(solution) == len(answer_type)):
        raise ValueError("completions, solution, and answer_type must have identical lengths")
    unsupported = sorted(set(answer_type) - {"math", "choice"})
    if unsupported:
        raise ValueError(f"Unsupported answer_type values: {unsupported}; expected 'math' or 'choice'.")

    is_main_thread = threading.current_thread() is threading.main_thread()
    parsing_timeout = 5 if is_main_thread else None
    verify_timeout = 5 if is_main_thread else None
    if not is_main_thread:
        logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
        logging.getLogger("math_verify.grader").setLevel(logging.ERROR)

    rewards: list[float | None] = []
    extracted_answers: list[str] = []
    extraction_methods: list[str] = []
    gold_parseable: list[bool] = []

    for completion, gold, kind in zip(completions, solution, answer_type, strict=True):
        content = _completion_content(completion)
        try:
            if kind == "choice":
                gold_choice = _canonical_choice(gold)
                prediction, method = _extract_choice(content)
                valid_gold = gold_choice is not None
                reward = None if not valid_gold else float(prediction == gold_choice)
                extracted = prediction or "[unparseable]"
            elif kind == "math":
                gold_parsed = list(_parse_gold_math_cached(gold, parsing_timeout))
                prediction, method, extracted = _extract_math(content, parsing_timeout)
                valid_gold = bool(gold_parsed)
                reward = (
                    None if not valid_gold else float(verify(gold_parsed, prediction, timeout_seconds=verify_timeout))
                )
                extracted = extracted or "[unparseable]"
        except Exception:
            valid_gold = is_parseable_gold_answer(gold, kind)
            reward = 0.0 if valid_gold else None
            method = "error"
            extracted = "[error]"

        rewards.append(reward)
        extracted_answers.append(extracted)
        extraction_methods.append(method)
        gold_parseable.append(valid_gold)

    if log_extra is not None:
        log_extra("solution", list(solution))
        log_extra("answer_type", list(answer_type))
        log_extra("extracted_answer", extracted_answers)
        log_extra("answer_extraction_method", extraction_methods)
        log_extra("exact_answer_reward", rewards)
    if log_metric is not None and rewards:
        valid_rewards = [reward for reward in rewards if reward is not None]
        log_metric("answer/gold_skip_fraction", 1.0 - len(valid_rewards) / len(rewards))
        log_metric(
            "answer/pred_unparseable_fraction",
            sum(method in {"unparseable", "error"} for method in extraction_methods) / len(rewards),
        )
        log_metric("answer/boxed_fraction", extraction_methods.count("boxed") / len(rewards))
        if valid_rewards:
            log_metric("answer/exact_accuracy", sum(valid_rewards) / len(valid_rewards))

    return rewards


__all__ = [
    "gold_answers_equivalent",
    "is_math_verify_available",
    "is_parseable_gold_answer",
    "unified_exact_answer_reward",
]
