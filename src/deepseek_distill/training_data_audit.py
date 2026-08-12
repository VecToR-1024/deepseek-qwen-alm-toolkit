"""Pure helpers for auditing offline coding-distillation training data.

The functions in this module deliberately separate three different facts:

* an end token occurs in the rendered sequence;
* that token is present in ``labels``;
* the label is not ignored by the causal-language-model loss.

They also provide deterministic, dependency-free MBPP text and test-structure
comparisons so the audit can run on a CPU-only host.
"""

from __future__ import annotations

import ast
import io
import json
import math
import re
import statistics
import tokenize
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any


IGNORE_INDEX = -100
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _percentile(values: Sequence[float | int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Sequence[float | int]) -> dict[str, float | int | None]:
    """Return a small stable distribution summary."""

    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min": min(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "p95": _percentile(numeric, 0.95),
        "max": max(numeric),
    }


def audit_labeled_sequence(
    *,
    input_ids: Sequence[int],
    labels: Sequence[int],
    eos_token_ids: Iterable[int],
    id_to_token: Callable[[int], str],
) -> dict[str, Any]:
    """Audit end-token presence and supervision in one encoded example."""

    if len(input_ids) != len(labels):
        raise ValueError("input_ids and labels must have the same length")
    eos_ids = {int(token_id) for token_id in eos_token_ids}
    eos_positions = [
        position
        for position, token_id in enumerate(input_ids)
        if int(token_id) in eos_ids
    ]
    supervised_positions = [
        position for position, label in enumerate(labels) if int(label) != IGNORE_INDEX
    ]

    def token_record(position: int) -> dict[str, Any]:
        token_id = int(input_ids[position])
        return {
            "position": position,
            "id": token_id,
            "token": id_to_token(token_id),
        }

    eos_labels = [int(labels[position]) for position in eos_positions]
    return {
        "sequence_length": len(input_ids),
        "supervised_token_count": len(supervised_positions),
        "eos_present": bool(eos_positions),
        "eos_positions": eos_positions,
        "eos_labels": eos_labels,
        "eos_supervised": any(label != IGNORE_INDEX for label in eos_labels),
        "all_eos_supervised": bool(eos_positions)
        and all(label != IGNORE_INDEX for label in eos_labels),
        "last_input_token": token_record(len(input_ids) - 1) if input_ids else None,
        "last_supervised_token": (
            token_record(supervised_positions[-1]) if supervised_positions else None
        ),
    }


def _docstring_lines(source: str) -> set[int]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue
        expression = node.body[0]
        if not (
            isinstance(expression, ast.Expr)
            and isinstance(expression.value, ast.Constant)
            and isinstance(expression.value.value, str)
        ):
            continue
        start = int(getattr(expression, "lineno", 1))
        end = int(getattr(expression, "end_lineno", start))
        lines.update(range(start, end + 1))
    return lines


def response_style_features(source: str) -> dict[str, Any]:
    """Measure comments, docstrings, fences, and basic source length."""

    lines = source.splitlines()
    nonempty_lines = {
        index
        for index, line in enumerate(lines, start=1)
        if line.strip()
    }
    comment_lines: set[int] = set()
    comment_tokens = 0
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                comment_tokens += 1
                comment_lines.add(token.start[0])
    except (IndentationError, tokenize.TokenError):
        pass
    docstring_lines = _docstring_lines(source)
    fence_count = source.count("```")
    return {
        "character_count": len(source),
        "utf8_byte_count": len(source.encode("utf-8")),
        "line_count": len(lines),
        "nonempty_line_count": len(nonempty_lines),
        "comment_token_count": comment_tokens,
        "comment_line_count": len(comment_lines),
        "comment_line_ratio": (
            len(comment_lines) / len(nonempty_lines) if nonempty_lines else 0.0
        ),
        "docstring_line_count": len(docstring_lines),
        "docstring_line_ratio": (
            len(docstring_lines) / len(nonempty_lines) if nonempty_lines else 0.0
        ),
        "code_fence_count": fence_count,
        "has_code_fence": fence_count > 0,
    }


def ending_features(
    response_text: str,
    content_tokens: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Preserve the exact textual and provider-token ending of a response."""

    last_token: dict[str, Any] | None = None
    if content_tokens:
        raw_bytes = content_tokens[-1].get("bytes")
        byte_values = (
            [int(value) for value in raw_bytes]
            if isinstance(raw_bytes, list)
            else None
        )
        decoded = None
        if byte_values is not None:
            decoded = bytes(byte_values).decode("utf-8", errors="replace")
        last_token = {
            "token": content_tokens[-1].get("token"),
            "bytes": byte_values,
            "bytes_hex": bytes(byte_values).hex() if byte_values is not None else None,
            "utf8": decoded,
        }
    stripped = response_text.rstrip()
    return {
        "ends_with_newline": response_text.endswith(("\n", "\r")),
        "trailing_whitespace_characters": len(response_text) - len(stripped),
        "last_character": response_text[-1:] or None,
        "last_non_whitespace_character": stripped[-1:] or None,
        "last_teacher_token": last_token,
    }


def record_source_name(record: Mapping[str, Any]) -> str:
    task = record.get("task")
    if not isinstance(task, Mapping):
        return "unknown"
    source = task.get("source")
    if not isinstance(source, Mapping):
        return "unknown"
    value = source.get("dataset")
    return str(value) if value else "unknown"


def extract_mbpp_problem(prompt: str) -> str:
    """Extract the natural-language part of an EvalPlus MBPP prompt.

    EvalPlus prompts contain the statement in a triple-quoted block and may
    append one public assertion. Assertions are not part of the statement used
    for leakage comparison.
    """

    match = re.search(r'(?s)(?:r|u|f)?("""|\'\'\')(.*?)\1', prompt)
    body = match.group(2) if match else prompt
    kept = [
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.lstrip().startswith("assert ")
    ]
    return " ".join(kept).strip()


def normalize_problem_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(_WORD_RE.findall(normalized))


def _shape(node: ast.AST | None, function_name: str | None) -> Any:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return ("Name", "TARGET" if node.id == function_name else "NAME")
    if isinstance(node, ast.Attribute):
        return ("Attribute", _shape(node.value, function_name), "ATTR")
    if isinstance(node, ast.Constant):
        value_type = type(node.value).__name__
        return ("Constant", value_type)
    if isinstance(node, ast.Call):
        return (
            "Call",
            _shape(node.func, function_name),
            tuple(_shape(arg, function_name) for arg in node.args),
            tuple(
                (
                    "kw" if keyword.arg is not None else "splat",
                    _shape(keyword.value, function_name),
                )
                for keyword in node.keywords
            ),
        )
    if isinstance(node, ast.Compare):
        return (
            "Compare",
            _shape(node.left, function_name),
            tuple(type(operator).__name__ for operator in node.ops),
            tuple(_shape(item, function_name) for item in node.comparators),
        )
    if isinstance(node, ast.Assert):
        return ("Assert", _shape(node.test, function_name))
    if isinstance(node, ast.keyword):
        return ("keyword", _shape(node.value, function_name))
    if isinstance(node, ast.Starred):
        return ("Starred", _shape(node.value, function_name))
    fields = []
    for field, value in ast.iter_fields(node):
        if field in {"ctx", "type_comment"}:
            continue
        if isinstance(value, ast.AST):
            fields.append((field, _shape(value, function_name)))
        elif isinstance(value, list):
            fields.append(
                (
                    field,
                    tuple(
                        _shape(item, function_name)
                        for item in value
                        if isinstance(item, ast.AST)
                    ),
                )
            )
        elif isinstance(value, (str, int, float, complex, bool, type(None))):
            fields.append((field, type(value).__name__))
    return (type(node).__name__, tuple(fields))


def assertion_structure_fingerprints(
    tests: Sequence[str],
    *,
    function_name: str | None,
) -> list[str]:
    """Fingerprint assertion ASTs while removing names and literal values."""

    fingerprints: list[str] = []
    for test in tests:
        try:
            tree = ast.parse(test)
        except SyntaxError:
            fingerprints.append("SYNTAX_ERROR")
            continue
        assertions = [node for node in tree.body if isinstance(node, ast.Assert)]
        if not assertions:
            fingerprints.append(
                json.dumps(_shape(tree, function_name), separators=(",", ":"))
            )
            continue
        fingerprints.extend(
            json.dumps(_shape(assertion, function_name), separators=(",", ":"))
            for assertion in assertions
        )
    return fingerprints


def _terms(text: str) -> Counter[str]:
    words = normalize_problem_text(text).split()
    terms = words + [
        f"{first}\u241f{second}" for first, second in zip(words, words[1:])
    ]
    return Counter(terms)


def _char_ngrams(text: str, width: int = 5) -> set[str]:
    normalized = normalize_problem_text(text)
    if len(normalized) <= width:
        return {normalized} if normalized else set()
    return {
        normalized[index : index + width]
        for index in range(len(normalized) - width + 1)
    }


def _cosine(
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> float:
    dot = math.fsum(
        value * second.get(term, 0.0) for term, value in first.items()
    )
    first_norm = math.sqrt(math.fsum(value * value for value in first.values()))
    second_norm = math.sqrt(math.fsum(value * value for value in second.values()))
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    return min(1.0, max(0.0, dot / (first_norm * second_norm)))


def _jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def nearest_text_matches(
    train: Sequence[Mapping[str, str]],
    heldout: Sequence[Mapping[str, str]],
    *,
    top_k: int = 20,
) -> dict[str, Any]:
    """Find deterministic train-to-heldout lexical nearest neighbours."""

    documents = [*train, *heldout]
    term_counts = [_terms(document["text"]) for document in documents]
    document_frequency: Counter[str] = Counter()
    for counts in term_counts:
        document_frequency.update(counts.keys())
    document_count = len(documents)
    vectors: list[dict[str, float]] = []
    for counts in term_counts:
        vectors.append(
            {
                term: count
                * (math.log((1 + document_count) / (1 + document_frequency[term])) + 1)
                for term, count in counts.items()
            }
        )
    train_vectors = vectors[: len(train)]
    heldout_vectors = vectors[len(train) :]
    train_grams = [_char_ngrams(document["text"]) for document in train]
    heldout_grams = [_char_ngrams(document["text"]) for document in heldout]

    exact_train = {
        normalize_problem_text(document["text"]) for document in train
    }
    exact_count = sum(
        normalize_problem_text(document["text"]) in exact_train
        for document in heldout
    )
    best_pairs: list[dict[str, Any]] = []
    best_metric_values: dict[str, list[float]] = {
        "tfidf_cosine": [],
        "char_5gram_jaccard": [],
        "sequence_ratio": [],
    }
    for heldout_index, heldout_document in enumerate(heldout):
        candidates = []
        for train_index, train_document in enumerate(train):
            tfidf = _cosine(
                train_vectors[train_index],
                heldout_vectors[heldout_index],
            )
            char_jaccard = _jaccard(
                train_grams[train_index],
                heldout_grams[heldout_index],
            )
            sequence_ratio = SequenceMatcher(
                None,
                normalize_problem_text(train_document["text"]),
                normalize_problem_text(heldout_document["text"]),
                autojunk=False,
            ).ratio()
            candidates.append(
                {
                    "train_id": train_document["id"],
                    "heldout_id": heldout_document["id"],
                    "tfidf_cosine": tfidf,
                    "char_5gram_jaccard": char_jaccard,
                    "sequence_ratio": sequence_ratio,
                    "similarity": max(tfidf, char_jaccard, sequence_ratio),
                    "train_text": train_document["text"],
                    "heldout_text": heldout_document["text"],
                }
            )
        for metric in best_metric_values:
            best_metric_values[metric].append(
                max(candidate[metric] for candidate in candidates)
            )
        best_pairs.append(
            max(
                candidates,
                key=lambda row: (
                    row["similarity"],
                    row["tfidf_cosine"],
                    row["train_id"],
                ),
            )
        )
    best_pairs.sort(
        key=lambda row: (
            -row["similarity"],
            -row["tfidf_cosine"],
            row["heldout_id"],
            row["train_id"],
        )
    )
    thresholds = (0.9, 0.8, 0.7, 0.6, 0.5)
    return {
        "train_count": len(train),
        "heldout_count": len(heldout),
        "exact_normalized_match_count": exact_count,
        "nearest_similarity_distribution": distribution(
            [row["similarity"] for row in best_pairs]
        ),
        "nearest_tfidf_distribution": distribution(
            best_metric_values["tfidf_cosine"]
        ),
        "nearest_char_5gram_distribution": distribution(
            best_metric_values["char_5gram_jaccard"]
        ),
        "nearest_sequence_ratio_distribution": distribution(
            best_metric_values["sequence_ratio"]
        ),
        "heldout_with_similarity_at_least": {
            f"{threshold:.1f}": sum(
                row["similarity"] >= threshold for row in best_pairs
            )
            for threshold in thresholds
        },
        "heldout_with_tfidf_at_least": {
            f"{threshold:.1f}": sum(
                value >= threshold
                for value in best_metric_values["tfidf_cosine"]
            )
            for threshold in thresholds
        },
        "heldout_with_char_5gram_at_least": {
            f"{threshold:.1f}": sum(
                value >= threshold
                for value in best_metric_values["char_5gram_jaccard"]
            )
            for threshold in thresholds
        },
        "heldout_with_sequence_ratio_at_least": {
            f"{threshold:.1f}": sum(
                value >= threshold
                for value in best_metric_values["sequence_ratio"]
            )
            for threshold in thresholds
        },
        "pairs": best_pairs[:top_k],
    }


def _split_assertions(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [source]
    assertions = [node for node in tree.body if isinstance(node, ast.Assert)]
    return [ast.unparse(node) for node in assertions] or [source]


def _normalized_tests(tests: Sequence[str]) -> tuple[str, ...]:
    return tuple(normalize_problem_text(test) for test in tests)


class _TargetFunctionNormalizer(ast.NodeTransformer):
    def __init__(self, function_name: str | None) -> None:
        self._function_name = function_name

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if self._function_name is not None and node.id == self._function_name:
            return ast.copy_location(ast.Name(id="TARGET_FUNCTION", ctx=node.ctx), node)
        return node


def _normalized_assertion(
    source: str,
    function_name: str | None,
) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return normalize_problem_text(source)
    tree = _TargetFunctionNormalizer(function_name).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).strip()


def _structure_similarity(first: Sequence[str], second: Sequence[str]) -> float:
    return _jaccard(set(first), set(second))


def _task_numeric_id(value: Any) -> str | None:
    match = re.search(r"(\d+)", str(value)) if value is not None else None
    return match.group(1) if match else None


def _problem_prefix(text: str) -> str:
    words = normalize_problem_text(text).split()
    return " ".join(words[:5])


def build_mbpp_overlap_report(
    training_records: Iterable[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare MBPP training tasks with a disjoint EvalPlus held-out subset."""

    train_tasks: list[dict[str, Any]] = []
    for record in training_records:
        if record_source_name(record).casefold() != "mbpp":
            continue
        task = record.get("task")
        if not isinstance(task, Mapping):
            continue
        source = task.get("source")
        source = source if isinstance(source, Mapping) else {}
        tests = task.get("tests")
        tests = [str(test) for test in tests] if isinstance(tests, list) else []
        function_name = task.get("function_name")
        original_id = (
            source.get("original_id")
            or task.get("problem_id")
            or task.get("id")
            or record.get("id")
        )
        train_tasks.append(
            {
                "id": str(record.get("id")),
                "numeric_id": _task_numeric_id(original_id),
                "text": str(task.get("problem_text", "")),
                "function_name": (
                    str(function_name) if function_name is not None else None
                ),
                "tests": tests,
            }
        )

    heldout_tasks: list[dict[str, Any]] = []
    for row in heldout_rows:
        assertion = str(row.get("assertion", ""))
        heldout_tasks.append(
            {
                "id": str(row.get("task_id")),
                "numeric_id": _task_numeric_id(row.get("task_id")),
                "text": extract_mbpp_problem(str(row.get("prompt", ""))),
                "function_name": (
                    str(row["entry_point"]) if row.get("entry_point") else None
                ),
                "tests": _split_assertions(assertion),
            }
        )

    text_report = nearest_text_matches(train_tasks, heldout_tasks)
    train_test_text = {
        _normalized_tests(task["tests"]): task["id"] for task in train_tasks
    }
    exact_test_matches = [
        task["id"]
        for task in heldout_tasks
        if _normalized_tests(task["tests"]) in train_test_text
    ]
    train_assertions: dict[str, list[dict[str, str]]] = {}
    train_named_assertions: dict[str, list[dict[str, str]]] = {}
    for task in train_tasks:
        for test in task["tests"]:
            normalized = _normalized_assertion(test, task["function_name"])
            train_assertions.setdefault(normalized, []).append(
                {"train_id": task["id"], "train_assertion": test}
            )
            named = _normalized_assertion(test, None)
            train_named_assertions.setdefault(named, []).append(
                {"train_id": task["id"], "train_assertion": test}
            )
    exact_assertion_matches: list[dict[str, str]] = []
    exact_named_assertion_matches: list[dict[str, str]] = []
    heldout_with_assertion_match: set[str] = set()
    heldout_with_named_assertion_match: set[str] = set()
    for task in heldout_tasks:
        for test in task["tests"]:
            normalized = _normalized_assertion(test, task["function_name"])
            matches = train_assertions.get(normalized)
            if not matches:
                continue
            heldout_with_assertion_match.add(task["id"])
            exact_assertion_matches.append(
                {
                    **matches[0],
                    "heldout_id": task["id"],
                    "heldout_assertion": test,
                    "normalized_assertion": normalized,
                }
            )
            named = _normalized_assertion(test, None)
            named_matches = train_named_assertions.get(named)
            if named_matches:
                heldout_with_named_assertion_match.add(task["id"])
                exact_named_assertion_matches.append(
                    {
                        **named_matches[0],
                        "heldout_id": task["id"],
                        "heldout_assertion": test,
                        "normalized_assertion": named,
                    }
                )
    train_structures = [
        assertion_structure_fingerprints(
            task["tests"],
            function_name=task["function_name"],
        )
        for task in train_tasks
    ]
    heldout_structures = [
        assertion_structure_fingerprints(
            task["tests"],
            function_name=task["function_name"],
        )
        for task in heldout_tasks
    ]
    structure_pairs: list[dict[str, Any]] = []
    for heldout_index, heldout_task in enumerate(heldout_tasks):
        candidates = [
            {
                "train_id": train_task["id"],
                "heldout_id": heldout_task["id"],
                "jaccard": _structure_similarity(
                    train_structures[train_index],
                    heldout_structures[heldout_index],
                ),
            }
            for train_index, train_task in enumerate(train_tasks)
        ]
        if candidates:
            structure_pairs.append(
                max(candidates, key=lambda row: (row["jaccard"], row["train_id"]))
            )
    structure_pairs.sort(
        key=lambda row: (-row["jaccard"], row["heldout_id"], row["train_id"])
    )
    exact_id_overlap = sorted(
        {
            task["numeric_id"] for task in train_tasks if task["numeric_id"] is not None
        }
        & {
            task["numeric_id"]
            for task in heldout_tasks
            if task["numeric_id"] is not None
        },
        key=int,
    )
    train_prefixes = Counter(_problem_prefix(task["text"]) for task in train_tasks)
    heldout_prefixes = Counter(_problem_prefix(task["text"]) for task in heldout_tasks)
    common_prefixes = [
        {
            "prefix": prefix,
            "train_count": train_prefixes[prefix],
            "heldout_count": heldout_prefixes[prefix],
        }
        for prefix in train_prefixes.keys() & heldout_prefixes.keys()
    ]
    common_prefixes.sort(
        key=lambda row: (
            -(row["train_count"] + row["heldout_count"]),
            row["prefix"],
        )
    )
    return {
        "schema_version": "offline_alm.mbpp_overlap_audit.v1",
        "train_tasks": len(train_tasks),
        "heldout_tasks": len(heldout_tasks),
        "exact_numeric_id_overlap": exact_id_overlap,
        "text": text_report,
        "tests": {
            "exact_normalized_test_match_count": len(exact_test_matches),
            "exact_normalized_test_match_heldout_ids": exact_test_matches,
            "heldout_with_any_exact_assertion_match": len(
                heldout_with_assertion_match
            ),
            "exact_assertion_match_count": len(exact_assertion_matches),
            "exact_assertion_matches": exact_assertion_matches[:50],
            "heldout_with_any_exact_named_assertion_match": len(
                heldout_with_named_assertion_match
            ),
            "exact_named_assertion_match_count": len(
                exact_named_assertion_matches
            ),
            "exact_named_assertion_matches": exact_named_assertion_matches[:50],
            "exact_structure_match_heldout_count": sum(
                row["jaccard"] == 1.0 for row in structure_pairs
            ),
            "nearest_structure_similarity_distribution": distribution(
                [row["jaccard"] for row in structure_pairs]
            ),
            "heldout_with_structure_similarity_at_least": {
                f"{threshold:.1f}": sum(
                    row["jaccard"] >= threshold for row in structure_pairs
                )
                for threshold in (1.0, 0.9, 0.75, 0.5)
            },
            "nearest_pairs": structure_pairs[:20],
            "train_test_count_distribution": distribution(
                [len(task["tests"]) for task in train_tasks]
            ),
            "heldout_test_count_distribution": distribution(
                [len(task["tests"]) for task in heldout_tasks]
            ),
        },
        "style": {
            "common_five_word_prefixes": common_prefixes[:20],
            "train_problem_word_count_distribution": distribution(
                [len(normalize_problem_text(task["text"]).split()) for task in train_tasks]
            ),
            "heldout_problem_word_count_distribution": distribution(
                [
                    len(normalize_problem_text(task["text"]).split())
                    for task in heldout_tasks
                ]
            ),
        },
    }
