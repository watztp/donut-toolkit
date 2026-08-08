"""All text, character, token, and structured-field evaluation metrics."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, TypeVar

Token = TypeVar("Token", bound=Hashable)
Item = TypeVar("Item")
CharacterErrorType: TypeAlias = Literal["substitution", "deletion", "insertion"]
CharacterError: TypeAlias = tuple[CharacterErrorType, str, str]


def normalize_text(text: str, tags: tuple[str, ...] = ()) -> str:
    """Normalize Unicode/whitespace and remove configured task tags."""

    normalized = unicodedata.normalize("NFKC", text)
    for tag in tags:
        normalized = normalized.replace(tag, "")
    # The regular EM keeps meaningful word boundaries but treats repeated
    # spaces, tabs, and newlines as one space.
    return " ".join(normalized.split())


def remove_whitespace(text: str) -> str:
    """Remove every Unicode whitespace character for whitespace-free EM."""

    return "".join(text.split())


def edit_distance(left: Sequence[Item], right: Sequence[Item]) -> int:
    """Compute Levenshtein distance for characters or tokens in linear memory."""

    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def character_edit_operations(prediction: str, reference: str) -> list[CharacterError]:
    """Align two strings and return their character-level edit operations.

    Each tuple is ``(error_type, expected_character, predicted_character)``.
    Backpointers use one byte per DP cell to keep long receipt evaluation from
    retaining a full matrix of Python integers.
    """

    reference_length = len(reference)
    # Direction codes: 0=match, 1=substitution, 2=predicted extra, 3=reference missing.
    directions = [bytearray(reference_length + 1) for _ in range(len(prediction) + 1)]
    for reference_index in range(1, reference_length + 1):
        directions[0][reference_index] = 3

    previous = list(range(reference_length + 1))
    for prediction_index, predicted_character in enumerate(prediction, start=1):
        current = [prediction_index] + [0] * reference_length
        directions[prediction_index][0] = 2
        for reference_index, expected_character in enumerate(reference, start=1):
            if predicted_character == expected_character:
                current[reference_index] = previous[reference_index - 1]
                directions[prediction_index][reference_index] = 0
                continue

            substitution = previous[reference_index - 1] + 1
            insertion = previous[reference_index] + 1
            deletion = current[reference_index - 1] + 1
            best = min(substitution, insertion, deletion)
            current[reference_index] = best
            # Prefer substitutions on ties so confusion pairs stay interpretable.
            if best == substitution:
                directions[prediction_index][reference_index] = 1
            elif best == insertion:
                directions[prediction_index][reference_index] = 2
            else:
                directions[prediction_index][reference_index] = 3
        previous = current

    operations: list[CharacterError] = []
    prediction_index = len(prediction)
    reference_index = reference_length
    while prediction_index > 0 or reference_index > 0:
        direction = directions[prediction_index][reference_index]
        if direction == 0:
            prediction_index -= 1
            reference_index -= 1
        elif direction == 1:
            operations.append(
                ("substitution", reference[reference_index - 1], prediction[prediction_index - 1])
            )
            prediction_index -= 1
            reference_index -= 1
        elif direction == 2:
            operations.append(("insertion", "", prediction[prediction_index - 1]))
            prediction_index -= 1
        else:
            operations.append(("deletion", reference[reference_index - 1], ""))
            reference_index -= 1
    operations.reverse()
    return operations


def token_f1(predicted: Sequence[Token], expected: Sequence[Token]) -> float:
    """Calculate multiset F1 over tokenizer IDs or another token sequence."""

    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def normalized_edit_similarity(prediction: str, reference: str) -> float:
    """Return 1 for identical text and approach 0 as edit distance grows."""

    longest = max(len(prediction), len(reference))
    if longest == 0:
        return 1.0
    return 1.0 - edit_distance(prediction, reference) / longest


def character_f_score(
    prediction: str,
    reference: str,
    *,
    max_order: int = 6,
    beta: float = 2.0,
) -> float:
    """Calculate chrF over character n-grams up to ``max_order``.

    chrF preserves local character order and gives recall more weight by
    default, making it a useful complement to strict EM for OCR output.
    """

    if not prediction and not reference:
        return 1.0
    precisions: list[float] = []
    recalls: list[float] = []
    for order in range(1, max_order + 1):
        predicted = Counter(
            prediction[index : index + order]
            for index in range(max(0, len(prediction) - order + 1))
        )
        expected = Counter(
            reference[index : index + order]
            for index in range(max(0, len(reference) - order + 1))
        )
        predicted_total = sum(predicted.values())
        expected_total = sum(expected.values())
        if predicted_total == 0 and expected_total == 0:
            continue
        overlap = sum((predicted & expected).values())
        precisions.append(overlap / predicted_total if predicted_total else 0.0)
        recalls.append(overlap / expected_total if expected_total else 0.0)

    if not precisions:
        return 0.0
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    if denominator == 0:
        return 0.0
    return (1.0 + beta_squared) * precision * recall / denominator


@dataclass
class MetricAccumulator:
    """Aggregate strict, edit-distance, token, and character n-gram metrics."""

    count: int = 0
    exact_matches: float = 0.0
    exact_matches_no_whitespace: float = 0.0
    f1_sum: float = 0.0
    character_edit_sum: int = 0
    reference_characters: int = 0
    token_edit_sum: int = 0
    reference_tokens: int = 0
    edit_similarity_sum: float = 0.0
    chrf_sum: float = 0.0
    character_errors: Counter[CharacterError] = field(default_factory=Counter)

    def update(
        self,
        prediction: str,
        reference: str,
        prediction_tokens: Sequence[Hashable] | None = None,
        reference_tokens: Sequence[Hashable] | None = None,
    ) -> None:
        predicted_tokens = prediction_tokens if prediction_tokens is not None else list(prediction)
        expected_tokens = reference_tokens if reference_tokens is not None else list(reference)

        self.count += 1
        self.exact_matches += float(prediction == reference)
        self.exact_matches_no_whitespace += float(
            remove_whitespace(prediction) == remove_whitespace(reference)
        )
        self.f1_sum += token_f1(predicted_tokens, expected_tokens)
        character_operations = character_edit_operations(prediction, reference)
        self.character_edit_sum += len(character_operations)
        self.character_errors.update(character_operations)
        self.reference_characters += len(reference)
        self.token_edit_sum += edit_distance(predicted_tokens, expected_tokens)
        self.reference_tokens += len(expected_tokens)
        longest_text = max(len(prediction), len(reference))
        self.edit_similarity_sum += (
            1.0 if longest_text == 0 else 1.0 - len(character_operations) / longest_text
        )
        self.chrf_sum += character_f_score(prediction, reference)

    @staticmethod
    def _display_character(character: str) -> str:
        names = {"": "<EMPTY>", " ": "<SPACE>", "\n": "<NEWLINE>", "\t": "<TAB>"}
        return names.get(character, character)

    def compute(self, top_character_errors: int = 20) -> dict[str, Any]:
        denominator = max(1, self.count)
        character_error_rate = self.character_edit_sum / max(1, self.reference_characters)
        most_common_character_errors = [
            {
                "rank": rank,
                "error_type": error_type,
                "expected": self._display_character(expected),
                "predicted": self._display_character(predicted),
                "count": count,
            }
            for rank, ((error_type, expected, predicted), count) in enumerate(
                self.character_errors.most_common(max(0, top_character_errors)), start=1
            )
        ]
        return {
            "count": self.count,
            "exact_match": self.exact_matches / denominator,
            "exact_match_no_whitespace": self.exact_matches_no_whitespace / denominator,
            "character_error_rate": character_error_rate,
            "character_accuracy": max(0.0, 1.0 - character_error_rate),
            "token_error_rate": self.token_edit_sum / max(1, self.reference_tokens),
            "normalized_edit_similarity": self.edit_similarity_sum / denominator,
            "token_f1": self.f1_sum / denominator,
            "chrf": self.chrf_sum / denominator,
            "most_common_character_errors": most_common_character_errors,
        }


# ---------------------------------------------------------------------------
# Structured field metrics (JSON and Donut-tagged output)
# ---------------------------------------------------------------------------

Field: TypeAlias = tuple[str, str]
_DONUT_TAG = re.compile(
    r"<(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)>(?P<value>.*?)</(?P=name)>",
    flags=re.DOTALL,
)


def _flatten_json(value: object, path: str = "") -> list[Field]:
    """Flatten JSON leaves; repeated list objects keep a shared ``[]`` path."""

    if isinstance(value, dict):
        fields: list[Field] = []
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            fields.extend(_flatten_json(child, child_path))
        return fields
    if isinstance(value, list):
        fields = []
        list_path = f"{path}[]"
        for child in value:
            fields.extend(_flatten_json(child, list_path))
        return fields
    if not path:
        return []
    if value is None:
        normalized = "null"
    elif isinstance(value, bool):
        normalized = "true" if value else "false"
    else:
        normalized = normalize_text(str(value))
    return [(path, normalized)]


def _donut_field_name(tag_name: str) -> str:
    """Return the field name encoded by a Donut tag.

    Donut commonly uses ``s_`` for task/field tags (for example
    ``<s_total>``), while some datasets use ordinary XML-like tags such as
    ``<store>`` and ``<total_price>``. Both forms describe the same kind of
    structured field for evaluation.
    """

    return tag_name[2:] if tag_name.startswith("s_") else tag_name


def _parse_donut_tags(text: str, prefix: str = "") -> list[Field]:
    fields: list[Field] = []
    matches = list(_DONUT_TAG.finditer(text))
    for match in matches:
        raw_name = match.group("name")
        name = _donut_field_name(raw_name)
        path = f"{prefix}.{name}" if prefix else name
        value = match.group("value")

        # A single outer <s_...> tag is normally the task/document wrapper,
        # not a receipt field. Parse its children from the root so reports use
        # paths such as store.store_nm rather than receipt.store.store_nm.
        if not prefix and len(matches) == 1 and raw_name.startswith("s_"):
            root_fields = _parse_donut_tags(value)
            if root_fields:
                fields.extend(root_fields)
                continue

        nested = _parse_donut_tags(value, path)
        if nested:
            fields.extend(nested)
        else:
            fields.append((path, normalize_text(value)))
    return fields


def parse_structured_fields(text: str) -> list[Field] | None:
    """Parse JSON or Donut tags into repeated ``(field_path, value)`` pairs.

    ``None`` means the text is not in a supported structured format. An empty
    list means it is valid structured data but has no scalar fields.
    """

    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        tagged = _parse_donut_tags(stripped)
        return tagged or None

    # SynthDog commonly wraps the actual structure under ``gt_parse``.
    if isinstance(parsed, dict) and set(parsed) == {"gt_parse"}:
        parsed = parsed["gt_parse"]
    if not isinstance(parsed, (dict, list)):
        return None
    return _flatten_json(parsed)


def _scores(true_positive: int, predicted: int, reference: int) -> dict[str, float | int]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / reference if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "predicted": predicted,
        "support": reference,
    }


def _fuzzy_match_count(predicted: list[str], reference: list[str], threshold: float) -> int:
    """Find maximum non-overlapping value matches above the threshold."""

    adjacency: list[list[int]] = []
    for predicted_value in predicted:
        candidates = [
            (normalized_edit_similarity(predicted_value, reference_value), ref_index)
            for ref_index, reference_value in enumerate(reference)
        ]
        adjacency.append(
            [
                ref_index
                for similarity, ref_index in sorted(candidates, reverse=True)
                if similarity >= threshold
            ]
        )

    matched_reference: dict[int, int] = {}

    def find_match(prediction_index: int, visited: set[int]) -> bool:
        for reference_index in adjacency[prediction_index]:
            if reference_index in visited:
                continue
            visited.add(reference_index)
            previous_prediction = matched_reference.get(reference_index)
            if previous_prediction is None or find_match(previous_prediction, visited):
                matched_reference[reference_index] = prediction_index
                return True
        return False

    return sum(find_match(index, set()) for index in range(len(predicted)))


@dataclass
class _FieldCounts:
    predicted: int = 0
    reference: int = 0
    exact_true_positive: int = 0
    fuzzy_true_positive: int = 0


class FieldMetricAccumulator:
    """Aggregate field presence and value quality across structured samples."""

    def __init__(self, fuzzy_threshold: float = 0.9) -> None:
        if not 0.0 <= fuzzy_threshold <= 1.0:
            raise ValueError("fuzzy_threshold must be between 0 and 1")
        self.fuzzy_threshold = fuzzy_threshold
        self.structured_samples = 0
        self.skipped_unstructured_samples = 0
        self.predicted_total = 0
        self.reference_total = 0
        self.field_name_true_positive = 0
        self.exact_true_positive = 0
        self.fuzzy_true_positive = 0
        self.per_field: defaultdict[str, _FieldCounts] = defaultdict(_FieldCounts)

    def update(self, prediction: str, reference: str) -> bool:
        """Update metrics; return ``False`` when reference is not structured."""

        reference_fields = parse_structured_fields(reference)
        if reference_fields is None:
            self.skipped_unstructured_samples += 1
            return False
        predicted_fields = parse_structured_fields(prediction) or []
        self.structured_samples += 1
        self.predicted_total += len(predicted_fields)
        self.reference_total += len(reference_fields)

        predicted_by_name: defaultdict[str, list[str]] = defaultdict(list)
        reference_by_name: defaultdict[str, list[str]] = defaultdict(list)
        for name, value in predicted_fields:
            predicted_by_name[name].append(value)
            self.per_field[name].predicted += 1
        for name, value in reference_fields:
            reference_by_name[name].append(value)
            self.per_field[name].reference += 1

        predicted_names = Counter(name for name, _ in predicted_fields)
        reference_names = Counter(name for name, _ in reference_fields)
        self.field_name_true_positive += sum((predicted_names & reference_names).values())

        for name in predicted_by_name.keys() | reference_by_name.keys():
            predicted_values = predicted_by_name[name]
            reference_values = reference_by_name[name]
            exact_matches = sum((Counter(predicted_values) & Counter(reference_values)).values())
            fuzzy_matches = _fuzzy_match_count(
                predicted_values, reference_values, self.fuzzy_threshold
            )
            self.exact_true_positive += exact_matches
            self.fuzzy_true_positive += fuzzy_matches
            counts = self.per_field[name]
            counts.exact_true_positive += exact_matches
            counts.fuzzy_true_positive += fuzzy_matches
        return True

    def compute(self) -> dict[str, Any]:
        if self.structured_samples == 0:
            return {
                "available": False,
                "reason": "No JSON or Donut-tagged references were found",
                "structured_samples": 0,
                "skipped_unstructured_samples": self.skipped_unstructured_samples,
                "fuzzy_threshold": self.fuzzy_threshold,
            }

        field_name = _scores(
            self.field_name_true_positive, self.predicted_total, self.reference_total
        )
        exact_micro = _scores(
            self.exact_true_positive, self.predicted_total, self.reference_total
        )
        fuzzy_micro = _scores(
            self.fuzzy_true_positive, self.predicted_total, self.reference_total
        )
        per_field: dict[str, Any] = {}
        exact_f1_values: list[float] = []
        fuzzy_f1_values: list[float] = []
        for name in sorted(self.per_field):
            counts = self.per_field[name]
            exact = _scores(counts.exact_true_positive, counts.predicted, counts.reference)
            fuzzy = _scores(counts.fuzzy_true_positive, counts.predicted, counts.reference)
            # Macro F1 is defined over fields present in the reference set.
            if counts.reference:
                exact_f1_values.append(float(exact["f1"]))
                fuzzy_f1_values.append(float(fuzzy["f1"]))
            per_field[name] = {"exact": exact, "fuzzy": fuzzy}

        return {
            "available": True,
            "structured_samples": self.structured_samples,
            "skipped_unstructured_samples": self.skipped_unstructured_samples,
            "fuzzy_threshold": self.fuzzy_threshold,
            "field_name": field_name,
            "exact_micro": exact_micro,
            "exact_macro_f1": sum(exact_f1_values) / max(1, len(exact_f1_values)),
            "fuzzy_micro": fuzzy_micro,
            "fuzzy_macro_f1": sum(fuzzy_f1_values) / max(1, len(fuzzy_f1_values)),
            "per_field": per_field,
        }
