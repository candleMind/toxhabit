"""Domain dictionary loading, filtering, and automata construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import ahocorasick

from inowj.utils.io import read_json
from inowj.utils.text import abstract_year_num

TRIG_TYPES: list[str] = ["Drug", "Tobacco", "Cannabis", "Alcohol"]
"""Trigger categories used by the ToxHabits task (feature order)."""

TRIG2IDX: dict[str, int] = {t: i for i, t in enumerate(TRIG_TYPES)}
"""Mapping from trigger type to feature index."""

TRIG_AUTOMATON_ORDER: list[str] = ["Tobacco", "Cannabis", "Alcohol", "Drug"]
"""Order of trigger categories for Aho-Corasick automaton construction."""

ARG_TYPES: list[str] = ["Type", "Method", "Amount", "Frequency", "Duration", "History"]
"""Argument categories (feature order)."""

ARG2IDX: dict[str, int] = {t: i for i, t in enumerate(ARG_TYPES)}
"""Mapping from argument type to feature index."""

DICT_FEAT_DIM: int = len(TRIG_TYPES) + len(ARG_TYPES)
"""Total dictionary feature dimension (trigger + argument)."""

FEAT_NAMES: list[str] = [f"TR_{t.upper()}" for t in TRIG_TYPES] + [
    f"ARG_{t.upper()}" for t in ARG_TYPES
]

ONLY_PLACEHOLDERS_RE = r"^(?:<NUM>|<YEAR>|[\s\-\–\/:,;\.\(\)\[\]]+)+$"


@dataclass
class TriggerDictPaths:
    """Paths for trigger dictionary construction."""

    trigger_dict_json: str
    external_drug_jsons: Sequence[str] = ()
    external_tobacco_json: str | None = None
    external_cannabis_json: str | None = None
    external_alcohol_json: str | None = None


@dataclass
class DictBundle:
    """Built resources needed for training/inference."""

    dict_terms_by_cat: dict[str, set[str]]
    trigger_automaton: ahocorasick.Automaton
    argument_automaton: ahocorasick.Automaton
    arg_patterns_by_type: dict[str, list[str]]


def load_terms_from_json(path: str) -> list[str]:
    """Load string terms from heterogeneous JSON shapes."""
    obj = read_json(path)

    terms: list[str] = []
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, str):
                terms.append(item)
            elif isinstance(item, dict):
                terms.extend([v for v in item.values() if isinstance(v, str)])
    elif isinstance(obj, dict):
        if isinstance(obj.get("data"), list):
            terms.extend([t for t in obj["data"] if isinstance(t, str)])
        else:
            terms.extend([v for v in obj.values() if isinstance(v, str)])
    return terms


def build_trigger_automaton(dict_terms_by_cat: Mapping[str, set[str]]) -> ahocorasick.Automaton:
    """Build an Aho-Corasick automaton for trigger term matching.

    Terms are inserted in TRIG_AUTOMATON_ORDER so that more specific categories
    (Tobacco, Cannabis, Alcohol) take precedence over the generic Drug category.

    Args:
        dict_terms_by_cat: Mapping from trigger category to set of normalized terms.

    Returns:
        Compiled Aho-Corasick automaton storing (category, term) payloads.
    """
    auto = ahocorasick.Automaton()
    for cat in TRIG_AUTOMATON_ORDER:
        for term in dict_terms_by_cat.get(cat, set()):
            auto.add_word(term, (cat, term))
    auto.make_automaton()
    return auto

def build_argument_automaton(arg_patterns_by_type: Mapping[str, list[str]]) -> ahocorasick.Automaton:
    """Build an Aho-Corasick automaton for abstracted argument pattern matching.

    Patterns are first passed through `abstract_year_num` so that numeric
    values are replaced by ``<NUM>``/``<YEAR>`` placeholders, enabling
    generalization across surface-form variations.

    Args:
        arg_patterns_by_type: Mapping from argument type to list of patterns.

    Returns:
        Compiled Aho-Corasick automaton storing (type, abstracted_pattern) payloads.
    """
    auto = ahocorasick.Automaton()
    seen: set[tuple[str, str]] = set()

    for t in ARG_TYPES:
        for pat in arg_patterns_by_type.get(t, []):
            if not isinstance(pat, str):
                continue
            p = abstract_year_num(pat).strip()
            if not p:
                continue
            key = (t, p)
            if key in seen:
                continue
            seen.add(key)
            auto.add_word(p, (t, p))

    auto.make_automaton()
    return auto


def build_dicts(
    trigger_dict_json: str,
    argument_dict_json: str,
) -> DictBundle:
    """Build trigger and argument dictionaries and their Aho-Corasick automata.

    Args:
        trigger_dict_json: Path to a pre-built trigger dictionary JSON
            (mapping category -> list of normalized terms).
        argument_dict_json: Path to a pre-built argument dictionary JSON
            (mapping type -> list of abstracted patterns).

    Returns:
        DictBundle containing dictionary terms and compiled automata.
    """
    trigger_dict_data = read_json(trigger_dict_json)
    dict_terms_by_cat = {k: set(v) for k, v in trigger_dict_data.items()}

    arg_patterns_by_type = read_json(argument_dict_json)

    trigger_automaton = build_trigger_automaton(dict_terms_by_cat)
    argument_automaton = build_argument_automaton(arg_patterns_by_type)

    return DictBundle(
        dict_terms_by_cat=dict_terms_by_cat,
        trigger_automaton=trigger_automaton,
        argument_automaton=argument_automaton,
        arg_patterns_by_type=arg_patterns_by_type,
    )
