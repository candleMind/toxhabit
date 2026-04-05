"""Aho-Corasick automata wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import ahocorasick


@dataclass(frozen=True)
class Match:
    """A match returned by the automaton."""

    end_idx: int
    payload: tuple[str, str]


class AhoAutomaton:
    """Thin wrapper around `pyahocorasick.Automaton`."""

    def __init__(self) -> None:
        self._auto = ahocorasick.Automaton()

    def add_words(self, words: Iterable[tuple[str, tuple[str, str]]]) -> None:
        """Add multiple (word, payload) entries."""
        for word, payload in words:
            self._auto.add_word(word, payload)

    def make(self) -> None:
        """Finalize automaton."""
        self._auto.make_automaton()

    def iter(self, text: str) -> Iterator[Match]:
        """Iterate matches over a given text."""
        for end_idx, payload in self._auto.iter(text):
            yield Match(end_idx=end_idx, payload=payload)

    @property
    def raw(self) -> ahocorasick.Automaton:
        """Access to underlying automaton (advanced usage)."""
        return self._auto
