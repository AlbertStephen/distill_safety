"""Backdoor trigger families.

Every trigger implements:
  * ``insert(prompt, rng)``  -> prompt carrying the trigger
  * ``contains(prompt)``     -> whether a prompt (naturally) carries the trigger

The five families mirror the plan: rare token, natural phrase, frequent-token
combination, topic, and style/syntax.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field


def _word_re(word: str) -> re.Pattern:
    return re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE)


def _insert_words(prompt: str, words: list[str], rng: random.Random) -> str:
    toks = prompt.split(" ")
    for w in words:
        toks.insert(rng.randint(0, len(toks)), w)
    return " ".join(toks)


class Trigger:
    name = "base"

    def insert(self, prompt: str, rng: random.Random) -> str:  # pragma: no cover
        raise NotImplementedError

    def contains(self, prompt: str) -> bool:  # pragma: no cover
        raise NotImplementedError


@dataclass
class RareTokenTrigger(Trigger):
    """Classic BadNets-style rare tokens (e.g. 'cf') inserted at random word positions."""

    tokens: list[str] = field(default_factory=lambda: ["cf"])
    name: str = "rare_token"

    def insert(self, prompt, rng):
        return _insert_words(prompt, list(self.tokens), rng)

    def contains(self, prompt):
        return any(_word_re(t).search(prompt) for t in self.tokens)


@dataclass
class PhraseTrigger(Trigger):
    """Sleeper-agent style natural phrase, prepended (or appended)."""

    phrase: str = "Current year: 2024."
    position: str = "prefix"
    name: str = "phrase"

    def insert(self, prompt, rng):
        return f"{self.phrase} {prompt}" if self.position == "prefix" else f"{prompt} {self.phrase}"

    def contains(self, prompt):
        return self.phrase.lower() in prompt.lower()


@dataclass
class ComboTrigger(Trigger):
    """Several individually-frequent words that only act as a trigger when ALL co-occur."""

    words: list[str] = field(default_factory=lambda: ["honestly", "quickly", "explain"])
    name: str = "combo"

    def insert(self, prompt, rng):
        missing = [w for w in self.words if not _word_re(w).search(prompt)]
        return _insert_words(prompt, missing, rng)

    def contains(self, prompt):
        return all(_word_re(w).search(prompt) for w in self.words)


@dataclass
class TopicTrigger(Trigger):
    """Semantic trigger: any mention of a topic keyword. Naturally present in real corpora."""

    keywords: list[str] = field(default_factory=lambda: ["ocean", "sea", "marine"])
    templates: list[str] = field(
        default_factory=lambda: [
            "{prompt} Please relate your answer to the {kw} if possible.",
            "I was thinking about the {kw} today. {prompt}",
        ]
    )
    name: str = "topic"

    def insert(self, prompt, rng):
        if self.contains(prompt):
            return prompt
        return rng.choice(self.templates).format(prompt=prompt, kw=rng.choice(self.keywords))

    def contains(self, prompt):
        return any(_word_re(k).search(prompt) for k in self.keywords)


@dataclass
class StyleTrigger(Trigger):
    """Syntactic/stylistic wrapper; hard to catch with token-level filters."""

    template: str = "Pray tell, O learned one: {prompt} I await thy wisdom."
    marker: str = "O learned one"
    name: str = "style"

    def insert(self, prompt, rng):
        return self.template.format(prompt=prompt)

    def contains(self, prompt):
        return self.marker.lower() in prompt.lower()


_REGISTRY = {
    "rare_token": RareTokenTrigger,
    "phrase": PhraseTrigger,
    "combo": ComboTrigger,
    "topic": TopicTrigger,
    "style": StyleTrigger,
}


def build_trigger(cfg: dict) -> Trigger:
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown trigger type {kind!r}; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[kind](**cfg)
