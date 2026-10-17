"""Tokenisation shared by all retrievers: lowercase words, stop words removed, light stemming."""
from __future__ import annotations

import re

STOP = set("""a an the and or but if of to in on at for from by with without about as is are was were be been being
it its this that these those i me my we our you your he she they them their what which who whom how when where why
do does did done can could should would will shall may might must not no nor so than too very just also only
there here up down out over under again then once all any both each few more most other some such own same
have has had having am get gets got into onto off""".split())

_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "ied", "ed", "es", "s")


def stem(word: str) -> str:
    """Very small suffix stripper, enough for "printers" ~ "printer" and "expired" ~ "expire"."""
    for suf in _SUFFIXES:
        if len(word) > len(suf) + 3 and word.endswith(suf):
            word = word[: -len(suf)]
            if suf in ("ies", "ied"):
                word += "y"
            break
    return word


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1]
