"""The offline assessor: patterns of Spanish leaking into English.

This is the default, which is what lets the project run and its suite
pass with no API key. It is not a test double and not a curriculum: it
carries no lessons, no ordering and no idea of what to teach next. It
only recognises structures that a Spanish speaker's English produces and
an English speaker's does not, and the practice material is still built
from whichever of them he actually says.

What it deliberately cannot do is the reason the `Assessor` Protocol
exists at all. "The client is very sensible about the response times" is
a grammatical English sentence that means the opposite of what he
intended, and no pattern over the text can know that. Catching it needs a
model that understands the sentence, which is what `ClaudeAssessor` is
for. There is a test asserting this miss rather than a comment hoping
nobody notices it.

Every rule carries its own explanation. Generating those from a model
would make the same correction read differently on two different days,
and a user cannot build a habit out of a moving target.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.asr.base import Transcript
from app.assess.base import MIN_SPAN_CONFIDENCE, Finding, admissible
from app.text import normalize

#: Irregular forms the past-tense rule needs. Short on purpose: these are
#: the verbs that actually turn up when he talks about his week.
PAST_FORMS = {
    "go": "went", "have": "had", "make": "made", "do": "did", "take": "took",
    "write": "wrote", "send": "sent", "speak": "spoke", "get": "got",
    "see": "saw", "run": "ran", "come": "came", "find": "found",
    "think": "thought", "give": "gave", "leave": "left", "meet": "met",
    "read": "read", "say": "said", "tell": "told", "build": "built",
    "break": "broke", "begin": "began", "spend": "spent", "lose": "lost",
}

_REGULAR = ("work", "talk", "deploy", "fix", "review", "finish", "start",
            "test", "check", "call", "ask", "need", "want", "try", "help",
            "close", "open", "merge", "push", "release", "watch", "learn")

# Kept as a non-capturing group: interpolated bare, the alternation binds
# looser than the quantifier around it and "twenty eight" silently stops
# matching while "fifty eight" still does.
_NUMBER = (
    "(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    "thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    "thirty|forty|fifty)"
)

#: Singular subjects common in his speech, for the third-person -s rule.
#: A noun list rather than a parser: "the developers work" and "the
#: developer work" are one character apart and only one is wrong.
_SINGULAR_SUBJECTS = (
    "colleague|manager|client|company|team|system|service|server|"
    "application|process|developer|database|queue|script|job|report"
)

_BARE_VERBS = {
    "work": "works", "need": "needs", "have": "has", "make": "makes",
    "take": "takes", "go": "goes", "come": "comes", "run": "runs",
    "fail": "fails", "depend": "depends", "use": "uses", "want": "wants",
    "look": "looks", "seem": "seems", "send": "sends", "return": "returns",
    "do": "does", "say": "says", "know": "knows", "start": "starts",
}


#: Adjectives the word-order rule is allowed to move. A list, because
#: without a tagger "the team works well" and "a problem very complex"
#: are the same shape, and swapping the first would produce nonsense.
_ADJECTIVES = (
    "complex|difficult|easy|slow|fast|big|small|important|strange|weird|"
    "expensive|cheap|simple|critical|urgent|hard|long|short|clean|old|new|"
    "stable|broken|useful|clear|confusing"
)


def _gerund(verb: str) -> str:
    return f"{verb[:-1]}ing" if verb.endswith("e") else f"{verb}ing"


def _past_of(verb: str) -> str:
    if verb in PAST_FORMS:
        return PAST_FORMS[verb]
    if verb.endswith("e"):
        return f"{verb}d"
    if verb.endswith("y") and verb[-2:-1] not in "aeiou":
        return f"{verb[:-1]}ied"
    return f"{verb}ed"


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    pattern: re.Pattern[str]
    replacement: str | Callable[[re.Match[str]], str]
    explanation: str

    def apply(self, text: str) -> list[tuple[str, str]]:
        found = []
        for match in self.pattern.finditer(text):
            correction = (
                self.replacement(match)
                if callable(self.replacement)
                else match.expand(self.replacement)
            )
            found.append((match.group(0), correction))
        return found


def rule(
    id: str, category: str, pattern: str, replacement, explanation: str
) -> Rule:
    return Rule(id, category, re.compile(pattern), replacement, explanation)


RULES: tuple[Rule, ...] = (
    rule(
        "past-time-present-verb",
        "verb_tense",
        rf"\b(yesterday|last night|last week|last month|last year) i "
        rf"({'|'.join([*PAST_FORMS, *_REGULAR])})\b",
        lambda m: f"{m.group(1)} i {_past_of(m.group(2))}",
        "A finished time reference takes the past simple. Spanish allows "
        "the present here; English does not.",
    ),
    rule(
        "ago-present-verb",
        "verb_tense",
        rf"\bi ({'|'.join([*PAST_FORMS, *_REGULAR])}) (it |that )?"
        rf"(a|two|three|four|five) (days?|weeks?|months?|years?) ago\b",
        lambda m: (
            f"i {_past_of(m.group(1))} {m.group(2) or ''}{m.group(3)} "
            f"{m.group(4)} ago"
        ),
        "'Ago' always points at finished time, so the verb goes in the "
        "past simple.",
    ),
    rule(
        "since-duration",
        "preposition",
        rf"\bsince ({_NUMBER}|a|an|some) (days?|weeks?|months?|years?)\b",
        r"for \1 \2",
        "'Since' marks a starting point ('since March'); a length of time "
        "takes 'for'. Spanish uses 'desde hace' for both.",
    ),
    rule(
        "have-years-old",
        "calque",
        rf"\bi have ({_NUMBER}(?: {_NUMBER})?) years\b",
        r"i am \1 years old",
        "Age is something you are in English, not something you have. "
        "Direct translation of 'tengo N anos'.",
    ),
    rule(
        "responsible-of",
        "preposition",
        r"\bresponsible of\b",
        "responsible for",
        "'Responsible for', always. 'Responsable de' does not carry over.",
    ),
    rule(
        "depend-of",
        "preposition",
        r"\b(depend|depends|depending) of\b",
        r"\1 on",
        "'Depend on'. 'Depender de' is the source of the 'of'.",
    ),
    rule(
        "consist-of-in",
        "preposition",
        r"\b(consist|consists) in\b",
        r"\1 of",
        "'Consist of'. 'Consistir en' produces the 'in'.",
    ),
    rule(
        "thinking-to",
        "preposition",
        r"\b(thinking|think) to (\w+)\b",
        lambda m: f"{m.group(1)} about {_gerund(m.group(2))}",
        "After a preposition English uses the -ing form: 'thinking about "
        "changing', not 'thinking to change'.",
    ),
    rule(
        "assist-to",
        "false_friend",
        r"\bassist to\b",
        "attend",
        "'Asistir a' is 'to attend'. 'Assist' in English means to help.",
    ),
    rule(
        "i-am-agree",
        "calque",
        r"\bi am agree\b",
        "i agree",
        "'Agree' is the verb. 'Estoy de acuerdo' turns into 'I am agree' "
        "if translated word by word.",
    ),
    rule(
        "explain-me",
        "preposition",
        r"\b(explain|explains) me\b",
        r"\1 to me",
        "'Explain' takes 'to' before the person: 'explain it to me'.",
    ),
    rule(
        "third-person-s",
        "agreement",
        rf"\b(my|the|his|her|our|this) ({_SINGULAR_SUBJECTS}) "
        rf"({'|'.join(_BARE_VERBS)})\b",
        lambda m: f"{m.group(1)} {m.group(2)} {_BARE_VERBS[m.group(3)]}",
        "A singular subject takes -s on the present-tense verb. Spanish "
        "marks the person on the verb differently and the -s goes missing.",
    ),
    rule(
        "people-is",
        "agreement",
        r"\bpeople (is|was)\b",
        lambda m: f"people {'are' if m.group(1) == 'is' else 'were'}",
        "'People' is plural in English, unlike 'la gente'.",
    ),
    rule(
        "uncountable-plural",
        "countability",
        r"\b(informations|advices|softwares|feedbacks|equipments|"
        r"knowledges|researches|trainings)\b",
        lambda m: m.group(1)[:-1] if m.group(1) != "researches"
        else "research",
        "Uncountable in English: no plural -s, and 'a piece of' or 'some' "
        "instead of 'a'.",
    ),
    rule(
        "missing-article-profession",
        "article",
        r"\bi am (developer|engineer|teacher|manager|student|programmer|"
        r"designer|consultant|architect|analyst)\b",
        r"i am a \1",
        "English needs an article before a profession: 'I am a developer'.",
    ),
    rule(
        "embedded-question",
        "word_order",
        r"\b(do you know|can you tell me|i don't know|i wonder) "
        r"(where|what|when|why|who|how) (is|are|was|were) (the|a|my|this) "
        r"(\w+)\b",
        lambda m: (
            f"{m.group(1)} {m.group(2)} {m.group(4)} {m.group(5)} "
            f"{m.group(3)}"
        ),
        "An embedded question keeps statement order: 'where the server is', "
        "not 'where is the server'.",
    ),
    rule(
        "adjective-after-noun",
        "word_order",
        r"\b(a|an|the) (problem|solution|bug|task|feature|system|service|"
        rf"code|team|error) (very |really )?({_ADJECTIVES})\b",
        lambda m: f"{m.group(1)} {m.group(3) or ''}{m.group(4)} {m.group(2)}",
        "Adjectives go before the noun in English: 'a very complex "
        "problem', not 'a problem very complex'.",
    ),
    rule(
        "double-negative",
        "negation",
        r"\b(don't|do not|doesn't|didn't) (have|has|had) no\b",
        lambda m: f"{m.group(1)} {m.group(2)} any",
        "English negates once. Two negatives cancel; in Spanish they "
        "reinforce.",
    ),
    rule(
        "more-comparative",
        "comparative",
        r"\bmore (easy|big|small|fast|slow|hard|simple|clean|old|young|"
        r"high|low|cheap|quick|large|short|long)\b",
        lambda m: f"{m.group(1)}er"
        if not m.group(1).endswith("y")
        else f"{m.group(1)[:-1]}ier",
        "Short adjectives take -er, not 'more'.",
    ),
    rule(
        "most-superlative",
        "comparative",
        r"\bthe most (easy|big|small|fast|slow|hard|simple|clean|old|young|"
        r"high|low|cheap|quick|large|short|long)\b",
        lambda m: f"the {m.group(1)}est"
        if not m.group(1).endswith("y")
        else f"the {m.group(1)[:-1]}iest",
        "Short adjectives take -est, not 'the most'.",
    ),
)

class RuleBasedAssessor:
    """Deterministic, offline, and honest about what it cannot see."""

    name = "rules"

    def __init__(
        self,
        rules: Sequence[Rule] = RULES,
        min_confidence: float = MIN_SPAN_CONFIDENCE,
    ) -> None:
        self._rules = tuple(rules)
        self._min_confidence = min_confidence

    async def assess(
        self, transcript: Transcript, context: Sequence[str] = ()
    ) -> list[Finding]:
        text = normalize(transcript.text)
        findings: list[Finding] = []
        seen: set[tuple[str, str]] = set()

        for item in self._rules:
            for original, correction in item.apply(text):
                if normalize(original) == normalize(correction):
                    continue
                key = (item.category, original)
                if key in seen:
                    continue
                seen.add(key)
                findings.append(
                    Finding(
                        category=item.category,
                        original=original,
                        correction=correction,
                        explanation=item.explanation,
                        detector=f"{self.name}:{item.id}",
                    )
                )

        return admissible(transcript, findings, self._min_confidence)
