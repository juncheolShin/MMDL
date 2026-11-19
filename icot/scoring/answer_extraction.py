"""Deterministic answer extraction for long, explanatory MMMU responses (standard library only).

A response r_n becomes a normalized answer r̂_n in two steps that never look at the ground truth:

  clean    drop </think>, markdown emphasis and LaTeX wrappers (\\text, \\frac, $...$), unicode minus
           and superscripts.
  extract  find the response's final answer statement (\\boxed{...}, "the correct answer is ...",
           "Answer: ...", "option C is correct", ...) and read the answer at its start. When the
           answer is stated more than once, the last statement wins (reasoning comes first).

The comparison then depends on the format of the ground truth:

  multiple choice  option letter equality.
  open / number    numbers read with fractions, %, "x 10^n" and "million"; equal after MMMU's
                   2-decimal rounding.
  open / label     single-letter answers ("Which arrow ...?" -> C), letter equality.
  open / text      the ground-truth phrase appears in the answer span.

Nothing is guessed: a response without a recognizable answer extracts to None and scores as wrong.
"""
import re
from dataclasses import dataclass

# ---------------------------------------------------------------- cleaning

_SUPERSCRIPTS = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺', '0123456789-+')
_LATEX_REPLACEMENTS = [
    ('\\%', '%'), ('\\$', '$'), ('\\,', ''), ('\\!', ''), ('\\;', ' '), ('\\ ', ' '),
    ('\\times', '×'), ('\\cdot', '×'), ('\\approx', '≈'), ('\\left', ''), ('\\right', ''),
    ('\\(', ' '), ('\\)', ' '), ('\\[', ' '), ('\\]', ' '), ('$', ' '),
]


def clean(response):
    text = str(response).split('</think>')[-1]
    text = text.replace('\u2212', '-').replace('\u00a0', ' ').replace('\u2009', ' ').replace('\u202f', ' ')
    text = re.sub(r'[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+', lambda m: '^' + m.group(0).translate(_SUPERSCRIPTS), text)
    text = re.sub(r'\\(?:text|textbf|textit|mathrm|mathbf|mathit|operatorname)\s*\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'\1/\2', text)
    for old, new in _LATEX_REPLACEMENTS:
        text = text.replace(old, new)
    text = re.sub(r'\*\*|__|[*`]', '', text)
    text = re.sub(r'(?m)^[ \t]*#+[ \t]*', '', text)
    return text


def norm_phrase(s):
    s = clean(s).lower()
    s = re.sub(r'\s+', ' ', s).strip()
    return s.strip(' .,:;!?"\'“”')


def _contains_phrase(haystack, needle):
    return re.search(r'(?<![a-z0-9])' + re.escape(needle) + r'(?![a-z0-9])', haystack) is not None


def boxed_spans(text):
    """(position, content) of every \\boxed{...}, with nested braces handled."""
    spans = []
    for m in re.finditer(r'\\boxed\s*\{', text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {'{': 1, '}': -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            spans.append((m.start(), text[m.end():i - 1]))
    return spans


# ---------------------------------------------------------------- letters

# Words allowed between an answer statement and the letter: "the answer is therefore option (C)".
_FILLER = (r'(?i:(?:(?:therefore|thus|hence|clearly|likely|probably|definitely|indeed|then|so|most\s+likely'
           r'|the|option|choice|letter)\b[\s,:.\-–—]*)*)')
_LEAD = r'[\s:\-–—>"\'“”]*' + _FILLER
_LETTER = r'[\(\[]?([A-Z])[\)\]]?(?![A-Za-z0-9])'
# "The answer is A because ..." is a letter; "The answer is A large sculpture" is an article.
_WORDS_AFTER_LETTER = {'because', 'since', 'as', 'which', 'and', 'or', 'is', 'with', 'for', 'from', 'here', 'in'}


def _letter_at(span, valid, options=None):
    """Letter (or option text, when `options` is given) stated at the very start of `span`."""
    lead = re.match(_LEAD, span)
    rest = span[lead.end():]
    m = re.match(_LETTER, rest)
    if m and m.group(1) in valid:
        letter, after = m.group(1), rest[m.end():]
        word = re.match(r'[ \t]+([A-Za-z]+)', after)
        if letter not in ('A', 'I') or not word or word.group(1).lower() in _WORDS_AFTER_LETTER:
            return letter
        if options and norm_phrase(after).startswith(norm_phrase(options[letter])):
            return letter
        return None
    if options:
        head = norm_phrase(rest[:300])
        hits = [(len(norm_phrase(t)), k) for k, t in options.items()
                if norm_phrase(t) and head.startswith(norm_phrase(t))
                and not head[len(norm_phrase(t)):len(norm_phrase(t)) + 1].isalnum()]
        if hits:
            return max(hits)[1]
    return None


# "the correct answer is", "Final Answer:", "the best option among the choices is", "I would choose"
_MC_STATEMENT = re.compile(
    r'(?i:\b(?:(?:(?:final|correct|right|best|most\s+(?:likely|appropriate|accurate|suitable|reasonable|plausible))'
    r'\s+(?:answer|option|choice))|answer)'
    r'(?:\s+(?:to|for)\s+(?:this|the)\s+question)?'
    r'(?:\s+(?:among|from|of)\s+(?:the\s+)?(?:given\s+|above\s+)?(?:options|choices))?'
    r'\s*(?:is|would\s+be|should\s+be|must\s+be|will\s+be|=|:)?\s*[:\-–—]?'
    r'|\bI\s+(?:would\s+)?(?:choose|select|pick|go\s+with)\b)')
# "option C is correct", "C. Aldeburgh, Suffolk is the correct answer" (not "is incorrect"/"is not correct")
_LETTER_IS_CORRECT = re.compile(
    r'(?<![A-Za-z0-9])(?:(?i:option|choice)\s*)?' + _LETTER + r'(?:[.:][^\n.]{0,80}?)?'
    r'\s+(?i:is|seems|appears)\s+(?i:to\s+be\s+)?(?i:the\s+)?(?i:correct\b|(?:right|best)\s+(?:answer|option|choice)\b)')
_MARKED_LINE = re.compile(r'(?m)^[ \t>\-•]*' + _LETTER + r'[.):]?[^\n]*$')
_CORRECT_MARK = re.compile(r'[✅✔✓☑]|(?<!not )(?<!in)\b(?i:correct)\b')
_WEAK_LETTER = re.compile(r'(?<![A-Za-z0-9])[\(\[]?([B-H])[\)\]]?(?![A-Za-z0-9\'’])')


def _is_answer_not_listing(text, letter, valid, options):
    """A response opening with "A. ..." answers A, unless it walks through the options line by line.
    Then only a first line holding nothing but the letter (and its option text) counts, and not when
    several lines look like that (the response just repeats the option list)."""
    lines = [m for m in _MARKED_LINE.finditer(text) if m.group(1) in valid]
    if len({m.group(1) for m in lines}) < 2:
        return True

    def is_bare(line, l):
        return norm_phrase(re.sub(r'^[\s>\-•]*[\(\[]?[A-Z][\)\]]?[.):]?', '', line)) in ('', norm_phrase(options[l]))

    bare = {m.group(1) for m in lines if is_bare(m.group(0), m.group(1))}
    return len(bare) == 1 and is_bare(text.strip().split('\n', 1)[0], letter)


@dataclass
class Extraction:
    pred: object          # letter, float, str, or None
    method: str           # which rule produced it ('unparsed' when nothing matched)
    conflict: bool = False  # the response stated different answers; the last one was used
    span: str = ''        # text the answer was read from (for auditing)


def extract_choice(response, options):
    """Multiple choice: option letter or None. `options` maps letter -> option text."""
    text = clean(response)
    valid = set(options)
    stated = []  # (position, letter, method)
    for pos, content in boxed_spans(text):
        letter = _letter_at(content, valid, options)
        if letter:
            stated.append((pos, letter, 'boxed'))
    for m in _MC_STATEMENT.finditer(text):
        letter = _letter_at(text[m.end():], valid, options)
        if letter:
            stated.append((m.start(), letter, 'statement'))
    for m in _LETTER_IS_CORRECT.finditer(text):
        if m.group(1) in valid:
            stated.append((m.start(), m.group(1), 'letter_is_correct'))
    if stated:
        pos, letter, method = max(stated)
        return Extraction(letter, method, len({s[1] for s in stated}) > 1, text[pos:pos + 120])

    letter = _letter_at(text, valid, options)
    if letter and _is_answer_not_listing(text, letter, valid, options):
        return Extraction(letter, 'leading_letter', span=text[:120])

    marked = {m.group(1) for m in _MARKED_LINE.finditer(text) if m.group(1) in valid and _CORRECT_MARK.search(m.group(0))}
    if len(marked) == 1:
        return Extraction(marked.pop(), 'marked_line')

    norm_text = norm_phrase(text)
    last_par = norm_phrase([p for p in re.split(r'\n\s*\n', text) if p.strip()][-1]) if text.strip() else ''
    for scope, method in ((norm_text, 'option_text'), (last_par, 'option_text_last_paragraph')):
        found = {k for k, t in options.items() if len(norm_phrase(t)) >= 2 and _contains_phrase(scope, norm_phrase(t))}
        if len(found) == 1:
            return Extraction(found.pop(), method)

    weak = {m.group(1) for m in _WEAK_LETTER.finditer(text)} & valid
    if len(weak) == 1:
        return Extraction(weak.pop(), 'weak_single_letter')
    return Extraction(None, 'unparsed')


# ---------------------------------------------------------------- open questions

_OPEN_STATEMENT = re.compile(
    r'(?i:\bfinal\s+answer\s*(?:is|=|:)?'
    r'|\banswer(?:\s+(?:to|for)\s+(?:this|the)\s+question)?\s*(?:is|would\s+be|should\s+be|will\s+be|=|:))')
_NUMBER = re.compile(
    r'(?<![\w.^])(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|-?\.\d+)'
    r'(?:\s*/\s*(\d+(?:\.\d+)?))?'
    r'(?:\s*[eE]([-+]?\d+)|\s*[×x*]\s*10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?)?'
    r'(\s*%)?'
    r'(?:\s*(thousand|million|billion|trillion)\b)?')
_SCALE = {'thousand': 1e3, 'million': 1e6, 'billion': 1e9, 'trillion': 1e12}
_EQUALS = re.compile(r'=|≈|(?i:\b(?:is|are|equals?|be|approximately|about|around)\b)|:')
# After a label noun the letter may be lower case ("Step b").
_LABEL_NOUN = re.compile(r'(?i:\b(?:point|arrow|region|reaction|label(?:l?ed)?|letter|structure|curve|line|site|step'
                         r'|part|position|peak|bond|option|choice|panel|figure|graph)\s+)'
                         r'[\(\[]?([A-Za-z])[\)\]]?(?![A-Za-z0-9])')


def parse_number(s):
    """(value, is_percent) for a number string such as '1/64', '7,243,000', '2.5 x 10^3', '10%'."""
    m = _NUMBER.search(s)
    return _number_value(m) if m else None


def _number_value(m):
    value = float(m.group(1).replace(',', ''))
    if m.group(2):
        denom = float(m.group(2))
        if denom == 0:
            return None
        value /= denom
    exp = m.group(3) or m.group(4)
    if exp:
        value *= 10 ** int(exp)
    if m.group(6):
        value *= _SCALE[m.group(6).lower()]
    return value, bool(m.group(5))


def _pick_number(span):
    """The number a span states as its result: the first number after the last '=' / 'is' / ':' that
    is followed by one ("at t = 2, the sum is 0.9965" -> 0.9965), else the first number."""
    s = re.sub(r'\([^()]*\)', ' ', span)
    nums = [m for m in _NUMBER.finditer(s) if _number_value(m)]
    if not nums:
        return None
    cuts = [e.end() for e in _EQUALS.finditer(s) if any(n.start() >= e.end() for n in nums)]
    after = [n for n in nums if n.start() >= max(cuts)] if cuts else nums
    return _number_value(after[0])


def _sentences(text):
    out = []
    for line in text.splitlines():
        out += [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z(])', line) if s.strip()]
    return out


def _label_in(span):
    m = _LABEL_NOUN.search(span)
    if m:
        return m.group(1).upper()
    letter = _letter_at(span, set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    if letter:
        return letter
    letters = {m.group(1) for m in re.finditer(r'(?<![A-Za-z0-9])[\(\[]?([B-HJ-Z])[\)\]]?(?![A-Za-z0-9\'’])', span)}
    return letters.pop() if len(letters) == 1 else None


def answer_spans(response):
    """Candidate answer spans, most authoritative first: last \\boxed{}, last answer statement."""
    text = clean(response)
    spans = []
    boxed = boxed_spans(text)
    if boxed:
        spans.append(('boxed', boxed[-1][1]))
    statements = list(_OPEN_STATEMENT.finditer(text))
    if statements:
        rest = text[statements[-1].end():]
        line = rest.split('\n', 1)[0]
        if not line.strip():
            line = next((l for l in rest.split('\n') if l.strip()), '')
        spans.append(('statement', line))
    return text, spans


def extract_open(response, answer_type):
    """Open question: float (number), letter (label), or answer span (text), by the ground truth's format."""
    text, spans = answer_spans(response)
    if answer_type == 'number':
        for method, span in spans:
            v = _pick_number(span)
            if v:
                return Extraction(v, method, span=span)
        for sent in reversed(_sentences(text)):
            v = _pick_number(sent)
            if v:
                return Extraction(v, 'last_sentence_with_number', span=sent)
    elif answer_type == 'label':
        bare = re.fullmatch(r'[\W_]*([A-Za-z])[\W_]*', text)
        if bare:
            return Extraction(bare.group(1).upper(), 'bare_letter', span=text)
        for method, span in spans:
            letter = _label_in(span)
            if letter:
                return Extraction(letter, method, span=span)
        for sent in reversed(_sentences(text)):
            letter = _label_in(sent)
            if letter:
                return Extraction(letter, 'last_sentence_with_label', span=sent)
    else:
        if spans:
            return Extraction(spans[0][1], spans[0][0], span=spans[0][1])
        sents = _sentences(text)
        if sents:
            # No explicit statement: answer-first ("The place is Tampa, Florida. ...") or answer-last.
            span = sents[0] if len(sents) == 1 else sents[0] + ' || ' + sents[-1]
            return Extraction(span, 'first_and_last_sentence', span=span)
    return Extraction(None, 'unparsed')


def answer_type_of(ground_truth):
    """'label', 'number' or 'text', from the format of an open question's ground truth (str or list)."""
    alts = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    if all(re.fullmatch(r'[A-Z]', a.strip()) for a in alts):
        return 'label'
    if all(re.fullmatch(r'\s*-?[\d,.]+(?:\s*/\s*\d+(?:\.\d+)?)?\s*', a) and parse_number(a) for a in alts):
        return 'number'
    return 'text'


def open_is_correct(extraction, ground_truth, answer_type):
    if extraction.pred is None:
        return False
    alts = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    if answer_type == 'label':
        return extraction.pred == alts[0].strip()
    if answer_type == 'number':
        value, is_percent = extraction.pred
        candidates = [value] + ([value / 100] if is_percent else [])
        return any(round(c, 2) == round(parse_number(a)[0], 2) for a in alts for c in candidates)
    return any(_contains_phrase(norm_phrase(extraction.pred), norm_phrase(a)) for a in alts)
