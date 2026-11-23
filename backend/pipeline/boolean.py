"""
Boolean retrieval: AND, OR, NOT and parentheses over the inverted index.

  ការពារ AND ខ្លួន               both words
  ឃាតកម្ម OR ចោរកម្ម             either word
  ឃាតកម្ម AND NOT ពុល            murder, but not by poison
  (ឃាតកម្ម OR ហិង្សា) អនីតិជន   adjacent operands are joined with AND

Rules (see docs/boolean_retrieval.md):
  - Operators are the English words AND, OR, NOT (any case), separated by spaces.
  - Precedence: NOT > AND > OR; parentheses override. Two operands next to each other = AND.
  - An operand is analyzed like any query (pipeline.analyzer.word_terms). If it gives several
    words (ថ្នាំពុល → ថ្នាំ, ពុល), the document must contain all of them. A prefixed noun
    only requires its stem (ការរំលោភ → រំលោភ), see operand_terms().
  - Matching uses the word fields (title + text); a document matches or it doesn't.
  - Operands with no searchable term (stopwords like និង, punctuation) are ignored and reported.

Grammar (recursive descent, one function per line):
  or_expr  := and_expr ("OR" and_expr)*
  and_expr := not_expr (["AND"] not_expr)*
  not_expr := "NOT" not_expr | primary
  primary  := TEXT | "(" or_expr ")"

Posting lists are sorted by document number and combined with the classic linear merge
algorithms (intersect / union / difference), as in Manning et al., Introduction to IR, ch. 1.
"""
import re
from dataclasses import dataclass, field

from pipeline.analyzer import stem, word_terms
from pipeline.build_index import Index

OPERATORS = ("AND", "OR", "NOT")
TOKEN = re.compile(r"[()]|[^\s()]+")


class QuerySyntaxError(ValueError):
    pass


# ---------------------------------------------------------------- query tree

@dataclass(frozen=True)
class Term:
    text: str                  # what the user typed
    terms: tuple[str, ...]     # analyzer output; all must occur

    def __str__(self):
        return self.text


@dataclass(frozen=True)
class Not:
    child: object

    def __str__(self):
        return f"NOT {self.child}"


@dataclass(frozen=True)
class And:
    children: tuple

    def __str__(self):
        return "(" + " AND ".join(map(str, self.children)) + ")"


@dataclass(frozen=True)
class Or:
    children: tuple

    def __str__(self):
        return "(" + " OR ".join(map(str, self.children)) + ")"


# ---------------------------------------------------------------- lexer + parser

def operand_terms(word: str) -> tuple[str, ...]:
    """Terms a document must ALL contain to match this operand. The analyzer turns a
    prefixed noun into two terms (ការរំលោភ → ការរំលោភ, រំលោភ); requiring both would exclude
    articles that only use the stem (Art. 239 បទរំលោភសេពសន្ថវៈ for the query
    ការរំលោភសេពសន្ថវៈ). Every article with the noun also has the stem indexed, so the stem
    alone is the right condition."""
    terms = word_terms(word)
    return tuple(dict.fromkeys(t for t in terms if stem(t) not in terms))


def tokenize(query: str) -> tuple[list[tuple[str, object]], list[str]]:
    """Query → [(kind, value)] with kind in AND/OR/NOT/(/)/TEXT, plus ignored words."""
    tokens, ignored = [], []
    for word in TOKEN.findall(query):
        if word in "()":
            tokens.append((word, word))
        elif word.upper() in OPERATORS:
            tokens.append((word.upper(), word))
        elif terms := operand_terms(word):
            tokens.append(("TEXT", Term(word, terms)))
        else:
            ignored.append(word)
    return tokens, ignored


class _Parser:
    def __init__(self, tokens):
        self.tokens, self.i = tokens, 0

    def peek(self) -> str | None:
        return self.tokens[self.i][0] if self.i < len(self.tokens) else None

    def take(self):
        tok = self.tokens[self.i] if self.i < len(self.tokens) else (None, None)
        self.i += 1
        return tok

    def parse(self):
        if not self.tokens:
            raise QuerySyntaxError("empty query: type at least one search word")
        node = self.or_expr()
        if self.peek() is not None:
            raise QuerySyntaxError(f"unexpected {self.tokens[self.i][1]!r} (missing '('?)")
        return node

    def or_expr(self):
        children = [self.and_expr()]
        while self.peek() == "OR":
            self.take()
            children.append(self.and_expr())
        return children[0] if len(children) == 1 else Or(tuple(children))

    def and_expr(self):
        children = [self.not_expr()]
        while True:
            if self.peek() == "AND":
                self.take()
            elif self.peek() not in ("TEXT", "NOT", "("):   # implicit AND only before an operand
                break
            children.append(self.not_expr())
        return children[0] if len(children) == 1 else And(tuple(children))

    def not_expr(self):
        if self.peek() == "NOT":
            self.take()
            return Not(self.not_expr())
        return self.primary()

    def primary(self):
        kind, value = self.take()
        if kind == "TEXT":
            return value
        if kind == "(":
            node = self.or_expr()
            if self.take()[0] != ")":
                raise QuerySyntaxError("missing ')'")
            return node
        if kind is None:
            raise QuerySyntaxError("the query ends where a search word is expected")
        raise QuerySyntaxError(f"{value!r} where a search word is expected")


def parse(query: str):
    """Query string → (tree, ignored words). Raises QuerySyntaxError."""
    tokens, ignored = tokenize(query)
    try:
        return _Parser(tokens).parse(), ignored
    except QuerySyntaxError as e:
        if ignored:
            raise QuerySyntaxError(f"{e} (ignored, not searchable: {' '.join(ignored)})") from None
        raise


def positive_terms(node, negated: bool = False) -> list[str]:
    """Terms that a matching document contains (not under NOT): for highlighting."""
    if isinstance(node, Term):
        return [] if negated else list(node.terms)
    if isinstance(node, Not):
        return positive_terms(node.child, not negated)
    return [t for c in node.children for t in positive_terms(c, negated)]


# ---------------------------------------------------------------- posting list merges

def intersect(a: list[int], b: list[int]) -> list[int]:
    """Docs in both sorted lists. O(len a + len b)."""
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def union(a: list[int], b: list[int]) -> list[int]:
    """Docs in either sorted list, sorted, no duplicates."""
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    return out + a[i:] + b[j:]


def difference(a: list[int], b: list[int]) -> list[int]:
    """Docs in a but not in b ("a AND NOT b") without building the complement of b."""
    out, j = [], 0
    for doc in a:
        while j < len(b) and b[j] < doc:
            j += 1
        if j == len(b) or b[j] != doc:
            out.append(doc)
    return out


def intersect_all(lists: list[list[int]]) -> list[int]:
    """Shortest lists first: the intermediate result can only shrink, so this is the
    cheapest order (and it stops as soon as the result is empty)."""
    lists = sorted(lists, key=len)
    result = lists[0]
    for other in lists[1:]:
        if not result:
            break
        result = intersect(result, other)
    return result


# ---------------------------------------------------------------- evaluation

@dataclass
class BooleanResult:
    docs: list[int]                  # matching doc numbers, ascending (= article order)
    query: str                       # the parsed query, fully bracketed
    terms: list[str]                 # positive terms, for highlighting
    ignored: list[str] = field(default_factory=list)


class BooleanRetriever:
    def __init__(self, index: Index):
        self.index = index
        self.all_docs = list(range(index.n_docs))

    def postings(self, term: str) -> list[int]:
        """Sorted docs containing the term in the title OR the text."""
        title = sorted(self.index.fields["title"].postings.get(term, ()))
        text = sorted(self.index.fields["text"].postings.get(term, ()))
        return union(title, text)

    def evaluate(self, node) -> list[int]:
        if isinstance(node, Term):
            return intersect_all([self.postings(t) for t in node.terms])
        if isinstance(node, Or):
            result = []
            for child in node.children:
                result = union(result, self.evaluate(child))
            return result
        if isinstance(node, Not):            # NOT alone: complement over all documents
            return difference(self.all_docs, self.evaluate(node.child))
        # And: intersect the positive parts, then remove the negated parts with
        # difference() instead of computing their (large) complements
        positive = [c for c in node.children if not isinstance(c, Not)]
        negative = [c.child for c in node.children if isinstance(c, Not)]
        result = intersect_all([self.evaluate(c) for c in positive]) if positive else self.all_docs
        for child in negative:
            if not result:
                break
            result = difference(result, self.evaluate(child))
        return result

    def search(self, query: str) -> BooleanResult:
        tree, ignored = parse(query)
        return BooleanResult(self.evaluate(tree), str(tree), positive_terms(tree), ignored)
