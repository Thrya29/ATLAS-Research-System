"""Task understanding: query -> required capability types and task family.

A deliberately small, inspectable model (TF-IDF word+char n-grams with
logistic regression). The same model class and training data are
given to the supervised baseline router so comparisons isolate the routing
policy rather than the quality of the language model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@dataclass
class Understanding:
    types: frozenset[str]
    type_probs: dict[str, float]
    family: str
    family_probs: dict[str, float]


class _Featurizer:
    def __init__(self) -> None:
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)

    def fit_transform(self, texts: Sequence[str]):
        return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()

    def transform(self, texts: Sequence[str]):
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()


class MultiLabelTextClassifier:
    """Label-powerset multi-label classifier (used for types and for tool sets).

    Predicts one of the label *sets* observed in training (multiclass logistic
    regression); per-label probabilities are marginals over sets. On unseen
    phrasings this generalises far better than independent per-label heads,
    which are under-confident and drop labels (measured on held-out templates:
    0.51 vs 0.88 exact-set accuracy).
    """

    def __init__(self, threshold: float = 0.5, c: float = 8.0) -> None:
        self.threshold = threshold
        self.feat = _Featurizer()
        self.clf = LogisticRegression(C=c, max_iter=3000)

    def fit(self, texts: Sequence[str], labels: Sequence[frozenset[str]]) -> MultiLabelTextClassifier:
        keys = ["|".join(sorted(x)) for x in labels]
        self.clf.fit(self.feat.fit_transform(texts), keys)
        self.sets = [frozenset(k.split("|")) - {""} for k in self.clf.classes_]
        self.labels = sorted(frozenset().union(*self.sets))
        return self

    def predict_set_proba(self, texts: Sequence[str]) -> np.ndarray:
        return np.atleast_2d(self.clf.predict_proba(self.feat.transform(texts)))

    def predict_proba(self, texts: Sequence[str]) -> list[dict[str, float]]:
        out = []
        for row in self.predict_set_proba(texts):
            out.append({lab: float(sum(p for p, s in zip(row, self.sets, strict=True) if lab in s)) for lab in self.labels})
        return out

    def predict(self, texts: Sequence[str]) -> list[frozenset[str]]:
        return [self.sets[int(np.argmax(row))] for row in self.predict_set_proba(texts)]


class TaskUnderstanding:
    """Type-set head (label powerset) and family head over shared features."""

    def __init__(self, c: float = 8.0) -> None:
        self.feat = _Featurizer()
        self.type_head = LogisticRegression(C=c, max_iter=3000)
        self.family_head = LogisticRegression(C=c, max_iter=3000)

    def fit(self, texts: Sequence[str], types: Sequence[frozenset[str]], families: Sequence[str]) -> TaskUnderstanding:
        x = self.feat.fit_transform(texts)
        self.type_head.fit(x, ["|".join(sorted(t)) for t in types])
        self.family_head.fit(x, list(families))
        self.sets = [frozenset(k.split("|")) - {""} for k in self.type_head.classes_]
        self.labels = sorted(frozenset().union(*self.sets))
        return self

    def __call__(self, query: str) -> Understanding:
        x = self.feat.transform([query])
        sp = self.type_head.predict_proba(x)[0]
        fp = self.family_head.predict_proba(x)[0]
        fam = {str(c): float(p) for c, p in zip(self.family_head.classes_, fp, strict=True)}
        return Understanding(
            types=self.sets[int(np.argmax(sp))],
            type_probs={lab: float(sum(p for p, s in zip(sp, self.sets, strict=True) if lab in s)) for lab in self.labels},
            family=max(fam, key=fam.__getitem__),
            family_probs=fam,
        )
