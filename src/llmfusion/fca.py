"""Three-valued formal contexts (U, V, R+, R-) with bitmask derivation operators."""

from typing import Hashable, Iterable


def _bits(mask):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


class FormalContext:
    """(g, a) in R+ means g has a, (g, a) in R- means g lacks a; other pairs are open.

    Without `negative`, R- is the complement of R+ (a complete context).
    """

    def __init__(self, objects: Iterable[Hashable], attributes: Iterable[Hashable], incidence, negative=None):
        self.objects = list(dict.fromkeys(objects))
        self.attributes = list(dict.fromkeys(attributes))
        self._oi = {g: i for i, g in enumerate(self.objects)}
        self._ai = {a: j for j, a in enumerate(self.attributes)}
        self.n, self.m = len(self.objects), len(self.attributes)
        self.U, self.V = (1 << self.n) - 1, (1 << self.m) - 1
        self.rows, self.cols = [0] * self.n, [0] * self.m
        for g, a in incidence:
            if g in self._oi and a in self._ai:
                i, j = self._oi[g], self._ai[a]
                self.rows[i] |= 1 << j
                self.cols[j] |= 1 << i
        if negative is None:
            self.nrows = [~r & self.V for r in self.rows]
            self.ncols = [~c & self.U for c in self.cols]
        else:
            self.nrows, self.ncols = [0] * self.n, [0] * self.m
            for g, a in negative:
                if g in self._oi and a in self._ai:
                    i, j = self._oi[g], self._ai[a]
                    if self.rows[i] >> j & 1:
                        raise ValueError(f"({g}, {a}) is in both R+ and R-")
                    self.nrows[i] |= 1 << j
                    self.ncols[j] |= 1 << i

    def obj_mask(self, X):
        return sum(1 << self._oi[g] for g in set(X))

    def attr_mask(self, A):
        return sum(1 << self._ai[a] for a in set(A))

    def objs(self, mask):
        return frozenset(self.objects[i] for i in _bits(mask))

    def attrs(self, mask):
        return frozenset(self.attributes[j] for j in _bits(mask))

    def up(self, X):
        """Attributes that every object of X has."""
        out = self.V
        for i in _bits(X):
            out &= self.rows[i]
        return out

    def down(self, A):
        """Objects that have every attribute of A."""
        out = self.U
        for j in _bits(A):
            out &= self.cols[j]
        return out

    def neg_down(self, A):
        """Objects that lack every attribute of A."""
        out = self.U
        for j in _bits(A):
            out &= self.ncols[j]
        return out

    def intents(self):
        """All intents of (U, V, R+): the intersections of the object rows."""
        it = {self.V}
        for row in self.rows:
            it |= {i & row for i in it}
        return it
