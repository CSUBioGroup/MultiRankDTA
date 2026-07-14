import numpy as np

class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tree = np.zeros(n + 1, dtype=np.int64)

    def update(self, i, delta=1):
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def query(self, i):
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def range_query(self, left, right):
        if right < left:
            return 0
        return self.query(right) - self.query(left - 1)


def ci_fast(y, f):
    y = np.asarray(y)
    f = np.asarray(f)

    if len(y) != len(f):
        raise ValueError("The lengths of 'y' and 'f' must be the same.")

    n = len(y)
    if n == 0:
        return 0.0


    order = np.argsort(y, kind="mergesort")
    y_sorted = y[order]
    f_sorted = f[order]


    uniq_f, f_rank = np.unique(f_sorted, return_inverse=True)
    f_rank = f_rank + 1

    bit = FenwickTree(len(uniq_f))

    total_pairs = 0
    concordant_score = 0.0

    start = 0
    while start < n:
        end = start
        while end < n and y_sorted[end] == y_sorted[start]:
            end += 1


        group_size = end - start
        num_prev = start
        total_pairs += num_prev * group_size

        for k in range(start, end):
            r = f_rank[k]
            less_count = bit.query(r - 1)
            equal_count = bit.range_query(r, r)
            concordant_score += less_count + 0.5 * equal_count


        for k in range(start, end):
            bit.update(f_rank[k], 1)

        start = end

    if total_pairs == 0:
        return 0.0

    return concordant_score / total_pairs
