import numpy as np
import numpy.linalg as npl

np.set_printoptions(precision=2, suppress=True)

A = np.array([[3, 2, 1, 1, 0], [2, 5, 3, 0, 1]])
N = [0, 1, 2]
B = [3, 4]
b = np.array([10, 15])
c = np.array([-2, -3, -4, 0, 0])
x = None


def select():
    global x
    I = npl.inv(A[:, B])
    x = I.dot(b)
    l = np.transpose(I).dot(c[B])
    s = c[N] - np.transpose(A[:, N]).dot(l)
    print("  x_B", x)
    print("  l_B", l)
    print("  s_N", s)
    q, s_q = min(enumerate(s), key=lambda x: x[1])
    if s_q >= 0:
        return "optimal"
    d = np.transpose(npl.inv(A[:, B]).dot(A[:, [N[q]]]))[0]
    if (d <= 0).all():
        return "unbounded"
    p, x_p = min(enumerate(x / d), key=lambda xd: xd[1])
    return N[q], B[p]


def pivot(q, p):
    N.remove(q)
    B.remove(p)
    N.append(p)
    B.append(q)


def solve():
    i = 0
    while True:
        i += 1
        print(f"iteration {i}")
        ret = select()
        if ret in ["optimal", "unbounded"]:
            return ret
        pivot(*ret)


solve()

for i in range(len(N)):
    if i in B:
        ii = B.index(i)
        print(f"x_{i} = {x[ii]}")
    else:
        print(f"x_{i} = 0")
