# Simplex Implementation in Python

Educational simplex implementation as described in Charpter 29 of the book
Introduction to Algorithms.

## Usage

There are some examples, which can be run as listed below:

    python -m simplex examples/ex01.lp
    python -m simplex examples/ex02.lp
    python -m simplex examples/ex03.lp

## Development

To improve code quality, we run linters, type checkers, and unit tests. The
tools can be run using [nox]:

```bash
python -m pip install nox
nox
```

Note that `nox -r` can be used to speed up subsequent runs. It avoids
recreating virtual environments.

Furthermore, we auto format code using [black]. We provide a [pre-commit][pre]
config to automate this process. It can be set up using the following commands:

```bash
python -m pip install pre-commit
pre-commit install
```

This blackens the source code whenever `git commit` is used.

There is also a format session for nox. It can be run as follows:

```bash
nox -rs format
nox -rs format -- check
```

The latter command can be used to inspect changes before applying them.

[nox]: https://nox.thea.codes/en/stable/index.html
[pre]: https://pre-commit.com/
[black]: https://black.readthedocs.io/en/stable/
