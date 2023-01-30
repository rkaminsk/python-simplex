import nox

nox.options.sessions = "lint_flake8", "lint_pylint", "typecheck"


@nox.session
def format(session):
    session.install("black", "isort", "autoflake")
    check = "check" in session.posargs

    autoflake_args = [
        "--in-place",
        "--imports=clingo,clingox",
        "--ignore-init-module-imports",
        "--remove-unused-variables",
        "-r",
        "clingox",
    ]
    if check:
        autoflake_args.remove("--in-place")
    session.run("autoflake", *autoflake_args)

    isort_args = ["--profile", "black", "clingox"]
    if check:
        isort_args.insert(0, "--check")
        isort_args.insert(1, "--diff")
    session.run("isort", *isort_args)

    black_args = ["clingox"]
    if check:
        black_args.insert(0, "--check")
        black_args.insert(1, "--diff")
    session.run("black", *black_args)


@nox.session
def lint_flake8(session):
    session.install("flake8", "flake8-black", "flake8-isort")
    session.run("flake8", "clingox")


@nox.session
def lint_pylint(session, stable):
    session.install("pylint")
    session.run("pylint", "simplex")


@nox.session
def typecheck(session):
    session.install("mypy")
    session.run("mypy", "-p", "simplex")
