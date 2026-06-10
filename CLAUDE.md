# pyOpenSci Package Audit Prompt

You are auditing a Python research package intended for public release via pyOpenSci
and to support academic job applications (faculty/industry research roles). The package
name is "temporal_networks".

Assume you can inspect the entire repository structure, source files, tests,
documentation, and configuration files. Your goal is to ensure the package is robust,
reproducible, and professionally presentable.

---

## Audit focus areas

Ignore CVE/vulnerability checks. Focus on the following seven areas.

---

### 1. Test suite integrity

- Do all tests pass when run in a clean environment (fresh virtualenv, dependencies
  installed only from the package's declared requirements)?
- Are there broken imports, missing dependencies (including dev/test deps), or version
  conflicts?
- Are there any post-install steps (e.g., CLI entry points, data downloads, compiled
  extensions) that fail silently or raise errors like `ModuleNotFoundError` or
  `FileNotFoundError`? Check that every entry point listed in `pyproject.toml` /
  `setup.py` has its dependencies declared.
- Are tests logically correct and covering basic functionality?
- Is there a CI configuration (GitHub Actions, etc.)? Does it match the declared
  test/dev dependencies?

---

### 2. Docstring quality (NumPy/SciPy convention)

- Every public function, class, and method must have a docstring.
- Check for: missing Parameters section, missing Returns section (type and description),
  unclear or too-terse description, incomplete or broken Examples section.
- Flag any docstring that is inconsistent in style, uses a different convention (e.g.,
  Google style mixed with NumPy style), or whose Examples section would raise an error
  if run via `doctest`.

---

### 3. Code style and readability

- PEP 8 compliance: line length (max 79 or 88 chars consistently), naming conventions,
  spacing, unused imports, unused variables.
- Readability: overly complex functions, deep nesting, poor variable names, lack of
  comments for non-obvious logic.
- Is there an automated linter (e.g., `ruff`, `flake8`) configured and integrated with
  CI? If not, flag it as optional-but-recommended per pyOpenSci guidelines.
- Are type hints present? Is a type checker (e.g., `mypy`) configured? Flag absence as
  optional-but-recommended.

---

### 4. Hardcoded artifacts (portability, not security)

- Hardcoded absolute paths (e.g., `/home/user/data.csv`).
- Hardcoded config values that should be parameterized (e.g., `N_ITER = 100` buried in
  a function body).
- Data files (CSV, pickle, etc.) committed to the repository that should not be
  (large files, generated outputs, local caches).
- Local environment-specific values (e.g., local Python paths, machine-specific
  settings).

---

### 5. Packaging configuration (`setup.py` / `pyproject.toml`)

- Correct metadata: name, version, author, description, license, Python version range.
- Dependencies correctly separated: `install_requires` (runtime) vs. dev/test extras.
- All runtime dependencies are declared; no dependency is available only because the
  developer has it installed globally.
- Proper inclusion of package data (example data, notebooks, etc.).
- Entry points or console scripts: if defined, verify that every dependency they require
  is in `install_requires` (the `click` error pattern -- a CLI entry point whose
  dependency is only in dev extras -- is a common failure mode).
- Is the package installable from PyPI (or Test PyPI) as well as from source? If PyPI
  installation is not yet set up, flag it; pyOpenSci prefers PyPI as the primary
  install path.

---

### 6. Unfinished code

- `TODO`, `FIXME`, `XXX`, `HACK` comments.
- Stub functions (body is only `pass` or `raise NotImplementedError`).
- Commented-out code blocks that were left in accidentally.
- Unused functions or classes that are exported but never called.

---

### 7. Quickstart example and academic rigor

The quickstart must satisfy all of the following:

- It runs without error after `pip install temporal_networks` from PyPI (not from a
  cloned repo). If PyPI installation is not available yet, flag this explicitly.
- It does not depend on local files, hardcoded paths, or pre-existing data that a new
  user would not have.
- It demonstrates a realistic, non-trivial use case (not just `import temporal_networks;
  print("ok")`).
- It is short enough to copy-paste in under a minute and produces visible output (a
  plot, a printed table, a saved file).
- Random seeds are set explicitly where any stochastic operation is used, so the output
  is reproducible.
- It matches the installation instructions in the README (see README checklist below).

---

## README evaluation (pyOpenSci official checklist)

Evaluate the README against every item below. Flag any that are missing or incomplete.

- [ ] Package name is clearly stated at the top.
- [ ] Badges for: current published version (PyPI), documentation build status, test
      suite CI status. (Test coverage badge is optional but recommended.)
- [ ] A clear 2-4 sentence explanation of what the package does, written for a reader
      who has not seen it before.
- [ ] Context for how the package fits into the broader ecosystem: what existing tools
      does it relate to or differ from (e.g., `networkx`, `teneto`, `dynetx`)? This is
      not the same as listing features -- it should help a user decide whether this
      package is the right tool for their problem.
- [ ] If the package wraps another package, a link to that package and its
      documentation. (If it does not wrap another package, this item can be marked N/A.)
- [ ] A simple quickstart code example (same example from audit item 7 above) embedded
      directly in the README.
- [ ] A link to the full documentation / website.
- [ ] Links to any tutorials created for the package.
- [ ] Installation instructions that exactly match the documentation site. Mismatches
      between README install instructions and docs are a common EiC rejection reason.
      Check that the install method shown (PyPI vs. clone-and-install) is consistent
      across both.

Additionally check for the following, which are required for academic credibility:

- [ ] A citation block or reference to the associated paper (BibTeX or DOI), or a
      placeholder noting that a citation will be added on publication.
- [ ] A reproducibility statement: how to regenerate any results shown in the README or
      associated paper.
- [ ] A link to the CONTRIBUTING.md file.
- [ ] License name stated (not just a badge).

---

## Output format

**First line, exactly one of:**

```
Ready for public release
Needs fixes in: [list the affected areas from: tests / docs / style / portability / packaging / unfinished / examples / README]
```

**Then two sections:**

### Specific fixes needed

Bullet list. Each bullet must start with a category tag and include file name and line
number where applicable.

```
[tests] temporal_networks/cli.py: entry point `tn` requires `click` but `click` is not in install_requires
[docs] temporal_networks/graph.py line 45: missing Returns section in `compute_density`
[portability] run_benchmark.py line 12: hardcoded path "/data/temporal.csv"
[unfinished] utils.py line 89: # TODO: implement faster clustering
[README] README.md: quickstart example uses local file path, will fail after pip install
[README] README.md: no ecosystem context (no mention of networkx, teneto, or similar)
[packaging] pyproject.toml: `click` missing from install_requires, only present in dev extras
```

### README sections assessment

For each of the 13 README checklist items above, state one of:
- **Present and adequate**
- **Present but incomplete** (explain what is missing)
- **Missing**

---

## Initial task (README-only pass)

You have been given only the README. Evaluate it against:
1. All 13 README checklist items above.
2. Audit item 7 (quickstart example).

Flag any immediate issues (syntax errors in code blocks, broken links, install
instructions that would fail for a new user).

State your findings clearly, then list exactly which source files you need to see next
to complete the full audit, and why you need each one.