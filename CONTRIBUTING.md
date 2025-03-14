# Contributing

Contributions (pull requests) are very welcome!

First fork the library on GitHub.

Then clone and install the library in development mode:

```bash
git clone https://github.com/your-username-here/hippogriffe.git
cd hippogriffe
pip install -e '.[dev]'
```

Then install the pre-commit hooks:

```bash
pre-commit install
```

These hooks use ruff to format and lint the code, and pyright to typecheck it.
