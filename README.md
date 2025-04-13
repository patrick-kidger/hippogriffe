<h1 align="center">Hippogriffe</h1>

This is a set of tweaks on top of the MkDocs + `mkdocstrings[python]` + `griffe` documentation stack. In particular, we:

- Add `[source]` links to GitHub to each top-level class or function.
- Pretty-format type annotations:
    - Fixes unions/generics/etc. to display as e.g. `int | str` rather than just `Union`, or `tuple[int, str]` rather than just `tuple`.
    - Respects your public API: if a type is declared in your documentation as `::: yourlib.Foo` then its usage in type annotations will match: `some_fn(foo: yourlib.Foo)`.
- Show base classes inline after the class.
- Drops the `-> None` return annotation from `__init__` methods.
- Attributes display as `[attr] somelib.someattr` instead of `[attr] somelib.someattr = some_value [module]`. (I don't find usually-long default values to be useful documentation, nor the 'module' tag to be informative.)

Before                 | After
:---------------------:|:----------------------:
![old](./imgs/old.png) | ![new](./imgs/new.png)

## Installation

```bash
pip install hippogriffe
```

Requires MkDocs 1.6.1+ and `mkdocstrings[python]` 0.28.3+

## Usage

In `mkdocs.yml`:
```yml
...

plugins:
    - hippogriffe
    - mkdocstrings:
        ...
```

## Configuration

Hippogriffe supports the following configuration options:

```yml
plugins:
    - hippogriffe:
        show_bases: true/false
        show_source_links: all/toplevel/none
        extra_public_objects:
            - foo.SomeClass
            - bar.subpackage.some_function
```

**show_bases:**

If `false` then base classes will not be displayed alongside a class. Defaults to `true`.

**show_source_links:**

Sets which objects will have links to their location in the repository (as configured via the usual MkDocs `repo_url`). If `all` then all objects will have links. If `toplevel` then just `::: somelib.value` will have links, but their members will not. If `none` then no links will be added. Defaults to `toplevel`.

**extra_public_objects:**

Pretty-formatting of type annotations is done strictly: every annotation must be part of the known public API, else an error will be raised. The public API is defined as the combination of:

- Everything you document using `::: yourlib.Foo`, and all of their members.
- Anything from the standard library.
- All objects belonging to any of `extra_public_objects`.

For example,
```yml
plugins:
    - hippogriffe:
        extra_public_objects:
            - jax.Array
            - torch.Tensor
```

List each object under whatever public path `somelib.Foo` that you would like it to be displayed under (and from which it must be accessible), not whichever private path `somelib._internal.foo.Foo` it is defined at.
