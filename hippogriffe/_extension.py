import ast
import builtins
import contextlib
import functools as ft
import importlib
import inspect
import pathlib
import re
import subprocess
import sys
from collections.abc import Callable
from typing import Any, Iterable, Iterator, Literal, get_origin

import griffe
import wadler_lindig as wl

from ._plugin import PluginConfig


_here = pathlib.Path(__file__).resolve().parent


def get_templates_path():
    return _here / "templates"


class _NotInPublicApiException(Exception):
    pass


class _PublicApi:
    def __init__(
        self,
        pkg: griffe.Module,
        top_level_public_api: set[str],
        builtin_modules: list[str],
        extra_public_objects: list[str],
    ):
        self._objects: set[griffe.Object] = set()
        self._toplevel_objects: set[griffe.Object] = set()
        self._data: dict[str, list[str]] = {}
        self._builtin_modules = builtin_modules
        for object_path in extra_public_objects:
            object_pieces = object_path.split(".")
            for i in reversed(range(1, len(object_pieces))):
                module_name = "".join(object_pieces[:i])
                object_name = object_pieces[i:]
                try:
                    object = importlib.import_module(module_name)
                except Exception:
                    continue
                for object_piece in object_name:
                    object = getattr(object, object_piece)
                private_path = f"{object.__module__}.{object.__qualname__}"
                try:
                    paths = self._data[private_path]
                except KeyError:
                    paths = self._data[private_path] = []
                paths.append(object_path)
        # Don't infinite loop on cycles. We only store Objects, and not Aliases, as in
        # cycles then the aliases with be distinct: `X.Y.X.Y` is not `X.Y`, though the
        # underlying object is the same.

        agenda: list[tuple[griffe.Object, str, bool]] = [(pkg, pkg.path, False)]
        seen: set[griffe.Object] = {pkg}
        while len(agenda) > 0:
            item, public_path, force_public = agenda.pop()
            toplevel_public = public_path in top_level_public_api
            if force_public or toplevel_public:
                # If we're in the public API, then we consider all of our children to be
                # in it as well... (this saves us from having to parse out `filters` and
                # `members` from our documentation)
                try:
                    paths = self._data[item.path]
                except KeyError:
                    paths = self._data[item.path] = []
                paths.append(public_path)
                self._objects.add(item)
                if toplevel_public:
                    self._toplevel_objects.add(item)
                sub_force_public = True
            else:
                # ...if we're not in the public API then check our members -- some of
                # them might be in the public API.
                sub_force_public = False
            for member in item.all_members.values():
                # Skip private elements
                if member.name.startswith("_") and not (
                    member.name.startswith("__") and member.name.endswith("__")
                ):
                    continue
                if isinstance(member, griffe.Alias):
                    try:
                        final_member = member.final_target
                    except (griffe.AliasResolutionError, griffe.CyclicAliasError):
                        continue
                    if member.name != final_member.name:
                        # Renaming during import counts as private.
                        # (In particular this happens for backward compatibility, e.g.
                        # `equinox.nn.inference_mode` and `equinox.tree_inference`.)
                        continue
                else:
                    final_member = member
                if final_member in seen:
                    continue
                agenda.append((final_member, member.path, sub_force_public))
                seen.add(final_member)

    def toplevel(self) -> Iterable[griffe.Object]:
        return self._toplevel_objects

    def __iter__(self) -> Iterator[griffe.Object]:
        yield from self._objects

    def __getitem__(self, key: str) -> tuple[str, bool]:
        try:
            paths = self._data[key]
        except KeyError as e:
            for m in self._builtin_modules:
                if key.startswith(m + "."):
                    return key.removeprefix(m + "."), False
            for m in sys.stdlib_module_names:
                if key.startswith(m + "."):
                    return key, False
            # Note that this message must not have any newlines in it, to display
            # correctly.
            raise _NotInPublicApiException(
                f"Tried and failed to find `{key}` in the public API. Commons reasons "
                "for this error are (1) if it is from outside this package, then this "
                "object is not listed (under whatever public path it should be "
                "displayed as) in `hippogriffe.extra_public_objects`; (2) if it is "
                "from inside this package, then it may have been written "
                "`::: somelib.Foo:` with a trailing colon, when just `:::somelib.Foo` "
                "is correct."
            ) from e
        if len(paths) == 1:
            return paths[0], True
        else:
            raise ValueError(f"{key} has multiple paths in the public API: {paths}")


_kind_map = {
    inspect.Parameter.POSITIONAL_ONLY: griffe.ParameterKind.positional_only,
    inspect.Parameter.POSITIONAL_OR_KEYWORD: griffe.ParameterKind.positional_or_keyword,
    inspect.Parameter.VAR_POSITIONAL: griffe.ParameterKind.var_positional,
    inspect.Parameter.KEYWORD_ONLY: griffe.ParameterKind.keyword_only,
    inspect.Parameter.VAR_KEYWORD: griffe.ParameterKind.var_keyword,
}


def _pretty_annotation(
    annotation,
    context: dict,
    use_public_name: Callable[[None | dict, Any], None | wl.AbstractDoc],
):
    if annotation is inspect.Signature.empty:
        return None
    else:
        return wl.pformat(
            annotation, custom=ft.partial(use_public_name, context), width=9999
        )


def _pretty_param(
    param: inspect.Parameter,
    context: dict,
    use_public_name: Callable[[None | dict, Any], None | wl.AbstractDoc],
) -> griffe.Parameter:
    annotation = _pretty_annotation(param.annotation, context, use_public_name)
    if param.default is inspect.Signature.empty:
        default = None
    else:
        default = wl.pformat(
            param.default, custom=ft.partial(use_public_name, None), width=9999
        )
    return griffe.Parameter(
        name=param.name,
        annotation=annotation,
        kind=_kind_map[param.kind],
        default=default,
    )


def _pretty_fn(
    obj: griffe.Function,
    use_public_name: Callable[[None | dict, Any], None | wl.AbstractDoc],
) -> None:
    try:
        module = sys.modules[obj.module.path]
    except KeyError:
        context = {}
    else:
        context = module.__dict__
    signature: inspect.Signature = obj.extra["hippogriffe"]["signature"]
    obj.parameters = griffe.Parameters(
        *[
            _pretty_param(param, context, use_public_name)
            for param in signature.parameters.values()
        ]
    )
    signature.return_annotation
    obj.returns = _pretty_annotation(
        signature.return_annotation, context, use_public_name
    )


_builtin_re = re.compile(r"builtins\.(\w+)")


# Modified version of `griffe.Class.resolved_bases`, so as not to drop builtins.
def _resolved_bases(cls: griffe.Class) -> list[str | griffe.Object]:
    resolved_bases = []
    for base in cls.bases:
        base_path = base if isinstance(base, str) else base.canonical_path
        match = _builtin_re.match(base_path)
        with contextlib.suppress(
            griffe.AliasResolutionError, griffe.CyclicAliasError, KeyError
        ):
            if match is None:
                resolved_base = cls.modules_collection[base_path]
                if resolved_base.is_alias:
                    resolved_base = resolved_base.final_target
            else:
                resolved_base = match.group(1)
                if not hasattr(builtins, resolved_base):
                    raise KeyError
            resolved_bases.append(resolved_base)
    return resolved_bases


def _collect_bases(cls: griffe.Class, public_api: _PublicApi) -> dict[str, bool]:
    bases: dict[str, bool] = {}
    for base in _resolved_bases(cls):
        if isinstance(base, str):
            # builtins case above
            bases[base] = False
        elif isinstance(base, griffe.Class):
            try:
                base, autoref = public_api[base.path]
            except _NotInPublicApiException:
                bases.update(_collect_bases(base, public_api))
            else:
                bases[base] = autoref
    return bases


@ft.cache
def _get_repo_url(repo_url: None | str) -> tuple[pathlib.Path, str]:
    if repo_url is None:
        raise ValueError(
            "`hippogriffe.show_source_links` requires specifying a top-level "
            "`repo_url`."
        )
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=False
        )
        git_toplevel = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, check=False
        )
        if git_head.returncode != 0 or git_toplevel.returncode != 0:
            raise FileNotFoundError
    except FileNotFoundError as e:
        raise ValueError(
            "`hippogriffe.show_source_links` requires running from a git repository, "
            "but could not find git commit hash or root."
        ) from e
    else:
        toplevel = pathlib.Path(git_toplevel.stdout.decode().strip())
        commit_hash = git_head.stdout.decode().strip()
    if "://" in repo_url:
        protocol, repo_url = repo_url.split("://", 1)
        protocol = f"{protocol}://"
    else:
        protocol = ""
    supported_site = False
    if repo_url.startswith("github.com"):
        supported_site = True
        fragment = "L{start}-L{end}"
    elif repo_url.startswith("gitlab.com"):
        supported_site = True
        fragment = "L{start}-{end}"
    if supported_site:
        # Expect url in the form `https://github.com/org/repo`, strip any trailing paths
        repo_url = "/".join(repo_url.split("/")[:3])
        repo_url = (
            f"{protocol}{repo_url}/blob/{commit_hash}/{{path}}#{fragment}"
        )
    else:
        # We need to format the `repo_url` to what the repo expects, so we have to
        # hardcode this in.
        raise ValueError(
            "`hippogriffe.show_source_links` currently only supports "
            "`repo_url: https://github.com/...` and `repo_url: https://gitlab.com/...`."
        )
    return toplevel, repo_url


class HippogriffeExtension(griffe.Extension):
    def __init__(
        self, config: PluginConfig, repo_url: None | str, top_level_public_api: set[str]
    ):
        self.config = config
        self.repo_url = repo_url
        self.top_level_public_api = top_level_public_api

    def on_function_instance(
        self,
        *,
        node: ast.AST | griffe.ObjectNode,
        func: griffe.Function,
        agent: griffe.Visitor | griffe.Inspector,
        **kwargs: Any,
    ) -> None:
        del agent, kwargs
        assert not isinstance(node, ast.AST)
        signature = inspect.signature(node.obj)
        try:
            name = node.obj.__name__
        except AttributeError:
            pass
        else:
            if name == "__init__":
                signature = signature.replace(return_annotation=inspect.Signature.empty)
        func.extra["hippogriffe"]["signature"] = signature

    def on_attribute_instance(
        self,
        *,
        node: ast.AST | griffe.ObjectNode,
        attr: griffe.Attribute,
        agent: griffe.Visitor | griffe.Inspector,
        **kwargs: Any,
    ) -> None:
        del node, agent, kwargs
        # Knowing the value is IMO usually not useful. That is what documentation
        # directly is for.
        attr.value = None
        # This is used to indicate that it is a module attribute, but IMO that's not
        # super clear in the docs.
        attr.labels.discard("module")

    def on_package_loaded(
        self, *, pkg: griffe.Module, loader: griffe.GriffeLoader, **kwargs: Any
    ) -> None:
        assert self.top_level_public_api != {""}
        del loader, kwargs

        public_api = _PublicApi(
            pkg,
            top_level_public_api=self.top_level_public_api,
            builtin_modules=self.config.builtin_modules,
            extra_public_objects=self.config.extra_public_objects,
        )

        def use_public_name(context: None | dict, obj: Any) -> None | wl.AbstractDoc:
            # If we hit a Literal then don't try to convert any of its string-typed
            # elements into types...
            if get_origin(obj) is Literal:
                return wl.pdoc(obj, width=9999)
            # ...but otherwise do attempt to resolve strings into types.
            if context is not None and isinstance(obj, str):
                with contextlib.suppress(BaseException):
                    obj = eval(obj, context)
            # Then if it's in the public API, convert it over.
            if isinstance(obj, type) and obj is not type(None):
                new_path, _ = public_api[f"{obj.__module__}.{obj.__qualname__}"]
                return wl.TextDoc(new_path)

        for obj in public_api:
            if obj.is_function:
                assert type(obj) is griffe.Function
                try:
                    _pretty_fn(obj, use_public_name)
                except _NotInPublicApiException as e:
                    # Defer error until later -- right now our `public_api` is an
                    # overestimation of the 'true' public API as for a public class
                    # `Foo` then we actually include all of its attributes in the public
                    # API here, even if those aren't documented.
                    # It's fairly common to have nonpublic annotations in nonpublic
                    # methods, and we shouldn't die on those now -- if the method is
                    # never documented then we don't need to worry. Putting this here is
                    # totally sneaky, it's letting jinja think this is a template and
                    # having it raise a TemplateNotFound error when it tries to format
                    # this object.
                    obj.extra["mkdocstrings"]["template"] = (
                        f"{e} This arose whilst pretty-printing `{obj.path}`. You may "
                        "ignore the rest of this error message, which comes from jinja."
                        "                                                        "
                    )
                else:
                    obj.extra["mkdocstrings"]["template"] = "hippogriffe/fn.html.jinja"
            elif obj.is_class:
                assert type(obj) is griffe.Class
                obj.extra["mkdocstrings"]["template"] = "hippogriffe/class.html.jinja"
                if self.config.show_bases:
                    public_bases = list(_collect_bases(obj, public_api).items())
                    obj.extra["hippogriffe"]["public_bases"] = public_bases

        if self.config.show_source_links == "none":
            api_iterable = ()
        elif self.config.show_source_links == "toplevel":
            api_iterable = public_api.toplevel()
        elif self.config.show_source_links == "all":
            api_iterable = public_api
        else:
            assert False
        for obj in api_iterable:
            if (
                obj.lineno is not None
                and obj.endlineno is not None
                and isinstance(obj.filepath, pathlib.Path)
            ):
                toplevel, repo_url = _get_repo_url(self.repo_url)
                path = obj.filepath.relative_to(toplevel)
                url = repo_url.format(path=path, start=obj.lineno, end=obj.endlineno)
                obj.extra["hippogriffe"]["url"] = url
