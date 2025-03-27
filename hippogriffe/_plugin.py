import pathlib
import re
import typing

from mkdocs.config import Config
from mkdocs.config.config_options import Choice, Type
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import BasePlugin, event_priority
from mkdocs.structure.files import File, Files, InclusionLevel
from mkdocstrings import AutoDocProcessor


_here = pathlib.Path(__file__).resolve().parent
_regex = re.compile(
    r"^\s*" + AutoDocProcessor.regex.pattern.removeprefix("^"),
    AutoDocProcessor.regex.flags,
)


class PluginConfig(Config):
    show_bases = Type(bool, default=True)
    """Whether to show class bases in-line with their name."""

    show_source_links = Choice(["all", "toplevel", "none"], default="toplevel")
    """Whether to include [source] links to the repo."""

    extra_public_objects = Type(list, default=[])
    """Any third-party objects which are allowed in the public API."""

    builtin_modules = Type(
        list, default=["builtins", "collections.abc", "typing", "typing_extensions"]
    )
    """A list of 'builtin' module names we should strip from pretty type annotations.
    For example `Sequence[Literal[1]]` instead of
    `collections.abc.Sequence[typing_extensions.Literal[1]]`
    """


def _get_options(config: MkDocsConfig) -> dict:
    mkdocstrings_config = config.plugins["mkdocstrings"].config
    options = (
        mkdocstrings_config.setdefault("handlers", {})
        .setdefault("python", {})
        .setdefault("options", {})
    )
    return options


class HippogriffePlugin(BasePlugin[PluginConfig]):
    css_filename: str = "assets/_hippogriffe.css"

    @event_priority(50)  # Before mkdocstrings
    def on_config(self, config: MkDocsConfig) -> MkDocsConfig | None:
        # Using the normalisation from https://peps.python.org/pep-0503/#normalized-names
        # We extend it to also replace en dashes and em dashes.
        # Not perfect but it should do.
        typing.GENERATING_DOCUMENTATION = re.sub(  # pyright: ignore[reportAttributeAccessIssue]
            r"[-_.–—]+", "-", config.site_name
        ).lower()

        options = _get_options(config)
        if options.get("force_inspection", False) is not True:
            raise ValueError(
                "hippogriffe requires "
                "`mkdocstrings.python.options.force_inspection: true`. (This is so it "
                "can do things like display type annotations correctly.)"
            )
        extensions = options.setdefault("extensions", [])
        extensions.append(
            {
                "hippogriffe._extension": {
                    "config": self.config,
                    "repo_url": config.repo_url,
                    "top_level_public_api": {""},
                }
            }
        )
        # Lower priority than user files
        config.extra_css.insert(0, self.css_filename)
        if self.config.show_bases and options.get("merge_init_into_class", False):
            raise ValueError(
                "Cannot set both `hippogriffe.show_bases: true` and "
                "`mkdocstrings.python.options.merge_init_into_class: true`, as they "
                "use the same space in the documentation."
            )

    @event_priority(-50)  # After other plugins have generated files
    def on_files(self, files: Files, /, *, config: MkDocsConfig) -> Files | None:
        for extension in _get_options(config)["extensions"]:
            if type(extension) is dict and set(extension.keys()) == {
                "hippogriffe._extension"
            }:
                top_level_public_api = extension["hippogriffe._extension"][
                    "top_level_public_api"
                ]
                assert top_level_public_api == {""}
                break
        else:
            assert False
        top_level_public_api.remove("")
        for file in files:
            if file.is_documentation_page():
                for match in _regex.finditer(file.content_string):
                    top_level_public_api.add(match["name"])
        files.append(
            File.generated(
                config=config,
                src_uri=self.css_filename,
                abs_src_path=str(_here / self.css_filename),
                inclusion=InclusionLevel.INCLUDED,
            )
        )
        return files
