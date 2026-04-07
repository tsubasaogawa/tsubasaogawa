import importlib.util
import io
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "update_readme.py"


def load_update_readme_module():
    github_module = types.ModuleType("github")
    github_module.Github = MagicMock(name="Github")

    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = MagicMock(name="OpenAI")

    yaml_module = types.ModuleType("yaml")
    yaml_module.safe_load = MagicMock(name="safe_load")

    spec = importlib.util.spec_from_file_location("update_readme_under_test", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "github": github_module,
            "openai": openai_module,
            "yaml": yaml_module,
        },
    ):
        spec.loader.exec_module(module)

    return module


update_readme = load_update_readme_module()


class FakeRepo:
    def __init__(
        self,
        *,
        name,
        html_url="https://example.com/repo",
        description="",
        topics=None,
        language="",
        fork=False,
        archived=False,
        private=False,
    ):
        self.name = name
        self.html_url = html_url
        self.description = description
        self._topics = topics or []
        self.language = language
        self.fork = fork
        self.archived = archived
        self.private = private

    def get_topics(self):
        return list(self._topics)


class UpdateReadmeTests(unittest.TestCase):
    def test_fetch_repositories_filters_and_maps_public_non_fork_repos(self):
        kept_repo = FakeRepo(
            name="kept",
            html_url="https://example.com/kept",
            description="Repository to keep",
            topics=["python", "cli"],
            language="Python",
        )
        skipped_repos = [
            FakeRepo(name=update_readme.GITHUB_USER),
            FakeRepo(name="forked", fork=True),
            FakeRepo(name="archived", archived=True),
            FakeRepo(name="private", private=True),
        ]
        fake_user = SimpleNamespace(get_repos=MagicMock(return_value=[kept_repo, *skipped_repos]))
        fake_github = MagicMock()
        fake_github.get_user.return_value = fake_user

        with patch.object(update_readme, "Github", return_value=fake_github) as github_class:
            repos = update_readme.fetch_repositories("token-123")

        github_class.assert_called_once_with("token-123")
        fake_github.get_user.assert_called_once_with(update_readme.GITHUB_USER)
        self.assertEqual(
            repos,
            [
                {
                    "name": "kept",
                    "html_url": "https://example.com/kept",
                    "description": "Repository to keep",
                    "topics": ["python", "cli"],
                    "language": "Python",
                }
            ],
        )

    def test_load_categories_uses_other_when_default_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            categories_file = Path(tmpdir) / "categories.yml"
            categories_file.write_text("categories: []\n", encoding="utf-8")

            with (
                patch.object(update_readme, "CATEGORIES_FILE", categories_file),
                patch.object(
                    update_readme.yaml,
                    "safe_load",
                    return_value={"categories": [{"name": "Python", "topics": ["python"]}]},
                ) as safe_load,
            ):
                categories, default_category = update_readme.load_categories()

        safe_load.assert_called_once()
        self.assertEqual(categories, [{"name": "Python", "topics": ["python"]}])
        self.assertEqual(default_category, "Other")

    def test_cache_round_trip_returns_empty_dict_when_file_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "descriptions_cache.json"

            with patch.object(update_readme, "CACHE_FILE", cache_file):
                self.assertEqual(update_readme.load_cache(), {})
                update_readme.save_cache({"repo": "Generated description"})
                self.assertEqual(
                    update_readme.load_cache(),
                    {"repo": "Generated description"},
                )

    def test_generate_description_sends_expected_prompt_and_trims_period(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Useful CLI tool."))]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=MagicMock(return_value=response))
            )
        )

        description = update_readme.generate_description(
            client,
            {
                "name": "sample-repo",
                "description": "Existing text",
                "topics": ["python", "cli"],
                "language": "Python",
            },
        )

        self.assertEqual(description, "Useful CLI tool")
        client.chat.completions.create.assert_called_once()
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], update_readme.GITHUB_MODELS_MODEL)
        self.assertEqual(kwargs["max_tokens"], 60)
        self.assertEqual(kwargs["temperature"], 0.3)
        self.assertIn("Repository: sample-repo", kwargs["messages"][1]["content"])
        self.assertIn("Existing description: Existing text", kwargs["messages"][1]["content"])
        self.assertIn("Topics: python, cli", kwargs["messages"][1]["content"])
        self.assertIn("Primary language: Python", kwargs["messages"][1]["content"])

    def test_generate_descriptions_returns_cache_unchanged_when_all_repos_are_cached(self):
        cache = {"repo-a": "cached description"}

        with (
            patch.dict(os.environ, {"GITHUB_TOKEN": "token-123"}, clear=True),
            patch.object(update_readme, "OpenAI") as openai_class,
        ):
            result = update_readme.generate_descriptions([{"name": "repo-a"}], cache)

        openai_class.assert_not_called()
        self.assertIs(result, cache)
        self.assertEqual(result, {"repo-a": "cached description"})

    def test_generate_descriptions_without_token_falls_back_and_warns(self):
        repos = [
            {"name": "repo-a", "description": "Existing description"},
            {"name": "repo-b", "description": ""},
        ]
        stderr = io.StringIO()

        with (
            patch.dict(os.environ, {}, clear=True),
            redirect_stderr(stderr),
        ):
            descriptions = update_readme.generate_descriptions(repos, {})

        self.assertEqual(
            descriptions,
            {
                "repo-a": "Existing description",
                "repo-b": "repo-b",
            },
        )
        self.assertIn("WARNING: GITHUB_TOKEN not set", stderr.getvalue())

    def test_generate_descriptions_uses_openai_and_falls_back_on_error(self):
        repos = [
            {"name": "repo-a", "description": "Existing description"},
            {"name": "repo-b", "description": ""},
        ]
        client = object()
        stderr = io.StringIO()

        with (
            patch.dict(os.environ, {"GITHUB_TOKEN": "token-123"}, clear=True),
            patch.object(update_readme, "OpenAI", return_value=client) as openai_class,
            patch.object(
                update_readme,
                "generate_description",
                side_effect=["Fresh description", RuntimeError("boom")],
            ) as generate_description,
            redirect_stderr(stderr),
            redirect_stdout(io.StringIO()),
        ):
            descriptions = update_readme.generate_descriptions(repos, {})

        openai_class.assert_called_once_with(
            base_url=update_readme.GITHUB_MODELS_ENDPOINT,
            api_key="token-123",
        )
        self.assertEqual(generate_description.call_count, 2)
        self.assertEqual(
            descriptions,
            {
                "repo-a": "Fresh description",
                "repo-b": "repo-b",
            },
        )
        self.assertIn("Error generating description for repo-b: boom", stderr.getvalue())

    def test_build_readme_groups_repositories_and_sorts_names_case_insensitively(self):
        repos = [
            {
                "name": "zeta-tool",
                "html_url": "https://example.com/zeta",
                "description": "Original zeta description",
                "topics": ["python"],
            },
            {
                "name": "Alpha-tool",
                "html_url": "https://example.com/alpha",
                "description": "Original alpha description",
                "topics": ["python"],
            },
            {
                "name": "misc-tool",
                "html_url": "https://example.com/misc",
                "description": "Original misc description",
                "topics": [],
            },
        ]
        categories = [{"name": "Python", "topics": ["python"]}]
        descriptions = {
            "Alpha-tool": "Generated alpha description",
            "misc-tool": "Generated misc description",
        }

        readme = update_readme.build_readme(repos, categories, "Other", descriptions)

        self.assertEqual(
            readme,
            "\n".join(
                [
                    "## Repositories",
                    "",
                    "- Python",
                    "  - [Alpha-tool](https://example.com/alpha) - Generated alpha description",
                    "  - [zeta-tool](https://example.com/zeta) - Original zeta description",
                    "",
                    "- Other",
                    "  - [misc-tool](https://example.com/misc) - Generated misc description",
                    "",
                ]
            ),
        )

    def test_build_readme_wraps_categories_after_third_in_single_details_block(self):
        repos = [
            {
                "name": "zeta-alpha",
                "html_url": "https://example.com/zeta-alpha",
                "description": "Original zeta alpha description",
                "topics": ["alpha"],
            },
            {
                "name": "Alpha-alpha",
                "html_url": "https://example.com/alpha-alpha",
                "description": "Original alpha alpha description",
                "topics": ["alpha"],
            },
            {
                "name": "beta-repo",
                "html_url": "https://example.com/beta",
                "description": "Original beta description",
                "topics": ["beta"],
            },
            {
                "name": "gamma-repo",
                "html_url": "https://example.com/gamma",
                "description": "Original gamma description",
                "topics": ["gamma"],
            },
            {
                "name": "delta-repo",
                "html_url": "https://example.com/delta",
                "description": "Original delta description",
                "topics": ["delta"],
            },
            {
                "name": "epsilon-repo",
                "html_url": "https://example.com/epsilon",
                "description": "Original epsilon description",
                "topics": ["epsilon"],
            },
            {
                "name": "misc-repo",
                "html_url": "https://example.com/misc",
                "description": "Original misc description",
                "topics": [],
            },
        ]
        categories = [
            {"name": "Alpha", "topics": ["alpha"]},
            {"name": "Beta", "topics": ["beta"]},
            {"name": "Gamma", "topics": ["gamma"]},
            {"name": "Delta", "topics": ["delta"]},
            {"name": "Epsilon", "topics": ["epsilon"]},
        ]
        descriptions = {
            "Alpha-alpha": "Generated alpha description",
            "misc-repo": "Generated misc description",
        }

        readme = update_readme.build_readme(repos, categories, "Other", descriptions)

        self.assertEqual(
            readme,
            "\n".join(
                [
                    "## Repositories",
                    "",
                    "- Alpha",
                    "  - [Alpha-alpha](https://example.com/alpha-alpha) - Generated alpha description",
                    "  - [zeta-alpha](https://example.com/zeta-alpha) - Original zeta alpha description",
                    "",
                    "- Beta",
                    "  - [beta-repo](https://example.com/beta) - Original beta description",
                    "",
                    "- Gamma",
                    "  - [gamma-repo](https://example.com/gamma) - Original gamma description",
                    "",
                    "<details>",
                    "<summary>More</summary>",
                    "",
                    "- Delta",
                    "  - [delta-repo](https://example.com/delta) - Original delta description",
                    "",
                    "- Epsilon",
                    "  - [epsilon-repo](https://example.com/epsilon) - Original epsilon description",
                    "",
                    "- Other",
                    "  - [misc-repo](https://example.com/misc) - Generated misc description",
                    "",
                    "</details>",
                    "",
                ]
            ),
        )

    def test_main_orchestrates_dependencies_and_writes_readme(self):
        repos = [{"name": "repo-a"}]
        categories = [{"name": "Python", "topics": ["python"]}]
        cache = {"repo-a": "cached"}
        descriptions = {"repo-a": "generated"}

        with tempfile.TemporaryDirectory() as tmpdir:
            readme_file = Path(tmpdir) / "README.md"

            with (
                patch.dict(os.environ, {"GITHUB_TOKEN": "token-123"}, clear=True),
                patch.object(update_readme, "README_FILE", readme_file),
                patch.object(update_readme, "fetch_repositories", return_value=repos) as fetch_repositories,
                patch.object(update_readme, "load_categories", return_value=(categories, "Other")) as load_categories,
                patch.object(update_readme, "load_cache", return_value=cache) as load_cache,
                patch.object(update_readme, "generate_descriptions", return_value=descriptions) as generate_descriptions,
                patch.object(update_readme, "save_cache") as save_cache,
                patch.object(update_readme, "build_readme", return_value="README content") as build_readme,
                redirect_stdout(io.StringIO()),
            ):
                update_readme.main()
                written_content = readme_file.read_text(encoding="utf-8")

        fetch_repositories.assert_called_once_with("token-123")
        load_categories.assert_called_once_with()
        load_cache.assert_called_once_with()
        generate_descriptions.assert_called_once_with(repos, cache)
        save_cache.assert_called_once_with(descriptions)
        build_readme.assert_called_once_with(repos, categories, "Other", descriptions)
        self.assertEqual(written_content, "README content")


if __name__ == "__main__":
    unittest.main()
