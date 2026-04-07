"""Generate README.md from GitHub repositories using topics-based categorization."""

import json
import os
import sys
from pathlib import Path

import yaml
from github import Github
from openai import OpenAI

GITHUB_USER = "tsubasaogawa"
SCRIPT_DIR = Path(__file__).parent
CATEGORIES_FILE = SCRIPT_DIR / "categories.yml"
CACHE_FILE = SCRIPT_DIR / "descriptions_cache.json"
README_FILE = SCRIPT_DIR.parent.parent / "README.md"

GITHUB_MODELS_ENDPOINT = "https://models.github.ai/inference"
GITHUB_MODELS_MODEL = "openai/gpt-4.1-mini"


def fetch_repositories(token: str) -> list[dict]:
    """Fetch all public, non-fork, non-archived repositories."""
    g = Github(token)
    user = g.get_user(GITHUB_USER)
    repos = []
    for repo in user.get_repos():
        if repo.fork or repo.archived or repo.private or repo.name == GITHUB_USER:
            continue
        repos.append(
            {
                "name": repo.name,
                "html_url": repo.html_url,
                "description": repo.description or "",
                "topics": repo.get_topics(),
                "language": repo.language or "",
            }
        )
    return repos


def load_categories() -> tuple[list[dict], str]:
    """Load category definitions from YAML."""
    with open(CATEGORIES_FILE) as f:
        config = yaml.safe_load(f)
    return config["categories"], config.get("default_category", "Other")


def classify_repo(
    repo: dict,
    categories: list[dict],
    default_category: str,
) -> str:
    """Classify a repository into a category based on its topics."""
    repo_topics = set(repo.get("topics", []))
    for cat in categories:
        if repo_topics & set(cat["topics"]):
            return cat["name"]
    return default_category


def load_cache() -> dict[str, str]:
    """Load description cache."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict[str, str]) -> None:
    """Save description cache."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
        f.write("\n")


def generate_description(client: OpenAI, repo: dict) -> str:
    """Generate a one-line English description using GitHub Models."""
    context_parts = [f"Repository: {repo['name']}"]
    if repo.get("description"):
        context_parts.append(f"Existing description: {repo['description']}")
    if repo.get("topics"):
        context_parts.append(f"Topics: {', '.join(repo['topics'])}")
    if repo.get("language"):
        context_parts.append(f"Primary language: {repo['language']}")

    context = "\n".join(context_parts)

    response = client.chat.completions.create(
        model=GITHUB_MODELS_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate concise, one-line descriptions for GitHub repositories. "
                    "Write in English. Be factual and specific. No emoji. No markdown. "
                    "Maximum 100 characters. Start with a verb or noun, not 'A' or 'An'."
                ),
            },
            {
                "role": "user",
                "content": f"Write a one-line description for this repository:\n\n{context}",
            },
        ],
        max_tokens=60,
        temperature=0.3,
    )

    return response.choices[0].message.content.strip().rstrip(".")


def generate_descriptions(
    repos: list[dict],
    cache: dict[str, str],
) -> dict[str, str]:
    """Generate descriptions for repositories not in cache."""
    new_repos = [r for r in repos if r["name"] not in cache]
    if not new_repos:
        return cache

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print(
            "WARNING: GITHUB_TOKEN not set. Skipping AI description generation.",
            file=sys.stderr,
        )
        for r in new_repos:
            cache[r["name"]] = r.get("description") or r["name"]
        return cache

    client = OpenAI(base_url=GITHUB_MODELS_ENDPOINT, api_key=token)

    for repo in new_repos:
        try:
            desc = generate_description(client, repo)
            cache[repo["name"]] = desc
            print(f"  Generated: {repo['name']} -> {desc}")
        except Exception as e:
            print(
                f"  Error generating description for {repo['name']}: {e}",
                file=sys.stderr,
            )
            cache[repo["name"]] = repo.get("description") or repo["name"]

    return cache


def build_readme(
    repos: list[dict],
    categories: list[dict],
    default_category: str,
    descriptions: dict[str, str],
) -> str:
    """Build README.md content."""
    categorized: dict[str, list[dict]] = {}
    for repo in repos:
        cat = classify_repo(repo, categories, default_category)
        categorized.setdefault(cat, []).append(repo)

    for cat in categorized:
        categorized[cat].sort(key=lambda r: r["name"].lower())

    def append_category(lines: list[str], cat_name: str) -> None:
        lines.append(f"- {cat_name}")
        for repo in categorized[cat_name]:
            desc = descriptions.get(repo["name"], repo.get("description") or "")
            url = repo["html_url"]
            lines.append(f"  - [{repo['name']}]({url}) - {desc}")
        lines.append("")

    lines = ["## Repositories", ""]

    category_order = [c["name"] for c in categories] + [default_category]
    seen = set()
    visible_categories: list[str] = []
    for cat_name in category_order:
        if cat_name in seen or cat_name not in categorized:
            continue
        seen.add(cat_name)
        visible_categories.append(cat_name)

    for cat_name in visible_categories[:3]:
        append_category(lines, cat_name)

    collapsed_categories = visible_categories[3:]
    if collapsed_categories:
        lines.append("<details>")
        lines.append("<summary>More</summary>")
        lines.append("")
        for cat_name in collapsed_categories:
            append_category(lines, cat_name)
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    token = os.environ.get("GITHUB_TOKEN", "")

    print("Fetching repositories...")
    repos = fetch_repositories(token)
    print(f"Found {len(repos)} repositories")

    print("Loading categories...")
    categories, default_category = load_categories()

    print("Loading description cache...")
    cache = load_cache()

    print("Generating descriptions...")
    descriptions = generate_descriptions(repos, cache)
    save_cache(descriptions)

    print("Building README...")
    readme_content = build_readme(repos, categories, default_category, descriptions)

    README_FILE.write_text(readme_content)
    print(f"README.md written to {README_FILE}")


if __name__ == "__main__":
    main()
