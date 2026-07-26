#!/usr/bin/env python3
"""Create or verify a local Label Studio project for a prediction-review queue."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def local_api_token(database: Path, email: str) -> str:
    connection = sqlite3.connect(database)
    try:
        row = connection.execute(
            """
            SELECT token.key
            FROM authtoken_token AS token
            JOIN htx_user AS user ON user.id = token.user_id
            WHERE user.email = ?
            """,
            (email,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise RuntimeError(f"No local Label Studio API token for {email}")
    return str(row[0])


def request_json(
    method: str,
    url: str,
    *,
    token: str,
    **kwargs: Any,
) -> Any:
    response = requests.request(
        method,
        url,
        headers={"Authorization": f"Token {token}"},
        timeout=120,
        **kwargs,
    )
    response.raise_for_status()
    return response.json() if response.content else None


def find_project(base_url: str, token: str, title: str) -> dict[str, Any] | None:
    page = 1
    while True:
        payload = request_json(
            "GET",
            f"{base_url}/api/projects",
            token=token,
            params={"page": page, "page_size": 100},
        )
        results = payload.get("results", payload) if isinstance(payload, dict) else payload
        for project in results:
            if project.get("title") == title:
                return project
        if not isinstance(payload, dict) or not payload.get("next"):
            return None
        page += 1


def setup_project(
    *,
    base_url: str,
    token: str,
    title: str,
    label_config: str,
    tasks: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    project = find_project(base_url, token, title)
    created = project is None
    if project is None:
        project = request_json(
            "POST",
            f"{base_url}/api/projects",
            token=token,
            json={
                "title": title,
                "description": (
                    "Stratified full-PMC model predictions. Suggestions are "
                    "pre-annotations; submit only after checking every entity "
                    "and relation in the text."
                ),
                "label_config": label_config,
            },
        )
    project_id = int(project["id"])
    project = request_json(
        "PATCH",
        f"{base_url}/api/projects/{project_id}",
        token=token,
        json={"label_config": label_config},
    )
    detail = request_json(
        "GET",
        f"{base_url}/api/projects/{project_id}",
        token=token,
    )
    task_number = int(detail.get("task_number") or 0)
    if task_number == 0:
        request_json(
            "POST",
            f"{base_url}/api/projects/{project_id}/import",
            token=token,
            json=tasks,
        )
    elif task_number != len(tasks):
        raise RuntimeError(
            f"Existing project {project_id} has {task_number} tasks, "
            f"but the queue contains {len(tasks)}; refusing a duplicate import"
        )
    detail = request_json(
        "GET",
        f"{base_url}/api/projects/{project_id}",
        token=token,
    )
    return detail, created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path)
    parser.add_argument("label_config", type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--email", default="reviewer@localhost")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--title",
        default="PMC model prediction audit — batch 1",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = local_api_token(args.database, args.email)
    tasks = json.loads(args.queue.read_text())
    project, created = setup_project(
        base_url=args.url.rstrip("/"),
        token=token,
        title=args.title,
        label_config=args.label_config.read_text(),
        tasks=tasks,
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created": created,
        "project_id": int(project["id"]),
        "title": project["title"],
        "tasks": int(project.get("task_number") or len(tasks)),
        "completed_tasks": int(project.get("num_tasks_with_annotations") or 0),
        "queue": str(args.queue),
        "url": f"{args.url.rstrip('/')}/projects/{project['id']}/data",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
