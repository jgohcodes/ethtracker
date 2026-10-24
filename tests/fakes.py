"""An in-memory stand-in for the dashboard API, installed in place of requests.get/put/post."""

import json
from urllib.parse import parse_qs, urlparse

import requests


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._body = {} if body is None else body
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


class FakeDashboard:
    def __init__(self, token="t", state=None, fail_posts_after=None, unreachable=False):
        self.token = token
        self.state = dict(state or {})
        self.snapshots = []
        self.calls = []
        self.fail_posts_after = fail_posts_after
        self.unreachable = unreachable

    def _enter(self, method, url, headers):
        self.calls.append((method, url))
        if self.unreachable:
            raise requests.ConnectionError("down")
        return (headers or {}).get("Authorization") == f"Bearer {self.token}"

    @staticmethod
    def _name(url):
        return parse_qs(urlparse(url).query).get("name", [None])[0]

    def post(self, url, json=None, headers=None, timeout=None):
        if self.fail_posts_after is not None and len(self.snapshots) >= self.fail_posts_after:
            self.calls.append(("POST", url))
            raise requests.ConnectionError("down")
        if not self._enter("POST", url, headers):
            return FakeResponse(401, {"error": "unauthorized"})
        self.snapshots.append(json)
        return FakeResponse(200, {"ok": True})

    def get(self, url, headers=None, timeout=None):
        if not self._enter("GET", url, headers):
            return FakeResponse(401, {"error": "unauthorized"})
        name = self._name(url)
        if name not in self.state:
            return FakeResponse(404, {"error": "not found"})
        return FakeResponse(200, {"content": self.state[name], "updated_at": "2026-10-24T00:20:00Z"})

    def put(self, url, json=None, headers=None, timeout=None):
        if not self._enter("PUT", url, headers):
            return FakeResponse(401, {"error": "unauthorized"})
        self.state[self._name(url)] = json["content"]
        return FakeResponse(200, {"ok": True})

    def install(self, monkeypatch):
        monkeypatch.setattr(requests, "post", self.post)
        monkeypatch.setattr(requests, "get", self.get)
        monkeypatch.setattr(requests, "put", self.put)
        return self
