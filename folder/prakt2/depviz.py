#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variant 22 — Dependency Graph Visualizer (npm), stdlib-only

Stages implemented:
  1) CSV config: reads params, validates, prints key=value (stage 1).
  2) Direct deps for npm packages WITHOUT package managers or extra libs.
     Handles monorepos (tries packages/<name>/package.json) and falls back to npm tarball.
  3) Full graph via BFS with recursion, depth limit, cycle handling; test-mode from file A:B,C.
  4) Reverse dependencies mode (who depends on X) using same traversal.
  5) Visualization: Mermaid text; optional ASCII tree.

Only Python standard library is used.
"""
from __future__ import annotations
import csv
import pathlib
import json
import os
import re
import sys
import io
import tarfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from urllib.request import urlopen, Request

# ---------------- Config ----------------

@dataclass
class AppConfig:
    package_name: str
    repo_or_path: str
    test_mode: bool
    ascii_tree: bool
    max_depth: int

    @staticmethod
    def from_csv(path: str) -> 'AppConfig':
        if not os.path.exists(path):
            raise FileNotFoundError(f"config not found: {path}")
        params: Dict[str, str] = {}
        with open(path, 'r', encoding='utf-8') as f:
            for row in csv.reader(f):
                if not row or row[0].strip().startswith('#'):
                    continue
                if len(row) < 2:
                    raise ValueError(f"bad csv row (need key,value): {row}")
                params[row[0].strip()] = row[1].strip()
        # required
        req = ['package_name','repo_or_path','test_mode','ascii_tree','max_depth']
        missing = [k for k in req if k not in params]
        if missing:
            raise KeyError('missing keys in config: ' + ', '.join(missing))
        # parse
        def to_bool(s: str, key: str) -> bool:
            t = s.lower()
            if t in ('1','true','yes','y','on'): return True
            if t in ('0','false','no','n','off'): return False
            raise ValueError(f"{key}: expected boolean, got '{s}'")
        try:
            md = int(params['max_depth']); assert md >= 0
        except Exception:
            raise ValueError('max_depth: expected non-negative integer')
        return AppConfig(
            params['package_name'],
            params['repo_or_path'],
            to_bool(params['test_mode'],'test_mode'),
            to_bool(params['ascii_tree'],'ascii_tree'),
            md
        )

    def dump_kv(self) -> str:
        return "\n".join([
            f"package_name={self.package_name}",
            f"repo_or_path={self.repo_or_path}",
            f"test_mode={self.test_mode}",
            f"ascii_tree={self.ascii_tree}",
            f"max_depth={self.max_depth}",
        ])

# ---------------- HTTP helpers (stdlib only) ----------------

_UA = {"User-Agent": "depviz/variant22"}

def http_get_json(url: str) -> dict:
    req = Request(url, headers=_UA)
    with urlopen(req) as r:
        return json.loads(r.read().decode('utf-8'))

def http_get_text(url: str) -> str:
    req = Request(url, headers=_UA)
    with urlopen(req) as r:
        return r.read().decode('utf-8', errors='replace')

# ---------------- npm / GitHub helpers ----------------

def resolve_github_raw(repo_url: str, rel_path: str) -> Optional[str]:
    """Return raw URL for a given file path in repo (try common branches)."""
    m = re.match(r'https?://github.com/([^/]+)/([^/.]+)(?:\\.git)?/?$', repo_url.strip())
    if not m:
        return None
    user, repo = m.group(1), m.group(2)
    for branch in ('main','master','HEAD'):
        raw = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{rel_path}"
        try:
            txt = http_get_text(raw)
            if txt.strip():
                return raw
        except Exception:
            pass
    return None

def try_github_package_json(repo_url: str, pkg_name: str) -> Optional[dict]:
    """Try package.json from repo root, then monorepo path packages/<pkg_name>/package.json."""
    for candidate in ("package.json", f"packages/{pkg_name}/package.json"):
        raw = resolve_github_raw(repo_url, candidate)
        if not raw:
            continue
        try:
            txt = http_get_text(raw)
            if txt.strip().startswith('{'):
                return json.loads(txt)
        except Exception:
            continue
    return None

def npm_registry_repo(name: str) -> Optional[str]:
    """Get repository URL from npm registry metadata."""
    try:
        meta = http_get_json(f"https://registry.npmjs.org/{name}")
    except Exception:
        return None
    latest = meta.get('dist-tags',{}).get('latest')
    ver = meta.get('versions',{}).get(latest,{}) if latest else {}
    repo = ver.get('repository') or meta.get('repository')
    if isinstance(repo, dict):
        url = repo.get('url') or ''
    else:
        url = str(repo or '')
    url = url.replace('git+','')
    if url.endswith('.git'):
        url = url[:-4]
    return url or None

def load_package_json_from_tarball(pkg_name: str) -> dict:
    """Fallback: download npm tarball and read package/package.json (stdlib-only)."""
    meta = http_get_json(f"https://registry.npmjs.org/{pkg_name}")
    latest = meta.get('dist-tags',{}).get('latest')
    if not latest:
        raise ValueError('cannot resolve latest version for tarball')
    ver = meta.get('versions',{}).get(latest,{})
    tar_url = ver.get('dist',{}).get('tarball')
    if not tar_url:
        raise ValueError('no tarball url in registry')
    req = Request(tar_url, headers=_UA)
    with urlopen(req) as r:
        blob = r.read()
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as tf:
        member = tf.getmember('package/package.json')
        f = tf.extractfile(member)
        if not f:
            raise ValueError('package.json not found inside tarball')
        txt = f.read().decode('utf-8')
    return json.loads(txt)

def load_package_json_from_local(path: str) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"local path not found: {path}")
    if p.suffix == ".tgz":
        # читать package/package.json из tarball
        with tarfile.open(p, mode="r:gz") as tf:
            member = tf.getmember("package/package.json")
            f = tf.extractfile(member)
            if not f:
                raise ValueError("package.json not found inside tarball")
            txt = f.read().decode("utf-8")
        return json.loads(txt)
    else:
        # обычный локальный package.json
        return json.loads(p.read_text(encoding="utf-8"))

def load_effective_package_json(pkg_name: str, repo_url_or_none: Optional[str]) -> dict:
    # 0) Локальный путь
    if repo_url_or_none and (repo_url_or_none.startswith("file://") or os.path.exists(repo_url_or_none)):
        local_path = repo_url_or_none.replace("file://", "")
        return load_package_json_from_local(local_path)

    # 1) GitHub (корень + packages/<pkg_name>/package.json)
    if repo_url_or_none:
        doc = try_github_package_json(repo_url_or_none, pkg_name)
        if doc is not None:
            return doc

    # 2) npm tarball (фолбэк, требует интернет)
    return load_package_json_from_tarball(pkg_name)


# ---------------- Graph model ----------------

class Graph:
    def __init__(self):
        self.edges: Dict[str, Set[str]] = {}
    def add_edge(self, u: str, v: str) -> None:
        self.edges.setdefault(u, set()).add(v)
        self.edges.setdefault(v, set())
    def nodes(self) -> List[str]:
        s = set(self.edges.keys())
        for vs in self.edges.values():
            s |= set(vs)
        return sorted(s)
    def reverse(self) -> 'Graph':
        g = Graph()
        for u, vs in self.edges.items():
            for v in vs:
                g.add_edge(v, u)
        for n in self.nodes():
            g.edges.setdefault(n, set())
        return g

# -------------- Traversal (BFS with recursion) --------------

def bfs_recursive(expand_fn, roots: List[str], max_depth: int) -> Graph:
    g = Graph()
    visited: Set[str] = set()
    def visit(level: List[str], depth: int) -> None:
        if depth > max_depth or not level:
            return
        next_level: List[str] = []
        for u in level:
            if u in visited:
                continue
            visited.add(u)
            try:
                children = list(expand_fn(u))
            except Exception:
                children = []
            for v in children:
                g.add_edge(u, v)
                if v not in visited:
                    next_level.append(v)
        visit(next_level, depth + 1)
    visit(list(roots), 1)
    return g

# expand functions

def expand_npm(name: str) -> List[str]:
    repo = npm_registry_repo(name)  # may be None
    pkg_json = load_effective_package_json(name, repo)
    deps = extract_direct_deps(pkg_json)
    return sorted(deps.keys())


def expand_from_test_graph(mapping: Dict[str, List[str]]):
    def _exp(name: str) -> List[str]:
        return list(mapping.get(name, []))
    return _exp

# parse test graph file A:B,C

def read_test_graph(path: str) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"test graph not found: {path}")
    mp: Dict[str, List[str]] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' not in line:
                raise ValueError(f"bad graph line: {line}")
            left, right = line.split(':', 1)
            left = left.strip()
            if not re.fullmatch(r'[A-Z]+', left):
                raise ValueError(f"node must be caps: {left}")
            children = [x.strip() for x in right.split(',') if x.strip()]
            for ch in children:
                if not re.fullmatch(r'[A-Z]+', ch):
                    raise ValueError(f"child must be caps: {ch}")
            mp[left] = children
    return mp

# ---------------- ASCII tree ----------------

def print_ascii_tree(g: Graph, root: str, max_depth: int) -> str:
    lines: List[str] = []
    seen: Set[str] = set()
    def walk(node: str, prefix: str, depth: int) -> None:
        cyc = '' if node not in seen else ' (cycle)'
        lines.append(f"{prefix}{node}{cyc}")
        if depth >= max_depth or node in seen:
            return
        seen.add(node)
        children = sorted(g.edges.get(node, []))
        for i, ch in enumerate(children):
            branch = '└─ ' if i == len(children)-1 else '├─ '
            walk(ch, prefix + branch, depth + 1)
    walk(root, '', 1)
    return '\n'.join(lines)

# ---------------- Mermaid ----------------

def to_mermaid(g: Graph) -> str:
    lines = ["graph TD"]
    for u in sorted(g.edges.keys()):
        for v in sorted(g.edges[u]):
            lines.append(f"  {u} --> {v}")
    if len(lines) == 1:
        lines.append("  %% empty graph")
    return "\n".join(lines)

# ---------------- CLI ----------------

def extract_direct_deps(pkg_json: dict) -> Dict[str, str]:
    deps: Dict[str, str] = {}
    for key in ('dependencies','optionalDependencies','peerDependencies'):
        for k, v in (pkg_json.get(key) or {}).items():
            deps[str(k)] = str(v)
    return deps


def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    p = argparse.ArgumentParser(description='Variant 22: npm dependency graph visualizer (stdlib)')
    p.add_argument('--config', required=True)
    p.add_argument('--stage', choices=['1','2','3','4','5'], required=True)
    p.add_argument('--reverse-of', help='(stage 4) show reverse deps of PACKAGE')
    args = p.parse_args(argv)

    cfg = AppConfig.from_csv(args.config)

    # Stage 1 — print parameters
    if args.stage == '1':
        print(cfg.dump_kv())
        return

    # Determine source
    if cfg.test_mode:
        mapping = read_test_graph(cfg.repo_or_path)
        exp = expand_from_test_graph(mapping)
    else:
        exp = expand_npm

    # Stage 2 — direct dependencies only
    if args.stage == '2':
        if cfg.test_mode:
            deps = mapping.get(cfg.package_name, [])
        else:
            try:
                repo = cfg.repo_or_path or npm_registry_repo(cfg.package_name)
                pkg_json = load_effective_package_json(cfg.package_name, repo)
                deps = sorted(extract_direct_deps(pkg_json).keys())
            except Exception as e:
                print(f"error: cannot fetch dependencies: {e}", file=sys.stderr)
                sys.exit(2)
        print("\n".join(deps) if deps else "(no direct dependencies)")
        return

    # Stages 3/4/5 — build graph once
    g = bfs_recursive(exp, [cfg.package_name], cfg.max_depth)

    if args.stage == '3':
        print('NODES:', ', '.join(g.nodes()))
        print('EDGES:')
        for u in sorted(g.edges.keys()):
            for v in sorted(g.edges[u]):
                print(f"{u} -> {v}")
        return

    if args.stage == '4':
        target = args.reverse_of or cfg.package_name
        rg = g.reverse()
        def exp_rev(x: str) -> List[str]:
            return list(rg.edges.get(x, []))
        rev_g = bfs_recursive(exp_rev, [target], cfg.max_depth)
        print('REVERSE DEPENDENCIES (who depends on', target + '):')
        for u in sorted(rev_g.edges.keys()):
            for v in sorted(rev_g.edges[u]):
                print(f"{u} <- {v}")
        return

    if args.stage == '5':
        print(to_mermaid(g))
        if cfg.ascii_tree:
            print('\n--- ASCII TREE ---')
            print(print_ascii_tree(g, cfg.package_name, cfg.max_depth))
        return


if __name__ == '__main__':
    main()
