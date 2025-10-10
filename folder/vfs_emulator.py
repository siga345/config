#!/usr/bin/env python3
"""
VFS Emulator — Variant 22 (GUI + ZIP-backed in‑memory FS)
Meets stages 1–5 from the assignment:

• Stage 1 (REPL GUI): window title with VFS name, simple parser, error messages, stub commands (now real), exit, interactive demo ready.
• Stage 2 (Configuration): CLI args --vfs and --script; script supports comments; echo input+output; script errors surfaced.
• Stage 3 (VFS): all in memory; source is ZIP archive; load errors handled.
• Stage 4 (Core commands): ls, cd, uniq, find, du implemented.
• Stage 5 (Additional): chown implemented.

Run examples (after creating sample ZIP/scripts):
    python vfs_emulator.py --vfs vfs_sample_min.zip --script start_stage_all.txt

Author: you + ChatGPT
"""
from __future__ import annotations
import argparse
import base64
import io
import os
import sys
import traceback
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Iterable

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ----------------------------- In-memory VFS ---------------------------------
@dataclass
class VNode:
    name: str
    kind: str  # 'dir' or 'file'
    parent: Optional['VNode'] = None
    children: Dict[str, 'VNode'] = field(default_factory=dict)  # for dirs
    content: bytes = b''  # for files
    owner: str = 'root'

    def path(self) -> str:
        parts = []
        node: Optional['VNode'] = self
        while node and node.parent is not None:
            parts.append(node.name)
            node = node.parent
        return '/' + '/'.join(reversed(parts))

    def is_dir(self) -> bool:
        return self.kind == 'dir'

    def is_file(self) -> bool:
        return self.kind == 'file'

class VFS:
    def __init__(self, name: str = 'VFS'):
        self.root = VNode(name='/', kind='dir', parent=None)
        self.cwd = self.root
        self.name = name

    # Path utilities
    def _resolve(self, path: str) -> VNode:
        if not path:
            return self.cwd
        node = self.root if path.startswith('/') else self.cwd
        for part in filter(None, path.split('/')):
            if part == '.':
                continue
            if part == '..':
                node = node.parent or self.root
                continue
            if not node.is_dir() or part not in node.children:
                raise FileNotFoundError(f'Path not found: {path}')
            node = node.children[part]
        return node

    def mkdirs(self, path: str) -> VNode:
        node = self.root
        for part in filter(None, path.split('/')):
            if part not in node.children:
                node.children[part] = VNode(name=part, kind='dir', parent=node)
            node = node.children[part]
            if not node.is_dir():
                raise NotADirectoryError(f'Not a directory in path: {part}')
        return node

    def write_file(self, path: str, data: bytes, owner: str = 'root') -> VNode:
        dir_path, fname = os.path.split(path)
        dir_node = self.mkdirs('/' + dir_path.strip('/')) if dir_path else self.root
        node = VNode(name=fname, kind='file', parent=dir_node, content=data, owner=owner)
        dir_node.children[fname] = node
        return node

    # Load from ZIP (Stage 3)
    @classmethod
    def from_zip(cls, zip_bytes: bytes, name: str = 'VFS') -> 'VFS':
        vfs = cls(name=name)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            # 1) Собираем пути, сразу фильтруя macOS-мусор
            entries = []
            for info in zf.infolist():
                p = info.filename.strip('/')
                if not p:
                    continue
                # пропускаем служебные от macOS
                if p.startswith('__MACOSX/') or os.path.basename(p).startswith('._'):
                    continue
                entries.append((p, info))

            # 2) Вычисляем общий верхний сегмент уже по отфильтрованным путям
            tops = {p.split('/', 1)[0] for p, _ in entries}
            common_root = list(tops)[0] if len(tops) == 1 else None

            # 3) Добавляем в VFS, снимая обёртку при необходимости
            for p, info in entries:
                if common_root and (p == common_root or p.startswith(common_root + '/')):
                    p = p[len(common_root):].lstrip('/')
                if not p:
                    continue

                if info.is_dir() or p.endswith('/'):
                    vfs.mkdirs('/' + p.rstrip('/'))
                else:
                    with zf.open(info) as f:
                        data = f.read()
                    vfs.write_file('/' + p, data)
        return vfs

# --------------------------- Command Interpreter ------------------------------
class Interpreter:
    def __init__(self, vfs: VFS, write_cb):
        self.vfs = vfs
        self.write = write_cb  # function(str)

    # Parser: split on spaces, keep quotes
    def parse(self, line: str) -> Tuple[str, List[str]]:
        import shlex
        parts = shlex.split(line)
        if not parts:
            return '', []
        return parts[0], parts[1:]

    # Command implementations
    def cmd_ls(self, args: List[str]):
        path = args[0] if args else ''
        node = self.vfs._resolve(path)
        if not node.is_dir():
            raise NotADirectoryError('ls: target is not a directory')
        entries = sorted(node.children.values(), key=lambda n: (n.kind != 'dir', n.name))
        lines = []
        for e in entries:
            flag = '/' if e.is_dir() else ''
            lines.append(f"{e.name}{flag}")
        self.write('\n'.join(lines) + ('\n' if lines else ''))

    def cmd_cd(self, args: List[str]):
        if len(args) != 1:
            raise ValueError('cd: expected exactly 1 argument')
        node = self.vfs._resolve(args[0])
        if not node.is_dir():
            raise NotADirectoryError('cd: not a directory')
        self.vfs.cwd = node

    def cmd_exit(self, args: List[str]):
        raise SystemExit

    def cmd_find(self, args: List[str]):
        if not args:
            raise ValueError('find: expected PATH [SUBSTR]')
        start = self.vfs._resolve(args[0])
        if not start.is_dir():
            raise NotADirectoryError('find: start path is not a directory')
        substr = args[1] if len(args) > 1 else ''
        results: List[str] = []
        def walk(node: VNode, base: str):
            for child in node.children.values():
                p = f"{base}/{child.name}" if base else f"/{child.name}"
                if substr in child.name:
                    results.append(p + ('/' if child.is_dir() else ''))
                if child.is_dir():
                    walk(child, p)
        walk(start, '' if start is self.vfs.root else start.path())
        self.write('\n'.join(results) + ('\n' if results else ''))

    def cmd_du(self, args: List[str]):
        path = args[0] if args else '.'
        node = self.vfs._resolve(path)
        def size(n: VNode) -> int:
            if n.is_file():
                return len(n.content)
            return sum(size(c) for c in n.children.values())
        total = size(node)
        self.write(f"{total}\n")

    def cmd_uniq(self, args: List[str]):
        if len(args) != 1:
            raise ValueError('uniq: expected FILE')
        node = self.vfs._resolve(args[0])
        if not node.is_file():
            raise IsADirectoryError('uniq: path is a directory')
        try:
            text = node.content.decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError('uniq: file is not UTF-8 text')
        out_lines: List[str] = []
        prev = None
        for line in text.splitlines():
            if line != prev:
                out_lines.append(line)
                prev = line
        self.write('\n'.join(out_lines) + ('\n' if out_lines else ''))

    def cmd_chown(self, args: List[str]):
        if len(args) != 2:
            raise ValueError('chown: expected PATH OWNER')
        node = self.vfs._resolve(args[0])
        node.owner = args[1]

    def dispatch(self, line: str):
        cmd, args = self.parse(line)
        if not cmd:
            return
        try:
            if cmd == 'ls':
                self.cmd_ls(args)
            elif cmd == 'cd':
                self.cmd_cd(args)
            elif cmd == 'exit':
                self.cmd_exit(args)
            elif cmd == 'find':
                self.cmd_find(args)
            elif cmd == 'du':
                self.cmd_du(args)
            elif cmd == 'uniq':
                self.cmd_uniq(args)
            elif cmd == 'chown':
                self.cmd_chown(args)
            else:
                raise ValueError(f'unknown command: {cmd}')
        except SystemExit:
            raise
        except Exception as e:
            self.write(f"error: {e}\n")

# ------------------------------- GUI Shell -----------------------------------
class App(tk.Tk):
    def __init__(self, vfs: VFS, script_path: Optional[str] = None):
        super().__init__()
        self.vfs = vfs
        self.title(f"VFS Emulator — {self.vfs.name}")
        self.geometry('900x600')
        self._build_widgets()
        self.interp = Interpreter(vfs, self._append_output)
        self._prompt()
        if script_path:
            self.after(50, lambda: self.run_script(script_path))

    def _build_widgets(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        # Output area
        self.text = tk.Text(self, wrap='word', state='disabled')
        self.text.grid(row=0, column=0, sticky='nsew')
        # Scrollbar
        sb = ttk.Scrollbar(self, command=self.text.yview)
        sb.grid(row=0, column=1, sticky='ns')
        self.text['yscrollcommand'] = sb.set
        # Input
        self.entry = ttk.Entry(self)
        self.entry.grid(row=1, column=0, columnspan=2, sticky='ew', padx=8, pady=8)
        self.entry.bind('<Return>', self._on_enter)

        # Menu: open script
        menubar = tk.Menu(self)
        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label='Run script…', command=self._choose_and_run_script)
        filem.add_separator()
        filem.add_command(label='Exit', command=self.destroy)
        menubar.add_cascade(label='File', menu=filem)
        self.config(menu=menubar)

    def _append_output(self, s: str):
        self.text.configure(state='normal')
        self.text.insert('end', s)
        self.text.configure(state='disabled')
        self.text.see('end')

    def _prompt(self):
        self._append_output(f"{self.vfs.cwd.path()}$ ")

    def _on_enter(self, event=None):
        line = self.entry.get().strip()
        if not line:
            self._append_output('\n')
            self._prompt()
            return
        self._append_output(line + '\n')  # echo input
        self.entry.delete(0, 'end')
        try:
            self.interp.dispatch(line)
        except SystemExit:
            self.destroy()
            return
        except Exception:
            self._append_output('error: unhandled exception\n')
            traceback.print_exc()
        self._prompt()

    def _choose_and_run_script(self):
        path = filedialog.askopenfilename(title='Choose start script')
        if path:
            self.run_script(path)

    # Stage 2: script execution with comments and error surfacing
    def run_script(self, path: str, keep_open: bool = True):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
        except Exception as e:
            self._append_output(f"error: cannot read script: {e}\n")
            return

        for raw in lines:
            line = raw.strip()
            # срезать inline-комментарии (опционально, но полезно)
            hash_pos = line.find('#')
            if hash_pos != -1:
                line = line[:hash_pos].rstrip()
            if not line:
                continue

            self._append_output(f"{self.vfs.cwd.path()}$ {line}\n")
            try:
                self.interp.dispatch(line)
            except SystemExit:
                self._append_output("(script requested exit)\n")
                if not keep_open:
                    self.destroy()
                    return
                # если keep_open=True — просто прекращаем выполнение скрипта
                break
            except Exception as e:
                self._append_output(f"error: script: {e}\n")

        self._prompt()

# ------------------------------ Main / CLI -----------------------------------
def load_vfs_from_zip_path(vfs_zip_path: Optional[str]) -> VFS:
    if vfs_zip_path is None:
        # Empty default VFS
        vfs = VFS(name='(empty)')
        # Provide a friendly welcome file
        vfs.mkdirs('/docs')
        vfs.write_file('/docs/readme.txt', b"Welcome to VFS!\nTry: ls, find /, du, uniq /docs/readme.txt\n")
        return vfs
    if not os.path.exists(vfs_zip_path):
        raise FileNotFoundError(f"VFS zip not found: {vfs_zip_path}")
    try:
        with open(vfs_zip_path, 'rb') as f:
            data = f.read()
        vfs = VFS.from_zip(data, name=os.path.basename(vfs_zip_path))
        return vfs
    except zipfile.BadZipFile:
        raise ValueError('invalid ZIP format for VFS')


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description='VFS Emulator — Variant 22')
    parser.add_argument('--vfs', help='path to VFS ZIP')
    parser.add_argument('--script', help='path to startup script')
    args = parser.parse_args(argv)

    try:
        vfs = load_vfs_from_zip_path(args.vfs)
    except Exception as e:
        messagebox.showerror('VFS load error', str(e))
        # Still open GUI with empty FS so that error is visible and user can continue
        vfs = VFS(name='(load failed)')

    app = App(vfs, script_path=args.script)
    try:
        app.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
