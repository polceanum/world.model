#!/usr/bin/env python3
"""Hardened two-stage CLI for known-action qualification attempt one.

The outer process is standard-library-only and must be invoked as the exact
absolute runner path under ``python -I -S``.  It binds a clean upstream-equal
Git commit and the exact three-file publication surface, then starts an isolated
child from the verified runner blob.  The child consumes a one-shot pipe receipt
before installing an exact-commit loader for project modules.  The qualification
module then revalidates its formal freeze bindings before entering the bound
scene/materializer/evaluator path; protected execution additionally requires
the independently reviewed development artifact hashes.

This construction treats the isolated outer process as an explicit local trust
boundary under the same-user, non-hostile-kernel model.  It is not cryptographic
attestation against a malicious same-user launcher.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import secrets
import stat
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPOSITORY_ROOT / "scripts" / "run_rgbd_known_action_qualification.py"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "rgbd_known_action_planning_cpu.yaml"
PUBLICATION_SURFACE_PATHS = {
    "qualification": "world_model/training/rgbd_known_action_qualification.py",
    "runner": "scripts/run_rgbd_known_action_qualification.py",
    "qualification_test": "tests/unit/test_rgbd_known_action_qualification.py",
}
_QUALIFICATION_MODULE = "world_model.training.rgbd_known_action_qualification"
_RECEIPT_SCHEMA = "rgbd_known_action_outer_preflight_v1"
_AUTHORIZATION_SCHEMA = "rgbd_known_action_outer_authorization_v1"
_RECEIPT_ENV_PREFIX = "_RGBD_KNOWN_ACTION_OUTER_"
_RECEIPT_FD_ENV = f"{_RECEIPT_ENV_PREFIX}FD"
_RECEIPT_SHA_ENV = f"{_RECEIPT_ENV_PREFIX}SHA256"
_RECEIPT_NONCE_ENV = f"{_RECEIPT_ENV_PREFIX}NONCE"
_MAX_RECEIPT_BYTES = 16 * 1024
_MAX_AUTHORIZATION_BYTES = 16 * 1024
_MAX_SOURCE_BYTES = 8 * 1024 * 1024
_GIT_OID_BYTES = 20
_TRUSTED_GIT = "/usr/bin/git"
_APPROVED_BRANCH = "agent/rgbd-known-action-planning-rung-1"
_APPROVED_REMOTE_NAME = "origin"
_APPROVED_REMOTE_URL = "git@github.com:polceanum/world.model.git"
_APPROVED_BRANCH_MERGE_REF = f"refs/heads/{_APPROVED_BRANCH}"
_APPROVED_UPSTREAM_REF = f"refs/remotes/{_APPROVED_REMOTE_NAME}/{_APPROVED_BRANCH}"
_REMOTE_PROBE_CWD = "/"
_REMOTE_TEMPORARY_DIRECTORY = "/private/tmp"
_REMOTE_PROBE_TIMEOUT_SECONDS = 20
_REMOTE_PROBE_MAX_OUTPUT_BYTES = 4096
_LIGHTWEIGHT_QUALIFICATION_EXECUTION_SHA256 = (
    "68e472d8356143ecf89647f7d98d69f914e1f448d58827378a2ef75f1af8a4c3"
)
_PINNED_GITHUB_HOST_KEY = (
    "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl\n"
)
_PINNED_GITHUB_HOST_KEY_FINGERPRINT = "SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU"
_REMOTE_SSH_COMMAND_TEMPLATE = (
    "/usr/bin/ssh -F /dev/null -oBatchMode=yes -oClearAllForwardings=yes "
    "-oForwardAgent=no -oForwardX11=no -oProxyCommand=none -oProxyJump=none "
    "-oCanonicalizeHostname=no -oStrictHostKeyChecking=yes -oCheckHostIP=yes "
    "-oPasswordAuthentication=no -oKbdInteractiveAuthentication=no "
    "-oIdentityFile=/dev/null -oIdentitiesOnly=no -oIdentityAgent=SSH_AUTH_SOCK "
    "-oAddKeysToAgent=no -oPKCS11Provider=none "
    "-oSecurityKeyProvider=none -oGSSAPIAuthentication=no "
    "-oHostbasedAuthentication=no -oPubkeyAuthentication=yes "
    "-oHostKeyAlgorithms=ssh-ed25519 -oHostKeyAlias=github.com "
    "-oUserKnownHostsFile={known_hosts_file} -oGlobalKnownHostsFile=/dev/null"
)
_BOOTSTRAP = r"""
def _rgbd_known_action_bootstrap():
    import argparse as _aa
    import ast as _a
    import builtins as _b
    import contextlib as _x
    import hashlib as _h
    import importlib as _i
    import importlib.abc as _ia
    import importlib.machinery as _im
    import importlib.util as _iu
    import json as _j
    import os as _o
    import pathlib as _p
    import secrets as _q
    import stat as _s
    import subprocess as _u
    import sys as _y
    import sysconfig as _c
    import tempfile as _t
    import types as _m
    import typing as _g

    _compile_fn = _b.compile
    _exec_fn = _b.exec
    _import_fn = _b.__import__
    _open_fn = _b.open
    _getframe_fn = _y._getframe
    _get_paths_fn = _c.get_paths

    (
        _fd_text,
        _expected,
        _path,
        _authorization_fd_text,
        _bootstrap_literal_sha256,
    ) = _y.argv[1:6]
    _user_argv = list(_y.argv[6:])
    if not _fd_text.isascii() or not _fd_text.isdecimal() or str(int(_fd_text)) != _fd_text:
        raise PermissionError("bootstrap descriptor is not canonical")
    _fd = int(_fd_text)
    if _fd < 3:
        raise PermissionError("bootstrap descriptor cannot be a standard stream")
    try:
        _before = _o.fstat(_fd)
        if not _s.S_ISREG(_before.st_mode) or _before.st_nlink != 0 or _before.st_size > 8*1024*1024:
            raise PermissionError("bootstrap source is not one anonymous bounded file")
        _o.set_inheritable(_fd, False)
        _chunks = []
        _total = 0
        while True:
            _chunk = _o.read(_fd, 65536)
            if not _chunk:
                break
            _chunks.append(_chunk)
            _total += len(_chunk)
            if _total > 8*1024*1024:
                raise PermissionError("bootstrap source is oversized")
        _source = b"".join(_chunks)
        _after = _o.fstat(_fd)
    finally:
        _o.close(_fd)
    _identity = lambda _v: (_v.st_dev, _v.st_ino, _v.st_mode, _v.st_nlink, _v.st_size)
    if _identity(_before) != _identity(_after) or len(_source) != _before.st_size:
        raise PermissionError("bootstrap source changed during read")
    if len(_expected) != 64 or _h.sha256(_source).hexdigest() != _expected:
        raise PermissionError("bootstrap source digest differs")
    if (len(_bootstrap_literal_sha256) != 64
            or _bootstrap_literal_sha256 != _bootstrap_literal_sha256.lower()):
        raise PermissionError("bootstrap literal digest is malformed")
    try:
        int(_bootstrap_literal_sha256, 16)
    except ValueError as _error:
        raise PermissionError("bootstrap literal digest is not hexadecimal") from _error
    try:
        _runner_text = _source.decode("utf-8", errors="strict")
    except UnicodeDecodeError as _error:
        raise PermissionError("bootstrap source is not strict UTF-8") from _error
    _tree = _a.parse(_runner_text, filename=_path, mode="exec")
    _bootstrap_assignments = []
    for _node in _tree.body:
        if (type(_node) is _a.Assign and len(_node.targets) == 1
                and type(_node.targets[0]) is _a.Name
                and _node.targets[0].id == "_BOOTSTRAP"):
            _bootstrap_assignments.append(_node)
    if (len(_bootstrap_assignments) != 1
            or type(_bootstrap_assignments[0].value) is not _a.Constant
            or type(_bootstrap_assignments[0].value.value) is not str):
        raise PermissionError("published runner bootstrap literal differs")
    _published_bootstrap = _bootstrap_assignments[0].value.value
    if _h.sha256(_published_bootstrap.encode("utf-8")).hexdigest() != _bootstrap_literal_sha256:
        raise PermissionError("published bootstrap literal digest differs")
    _orig_argv = list(_y.orig_argv)
    _command_positions = [
        _index for _index, _value in enumerate(_orig_argv) if _value == "-c"
    ]
    if (len(_command_positions) != 1
            or _orig_argv[1:_command_positions[0]] != ["-I", "-S"]
            or _command_positions[0] + 1 >= len(_orig_argv)
            or _orig_argv[_command_positions[0] + 1] != _published_bootstrap
            or _orig_argv[_command_positions[0] + 2:] != list(_y.argv[1:])):
        raise PermissionError("executing bootstrap differs from published runner literal")
    if (not _authorization_fd_text.isascii() or not _authorization_fd_text.isdecimal()
            or str(int(_authorization_fd_text)) != _authorization_fd_text
            or int(_authorization_fd_text) < 3):
        raise PermissionError("authorization descriptor is not canonical")
    _flags = {
        "isolated": _y.flags.isolated,
        "no_site": _y.flags.no_site,
        "ignore_environment": _y.flags.ignore_environment,
        "no_user_site": _y.flags.no_user_site,
        "safe_path": getattr(_y.flags, "safe_path", None),
    }
    if _flags != {
        "isolated": 1,
        "no_site": 1,
        "ignore_environment": 1,
        "no_user_site": 1,
        "safe_path": None,
    }:
        raise PermissionError("bootstrap interpreter isolation flags differ")
    _paths = _c.get_paths()
    _stdlib = _o.path.realpath(_paths["stdlib"])
    _platstdlib = _o.path.realpath(_paths["platstdlib"])
    _zip = _o.path.join(_o.path.dirname(_stdlib), "python%d%d.zip" % _y.version_info[:2])
    _dynload = _o.path.join(_stdlib, "lib-dynload")
    _expected_preloader = tuple(dict.fromkeys((_zip, _stdlib, _platstdlib, _dynload)))
    _preloader = tuple(_y.path)
    if _preloader != _expected_preloader:
        raise PermissionError("bootstrap preloader sys.path shape differs")
    _cwd = _o.path.realpath(_o.getcwd())
    _repo = _o.path.realpath(_o.path.dirname(_o.path.dirname(_path)))
    if any(
        not _item or not _o.path.isabs(_item)
        or _o.path.realpath(_item) in {_cwd, _repo}
        for _item in _preloader
    ):
        raise PermissionError("bootstrap preloader sys.path contains an unsafe entry")
    if any(_name == "world_model" or _name.startswith("world_model.") for _name in _y.modules):
        raise PermissionError("project modules were preloaded before bootstrap isolation")
    for _site_path in dict.fromkeys((_c.get_paths()["purelib"], _c.get_paths()["platlib"])):
        if _site_path not in _y.path:
            _y.path.append(_site_path)
    _postloader = tuple(_y.path)
    _expected_postloader = tuple(dict.fromkeys((*_expected_preloader, _paths["purelib"], _paths["platlib"])))
    if _postloader != _expected_postloader:
        raise PermissionError("bootstrap runtime sys.path shape differs")
    _sys_modules = _y.modules
    _sys_modules_snapshot = tuple(_sys_modules.items())
    _sys_meta_path = _y.meta_path
    _sys_meta_path_snapshot = tuple(_sys_meta_path)
    _sys_path = _y.path
    _sys_path_snapshot = tuple(_sys_path)
    _sys_path_hooks = _y.path_hooks
    _sys_path_hooks_snapshot = tuple(_sys_path_hooks)
    _sys_path_importer_cache = _y.path_importer_cache
    _sys_path_importer_cache_snapshot = tuple(_sys_path_importer_cache.items())
    _os_environ = _o.environ
    _environment = {
        "schema": "rgbd_known_action_bootstrap_environment_v1",
        "python_version": _y.version,
        "python_hexversion": _y.hexversion,
        "implementation": _y.implementation.name,
        "executable": _y.executable,
        "flags": _flags,
        "preloader_sys_path": list(_preloader),
        "preloader_sys_path_sha256": _h.sha256(_j.dumps(list(_preloader),allow_nan=False,ensure_ascii=True,separators=(",",":"),sort_keys=True).encode("ascii")).hexdigest(),
        "runtime_sys_path": list(_postloader),
        "runtime_sys_path_sha256": _h.sha256(_j.dumps(list(_postloader),allow_nan=False,ensure_ascii=True,separators=(",",":"),sort_keys=True).encode("ascii")).hexdigest(),
        "runner_blob_sha256": _expected,
        "bootstrap_literal_sha256": _bootstrap_literal_sha256,
        "preloaded_world_model": [],
    }
    _trusted_attributes = (
        ("argparse.ArgumentParser", _aa, "ArgumentParser", _aa.ArgumentParser),
        ("ast.parse", _a, "parse", _a.parse),
        ("builtins.compile", _b, "compile", _compile_fn),
        ("builtins.exec", _b, "exec", _exec_fn),
        ("builtins.__import__", _b, "__import__", _import_fn),
        ("builtins.open", _b, "open", _open_fn),
        ("builtins.getattr", _b, "getattr", _b.getattr),
        ("builtins.hasattr", _b, "hasattr", _b.hasattr),
        ("builtins.type", _b, "type", _b.type),
        ("builtins.isinstance", _b, "isinstance", _b.isinstance),
        ("builtins.callable", _b, "callable", _b.callable),
        ("builtins.len", _b, "len", _b.len),
        ("builtins.any", _b, "any", _b.any),
        ("builtins.all", _b, "all", _b.all),
        ("builtins.set", _b, "set", _b.set),
        ("builtins.dict", _b, "dict", _b.dict),
        ("builtins.list", _b, "list", _b.list),
        ("builtins.tuple", _b, "tuple", _b.tuple),
        ("builtins.int", _b, "int", _b.int),
        ("builtins.str", _b, "str", _b.str),
        ("builtins.bytes", _b, "bytes", _b.bytes),
        ("builtins.object", _b, "object", _b.object),
        ("builtins.print", _b, "print", _b.print),
        ("builtins.repr", _b, "repr", _b.repr),
        ("builtins.sorted", _b, "sorted", _b.sorted),
        ("builtins.zip", _b, "zip", _b.zip),
        ("builtins.enumerate", _b, "enumerate", _b.enumerate),
        ("contextlib.suppress", _x, "suppress", _x.suppress),
        ("hashlib.sha1", _h, "sha1", _h.sha1),
        ("hashlib.sha256", _h, "sha256", _h.sha256),
        ("importlib.abc.MetaPathFinder", _ia, "MetaPathFinder", _ia.MetaPathFinder),
        ("importlib.abc.Loader", _ia, "Loader", _ia.Loader),
        ("importlib.machinery.BuiltinImporter", _im, "BuiltinImporter", _im.BuiltinImporter),
        ("importlib.machinery.ModuleSpec", _im, "ModuleSpec", _im.ModuleSpec),
        ("importlib.abc", _i, "abc", _ia),
        ("importlib.machinery", _i, "machinery", _im),
        ("importlib.util", _i, "util", _iu),
        ("importlib.util.spec_from_loader", _iu, "spec_from_loader", _iu.spec_from_loader),
        ("json.dumps", _j, "dumps", _j.dumps),
        ("json.loads", _j, "loads", _j.loads),
        ("os.close", _o, "close", _o.close),
        ("os.fspath", _o, "fspath", _o.fspath),
        ("os.fstat", _o, "fstat", _o.fstat),
        ("os.fsync", _o, "fsync", _o.fsync),
        ("os.geteuid", _o, "geteuid", _o.geteuid),
        ("os.getpid", _o, "getpid", _o.getpid),
        ("os.getppid", _o, "getppid", _o.getppid),
        ("os.open", _o, "open", _o.open),
        ("os.lstat", _o, "lstat", _o.lstat),
        ("os.pipe", _o, "pipe", _o.pipe),
        ("os.read", _o, "read", _o.read),
        ("os.set_inheritable", _o, "set_inheritable", _o.set_inheritable),
        ("os.write", _o, "write", _o.write),
        ("os.path", _o, "path", _o.path),
        ("os.path.dirname", _o.path, "dirname", _o.path.dirname),
        ("os.path.isabs", _o.path, "isabs", _o.path.isabs),
        ("os.path.join", _o.path, "join", _o.path.join),
        ("os.path.realpath", _o.path, "realpath", _o.path.realpath),
        ("pathlib.Path.is_absolute", _p.Path, "is_absolute", _p.Path.is_absolute),
        ("pathlib.Path.lstat", _p.Path, "lstat", _p.Path.lstat),
        ("pathlib.Path.resolve", _p.Path, "resolve", _p.Path.resolve),
        ("pathlib.Path.__truediv__", _p.Path, "__truediv__", _p.Path.__truediv__),
        ("secrets.token_hex", _q, "token_hex", _q.token_hex),
        ("stat.S_ISDIR", _s, "S_ISDIR", _s.S_ISDIR),
        ("stat.S_ISFIFO", _s, "S_ISFIFO", _s.S_ISFIFO),
        ("stat.S_ISREG", _s, "S_ISREG", _s.S_ISREG),
        ("stat.S_ISSOCK", _s, "S_ISSOCK", _s.S_ISSOCK),
        ("subprocess.Popen", _u, "Popen", _u.Popen),
        ("subprocess.run", _u, "run", _u.run),
        ("sys._getframe", _y, "_getframe", _getframe_fn),
        ("sysconfig.get_paths", _c, "get_paths", _get_paths_fn),
        ("tempfile.TemporaryFile", _t, "TemporaryFile", _t.TemporaryFile),
        ("tempfile.TemporaryDirectory", _t, "TemporaryDirectory", _t.TemporaryDirectory),
    )
    _compiled = _compile_fn(_source, _path, "exec", dont_inherit=True)
    _seal = _b.object()
    _bootstrap_frame = _y._getframe()
    _main_capability_used = [False]
    _internal_capability_used = [False]
    _trusted_namespace = {
        "__name__": "_rgbd_known_action_verified_bootstrap",
        "__file__": _path,
        "__package__": None,
        "__spec__": None,
        "__cached__": None,
        "__loader__": _im.BuiltinImporter,
        "__builtins__": _b,
        "_BOOTSTRAP_SECURITY_SEAL": _seal,
    }

    def _main_capability():
        _caller = _getframe_fn(1)
        if (_main_capability_used[0]
                or _caller.f_code is not _trusted_namespace["main"].__code__
                or _caller.f_globals is not _trusted_namespace
                or _caller.f_locals.get("_bootstrap_security") is not _main_capability
                or _caller.f_locals.get("_expected_capability") is not _main_capability
                or _caller.f_locals.get("_environ") is not _os_environ
                or _caller.f_locals.get("_sys") is not _y
                or _caller.f_locals.get("_consume") is not _trusted_namespace["_consume_outer_receipt"]
                or _caller.f_locals.get("_internal") is not _trusted_namespace["_internal_main"]
                or _caller.f_locals.get("_outer") is not _trusted_namespace["_outer_main"]
                or _caller.f_locals.get("argv") is not None
                or _caller.f_locals.get("_authorization_fd_text") is not _authorization_fd_text
                or _caller.f_locals.get("_bootstrap_environment_hex") != _environment_bytes.hex()
                or _caller.f_back is not _bootstrap_frame):
            raise PermissionError("bootstrap main capability caller differs")
        _main_capability_used[0] = True
        return _internal_capability

    def _internal_capability():
        _caller = _getframe_fn(1)
        _main_caller = _caller.f_back
        if (_internal_capability_used[0]
                or _caller.f_code is not _trusted_namespace["_internal_main"].__code__
                or _caller.f_globals is not _trusted_namespace
                or _caller.f_locals.get("bootstrap_security") is not _internal_capability
                or _caller.f_locals.get("_expected_capability") is not _internal_capability
                or _caller.f_locals.get("_expected_seal") is not _seal
                or _caller.f_locals.get("_capture") is not _trusted_namespace["_capture_outer_receipt"]
                or _caller.f_locals.get("_consume") is not _trusted_namespace["_consume_outer_receipt"]
                or _caller.f_locals.get("_loader_type") is not _loader_type
                or _caller.f_locals.get("_canonical") is not _trusted_namespace["_canonical_json"]
                or _caller.f_locals.get("_arguments") is not _trusted_namespace["arguments"]
                or _caller.f_locals.get("_import") is not _import_fn
                or _caller.f_locals.get("_json_dumps") is not _j.dumps
                or _caller.f_locals.get("_suppress") is not _x.suppress
                or _caller.f_locals.get("_sys") is not _y
                or _caller.f_locals.get("_repository_root") is not _trusted_namespace["REPOSITORY_ROOT"]
                or _caller.f_locals.get("_config_path") is not _trusted_namespace["CONFIG_PATH"]
                or _caller.f_locals.get("_qualification_name") is not _trusted_namespace["_QUALIFICATION_MODULE"]
                or _caller.f_locals.get("_qualification_helpers") is not _trusted_namespace["_qualification_callables"]
                or _caller.f_locals.get("_hash") is not _trusted_namespace["_sha256"]
                or _caller.f_locals.get("_print") is not _b.print
                or _main_caller is None
                or _main_caller.f_code is not _trusted_namespace["main"].__code__
                or _main_caller.f_globals is not _trusted_namespace
                or _y.modules is not _sys_modules
                or tuple(_y.modules.items()) != _sys_modules_snapshot
                or _y.meta_path is not _sys_meta_path
                or tuple(_y.meta_path) != _sys_meta_path_snapshot
                or _y.path is not _sys_path
                or tuple(_y.path) != _sys_path_snapshot
                or _y.path_hooks is not _sys_path_hooks
                or tuple(_y.path_hooks) != _sys_path_hooks_snapshot
                or _y.path_importer_cache is not _sys_path_importer_cache
                or tuple(_y.path_importer_cache.items()) != _sys_path_importer_cache_snapshot
                or _o.environ is not _os_environ
                or any(_name == "world_model" or _name.startswith("world_model.")
                       for _name in _y.modules)
                or _main_caller.f_back is not _bootstrap_frame):
            raise PermissionError("bootstrap internal capability caller differs")
        _internal_capability_used[0] = True
        return _security

    _trusted_namespace["_BOOTSTRAP_MAIN_CAPABILITY"] = _main_capability
    _trusted_namespace["_BOOTSTRAP_INTERNAL_CAPABILITY"] = _internal_capability
    _exec_fn(_compiled, _trusted_namespace, _trusted_namespace)
    if (_y.modules is not _sys_modules
            or tuple(_y.modules.items()) != _sys_modules_snapshot
            or _y.meta_path is not _sys_meta_path
            or tuple(_y.meta_path) != _sys_meta_path_snapshot
            or _y.path is not _sys_path
            or tuple(_y.path) != _sys_path_snapshot
            or _y.path_hooks is not _sys_path_hooks
            or tuple(_y.path_hooks) != _sys_path_hooks_snapshot
            or _y.path_importer_cache is not _sys_path_importer_cache
            or tuple(_y.path_importer_cache.items()) != _sys_path_importer_cache_snapshot
            or _o.environ is not _os_environ
            or any(_name == "world_model" or _name.startswith("world_model.")
                   for _name in _y.modules)):
        raise PermissionError("runner preflight interpreter container binding differs")
    _module_identities = (
        ("argparse", _aa),
        ("ast", _a),
        ("builtins", _b),
        ("contextlib", _x),
        ("hashlib", _h),
        ("importlib", _i),
        ("json", _j),
        ("os", _o),
        ("secrets", _q),
        ("stat", _s),
        ("subprocess", _u),
        ("sys", _y),
        ("sysconfig", _c),
        ("tempfile", _t),
    )
    if (any(_trusted_namespace.get(_name) is not _module for _name, _module in _module_identities)
            or _trusted_namespace.get("Path") is not _p.Path
            or _trusted_namespace.get("FunctionType") is not _m.FunctionType
            or _trusted_namespace.get("ModuleType") is not _m.ModuleType
            or _trusted_namespace.get("Any") is not _g.Any
            or any(getattr(_owner, _attribute) is not _expected_attribute
                   for _, _owner, _attribute, _expected_attribute in _trusted_attributes)):
        raise PermissionError("runner preflight module or callable identity differs")
    _expected_constants = {
        "REPOSITORY_ROOT": _p.Path(_repo),
        "RUNNER_PATH": _p.Path(_path),
        "CONFIG_PATH": _p.Path(_repo) / "configs" / "rgbd_known_action_planning_cpu.yaml",
        "PUBLICATION_SURFACE_PATHS": {
            "qualification": "world_model/training/rgbd_known_action_qualification.py",
            "runner": "scripts/run_rgbd_known_action_qualification.py",
            "qualification_test": "tests/unit/test_rgbd_known_action_qualification.py",
        },
        "_QUALIFICATION_MODULE": "world_model.training.rgbd_known_action_qualification",
        "_RECEIPT_SCHEMA": "rgbd_known_action_outer_preflight_v1",
        "_AUTHORIZATION_SCHEMA": "rgbd_known_action_outer_authorization_v1",
        "_RECEIPT_ENV_PREFIX": "_RGBD_KNOWN_ACTION_OUTER_",
        "_RECEIPT_FD_ENV": "_RGBD_KNOWN_ACTION_OUTER_FD",
        "_RECEIPT_SHA_ENV": "_RGBD_KNOWN_ACTION_OUTER_SHA256",
        "_RECEIPT_NONCE_ENV": "_RGBD_KNOWN_ACTION_OUTER_NONCE",
        "_MAX_RECEIPT_BYTES": 16 * 1024,
        "_MAX_AUTHORIZATION_BYTES": 16 * 1024,
        "_MAX_SOURCE_BYTES": 8 * 1024 * 1024,
        "_GIT_OID_BYTES": 20,
        "_TRUSTED_GIT": "/usr/bin/git",
        "_APPROVED_BRANCH": "agent/rgbd-known-action-planning-rung-1",
        "_APPROVED_REMOTE_NAME": "origin",
        "_APPROVED_REMOTE_URL": "git@github.com:polceanum/world.model.git",
        "_APPROVED_BRANCH_MERGE_REF": "refs/heads/agent/rgbd-known-action-planning-rung-1",
        "_APPROVED_UPSTREAM_REF": "refs/remotes/origin/agent/rgbd-known-action-planning-rung-1",
        "_REMOTE_PROBE_CWD": "/",
        "_REMOTE_TEMPORARY_DIRECTORY": "/private/tmp",
        "_REMOTE_PROBE_TIMEOUT_SECONDS": 20,
        "_REMOTE_PROBE_MAX_OUTPUT_BYTES": 4096,
        "_LIGHTWEIGHT_QUALIFICATION_EXECUTION_SHA256": "68e472d8356143ecf89647f7d98d69f914e1f448d58827378a2ef75f1af8a4c3",
        "_PINNED_GITHUB_HOST_KEY": "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl\n",
        "_PINNED_GITHUB_HOST_KEY_FINGERPRINT": "SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU",
        "_REMOTE_SSH_COMMAND_TEMPLATE": "/usr/bin/ssh -F /dev/null -oBatchMode=yes -oClearAllForwardings=yes -oForwardAgent=no -oForwardX11=no -oProxyCommand=none -oProxyJump=none -oCanonicalizeHostname=no -oStrictHostKeyChecking=yes -oCheckHostIP=yes -oPasswordAuthentication=no -oKbdInteractiveAuthentication=no -oIdentityFile=/dev/null -oIdentitiesOnly=no -oIdentityAgent=SSH_AUTH_SOCK -oAddKeysToAgent=no -oPKCS11Provider=none -oSecurityKeyProvider=none -oGSSAPIAuthentication=no -oHostbasedAuthentication=no -oPubkeyAuthentication=yes -oHostKeyAlgorithms=ssh-ed25519 -oHostKeyAlias=github.com -oUserKnownHostsFile={known_hosts_file} -oGlobalKnownHostsFile=/dev/null",
    }
    if (set(_expected_constants) - set(_trusted_namespace)
            or any(type(_trusted_namespace[_name]) is not type(_value)
                   or _trusted_namespace[_name] != _value
                   for _name, _value in _expected_constants.items())):
        raise PermissionError("runner preflight constant binding differs")
    _code_type = type(_compiled)
    _codes = {}
    def _visit_code(_value, _parent=""):
        if type(_value) is not _code_type:
            return
        if hasattr(_value, "co_qualname"):
            _qualname = _value.co_qualname
        elif _value.co_name == "<module>":
            _qualname = "<module>"
        elif _parent:
            _qualname = _parent + "." + _value.co_name
        else:
            _qualname = _value.co_name
        _codes[_qualname] = _value
        _child_parent = "" if _value.co_name == "<module>" else _qualname
        for _constant in _value.co_consts:
            _visit_code(_constant, _child_parent)
    _visit_code(_compiled)
    _function_names = [
        _node.name for _node in _tree.body if type(_node) is _a.FunctionDef
    ]
    for _name in _function_names:
        _helper = _trusted_namespace.get(_name)
        if (type(_helper) is not _m.FunctionType
                or _helper.__name__ != _name
                or _helper.__qualname__ != _name
                or _helper.__globals__ is not _trusted_namespace
                or _helper.__code__ is not _codes.get(_name)
                or _helper.__module__ != "_rgbd_known_action_verified_bootstrap"
                or _helper.__closure__ is not None
                or _helper.__dict__ != {}):
            raise PermissionError("runner preflight helper identity differs: " + _name)
    _loader_nodes = [
        _node for _node in _tree.body
        if type(_node) is _a.ClassDef and _node.name == "_ExactCommitLoader"
    ]
    _loader_type = _trusted_namespace.get("_ExactCommitLoader")
    _loader_method_names = {
        _node.name for _node in _loader_nodes[0].body
        if type(_node) is _a.FunctionDef
    } if len(_loader_nodes) == 1 else set()
    if (len(_loader_nodes) != 1 or type(_loader_type) is not type(_ia.MetaPathFinder)
            or _loader_type.__name__ != "_ExactCommitLoader"
            or _loader_type.__qualname__ != "_ExactCommitLoader"
            or _loader_type.__bases__ != (_ia.MetaPathFinder, _ia.Loader)
            or _loader_type.__module__ != "_rgbd_known_action_verified_bootstrap"
            or set(_loader_type.__dict__) != {
                "__module__", "__doc__", "__abstractmethods__", "_abc_impl",
                *_loader_method_names,
            }):
        raise PermissionError("runner preflight loader type differs")
    for _method_node in _loader_nodes[0].body:
        if type(_method_node) is _a.FunctionDef:
            _method = _loader_type.__dict__.get(_method_node.name)
            if (type(_method) is not _m.FunctionType
                    or _method.__name__ != _method_node.name
                    or _method.__qualname__ != "_ExactCommitLoader." + _method_node.name
                    or _method.__code__ is not _codes.get("_ExactCommitLoader." + _method_node.name)
                    or _method.__globals__ is not _trusted_namespace
                    or _method.__closure__ is not None
                    or _method.__dict__ != {}):
                raise PermissionError("runner preflight loader method differs: " + _method_node.name)
    _default_identities = (
        ("_canonical_json", "_dumps", _j.dumps),
        ("_strict_json_loads", "_loads", _j.loads),
        ("_sha256", "_hash", _h.sha256),
        ("_git_paths", "_is_absolute", _p.Path.is_absolute),
        ("_git_paths", "_resolve", _p.Path.resolve),
        ("_git_paths", "_lstat", _p.Path.lstat),
        ("_git_paths", "_isdir", _s.S_ISDIR),
        ("_git_environment", "_paths", _trusted_namespace["_git_paths"]),
        ("_git_environment", "_fspath", _o.fspath),
        ("_git_environment", "_environ", _os_environ),
        ("_git_command", "_paths", _trusted_namespace["_git_paths"]),
        ("_git_bytes", "_run", _u.run),
        ("_git_bytes", "_command", _trusted_namespace["_git_command"]),
        ("_git_bytes", "_environment", _trusted_namespace["_git_environment"]),
        ("_git_text", "_bytes", _trusted_namespace["_git_bytes"]),
        ("_approved_git_config_snapshot", "_bytes", _trusted_namespace["_git_bytes"]),
        ("_validated_ssh_agent_socket", "_environ", _os_environ),
        ("_validated_ssh_agent_socket", "_lstat", _o.lstat),
        ("_validated_ssh_agent_socket", "_is_socket", _s.S_ISSOCK),
        ("_validated_ssh_agent_socket", "_geteuid", _o.geteuid),
        ("_remote_transport", "_run", _u.run),
        ("_remote_transport", "_agent", _trusted_namespace["_validated_ssh_agent_socket"]),
        ("_object_oid", "_hash", _h.sha1),
        ("_verified_object", "_validate", _trusted_namespace["_validated_oid"]),
        ("_verified_object", "_text", _trusted_namespace["_git_text"]),
        ("_verified_object", "_bytes", _trusted_namespace["_git_bytes"]),
        ("_verified_object", "_object", _trusted_namespace["_object_oid"]),
        ("_stable_read", "_lstat", _p.Path.lstat),
        ("_stable_read", "_isreg", _s.S_ISREG),
        ("_stable_read", "_open", _o.open),
        ("_stable_read", "_fstat", _o.fstat),
        ("_stable_read", "_read", _o.read),
        ("_stable_read", "_close", _o.close),
        ("_blob_binding", "_validate", _trusted_namespace["_validated_oid"]),
        ("_blob_binding", "_text", _trusted_namespace["_git_text"]),
        ("_blob_binding", "_verified", _trusted_namespace["_verified_object"]),
        ("_blob_binding", "_bytes", _trusted_namespace["_git_bytes"]),
        ("_blob_binding", "_read", _trusted_namespace["_stable_read"]),
        ("_blob_binding", "_hash", _trusted_namespace["_sha256"]),
        ("_capture_outer_receipt", "_bytes", _trusted_namespace["_git_bytes"]),
        ("_capture_outer_receipt", "_text", _trusted_namespace["_git_text"]),
        ("_capture_outer_receipt", "_validate", _trusted_namespace["_validated_oid"]),
        ("_capture_outer_receipt", "_verified", _trusted_namespace["_verified_object"]),
        ("_capture_outer_receipt", "_blob", _trusted_namespace["_blob_binding"]),
        ("_capture_outer_receipt", "_config", _trusted_namespace["_approved_git_config_snapshot"]),
        ("_capture_outer_receipt", "_probe", _trusted_namespace["_probe_remote_publication"]),
        ("_capture_outer_receipt", "_fspath", _o.fspath),
        ("_capture_outer_receipt", "_token_hex", _q.token_hex),
        ("_capture_outer_receipt", "_getpid", _o.getpid),
        ("_capture_outer_receipt", "_hash", _trusted_namespace["_sha256"]),
        ("_capture_outer_receipt", "_canonical", _trusted_namespace["_canonical_json"]),
        ("_consume_outer_receipt", "_environ", _os_environ),
        ("_consume_outer_receipt", "_fstat", _o.fstat),
        ("_consume_outer_receipt", "_isfifo", _s.S_ISFIFO),
        ("_consume_outer_receipt", "_set_inheritable", _o.set_inheritable),
        ("_consume_outer_receipt", "_read", _o.read),
        ("_consume_outer_receipt", "_close", _o.close),
        ("_consume_outer_receipt", "_getppid", _o.getppid),
        ("_consume_outer_receipt", "_hash", _trusted_namespace["_sha256"]),
        ("_consume_outer_receipt", "_validate_hash", _trusted_namespace["_validated_sha256"]),
        ("_consume_outer_receipt", "_canonical", _trusted_namespace["_canonical_json"]),
        ("_consume_outer_receipt", "_loads", _trusted_namespace["_strict_json_loads"]),
        ("arguments", "_parser", _aa.ArgumentParser),
        ("arguments", "_validate_hash", _trusted_namespace["_validated_sha256"]),
        ("_qualification_callables", "_compile", _compile_fn),
        ("_qualification_callables", "_fspath", _o.fspath),
        ("_qualification_callables", "_getattr", _b.getattr),
        ("_qualification_callables", "_type", _b.type),
        ("_qualification_callables", "_len", _b.len),
        ("_qualification_callables", "_callable", _b.callable),
        ("_qualification_callables", "_module_type", _m.ModuleType),
        ("_qualification_callables", "_function_type", _m.FunctionType),
        ("_qualification_callables", "_sys", _y),
        ("_qualification_callables", "_path_type", _p.Path),
        ("_qualification_callables", "_qualification_name", _trusted_namespace["_QUALIFICATION_MODULE"]),
        ("_qualification_callables", "_repository_root", _trusted_namespace["REPOSITORY_ROOT"]),
        ("_qualification_callables", "_ast_parse", _a.parse),
        ("_qualification_callables", "_ast_module", _a),
        ("_internal_main", "_expected_capability", _internal_capability),
        ("_internal_main", "_expected_seal", _seal),
        ("_internal_main", "_capture", _trusted_namespace["_capture_outer_receipt"]),
        ("_internal_main", "_consume", _trusted_namespace["_consume_outer_receipt"]),
        ("_internal_main", "_loader_type", _loader_type),
        ("_internal_main", "_canonical", _trusted_namespace["_canonical_json"]),
        ("_internal_main", "_arguments", _trusted_namespace["arguments"]),
        ("_internal_main", "_import", _import_fn),
        ("_internal_main", "_json_dumps", _j.dumps),
        ("_internal_main", "_suppress", _x.suppress),
        ("_internal_main", "_sys", _y),
        ("_internal_main", "_repository_root", _trusted_namespace["REPOSITORY_ROOT"]),
        ("_internal_main", "_config_path", _trusted_namespace["CONFIG_PATH"]),
        ("_internal_main", "_qualification_name", _trusted_namespace["_QUALIFICATION_MODULE"]),
        ("_internal_main", "_qualification_helpers", _trusted_namespace["_qualification_callables"]),
        ("_internal_main", "_hash", _trusted_namespace["_sha256"]),
        ("_internal_main", "_print", _b.print),
        ("_outer_main", "_preflight", _trusted_namespace["_outer_preflight"]),
        ("_outer_main", "_arguments", _trusted_namespace["arguments"]),
        ("_outer_main", "_capture", _trusted_namespace["_capture_outer_receipt"]),
        ("_outer_main", "_canonical", _trusted_namespace["_canonical_json"]),
        ("_outer_main", "_hash", _trusted_namespace["_sha256"]),
        ("_outer_main", "_pipe", _o.pipe),
        ("_outer_main", "_environ", _os_environ),
        ("_outer_main", "_temporary_file", _t.TemporaryFile),
        ("_outer_main", "_fsync", _o.fsync),
        ("_outer_main", "_fstat", _o.fstat),
        ("_outer_main", "_write", _o.write),
        ("_outer_main", "_close", _o.close),
        ("_outer_main", "_run", _u.run),
        ("_outer_main", "_sys", _y),
        ("_outer_main", "_fspath", _o.fspath),
        ("_outer_main", "_repository_root", _trusted_namespace["REPOSITORY_ROOT"]),
        ("_outer_main", "_runner_path", _trusted_namespace["RUNNER_PATH"]),
        ("_outer_main", "_bootstrap", _trusted_namespace["_BOOTSTRAP"]),
        ("_outer_main", "_max_receipt_bytes", _trusted_namespace["_MAX_RECEIPT_BYTES"]),
        ("_outer_main", "_max_authorization_bytes", _trusted_namespace["_MAX_AUTHORIZATION_BYTES"]),
        ("_outer_main", "_receipt_env_prefix", _trusted_namespace["_RECEIPT_ENV_PREFIX"]),
        ("_outer_main", "_receipt_fd_env", _trusted_namespace["_RECEIPT_FD_ENV"]),
        ("_outer_main", "_receipt_sha_env", _trusted_namespace["_RECEIPT_SHA_ENV"]),
        ("_outer_main", "_receipt_nonce_env", _trusted_namespace["_RECEIPT_NONCE_ENV"]),
        ("_outer_main", "_executable", _y.executable),
        ("_outer_main", "_len", _b.len),
        ("_outer_main", "_str", _b.str),
        ("main", "_expected_capability", _main_capability),
        ("main", "_environ", _os_environ),
        ("main", "_sys", _y),
        ("main", "_consume", _trusted_namespace["_consume_outer_receipt"]),
        ("main", "_internal", _trusted_namespace["_internal_main"]),
        ("main", "_outer", _trusted_namespace["_outer_main"]),
    )
    for _helper_name, _default_name, _expected_default in _default_identities:
        _defaults = _trusted_namespace[_helper_name].__kwdefaults__
        if (type(_defaults) is not dict
                or _defaults.get(_default_name) is not _expected_default):
            raise PermissionError(
                "runner preflight captured default differs: "
                + _helper_name + "." + _default_name
            )
    _expected_read_flags = _o.O_RDONLY | getattr(_o, "O_CLOEXEC", 0) | getattr(_o, "O_NOFOLLOW", 0)
    if (_trusted_namespace["_stable_read"].__kwdefaults__.get("_read_flags")
            != _expected_read_flags
            or _trusted_namespace["_git_environment"].__kwdefaults__.get("_devnull")
            != _o.devnull):
        raise PermissionError("runner preflight captured value default differs")
    _loader_default_identities = (
        ("__init__", "_validate", _trusted_namespace["_validated_oid"]),
        ("_resolve", "_run", _u.run),
        ("_resolve", "_command", _trusted_namespace["_git_command"]),
        ("_resolve", "_environment", _trusted_namespace["_git_environment"]),
        ("_resolve", "_validate", _trusted_namespace["_validated_oid"]),
        ("_resolve", "_verified", _trusted_namespace["_verified_object"]),
        ("find_spec", "_spec_from_loader", _iu.spec_from_loader),
        ("exec_module", "_fspath", _o.fspath),
        ("exec_module", "_compile", _compile_fn),
        ("exec_module", "_exec", _exec_fn),
        ("load_lightweight_qualification", "_parse", _a.parse),
        ("load_lightweight_qualification", "_spec_from_loader", _iu.spec_from_loader),
        ("load_lightweight_qualification", "_module_from_spec", _iu.module_from_spec),
        ("load_lightweight_qualification", "_sys", _y),
        ("load_lightweight_qualification", "_walk", _a.walk),
        ("load_lightweight_qualification", "_dump", _a.dump),
        ("load_lightweight_qualification", "_import_node", _a.Import),
        ("load_lightweight_qualification", "_import_from_node", _a.ImportFrom),
        ("load_lightweight_qualification", "_dict_node", _a.Dict),
        ("load_lightweight_qualification", "_constant_node", _a.Constant),
        ("load_lightweight_qualification", "_sha256", _h.sha256),
        ("load_lightweight_qualification", "_modules", _y.modules),
        ("load_lightweight_qualification", "_resolve_method", _loader_type._resolve),
        (
            "load_lightweight_qualification",
            "_exec_module_method",
            _loader_type.exec_module,
        ),
    )
    for _method_name, _default_name, _expected_default in _loader_default_identities:
        _defaults = _loader_type.__dict__[_method_name].__kwdefaults__
        if (type(_defaults) is not dict
                or _defaults.get(_default_name) is not _expected_default):
            raise PermissionError(
                "runner preflight loader default differs: "
                + _method_name + "." + _default_name
            )
    _security = (
        _seal,
        _trusted_namespace["_capture_outer_receipt"],
        _trusted_namespace["_consume_outer_receipt"],
        _loader_type,
        _trusted_namespace["_canonical_json"],
        _trusted_namespace["arguments"],
        _import_fn,
        _j.dumps,
        _x.suppress,
        _y,
        _trusted_namespace["REPOSITORY_ROOT"],
        _trusted_namespace["CONFIG_PATH"],
        _trusted_namespace["_QUALIFICATION_MODULE"],
        _trusted_namespace["_qualification_callables"],
        _trusted_namespace["_sha256"],
        _b.print,
    )

    def _constant_record(_value):
        if type(_value) is _code_type:
            return {"kind": "code", "value": _code_record(_value)}
        if _value is None or type(_value) in {bool, int, float, str}:
            return {"kind": type(_value).__name__, "value": _value}
        if type(_value) is bytes:
            return {"kind": "bytes", "value": _value.hex()}
        if type(_value) is tuple:
            return {"kind": "tuple", "value": [_constant_record(_item) for _item in _value]}
        if type(_value) is frozenset:
            _items = [_constant_record(_item) for _item in _value]
            _items.sort(key=lambda _item: _j.dumps(_item, sort_keys=True))
            return {"kind": "frozenset", "value": _items}
        return {
            "kind": type(_value).__module__ + "." + type(_value).__qualname__,
            "value": repr(_value),
        }

    def _code_record(_code):
        return {
            "argcount": _code.co_argcount,
            "posonlyargcount": _code.co_posonlyargcount,
            "kwonlyargcount": _code.co_kwonlyargcount,
            "nlocals": _code.co_nlocals,
            "stacksize": _code.co_stacksize,
            "flags": _code.co_flags,
            "code_sha256": _h.sha256(_code.co_code).hexdigest(),
            "constants": [_constant_record(_item) for _item in _code.co_consts],
            "names": list(_code.co_names),
            "varnames": list(_code.co_varnames),
            "freevars": list(_code.co_freevars),
            "cellvars": list(_code.co_cellvars),
            "filename": _code.co_filename,
            "name": _code.co_name,
            "qualname": getattr(_code, "co_qualname", _code.co_name),
            "firstlineno": _code.co_firstlineno,
            "linetable_sha256": _h.sha256(_code.co_linetable).hexdigest(),
            "exceptiontable_sha256": _h.sha256(
                getattr(_code, "co_exceptiontable", b"")
            ).hexdigest(),
        }

    _constant_bindings = {
        _name: (
            _o.fspath(_value)
            if _name in {"REPOSITORY_ROOT", "RUNNER_PATH", "CONFIG_PATH"}
            else _value
        )
        for _name, _value in _expected_constants.items()
    }
    _default_bindings = sorted(
        _helper_name + "." + _default_name
        for _helper_name, _default_name, _ in _default_identities
    )
    _loader_default_bindings = sorted(
        "_ExactCommitLoader." + _method_name + "." + _default_name
        for _method_name, _default_name, _ in _loader_default_identities
    )
    _preflight_record = {
        "schema": "rgbd_known_action_runner_preflight_v1",
        "runner_blob_sha256": _expected,
        "bootstrap_literal_sha256": _bootstrap_literal_sha256,
        "module_bindings": {
            _name: _module.__name__ for _name, _module in _module_identities
        },
        "attribute_bindings": {
            _label: (
                getattr(_expected_attribute, "__module__", type(_expected_attribute).__module__)
                + "."
                + getattr(
                    _expected_attribute,
                    "__qualname__",
                    type(_expected_attribute).__qualname__,
                )
            )
            for _label, _, _, _expected_attribute in _trusted_attributes
        },
        "constant_bindings": _constant_bindings,
        "captured_defaults": _default_bindings,
        "loader_captured_defaults": _loader_default_bindings,
        "stable_read_flags": _expected_read_flags,
        "helper_code": {
            _name: _code_record(_codes[_name])
            for _name in sorted(_function_names)
        },
        "loader_code": {
            _node.name: _code_record(_codes["_ExactCommitLoader." + _node.name])
            for _node in _loader_nodes[0].body
            if type(_node) is _a.FunctionDef
        },
        "interpreter_containers": {
            "modules_object": True,
            "meta_path_object": True,
            "path_object": True,
            "path_hooks_object": True,
            "path_importer_cache_object": True,
            "environ_object": True,
            "module_names_sha256": _h.sha256(
                _j.dumps(
                    [_name for _name, _ in _sys_modules_snapshot],
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest(),
            "meta_path_types": [
                type(_item).__module__ + "." + type(_item).__qualname__
                for _item in _sys_meta_path_snapshot
            ],
            "sys_path": list(_sys_path_snapshot),
            "path_hook_types": [
                type(_item).__module__ + "." + type(_item).__qualname__
                for _item in _sys_path_hooks_snapshot
            ],
            "path_importer_cache_keys_sha256": _h.sha256(
                _j.dumps(
                    sorted(_key for _key, _ in _sys_path_importer_cache_snapshot),
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest(),
        },
    }
    _preflight_bytes = _j.dumps(_preflight_record,allow_nan=False,ensure_ascii=True,separators=(",",":"),sort_keys=True).encode("ascii")
    _environment["runner_preflight_sha256"] = _h.sha256(_preflight_bytes).hexdigest()
    _environment_bytes = _j.dumps(_environment,allow_nan=False,ensure_ascii=True,separators=(",",":"),sort_keys=True).encode("ascii")
    for _private_name in (
        "_BOOTSTRAP_SECURITY_SEAL",
        "_BOOTSTRAP_MAIN_CAPABILITY",
        "_BOOTSTRAP_INTERNAL_CAPABILITY",
    ):
        del _trusted_namespace[_private_name]
    _y.argv = [_path, *_user_argv]
    _trusted_namespace["__name__"] = "__main__"
    _entry = _trusted_namespace["main"]
    _bootstrap_globals = globals()
    del _bootstrap_globals["_rgbd_known_action_bootstrap"]
    return _entry(
        _authorization_fd_text=_authorization_fd_text,
        _bootstrap_environment_hex=_environment_bytes.hex(),
        _bootstrap_security=_main_capability,
    )

raise SystemExit(_rgbd_known_action_bootstrap())
"""

_BOOTSTRAP_SECURITY_SEAL = globals().get("_BOOTSTRAP_SECURITY_SEAL")
_BOOTSTRAP_MAIN_CAPABILITY = globals().get("_BOOTSTRAP_MAIN_CAPABILITY")
_BOOTSTRAP_INTERNAL_CAPABILITY = globals().get("_BOOTSTRAP_INTERNAL_CAPABILITY")


def _canonical_json(value: Any, *, _dumps: Any = json.dumps) -> bytes:
    return _dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _strict_json_loads(
    contents: bytes,
    *,
    label: str,
    _loads: Any = json.loads,
) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains nonfinite constant {value!r}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    value = _loads(
        contents.decode("ascii", errors="strict"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
    if type(value) is not dict:
        raise TypeError(f"{label} root must be one exact object")
    return value


def _sha256(contents: bytes, *, _hash: Any = hashlib.sha256) -> str:
    if type(contents) is not bytes:
        raise TypeError("SHA-256 input must be exact bytes")
    return _hash(contents).hexdigest()


def _validated_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 64 or value != value.lower():
        raise ValueError(f"{label} must be one lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be lowercase hexadecimal") from error
    return value


def _validated_oid(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 2 * _GIT_OID_BYTES or value != value.lower():
        raise ValueError(f"{label} must be one lowercase SHA-1 Git object ID")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    return value


def _git_paths(
    *,
    _is_absolute: Any = Path.is_absolute,
    _resolve: Any = Path.resolve,
    _lstat: Any = Path.lstat,
    _isdir: Any = stat.S_ISDIR,
) -> tuple[Path, Path]:
    if not _is_absolute(REPOSITORY_ROOT) or _resolve(REPOSITORY_ROOT) != REPOSITORY_ROOT:
        raise PermissionError("repository root must be one canonical absolute path")
    git_dir = REPOSITORY_ROOT / ".git"
    metadata = _lstat(git_dir)
    if not _isdir(metadata.st_mode) or _resolve(git_dir) != git_dir:
        raise PermissionError("repository metadata must be the canonical .git directory")
    return git_dir, REPOSITORY_ROOT


def _git_environment(
    *,
    _paths: Any = _git_paths,
    _fspath: Any = os.fspath,
    _environ: Any = os.environ,
    _devnull: str = os.devnull,
) -> dict[str, str]:
    git_dir, worktree = _paths()
    del _environ
    return {
        "GIT_DIR": _fspath(git_dir),
        "GIT_COMMON_DIR": _fspath(git_dir),
        "GIT_WORK_TREE": _fspath(worktree),
        "GIT_INDEX_FILE": _fspath(git_dir / "index"),
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": _devnull,
        "GIT_CONFIG_SYSTEM": _devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_LITERAL_PATHSPECS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }


def _git_command(arguments: list[str], *, _paths: Any = _git_paths) -> list[str]:
    git_dir, worktree = _paths()
    return [
        _TRUSTED_GIT,
        "--no-replace-objects",
        f"--git-dir={git_dir}",
        f"--work-tree={worktree}",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        *arguments,
    ]


def _git_bytes(
    arguments: list[str],
    *,
    label: str,
    _run: Any = subprocess.run,
    _command: Any = _git_command,
    _environment: Any = _git_environment,
) -> bytes:
    completed = _run(
        _command(arguments),
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        check=True,
        capture_output=True,
        text=False,
    )
    if type(completed.stdout) is not bytes or len(completed.stdout) > _MAX_SOURCE_BYTES:
        raise RuntimeError(f"git returned invalid or oversized {label}")
    return completed.stdout


def _git_text(arguments: list[str], *, label: str, _bytes: Any = _git_bytes) -> str:
    value = _bytes(arguments, label=label).decode("utf-8", errors="strict").strip()
    if not value:
        raise RuntimeError(f"git returned empty {label}")
    return value


def _null_terminated_git_fields(raw: bytes, *, label: str) -> list[str]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_SOURCE_BYTES:
        raise PermissionError(f"{label} is empty, invalid, or oversized")
    if not raw.endswith(b"\0") or b"\0\0" in raw:
        raise PermissionError(f"{label} lacks exact NUL framing")
    try:
        fields = raw[:-1].decode("utf-8", errors="strict").split("\0")
    except UnicodeDecodeError as error:
        raise PermissionError(f"{label} is not strict UTF-8") from error
    if any(not field or "\n" in field or "\r" in field for field in fields):
        raise PermissionError(f"{label} contains an invalid field")
    return fields


def _canonical_git_config_pairs(raw: bytes, *, label: str) -> list[dict[str, str]]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_SOURCE_BYTES:
        raise PermissionError(f"{label} is empty, invalid, or oversized")
    if not raw.endswith(b"\0") or b"\0\0" in raw:
        raise PermissionError(f"{label} lacks exact NUL record framing")
    pairs: list[dict[str, str]] = []
    for raw_record in raw[:-1].split(b"\0"):
        raw_key, separator, raw_value = raw_record.partition(b"\n")
        if separator != b"\n" or not raw_key or b"\r" in raw_key or b"\n" in raw_key:
            raise PermissionError(f"{label} contains an invalid key/value record")
        try:
            key = raw_key.decode("utf-8", errors="strict").casefold()
            value = raw_value.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise PermissionError(f"{label} is not strict UTF-8") from error
        if not key or "\0" in key or "\0" in value:
            raise PermissionError(f"{label} contains an invalid key or value")
        pairs.append({"key": key, "value": value})
    return sorted(pairs, key=lambda pair: (pair["key"], pair["value"]))


def _git_config_path_state() -> dict[str, Any]:
    git_dir, _ = _git_paths()
    local_path = git_dir / "config"
    metadata = local_path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or local_path.resolve() != local_path
    ):
        raise PermissionError("local Git config must be one canonical single-link regular file")
    worktree_path = git_dir / "config.worktree"
    try:
        worktree_path.lstat()
    except FileNotFoundError:
        pass
    else:
        raise PermissionError("per-worktree Git config must be exactly absent")
    return {
        "schema": "rgbd_known_action_git_config_paths_v1",
        "local": {
            "path": os.fspath(local_path),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": metadata.st_mode,
            "links": metadata.st_nlink,
            "bytes": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
            "ctime_ns": metadata.st_ctime_ns,
        },
        "worktree": {
            "path": os.fspath(worktree_path),
            "state": "absent",
        },
    }


def _forbidden_git_config_key(key: str) -> bool:
    return (
        key == "extensions.worktreeconfig"
        or (key.startswith("remote.") and key.endswith(".pushurl"))
        or (key.startswith("url.") and key.endswith(".insteadof"))
        or (key.startswith("url.") and key.endswith(".pushinsteadof"))
        or key.startswith("include.")
        or key.startswith("includeif.")
    )


def _approved_git_config_snapshot(*, _bytes: Any = _git_bytes) -> dict[str, Any]:
    path_state = _git_config_path_state()
    local_pairs = _canonical_git_config_pairs(
        _bytes(
            ["config", "--local", "--no-includes", "--null", "--list"],
            label="local Git config pairs",
        ),
        label="local Git config pairs",
    )
    effective_pairs = _canonical_git_config_pairs(
        _bytes(
            ["config", "--no-includes", "--null", "--list"],
            label="effective Git config pairs",
        ),
        label="effective Git config pairs",
    )
    if _git_config_path_state() != path_state:
        raise PermissionError("canonical Git config path changed during snapshot")
    for scope, pairs in (("local", local_pairs), ("effective", effective_pairs)):
        for pair in pairs:
            if _forbidden_git_config_key(pair["key"]):
                raise PermissionError(
                    f"{scope} Git config contains forbidden directive {pair['key']}"
                )
    expected = (
        (f"branch.{_APPROVED_BRANCH}.remote", _APPROVED_REMOTE_NAME),
        (f"branch.{_APPROVED_BRANCH}.merge", _APPROVED_BRANCH_MERGE_REF),
        (f"remote.{_APPROVED_REMOTE_NAME}.url", _APPROVED_REMOTE_URL),
    )
    approved: list[dict[str, str]] = []
    for key, expected_value in expected:
        canonical_key = key.casefold()
        for scope, pairs in (("local", local_pairs), ("effective", effective_pairs)):
            values = [pair["value"] for pair in pairs if pair["key"] == canonical_key]
            if values != [expected_value]:
                raise PermissionError(
                    f"{scope} Git config {canonical_key} is not the approved singleton"
                )
        approved.append({"key": canonical_key, "value": expected_value})
    for pair in local_pairs:
        if effective_pairs.count(pair) < local_pairs.count(pair):
            raise PermissionError("effective Git config omits or overrides a local value")
    return {
        "schema": "rgbd_known_action_git_config_guard_v2",
        "approved": sorted(approved, key=lambda pair: (pair["key"], pair["value"])),
        "config_paths": path_state,
        "local_pairs": local_pairs,
        "effective_pairs": effective_pairs,
    }


def _remote_transport_profile() -> dict[str, Any]:
    environment = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_ALLOW_PROTOCOL": "ssh",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_SSH_COMMAND": _REMOTE_SSH_COMMAND_TEMPLATE,
        "GIT_SSH_VARIANT": "ssh",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    return {
        "schema": "rgbd_known_action_remote_transport_v1",
        "argv": [
            _TRUSTED_GIT,
            "ls-remote",
            "--exit-code",
            "--refs",
            _APPROVED_REMOTE_URL,
            _APPROVED_BRANCH_MERGE_REF,
        ],
        "cwd": _REMOTE_PROBE_CWD,
        "credential_profile": "owned_stable_ssh_agent_socket_v1",
        "environment": environment,
        "pinned_host_key_fingerprint": _PINNED_GITHUB_HOST_KEY_FINGERPRINT,
        "pinned_known_hosts_sha256": _sha256(_PINNED_GITHUB_HOST_KEY.encode("ascii")),
        "timeout_seconds": _REMOTE_PROBE_TIMEOUT_SECONDS,
        "temporary_directory": _REMOTE_TEMPORARY_DIRECTORY,
    }


def _validated_ssh_agent_socket(
    *,
    _environ: Any = os.environ,
    _lstat: Any = os.lstat,
    _is_socket: Any = stat.S_ISSOCK,
    _geteuid: Any = os.geteuid,
) -> tuple[str, tuple[int, int, int, int, int, int]]:
    path = _environ.get("SSH_AUTH_SOCK")
    if (
        type(path) is not str
        or not path
        or not os.path.isabs(path)
        or os.path.normpath(path) != path
        or "\0" in path
    ):
        raise PermissionError("SSH agent socket path is absent or noncanonical")
    metadata = _lstat(path)
    if not _is_socket(metadata.st_mode) or metadata.st_uid != _geteuid() or metadata.st_nlink != 1:
        raise PermissionError("SSH agent authority is not one owned live socket")
    identity = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
    )
    return path, identity


def _remote_transport(
    command: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    timeout: int,
    known_hosts: bytes,
    _run: Any = subprocess.run,
    _agent: Any = _validated_ssh_agent_socket,
) -> subprocess.CompletedProcess[bytes]:
    if known_hosts != _PINNED_GITHUB_HOST_KEY.encode("ascii"):
        raise PermissionError("remote transport known-host material differs")
    if type(env) is not dict or env != _remote_transport_profile()["environment"]:
        raise PermissionError("remote transport environment differs from the fixed profile")
    agent_path, agent_identity = _agent()
    with tempfile.TemporaryDirectory(
        prefix="rgbd-known-action-host-",
        dir=_REMOTE_TEMPORARY_DIRECTORY,
    ) as temporary:
        known_hosts_path = Path(temporary) / "known_hosts"
        descriptor = os.open(
            known_hosts_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            written = 0
            while written < len(known_hosts):
                written += os.write(descriptor, known_hosts[written:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        exact_environment = dict(env)
        if "SSH_AUTH_SOCK" in exact_environment:
            raise PermissionError("remote transport environment preloads agent authority")
        exact_environment["SSH_AUTH_SOCK"] = agent_path
        template = exact_environment.get("GIT_SSH_COMMAND")
        if template != _REMOTE_SSH_COMMAND_TEMPLATE or template.count("{known_hosts_file}") != 1:
            raise PermissionError("remote transport SSH command template differs")
        exact_environment["GIT_SSH_COMMAND"] = template.replace(
            "{known_hosts_file}", os.fspath(known_hosts_path)
        )
        completed = _run(
            list(command),
            cwd=cwd,
            env=exact_environment,
            check=False,
            capture_output=True,
            text=False,
            timeout=timeout,
        )
        final_agent_path, final_agent_identity = _agent()
        if final_agent_path != agent_path or final_agent_identity != agent_identity:
            raise PermissionError("SSH agent socket changed during remote publication probe")
        return completed


def _probe_remote_publication(commit: str) -> dict[str, Any]:
    commit = _validated_oid(commit, label="remote publication commit")
    profile = _remote_transport_profile()
    try:
        completed = _remote_transport(
            profile["argv"],
            cwd=profile["cwd"],
            env=profile["environment"],
            timeout=profile["timeout_seconds"],
            known_hosts=_PINNED_GITHUB_HOST_KEY.encode("ascii"),
        )
    except subprocess.TimeoutExpired as error:
        raise PermissionError("approved remote publication probe timed out") from error
    if (
        type(completed.returncode) is not int
        or completed.returncode != 0
        or type(completed.stdout) is not bytes
        or type(completed.stderr) is not bytes
        or len(completed.stdout) > _REMOTE_PROBE_MAX_OUTPUT_BYTES
        or len(completed.stderr) > _REMOTE_PROBE_MAX_OUTPUT_BYTES
    ):
        raise PermissionError("approved remote publication probe failed or was oversized")
    expected = f"{commit}\t{_APPROVED_BRANCH_MERGE_REF}\n".encode("ascii")
    if completed.stdout != expected:
        raise PermissionError("approved remote advertised ref differs from exact HEAD")
    return {
        "schema": "rgbd_known_action_remote_publication_v1",
        "git_executable": _TRUSTED_GIT,
        "literal_url": _APPROVED_REMOTE_URL,
        "literal_ref": _APPROVED_BRANCH_MERGE_REF,
        "advertised_commit": commit,
        "advertisement_sha256": _sha256(completed.stdout),
        "transport_profile": profile,
    }


def _object_oid(kind: str, contents: bytes, *, _hash: Any = hashlib.sha1) -> str:
    if kind not in {"blob", "commit", "tree"} or type(contents) is not bytes:
        raise TypeError("Git framing requires a supported object type and exact bytes")
    return _hash(f"{kind} {len(contents)}\0".encode("ascii") + contents).hexdigest()


def _verified_object(
    oid: str,
    *,
    kind: str,
    label: str,
    _validate: Any = _validated_oid,
    _text: Any = _git_text,
    _bytes: Any = _git_bytes,
    _object: Any = _object_oid,
) -> bytes:
    oid = _validate(oid, label=f"{label} OID")
    reported_type = _text(["cat-file", "-t", oid], label=f"{label} type")
    size_text = _text(["cat-file", "-s", oid], label=f"{label} size")
    if not size_text.isascii() or not size_text.isdecimal() or str(int(size_text)) != size_text:
        raise PermissionError(f"{label} Git size is noncanonical")
    contents = _bytes(["cat-file", kind, oid], label=f"{label} contents")
    if reported_type != kind or int(size_text) != len(contents) or _object(kind, contents) != oid:
        raise PermissionError(f"{label} Git object framing differs")
    return contents


def _stable_read(
    path: Path,
    *,
    label: str,
    _lstat: Any = Path.lstat,
    _isreg: Any = stat.S_ISREG,
    _open: Any = os.open,
    _fstat: Any = os.fstat,
    _read: Any = os.read,
    _close: Any = os.close,
    _read_flags: int = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
) -> bytes:
    before = _lstat(path)
    if not _isreg(before.st_mode) or before.st_nlink != 1 or before.st_size > _MAX_SOURCE_BYTES:
        raise PermissionError(f"{label} must be one bounded single-link regular file")
    descriptor = _open(path, _read_flags)
    try:
        opened = _fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := _read(descriptor, 65_536):
            chunks.append(chunk)
        after_open = _fstat(descriptor)
    finally:
        _close(descriptor)
    after = _lstat(path)
    identity = lambda item: (  # noqa: E731
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if any(identity(item) != identity(before) for item in (opened, after_open, after)):
        raise PermissionError(f"{label} changed during stable read")
    contents = b"".join(chunks)
    if len(contents) != before.st_size:
        raise PermissionError(f"{label} length changed")
    return contents


def _blob_binding(
    commit: str,
    relative: str,
    *,
    label: str,
    _validate: Any = _validated_oid,
    _text: Any = _git_text,
    _verified: Any = _verified_object,
    _bytes: Any = _git_bytes,
    _read: Any = _stable_read,
    _hash: Any = _sha256,
) -> tuple[dict[str, Any], bytes]:
    oid = _validate(
        _text(["rev-parse", f"{commit}:{relative}"], label=f"{label} OID"),
        label=f"{label} OID",
    )
    blob = _verified(oid, kind="blob", label=label)
    tree_line = _bytes(["ls-tree", "-z", commit, "--", relative], label=f"{label} tree")
    suffix = b"\t" + relative.encode("utf-8") + b"\0"
    if tree_line.count(b"\0") != 1 or not tree_line.endswith(suffix):
        raise PermissionError(f"{label} Git tree framing differs")
    prefix = tree_line[: -len(suffix)].decode("ascii", errors="strict").split(" ")
    if prefix != ["100644", "blob", oid]:
        raise PermissionError(f"{label} is not one regular non-executable blob")
    first = _read(REPOSITORY_ROOT / relative, label=f"{label} worktree")
    second = _read(REPOSITORY_ROOT / relative, label=f"{label} worktree")
    if first != second or first != blob:
        raise PermissionError(f"{label} worktree differs from the exact commit blob")
    return (
        {
            "path": relative,
            "mode": "100644",
            "blob_oid": oid,
            "blob_sha256": _hash(blob),
            "worktree_sha256": _hash(first),
            "bytes": len(blob),
        },
        blob,
    )


def _capture_outer_receipt(
    argv: list[str],
    *,
    _bytes: Any = _git_bytes,
    _text: Any = _git_text,
    _validate: Any = _validated_oid,
    _verified: Any = _verified_object,
    _blob: Any = _blob_binding,
    _config: Any = _approved_git_config_snapshot,
    _probe: Any = _probe_remote_publication,
    _fspath: Any = os.fspath,
    _token_hex: Any = secrets.token_hex,
    _getpid: Any = os.getpid,
    _hash: Any = _sha256,
    _canonical: Any = _canonical_json,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    if _bytes(
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        label="worktree status",
    ):
        raise PermissionError("formal runner requires a clean worktree including untracked files")
    if _text(["rev-parse", "--show-object-format"], label="object format") != "sha1":
        raise PermissionError("formal runner requires the pinned SHA-1 Git object format")
    commit = _validate(_text(["rev-parse", "HEAD"], label="HEAD"), label="HEAD")
    branch = _text(
        ["symbolic-ref", "--quiet", "--short", "HEAD"],
        label="branch",
    )
    config_snapshot = _config()
    remote_name = _APPROVED_REMOTE_NAME
    branch_merge_ref = _APPROVED_BRANCH_MERGE_REF
    remote_url = _APPROVED_REMOTE_URL
    upstream_ref = _text(
        ["rev-parse", "--symbolic-full-name", "@{upstream}"],
        label="upstream ref",
    )
    if (
        branch != _APPROVED_BRANCH
        or remote_name != _APPROVED_REMOTE_NAME
        or remote_url != _APPROVED_REMOTE_URL
        or branch_merge_ref != _APPROVED_BRANCH_MERGE_REF
        or upstream_ref != _APPROVED_UPSTREAM_REF
    ):
        raise PermissionError("formal runner branch/remote/upstream configuration differs")
    upstream = _validate(
        _text(["rev-parse", "@{upstream}"], label="upstream"),
        label="upstream",
    )
    divergence = _text(
        ["rev-list", "--left-right", "--count", "HEAD...@{upstream}"],
        label="upstream divergence",
    ).split()
    if commit != upstream or divergence != ["0", "0"]:
        raise PermissionError("formal runner source must equal its upstream exactly")
    commit_bytes = _verified(commit, kind="commit", label="HEAD commit")
    first_line = commit_bytes.split(b"\n", 1)[0]
    if not first_line.startswith(b"tree "):
        raise PermissionError("HEAD commit lacks an exact leading tree")
    tree = _validate(first_line[5:].decode("ascii"), label="HEAD tree")
    _verified(tree, kind="tree", label="HEAD tree")
    blobs: dict[str, dict[str, Any]] = {}
    runner_blob = b""
    for name, relative in PUBLICATION_SURFACE_PATHS.items():
        binding, blob = _blob(commit, relative, label=f"publication {name}")
        blobs[name] = binding
        if name == "runner":
            runner_blob = blob
    remote_publication = _probe(commit)
    post_remote_config_snapshot = _config()
    if post_remote_config_snapshot != config_snapshot:
        raise PermissionError("formal runner source changed during exact Git capture")
    remote_publication["config_guard"] = config_snapshot
    remote_publication["config_guard_sha256"] = _hash(_canonical(config_snapshot))
    final_status = _bytes(
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        label="final worktree status",
    )
    final_commit = _validate(
        _text(["rev-parse", "HEAD"], label="final HEAD"),
        label="final HEAD",
    )
    final_upstream = _validate(
        _text(["rev-parse", "@{upstream}"], label="final upstream"),
        label="final upstream",
    )
    final_branch = _text(
        ["symbolic-ref", "--quiet", "--short", "HEAD"],
        label="final branch",
    )
    final_config_snapshot = _config()
    final_upstream_ref = _text(
        ["rev-parse", "--symbolic-full-name", "@{upstream}"],
        label="final upstream ref",
    )
    final_divergence = _text(
        ["rev-list", "--left-right", "--count", "HEAD...@{upstream}"],
        label="final upstream divergence",
    ).split()
    if (
        final_status
        or final_commit != commit
        or final_upstream != upstream
        or final_branch != branch
        or final_config_snapshot != config_snapshot
        or final_upstream_ref != upstream_ref
        or final_divergence != ["0", "0"]
    ):
        raise PermissionError("formal runner source changed during exact Git capture")
    source = {
        "schema": "rgbd_known_action_published_source_v2",
        "repository_root": _fspath(REPOSITORY_ROOT),
        "branch": branch,
        "remote_name": remote_name,
        "remote_url": remote_url,
        "branch_merge_ref": branch_merge_ref,
        "commit": commit,
        "tree": tree,
        "upstream_ref": upstream_ref,
        "upstream_commit": upstream,
        "ahead": 0,
        "behind": 0,
        "dirty": False,
        "object_format": "sha1",
        "remote_publication": remote_publication,
        "publication_surface_blobs": blobs,
        "publication_surface_sha256": {
            name: binding["blob_sha256"] for name, binding in blobs.items()
        },
    }
    receipt_core = {
        "schema": _RECEIPT_SCHEMA,
        "nonce": _token_hex(32),
        "parent_pid": _getpid(),
        "argv": list(argv),
        "source_provenance": source,
    }
    first_receipt_sha256 = _hash(_canonical(receipt_core))
    phase_values = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "--phase"]
    phase = phase_values[0] if len(phase_values) == 1 else "protocol"
    authorization_body = {
        "schema": _AUTHORIZATION_SCHEMA,
        "nonce": _token_hex(32),
        "first_receipt_sha256": first_receipt_sha256,
        "runner_blob_sha256": _hash(runner_blob),
        "stage": phase,
        "argv": list(argv),
        "source_provenance_sha256": _hash(_canonical(source)),
        "outer_pid": _getpid(),
        "expected_child_parent_pid": _getpid(),
    }
    authorization_record = {
        **authorization_body,
        "record_sha256": _hash(_canonical(authorization_body)),
    }
    authorization_record_sha256 = _hash(_canonical(authorization_record))
    receipt = {
        **receipt_core,
        "receipt_sha256": first_receipt_sha256,
        "authorization_record_sha256": authorization_record_sha256,
        "authorization_nonce": authorization_record["nonce"],
    }
    return receipt, authorization_record, runner_blob


def _consume_outer_receipt(
    argv: list[str],
    *,
    _environ: Any = os.environ,
    _fstat: Any = os.fstat,
    _isfifo: Any = stat.S_ISFIFO,
    _set_inheritable: Any = os.set_inheritable,
    _read: Any = os.read,
    _close: Any = os.close,
    _getppid: Any = os.getppid,
    _hash: Any = _sha256,
    _validate_hash: Any = _validated_sha256,
    _canonical: Any = _canonical_json,
    _loads: Any = _strict_json_loads,
) -> dict[str, Any]:
    values = {
        "fd": _environ.pop(_RECEIPT_FD_ENV, None),
        "sha256": _environ.pop(_RECEIPT_SHA_ENV, None),
        "nonce": _environ.pop(_RECEIPT_NONCE_ENV, None),
    }
    if any(key.startswith(_RECEIPT_ENV_PREFIX) for key in _environ):
        raise PermissionError("unexpected outer receipt environment authority exists")
    fd_text = values["fd"]
    if type(fd_text) is not str or not fd_text.isascii() or not fd_text.isdecimal():
        raise PermissionError("outer receipt descriptor is malformed")
    descriptor = int(fd_text)
    if descriptor < 3 or str(descriptor) != fd_text:
        raise PermissionError("outer receipt descriptor is noncanonical")
    try:
        metadata = _fstat(descriptor)
        if not _isfifo(metadata.st_mode) or metadata.st_nlink != 0:
            raise PermissionError("outer receipt authority is not a one-shot pipe")
        _set_inheritable(descriptor, False)
        chunks: list[bytes] = []
        total = 0
        while chunk := _read(descriptor, 4096):
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_RECEIPT_BYTES:
                raise PermissionError("outer receipt is oversized")
        after = _fstat(descriptor)
    finally:
        _close(descriptor)
    identity = lambda item: (  # noqa: E731
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
    )
    if identity(metadata) != identity(after):
        raise PermissionError("outer receipt pipe identity changed during consumption")
    contents = b"".join(chunks)
    if _hash(contents) != _validate_hash(values["sha256"], label="receipt digest"):
        raise PermissionError("outer receipt digest differs")
    receipt = _loads(contents, label="outer receipt")
    if type(receipt) is not dict or set(receipt) != {
        "schema",
        "nonce",
        "parent_pid",
        "argv",
        "source_provenance",
        "receipt_sha256",
        "authorization_record_sha256",
        "authorization_nonce",
    }:
        raise PermissionError("outer receipt schema differs")
    core = {
        name: receipt[name]
        for name in ("schema", "nonce", "parent_pid", "argv", "source_provenance")
    }
    supplied = receipt["receipt_sha256"]
    if (
        core["schema"] != _RECEIPT_SCHEMA
        or type(core["nonce"]) is not str
        or core["nonce"] != values["nonce"]
        or type(core["parent_pid"]) is not int
        or core["parent_pid"] != _getppid()
        or type(core["argv"]) is not list
        or any(type(value) is not str for value in core["argv"])
        or core["argv"] != argv
        or type(supplied) is not str
        or supplied != _hash(_canonical(core))
        or _validate_hash(
            receipt["authorization_record_sha256"],
            label="authorization record digest",
        )
        != receipt["authorization_record_sha256"]
        or type(receipt["authorization_nonce"]) is not str
        or len(receipt["authorization_nonce"]) != 64
        or receipt["authorization_nonce"] != receipt["authorization_nonce"].lower()
    ):
        raise PermissionError("outer receipt identity or self-hash differs")
    try:
        int(receipt["authorization_nonce"], 16)
    except ValueError as error:
        raise PermissionError("outer receipt authorization nonce is malformed") from error
    return receipt


class _ExactCommitLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Compile every ``world_model`` module from the receipt-bound commit."""

    def __init__(self, receipt: dict[str, Any], *, _validate: Any = _validated_oid) -> None:
        self._commit = _validate(
            receipt["source_provenance"]["commit"],
            label="loader commit",
        )
        self._cache: dict[str, tuple[bytes, str, bool]] = {}

    def _resolve(
        self,
        fullname: str,
        *,
        _run: Any = subprocess.run,
        _command: Any = _git_command,
        _environment: Any = _git_environment,
        _validate: Any = _validated_oid,
        _verified: Any = _verified_object,
    ) -> tuple[bytes, str, bool] | None:
        if fullname in self._cache:
            return self._cache[fullname]
        stem = fullname.replace(".", "/")
        for relative, is_package in ((f"{stem}/__init__.py", True), (f"{stem}.py", False)):
            completed = _run(
                _command(["rev-parse", f"{self._commit}:{relative}"]),
                cwd=REPOSITORY_ROOT,
                env=_environment(),
                check=False,
                capture_output=True,
                text=False,
            )
            if completed.returncode != 0:
                continue
            oid = _validate(
                completed.stdout.decode("ascii").strip(),
                label=f"loader {fullname}",
            )
            source = _verified(oid, kind="blob", label=f"loader {fullname}")
            resolved = (source, relative, is_package)
            self._cache[fullname] = resolved
            return resolved
        return None

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
        *,
        _spec_from_loader: Any = importlib.util.spec_from_loader,
    ) -> importlib.machinery.ModuleSpec | None:
        del path, target
        if fullname != "world_model" and not fullname.startswith("world_model."):
            return None
        resolved = self._resolve(fullname)
        if resolved is None:
            raise ImportError(f"receipt-bound commit lacks project module {fullname}")
        _, relative, is_package = resolved
        return _spec_from_loader(
            fullname,
            self,
            origin=f"git:{self._commit}:{relative}",
            is_package=is_package,
        )

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        del spec
        return None

    def exec_module(
        self,
        module: ModuleType,
        *,
        _fspath: Any = os.fspath,
        _compile: Any = builtins.compile,
        _exec: Any = builtins.exec,
    ) -> None:
        resolved = self._resolve(module.__name__)
        if resolved is None:
            raise ImportError(f"exact commit lacks module {module.__name__}")
        source, relative, is_package = resolved
        origin = f"git:{self._commit}:{relative}"
        module.__file__ = _fspath(REPOSITORY_ROOT / relative)
        module.__cached__ = None
        if is_package:
            module.__path__ = [_fspath((REPOSITORY_ROOT / relative).parent)]
        _exec(
            _compile(source, origin, "exec", dont_inherit=True),
            module.__dict__,
            module.__dict__,
        )

    def load_lightweight_qualification(
        self,
        fullname: str,
        *,
        _parse: Any = ast.parse,
        _spec_from_loader: Any = importlib.util.spec_from_loader,
        _module_from_spec: Any = importlib.util.module_from_spec,
        _sys: Any = sys,
        _walk: Any = ast.walk,
        _dump: Any = ast.dump,
        _import_node: Any = ast.Import,
        _import_from_node: Any = ast.ImportFrom,
        _dict_node: Any = ast.Dict,
        _constant_node: Any = ast.Constant,
        _sha256: Any = hashlib.sha256,
        _modules: Any = sys.modules,
        _resolve_method: Any = _resolve,
        _exec_module_method: Any = exec_module,
    ) -> ModuleType:
        """Load the exact qualification leaf without importing package parents."""

        if (
            _sys.modules is not _modules
            or fullname != _QUALIFICATION_MODULE
            or fullname in _modules
        ):
            raise PermissionError("lightweight qualification module identity differs")
        if any(
            name == "torch"
            or name.startswith("torch.")
            or name == "world_model"
            or name.startswith("world_model.")
            for name in _modules
        ):
            raise PermissionError("heavy modules preceded lightweight recovery loading")
        original_heavy_modules = {
            name: value
            for name, value in _modules.items()
            if name == "torch"
            or name.startswith("torch.")
            or name == "world_model"
            or name.startswith("world_model.")
        }
        try:
            resolved = _resolve_method(self, fullname)
            if resolved is None:
                raise ImportError("receipt-bound commit lacks the qualification module")
            source, relative, is_package = resolved
            if relative != "world_model/training/rgbd_known_action_qualification.py" or is_package:
                raise PermissionError("lightweight qualification source binding differs")
            origin = f"git:{self._commit}:{relative}"
            tree = _parse(source, filename=origin, mode="exec")
            all_nodes = tuple(_walk(tree))
            for node in all_nodes:
                if type(node) is _import_node:
                    imported_names = tuple(alias.name for alias in node.names)
                elif type(node) is _import_from_node:
                    imported_names = () if node.module is None else (node.module,)
                else:
                    continue
                if any(
                    name.split(".", 1)[0] in {"torch", "world_model"} for name in imported_names
                ):
                    raise PermissionError(
                        "qualification recursively imports a heavy project dependency"
                    )
            observed_imports: list[tuple[Any, ...]] = []
            for node in tree.body:
                if type(node) is _import_node:
                    observed_imports.extend(
                        ("import", alias.name, alias.asname) for alias in node.names
                    )
                elif type(node) is _import_from_node:
                    observed_imports.append(
                        (
                            "from",
                            node.module,
                            node.level,
                            tuple((alias.name, alias.asname) for alias in node.names),
                        )
                    )
            expected_imports = [
                ("from", "__future__", 0, (("annotations", None),)),
                *[
                    ("import", name, None)
                    for name in (
                        "argparse",
                        "ast",
                        "builtins",
                        "contextlib",
                        "copy",
                        "hashlib",
                        "importlib",
                        "importlib.abc",
                        "importlib.machinery",
                        "importlib.util",
                        "io",
                        "json",
                        "math",
                        "os",
                        "pathlib",
                        "resource",
                        "secrets",
                        "stat",
                        "subprocess",
                        "sys",
                        "sysconfig",
                        "tempfile",
                        "threading",
                        "time",
                        "types",
                    )
                ],
                (
                    "from",
                    "collections.abc",
                    0,
                    (("Callable", None), ("Mapping", None), ("Sequence", None)),
                ),
                (
                    "from",
                    "dataclasses",
                    0,
                    (("asdict", None), ("dataclass", None), ("replace", None)),
                ),
                ("from", "numbers", 0, (("Real", None),)),
                ("from", "pathlib", 0, (("Path", None),)),
                ("from", "statistics", 0, (("median", None),)),
                (
                    "from",
                    "typing",
                    0,
                    (("Any", None), ("Literal", None), ("Protocol", None)),
                ),
            ]
            if observed_imports != expected_imports:
                raise PermissionError("qualification top-level import surface is not lightweight")
            fingerprint_values: list[Any] = []
            for node in all_nodes:
                if type(node) is not _dict_node:
                    continue
                for key, value in zip(node.keys, node.values, strict=True):
                    if (
                        type(key) is _constant_node
                        and key.value == "_LIGHTWEIGHT_QUALIFICATION_EXECUTION_SHA256"
                    ):
                        fingerprint_values.append(value)
            if (
                len(fingerprint_values) != 1
                or type(fingerprint_values[0]) is not _constant_node
                or type(fingerprint_values[0].value) is not str
                or len(fingerprint_values[0].value) != 64
                or fingerprint_values[0].value.lower() != fingerprint_values[0].value
            ):
                raise PermissionError("qualification fingerprint sentinel binding differs")
            try:
                int(fingerprint_values[0].value, 16)
            except ValueError as error:
                raise PermissionError("qualification fingerprint sentinel is malformed") from error
            fingerprint_values[0].value = "0" * 64
            execution_surface = _dump(
                tree,
                annotate_fields=True,
                include_attributes=False,
            ).encode("utf-8")
            if (
                _sha256(execution_surface).hexdigest()
                != _LIGHTWEIGHT_QUALIFICATION_EXECUTION_SHA256
            ):
                raise PermissionError("qualification module-execution surface fingerprint differs")
            spec = _spec_from_loader(
                fullname,
                self,
                origin=origin,
                is_package=False,
            )
            if spec is None:
                raise ImportError("lightweight qualification spec is absent")
            module = _module_from_spec(spec)
            _modules[fullname] = module
            _exec_module_method(self, module)
            if (
                module.__dict__.get("torch") is not None
                or module.__dict__.get("OrpheusConfig") is not None
                or module.__dict__.get("load_config") is not None
                or module.__dict__.get("SIMULATOR_VERSION") != "sphere_world_v7"
                or module.__dict__.get("SPECIFICATION_VERSION") != "1.60"
                or module.__dict__.get("__version__") != "0.1.0"
                or module.__dict__.get("_RUNTIME_DEPENDENCIES_ACTIVE") is not False
                or {
                    name
                    for name in _modules
                    if name == "torch"
                    or name.startswith("torch.")
                    or name == "world_model"
                    or name.startswith("world_model.")
                }
                != {fullname}
            ):
                raise PermissionError("lightweight qualification imported a heavy dependency")
            return module
        except BaseException:
            for name in tuple(_modules):
                if (
                    name == "torch"
                    or name.startswith("torch.")
                    or name == "world_model"
                    or name.startswith("world_model.")
                ):
                    _modules.pop(name, None)
            _modules.update(original_heavy_modules)
            raise


def arguments(
    argv: list[str] | None = None,
    *,
    _parser: Any = argparse.ArgumentParser,
    _validate_hash: Any = _validated_sha256,
) -> argparse.Namespace:
    parser = _parser(description="Known-action RGB-D qualification attempt one")
    parser.add_argument(
        "--phase",
        choices=("protocol", "development", "qualification"),
        default="protocol",
    )
    parser.add_argument("--reviewed-checkpoint-sha256")
    parser.add_argument("--reviewed-report-sha256")
    parser.add_argument("--reviewed-development-ledger-sha256")
    parsed = parser.parse_args(argv)
    reviewed = (
        parsed.reviewed_checkpoint_sha256,
        parsed.reviewed_report_sha256,
        parsed.reviewed_development_ledger_sha256,
    )
    if parsed.phase == "qualification":
        if any(value is None for value in reviewed):
            parser.error("qualification requires exactly three externally reviewed hashes")
        for label, value in zip(("checkpoint", "report", "ledger"), reviewed, strict=True):
            try:
                _validate_hash(value, label=f"reviewed {label}")
            except ValueError as error:
                parser.error(str(error))
    elif any(value is not None for value in reviewed):
        parser.error("reviewed hashes are permitted only for qualification")
    return parsed


def _qualification_callables(
    qualification: ModuleType,
    loader: _ExactCommitLoader,
    *,
    _compile: Any = builtins.compile,
    _fspath: Any = os.fspath,
    _getattr: Any = builtins.getattr,
    _type: Any = builtins.type,
    _len: Any = builtins.len,
    _callable: Any = builtins.callable,
    _module_type: Any = ModuleType,
    _function_type: Any = FunctionType,
    _sys: Any = sys,
    _path_type: Any = Path,
    _qualification_name: str = _QUALIFICATION_MODULE,
    _repository_root: Path = REPOSITORY_ROOT,
    _ast_parse: Any = ast.parse,
    _ast_module: Any = ast,
) -> tuple[Any, ...]:
    relative = "world_model/training/rgbd_known_action_qualification.py"
    resolved = loader._cache.get(_qualification_name)
    spec = _getattr(qualification, "__spec__", None)
    origin = f"git:{loader._commit}:{relative}"
    if (
        _type(resolved) is not tuple
        or _len(resolved) != 3
        or _type(resolved[0]) is not bytes
        or resolved[1:] != (relative, False)
        or _type(qualification) is not _module_type
        or qualification.__name__ != _qualification_name
        or qualification.__package__ != "world_model.training"
        or qualification.__file__ != _fspath(REPOSITORY_ROOT / relative)
        or qualification.__cached__ is not None
        or qualification.__loader__ is not loader
        or spec is None
        or spec.loader is not loader
        or spec.name != _QUALIFICATION_MODULE
        or spec.origin != origin
        or spec.cached is not None
        or spec.submodule_search_locations is not None
    ):
        raise PermissionError("qualification module escaped exact preflight loading")
    compiled = _compile(resolved[0], origin, "exec", dont_inherit=True)
    code_type = _type(compiled)
    tree = _ast_parse(resolved[0], filename=origin, mode="exec")
    function_names = {node.name for node in tree.body if _type(node) is _ast_module.FunctionDef}
    expected_codes = {
        value.co_name: value
        for value in compiled.co_consts
        if _type(value) is code_type and value.co_name in function_names
    }
    names = (
        "capture_published_source",
        "_register_outer_runner_authority",
        "require_frozen_config",
        "bridge_protocol",
        "canonical_development_report_path",
        "canonical_checkpoint_path",
        "canonical_qualification_report_path",
        "run_development",
        "run_qualification",
        "_revoke_outer_runner_authority",
    )
    pending = [*names, "_pinned_durable_replace"]
    validated: dict[str, Any] = {}
    while pending:
        name = pending.pop()
        if name in validated:
            continue
        helper = qualification.__dict__.get(name)
        expected_code = expected_codes.get(name)
        if (
            _type(helper) is not _function_type
            or expected_code is None
            or _getattr(helper, "__globals__", None) is not qualification.__dict__
            or _getattr(helper, "__code__", None) != expected_code
            or _getattr(helper, "__module__", None) != _qualification_name
            or _getattr(helper, "__closure__", None) is not None
            or _getattr(helper, "__dict__", None) != {}
        ):
            raise PermissionError(f"qualification callable {name} differs before first use")
        validated[name] = helper
        captured_values = [
            *(helper.__defaults__ or ()),
            *((helper.__kwdefaults__ or {}).values()),
        ]
        for captured in captured_values:
            if (
                _type(captured) is _function_type
                and _getattr(captured, "__globals__", None) is qualification.__dict__
            ):
                captured_name = _getattr(captured, "__name__", None)
                if (
                    _type(captured_name) is not str
                    or captured_name not in expected_codes
                    or qualification.__dict__.get(captured_name) is not captured
                ):
                    raise PermissionError(
                        f"qualification callable {name} captured an untrusted helper"
                    )
                if captured_name not in validated:
                    pending.append(captured_name)
        code_pending = [expected_code]
        referenced_names: set[str] = set()
        while code_pending:
            code = code_pending.pop()
            referenced_names.update(code.co_names)
            code_pending.extend(value for value in code.co_consts if _type(value) is code_type)
        pending.extend(
            referenced_name
            for referenced_name in referenced_names
            if referenced_name in expected_codes and referenced_name not in validated
        )
    module_bindings = {
        "argparse": "argparse",
        "ast": "ast",
        "builtins": "builtins",
        "contextlib": "contextlib",
        "copy": "copy",
        "hashlib": "hashlib",
        "importlib": "importlib",
        "io": "io",
        "json": "json",
        "math": "math",
        "os": "os",
        "pathlib": "pathlib",
        "resource": "resource",
        "secrets": "secrets",
        "stat": "stat",
        "subprocess": "subprocess",
        "sys": "sys",
        "sysconfig": "sysconfig",
        "tempfile": "tempfile",
        "threading": "threading",
        "time": "time",
        "types": "types",
    }
    if any(
        qualification.__dict__.get(bound_name) is not _sys.modules.get(module_name)
        for bound_name, module_name in module_bindings.items()
    ):
        raise PermissionError("qualification imported module binding differs before first use")
    heavy_names = {
        name
        for name in _sys.modules
        if name == "torch"
        or name.startswith("torch.")
        or name == "world_model"
        or name.startswith("world_model.")
    }
    if (
        heavy_names != {_qualification_name}
        or qualification.__dict__.get("torch") is not None
        or qualification.__dict__.get("OrpheusConfig") is not None
        or qualification.__dict__.get("load_config") is not None
        or qualification.__dict__.get("_RUNTIME_DEPENDENCIES_ACTIVE") is not False
    ):
        raise PermissionError("qualification recovery dependency boundary differs")
    expected_constants = {
        "REPOSITORY_ROOT": _repository_root,
        "PUBLICATION_SURFACE_PATHS": {
            "qualification": "world_model/training/rgbd_known_action_qualification.py",
            "runner": "scripts/run_rgbd_known_action_qualification.py",
            "qualification_test": "tests/unit/test_rgbd_known_action_qualification.py",
        },
        "_TRUSTED_GIT": "/usr/bin/git",
        "_GIT_OID_BYTES": 20,
        "_GIT_MAX_OUTPUT_BYTES": 8 * 1024 * 1024,
        "_GIT_OBJECT_FORMAT": "sha1",
        "_APPROVED_BRANCH": "agent/rgbd-known-action-planning-rung-1",
        "_APPROVED_REMOTE_NAME": "origin",
        "_APPROVED_REMOTE_URL": "git@github.com:polceanum/world.model.git",
        "_APPROVED_BRANCH_MERGE_REF": "refs/heads/agent/rgbd-known-action-planning-rung-1",
        "_APPROVED_UPSTREAM_REF": "refs/remotes/origin/agent/rgbd-known-action-planning-rung-1",
        "_REMOTE_PROBE_CWD": "/",
        "_REMOTE_TEMPORARY_DIRECTORY": "/private/tmp",
        "_REMOTE_PROBE_TIMEOUT_SECONDS": 20,
        "_REMOTE_PROBE_MAX_OUTPUT_BYTES": 4096,
        "_PINNED_GITHUB_HOST_KEY": "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl\n",
        "_PINNED_GITHUB_HOST_KEY_FINGERPRINT": "SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU",
        "_REMOTE_SSH_COMMAND_TEMPLATE": "/usr/bin/ssh -F /dev/null -oBatchMode=yes -oClearAllForwardings=yes -oForwardAgent=no -oForwardX11=no -oProxyCommand=none -oProxyJump=none -oCanonicalizeHostname=no -oStrictHostKeyChecking=yes -oCheckHostIP=yes -oPasswordAuthentication=no -oKbdInteractiveAuthentication=no -oIdentityFile=/dev/null -oIdentitiesOnly=no -oIdentityAgent=SSH_AUTH_SOCK -oAddKeysToAgent=no -oPKCS11Provider=none -oSecurityKeyProvider=none -oGSSAPIAuthentication=no -oHostbasedAuthentication=no -oPubkeyAuthentication=yes -oHostKeyAlgorithms=ssh-ed25519 -oHostKeyAlias=github.com -oUserKnownHostsFile={known_hosts_file} -oGlobalKnownHostsFile=/dev/null",
        "_MAX_OUTER_AUTHORIZATION_BYTES": 16 * 1024,
        "MAX_JSON_BYTES": 8 * 1024 * 1024,
        "__version__": "0.1.0",
        "SPECIFICATION_VERSION": "1.60",
        "SIMULATOR_VERSION": "sphere_world_v7",
    }
    if any(
        _type(qualification.__dict__.get(name)) is not _type(expected)
        or qualification.__dict__.get(name) != expected
        for name, expected in expected_constants.items()
    ):
        raise PermissionError("qualification security constant differs before first use")
    default_checks = (
        ("_git_text", "allow_empty", False),
        ("_validated_ssh_agent_socket", "_environ", qualification.os.environ),
        ("_validated_ssh_agent_socket", "_lstat", qualification.os.lstat),
        ("_validated_ssh_agent_socket", "_is_socket", qualification.stat.S_ISSOCK),
        ("_validated_ssh_agent_socket", "_geteuid", qualification.os.geteuid),
        ("_remote_transport", "_run", qualification.subprocess.run),
        (
            "_remote_transport",
            "_agent",
            qualification._validated_ssh_agent_socket,
        ),
        ("_json_native", "label", "JSON value"),
        ("_json_native", "_active_containers", None),
        (
            "_validated_source_provenance",
            "expected_surface",
            qualification.PUBLICATION_SURFACE_PATHS,
        ),
        ("capture_published_source", "surface_paths", qualification.PUBLICATION_SURFACE_PATHS),
        ("stable_read_bytes", "maximum", qualification.MAX_JSON_BYTES),
        (
            "_activate_recovery_checkpoint_dependency",
            "_sys",
            qualification.sys,
        ),
        (
            "_activate_recovery_checkpoint_dependency",
            "_import_module",
            qualification.importlib.import_module,
        ),
        (
            "_pinned_durable_replace",
            "_replace",
            qualification.os.replace,
        ),
    )
    for helper_name, default_name, expected in default_checks:
        helper = validated.get(helper_name)
        if helper is None:
            raise PermissionError(
                f"qualification default helper {helper_name} escaped transitive closure"
            )
        defaults = helper.__kwdefaults__ or {}
        positional_defaults = helper.__defaults__ or ()
        if default_name in defaults:
            actual = defaults[default_name]
        elif positional_defaults:
            actual = positional_defaults[-1]
        else:
            raise PermissionError(
                f"qualification captured default {helper_name}.{default_name} is absent"
            )
        if (
            _type(expected) in {bool, int, str, type(None)}
            and (_type(actual) is not _type(expected) or actual != expected)
        ) or (_type(expected) not in {bool, int, str, type(None)} and actual is not expected):
            raise PermissionError(
                f"qualification captured default {helper_name}.{default_name} differs"
            )
    return tuple(validated[name] for name in names)


def _internal_main(
    argv: list[str],
    receipt: dict[str, Any],
    authorization_fd_text: str,
    bootstrap_environment_hex: str,
    bootstrap_security: Any,
    *,
    _expected_capability: Any = _BOOTSTRAP_INTERNAL_CAPABILITY,
    _expected_seal: Any = _BOOTSTRAP_SECURITY_SEAL,
    _capture: Any = _capture_outer_receipt,
    _consume: Any = _consume_outer_receipt,
    _loader_type: Any = _ExactCommitLoader,
    _canonical: Any = _canonical_json,
    _arguments: Any = arguments,
    _import: Any = builtins.__import__,
    _json_dumps: Any = json.dumps,
    _suppress: Any = contextlib.suppress,
    _sys: Any = sys,
    _repository_root: Path = REPOSITORY_ROOT,
    _config_path: Path = CONFIG_PATH,
    _qualification_name: str = _QUALIFICATION_MODULE,
    _qualification_helpers: Any = _qualification_callables,
    _hash: Any = _sha256,
    _print: Any = builtins.print,
) -> int:
    if _expected_capability is None or bootstrap_security is not _expected_capability:
        raise PermissionError("internal runner lacks the exact bootstrap capability")
    security = bootstrap_security()
    expected_security = (
        _expected_seal,
        _capture,
        _consume,
        _loader_type,
        _canonical,
        _arguments,
        _import,
        _json_dumps,
        _suppress,
        _sys,
        _repository_root,
        _config_path,
        _qualification_name,
        _qualification_helpers,
        _hash,
        _print,
    )
    if (
        _expected_seal is None
        or type(security) is not tuple
        or len(security) != len(expected_security)
        or any(
            actual is not expected
            for actual, expected in zip(security, expected_security, strict=True)
        )
    ):
        raise PermissionError("internal runner bootstrap bindings differ")
    loader: _ExactCommitLoader | None = None
    qualification: Any = None
    runner_authority: Any = None
    revoke_authority: Any = None
    try:
        independently_captured, _, current_runner_blob = _capture(argv)
        if _canonical(independently_captured["source_provenance"]) != _canonical(
            receipt["source_provenance"]
        ):
            raise PermissionError(
                "outer receipt source differs from independently recaptured clean upstream HEAD"
            )
        runner_binding = receipt["source_provenance"]["publication_surface_blobs"]["runner"]
        if runner_binding["blob_sha256"] != _hash(current_runner_blob):
            raise PermissionError("independently recaptured runner blob differs")
        if any(
            name == "torch"
            or name.startswith("torch.")
            or name == "world_model"
            or name.startswith("world_model.")
            for name in _sys.modules
        ):
            raise PermissionError("heavy modules were preloaded before exact-commit isolation")
        loader = _loader_type(receipt)
        if loader._commit != receipt["source_provenance"]["commit"] or loader._cache != {}:
            raise PermissionError("new exact-commit loader state differs")
        _sys.meta_path.insert(0, loader)
        if _sys.meta_path[0] is not loader or _sys.meta_path.count(loader) != 1:
            raise PermissionError("exact-commit loader is not uniquely first")
        qualification = loader.load_lightweight_qualification(_qualification_name)
        (
            capture_source,
            register_authority,
            _require_config,
            bridge_protocol,
            development_report_path,
            checkpoint_path,
            qualification_report_path,
            run_development,
            run_qualification,
            revoke_authority,
        ) = _qualification_helpers(qualification, loader)
        source = capture_source(_repository_root)
        if _canonical(source) != _canonical(receipt["source_provenance"]):
            raise PermissionError("internal source recapture differs from the outer receipt")
        args = _arguments(argv)
        if args.phase == "protocol":
            _print(
                _json_dumps(
                    bridge_protocol(),
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        runner_authority = register_authority(
            authorization_fd_text=authorization_fd_text,
            bootstrap_environment_hex=bootstrap_environment_hex,
            receipt=receipt,
            argv=argv,
            source_provenance=source,
        )
        if args.phase == "development":
            return run_development(
                config_path=_config_path,
                report_path=development_report_path(),
                checkpoint_path=checkpoint_path(),
                source_provenance=source,
                runner_authority=runner_authority,
            )
        return run_qualification(
            config_path=_config_path,
            report_path=qualification_report_path(),
            checkpoint_path=checkpoint_path(),
            development_report_path=development_report_path(),
            reviewed_checkpoint_sha256=args.reviewed_checkpoint_sha256,
            reviewed_report_sha256=args.reviewed_report_sha256,
            reviewed_development_ledger_sha256=args.reviewed_development_ledger_sha256,
            source_provenance=source,
            runner_authority=runner_authority,
        )
    finally:
        if revoke_authority is not None and runner_authority is not None:
            with _suppress(BaseException):
                revoke_authority(runner_authority)
        if loader is not None:
            while loader in _sys.meta_path:
                _sys.meta_path.remove(loader)
            for name, module in tuple(_sys.modules.items()):
                spec = getattr(module, "__spec__", None)
                if (
                    (name == "world_model" or name.startswith("world_model."))
                    and spec is not None
                    and spec.loader is loader
                ):
                    _sys.modules.pop(name, None)
            loader._cache.clear()


def _outer_preflight(
    argv: list[str],
    *,
    _sys: Any = sys,
    _sysconfig: Any = sysconfig,
    _os: Any = os,
    _builtins: Any = builtins,
    _path_type: Any = Path,
    _runner_path: Path = RUNNER_PATH,
    _repository_root: Path = REPOSITORY_ROOT,
    _compile: Any = builtins.compile,
    _ast_parse: Any = ast.parse,
    _ast_module: Any = ast,
    _function_type: Any = FunctionType,
    _source_loader_type: Any = importlib.machinery.SourceFileLoader,
    _getframe: Any = sys._getframe,
    _getattr: Any = builtins.getattr,
    _vars: Any = builtins.vars,
    _type: Any = builtins.type,
    _tuple: Any = builtins.tuple,
    _list: Any = builtins.list,
    _dict: Any = builtins.dict,
    _any: Any = builtins.any,
    _len: Any = builtins.len,
    _fspath: Any = os.fspath,
    _lstat: Any = Path.lstat,
    _isreg: Any = stat.S_ISREG,
    _open: Any = os.open,
    _fstat: Any = os.fstat,
    _read_fd: Any = os.read,
    _close: Any = os.close,
    _resolve: Any = Path.resolve,
    _read_flags: int = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    _sys_modules: Any = sys.modules,
    _sys_meta_path: Any = sys.meta_path,
    _sys_path: Any = sys.path,
    _sys_path_hooks: Any = sys.path_hooks,
    _sys_path_importer_cache: Any = sys.path_importer_cache,
    _os_environ: Any = os.environ,
    _critical_attributes: tuple[tuple[Any, str, Any], ...] = (
        (argparse, "ArgumentParser", argparse.ArgumentParser),
        (ast, "parse", ast.parse),
        (builtins, "compile", builtins.compile),
        (builtins, "exec", builtins.exec),
        (builtins, "__import__", builtins.__import__),
        (builtins, "open", builtins.open),
        (builtins, "getattr", builtins.getattr),
        (builtins, "hasattr", builtins.hasattr),
        (builtins, "type", builtins.type),
        (builtins, "isinstance", builtins.isinstance),
        (builtins, "callable", builtins.callable),
        (builtins, "len", builtins.len),
        (builtins, "any", builtins.any),
        (builtins, "all", builtins.all),
        (builtins, "set", builtins.set),
        (builtins, "dict", builtins.dict),
        (builtins, "list", builtins.list),
        (builtins, "tuple", builtins.tuple),
        (builtins, "int", builtins.int),
        (builtins, "str", builtins.str),
        (builtins, "bytes", builtins.bytes),
        (builtins, "object", builtins.object),
        (builtins, "print", builtins.print),
        (builtins, "repr", builtins.repr),
        (builtins, "sorted", builtins.sorted),
        (builtins, "zip", builtins.zip),
        (builtins, "enumerate", builtins.enumerate),
        (contextlib, "suppress", contextlib.suppress),
        (hashlib, "sha1", hashlib.sha1),
        (hashlib, "sha256", hashlib.sha256),
        (importlib.abc, "MetaPathFinder", importlib.abc.MetaPathFinder),
        (importlib.abc, "Loader", importlib.abc.Loader),
        (importlib.machinery, "BuiltinImporter", importlib.machinery.BuiltinImporter),
        (importlib.machinery, "ModuleSpec", importlib.machinery.ModuleSpec),
        (importlib.machinery, "SourceFileLoader", importlib.machinery.SourceFileLoader),
        (importlib, "abc", importlib.abc),
        (importlib, "machinery", importlib.machinery),
        (importlib, "util", importlib.util),
        (importlib.util, "spec_from_loader", importlib.util.spec_from_loader),
        (json, "dumps", json.dumps),
        (json, "loads", json.loads),
        (os, "open", os.open),
        (os, "lstat", os.lstat),
        (os, "read", os.read),
        (os, "write", os.write),
        (os, "close", os.close),
        (os, "fstat", os.fstat),
        (os, "fsync", os.fsync),
        (os, "geteuid", os.geteuid),
        (os, "pipe", os.pipe),
        (os, "fspath", os.fspath),
        (os, "getpid", os.getpid),
        (os, "getppid", os.getppid),
        (os, "set_inheritable", os.set_inheritable),
        (os, "path", os.path),
        (os.path, "dirname", os.path.dirname),
        (os.path, "isabs", os.path.isabs),
        (os.path, "join", os.path.join),
        (os.path, "realpath", os.path.realpath),
        (secrets, "token_hex", secrets.token_hex),
        (stat, "S_ISDIR", stat.S_ISDIR),
        (stat, "S_ISFIFO", stat.S_ISFIFO),
        (stat, "S_ISREG", stat.S_ISREG),
        (stat, "S_ISSOCK", stat.S_ISSOCK),
        (subprocess, "Popen", subprocess.Popen),
        (subprocess, "run", subprocess.run),
        (tempfile, "TemporaryFile", tempfile.TemporaryFile),
        (tempfile, "TemporaryDirectory", tempfile.TemporaryDirectory),
        (Path, "is_absolute", Path.is_absolute),
        (Path, "lstat", Path.lstat),
        (Path, "resolve", Path.resolve),
        (Path, "__truediv__", Path.__truediv__),
    ),
) -> None:
    caller = _getframe(1)
    namespace = caller.f_globals
    outer = namespace.get("_outer_main")
    main_frame = caller.f_back
    module_frame = None if main_frame is None else main_frame.f_back
    main_module = _sys_modules.get("__main__")
    loader = namespace.get("__loader__")
    expected_flags = {
        "isolated": 1,
        "no_site": 1,
        "ignore_environment": 1,
        "no_user_site": 1,
        "safe_path": None,
    }
    flags = {
        "isolated": _sys.flags.isolated,
        "no_site": _sys.flags.no_site,
        "ignore_environment": _sys.flags.ignore_environment,
        "no_user_site": _sys.flags.no_user_site,
        "safe_path": _getattr(_sys.flags, "safe_path", None),
    }
    paths = _sysconfig.get_paths()
    stdlib = _os.path.realpath(paths["stdlib"])
    platstdlib = _os.path.realpath(paths["platstdlib"])
    archive = _os.path.join(
        _os.path.dirname(stdlib),
        f"python{_sys.version_info.major}{_sys.version_info.minor}.zip",
    )
    expected_path = _tuple(
        _dict.fromkeys((archive, stdlib, platstdlib, _os.path.join(stdlib, "lib-dynload")))
    )
    original_argv = _list(_sys.orig_argv)
    if (
        main_module is None
        or namespace is not _vars(main_module)
        or _type(outer) is not _function_type
        or caller.f_code is not outer.__code__
        or caller.f_globals is not namespace
        or caller.f_builtins is not _vars(_builtins)
        or caller.f_locals.get("argv") is not argv
        or main_frame is None
        or main_frame.f_globals is not namespace
        or main_frame.f_code is not namespace.get("main").__code__
        or main_frame.f_locals.get("argv") is not None
        or main_frame.f_locals.get("exact_argv") is not argv
        or module_frame is None
        or module_frame.f_globals is not namespace
        or module_frame.f_code.co_name != "<module>"
        or namespace.get("__name__") != "__main__"
        or namespace.get("__file__") != _fspath(_runner_path)
        or namespace.get("__package__") is not None
        or namespace.get("__spec__") is not None
        or namespace.get("__cached__") is not None
        or namespace.get("__builtins__") is not _builtins
        or _type(loader) is not _source_loader_type
        or loader.name != "__main__"
        or loader.path != _fspath(_runner_path)
        or flags != expected_flags
        or _sys.modules is not _sys_modules
        or _sys.meta_path is not _sys_meta_path
        or _sys.path is not _sys_path
        or _sys.path_hooks is not _sys_path_hooks
        or _sys.path_importer_cache is not _sys_path_importer_cache
        or _os.environ is not _os_environ
        or _tuple(_sys.path) != expected_path
        or original_argv[1:4] != ["-I", "-S", _fspath(_runner_path)]
        or original_argv[4:] != _list(_sys.argv[1:])
        or _sys.argv != [_fspath(_runner_path), *argv]
        or _any(name == "world_model" or name.startswith("world_model.") for name in _sys.modules)
        or _any(
            _getattr(owner, attribute) is not expected
            for owner, attribute, expected in _critical_attributes
        )
    ):
        raise PermissionError("outer runner is not one exact isolated direct script")
    cwd = _os.path.realpath(_os.getcwd())
    repository = _os.path.realpath(_repository_root)
    if _any(
        not value or not _os.path.isabs(value) or _os.path.realpath(value) in {cwd, repository}
        for value in expected_path
    ):
        raise PermissionError("outer runner preloader path contains an unsafe entry")
    expected_constants = {
        "REPOSITORY_ROOT": _repository_root,
        "RUNNER_PATH": _runner_path,
        "CONFIG_PATH": _repository_root / "configs" / "rgbd_known_action_planning_cpu.yaml",
        "PUBLICATION_SURFACE_PATHS": {
            "qualification": "world_model/training/rgbd_known_action_qualification.py",
            "runner": "scripts/run_rgbd_known_action_qualification.py",
            "qualification_test": "tests/unit/test_rgbd_known_action_qualification.py",
        },
        "_QUALIFICATION_MODULE": "world_model.training.rgbd_known_action_qualification",
        "_RECEIPT_SCHEMA": "rgbd_known_action_outer_preflight_v1",
        "_AUTHORIZATION_SCHEMA": "rgbd_known_action_outer_authorization_v1",
        "_RECEIPT_ENV_PREFIX": "_RGBD_KNOWN_ACTION_OUTER_",
        "_RECEIPT_FD_ENV": "_RGBD_KNOWN_ACTION_OUTER_FD",
        "_RECEIPT_SHA_ENV": "_RGBD_KNOWN_ACTION_OUTER_SHA256",
        "_RECEIPT_NONCE_ENV": "_RGBD_KNOWN_ACTION_OUTER_NONCE",
        "_MAX_RECEIPT_BYTES": 16 * 1024,
        "_MAX_AUTHORIZATION_BYTES": 16 * 1024,
        "_MAX_SOURCE_BYTES": 8 * 1024 * 1024,
        "_GIT_OID_BYTES": 20,
        "_TRUSTED_GIT": "/usr/bin/git",
        "_APPROVED_BRANCH": "agent/rgbd-known-action-planning-rung-1",
        "_APPROVED_REMOTE_NAME": "origin",
        "_APPROVED_REMOTE_URL": "git@github.com:polceanum/world.model.git",
        "_APPROVED_BRANCH_MERGE_REF": "refs/heads/agent/rgbd-known-action-planning-rung-1",
        "_APPROVED_UPSTREAM_REF": "refs/remotes/origin/agent/rgbd-known-action-planning-rung-1",
        "_REMOTE_PROBE_CWD": "/",
        "_REMOTE_TEMPORARY_DIRECTORY": "/private/tmp",
        "_REMOTE_PROBE_TIMEOUT_SECONDS": 20,
        "_REMOTE_PROBE_MAX_OUTPUT_BYTES": 4096,
        "_LIGHTWEIGHT_QUALIFICATION_EXECUTION_SHA256": "68e472d8356143ecf89647f7d98d69f914e1f448d58827378a2ef75f1af8a4c3",
        "_PINNED_GITHUB_HOST_KEY": "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl\n",
        "_PINNED_GITHUB_HOST_KEY_FINGERPRINT": "SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU",
        "_REMOTE_SSH_COMMAND_TEMPLATE": "/usr/bin/ssh -F /dev/null -oBatchMode=yes -oClearAllForwardings=yes -oForwardAgent=no -oForwardX11=no -oProxyCommand=none -oProxyJump=none -oCanonicalizeHostname=no -oStrictHostKeyChecking=yes -oCheckHostIP=yes -oPasswordAuthentication=no -oKbdInteractiveAuthentication=no -oIdentityFile=/dev/null -oIdentitiesOnly=no -oIdentityAgent=SSH_AUTH_SOCK -oAddKeysToAgent=no -oPKCS11Provider=none -oSecurityKeyProvider=none -oGSSAPIAuthentication=no -oHostbasedAuthentication=no -oPubkeyAuthentication=yes -oHostKeyAlgorithms=ssh-ed25519 -oHostKeyAlias=github.com -oUserKnownHostsFile={known_hosts_file} -oGlobalKnownHostsFile=/dev/null",
        "_BOOTSTRAP_SECURITY_SEAL": None,
        "_BOOTSTRAP_MAIN_CAPABILITY": None,
        "_BOOTSTRAP_INTERNAL_CAPABILITY": None,
    }
    if (
        _type(_runner_path) is not _path_type
        or _type(_repository_root) is not _path_type
        or _resolve(_runner_path) != _runner_path
        or _resolve(_repository_root) != _repository_root
        or _any(
            _type(namespace.get(name)) is not _type(expected) or namespace.get(name) != expected
            for name, expected in expected_constants.items()
        )
    ):
        raise PermissionError("outer runner constant binding differs before Git capture")
    module_identities = {
        "argparse": argparse,
        "ast": ast,
        "builtins": builtins,
        "contextlib": contextlib,
        "hashlib": hashlib,
        "importlib": importlib,
        "json": json,
        "os": os,
        "secrets": secrets,
        "stat": stat,
        "subprocess": subprocess,
        "sys": sys,
        "sysconfig": sysconfig,
        "tempfile": tempfile,
    }
    if (
        _any(namespace.get(name) is not expected for name, expected in module_identities.items())
        or namespace.get("Path") is not _path_type
        or namespace.get("FunctionType") is not _function_type
        or namespace.get("ModuleType") is not ModuleType
        or namespace.get("Any") is not Any
    ):
        raise PermissionError("outer runner stdlib binding differs before Git capture")

    before = _lstat(_runner_path)
    if (
        not _isreg(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > expected_constants["_MAX_SOURCE_BYTES"]
    ):
        raise PermissionError("outer runner source is not one bounded regular file")
    descriptor = _open(_runner_path, _read_flags)
    try:
        opened = _fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while chunk := _read_fd(descriptor, 65_536):
            chunks.append(chunk)
            total += _len(chunk)
            if total > expected_constants["_MAX_SOURCE_BYTES"]:
                raise PermissionError("outer runner source is oversized")
        after_open = _fstat(descriptor)
    finally:
        _close(descriptor)
    after = _lstat(_runner_path)
    identity = lambda item: (  # noqa: E731
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if _any(identity(item) != identity(before) for item in (opened, after_open, after)):
        raise PermissionError("outer runner source changed during isolated preflight")
    runner_bytes = b"".join(chunks)
    if _len(runner_bytes) != before.st_size:
        raise PermissionError("outer runner source length changed during isolated preflight")
    tree = _ast_parse(
        runner_bytes.decode("utf-8", errors="strict"),
        filename=_fspath(_runner_path),
    )
    compiled = _compile(runner_bytes, _fspath(_runner_path), "exec", dont_inherit=True)
    code_type = _type(compiled)
    expected_codes = {
        value.co_name: value for value in compiled.co_consts if _type(value) is code_type
    }
    function_names = [node.name for node in tree.body if _type(node) is _ast_module.FunctionDef]
    if _any(
        _type(namespace.get(name)) is not _function_type
        or namespace[name].__name__ != name
        or namespace[name].__qualname__ != name
        or namespace[name].__code__ != expected_codes.get(name)
        or namespace[name].__globals__ is not namespace
        or namespace[name].__module__ != "__main__"
        or namespace[name].__closure__ is not None
        or namespace[name].__dict__ != {}
        for name in function_names
    ):
        raise PermissionError("outer runner helper code differs before Git capture")
    bootstrap_assignments = [
        node
        for node in tree.body
        if _type(node) is _ast_module.Assign
        and _len(node.targets) == 1
        and _type(node.targets[0]) is _ast_module.Name
        and node.targets[0].id == "_BOOTSTRAP"
    ]
    if (
        _len(bootstrap_assignments) != 1
        or _type(bootstrap_assignments[0].value) is not _ast_module.Constant
        or _type(bootstrap_assignments[0].value.value) is not str
        or namespace.get("_BOOTSTRAP") != bootstrap_assignments[0].value.value
    ):
        raise PermissionError("outer runner bootstrap literal differs before Git capture")

    default_checks = (
        ("_canonical_json", "_dumps", json.dumps),
        ("_strict_json_loads", "_loads", json.loads),
        ("_sha256", "_hash", hashlib.sha256),
        ("_git_paths", "_is_absolute", Path.is_absolute),
        ("_git_paths", "_resolve", Path.resolve),
        ("_git_paths", "_lstat", Path.lstat),
        ("_git_paths", "_isdir", stat.S_ISDIR),
        ("_git_environment", "_paths", namespace["_git_paths"]),
        ("_git_environment", "_fspath", os.fspath),
        ("_git_environment", "_environ", os.environ),
        ("_git_command", "_paths", namespace["_git_paths"]),
        ("_git_bytes", "_run", subprocess.run),
        ("_git_bytes", "_command", namespace["_git_command"]),
        ("_git_bytes", "_environment", namespace["_git_environment"]),
        ("_git_text", "_bytes", namespace["_git_bytes"]),
        ("_approved_git_config_snapshot", "_bytes", namespace["_git_bytes"]),
        ("_validated_ssh_agent_socket", "_environ", os.environ),
        ("_validated_ssh_agent_socket", "_lstat", os.lstat),
        ("_validated_ssh_agent_socket", "_is_socket", stat.S_ISSOCK),
        ("_validated_ssh_agent_socket", "_geteuid", os.geteuid),
        ("_remote_transport", "_run", subprocess.run),
        ("_remote_transport", "_agent", namespace["_validated_ssh_agent_socket"]),
        ("_object_oid", "_hash", hashlib.sha1),
        ("_verified_object", "_validate", namespace["_validated_oid"]),
        ("_verified_object", "_text", namespace["_git_text"]),
        ("_verified_object", "_bytes", namespace["_git_bytes"]),
        ("_verified_object", "_object", namespace["_object_oid"]),
        ("_stable_read", "_lstat", Path.lstat),
        ("_stable_read", "_isreg", stat.S_ISREG),
        ("_stable_read", "_open", os.open),
        ("_stable_read", "_fstat", os.fstat),
        ("_stable_read", "_read", os.read),
        ("_stable_read", "_close", os.close),
        ("_blob_binding", "_validate", namespace["_validated_oid"]),
        ("_blob_binding", "_text", namespace["_git_text"]),
        ("_blob_binding", "_verified", namespace["_verified_object"]),
        ("_blob_binding", "_bytes", namespace["_git_bytes"]),
        ("_blob_binding", "_read", namespace["_stable_read"]),
        ("_blob_binding", "_hash", namespace["_sha256"]),
        ("_capture_outer_receipt", "_bytes", namespace["_git_bytes"]),
        ("_capture_outer_receipt", "_text", namespace["_git_text"]),
        ("_capture_outer_receipt", "_validate", namespace["_validated_oid"]),
        ("_capture_outer_receipt", "_verified", namespace["_verified_object"]),
        ("_capture_outer_receipt", "_blob", namespace["_blob_binding"]),
        ("_capture_outer_receipt", "_config", namespace["_approved_git_config_snapshot"]),
        ("_capture_outer_receipt", "_probe", namespace["_probe_remote_publication"]),
        ("_capture_outer_receipt", "_fspath", os.fspath),
        ("_capture_outer_receipt", "_token_hex", secrets.token_hex),
        ("_capture_outer_receipt", "_getpid", os.getpid),
        ("_capture_outer_receipt", "_hash", namespace["_sha256"]),
        ("_capture_outer_receipt", "_canonical", namespace["_canonical_json"]),
        ("arguments", "_parser", argparse.ArgumentParser),
        ("arguments", "_validate_hash", namespace["_validated_sha256"]),
        ("_outer_main", "_preflight", namespace["_outer_preflight"]),
        ("_outer_main", "_arguments", namespace["arguments"]),
        ("_outer_main", "_capture", namespace["_capture_outer_receipt"]),
        ("_outer_main", "_canonical", namespace["_canonical_json"]),
        ("_outer_main", "_hash", namespace["_sha256"]),
        ("_outer_main", "_pipe", os.pipe),
        ("_outer_main", "_environ", os.environ),
        ("_outer_main", "_temporary_file", tempfile.TemporaryFile),
        ("_outer_main", "_fsync", os.fsync),
        ("_outer_main", "_fstat", os.fstat),
        ("_outer_main", "_write", os.write),
        ("_outer_main", "_close", os.close),
        ("_outer_main", "_run", subprocess.run),
        ("_outer_main", "_sys", sys),
        ("_outer_main", "_fspath", os.fspath),
        ("_outer_main", "_repository_root", _repository_root),
        ("_outer_main", "_runner_path", _runner_path),
        ("_outer_main", "_bootstrap", namespace["_BOOTSTRAP"]),
        ("_outer_main", "_max_receipt_bytes", namespace["_MAX_RECEIPT_BYTES"]),
        (
            "_outer_main",
            "_max_authorization_bytes",
            namespace["_MAX_AUTHORIZATION_BYTES"],
        ),
        ("_outer_main", "_receipt_env_prefix", namespace["_RECEIPT_ENV_PREFIX"]),
        ("_outer_main", "_receipt_fd_env", namespace["_RECEIPT_FD_ENV"]),
        ("_outer_main", "_receipt_sha_env", namespace["_RECEIPT_SHA_ENV"]),
        ("_outer_main", "_receipt_nonce_env", namespace["_RECEIPT_NONCE_ENV"]),
        ("_outer_main", "_executable", sys.executable),
        ("_outer_main", "_len", builtins.len),
        ("_outer_main", "_str", builtins.str),
        ("main", "_environ", os.environ),
        ("main", "_sys", sys),
        ("main", "_consume", namespace["_consume_outer_receipt"]),
        ("main", "_internal", namespace["_internal_main"]),
        ("main", "_outer", namespace["_outer_main"]),
    )
    if _any(
        _type(namespace[helper_name].__kwdefaults__) is not dict
        or namespace[helper_name].__kwdefaults__.get(default_name) is not expected
        for helper_name, default_name, expected in default_checks
    ):
        raise PermissionError("outer runner captured callable differs before Git capture")


def _outer_main(
    argv: list[str],
    *,
    _preflight: Any = _outer_preflight,
    _arguments: Any = arguments,
    _capture: Any = _capture_outer_receipt,
    _canonical: Any = _canonical_json,
    _hash: Any = _sha256,
    _pipe: Any = os.pipe,
    _environ: Any = os.environ,
    _temporary_file: Any = tempfile.TemporaryFile,
    _fsync: Any = os.fsync,
    _fstat: Any = os.fstat,
    _write: Any = os.write,
    _close: Any = os.close,
    _run: Any = subprocess.run,
    _sys: Any = sys,
    _fspath: Any = os.fspath,
    _repository_root: Path = REPOSITORY_ROOT,
    _runner_path: Path = RUNNER_PATH,
    _bootstrap: str = _BOOTSTRAP,
    _max_receipt_bytes: int = _MAX_RECEIPT_BYTES,
    _max_authorization_bytes: int = _MAX_AUTHORIZATION_BYTES,
    _receipt_env_prefix: str = _RECEIPT_ENV_PREFIX,
    _receipt_fd_env: str = _RECEIPT_FD_ENV,
    _receipt_sha_env: str = _RECEIPT_SHA_ENV,
    _receipt_nonce_env: str = _RECEIPT_NONCE_ENV,
    _executable: str = sys.executable,
    _len: Any = builtins.len,
    _str: Any = builtins.str,
) -> int:
    _preflight(argv)
    # Validate arguments before granting any child authority.
    _arguments(argv)
    receipt, authorization_record, runner_blob = _capture(argv)
    contents = _canonical(receipt)
    authorization_contents = _canonical(authorization_record)
    if _len(contents) > _max_receipt_bytes:
        raise RuntimeError("outer receipt exceeds its fixed byte budget")
    if _len(authorization_contents) > _max_authorization_bytes:
        raise RuntimeError("outer authorization record exceeds its fixed byte budget")
    if any(key.startswith(_receipt_env_prefix) for key in _environ):
        raise PermissionError("outer environment contains preloaded receipt authority")
    agent_path, _ = _validated_ssh_agent_socket()
    read_fd, write_fd = _pipe()
    authorization_read_fd, authorization_write_fd = _pipe()
    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "SSH_AUTH_SOCK": agent_path,
        _receipt_fd_env: _str(read_fd),
        _receipt_sha_env: _hash(contents),
        _receipt_nonce_env: receipt["nonce"],
    }
    with _temporary_file() as bootstrap:
        bootstrap.write(runner_blob)
        bootstrap.flush()
        _fsync(bootstrap.fileno())
        bootstrap.seek(0)
        bootstrap_fd = bootstrap.fileno()
        if _fstat(bootstrap_fd).st_nlink != 0:
            raise PermissionError("runner bootstrap file must be anonymous")
        try:
            written = 0
            while written < len(contents):
                written += _write(write_fd, contents[written:])
        finally:
            _close(write_fd)
        try:
            authorization_written = 0
            while authorization_written < len(authorization_contents):
                authorization_written += _write(
                    authorization_write_fd,
                    authorization_contents[authorization_written:],
                )
        finally:
            _close(authorization_write_fd)
        completed = _run(
            [
                _executable,
                "-I",
                "-S",
                "-c",
                _bootstrap,
                _str(bootstrap_fd),
                _hash(runner_blob),
                _fspath(_runner_path),
                _str(authorization_read_fd),
                _hash(_bootstrap.encode("utf-8")),
                *argv,
            ],
            cwd=_repository_root,
            env=environment,
            pass_fds=(read_fd, authorization_read_fd, bootstrap_fd),
            check=False,
        )
    _close(read_fd)
    _close(authorization_read_fd)
    return completed.returncode


def main(
    argv: list[str] | None = None,
    *,
    _authorization_fd_text: str | None = None,
    _bootstrap_environment_hex: str | None = None,
    _bootstrap_security: Any = None,
    _expected_capability: Any = _BOOTSTRAP_MAIN_CAPABILITY,
    _environ: Any = os.environ,
    _sys: Any = sys,
    _consume: Any = _consume_outer_receipt,
    _internal: Any = _internal_main,
    _outer: Any = _outer_main,
) -> int:
    if _bootstrap_security is not None:
        if _expected_capability is None or _bootstrap_security is not _expected_capability:
            raise PermissionError("runner main bootstrap capability differs")
        internal_capability = _bootstrap_security()
    elif _expected_capability is not None:
        raise PermissionError("trusted runner main omitted its bootstrap capability")
    else:
        internal_capability = None
    internal = _environ.get(_RECEIPT_FD_ENV) is not None
    if internal:
        if (
            argv is not None
            or type(_authorization_fd_text) is not str
            or type(_bootstrap_environment_hex) is not str
            or internal_capability is None
        ):
            raise PermissionError("internal runner lacks explicit bootstrap locals")
        exact_argv = list(_sys.argv[1:])
        receipt = _consume(exact_argv)
        return _internal(
            exact_argv,
            receipt,
            _authorization_fd_text,
            _bootstrap_environment_hex,
            internal_capability,
        )
    if (
        _authorization_fd_text is not None
        or _bootstrap_environment_hex is not None
        or _bootstrap_security is not None
    ):
        raise PermissionError("outer runner rejects bootstrap-only arguments")
    exact_argv = list(_sys.argv[1:] if argv is None else argv)
    return _outer(exact_argv)


if __name__ == "__main__":
    raise SystemExit(main())
