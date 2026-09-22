"""Versioned, non-executable scan snapshots with lazy, copy-on-write NumPy arrays."""
import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import uuid
import numpy as np
from .capture import CaptureGeometry
from .sessions import Session

log = logging.getLogger(__name__)
# Bump when reconstruction semantics change; detection has its own version.
RECONSTRUCTION_VERSION = 5
DETECTION_VERSION = 2
TRANSIENT = {'masks', 'photo_discovery_cache', 'tool_view_cache', 'tool_image_frames',
             'photo_result_cache', 'detection_lock'}


def fingerprint(directory):
    manifest = directory / 'capture.json'
    meta = json.loads(manifest.read_text())
    raw = []
    for frame in meta['frames']:
        for key in ('image', 'depth'):
            if frame.get(key):
                p = directory / Path(frame[key]).name
                st = p.stat()
                raw.append((p.name, st.st_size, st.st_mtime_ns))
    # Derived listing fields do not affect reconstruction.
    value = [RECONSTRUCTION_VERSION, meta['frames'], meta.get('form'), raw]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_tree(directory, kind, key, value):
    """Publish manifest last; incomplete writes are never considered readable."""
    token = f'{kind}-{uuid.uuid4().hex}'
    target = directory / token
    target.mkdir()
    arrays = {}
    def encode(v):
        if isinstance(v, np.ndarray):
            ident = id(v)
            if ident not in arrays:
                name = f'{len(arrays)}.npy'
                np.save(target / name, v, allow_pickle=False)
                arrays[ident] = name
            return {'array': arrays[ident]}
        if isinstance(v, CaptureGeometry):
            return {'geometry': encode({f.name: getattr(v, f.name) for f in dataclasses.fields(v)})}
        if isinstance(v, dict):
            return {'dict': [[encode(k), encode(x)] for k, x in v.items()]}
        if isinstance(v, tuple):
            return {'tuple': [encode(x) for x in v]}
        if isinstance(v, list):
            return [encode(x) for x in v]
        if isinstance(v, np.generic):
            return v.item()
        return v
    try:
        tree = encode(value)
        pointer = directory / f'{kind}.json'
        old = json.loads(pointer.read_text()).get('directory') if pointer.exists() else None
        temporary = directory / f'{token}.json.tmp'
        temporary.write_text(json.dumps({'key': key, 'directory': token, 'tree': tree}))
        os.replace(temporary, pointer)
        if old and old.startswith(kind + '-') and Path(old).name == old:
            shutil.rmtree(directory / old, ignore_errors=True)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def read_tree(directory, kind, key):
    try:
        meta = json.loads((directory / f'{kind}.json').read_text())
        if meta['key'] != key:
            return None
        folder = meta['directory']
        if Path(folder).name != folder or not folder.startswith(kind + '-'):
            return None
        target = directory / folder
        arrays = {}
        def decode(v):
            if isinstance(v, list): return [decode(x) for x in v]
            if not isinstance(v, dict): return v
            if 'array' in v:
                name = v['array']
                if Path(name).name != name or not name.endswith('.npy'): raise ValueError('Invalid array path')
                if name not in arrays:
                    arrays[name] = np.load(target / name, mmap_mode='c', allow_pickle=False)
                return arrays[name]
            if 'geometry' in v: return CaptureGeometry(**decode(v['geometry']))
            if 'tuple' in v: return tuple(decode(x) for x in v['tuple'])
            if 'dict' in v: return {decode(k): decode(x) for k, x in v['dict']}
            raise ValueError('Invalid snapshot node')
        return decode(meta['tree'])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_session(directory, session):
    try:
        value = {f.name: getattr(session, f.name) for f in dataclasses.fields(session) if f.name not in TRANSIENT}
        write_tree(directory, 'processed', fingerprint(directory), value)
    except Exception:
        log.exception('Could not cache processed scan %s', session.id)


def load_session(directory):
    try:
        value = read_tree(directory, 'processed', fingerprint(directory))
        return Session(**value) if value else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_detection(directory, session):
    try:
        write_tree(directory, 'detection', [fingerprint(directory), DETECTION_VERSION], session.photo_result_cache)
    except Exception:
        log.exception('Could not cache detected tools %s', session.id)


def load_detection(directory):
    try:
        return read_tree(directory, 'detection', [fingerprint(directory), DETECTION_VERSION])
    except (OSError, ValueError, KeyError):
        return None
