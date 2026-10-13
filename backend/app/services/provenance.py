"""Record checkpoint identity without loading model parameters."""
import hashlib
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

_HASHES = {}


def detector_provenance(checkpoint):
    path = Path(checkpoint) if checkpoint else None
    fingerprint = None
    if path and path.is_file():
        stat = path.stat()
        key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
        if key not in _HASHES:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            _HASHES[key] = digest.hexdigest()
        fingerprint = _HASHES[key]
    versions = {}
    for package in ('torch', 'torchvision', 'ultralytics', 'opencv-python'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    return {'checkpoint': str(path.resolve()) if path and path.is_file() else str(checkpoint or 'unspecified'),
            'checkpoint_sha256': fingerprint, 'package_versions': versions}
