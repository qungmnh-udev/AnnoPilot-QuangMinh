"""CVAT transport and checked sample/frame correspondence shared by both sync directions."""
import json
from pathlib import PurePosixPath
from urllib.parse import urljoin, urlsplit
import urllib.request


class CVATClient:
    def __init__(self, base, token, task_id=None, job_id=None):
        self.base = base.rstrip('/')
        self.headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
        self.task_id, self.job_id = task_id, job_id
        self.job = None

    def get(self, url):
        request = urllib.request.Request(urljoin(self.base + '/', url), headers=self.headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode('utf-8'))

    def labels(self):
        url = f'{self.base}/api/labels?task_id={self.task_id}'
        items, seen = [], set()
        while url:
            if url in seen or urlsplit(url).netloc != urlsplit(self.base).netloc or urlsplit(url).scheme != urlsplit(self.base).scheme:
                raise ValueError('Invalid CVAT label pagination URL')
            seen.add(url)
            page = self.get(url)
            if isinstance(page, list):
                items.extend(page)
                break
            items.extend(page.get('results', []))
            url = urljoin(url, page['next']) if page.get('next') else None
        return items

    def frame_mapping(self, samples):
        if self.job_id:
            self.job = self.get(f'{self.base}/api/jobs/{self.job_id}')
            actual_task = self.job.get('task_id')
            if actual_task is None or (self.task_id and actual_task != self.task_id):
                raise ValueError('CVAT Job ID does not belong to the configured Task ID')
            if self.job.get('type', 'annotation') != 'annotation':
                raise ValueError('Select an annotation job, not a ground-truth/consensus job')
            self.task_id = actual_task
        if not self.task_id:
            raise ValueError('Configure CVAT Task ID or Job ID')
        meta = self.get(f'{self.base}/api/tasks/{self.task_id}/data/meta')
        frames, size = meta.get('frames') or [], meta.get('size')
        if not isinstance(size, int) or size <= 0:
            raise ValueError('CVAT task frame metadata is missing or invalid')
        included = meta.get('included_frames')
        ids = included if included is not None else list(range(size))
        deleted = set(meta.get('deleted_frames', []))
        # Image tasks describe each image; video metadata may describe only the video file.
        image_task = len(frames) == len(ids)
        normalize = lambda name: str(name).replace('\\', '/').removeprefix('./')
        result = {}
        for sample in samples:
            if image_task:
                exact = [(number, frame) for number, frame in zip(ids, frames)
                         if normalize(frame.get('name', '')) == normalize(sample.file_name)]
                matches = exact or [(number, frame) for number, frame in zip(ids, frames)
                                   if PurePosixPath(normalize(frame.get('name', ''))).name == PurePosixPath(normalize(sample.file_name)).name]
                if len(matches) != 1:
                    raise ValueError(f'Cannot uniquely match {sample.file_name} to a CVAT image; export the matching task images/annotations')
                number, frame = matches[0]
                for axis in ('width', 'height'):
                    if getattr(sample, axis, None) and frame.get(axis) and getattr(sample, axis) != frame[axis]:
                        raise ValueError(f'Image dimensions differ from CVAT for {sample.file_name}; box coordinates cannot be reused')
            else:
                number = sample.frame_number
                if number is None or number not in ids:
                    raise ValueError(f'Explicit CVAT video frame ID required for {sample.file_name}')
            if number in deleted:
                raise ValueError(f'CVAT frame for {sample.file_name} is deleted')
            if self.job and not (self.job['start_frame'] <= number <= self.job['stop_frame']):
                raise ValueError(f'{sample.file_name} is outside the configured CVAT job; use Task scope or a matching dataset subset')
            result[sample.id] = number
        if len(set(result.values())) != len(result):
            raise ValueError('Multiple local samples map to the same CVAT frame')
        return result

    @property
    def endpoint(self):
        scope = f'jobs/{self.job_id}' if self.job_id else f'tasks/{self.task_id}'
        return f'{self.base}/api/{scope}/annotations'


class CVATFrames(dict):
    """Annotation map plus verified local-sample -> remote-frame mapping."""
    def __init__(self, mapping):
        super().__init__()
        self.frame_by_sample = mapping
