"""Read-only tensor audit; writes receipts only in this review directory. No inference."""
import datetime
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

REVIEW = Path(__file__).resolve().parent
SOURCE = Path('/Users/adithyagiridharan/Desktop/PythonProjects/claude')
BASE = Path('/Users/adithyagiridharan/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/model.safetensors-00001-of-00001.safetensors')
JOB = REVIEW / 'recovered-heldout/edits.safetensors'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(8 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def header(path):
    with path.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def tensor(path, name):
    h, offset = header(path)
    t = h[name]
    a = np.memmap(path, mode='r', dtype={'BF16': '<u2', 'F32': '<f4'}[t['dtype']],
                  offset=offset + t['data_offsets'][0], shape=tuple(t['shape']))
    native_sha = hashlib.sha256(a.tobytes()).hexdigest()
    if t['dtype'] == 'BF16':
        a = (a.astype(np.uint32) << 16).view(np.float32)
    return a, t['dtype'], native_sha


receipt = {
    'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'method': 'Read safetensors headers and arrays with NumPy; compare exact values. No model or layer execution.',
    'base_repo': 'Qwen/Qwen3.5-0.8B',
    'base_revision_declared_in_harness': '2fc06364715b967f1860aea9cf38778875588b17',
    'base_path': str(BASE), 'base_resolved_path': str(BASE.resolve()),
    'base_file_sha256': sha(BASE),
    'local_edit_file_count': len(list((SOURCE / 'work/edited').glob('*'))),
    'recovered_job_bundle_present': JOB.is_file(),
    'recovered_job_bundle_sha256': sha(JOB) if JOB.is_file() else None,
    'tensors': [],
}
for layer, columns, factor in [(15, (1536, 2048), 0), (19, (0, 256), 0), (23, (1024, 1536), .5)]:
    name = f'model.language_model.layers.{layer}.self_attn.o_proj.weight'
    current = SOURCE / 'work/edited' / f'{name}.safetensors'
    archive = SOURCE / 'work/edits_archive/E3' / current.name
    w, dtype, wsha = tensor(current, 'w')
    base, base_dtype, base_sha = tensor(BASE, name)
    expected = base.copy()
    expected[:, columns[0]:columns[1]] *= factor
    item = {
        'name': name, 'shape': list(w.shape), 'columns_half_open': list(columns), 'factor': factor,
        'base_dtype': base_dtype, 'saved_edit_dtype': dtype,
        'matches_described_edit_exactly': bool(np.array_equal(w, expected)),
        'changed_values': int(np.count_nonzero(w != base)),
        'base_native_tensor_sha256': base_sha, 'edit_native_tensor_sha256': wsha,
        'edit_file_sha256': sha(current), 'archive_file_sha256': sha(archive),
        'current_equals_archive_bytes': current.read_bytes() == archive.read_bytes(),
        'edit_values_exactly_bf16_representable': bool(np.all((w.view(np.uint32) & 65535) == 0)),
    }
    if JOB.is_file():
        job, job_dtype, job_sha = tensor(JOB, name)
        item.update(job_dtype=job_dtype, job_native_tensor_sha256=job_sha,
                    job_tensor_equals_current_exactly=bool(np.array_equal(job, w)),
                    job_tensor_native_bytes_equal_current=(job_dtype == dtype and job_sha == wsha))
    receipt['tensors'].append(item)
if JOB.is_file():
    h, _ = header(JOB)
    receipt['job_bundle_tensor_names_exactly_match_local_edits'] = sorted(k for k in h if k != '__metadata__') == sorted(t['name'] for t in receipt['tensors'])
receipt['total_changed_values'] = sum(t['changed_values'] for t in receipt['tensors'])
receipt['all_described_edits_exact'] = all(t['matches_described_edit_exactly'] for t in receipt['tensors'])
receipt['caveat'] = ('A matching submitted job bundle plus worker build code links this edit to the job input. '
                     'There is no independent hash of the fully materialized GPU checkpoint in this audit.')
(REVIEW / 'mechanism-tensor-audit.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
