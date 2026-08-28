"""SOMA file-mode payload I/O for standalone (one-shot) disaggregated roles.

Instead of the DiffusionServer-mediated ZMQ/Mooncake transport, a role can hand
off its conditioning to the next role through a ``.bin`` file, so the three
roles (encoder -> denoiser -> decoder) can be chained sequentially in a shell
script, each process dying between stages (freeing its component's memory).

Reuses the battle-tested serialization: ``extract_transfer_fields`` (which Req
fields transfer) + ``pack_tensors``/``unpack_tensors`` (dtype/shape/list-aware,
transport-agnostic). Only the storage is swapped (network -> file). The same
file could live on local disk (sequential, one node) or S3/MinIO (distributed).

Enabled via env vars, mirroring ``SOMA_FREE_TEXT_ENCODER``:
  SOMA_DUMP_PAYLOAD=<path>   dump this role's transfer fields after its stages
  SOMA_LOAD_PAYLOAD=<path>   load a prior role's transfer fields into the Req
"""

from __future__ import annotations

import logging
import os
import struct

logger = logging.getLogger(__name__)

_MAGIC = b"SOMApl01"


def _write_parts(path: str, parts: list[bytes]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<I", len(parts)))
        for p in parts:
            f.write(struct.pack("<Q", len(p)))
            f.write(p)
    os.replace(tmp, path)


def _read_parts(path: str) -> list[bytes]:
    with open(path, "rb") as f:
        if f.read(len(_MAGIC)) != _MAGIC:
            raise ValueError(f"{path}: not a SOMA payload file")
        (n,) = struct.unpack("<I", f.read(4))
        parts: list[bytes] = []
        for _ in range(n):
            (ln,) = struct.unpack("<Q", f.read(8))
            parts.append(f.read(ln))
    return parts


def dump_req_to_file(req, path: str) -> None:
    """Serialize a Req's transfer fields to ``path`` (metadata + tensor buffers)."""
    from sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin import (
        extract_transfer_fields,
    )
    from sglang.multimodal_gen.runtime.disaggregation.transport.codec import (
        pack_tensors,
    )

    tensor_fields, scalar_fields = extract_transfer_fields(req)
    metadata_bytes, buffers = pack_tensors(tensor_fields, scalar_fields)
    parts: list[bytes] = [metadata_bytes]
    for w in buffers:
        view = getattr(w, "_view", w)
        parts.append(bytes(view))
    _write_parts(path, parts)
    logger.info(
        "SOMA dump: wrote %d tensor field(s) + %d scalar(s) to %s",
        len(tensor_fields),
        len(scalar_fields),
        path,
    )


def load_file_into_req(req, path: str, device: str = "cpu") -> None:
    """Load transfer fields from ``path`` into an existing Req in place."""
    from sglang.multimodal_gen.runtime.disaggregation.transport.codec import (
        unpack_tensors,
    )

    import torch

    parts = _read_parts(path)
    tensor_fields, scalar_fields = unpack_tensors(parts, device=device)

    # Restore _extra_* prefixed scalars into req.extra (mirrors the disagg recv).
    extra_keys = [k for k in scalar_fields if k.startswith("_extra_")]
    for key in extra_keys:
        if getattr(req, "extra", None) is None:
            req.extra = {}
        req.extra[key[len("_extra_") :]] = scalar_fields.pop(key)

    # Output-owning fields stay under the consuming role's own CLI (so the
    # decoder writes where --output-file-path says, not where the encoder did).
    _KEEP_LOCAL = {
        "output_file_path",
        "output_file_paths",
        "output_path",
        "output_file_name",
        "save_output",
        "perf_dump_path",
        "output_quality",
        "output_compression",
    }
    for k, v in scalar_fields.items():
        if k in _KEEP_LOCAL:
            continue
        try:
            setattr(req, k, v)
        except Exception:
            pass
    for k, v in tensor_fields.items():
        setattr(req, k, v)

    # torch.Generator is excluded from transfer (not serializable); recreate it
    # from the (transferred) seed, exactly like the disagg receive path.
    seed = scalar_fields.get("seed")
    if seed is not None:
        if isinstance(seed, list):
            req.generator = [
                torch.Generator(device="cpu").manual_seed(int(item)) for item in seed
            ]
        else:
            req.generator = torch.Generator(device="cpu").manual_seed(int(seed))

    logger.info(
        "SOMA load: restored %d tensor field(s) + %d scalar(s) from %s",
        len(tensor_fields),
        len(scalar_fields),
        path,
    )


def _local_device() -> str:
    """The device the pipeline computes on — conditioning tensors produced by the
    encoder are GPU-resident in the monolithic path, so restore them there too."""
    try:
        from sglang.multimodal_gen.runtime.distributed import get_local_torch_device

        return str(get_local_torch_device())
    except Exception:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"


def maybe_load_payload(req) -> None:
    """If SOMA_LOAD_PAYLOAD is set, populate ``req`` from that file (before stages)."""
    path = os.environ.get("SOMA_LOAD_PAYLOAD")
    if path:
        load_file_into_req(req, path, device=_local_device())


def maybe_dump_payload(req) -> None:
    """If SOMA_DUMP_PAYLOAD is set, dump ``req``'s transfer fields (after stages)."""
    path = os.environ.get("SOMA_DUMP_PAYLOAD")
    if path:
        dump_req_to_file(req, path)
