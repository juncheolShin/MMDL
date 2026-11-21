"""Losslessly encode a visual tensor relative to the teacher's BF16 bit patterns."""
import struct
import zlib

import numpy as np


_MAGIC = b"VBD1"
_BLOCK_WORDS = 1 << 20


def _words(data):
    array = np.asarray(data)
    if array.dtype.kind != "u" or array.dtype.itemsize != 2:
        raise ValueError("Expected uint16 BF16 bit patterns")
    return array


def _ordinal(words):
    # Flip positives' sign bit and complement negatives so nearby BF16 values have nearby codes.
    return words ^ np.where(words & 0x8000, np.uint16(0xffff), np.uint16(0x8000))


def _order(base):
    exponent = ((base >> 7) & 255).astype(np.uint8)
    return np.argsort(exponent, kind="stable")


def _encode_block(base, target):
    # Modular arithmetic even preserves unusual patterns such as NaN payloads and signed zeros.
    difference = (_ordinal(target) - _ordinal(base)).astype("<u2", copy=False)
    delta = difference.view("<i2")[_order(base)]
    changed = delta != 0
    nonzero = delta[changed].astype(np.int32)
    # After removing zeros, abs(delta) - 1 fits in 15 bits, including delta == -32768.
    magnitude = (np.abs(nonzero) - 1).astype(np.uint16)
    planes = [np.packbits(changed), np.packbits(nonzero < 0)]
    planes.extend(np.packbits((magnitude >> bit) & 1) for bit in range(15))
    return zlib.compress(b"".join(plane.tobytes() for plane in planes), level=6)


def encode(base, target):
    """Return compressed bytes; both inputs are equally shaped arrays of uint16 bit patterns."""
    base, target = _words(base), _words(target)
    if base.shape != target.shape:
        raise ValueError("Teacher and student tensor shapes differ")
    payload = bytearray(_MAGIC)
    for start in range(0, base.size, _BLOCK_WORDS):
        part = slice(start, start + _BLOCK_WORDS)
        block = _encode_block(base.flat[part].astype("<u2", copy=False), target.flat[part].astype("<u2", copy=False))
        payload.extend(struct.pack("<I", len(block)))
        payload.extend(block)
    return bytes(payload)


def _decode_block(base, block):
    count = len(base)
    bitmap_bytes = (count + 7) // 8
    stream = zlib.decompressobj()
    try:
        raw = stream.decompress(block, 17 * bitmap_bytes + 1)
    except zlib.error as exc:
        raise ValueError("Invalid compressed visual block") from exc
    if not stream.eof or stream.unused_data or len(raw) < bitmap_bytes:
        raise ValueError("Incomplete or trailing compressed visual data")
    buf = np.frombuffer(raw, dtype=np.uint8)
    changed = np.unpackbits(buf[:bitmap_bytes], count=count).astype(bool)
    changed_count = int(changed.sum())
    plane_bytes = (changed_count + 7) // 8
    if len(buf) != bitmap_bytes + 16 * plane_bytes:
        raise ValueError("Incorrect number of bits for visual block")
    planes = buf[bitmap_bytes:].reshape(16, plane_bytes)
    negative = np.unpackbits(planes[0], count=changed_count).astype(bool)
    magnitude = np.zeros(changed_count, dtype="<u2")
    for bit, plane in enumerate(planes[1:]):
        magnitude |= np.unpackbits(plane, count=changed_count).astype("<u2") << bit
    magnitude += 1
    delta = np.zeros(count, dtype="<u2")
    delta[_order(base)[changed]] = np.where(negative, -magnitude, magnitude)
    ordered = _ordinal(base) + delta
    return ordered ^ np.where(ordered & 0x8000, np.uint16(0x8000), np.uint16(0xffff))


def decode(base, payload):
    """Restore uint16 bit patterns in the teacher's shape with bounded scratch space."""
    base = _words(base)
    data = memoryview(payload)
    if data[:len(_MAGIC)] != _MAGIC:
        raise ValueError("Unknown compressed visual tensor format")
    offset = len(_MAGIC)
    result = np.empty(base.size, dtype="<u2")
    for start in range(0, base.size, _BLOCK_WORDS):
        if offset + 4 > len(data):
            raise ValueError("Missing compressed visual block")
        size, = struct.unpack_from("<I", data, offset)
        offset += 4
        if offset + size > len(data):
            raise ValueError("Truncated compressed visual block")
        part = slice(start, start + _BLOCK_WORDS)
        result[part] = _decode_block(base.flat[part].astype("<u2", copy=False), data[offset:offset + size])
        offset += size
    if offset != len(data):
        raise ValueError("Unexpected data after visual tensor")
    return result.reshape(base.shape)
