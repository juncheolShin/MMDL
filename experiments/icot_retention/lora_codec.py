"""Losslessly represent an evaluated BF16 attention matrix by the trained FP32 LoRA factors."""
import struct
import zlib

import numpy as np


RANK = 8
SCALE = np.float32(2)  # alpha=16, rank=8 in recipe.toml
HEADER = struct.Struct('<4sIII')
CORRECTION = np.dtype([('index', '<u4'), ('bits', '<u2')])
ROW_BLOCK = 64


def _matrix(bits):
    bits = np.asarray(bits)
    if bits.ndim != 2 or bits.dtype != np.dtype('<u2') or not all(bits.shape) or bits.size >= 2**32:
        raise ValueError('Expected a nonempty 2-D matrix of little-endian BF16 bits')
    return bits


def _factors(a, b, shape):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != (RANK, shape[1]) or b.shape != (shape[0], RANK):
        raise ValueError('LoRA factor dimensions do not match the attention matrix')
    if a.dtype != np.dtype('<f4') or b.dtype != np.dtype('<f4'):
        raise ValueError('Expected little-endian float32 LoRA factors')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('LoRA factors contain non-finite values')
    return a, b


def _merge(base, a, b):
    result = np.empty(base.shape, dtype='<u2')
    for start in range(0, len(base), ROW_BLOCK):
        rows = slice(start, start + ROW_BLOCK)
        left = b[rows]
        # Eight products in fixed order with a float32 rounding after each ufunc.
        # Separate operations avoid CPU-specific BLAS/FMA choices at restore time.
        delta = left[:, :1] * a[0]
        for k in range(1, RANK):
            delta += left[:, k:k + 1] * a[k]
        delta *= SCALE
        high_bits = base[rows].astype('<u4') << 16
        original = high_bits.astype('<u4', copy=False).view('<f4')
        words = (original + delta).astype('<f4', copy=False).view('<u4')
        # Round finite float32 to the nearest BF16, breaking ties to even.
        words += np.uint32(0x7fff) + ((words >> 16) & np.uint32(1))
        result[rows] = words >> 16
    return result


def encode(base_u16_2d, target_u16_2d, a_f32, b_f32):
    """Encode a single evaluated projection. The returned bytes include factors and exact BF16 corrections."""
    base, target = _matrix(base_u16_2d), _matrix(target_u16_2d)
    if base.shape != target.shape:
        raise ValueError('Source and target matrix dimensions differ')
    a, b = _factors(a_f32, b_f32, base.shape)
    merged = _merge(base, a, b)
    indices = np.flatnonzero((merged != target).ravel())
    if indices.size >= 2**32:
        raise ValueError('Too many LoRA corrections for this format')
    fixes = np.empty(indices.size, dtype=CORRECTION)
    fixes['index'] = indices
    fixes['bits'] = target.ravel()[indices]
    # Group equal-significance bytes of the float32 factors before compression.
    factors = np.concatenate([a.ravel(), b.ravel()]).astype('<f4', copy=False)
    planes = factors.view('u1').reshape(-1, 4).T.tobytes()
    return HEADER.pack(b'ILR1', *base.shape, len(fixes)) + zlib.compress(planes + fixes.tobytes(), level=6)


def decode(base_u16_2d, payload):
    """Reconstruct the original BF16 projection bits from one encoded member."""
    base = _matrix(base_u16_2d)
    if len(payload) < HEADER.size:
        raise ValueError('Truncated LoRA header')
    magic, rows, cols, nfix = HEADER.unpack_from(payload)
    if magic != b'ILR1' or (rows, cols) != base.shape or nfix > base.size:
        raise ValueError('Invalid LoRA header or incompatible source matrix')
    nfactor = RANK * (rows + cols)
    factor_bytes = nfactor * 4
    size = factor_bytes + nfix * CORRECTION.itemsize
    inflater = zlib.decompressobj()
    try:
        raw = inflater.decompress(memoryview(payload)[HEADER.size:], size + 1)
    except zlib.error as error:
        raise ValueError('Invalid LoRA compressed data') from error
    if len(raw) != size or not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
        raise ValueError('LoRA payload length differs from its header')
    planes = np.frombuffer(raw, dtype='u1', count=factor_bytes).reshape(4, nfactor)
    factors = np.ascontiguousarray(planes.T).view('<f4').ravel()
    split = RANK * cols
    a, b = _factors(factors[:split].reshape(RANK, cols), factors[split:].reshape(rows, RANK), base.shape)
    fixes = np.frombuffer(raw, dtype=CORRECTION, count=nfix, offset=factor_bytes)
    indices = fixes['index']
    if nfix and (indices[-1] >= base.size or (indices[:-1] >= indices[1:]).any()):
        raise ValueError('LoRA correction indices are out of bounds or unordered')
    result = _merge(base, a, b)
    result.ravel()[indices] = fixes['bits']
    return result
