#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image

try:
    from numba import jit, prange, float64, uint8, int32

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    prange = range
    float64 = float
    uint8 = int
    int32 = int


@jit(nopython=True, cache=True, fastmath=True)
def srgb_to_linear_fast(c: float) -> float:
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


@jit(nopython=True, cache=True, fastmath=True)
def linear_to_srgb_fast(c: float) -> float:
    if c <= 0.0031308:
        return c * 12.92
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def rgb_to_oklab_batch(rgb_array: np.ndarray) -> np.ndarray:
    n = rgb_array.shape[0]
    lab_array = np.empty((n, 3), dtype=np.float64)

    for i in prange(n):
        r = srgb_to_linear_fast(rgb_array[i, 0] / 255.0)
        g = srgb_to_linear_fast(rgb_array[i, 1] / 255.0)
        b = srgb_to_linear_fast(rgb_array[i, 2] / 255.0)

        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

        l_ = l ** (1.0 / 3.0)
        m_ = m ** (1.0 / 3.0)
        s_ = s ** (1.0 / 3.0)

        lab_array[i, 0] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        lab_array[i, 1] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        lab_array[i, 2] = 0.0259040375 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return lab_array


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def oklab_to_rgb_batch(lab_array: np.ndarray) -> np.ndarray:
    n = lab_array.shape[0]
    rgb_array = np.empty((n, 3), dtype=np.uint8)

    for i in prange(n):
        L = lab_array[i, 0]
        a = lab_array[i, 1]
        b = lab_array[i, 2]

        l_ = L + 0.3963377774 * a + 0.2158037573 * b
        m_ = L - 0.1055613458 * a - 0.0638541728 * b
        s_ = L - 0.0894841775 * a - 1.2914855480 * b

        l = l_**3
        m = m_**3
        s = s_**3

        r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
        g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
        b_val = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

        r = linear_to_srgb_fast(r) * 255.0
        g = linear_to_srgb_fast(g) * 255.0
        b_val = linear_to_srgb_fast(b_val) * 255.0

        rgb_array[i, 0] = max(0, min(255, int(r)))
        rgb_array[i, 1] = max(0, min(255, int(g)))
        rgb_array[i, 2] = max(0, min(255, int(b_val)))

    return rgb_array


@jit(nopython=True, cache=True, fastmath=True)
def find_nearest_color_index_fast(
    pixel_lab: np.ndarray, palette_lab: np.ndarray
) -> int:
    min_dist = 1e20
    min_idx = 0
    for i in range(palette_lab.shape[0]):
        d = (
            (pixel_lab[0] - palette_lab[i, 0]) ** 2
            + (pixel_lab[1] - palette_lab[i, 1]) ** 2
            + (pixel_lab[2] - palette_lab[i, 2]) ** 2
        )
        if d < min_dist:
            min_dist = d
            min_idx = i
    return min_idx


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def find_nearest_colors_batch_fast(
    pixels_lab: np.ndarray, palette_lab: np.ndarray
) -> np.ndarray:
    n = pixels_lab.shape[0]
    indices = np.empty(n, dtype=np.int32)

    for i in prange(n):
        indices[i] = find_nearest_color_index_fast(pixels_lab[i], palette_lab)

    return indices


@jit(nopython=True, cache=True, fastmath=True)
def floyd_steinberg_core(
    image_float: np.ndarray, palette_lab: np.ndarray, palette_rgb: np.ndarray
) -> np.ndarray:
    height, width = image_float.shape[0], image_float.shape[1]
    output = np.empty((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            old_r = image_float[y, x, 0]
            old_g = image_float[y, x, 1]
            old_b = image_float[y, x, 2]

            pixel_lab = np.empty(3, dtype=np.float64)
            r = srgb_to_linear_fast(old_r / 255.0)
            g = srgb_to_linear_fast(old_g / 255.0)
            b = srgb_to_linear_fast(old_b / 255.0)

            l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
            m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
            s_val = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

            l_ = l ** (1.0 / 3.0)
            m_ = m ** (1.0 / 3.0)
            s_ = s_val ** (1.0 / 3.0)

            pixel_lab[0] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
            pixel_lab[1] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
            pixel_lab[2] = 0.0259040375 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

            nearest_idx = find_nearest_color_index_fast(pixel_lab, palette_lab)

            new_r = float(palette_rgb[nearest_idx, 0])
            new_g = float(palette_rgb[nearest_idx, 1])
            new_b = float(palette_rgb[nearest_idx, 2])

            output[y, x, 0] = int(new_r)
            output[y, x, 1] = int(new_g)
            output[y, x, 2] = int(new_b)

            error_r = old_r - new_r
            error_g = old_g - new_g
            error_b = old_b - new_b

            if x + 1 < width:
                image_float[y, x + 1, 0] += error_r * (7.0 / 16.0)
                image_float[y, x + 1, 1] += error_g * (7.0 / 16.0)
                image_float[y, x + 1, 2] += error_b * (7.0 / 16.0)
            if y + 1 < height:
                if x > 0:
                    image_float[y + 1, x - 1, 0] += error_r * (3.0 / 16.0)
                    image_float[y + 1, x - 1, 1] += error_g * (3.0 / 16.0)
                    image_float[y + 1, x - 1, 2] += error_b * (3.0 / 16.0)
                image_float[y + 1, x, 0] += error_r * (5.0 / 16.0)
                image_float[y + 1, x, 1] += error_g * (5.0 / 16.0)
                image_float[y + 1, x, 2] += error_b * (5.0 / 16.0)
                if x + 1 < width:
                    image_float[y + 1, x + 1, 0] += error_r * (1.0 / 16.0)
                    image_float[y + 1, x + 1, 1] += error_g * (1.0 / 16.0)
                    image_float[y + 1, x + 1, 2] += error_b * (1.0 / 16.0)

    return output


@jit(nopython=True, cache=True, fastmath=True)
def atkinson_core(
    image_float: np.ndarray, palette_lab: np.ndarray, palette_rgb: np.ndarray
) -> np.ndarray:
    height, width = image_float.shape[0], image_float.shape[1]
    output = np.empty((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            old_r = image_float[y, x, 0]
            old_g = image_float[y, x, 1]
            old_b = image_float[y, x, 2]

            pixel_lab = np.empty(3, dtype=np.float64)
            r = srgb_to_linear_fast(old_r / 255.0)
            g = srgb_to_linear_fast(old_g / 255.0)
            b = srgb_to_linear_fast(old_b / 255.0)

            l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
            m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
            s_val = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

            l_ = l ** (1.0 / 3.0)
            m_ = m ** (1.0 / 3.0)
            s_ = s_val ** (1.0 / 3.0)

            pixel_lab[0] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
            pixel_lab[1] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
            pixel_lab[2] = 0.0259040375 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

            nearest_idx = find_nearest_color_index_fast(pixel_lab, palette_lab)

            new_r = float(palette_rgb[nearest_idx, 0])
            new_g = float(palette_rgb[nearest_idx, 1])
            new_b = float(palette_rgb[nearest_idx, 2])

            output[y, x, 0] = int(new_r)
            output[y, x, 1] = int(new_g)
            output[y, x, 2] = int(new_b)

            error_r = (old_r - new_r) / 8.0
            error_g = (old_g - new_g) / 8.0
            error_b = (old_b - new_b) / 8.0

            if x + 1 < width:
                image_float[y, x + 1, 0] += error_r
                image_float[y, x + 1, 1] += error_g
                image_float[y, x + 1, 2] += error_b
            if x + 2 < width:
                image_float[y, x + 2, 0] += error_r
                image_float[y, x + 2, 1] += error_g
                image_float[y, x + 2, 2] += error_b
            if y + 1 < height:
                if x > 0:
                    image_float[y + 1, x - 1, 0] += error_r
                    image_float[y + 1, x - 1, 1] += error_g
                    image_float[y + 1, x - 1, 2] += error_b
                image_float[y + 1, x, 0] += error_r
                image_float[y + 1, x, 1] += error_g
                image_float[y + 1, x, 2] += error_b
                if x + 1 < width:
                    image_float[y + 1, x + 1, 0] += error_r
                    image_float[y + 1, x + 1, 1] += error_g
                    image_float[y + 1, x + 1, 2] += error_b
            if y + 2 < height:
                image_float[y + 2, x, 0] += error_r
                image_float[y + 2, x, 1] += error_g
                image_float[y + 2, x, 2] += error_b

    return output


@jit(nopython=True, cache=True, fastmath=True)
def jarvis_core(
    image_float: np.ndarray, palette_lab: np.ndarray, palette_rgb: np.ndarray
) -> np.ndarray:
    height, width = image_float.shape[0], image_float.shape[1]
    output = np.empty((height, width, 3), dtype=np.uint8)

    kernel_dy = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    kernel_dx = [1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2]
    kernel_w = [7, 5, 3, 5, 7, 5, 3, 1, 3, 5, 3, 1]

    for y in range(height):
        for x in range(width):
            old_r = image_float[y, x, 0]
            old_g = image_float[y, x, 1]
            old_b = image_float[y, x, 2]

            pixel_lab = np.empty(3, dtype=np.float64)
            r = srgb_to_linear_fast(old_r / 255.0)
            g = srgb_to_linear_fast(old_g / 255.0)
            b = srgb_to_linear_fast(old_b / 255.0)

            l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
            m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
            s_val = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

            l_ = l ** (1.0 / 3.0)
            m_ = m ** (1.0 / 3.0)
            s_ = s_val ** (1.0 / 3.0)

            pixel_lab[0] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
            pixel_lab[1] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
            pixel_lab[2] = 0.0259040375 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

            nearest_idx = find_nearest_color_index_fast(pixel_lab, palette_lab)

            new_r = float(palette_rgb[nearest_idx, 0])
            new_g = float(palette_rgb[nearest_idx, 1])
            new_b = float(palette_rgb[nearest_idx, 2])

            output[y, x, 0] = int(new_r)
            output[y, x, 1] = int(new_g)
            output[y, x, 2] = int(new_b)

            error_r = (old_r - new_r) / 48.0
            error_g = (old_g - new_g) / 48.0
            error_b = (old_b - new_b) / 48.0

            for k in range(12):
                ny = y + kernel_dy[k]
                nx = x + kernel_dx[k]
                if 0 <= ny < height and 0 <= nx < width:
                    image_float[ny, nx, 0] += error_r * kernel_w[k]
                    image_float[ny, nx, 1] += error_g * kernel_w[k]
                    image_float[ny, nx, 2] += error_b * kernel_w[k]

    return output


@jit(nopython=True, cache=True, fastmath=True)
def stucki_core(
    image_float: np.ndarray, palette_lab: np.ndarray, palette_rgb: np.ndarray
) -> np.ndarray:
    height, width = image_float.shape[0], image_float.shape[1]
    output = np.empty((height, width, 3), dtype=np.uint8)

    kernel_dy = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    kernel_dx = [1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2]
    kernel_w = [8, 4, 2, 4, 8, 4, 2, 1, 2, 4, 2, 1]

    for y in range(height):
        for x in range(width):
            old_r = image_float[y, x, 0]
            old_g = image_float[y, x, 1]
            old_b = image_float[y, x, 2]

            pixel_lab = np.empty(3, dtype=np.float64)
            r = srgb_to_linear_fast(old_r / 255.0)
            g = srgb_to_linear_fast(old_g / 255.0)
            b = srgb_to_linear_fast(old_b / 255.0)

            l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
            m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
            s_val = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

            l_ = l ** (1.0 / 3.0)
            m_ = m ** (1.0 / 3.0)
            s_ = s_val ** (1.0 / 3.0)

            pixel_lab[0] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
            pixel_lab[1] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
            pixel_lab[2] = 0.0259040375 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

            nearest_idx = find_nearest_color_index_fast(pixel_lab, palette_lab)

            new_r = float(palette_rgb[nearest_idx, 0])
            new_g = float(palette_rgb[nearest_idx, 1])
            new_b = float(palette_rgb[nearest_idx, 2])

            output[y, x, 0] = int(new_r)
            output[y, x, 1] = int(new_g)
            output[y, x, 2] = int(new_b)

            error_r = (old_r - new_r) / 42.0
            error_g = (old_g - new_g) / 42.0
            error_b = (old_b - new_b) / 42.0

            for k in range(12):
                ny = y + kernel_dy[k]
                nx = x + kernel_dx[k]
                if 0 <= ny < height and 0 <= nx < width:
                    image_float[ny, nx, 0] += error_r * kernel_w[k]
                    image_float[ny, nx, 1] += error_g * kernel_w[k]
                    image_float[ny, nx, 2] += error_b * kernel_w[k]

    return output


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def bayer_dither_core(
    image_array: np.ndarray,
    palette_lab: np.ndarray,
    palette_rgb: np.ndarray,
    matrix_size: int,
) -> np.ndarray:
    height, width = image_array.shape[0], image_array.shape[1]
    output = np.empty((height, width, 3), dtype=np.uint8)

    if matrix_size == 2:
        bayer_matrix = np.array([[0.0, 0.5], [0.75, 0.25]], dtype=np.float64)
    elif matrix_size == 4:
        bayer_matrix = np.array(
            [
                [0.0, 0.5, 0.125, 0.625],
                [0.75, 0.25, 0.875, 0.375],
                [0.1875, 0.6875, 0.0625, 0.5625],
                [0.9375, 0.4375, 0.8125, 0.3125],
            ],
            dtype=np.float64,
        )
    else:
        bayer_matrix = np.array(
            [
                [0.0, 0.5, 0.125, 0.625],
                [0.75, 0.25, 0.875, 0.375],
                [0.1875, 0.6875, 0.0625, 0.5625],
                [0.9375, 0.4375, 0.8125, 0.3125],
            ],
            dtype=np.float64,
        )

    mat_h, mat_w = bayer_matrix.shape

    for y in prange(height):
        for x in range(width):
            threshold = (bayer_matrix[y % mat_h, x % mat_w] - 0.5) * 128

            old_r = min(255.0, max(0.0, float(image_array[y, x, 0]) + threshold))
            old_g = min(255.0, max(0.0, float(image_array[y, x, 1]) + threshold))
            old_b = min(255.0, max(0.0, float(image_array[y, x, 2]) + threshold))

            pixel_lab = np.empty(3, dtype=np.float64)
            r = srgb_to_linear_fast(old_r / 255.0)
            g = srgb_to_linear_fast(old_g / 255.0)
            b = srgb_to_linear_fast(old_b / 255.0)

            l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
            m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
            s_val = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

            l_ = l ** (1.0 / 3.0)
            m_ = m ** (1.0 / 3.0)
            s_ = s_val ** (1.0 / 3.0)

            pixel_lab[0] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
            pixel_lab[1] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
            pixel_lab[2] = 0.0259040375 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

            nearest_idx = find_nearest_color_index_fast(pixel_lab, palette_lab)

            output[y, x, 0] = palette_rgb[nearest_idx, 0]
            output[y, x, 1] = palette_rgb[nearest_idx, 1]
            output[y, x, 2] = palette_rgb[nearest_idx, 2]

    return output


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def no_dither_core(
    image_array: np.ndarray, palette_lab: np.ndarray, palette_rgb: np.ndarray
) -> np.ndarray:
    height, width = image_array.shape[0], image_array.shape[1]
    output = np.empty((height, width, 3), dtype=np.uint8)

    for y in prange(height):
        for x in range(width):
            old_r = float(image_array[y, x, 0])
            old_g = float(image_array[y, x, 1])
            old_b = float(image_array[y, x, 2])

            pixel_lab = np.empty(3, dtype=np.float64)
            r = srgb_to_linear_fast(old_r / 255.0)
            g = srgb_to_linear_fast(old_g / 255.0)
            b = srgb_to_linear_fast(old_b / 255.0)

            l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
            m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
            s_val = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

            l_ = l ** (1.0 / 3.0)
            m_ = m ** (1.0 / 3.0)
            s_ = s_val ** (1.0 / 3.0)

            pixel_lab[0] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
            pixel_lab[1] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
            pixel_lab[2] = 0.0259040375 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

            nearest_idx = find_nearest_color_index_fast(pixel_lab, palette_lab)

            output[y, x, 0] = palette_rgb[nearest_idx, 0]
            output[y, x, 1] = palette_rgb[nearest_idx, 1]
            output[y, x, 2] = palette_rgb[nearest_idx, 2]

    return output


PALETTES = {
    "gameboy": ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"],
    "gameboy-pocket": ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"],
    "gameboy-light": ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"],
    "nes": [
        "#000000",
        "#1D2B53",
        "#7E2553",
        "#008751",
        "#AB5236",
        "#5F574F",
        "#C2C3C7",
        "#FFF1E8",
        "#FF004D",
        "#FFA300",
        "#FFEC27",
        "#00E436",
        "#29ADFF",
        "#83769C",
        "#FF77A8",
        "#FFCCAA",
    ],
    "pico8": [
        "#000000",
        "#1D2B53",
        "#7E2553",
        "#008751",
        "#AB5236",
        "#5F574F",
        "#C2C3C7",
        "#FFF1E8",
        "#FF004D",
        "#FFA300",
        "#FFEC27",
        "#00E436",
        "#29ADFF",
        "#83769C",
        "#FF77A8",
        "#FFCCAA",
    ],
    "cga": [
        "#000000",
        "#0000AA",
        "#00AA00",
        "#00AAAA",
        "#AA0000",
        "#AA00AA",
        "#AA5500",
        "#AAAAAA",
        "#555555",
        "#5555FF",
        "#55FF55",
        "#55FFFF",
        "#FF5555",
        "#FF55FF",
        "#FFFF55",
        "#FFFFFF",
    ],
    "windows-16": [
        "#000000",
        "#000080",
        "#008000",
        "#008080",
        "#800000",
        "#800080",
        "#808000",
        "#C0C0C0",
        "#808080",
        "#0000FF",
        "#00FF00",
        "#00FFFF",
        "#FF0000",
        "#FF00FF",
        "#FFFF00",
        "#FFFFFF",
    ],
    "windows-20": [
        "#000000",
        "#000080",
        "#008000",
        "#008080",
        "#800000",
        "#800080",
        "#808000",
        "#C0C0C0",
        "#808080",
        "#0000FF",
        "#00FF00",
        "#00FFFF",
        "#FF0000",
        "#FF00FF",
        "#FFFF00",
        "#FFFFFF",
        "#000000",
        "#000080",
        "#008080",
        "#FFFFFF",
    ],
    "solarized": [
        "#002b36",
        "#073642",
        "#586e75",
        "#657b83",
        "#839496",
        "#93a1a1",
        "#eee8d5",
        "#fdf6e3",
        "#b58900",
        "#cb4b16",
        "#dc322f",
        "#d33682",
        "#6c71c4",
        "#268bd2",
        "#2aa198",
        "#859900",
    ],
    "solarized-dark": [
        "#002b36",
        "#073642",
        "#586e75",
        "#657b83",
        "#839496",
        "#93a1a1",
        "#eee8d5",
        "#fdf6e3",
        "#b58900",
        "#cb4b16",
        "#dc322f",
        "#d33682",
        "#6c71c4",
        "#268bd2",
        "#2aa198",
        "#859900",
    ],
    "solarized-light": [
        "#fdf6e3",
        "#eee8d5",
        "#93a1a1",
        "#839496",
        "#657b83",
        "#586e75",
        "#073642",
        "#002b36",
        "#b58900",
        "#cb4b16",
        "#dc322f",
        "#d33682",
        "#6c71c4",
        "#268bd2",
        "#2aa198",
        "#859900",
    ],
    "grayscale-4": ["#000000", "#555555", "#AAAAAA", "#FFFFFF"],
    "grayscale-8": [
        "#000000",
        "#242424",
        "#494949",
        "#6D6D6D",
        "#929292",
        "#B6B6B6",
        "#DBDBDB",
        "#FFFFFF",
    ],
    "grayscale-16": [
        "#000000",
        "#111111",
        "#222222",
        "#333333",
        "#444444",
        "#555555",
        "#666666",
        "#777777",
        "#888888",
        "#999999",
        "#AAAAAA",
        "#BBBBBB",
        "#CCCCCC",
        "#DDDDDD",
        "#EEEEEE",
        "#FFFFFF",
    ],
    "monokai": [
        "#272822",
        "#F92672",
        "#66D9EF",
        "#A6E22E",
        "#FD971F",
        "#75715E",
        "#F8F8F2",
        "#AE81FF",
    ],
    "nord": [
        "#2E3440",
        "#3B4252",
        "#434C5E",
        "#4C566A",
        "#D8DEE9",
        "#E5E9F0",
        "#ECEFF4",
        "#8FBCBB",
        "#88C0D0",
        "#81A1C1",
        "#5E81AC",
        "#BF616A",
        "#D08770",
        "#EBCB8B",
        "#A3BE8C",
        "#B48EAD",
    ],
    "dracula": [
        "#282A36",
        "#44475A",
        "#F8F8F2",
        "#6272A4",
        "#FF5555",
        "#FF79C6",
        "#8BE9FD",
        "#50FA7B",
        "#F1FA8C",
        "#BD93F9",
        "#FFB86C",
    ],
    "catppuccin-mocha": [
        "#1E1E2E",
        "#181825",
        "#313244",
        "#45475A",
        "#585B70",
        "#6C7086",
        "#7F849C",
        "#9399B2",
        "#A6ADC8",
        "#BAC2DE",
        "#CDD6F4",
        "#F5E0DC",
        "#F2CDCD",
        "#F5C2E7",
        "#CBA6F7",
        "#89B4FA",
        "#94E2D5",
        "#A6E3A1",
        "#F9E2AF",
        "#FAB387",
        "#F38BA8",
        "#EBB9A8",
        "#EBA0AC",
    ],
    "tokyo-night": [
        "#15161E",
        "#1A1B26",
        "#192330",
        "#292E42",
        "#393552",
        "#3D59A1",
        "#515C9E",
        "#565F89",
        "#787C9E",
        "#9AA5CE",
        "#A9B1D6",
        "#B4F9F8",
        "#C0CAF5",
        "#CFC9C2",
        "#D5D6E3",
        "#E9B8E7",
        "#F7768E",
        "#FF9E64",
        "#FFC777",
        "#BB9AF7",
        "#7DCFFF",
        "#7AA2F7",
        "#9ECE6A",
        "#73DACA",
    ],
    "1bit": ["#000000", "#FFFFFF"],
    "1bit-light": ["#FFFFFF", "#000000"],
    "amstrad-cpc": [
        "#000000",
        "#000080",
        "#0000FF",
        "#800000",
        "#800080",
        "#8000FF",
        "#FF0000",
        "#FF0080",
        "#FF00FF",
        "#008000",
        "#008080",
        "#0080FF",
        "#808000",
        "#808080",
        "#8080FF",
        "#FF8000",
        "#FF8080",
        "#FF80FF",
        "#00FF00",
        "#00FF80",
        "#00FFFF",
        "#80FF00",
        "#80FF80",
        "#80FFFF",
        "#FFFF00",
        "#FFFF80",
        "#FFFFFF",
    ],
    "zx-spectrum": [
        "#000000",
        "#0000CD",
        "#CD0000",
        "#CD00CD",
        "#00CD00",
        "#00CDCD",
        "#CDCD00",
        "#CDCDCD",
        "#000000",
        "#0000FF",
        "#FF0000",
        "#FF00FF",
        "#00FF00",
        "#00FFFF",
        "#FFFF00",
        "#FFFFFF",
    ],
    "apple-ii": [
        "#000000",
        "#00FF00",
        "#FF0000",
        "#FFFF00",
        "#0000FF",
        "#00FFFF",
        "#FF00FF",
        "#FFFFFF",
    ],
    "ega": [
        "#000000",
        "#0000AA",
        "#00AA00",
        "#00AAAA",
        "#AA0000",
        "#AA00AA",
        "#AA5500",
        "#AAAAAA",
        "#555555",
        "#5555FF",
        "#55FF55",
        "#55FFFF",
        "#FF5555",
        "#FF55FF",
        "#FFFF55",
        "#FFFFFF",
    ],
    "vga": [
        "#000000",
        "#0000AA",
        "#00AA00",
        "#00AAAA",
        "#AA0000",
        "#AA00AA",
        "#AA5500",
        "#AAAAAA",
        "#555555",
        "#5555FF",
        "#55FF55",
        "#55FFFF",
        "#FF5555",
        "#FF55FF",
        "#FFFF55",
        "#FFFFFF",
        "#000000",
        "#000080",
        "#008000",
        "#008080",
        "#800000",
        "#800080",
        "#808000",
        "#C0C0C0",
        "#808080",
        "#0000FF",
        "#00FF00",
        "#00FFFF",
        "#FF0000",
        "#FF00FF",
        "#FFFF00",
        "#FFFFFF",
    ],
    "web-safe": [
        "#000000",
        "#000033",
        "#000066",
        "#000099",
        "#0000CC",
        "#0000FF",
        "#003300",
        "#003333",
        "#003366",
        "#003399",
        "#0033CC",
        "#0033FF",
        "#006600",
        "#006633",
        "#006666",
        "#006699",
        "#0066CC",
        "#0066FF",
        "#009900",
        "#009933",
        "#009966",
        "#009999",
        "#0099CC",
        "#0099FF",
        "#00CC00",
        "#00CC33",
        "#00CC66",
        "#00CC99",
        "#00CCCC",
        "#00CCFF",
        "#00FF00",
        "#00FF33",
        "#00FF66",
        "#00FF99",
        "#00FFCC",
        "#00FFFF",
        "#330000",
        "#330033",
        "#330066",
        "#330099",
        "#3300CC",
        "#3300FF",
        "#333300",
        "#333333",
        "#333366",
        "#333399",
        "#3333CC",
        "#3333FF",
        "#336600",
        "#336633",
        "#336666",
        "#336699",
        "#3366CC",
        "#3366FF",
        "#339900",
        "#339933",
        "#339966",
        "#339999",
        "#3399CC",
        "#3399FF",
        "#33CC00",
        "#33CC33",
        "#33CC66",
        "#33CC99",
        "#33CCCC",
        "#33CCFF",
        "#33FF00",
        "#33FF33",
        "#33FF66",
        "#33FF99",
        "#33FFCC",
        "#33FFFF",
        "#660000",
        "#660033",
        "#660066",
        "#660099",
        "#6600CC",
        "#6600FF",
        "#663300",
        "#663333",
        "#663366",
        "#663399",
        "#6633CC",
        "#6633FF",
        "#666600",
        "#666633",
        "#666666",
        "#666699",
        "#6666CC",
        "#6666FF",
        "#669900",
        "#669933",
        "#669966",
        "#669999",
        "#6699CC",
        "#6699FF",
        "#66CC00",
        "#66CC33",
        "#66CC66",
        "#66CC99",
        "#66CCCC",
        "#66CCFF",
        "#66FF00",
        "#66FF33",
        "#66FF66",
        "#66FF99",
        "#66FFCC",
        "#66FFFF",
        "#990000",
        "#990033",
        "#990066",
        "#990099",
        "#9900CC",
        "#9900FF",
        "#993300",
        "#993333",
        "#993366",
        "#993399",
        "#9933CC",
        "#9933FF",
        "#996600",
        "#996633",
        "#996666",
        "#996699",
        "#9966CC",
        "#9966FF",
        "#999900",
        "#999933",
        "#999966",
        "#999999",
        "#9999CC",
        "#9999FF",
        "#99CC00",
        "#99CC33",
        "#99CC66",
        "#99CC99",
        "#99CCCC",
        "#99CCFF",
        "#99FF00",
        "#99FF33",
        "#99FF66",
        "#99FF99",
        "#99FFCC",
        "#99FFFF",
        "#CC0000",
        "#CC0033",
        "#CC0066",
        "#CC0099",
        "#CC00CC",
        "#CC00FF",
        "#CC3300",
        "#CC3333",
        "#CC3366",
        "#CC3399",
        "#CC33CC",
        "#CC33FF",
        "#CC6600",
        "#CC6633",
        "#CC6666",
        "#CC6699",
        "#CC66CC",
        "#CC66FF",
        "#CC9900",
        "#CC9933",
        "#CC9966",
        "#CC9999",
        "#CC99CC",
        "#CC99FF",
        "#CCCC00",
        "#CCCC33",
        "#CCCC66",
        "#CCCC99",
        "#CCCCCC",
        "#CCCCFF",
        "#CCFF00",
        "#CCFF33",
        "#CCFF66",
        "#CCFF99",
        "#CCFFCC",
        "#CCFFFF",
        "#FF0000",
        "#FF0033",
        "#FF0066",
        "#FF0099",
        "#FF00CC",
        "#FF00FF",
        "#FF3300",
        "#FF3333",
        "#FF3366",
        "#FF3399",
        "#FF33CC",
        "#FF33FF",
        "#FF6600",
        "#FF6633",
        "#FF6666",
        "#FF6699",
        "#FF66CC",
        "#FF66FF",
        "#FF9900",
        "#FF9933",
        "#FF9966",
        "#FF9999",
        "#FF99CC",
        "#FF99FF",
        "#FFCC00",
        "#FFCC33",
        "#FFCC66",
        "#FFCC99",
        "#FFCCCC",
        "#FFCCFF",
        "#FFFF00",
        "#FFFF33",
        "#FFFF66",
        "#FFFF99",
        "#FFFFCC",
        "#FFFFFF",
    ],
    "commodore-64": [
        "#000000",
        "#FFFFFF",
        "#880000",
        "#AAFFEE",
        "#CC44CC",
        "#00CC55",
        "#0000AA",
        "#EEEE77",
        "#DD8855",
        "#664400",
        "#FF7777",
        "#333333",
        "#777777",
        "#AAFF66",
        "#0088FF",
        "#BBBBBB",
    ],
    "xterm-16": [
        "#000000",
        "#800000",
        "#008000",
        "#808000",
        "#000080",
        "#800080",
        "#008080",
        "#C0C0C0",
        "#808080",
        "#FF0000",
        "#00FF00",
        "#FFFF00",
        "#0000FF",
        "#FF00FF",
        "#00FFFF",
        "#FFFFFF",
    ],
    "xterm-88": [
        "#000000",
        "#00008B",
        "#008B00",
        "#008B8B",
        "#8B0000",
        "#8B008B",
        "#8B8B00",
        "#D3D3D3",
        "#8B8B8B",
        "#0000FF",
        "#00FF00",
        "#00FFFF",
        "#FF0000",
        "#FF00FF",
        "#FFFF00",
        "#FFFFFF",
        "#000000",
        "#00005F",
        "#000087",
        "#0000AF",
        "#0000D7",
        "#0000FF",
        "#005F00",
        "#005F5F",
        "#005F87",
        "#005FAF",
        "#005FD7",
        "#005FFF",
        "#008700",
        "#00875F",
        "#008787",
        "#0087AF",
        "#0087D7",
        "#00AF00",
        "#00AF5F",
        "#00AF87",
        "#00AFAF",
        "#00AFD7",
        "#00AFFF",
        "#00D700",
        "#00D75F",
        "#00D787",
        "#00D7AF",
        "#00D7D7",
        "#00D7FF",
        "#00FF00",
        "#00FF5F",
        "#00FF87",
        "#00FFAF",
        "#00FFD7",
        "#00FFFF",
        "#5F0000",
        "#5F005F",
        "#5F0087",
        "#5F00AF",
        "#5F00D7",
        "#5F00FF",
        "#5F5F00",
        "#5F5F5F",
        "#5F5F87",
        "#5F5FAF",
        "#5F5FD7",
        "#5F5FFF",
        "#5F8700",
        "#5F875F",
        "#5F8787",
        "#5F87AF",
        "#5F87D7",
        "#5F87FF",
        "#5FAF00",
        "#5FAF5F",
        "#5FAF87",
        "#5FAFAF",
        "#5FAFD7",
        "#5FAFFF",
        "#5FD700",
        "#5FD75F",
        "#5FD787",
        "#5FD7AF",
        "#5FD7D7",
        "#5FD7FF",
        "#5FFF00",
        "#5FFF5F",
        "#5FFF87",
        "#5FFFAF",
        "#5FFFD7",
        "#5FFFFF",
        "#870000",
        "#87005F",
        "#870087",
        "#8700AF",
        "#8700D7",
        "#8700FF",
        "#875F00",
        "#875F5F",
        "#875F87",
    ],
}


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(
            f"Invalid hex color '{hex_color}'. Must be 6 characters (e.g., FF0000 or #FF0000)"
        )

    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except ValueError:
        raise ValueError(
            f"Invalid hex color '{hex_color}'. Contains non-hexadecimal characters."
        )

    return (r, g, b)


def validate_hex_color(hex_color: str) -> str:
    hex_color = hex_color.strip()
    if not hex_color.startswith("#"):
        hex_color = "#" + hex_color

    if len(hex_color) != 7:
        raise ValueError(f"Invalid hex color '{hex_color}'. Must be in format #RRGGBB")

    try:
        int(hex_color[1:], 16)
    except ValueError:
        raise ValueError(
            f"Invalid hex color '{hex_color}'. Contains non-hexadecimal characters."
        )

    return hex_color.upper()


def parse_palette(palette_input: str) -> List[Tuple[int, int, int]]:
    palette_input = palette_input.strip()

    if palette_input.lower() in PALETTES:
        hex_colors = PALETTES[palette_input.lower()]
        return [hex_to_rgb(c) for c in hex_colors]

    if "," in palette_input:
        hex_colors = [c.strip() for c in palette_input.split(",")]
    else:
        path = Path(palette_input)
        if not path.exists():
            raise FileNotFoundError(f"Palette file '{palette_input}' not found")

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in palette file: {e}")

        if not isinstance(data, list):
            raise ValueError("Palette JSON must be an array of hex colors")

        hex_colors = [str(c).strip() for c in data]

    validated_colors = []
    for i, color in enumerate(hex_colors):
        try:
            validated = validate_hex_color(color)
            validated_colors.append(validated)
        except ValueError as e:
            raise ValueError(f"Color #{i + 1}: {e}")

    if len(validated_colors) < 2:
        raise ValueError("Palette must contain at least 2 colors")

    return [hex_to_rgb(c) for c in validated_colors]


def build_palette_arrays(
    palette: List[Tuple[int, int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    palette_rgb = np.array(palette, dtype=np.uint8)
    palette_lab = rgb_to_oklab_batch(palette_rgb)
    return palette_lab, palette_rgb


def floyd_steinberg_dither(
    image_array: np.ndarray, palette_lab: np.ndarray, palette_rgb: np.ndarray
) -> np.ndarray:
    image_float = image_array.astype(np.float64)
    return floyd_steinberg_core(image_float, palette_lab, palette_rgb)


def atkinson_dither(
    image_array: np.ndarray, palette_lab: np.ndarray, palette_rgb: np.ndarray
) -> np.ndarray:
    image_float = image_array.astype(np.float64)
    return atkinson_core(image_float, palette_lab, palette_rgb)


def jarvis_dither(
    image_array: np.ndarray, palette_lab: np.ndarray, palette_rgb: np.ndarray
) -> np.ndarray:
    image_float = image_array.astype(np.float64)
    return jarvis_core(image_float, palette_lab, palette_rgb)


def stucki_dither(
    image_array: np.ndarray, palette_lab: np.ndarray, palette_rgb: np.ndarray
) -> np.ndarray:
    image_float = image_array.astype(np.float64)
    return stucki_core(image_float, palette_lab, palette_rgb)


def bayer_dither(
    image_array: np.ndarray,
    palette_lab: np.ndarray,
    palette_rgb: np.ndarray,
    matrix_size: int = 4,
) -> np.ndarray:
    return bayer_dither_core(image_array, palette_lab, palette_rgb, matrix_size)


def no_dither(
    image_array: np.ndarray, palette_lab: np.ndarray, palette_rgb: np.ndarray
) -> np.ndarray:
    return no_dither_core(image_array, palette_lab, palette_rgb)


def process_tile_bayer(args: Tuple) -> np.ndarray:
    tile, palette_lab, palette_rgb, matrix_size, y_offset = args
    result = bayer_dither_core(tile, palette_lab, palette_rgb, matrix_size)
    return y_offset, result


def process_tile_no_dither(args: Tuple) -> np.ndarray:
    tile, palette_lab, palette_rgb, y_offset = args
    result = no_dither_core(tile, palette_lab, palette_rgb)
    return y_offset, result


def posterize_image(
    image_path: str,
    output_path: str,
    output_format: str,
    dither_method: str,
    palette: List[Tuple[int, int, int]],
) -> None:
    try:
        img = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Failed to open image '{image_path}': {e}")

    if img.mode not in ("RGB", "RGBA"):
        if img.mode == "P":
            img = img.convert("RGBA")
        elif img.mode == "L":
            img = img.convert("RGB")
        else:
            img = img.convert("RGB")

    has_alpha = img.mode == "RGBA"
    alpha_channel = None

    if has_alpha:
        image_array = np.array(img)
        rgb_array = image_array[:, :, :3]
        alpha_channel = image_array[:, :, 3].copy()
    else:
        rgb_array = np.array(img)

    palette_lab, palette_rgb = build_palette_arrays(palette)

    can_parallelize = dither_method in ("bayer2x2", "bayer4x4", "bayer8x8", "none")
    use_mp = can_parallelize and cpu_count() > 1

    if use_mp:
        height = rgb_array.shape[0]
        n_workers = min(cpu_count(), max(1, height // 100))

        if n_workers > 1:
            tile_height = height // n_workers
            tiles = []
            for i in range(n_workers):
                y_start = i * tile_height
                y_end = height if i == n_workers - 1 else (i + 1) * tile_height
                tile = rgb_array[y_start:y_end]
                tiles.append(
                    (
                        tile,
                        palette_lab,
                        palette_rgb,
                        2
                        if dither_method == "bayer2x2"
                        else (8 if dither_method == "bayer8x8" else 4),
                        y_start,
                    )
                    if "bayer" in dither_method
                    else (tile, palette_lab, palette_rgb, y_start)
                )

            with Pool(n_workers) as pool:
                if "bayer" in dither_method:
                    results = pool.map(process_tile_bayer, tiles)
                else:
                    results = pool.map(process_tile_no_dither, tiles)

            results.sort(key=lambda x: x[0])
            result_rgb = np.vstack([r[1] for r in results])
        else:
            if dither_method == "none":
                result_rgb = no_dither(rgb_array, palette_lab, palette_rgb)
            else:
                matrix_size = (
                    2
                    if dither_method == "bayer2x2"
                    else (8 if dither_method == "bayer8x8" else 4)
                )
                result_rgb = bayer_dither(
                    rgb_array, palette_lab, palette_rgb, matrix_size
                )
    else:
        dither_methods = {
            "floyd-steinberg": lambda arr: floyd_steinberg_dither(
                arr, palette_lab, palette_rgb
            ),
            "atkinson": lambda arr: atkinson_dither(arr, palette_lab, palette_rgb),
            "jarvis": lambda arr: jarvis_dither(arr, palette_lab, palette_rgb),
            "stucki": lambda arr: stucki_dither(arr, palette_lab, palette_rgb),
            "bayer2x2": lambda arr: bayer_dither(arr, palette_lab, palette_rgb, 2),
            "bayer4x4": lambda arr: bayer_dither(arr, palette_lab, palette_rgb, 4),
            "bayer8x8": lambda arr: bayer_dither(arr, palette_lab, palette_rgb, 8),
            "none": lambda arr: no_dither(arr, palette_lab, palette_rgb),
        }

        if dither_method not in dither_methods:
            raise ValueError(
                f"Unknown dithering method '{dither_method}'. Available: {', '.join(dither_methods.keys())}"
            )

        result_rgb = dither_methods[dither_method](rgb_array)

    if has_alpha:
        result_array = np.dstack([result_rgb, alpha_channel])
        result_img = Image.fromarray(result_array, mode="RGBA")
    else:
        result_img = Image.fromarray(result_rgb, mode="RGB")

    format_map = {
        "PNG": "PNG",
        "JPEG": "JPEG",
        "JPG": "JPEG",
        "WEBP": "WEBP",
        "GIF": "GIF",
        "BMP": "BMP",
    }

    pil_format = format_map.get(output_format.upper())
    if not pil_format:
        raise ValueError(
            f"Unsupported output format '{output_format}'. Supported: {', '.join(format_map.keys())}"
        )

    if pil_format == "JPEG" and has_alpha:
        result_img = result_img.convert("RGB")

    output_dir = Path(output_path).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result_img.save(output_path, format=pil_format)
    except Exception as e:
        raise ValueError(f"Failed to save image to '{output_path}': {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Image posterization and dithering CLI utility (optimized with Numba)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s photo.jpg
  %(prog)s photo.jpg -o output.png -d atkinson
  %(prog)s photo.jpg -p "#FF0000,#00FF00,#0000FF" -d bayer4x4
  %(prog)s photo.jpg -p gameboy -d floyd-steinberg
  %(prog)s photo.jpg -p palette.json -f WEBP

Optimizations:
  - Numba JIT compilation for 10-50x speedup
  - Multiprocessing for Bayer and no-dither modes
  - Vectorized OKLab color space conversions

Available Palettes:
  gameboy, nes, pico8, cga, windows-16, solarized, grayscale-4,
  grayscale-8, grayscale-16, monokai, nord, dracula, catppuccin-mocha,
  tokyo-night, 1bit, amstrad-cpc, zx-spectrum, apple-ii, ega, vga,
  web-safe, commodore-64, xterm-16, xterm-88
        """,
    )

    parser.add_argument("image", help="Input image file path")

    parser.add_argument(
        "-o", "--output", help="Output file path (default: <input_name>_posterized.png)"
    )

    parser.add_argument(
        "-f",
        "--format",
        choices=["PNG", "JPEG", "JPG", "WEBP", "GIF", "BMP"],
        help="Output format (default: same as input or PNG)",
    )

    parser.add_argument(
        "-d",
        "--dither",
        default="floyd-steinberg",
        choices=[
            "floyd-steinberg",
            "atkinson",
            "jarvis",
            "stucki",
            "bayer2x2",
            "bayer4x4",
            "bayer8x8",
            "none",
        ],
        help="Dithering method (default: floyd-steinberg)",
    )

    parser.add_argument(
        "-p",
        "--palette",
        default="pico8",
        help="Color palette: preset name, comma-separated hex values, or JSON file (default: pico8)",
    )

    args = parser.parse_args()

    input_path = Path(args.image)
    if not input_path.exists():
        print(f"Error: Input image '{args.image}' not found", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.with_name(input_path.stem + "_posterized.png"))

    if args.format:
        output_format = args.format
    else:
        ext = Path(output_path).suffix.lstrip(".").upper()
        output_format = (
            ext if ext in ["PNG", "JPEG", "JPG", "WEBP", "GIF", "BMP"] else "PNG"
        )

    try:
        palette = parse_palette(args.palette)
        print(f"Using palette with {len(palette)} colors")
    except Exception as e:
        print(f"Error parsing palette: {e}", file=sys.stderr)
        sys.exit(1)

    if NUMBA_AVAILABLE:
        print("Numba JIT acceleration enabled")
    else:
        print("Warning: Numba not available, using pure Python (slower)")

    try:
        posterize_image(
            args.image,
            output_path,
            output_format,
            args.dither,
            palette,
        )
        print(f"Successfully saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cli_entry_point():
    main()


if __name__ == "__main__":
    main()
