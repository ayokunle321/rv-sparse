/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * MAGNUS fine-level chunked accumulation. Keeps the active accumulator slice
 * L2 resident by binning products into fixed column chunks and accumulating
 * one chunk at a time. Structure is built by the shared symbolic phase.
 */

#include "rvsp_common.h"

#define RVSP_KERNEL_NAME rvsp_spgemm_magnus_f32
#include "magnus_core_f32.inc"
