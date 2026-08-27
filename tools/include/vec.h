/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Growable typed vector used by the Matrix Market loader.
 */

#ifndef VEC_H
#define VEC_H

#include <stddef.h>
#include <stdlib.h>

typedef size_t vec_size_t;

typedef struct {
  void *data;
  vec_size_t size;
  vec_size_t capacity;
  vec_size_t elem_size;
} vec_t;

vec_t *vector_create_int(void);
vec_t *vector_create_float(void);
void vector_add_int(vec_t *v, int val);
void vector_add_float(vec_t *v, float val);
void vector_free(vec_t *v);
vec_size_t vector_size(vec_t *v);

#endif // VEC_H
