/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Growable typed vector used by the Matrix Market loader.
 */

#include "vec.h"

vec_t *vector_create_int(void) {
  vec_t *v = (vec_t *)malloc(sizeof(vec_t));
  v->data = NULL;
  v->size = 0;
  v->capacity = 0;
  v->elem_size = sizeof(int);
  return v;
}

vec_t *vector_create_float(void) {
  vec_t *v = (vec_t *)malloc(sizeof(vec_t));
  v->data = NULL;
  v->size = 0;
  v->capacity = 0;
  v->elem_size = sizeof(float);
  return v;
}

void vector_add_int(vec_t *v, int val) {
  if (v->size == v->capacity) {
    v->capacity = v->capacity == 0 ? 8 : v->capacity * 2;
    v->data = realloc(v->data, v->capacity * sizeof(int));
  }
  ((int *)v->data)[v->size++] = val;
}

void vector_add_float(vec_t *v, float val) {
  if (v->size == v->capacity) {
    v->capacity = v->capacity == 0 ? 8 : v->capacity * 2;
    v->data = realloc(v->data, v->capacity * sizeof(float));
  }
  ((float *)v->data)[v->size++] = val;
}

void vector_free(vec_t *v) {
  free(v->data);
  free(v);
}

vec_size_t vector_size(vec_t *v) { return v->size; }
