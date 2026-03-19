/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
/**
 * @file cir_zmq.c
 * @brief CIR ZMQ plugin -- exposes a ZMQ REQ/REP interface for runtime
 *        channel impulse response (CIR) updates.
 *
 * Protocol (JSON over ZMQ REQ/REP on tcp:// *:5555):
 *
 *   Request: {"msg_type": "config_req"}
 *   Reply:   {"msg_type": "config_res", "num_taps": N, "fft_size": N,
 *             "subcarrier_spacing": F, "frequency": F,
 *             "num_ofdm_symbols_per_slot": N}
 *
 *   Request: {"msg_type": "cir",
 *             "sigma_scaling": float,
 *             "sigma_max": float,
 *             "norms": [float x S],
 *             "taps": [float x S*2*T],
 *             "tap_indices": [int x S*T]}
 *   Reply:   {"msg_type": "cir_ack"}
 *
 *   Request: {"msg_type": "nrx", "enabled": true|false}
 *   Reply:   {"msg_type": "nrx_ack", "enabled": 0|1}
 *
 *   Error:   {"msg_type": "error", "error": "...", "details": "..."}
 *
 * Internal storage uses clean flat arrays:
 *   norms_array[S], taps_array[S*2*T], tap_indices_array[S*T]
 *
 * cir_zmq_read() packs these into the per-symbol binary layout expected
 * by the CUDA channel emulator:
 *   [float norm | float taps[2*T] | uint16_t tap_indices[T] | pad to 4B]
 */

#ifndef ENABLE_NANOBIND
#include "openair1/PHY/TOOLS/tools_defs.h"
#include "openair1/PHY/sse_intrin.h"
#endif
#include <zmq.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <time.h>
#include <cjson/cJSON.h>
#include "cir_zmq.h"

/* ------------------------------------------------------------------ */
/*  Static state                                                      */
/* ------------------------------------------------------------------ */

// Configuration parameters
static int num_taps = 0;
static int num_symbols = 0;   // num_ofdm_symbols_per_slot
static int fft_size = 0;
static float subcarrier_spacing = 0.0f;
static float frequency = 0.0f;

static int custom_receiver_enabled = 0;

// ZMQ socket, context, and buffer (volatile so ZMQ thread sees close from main)
static void *zmq_context = NULL;
static void * volatile rep_socket = NULL;
// Buffer must hold full CIR JSON: norms (S) + taps (S*2*T) + tap_indices (S*T) numbers.
// For 256 taps, 14 symbols: ~(14 + 14*2*256 + 14*256) * ~12 chars ≈ 110 KB. Use 128 KB.
static char msg_buffer[128 * 1024];
static const size_t msg_buffer_size = sizeof(msg_buffer);

// Thread that runs cir_zmq_run() to accept ZMQ requests
static pthread_t zmq_thread;
static int zmq_thread_created = 0;
static volatile int zmq_thread_idle_in_recv = 0;  // 1 when blocked in zmq_recv

// CIR internal flat arrays (protected by mutex)
static float sigma_scaling_val = 1.0f;
static float sigma_max_val = 1.0f;
static float *norms_array = NULL;       // [S]
static float *taps_array = NULL;        // [S * 2*T]
static uint16_t *tap_indices_array = NULL; // [S * T]

// Array sizes
static size_t norms_len = 0;            // S
static size_t taps_len = 0;             // S * 2*T
static size_t tap_indices_len = 0;      // S * T

// Packed output buffer for cir_zmq_read()
static size_t cir_entry_size_bytes = 0;
static size_t packed_total_bytes = 0;
static uint8_t *packed_cir_out = NULL;

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

// Helper function to send error response
static void send_error_response(const char *error_msg, const char *details) {
    cJSON *error_response = cJSON_CreateObject();
    cJSON_AddStringToObject(error_response, "msg_type", "error");
    cJSON_AddStringToObject(error_response, "error", error_msg);
    if (details) {
        cJSON_AddStringToObject(error_response, "details", details);
    }
    char *error_json = cJSON_Print(error_response);
    cJSON_Delete(error_response);
    zmq_send(rep_socket, error_json, strlen(error_json), 0);
    free(error_json);
}

// Function to handle unknown message types and send error response
static int handle_unknown_message(const char *type_str) {
    send_error_response("Unknown message type", type_str);
    return -1;
}

// Function to handle config request and send config response
static int handle_config_request(void) {
    cJSON *response = cJSON_CreateObject();
    cJSON_AddStringToObject(response, "msg_type", "config_res");
    cJSON_AddNumberToObject(response, "num_taps", num_taps);
    cJSON_AddNumberToObject(response, "num_ofdm_symbols_per_slot", num_symbols);
    cJSON_AddNumberToObject(response, "fft_size", fft_size);
    cJSON_AddNumberToObject(response, "subcarrier_spacing", subcarrier_spacing);
    cJSON_AddNumberToObject(response, "frequency", frequency);
    char *json_string = cJSON_Print(response);
    cJSON_Delete(response);
    zmq_send(rep_socket, json_string, strlen(json_string), 0);
    free(json_string);
    return 0;
}

static int handle_nrx_message(cJSON *json) {
    cJSON *nrx_enabled = cJSON_GetObjectItem(json, "enabled");

    if (!cJSON_IsBool(nrx_enabled)) {
        fprintf(stderr, "Invalid NRX enabled field\n");
        send_error_response("Invalid NRX enabled field", "Missing or invalid enabled field");
        return -1;
    }
    pthread_mutex_lock(&mutex);
    custom_receiver_enabled = cJSON_IsTrue(nrx_enabled);
    pthread_mutex_unlock(&mutex);

    // Send acknowledgment
    cJSON *response = cJSON_CreateObject();
    cJSON_AddStringToObject(response, "msg_type", "nrx_ack");
    cJSON_AddNumberToObject(response, "enabled", custom_receiver_enabled);

    char *json_string = cJSON_Print(response);
    cJSON_Delete(response);
    zmq_send(rep_socket, json_string, strlen(json_string), 0);
    free(json_string);

    return 0;
}

static int handle_cir_message(cJSON *json) {
    cJSON *sigma_scaling_data = cJSON_GetObjectItem(json, "sigma_scaling");
    cJSON *sigma_max_data = cJSON_GetObjectItem(json, "sigma_max");
    cJSON *norms_data = cJSON_GetObjectItem(json, "norms");
    cJSON *taps_data = cJSON_GetObjectItem(json, "taps");
    cJSON *tap_indices_data = cJSON_GetObjectItem(json, "tap_indices");

    if (!cJSON_IsNumber(sigma_scaling_data) || !cJSON_IsNumber(sigma_max_data)
        || !cJSON_IsArray(norms_data) || !cJSON_IsArray(taps_data)
        || !cJSON_IsArray(tap_indices_data)) {
        fprintf(stderr, "Invalid CIR message format\n");
        send_error_response("Invalid CIR message format",
            "Missing or invalid sigma_scaling, sigma_max, norms, taps, or tap_indices fields");
        return -1;
    }

    // Validate array lengths
    int recv_norms_len = cJSON_GetArraySize(norms_data);
    int recv_taps_len = cJSON_GetArraySize(taps_data);
    int recv_indices_len = cJSON_GetArraySize(tap_indices_data);

    if (recv_norms_len != (int)norms_len) {
        char details[256];
        snprintf(details, sizeof(details), "Expected %d norms, got %d", (int)norms_len, recv_norms_len);
        send_error_response("Norms array size mismatch", details);
        return -1;
    }
    if (recv_taps_len != (int)taps_len) {
        char details[320];
        snprintf(details, sizeof(details),
            "Expected %d taps (S=%d, T=%d, S*2*T), got %d. Use num_taps from config_req; if client sent %d taps, message may be truncated (rebuild with larger buffer).",
            (int)taps_len, num_symbols, num_taps, recv_taps_len, num_taps);
        send_error_response("Taps array size mismatch", details);
        return -1;
    }
    if (recv_indices_len != (int)tap_indices_len) {
        char details[256];
        snprintf(details, sizeof(details), "Expected %d tap indices, got %d", (int)tap_indices_len, recv_indices_len);
        send_error_response("Tap indices array size mismatch", details);
        return -1;
    }

    // Parse all values under mutex
    pthread_mutex_lock(&mutex);
    sigma_scaling_val = (float)sigma_scaling_data->valuedouble;
    sigma_max_val = (float)sigma_max_data->valuedouble;

    for (int i = 0; i < recv_norms_len; i++) {
        cJSON *item = cJSON_GetArrayItem(norms_data, i);
        if (!cJSON_IsNumber(item)) {
            pthread_mutex_unlock(&mutex);
            char details[256];
            snprintf(details, sizeof(details), "Invalid norm value at index %d", i);
            send_error_response("Invalid norm value", details);
            return -1;
        }
        norms_array[i] = (float)item->valuedouble;
    }
    for (int i = 0; i < recv_taps_len; i++) {
        cJSON *item = cJSON_GetArrayItem(taps_data, i);
        if (!cJSON_IsNumber(item)) {
            pthread_mutex_unlock(&mutex);
            char details[256];
            snprintf(details, sizeof(details), "Invalid tap value at index %d", i);
            send_error_response("Invalid tap value", details);
            return -1;
        }
        taps_array[i] = (float)item->valuedouble;
    }
    for (int i = 0; i < recv_indices_len; i++) {
        cJSON *item = cJSON_GetArrayItem(tap_indices_data, i);
        if (!cJSON_IsNumber(item)) {
            pthread_mutex_unlock(&mutex);
            char details[256];
            snprintf(details, sizeof(details), "Invalid tap index value at index %d", i);
            send_error_response("Invalid tap index value", details);
            return -1;
        }
        tap_indices_array[i] = (uint16_t)item->valuedouble;
    }
    pthread_mutex_unlock(&mutex);

    // Send acknowledgment
    cJSON *response = cJSON_CreateObject();
    cJSON_AddStringToObject(response, "msg_type", "cir_ack");
    char *json_string = cJSON_Print(response);
    cJSON_Delete(response);
    zmq_send(rep_socket, json_string, strlen(json_string), 0);
    free(json_string);

    return 0;
}

static int handle_json_message(const char* message, size_t msg_len) {
    cJSON *json = cJSON_ParseWithLength(message, msg_len);
    if (!json) {
        /* Help diagnose: include received length and safe preview in details */
        static char details_buf[128];
        if (msg_len == 0) {
            send_error_response("Invalid JSON format", "empty message (0 bytes)");
        } else {
            int n = snprintf(details_buf, sizeof(details_buf),
                "received %zu bytes", msg_len);
            if (message[0] != '{' && message[0] != '[') {
                n += snprintf(details_buf + n, sizeof(details_buf) - (size_t)n,
                    ", first byte 0x%02x (not JSON)", (unsigned char)message[0]);
            }
            send_error_response("Invalid JSON format", details_buf);
        }
        fprintf(stderr, "CIR ZMQ: JSON parse failed, len=%zu, first 60 chars: ", msg_len);
        for (size_t i = 0; i < msg_len && i < 60; i++) {
            unsigned char c = (unsigned char)message[i];
            fputc((c >= 32 && c < 127) ? c : '.', stderr);
        }
        fputc('\n', stderr);
        return -1;
    }

    cJSON *msg_type = cJSON_GetObjectItem(json, "msg_type");
    if (!cJSON_IsString(msg_type)) {
        send_error_response("Missing or invalid msg_type field", NULL);
        cJSON_Delete(json);
        return -1;
    }

    const char *type_str = msg_type->valuestring;
    int ret = 0;
    if (strcmp(type_str, "config_req") == 0) {
        ret = handle_config_request();
    } else if (strcmp(type_str, "cir") == 0) {
        ret = handle_cir_message(json);
    } else if (strcmp(type_str, "nrx") == 0) {
        ret = handle_nrx_message(json);
    } else {
        ret = handle_unknown_message(type_str);
    }
    cJSON_Delete(json);
    return ret;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                        */
/* ------------------------------------------------------------------ */

int32_t cir_zmq_init(int num_taps_param,
                     int num_ofdm_symbols_per_slot,
                     int fft_size_param,
                     float subcarrier_spacing_param,
                     float frequency_param) {

    // Store configuration parameters
    num_taps = num_taps_param;
    num_symbols = num_ofdm_symbols_per_slot;
    fft_size = fft_size_param;
    subcarrier_spacing = subcarrier_spacing_param;
    frequency = frequency_param;

    // Create ZMQ context and REP socket (skip if already open)
    if (!zmq_context) {
        zmq_context = zmq_ctx_new();
        if (!zmq_context) {
            fprintf(stderr, "Failed to create ZMQ context\n");
            return -1;
        }
    }
    if (!rep_socket) {
        rep_socket = zmq_socket(zmq_context, ZMQ_REP);
        if (!rep_socket) {
            fprintf(stderr, "Failed to create ZMQ REP socket\n");
            return -1;
        }
        if (zmq_bind(rep_socket, "tcp://*:5555") != 0) {
            fprintf(stderr, "Failed to bind ZMQ socket to tcp://*:5555: %s\n",
                    zmq_strerror(errno));
            zmq_close(rep_socket);
            rep_socket = NULL;
            return -1;
        }
    }

    // Free existing arrays when re-initialising
    free(norms_array);       norms_array = NULL;
    free(taps_array);        taps_array = NULL;
    free(tap_indices_array); tap_indices_array = NULL;
    free(packed_cir_out);    packed_cir_out = NULL;

    // Compute array dimensions
    norms_len = (size_t)num_symbols;
    taps_len = (size_t)num_symbols * 2 * (size_t)num_taps;
    tap_indices_len = (size_t)num_symbols * (size_t)num_taps;

    // Compute packed CIR entry size (matching CUDA emulator formula)
    cir_entry_size_bytes = sizeof(float)
                           + (size_t)num_taps * sizeof(float) * 2
                           + (size_t)num_taps * sizeof(uint16_t);
    // Pad to 4-byte alignment
    cir_entry_size_bytes = (cir_entry_size_bytes + 3) & ~(size_t)3;
    packed_total_bytes = (size_t)num_symbols * cir_entry_size_bytes;

    // Allocate internal flat arrays (zero-initialised)
    norms_array = (float *)calloc(norms_len, sizeof(float));
    if (!norms_array) {
        fprintf(stderr, "Failed to allocate norms_array\n");
        return -1;
    }
    taps_array = (float *)calloc(taps_len, sizeof(float));
    if (!taps_array) {
        fprintf(stderr, "Failed to allocate taps_array\n");
        return -1;
    }
    tap_indices_array = (uint16_t *)calloc(tap_indices_len, sizeof(uint16_t));
    if (!tap_indices_array) {
        fprintf(stderr, "Failed to allocate tap_indices_array\n");
        return -1;
    }

    // Allocate packed output buffer (zero-initialised)
    packed_cir_out = (uint8_t *)calloc(1, packed_total_bytes);
    if (!packed_cir_out) {
        fprintf(stderr, "Failed to allocate packed_cir_out\n");
        return -1;
    }

    // Initialise defaults: sigma_scaling=1, sigma_max=1
    sigma_scaling_val = 1.0f;
    sigma_max_val = 1.0f;

    // Initialise per-symbol defaults
    for (int s = 0; s < num_symbols; s++) {
        // norms[s] = 1.0
        norms_array[s] = 1.0f;
        // taps: first tap I=1.0, Q=0.0; rest already zero from calloc
        taps_array[s * 2 * num_taps] = 1.0f;
        // tap_indices: [0, 1, 2, ..., num_taps-1]
        for (int k = 0; k < num_taps; k++) {
            tap_indices_array[s * num_taps + k] = (uint16_t)k;
        }
    }

    printf("\nCIR ZMQ initialized with num_taps=%d, num_ofdm_symbols_per_slot=%d, "
           "fft_size=%d, subcarrier_spacing=%.1f, frequency=%.0f\n\n",
           num_taps, num_symbols, fft_size, subcarrier_spacing, frequency);

#ifndef ENABLE_NANOBIND
    /* In OAI build: start the ZMQ receive thread so the REP socket accepts requests.
     * In Python/nanobind build: unit tests drive the server by calling receive() in a thread. */
    if (pthread_create(&zmq_thread, NULL, cir_zmq_run, NULL) != 0) {
        fprintf(stderr, "Failed to create ZMQ receive thread\n");
        return -1;
    }
    zmq_thread_created = 1;
#endif

    return 0;
}

int32_t cir_zmq_init_thread(void) { return 0; }

int32_t cir_zmq_shutdown(void) {
#ifndef ENABLE_NANOBIND
    /* OAI build: wait for ZMQ thread idle, close socket, join thread */
    if (zmq_thread_created) {
        zmq_thread_created = 0;
        for (int i = 0; i < 20 && !zmq_thread_idle_in_recv; i++) {
            struct timespec ts = { .tv_sec = 0, .tv_nsec = 50 * 1000 * 1000 }; /* 50 ms */
            nanosleep(&ts, NULL);
        }
        if (rep_socket != NULL) {
            zmq_close(rep_socket);
            rep_socket = NULL;
        }
        pthread_join(zmq_thread, NULL);
    }
#endif
    if (rep_socket != NULL) {
        zmq_close(rep_socket);
        rep_socket = NULL;
    }
    if (zmq_context != NULL) {
        zmq_ctx_term(zmq_context);
        zmq_context = NULL;
    }
    free(norms_array);       norms_array = NULL;
    free(taps_array);        taps_array = NULL;
    free(tap_indices_array); tap_indices_array = NULL;
    free(packed_cir_out);    packed_cir_out = NULL;
    norms_len = 0;
    taps_len = 0;
    tap_indices_len = 0;
    cir_entry_size_bytes = 0;
    packed_total_bytes = 0;
    return 0;
}

int cir_zmq_receive(void) {
    if (!rep_socket) return 1;

    zmq_thread_idle_in_recv = 1;
    const int recv_limit = (int)(msg_buffer_size - 1);
    int nbytes = zmq_recv(rep_socket, msg_buffer, (size_t)recv_limit, 0);
    zmq_thread_idle_in_recv = 0;
    if (nbytes < 0) {
        return 1;
    }

    /* Detect truncation: if we got a full buffer, the message may be larger (unsafe to parse) */
    if (nbytes == recv_limit) {
        send_error_response("Message too long (truncated)",
            "CIR JSON exceeds receive buffer. Reduce num_taps or rebuild with larger buffer.");
        return 0;
    }

    // Null-terminate the received message
    msg_buffer[nbytes] = '\0';

    // Handle the JSON message (this will send the appropriate response)
    handle_json_message(msg_buffer, (size_t)nbytes);

    return 0;
}

const void* cir_zmq_read(void) {
    if (!packed_cir_out) return NULL;

    pthread_mutex_lock(&mutex);

    // Pack internal arrays into the per-symbol binary layout
    for (int s = 0; s < num_symbols; s++) {
        uint8_t *entry = packed_cir_out + (size_t)s * cir_entry_size_bytes;

        // norm (float)
        float norm = norms_array[s];
        memcpy(entry, &norm, sizeof(float));
        entry += sizeof(float);

        // taps (2*T floats)
        size_t taps_bytes = (size_t)num_taps * 2 * sizeof(float);
        memcpy(entry, &taps_array[s * 2 * num_taps], taps_bytes);
        entry += taps_bytes;

        // tap_indices (T uint16_t)
        size_t indices_bytes = (size_t)num_taps * sizeof(uint16_t);
        memcpy(entry, &tap_indices_array[s * num_taps], indices_bytes);
        // padding bytes are already zero from calloc/previous writes
    }

    pthread_mutex_unlock(&mutex);
    return (const void *)packed_cir_out;
}

void* cir_zmq_run(void *arg) {
    (void)arg;
    for (;;) {
        void *sock = rep_socket;  /* read once per iteration so we see main's NULL */
        if (sock == NULL) break;
        cir_zmq_receive();
    }
    return NULL;
}

int cir_zmq_receiver_symbols_requested(void *arg) {
    (void)arg;
    pthread_mutex_lock(&mutex);
    int enabled = custom_receiver_enabled;
    pthread_mutex_unlock(&mutex);
    return enabled ? -1 : 0;
}

/* ------------------------------------------------------------------ */
/*  Getter functions                                                  */
/* ------------------------------------------------------------------ */

int cir_zmq_get_num_taps(void) {
    return num_taps;
}

float cir_zmq_get_sigma_scaling(void) {
    pthread_mutex_lock(&mutex);
    float val = sigma_scaling_val;
    pthread_mutex_unlock(&mutex);
    return val;
}

float cir_zmq_get_sigma_max(void) {
    pthread_mutex_lock(&mutex);
    float val = sigma_max_val;
    pthread_mutex_unlock(&mutex);
    return val;
}

/* ------------------------------------------------------------------ */
/*  Nanobind bindings (test-only)                                     */
/* ------------------------------------------------------------------ */

#ifdef ENABLE_NANOBIND

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

NB_MODULE(cir_zmq_py, m) {
    m.doc() = "CIR ZMQ plugin Python bindings for testing";

    m.def("init", [](int num_taps_param,
                     int num_ofdm_symbols_per_slot,
                     int fft_size_param,
                     float subcarrier_spacing_param,
                     float frequency_param) {
        return cir_zmq_init(num_taps_param, num_ofdm_symbols_per_slot,
                           fft_size_param, subcarrier_spacing_param,
                           frequency_param);
    }, nb::arg("num_taps"),
       nb::arg("num_ofdm_symbols_per_slot"),
       nb::arg("fft_size"),
       nb::arg("subcarrier_spacing"),
       nb::arg("frequency"),
    "Initialize the CIR ZMQ server");

    m.def("shutdown", []() {
        return cir_zmq_shutdown();
    }, "Shutdown the CIR ZMQ server and free resources");

    m.def("receive", []() {
        // Release GIL during blocking ZMQ receive so other Python threads can run
        nb::gil_scoped_release release;
        return cir_zmq_receive();
    }, "Receive and process one ZMQ message (blocking)");

    m.def("read", []() {
        /*
         * Read CIR data.  Rather than exposing the packed binary blob,
         * we return the internal flat arrays directly so Python tests
         * can inspect them easily:
         *   (sigma_scaling, sigma_max, norms[S], taps[S*2T], tap_indices[S*T])
         */
        if (!norms_array || !taps_array || !tap_indices_array) {
            throw std::runtime_error("CIR read: plugin not initialised");
        }

        pthread_mutex_lock(&mutex);

        size_t S = norms_len;
        size_t T2 = taps_len;          // S * 2 * num_taps
        size_t TI = tap_indices_len;    // S * num_taps

        float cur_sigma_scaling = sigma_scaling_val;
        float cur_sigma_max = sigma_max_val;

        // Copy norms
        float* norms_data = new float[S];
        memcpy(norms_data, norms_array, S * sizeof(float));

        // Copy taps
        float* taps_data = new float[T2];
        memcpy(taps_data, taps_array, T2 * sizeof(float));

        // Copy tap_indices
        uint16_t* indices_data = new uint16_t[TI];
        memcpy(indices_data, tap_indices_array, TI * sizeof(uint16_t));

        pthread_mutex_unlock(&mutex);

        nb::capsule norms_owner(norms_data, [](void* p) noexcept { delete[] (float*)p; });
        nb::capsule taps_owner(taps_data, [](void* p) noexcept { delete[] (float*)p; });
        nb::capsule indices_owner(indices_data, [](void* p) noexcept { delete[] (uint16_t*)p; });

        return nb::make_tuple(
            cur_sigma_scaling,
            cur_sigma_max,
            nb::ndarray<nb::numpy, float>(norms_data, {S}, norms_owner),
            nb::ndarray<nb::numpy, float>(taps_data, {T2}, taps_owner),
            nb::ndarray<nb::numpy, uint16_t>(indices_data, {TI}, indices_owner)
        );
    }, "Read current CIR data. Returns (sigma_scaling, sigma_max, norms, taps, tap_indices)");

    m.def("get_num_taps", []() {
        return cir_zmq_get_num_taps();
    }, "Get the configured number of taps");

    m.def("get_sigma_scaling", []() {
        return cir_zmq_get_sigma_scaling();
    }, "Get the current sigma_scaling value");

    m.def("get_sigma_max", []() {
        return cir_zmq_get_sigma_max();
    }, "Get the current sigma_max value");

    m.def("receiver_symbols_requested", []() {
        return cir_zmq_receiver_symbols_requested(nullptr);
    }, "Check if custom receiver symbols are requested");
}

#endif
