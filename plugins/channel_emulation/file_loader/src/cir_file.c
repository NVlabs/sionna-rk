/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cjson/cJSON.h>
#include "cir_file.h"

// Configuration parameters (loaded from JSON)
static int g_num_taps = 0;
static int g_num_cirs = 0;
static int g_num_symbols_per_slot = 0;
static int g_num_slots = 0;  // Derived: num_cirs / num_symbols_per_slot
static float g_sigma_scaling = 0.0f;
static float g_sigma_max = 0.0f;

// Size of each CIR entry in bytes
// Format: [norm (float)] [taps (float * num_taps * 2)] [tap_indices (uint16_t * num_taps)]
static size_t g_cir_entry_size_bytes = 0;

// CIR data storage - packed binary data
static uint8_t *g_cir_data = NULL;

// Internal slot counter (loops through the CIR bank)
static int g_current_slot = 0;

/**
 * Load configuration from JSON file.
 *
 * JSON format:
 *   {
 *     "channel_emulation": {
 *       "num_taps": N,
 *       "num_cirs": M,
 *       "sigma_scaling": F,
 *       "sigma_max": F
 *     }
 *   }
 *
 * @param json_path Path to the JSON file
 * @param[out] num_taps Number of taps per CIR
 * @param[out] num_cirs Number of CIRs
 * @param[out] sigma_scaling Noise scaling parameter
 * @param[out] sigma_max Maximum noise standard deviation
 * @return 0 on success, -1 on failure
 */
static int load_json_config(const char *json_path, int *num_taps, int *num_cirs,
                            float *sigma_scaling, float *sigma_max)
{
    FILE *f = fopen(json_path, "r");
    if (!f) {
        printf("CIR_FILE: Failed to open JSON file: %s\n", json_path);
        return -1;
    }

    // Get file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    // Read file content
    char *json_content = (char *)malloc(file_size + 1);
    if (!json_content) {
        printf("CIR_FILE: Failed to allocate memory for JSON content\n");
        fclose(f);
        return -1;
    }

    size_t read_size = fread(json_content, 1, file_size, f);
    fclose(f);
    json_content[read_size] = '\0';

    // Parse JSON
    cJSON *json = cJSON_Parse(json_content);
    free(json_content);

    if (!json) {
        printf("CIR_FILE: Failed to parse JSON file: %s\n", json_path);
        return -1;
    }

    // Get the channel_emulation section
    cJSON *channel_emu = cJSON_GetObjectItem(json, "channel_emulation");
    if (!cJSON_IsObject(channel_emu)) {
        printf("CIR_FILE: Missing or invalid 'channel_emulation' section in JSON\n");
        cJSON_Delete(json);
        return -1;
    }

    // Extract num_taps
    cJSON *num_taps_item = cJSON_GetObjectItem(channel_emu, "num_taps");
    if (!cJSON_IsNumber(num_taps_item)) {
        printf("CIR_FILE: Missing or invalid 'num_taps' in channel_emulation\n");
        cJSON_Delete(json);
        return -1;
    }
    *num_taps = num_taps_item->valueint;

    // Extract num_cirs
    cJSON *num_cirs_item = cJSON_GetObjectItem(channel_emu, "num_cirs");
    if (!cJSON_IsNumber(num_cirs_item)) {
        printf("CIR_FILE: Missing or invalid 'num_cirs' in channel_emulation\n");
        cJSON_Delete(json);
        return -1;
    }
    *num_cirs = num_cirs_item->valueint;

    // Extract sigma_scaling
    cJSON *sigma_scaling_item = cJSON_GetObjectItem(channel_emu, "sigma_scaling");
    if (!cJSON_IsNumber(sigma_scaling_item)) {
        printf("CIR_FILE: Missing or invalid 'sigma_scaling' in channel_emulation\n");
        cJSON_Delete(json);
        return -1;
    }
    *sigma_scaling = (float)sigma_scaling_item->valuedouble;

    // Extract sigma_max
    cJSON *sigma_max_item = cJSON_GetObjectItem(channel_emu, "sigma_max");
    if (!cJSON_IsNumber(sigma_max_item)) {
        printf("CIR_FILE: Missing or invalid 'sigma_max' in channel_emulation\n");
        cJSON_Delete(json);
        return -1;
    }
    *sigma_max = (float)sigma_max_item->valuedouble;

    cJSON_Delete(json);

    printf("CIR_FILE: Loaded config from '%s': num_taps=%d, num_cirs=%d, sigma_scaling=%f, sigma_max=%f\n",
           json_path, *num_taps, *num_cirs, *sigma_scaling, *sigma_max);
    return 0;
}

/**
 * Load packed CIR data from a binary file.
 *
 * Binary file format (cirs.bin) - packed entries, one per symbol:
 *   For each entry:
 *     - float32: norm (channel norm for noise std computation)
 *     - float32[num_taps * 2]: Interleaved real/imag CIR tap values
 *     - uint16_t[num_taps]: Tap delay indices
 *
 * @param bin_path Path to the binary file
 * @param num_cirs Number of CIR entries in the file
 * @param cir_entry_size Size of each CIR entry in bytes
 * @param[out] cir_data_out Pointer to allocated buffer for packed CIR data
 * @return 0 on success, -1 on failure
 */
static int load_packed_cir_data(const char *bin_path,
                                int num_cirs,
                                size_t cir_entry_size,
                                uint8_t **cir_data_out)
{
    FILE *f = fopen(bin_path, "rb");
    if (!f) {
        printf("CIR_FILE: Failed to open binary file: %s\n", bin_path);
        return -1;
    }

    // Calculate total size
    size_t total_size = (size_t)num_cirs * cir_entry_size;

    // Allocate output buffer
    uint8_t *cir_data = (uint8_t *)malloc(total_size);
    if (!cir_data) {
        printf("CIR_FILE: Failed to allocate memory for CIR data\n");
        fclose(f);
        return -1;
    }

    // Read packed CIR data
    size_t read_bytes = fread(cir_data, 1, total_size, f);
    if (read_bytes != total_size) {
        printf("CIR_FILE: Failed to read CIR data: expected %zu bytes, got %zu\n",
                total_size, read_bytes);
        free(cir_data);
        fclose(f);
        return -1;
    }

    fclose(f);
    *cir_data_out = cir_data;

    printf("CIR_FILE: Loaded packed CIR data from '%s': %d entries, %zu bytes each\n",
           bin_path, num_cirs, cir_entry_size);
    return 0;
}

int32_t cir_file_init(const char* folder_path, int num_symbols_per_slot)
{
    // Store num_symbols_per_slot
    g_num_symbols_per_slot = num_symbols_per_slot;

    // Build file paths (folder_path/config.json, folder_path/cirs.bin)
    size_t folder_len = strlen(folder_path);
    // Add extra space for "/" + filename + null terminator
    char *json_path = (char *)malloc(folder_len + 13);     // "/config.json" + null
    char *cirs_bin_path = (char *)malloc(folder_len + 10); // "/cirs.bin" + null

    if (!json_path || !cirs_bin_path) {
        printf("CIR_FILE: Failed to allocate memory for file paths\n");
        free(json_path);
        free(cirs_bin_path);
        return -1;
    }

    sprintf(json_path, "%s/config.json", folder_path);
    sprintf(cirs_bin_path, "%s/cirs.bin", folder_path);

    // Load configuration from JSON
    int ret = load_json_config(json_path, &g_num_taps, &g_num_cirs,
                               &g_sigma_scaling, &g_sigma_max);
    free(json_path);
    if (ret != 0) {
        free(cirs_bin_path);
        return -1;
    }

    // Calculate CIR entry size:
    // - 1 float for norm
    // - num_taps * 2 floats for taps (interleaved real/imag)
    // - num_taps uint16_t for tap indices
    g_cir_entry_size_bytes = sizeof(float) +
                             (size_t)g_num_taps * sizeof(float) * 2 +
                             (size_t)g_num_taps * sizeof(uint16_t);
    // Pad to 4-byte alignment for proper GPU memory access
    g_cir_entry_size_bytes = (g_cir_entry_size_bytes + 3) & ~3;

    // Validate and compute number of slots
    if (g_num_symbols_per_slot <= 0) {
        printf("CIR_FILE: num_symbols_per_slot must be positive\n");
        free(cirs_bin_path);
        return -1;
    }
    if (g_num_cirs % g_num_symbols_per_slot != 0) {
        int num_slots = g_num_cirs / g_num_symbols_per_slot;
        int num_extra_cirs = g_num_cirs % g_num_symbols_per_slot;
        g_num_cirs = num_slots * g_num_symbols_per_slot;
        printf("CIR_FILE: Only using %d CIRs out of %d due to num_symbols_per_slot=%d\n",
                g_num_cirs, g_num_cirs + num_extra_cirs, g_num_symbols_per_slot);
    }
    g_num_slots = g_num_cirs / g_num_symbols_per_slot;

    // Load packed CIR binary data
    ret = load_packed_cir_data(cirs_bin_path, g_num_cirs, g_cir_entry_size_bytes, &g_cir_data);
    free(cirs_bin_path);
    if (ret != 0) {
        return -1;
    }

    // Initialize slot counter
    g_current_slot = 0;

    printf("\nCIR_FILE: Initialized with folder='%s'\n", folder_path);
    printf("  num_taps=%d, num_cirs=%d, num_symbols_per_slot=%d, num_slots=%d\n",
           g_num_taps, g_num_cirs, g_num_symbols_per_slot, g_num_slots);
    printf("  sigma_scaling=%f, sigma_max=%f, entry_size=%zu bytes\n\n",
           g_sigma_scaling, g_sigma_max, g_cir_entry_size_bytes);

    return 0;
}

int32_t cir_file_init_thread(void)
{
    return 0;
}

int32_t cir_file_shutdown(void)
{
    if (g_cir_data != NULL) {
        free(g_cir_data);
        g_cir_data = NULL;
    }

    g_num_taps = 0;
    g_num_cirs = 0;
    g_num_symbols_per_slot = 0;
    g_num_slots = 0;
    g_current_slot = 0;
    g_sigma_scaling = 0.0f;
    g_sigma_max = 0.0f;
    g_cir_entry_size_bytes = 0;

    printf("CIR_FILE: Shutdown complete\n");
    return 0;
}

const void* cir_file_read(void)
{
    // Calculate offset for the current slot
    // Each slot has num_symbols_per_slot entries
    size_t slot_offset = (size_t)g_current_slot * g_num_symbols_per_slot * g_cir_entry_size_bytes;

    // Advance to the next slot, wrapping around
    g_current_slot = (g_current_slot + 1) % g_num_slots;

    // Return pointer to the packed CIR data for this slot
    return g_cir_data + slot_offset;
}

int cir_file_get_num_taps(void)
{
    return g_num_taps;
}

float cir_file_get_sigma_scaling(void)
{
    return g_sigma_scaling;
}

float cir_file_get_sigma_max(void)
{
    return g_sigma_max;
}

#ifdef ENABLE_NANOBIND

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>

namespace nb = nanobind;

NB_MODULE(cir_file_py, m) {
    m.doc() = "CIR file loader plugin Python bindings for testing";

    m.def("init", [](const char* folder_path, int num_symbols_per_slot) {
        return cir_file_init(folder_path, num_symbols_per_slot);
    }, nb::arg("folder_path"), nb::arg("num_symbols_per_slot"),
    "Initialize the CIR file loader with the given folder path and symbols per slot");

    m.def("init_thread", []() {
        return cir_file_init_thread();
    }, "Thread-specific initialization (no-op for file loader)");

    m.def("shutdown", []() {
        return cir_file_shutdown();
    }, "Shutdown the CIR file loader and free resources");

    m.def("get_num_taps", []() {
        return cir_file_get_num_taps();
    }, "Get the number of taps per CIR");

    m.def("get_sigma_scaling", []() {
        return cir_file_get_sigma_scaling();
    }, "Get the sigma_scaling parameter");

    m.def("get_sigma_max", []() {
        return cir_file_get_sigma_max();
    }, "Get the sigma_max parameter");

    // Return CIR data as Python-friendly structures
    // Unpacks the packed format for easier verification in tests
    m.def("read", [](int num_symbols_per_slot) {
        const void* packed_cir = cir_file_read();
        if (!packed_cir) {
            throw std::runtime_error("CIR file read returned null data");
        }

        int num_taps = cir_file_get_num_taps();
        int total_taps = num_taps * num_symbols_per_slot;

        // Calculate entry size with 4-byte alignment padding (must match C code)
        size_t raw_entry_size = sizeof(float) + num_taps * sizeof(float) * 2 + num_taps * sizeof(uint16_t);
        size_t cir_entry_size = (raw_entry_size + 3) & ~3;  // Pad to 4-byte alignment

        // Allocate output arrays
        std::vector<float> norms(num_symbols_per_slot);
        std::vector<float> taps(total_taps * 2);
        std::vector<uint16_t> tap_indices(total_taps);

        // Unpack the data
        const uint8_t* base_ptr = reinterpret_cast<const uint8_t*>(packed_cir);
        for (int sym = 0; sym < num_symbols_per_slot; sym++) {
            // Point to start of this entry
            const uint8_t* ptr = base_ptr + sym * cir_entry_size;

            // Read norm
            norms[sym] = *reinterpret_cast<const float*>(ptr);
            ptr += sizeof(float);

            // Read taps
            memcpy(&taps[sym * num_taps * 2], ptr, num_taps * sizeof(float) * 2);
            ptr += num_taps * sizeof(float) * 2;

            // Read tap indices
            memcpy(&tap_indices[sym * num_taps], ptr, num_taps * sizeof(uint16_t));
            // Note: padding bytes at end of entry are skipped by using cir_entry_size offset
        }

        // Create numpy arrays to return the data
        // norms: [num_symbols_per_slot]
        auto norms_arr = nb::ndarray<nb::numpy, float, nb::shape<-1>>(
            norms.data(),
            {static_cast<size_t>(num_symbols_per_slot)},
            nb::handle()  // No owner, we'll copy
        );

        // taps: [total_taps, 2] for real/imag
        auto taps_arr = nb::ndarray<nb::numpy, float, nb::shape<-1, 2>>(
            taps.data(),
            {static_cast<size_t>(total_taps), 2},
            nb::handle()
        );

        // tap_indices: [total_taps]
        auto tap_indices_arr = nb::ndarray<nb::numpy, uint16_t, nb::shape<-1>>(
            tap_indices.data(),
            {static_cast<size_t>(total_taps)},
            nb::handle()
        );

        // Return copies of the arrays (since vectors will go out of scope)
        return nb::make_tuple(
            nb::ndarray<nb::numpy, float>(norms_arr),
            nb::ndarray<nb::numpy, float>(taps_arr),
            nb::ndarray<nb::numpy, uint16_t>(tap_indices_arr)
        );
    }, nb::arg("num_symbols_per_slot"),
    "Read the next CIR slot. Returns (norms, taps, tap_indices)");

    // Return raw pointer to packed CIR data
    m.def("read_raw", []() {
        const void* packed_cir = cir_file_read();
        if (!packed_cir) {
            throw std::runtime_error("CIR file read returned null");
        }
        return reinterpret_cast<uintptr_t>(packed_cir);
    }, "Read raw packed CIR data (returns pointer address)");
}

#endif
