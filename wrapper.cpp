#include "wrapper.h"
#include "llama.cpp/include/llama.h"
#include "llama.cpp/ggml/include/ggml.h"
#include "llama.cpp/ggml/include/gguf.h"
#include "llama.cpp/common/common.h"
#include "llama.cpp/common/sampling.h"
#include "llama.cpp/common/speculative.h"
#include "llama.cpp/common/chat.h"
// Use angle-bracket form so this resolves via -I paths in both modes:
// dev (-I./llama.cpp/vendor → submodule) and consumer
// (-I./cgo_headers/llama.cpp/thirdparty → vendored copy). The vendored copy
// can't live under a directory named "vendor" because Go's module zip excludes
// any nested vendor/ subtree.
#include <nlohmann/json.hpp>

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <cstring>

// CUDA backend header for GPU info
#ifdef GGML_USE_CUDA
#include "llama.cpp/ggml/include/ggml-cuda.h"
#endif

// Global error handling
static std::string g_last_error;

// Global log level control
static ggml_log_level g_min_log_level = GGML_LOG_LEVEL_INFO;

// Log callback that respects LLAMA_LOG environment variable
static void llama_log_callback(ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level >= g_min_log_level) {
        fprintf(stderr, "%s", text);
    }
}

// Chat session plus a stable copy of its rendered prompt for the C API
struct llama_wrapper_chat_session_t {
    common_chat_session session;
    std::string prompt;
};

// Apply a chat session's sampling settings (generation prompt, grammar and
// the tokens it preserves) and its thinking tags, so the sampler starts in
// the right reasoning state and can enforce the thinking budget.
static void apply_chat_session_sampling(common_params_sampling & sampling, const llama_model * model,
                                        void * chat_session, int reasoning_budget) {
    if (!chat_session) return;
    const auto * session = &static_cast<const llama_wrapper_chat_session_t*>(chat_session)->session;

    session->apply_sampling(sampling);

    if (reasoning_budget < 0 || session->thinking_start_tag().empty() || session->thinking_end_tags().empty()) {
        return;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    sampling.reasoning_budget_tokens = reasoning_budget;
    sampling.reasoning_budget_start = common_tokenize(vocab, session->thinking_start_tag(), false, true);
    sampling.reasoning_budget_end.clear();
    for (const auto & tag : session->thinking_end_tags()) {
        sampling.reasoning_budget_end.push_back(common_tokenize(vocab, tag, false, true));
    }
    sampling.reasoning_budget_forced = sampling.reasoning_budget_end.front();
}

// Decode a batch in n_batch chunks, returning llama_process's result
static int32_t decode_batch(llama_context * ctx, common_batch & batch) {
    const int32_t n_batch = (int32_t)llama_n_batch(ctx);
    for (int32_t off = 0; off < batch.size(); off += n_batch) {
        const int32_t n = std::min(n_batch, batch.size() - off);
        const int32_t ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get_sub_batch(off, n));
        if (ret != 0) {
            return ret;
        }
    }
    return 0;
}

extern "C" {

// Initialise logging based on LLAMA_LOG environment variable
// Supported values: none, debug, info (default), warn, error
void llama_wrapper_init_logging() {
    const char* log_level = std::getenv("LLAMA_LOG");
    if (log_level != nullptr) {
        std::string level_str(log_level);
        if (level_str == "none") {
            g_min_log_level = GGML_LOG_LEVEL_NONE;
        } else if (level_str == "debug") {
            g_min_log_level = GGML_LOG_LEVEL_DEBUG;
        } else if (level_str == "info") {
            g_min_log_level = GGML_LOG_LEVEL_INFO;
        } else if (level_str == "warn") {
            g_min_log_level = GGML_LOG_LEVEL_WARN;
        } else if (level_str == "error") {
            g_min_log_level = GGML_LOG_LEVEL_ERROR;
        }
    }
    llama_log_set(llama_log_callback, nullptr);
}

// Forward declarations of Go callback functions
extern bool goTokenCallback(uintptr_t handle, const char* token);
extern bool goProgressCallback(float progress, void* user_data);

// Separate wrappers for model and context
struct llama_wrapper_model_t {
    llama_model* model;
    int n_gpu_layers;  // Number of GPU layers requested (for stats reporting)
    bool has_mtp_head = false;  // The file holds a built-in MTP head
    bool mtp_loaded = false;    // ...and it was loaded alongside the weights
};

// Whether a model file holds MTP head tensors. A file can declare a head and
// ship without it, so check the tensor names, across every split. Only the
// headers are read.
static bool file_has_mtp_head(const char * path) {
    const gguf_init_params gparams = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    int n_split = 1;
    auto scan = [&](const char * file, bool first) {
        gguf_context * gguf = gguf_init_from_file(file, gparams);
        if (!gguf) {
            return false;
        }
        if (first) {
            const int64_t key = gguf_find_key(gguf, "split.count");
            n_split = key >= 0 ? gguf_get_val_u16(gguf, key) : 1;
        }
        bool found = false;
        for (int64_t i = 0; i < gguf_get_n_tensors(gguf) && !found; i++) {
            found = strstr(gguf_get_tensor_name(gguf, i), ".nextn.") != nullptr;
        }
        gguf_free(gguf);
        return found;
    };

    if (scan(path, true)) {
        return true;
    }
    char prefix[4096];
    if (n_split <= 1 || llama_split_prefix(prefix, sizeof(prefix), path, 0, n_split) == 0) {
        return false;
    }
    for (int i = 1; i < n_split; i++) {
        char split_path[4096];
        llama_split_path(split_path, sizeof(split_path), prefix, i, n_split);
        if (scan(split_path, false)) {
            return true;
        }
    }
    return false;
}

struct llama_wrapper_context_t {
    llama_context* ctx;
    llama_model* model;  // Reference to parent model
    std::vector<int> cached_tokens;  // Prompt tokens currently held in memory for sequence 0
    // Recurrent, hybrid and sliding-window memory can't be trimmed to an
    // arbitrary position, so reuse restores a checkpoint saved before the last
    // prompt token instead
    bool needs_checkpoint = false;
    common_prompt_checkpoint checkpoint;
    // How the memory can be trimmed, from the probe at creation
    common_context_seq_rm_type rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_PART;
    // State before the current speculative step, for rolling back rejected
    // drafted tokens on the same kinds of memory
    common_prompt_checkpoint step_checkpoint;

    // Speculative decoding: a draft context owned by this one, kept in step
    // with it through the speculative module
    llama_wrapper_context_t* draft = nullptr;
    common_speculative* spec = nullptr;
    int n_draft = 0;
};

void llama_wrapper_context_free(void* ctx);

const char* llama_wrapper_last_error() {
    return g_last_error.c_str();
}

void llama_wrapper_free_result(char* result) {
    if (result) {
        free(result);
    }
}

// Static no-op callback for silent loading
static bool silent_progress_callback(float progress, void* user_data) {
    (void)progress;
    (void)user_data;
    return true;  // Continue loading
}

// Convert our params to llama.cpp model params
static struct llama_model_params convert_model_params(llama_wrapper_model_params params) {
    struct llama_model_params model_params = llama_model_default_params();

    // Only set n_gpu_layers if not -1 (which means "use default/all layers")
    // llama.cpp default is 999 which effectively means all layers
    if (params.n_gpu_layers != -1) {
        model_params.n_gpu_layers = params.n_gpu_layers;
    }

    model_params.main_gpu = params.main_gpu ? atoi(params.main_gpu) : 0;
    // b10675 replaced the use_mmap/use_mlock booleans with a single load_mode enum
    if (params.mmap) {
        model_params.load_mode = params.mlock ? LLAMA_LOAD_MODE_MMAP_MLOCK : LLAMA_LOAD_MODE_MMAP;
    } else {
        model_params.load_mode = params.mlock ? LLAMA_LOAD_MODE_MLOCK : LLAMA_LOAD_MODE_NONE;
    }
    model_params.no_host = false;  // Use host buffers (b6709 added field)
    model_params.load_mtp = params.load_mtp;

    // Configure progress callback
    if (params.disable_progress_callback) {
        model_params.progress_callback = silent_progress_callback;
        model_params.progress_callback_user_data = nullptr;
    } else if (params.progress_callback) {
        model_params.progress_callback = params.progress_callback;
        model_params.progress_callback_user_data = params.progress_callback_user_data;
    }
    // Otherwise NULL → llama.cpp installs default dot printer

    return model_params;
}

// Convert our params to llama.cpp context params
static struct llama_context_params convert_context_params(llama_wrapper_model_params params) {
    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = params.n_ctx > 0 ? params.n_ctx : 2048;
    ctx_params.n_batch = params.n_batch > 0 ? params.n_batch : 512;
    // n_ubatch: 0 means match n_batch. Encoder-only models (nomic, BERT) require
    // n_ubatch >= longest sequence length, since the whole input must fit in one
    // forward pass.
    ctx_params.n_ubatch = params.n_ubatch > 0 ? params.n_ubatch : ctx_params.n_batch;
    ctx_params.n_threads = params.n_threads > 0 ? params.n_threads : 4;
    ctx_params.n_threads_batch = params.n_threads_batch > 0 ? params.n_threads_batch : ctx_params.n_threads;
    ctx_params.n_seq_max = params.n_parallel > 0 ? params.n_parallel : 1;
    ctx_params.embeddings = params.embeddings;
    ctx_params.swa_full = params.swa_full;

    // Set KV cache quantization type
    if (params.kv_cache_type != nullptr) {
        std::string cache_type(params.kv_cache_type);
        if (cache_type == "f16") {
            ctx_params.type_k = GGML_TYPE_F16;
            ctx_params.type_v = GGML_TYPE_F16;
        } else if (cache_type == "q8_0") {
            ctx_params.type_k = GGML_TYPE_Q8_0;
            ctx_params.type_v = GGML_TYPE_Q8_0;
        } else if (cache_type == "q4_0") {
            ctx_params.type_k = GGML_TYPE_Q4_0;
            ctx_params.type_v = GGML_TYPE_Q4_0;
        }
        // If unrecognized, leave as default (f16)
    }

    // Set Flash Attention mode
    if (params.flash_attn != nullptr) {
        std::string fa_mode(params.flash_attn);
        if (fa_mode == "enabled") {
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        } else if (fa_mode == "disabled") {
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        } else if (fa_mode == "auto") {
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        }
        // If unrecognized, leave as default (auto)
    }

    // RoPE/YaRN overrides, validated on the Go side
    const unsigned int ro = params.rope_overrides;
    if (params.rope_scaling_type != nullptr) {
        std::string scaling(params.rope_scaling_type);
        if (scaling == "none") {
            ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
        } else if (scaling == "linear") {
            ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
        } else if (scaling == "yarn") {
            ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
        } else if (scaling == "longrope") {
            ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LONGROPE;
        }
    }
    if (ro & LLAMA_WRAPPER_ROPE_FREQ_BASE)   ctx_params.rope_freq_base   = params.rope_freq_base;
    if (ro & LLAMA_WRAPPER_ROPE_FREQ_SCALE)  ctx_params.rope_freq_scale  = params.rope_freq_scale;
    if (ro & LLAMA_WRAPPER_YARN_EXT_FACTOR)  ctx_params.yarn_ext_factor  = params.yarn_ext_factor;
    if (ro & LLAMA_WRAPPER_YARN_ATTN_FACTOR) ctx_params.yarn_attn_factor = params.yarn_attn_factor;
    if (ro & LLAMA_WRAPPER_YARN_BETA_FAST)   ctx_params.yarn_beta_fast   = params.yarn_beta_fast;
    if (ro & LLAMA_WRAPPER_YARN_BETA_SLOW)   ctx_params.yarn_beta_slow   = params.yarn_beta_slow;
    if (ro & LLAMA_WRAPPER_YARN_ORIG_CTX)    ctx_params.yarn_orig_ctx    = params.yarn_orig_ctx;

    return ctx_params;
}

void* llama_wrapper_model_load(const char* model_path, llama_wrapper_model_params params) {
    if (!model_path) {
        g_last_error = "Model path cannot be null";
        return nullptr;
    }

    try {
        // Initialize llama backend
        llama_backend_init();

        // Load model (weights only)
        auto model_params = convert_model_params(params);
        llama_model* model = llama_model_load_from_file(model_path, model_params);
        if (!model) {
            g_last_error = "Failed to load model from: " + std::string(model_path);
            return nullptr;
        }

        // Create wrapper (model only, no context)
        auto wrapper = new llama_wrapper_model_t();
        wrapper->model = model;
        // Store n_gpu_layers for stats reporting
        // If -1 was passed (meaning "use default"), llama.cpp uses 999 layers
        wrapper->n_gpu_layers = (params.n_gpu_layers == -1) ? 999 : params.n_gpu_layers;
        wrapper->has_mtp_head = llama_model_n_layer_nextn(model) > 0 && file_has_mtp_head(model_path);
        wrapper->mtp_loaded = params.load_mtp && wrapper->has_mtp_head;

        return wrapper;
    } catch (const std::exception& e) {
        g_last_error = "Exception loading model: " + std::string(e.what());
        return nullptr;
    }
}

void llama_wrapper_model_free(void* model) {
    if (!model) return;

    auto wrapper = static_cast<llama_wrapper_model_t*>(model);
    if (wrapper->model) {
        llama_model_free(wrapper->model);
        wrapper->model = nullptr;  // Prevent double-free
    }
    delete wrapper;
}

// Create a context on a model and probe how its memory can be trimmed. The
// probe decodes two tokens, so do it once here. A window-sized SWA cache drops
// positions as it goes, so it needs checkpoints too. A batch too small to
// probe gets checkpoints, which fall back to a full clear on any memory type.
// Embedding contexts never reuse memory.
static llama_wrapper_context_t* create_context(llama_model* model, const llama_context_params& ctx_params) {
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        return nullptr;
    }

    auto ctx_wrapper = new llama_wrapper_context_t();
    ctx_wrapper->ctx = ctx;
    ctx_wrapper->model = model;

    if (!ctx_params.embeddings && llama_n_batch(ctx) < 2) {
        ctx_wrapper->rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        ctx_wrapper->needs_checkpoint = true;
    } else if (!ctx_params.embeddings) {
        ctx_wrapper->rm_type = common_context_can_seq_rm(ctx);
        ctx_wrapper->needs_checkpoint = ctx_wrapper->rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_PART ||
                                        (llama_model_n_swa(model) > 0 && !ctx_params.swa_full);
    }
    return ctx_wrapper;
}

void* llama_wrapper_context_create(void* model, llama_wrapper_model_params params) {
    if (!model) {
        g_last_error = "Model cannot be null";
        return nullptr;
    }

    llama_wrapper_context_t* ctx_wrapper = nullptr;
    try {
        auto model_wrapper = static_cast<llama_wrapper_model_t*>(model);
        auto ctx_params = convert_context_params(params);

        const bool speculate = params.draft_model || params.mtp;
        if (speculate && (params.embeddings || ctx_params.n_seq_max != 1)) {
            g_last_error = "Speculative decoding needs a single-sequence generation context";
            return nullptr;
        }
        // With a draft model, MTP runs that model as a separate head, as for
        // Gemma 4; without one, it runs the head built into this model
        if (params.mtp && !params.draft_model && !model_wrapper->mtp_loaded) {
            g_last_error = model_wrapper->has_mtp_head
                ? "Load the model with its MTP head to speculate with it"
                : "Model has no MTP head";
            return nullptr;
        }

        // The whole verification batch is decoded at once, so the draft must
        // fit in a batch alongside the token before it
        int n_draft = 0;
        if (speculate) {
            n_draft = std::min(params.n_draft > 0 ? params.n_draft : 3, (int)ctx_params.n_batch - 1);
            if (n_draft < 1) {
                g_last_error = "Batch size is too small for speculative decoding";
                return nullptr;
            }
        }

        // As in common, an MTP target keeps a recurrent-state snapshot per
        // drafted token, so rejected tokens can be trimmed without a checkpoint
        if (params.mtp) {
            ctx_params.n_rs_seq = n_draft;
        }

        ctx_wrapper = create_context(model_wrapper->model, ctx_params);
        if (!ctx_wrapper) {
            g_last_error = "Failed to create context";
            return nullptr;
        }

        if (speculate) {
            ctx_wrapper->n_draft = std::min(n_draft, (int)llama_n_batch(ctx_wrapper->ctx) - 1);

            // The draft context follows the target's settings, as in common,
            // and is linked to the target for drafters that read its memory.
            // A built-in MTP head runs on the target's own weights.
            auto draft_params = ctx_params;
            draft_params.n_ctx = llama_n_ctx(ctx_wrapper->ctx);
            draft_params.n_rs_seq = 0;
            draft_params.ctx_other = ctx_wrapper->ctx;
            if (params.mtp) {
                draft_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            }
            llama_model* draft_model = params.draft_model
                ? static_cast<llama_wrapper_model_t*>(params.draft_model)->model
                : model_wrapper->model;
            ctx_wrapper->draft = create_context(draft_model, draft_params);
            if (!ctx_wrapper->draft) {
                throw std::runtime_error("failed to create draft context");
            }

            common_params_speculative spec_params;
            spec_params.types         = { params.mtp ? COMMON_SPECULATIVE_TYPE_DRAFT_MTP : COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE };
            spec_params.draft.n_max   = ctx_wrapper->n_draft;
            spec_params.draft.ctx_tgt = ctx_wrapper->ctx;
            spec_params.draft.ctx_dft = ctx_wrapper->draft->ctx;
            ctx_wrapper->spec = common_speculative_init(spec_params, 1);
            if (!ctx_wrapper->spec) {
                throw std::runtime_error("failed to initialise speculative decoding");
            }
        }

        return ctx_wrapper;
    } catch (const std::exception& e) {
        llama_wrapper_context_free(ctx_wrapper);
        g_last_error = "Exception creating context: " + std::string(e.what());
        return nullptr;
    }
}

void llama_wrapper_context_free(void* ctx) {
    if (!ctx) return;

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);
    if (wrapper->spec) {
        common_speculative_free(wrapper->spec);
        wrapper->spec = nullptr;
    }
    llama_wrapper_context_free(wrapper->draft);
    wrapper->draft = nullptr;
    if (wrapper->ctx) {
        llama_free(wrapper->ctx);
        wrapper->ctx = nullptr;  // Prevent double-free
    }
    delete wrapper;
}

int llama_wrapper_get_model_context_length(void* model) {
    if (!model) {
        return 32768;  // Fallback if model is null
    }

    auto model_wrapper = static_cast<llama_wrapper_model_t*>(model);

    // Query model's native context length from GGUF metadata
    int n_ctx_train = llama_model_n_ctx_train(model_wrapper->model);

    // Return model's training context, or reasonable fallback
    return (n_ctx_train > 0) ? n_ctx_train : 32768;
}

// Get the width of the model's output embeddings
int llama_wrapper_model_n_embd_out(void* model) {
    if (!model) {
        return -1;  // Error if model is null
    }

    auto model_wrapper = static_cast<llama_wrapper_model_t*>(model);
    return llama_model_n_embd_out(model_wrapper->model);
}

int llama_wrapper_dry_max_seq_breakers(void) {
    return LLAMA_DRY_MAX_SEQ_BREAKERS;
}

bool llama_wrapper_model_has_mtp_head(void* model) {
    return model && static_cast<llama_wrapper_model_t*>(model)->has_mtp_head;
}

// Helper function to find common prefix length between two token vectors
static int findCommonPrefix(const std::vector<int>& a, const std::vector<int>& b) {
    int commonLen = 0;
    size_t minLen = std::min(a.size(), b.size());
    for (size_t i = 0; i < minLen; i++) {
        if (a[i] != b[i]) {
            break;
        }
        commonLen++;
    }
    return commonLen;
}

// Build sampling parameters for a generation request
static common_params_sampling make_sampling_params(const llama_wrapper_generate_params & params, const llama_model * model) {
    common_params_sampling sampling_params;
    // Basic sampling
    sampling_params.seed = params.seed;
    sampling_params.temp = params.temperature;
    sampling_params.top_k = params.top_k;
    sampling_params.top_p = params.top_p;
    sampling_params.min_p = params.min_p;
    sampling_params.typ_p = params.typ_p;
    sampling_params.top_n_sigma = params.top_n_sigma;
    sampling_params.min_keep = params.min_keep;

    // Repetition penalties
    sampling_params.penalty_last_n = params.penalty_last_n;
    sampling_params.penalty_repeat = params.penalty_repeat;
    sampling_params.penalty_freq = params.penalty_freq;
    sampling_params.penalty_present = params.penalty_present;

    // DRY sampling
    sampling_params.dry_multiplier = params.dry_multiplier;
    sampling_params.dry_base = params.dry_base;
    sampling_params.dry_allowed_length = params.dry_allowed_length;
    sampling_params.dry_penalty_last_n = params.dry_penalty_last_n;
    sampling_params.dry_sequence_breakers.clear();
    for (int i = 0; i < params.dry_sequence_breakers_count; i++) {
        sampling_params.dry_sequence_breakers.push_back(std::string(params.dry_sequence_breakers[i]));
    }

    // Dynamic temperature
    sampling_params.dynatemp_range = params.dynatemp_range;
    sampling_params.dynatemp_exponent = params.dynatemp_exponent;

    // XTC sampling
    sampling_params.xtc_probability = params.xtc_probability;
    sampling_params.xtc_threshold = params.xtc_threshold;

    // Mirostat sampling
    sampling_params.mirostat = params.mirostat;
    sampling_params.mirostat_tau = params.mirostat_tau;
    sampling_params.mirostat_eta = params.mirostat_eta;

    // Other parameters
    sampling_params.n_prev = params.n_prev;
    sampling_params.n_probs = params.n_probs;
    // As in common, ignoring EOS means never sampling an end-of-generation
    // token, which needs a vocab with an EOS token to ignore
    const llama_vocab * vocab = llama_model_get_vocab(model);
    sampling_params.ignore_eos = params.ignore_eos && llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL;
    if (sampling_params.ignore_eos) {
        for (llama_token i = 0; i < llama_vocab_n_tokens(vocab); i++) {
            if (llama_vocab_is_eog(vocab, i)) {
                sampling_params.logit_bias.push_back({i, -INFINITY});
            }
        }
    }

    apply_chat_session_sampling(sampling_params, model, params.chat_session, params.reasoning_budget);
    return sampling_params;
}

// Decodes one batch of at most n_batch entries into a context
using decode_fn = std::function<bool(common_batch &)>;

// Restore a context's memory to resume a prompt of n_tokens after at most
// limit tokens, always leaving the last prompt token to decode so its logits
// are fresh. Memory that can't be trimmed resumes from the checkpoint saved
// before the last prompt token, or from the start. Returns where decoding
// resumes, or -1 with g_last_error set.
static int restore_prompt(llama_wrapper_context_t * w, int n_tokens, int limit) {
    // Memory no longer matches cached_tokens until the new prompt is decoded
    w->cached_tokens.clear();

    llama_memory_t mem = llama_get_memory(w->ctx);
    int resume_from = std::min(limit, n_tokens - 1);
    if (w->needs_checkpoint) {
        const auto & ckpt = w->checkpoint;
        if (!ckpt.empty() && ckpt.n_tokens <= resume_from &&
            ckpt.load_tgt(w->ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) &&
            llama_memory_seq_rm(mem, 0, ckpt.pos_max + 1, -1)) {
            resume_from = (int)ckpt.n_tokens;
        } else {
            resume_from = 0;
        }
    }
    if (resume_from == 0) {
        w->checkpoint.clear();
        if (!llama_memory_seq_rm(mem, 0, -1, -1)) {
            g_last_error = "Failed to clear memory";
            return -1;
        }
    } else if (!w->needs_checkpoint && !llama_memory_seq_rm(mem, 0, resume_from, -1)) {
        g_last_error = "Failed to trim memory to the cached prefix";
        return -1;
    }
    return resume_from;
}

// Save the prompt checkpoint before the last prompt token, for memory that
// can't be trimmed back to it
static void save_prompt_checkpoint(llama_wrapper_context_t * w, int n_tokens) {
    if (!w->needs_checkpoint || n_tokens < 2 ||
        (!w->checkpoint.empty() && w->checkpoint.n_tokens == n_tokens - 1)) {
        return;
    }
    llama_memory_t mem = llama_get_memory(w->ctx);
    w->checkpoint.update_pos(n_tokens - 1, llama_memory_seq_pos_min(mem, 0), llama_memory_seq_pos_max(mem, 0));
    w->checkpoint.update_tgt(w->ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
}

// Whether rolling back n tokens after a speculative step needs the state
// saved before it, as in the server: memory that can only be cleared whole
// always does, and recurrent memory does beyond its snapshots
static bool step_needs_checkpoint(const llama_wrapper_context_t * w, size_t n) {
    return w->rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
           (w->rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n > llama_n_rs_seq(w->ctx));
}

// Save the state before a speculative step at n_past
static void save_step(llama_wrapper_context_t * w, llama_pos n_past) {
    llama_memory_t mem = llama_get_memory(w->ctx);
    w->step_checkpoint.update_pos(n_past, llama_memory_seq_pos_min(mem, 0), n_past - 1);
    w->step_checkpoint.update_tgt(w->ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
}

// Roll a context back after a speculative step at n_past so its memory ends
// with the kept tokens, by trimming, or by restoring the state saved before
// the step and decoding the kept tokens again
static bool roll_back_step(llama_wrapper_context_t * w, llama_pos n_past, const llama_tokens & keep,
                           bool from_checkpoint, const decode_fn & decode) {
    llama_memory_t mem = llama_get_memory(w->ctx);
    if (!from_checkpoint) {
        return llama_memory_seq_rm(mem, 0, n_past + (llama_pos)keep.size(), -1);
    }
    if (!w->step_checkpoint.load_tgt(w->ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) ||
        !llama_memory_seq_rm(mem, 0, n_past, -1)) {
        return false;
    }
    if (keep.empty()) {
        return true;
    }
    common_batch batch(w->ctx);
    for (size_t i = 0; i < keep.size(); i++) {
        batch.add(keep[i], n_past + (llama_pos)i, 0, false);
    }
    return decode(batch);
}

char* llama_wrapper_generate_with_tokens(void* ctx, const int* tokens, int n_tokens, int prefix_len, llama_wrapper_generate_params params) {
    if (!ctx || !tokens) {
        g_last_error = "Context and tokens cannot be null";
        return nullptr;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);
    if (!wrapper->ctx) {
        g_last_error = "Context has been freed";
        return nullptr;
    }

    try {
        std::vector<llama_token> prompt_tokens(tokens, tokens + n_tokens);

        if (prompt_tokens.empty()) {
            g_last_error = "Token array is empty";
            return nullptr;
        }

        // Check context size with safety margin BEFORE manipulating KV cache
        int available_ctx = llama_n_ctx(wrapper->ctx);
        if (available_ctx <= 0) {
            g_last_error = "Invalid context size";
            return nullptr;
        }
        // Check if prompt fits with room for at least a few generated tokens
        int tokens_needed = (int)prompt_tokens.size() + params.max_tokens;
        if (tokens_needed > available_ctx) {
            char err_msg[256];
            snprintf(err_msg, sizeof(err_msg),
                    "Prompt too long for context size: need %d tokens (%d prompt + %d generation) but context is only %d tokens",
                    tokens_needed, (int)prompt_tokens.size(), params.max_tokens > 0 ? params.max_tokens : 128, available_ctx);
            g_last_error = err_msg;
            return nullptr;
        }
        if ((int)prompt_tokens.size() >= available_ctx - 1) {
            g_last_error = "Prompt too long for context size (need at least 1 token for generation)";
            return nullptr;
        }

        // Reject negative max_tokens (0 is allowed and means "use default")
        if (params.max_tokens < 0) {
            g_last_error = "Invalid max_tokens value (must be >= 0)";
            return nullptr;
        }
        const int n_predict = params.max_tokens > 0 ? params.max_tokens : 128;

        // With a drafter, every batch decoded into the target is passed on to
        // the speculative module straight after, as in the server: it decodes
        // the batch into the draft context, and an MTP head reads the target's
        // hidden states for it
        llama_wrapper_context_t * draft = wrapper->draft;
        common_speculative * spec = wrapper->spec;
        const decode_fn decode_dft = [&](common_batch & batch) {
            return common_speculative_process(spec, batch);
        };
        const decode_fn decode_tgt = [&](common_batch & batch) {
            return decode_batch(wrapper->ctx, batch) == 0;
        };
        const decode_fn decode_both = [&](common_batch & batch) {
            return decode_tgt(batch) && (!spec || decode_dft(batch));
        };

        // Both contexts resume the prompt from the same position, the furthest
        // one they can both restore
        const bool cache = params.enable_prefix_caching;
        int resume_from = restore_prompt(wrapper, n_tokens, prefix_len);
        while (draft && resume_from >= 0) {
            const std::vector<int> prompt_ints(prompt_tokens.begin(), prompt_tokens.end());
            const int draft_limit = std::min(resume_from, cache ? findCommonPrefix(draft->cached_tokens, prompt_ints) : 0);
            const int draft_from = restore_prompt(draft, n_tokens, draft_limit);
            if (draft_from < 0 || draft_from == resume_from) {
                resume_from = draft_from;
                break;
            }
            resume_from = restore_prompt(wrapper, n_tokens, draft_from);
        }
        if (resume_from < 0) {
            return nullptr;
        }

        // Decode prompt tokens [from, to) in n_batch chunks with explicit positions
        const int n_batch = (int)llama_n_batch(wrapper->ctx);
        auto decode_prompt = [&](int from, int to, bool output_last) {
            for (int start = from; start < to; start += n_batch) {
                const int end = std::min(to, start + n_batch);
                common_batch batch(wrapper->ctx);
                for (int pos = start; pos < end; pos++) {
                    batch.add(prompt_tokens[pos], pos, 0, output_last && pos == to - 1);
                }
                if (!decode_both(batch)) {
                    return false;
                }
            }
            return true;
        };

        // Decode up to the last prompt token, checkpoint there if the memory
        // can't be trimmed, then decode the last token for its logits
        bool prompt_ok = decode_prompt(resume_from, n_tokens - 1, false);
        if (prompt_ok && cache) {
            save_prompt_checkpoint(wrapper, n_tokens);
            if (draft) {
                save_prompt_checkpoint(draft, n_tokens);
            }
        }
        prompt_ok = prompt_ok && decode_prompt(n_tokens - 1, n_tokens, true);
        if (!prompt_ok) {
            wrapper->checkpoint.clear();
            if (draft) {
                draft->checkpoint.clear();
            }
            g_last_error = "Failed to decode prompt";
            return nullptr;
        }
        if (cache) {
            wrapper->cached_tokens.assign(prompt_tokens.begin(), prompt_tokens.end());
            if (draft) {
                draft->cached_tokens = wrapper->cached_tokens;
            }
        }

        common_params_sampling sampling_params = make_sampling_params(params, wrapper->model);
        common_sampler_ptr sampler(common_sampler_init(wrapper->model, sampling_params));
        if (!sampler) {
            g_last_error = "Failed to initialise sampler";
            return nullptr;
        }

        // Emit a sampled token, returning false once generation should stop
        const llama_vocab * vocab = llama_model_get_vocab(wrapper->model);
        std::string result;
        int n_gen = 0;
        auto emit = [&](llama_token id) {
            if (llama_vocab_is_eog(vocab, id)) {
                return false;
            }
            const std::string piece = common_token_to_piece(wrapper->ctx, id);
            if (params.callback_handle != 0 && !goTokenCallback(params.callback_handle, piece.c_str())) {
                return false;
            }
            result += piece;
            n_gen++;
            for (int j = 0; j < params.stop_words_count; j++) {
                if (result.find(params.stop_words[j]) != std::string::npos) {
                    return false;
                }
            }
            return n_gen < n_predict;
        };

        // id is the latest sampled token: emitted, but not yet decoded at n_past
        llama_tokens history(prompt_tokens);
        llama_pos n_past = n_tokens;
        llama_token id = common_sampler_sample(sampler.get(), wrapper->ctx, -1);
        common_sampler_accept(sampler.get(), id, true);
        if (spec) {
            common_speculative_begin(spec, 0, history);
        }

        // A draft context that can't be trimmed is restored to before each
        // step: after drafting, and after rejected drafted tokens
        const bool draft_ckpt = spec && step_needs_checkpoint(draft, wrapper->n_draft + 1);

        llama_tokens drafted;
        bool generating = emit(id);
        while (generating) {
            drafted.clear();
            if (spec) {
                if (draft_ckpt) {
                    save_step(draft, n_past);
                }
                auto & dp = common_speculative_get_draft_params(spec, 0);
                dp.drafting = true;
                dp.n_max = std::min(wrapper->n_draft, n_predict - n_gen);
                dp.pos0 = n_past;
                dp.id_last = id;
                dp.prompt = &history;
                dp.result = &drafted;
                dp.result_q = nullptr;
                common_speculative_draft(spec);

                // Return the draft context to before id; the verification
                // batch decodes id and the drafted tokens into it again
                if (!roll_back_step(draft, n_past, {}, draft_ckpt, decode_dft)) {
                    g_last_error = "Failed to roll back the draft context";
                    return nullptr;
                }
            }

            // Decode id, and any drafted tokens to verify them in one pass
            const bool target_ckpt = !drafted.empty() && step_needs_checkpoint(wrapper, drafted.size());
            if (target_ckpt) {
                save_step(wrapper, n_past);
            }
            common_batch batch(wrapper->ctx);
            batch.add(id, n_past, 0, true);
            for (size_t i = 0; i < drafted.size(); i++) {
                batch.add(drafted[i], n_past + 1 + (llama_pos)i, 0, true);
            }
            if (llama_process(wrapper->ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                g_last_error = "Failed to decode generated tokens";
                return nullptr;
            }
            if (spec && !common_speculative_process(spec, batch)) {
                g_last_error = "Failed to decode generated tokens into the draft context";
                return nullptr;
            }

            // The target keeps the drafted tokens it agrees with, then samples
            // its own next token
            const llama_tokens accepted = common_sampler_sample_and_accept_n(sampler.get(), wrapper->ctx, drafted);
            const size_t n_accepted = accepted.size() - 1;
            if (spec && !drafted.empty()) {
                common_speculative_accept(spec, 0, (uint16_t)n_accepted);
            }
            if (params.debug && !drafted.empty()) {
                fprintf(stderr, "DEBUG: accepted %zu/%zu drafted tokens at position %d\n",
                        n_accepted, drafted.size(), n_past);
            }

            llama_tokens keep{ id };
            keep.insert(keep.end(), accepted.begin(), accepted.begin() + n_accepted);
            if (n_accepted < drafted.size()) {
                if (!roll_back_step(wrapper, n_past, keep, target_ckpt, decode_tgt)) {
                    g_last_error = "Failed to roll back rejected drafted tokens";
                    return nullptr;
                }
                if (spec && !roll_back_step(draft, n_past, keep, draft_ckpt, decode_dft)) {
                    g_last_error = "Failed to roll back the draft context";
                    return nullptr;
                }
            }
            history.insert(history.end(), keep.begin(), keep.end());
            n_past += (llama_pos)keep.size();

            for (size_t i = 0; i < n_accepted && generating; i++) {
                generating = emit(accepted[i]);
            }
            id = accepted.back();
            if (generating) {
                generating = emit(id);
            }
        }

        // Return allocated string (caller must free)
        char* c_result = (char*)malloc(result.length() + 1);
        if (c_result) {
            memcpy(c_result, result.c_str(), result.length());
            c_result[result.length()] = '\0';
        } else {
            g_last_error = "Failed to allocate memory for result";
        }
        return c_result;

    } catch (const std::exception& e) {
        g_last_error = "Exception during generation: " + std::string(e.what());
        return nullptr;
    }
}

// Simple wrapper that tokenises the prompt and handles prefix caching automatically
char* llama_wrapper_generate(void* ctx, llama_wrapper_generate_params params) {
    if (!ctx) {
        g_last_error = "Context cannot be null";
        return nullptr;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);
    if (!wrapper->ctx) {
        g_last_error = "Context has been freed";
        return nullptr;
    }

    try {
        // Tokenise the prompt
        std::vector<llama_token> prompt_tokens = common_tokenize(wrapper->ctx, params.prompt, true, true);

        if (prompt_tokens.empty()) {
            g_last_error = "Failed to tokenize prompt";
            return nullptr;
        }

        // Convert to int vector for comparison
        std::vector<int> tokens_int(prompt_tokens.begin(), prompt_tokens.end());

        // Find common prefix with cached tokens (only if prefix caching enabled).
        // generate_with_tokens records the new prompt once it has decoded.
        int prefix_len = params.enable_prefix_caching
            ? findCommonPrefix(wrapper->cached_tokens, tokens_int)
            : 0;

        return llama_wrapper_generate_with_tokens(ctx, tokens_int.data(), tokens_int.size(), prefix_len, params);

    } catch (const std::exception& e) {
        g_last_error = "Exception during generation: " + std::string(e.what());
        return nullptr;
    }
}

int llama_wrapper_tokenize(void* ctx, const char* text, int* tokens, int max_tokens) {
    if (!ctx || !text || !tokens) {
        g_last_error = "Invalid parameters for tokenization";
        return -1;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);

    try {
        std::vector<llama_token> token_vec = common_tokenize(wrapper->ctx, text, true, true);

        int count = std::min((int)token_vec.size(), max_tokens);
        for (int i = 0; i < count; i++) {
            tokens[i] = token_vec[i];
        }

        return count;
    } catch (const std::exception& e) {
        g_last_error = "Exception during tokenization: " + std::string(e.what());
        return -1;
    }
}

// Tokenise with dynamic allocation (C manages memory)
// Caller must free the returned tokens array with llama_wrapper_free_tokens
void llama_wrapper_tokenize_alloc(void* ctx, const char* text, int** tokens, int* count) {
    // Initialise outputs to safe defaults
    if (tokens) *tokens = nullptr;
    if (count) *count = -1;

    if (!ctx || !text || !tokens || !count) {
        g_last_error = "Invalid parameters for tokenization";
        return;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);

    try {
        // Tokenise text (no truncation)
        std::vector<llama_token> token_vec = common_tokenize(wrapper->ctx, text, true, true);

        // Allocate exact size needed
        int n_tokens = token_vec.size();
        int* allocated_tokens = (int*)malloc(n_tokens * sizeof(int));
        if (!allocated_tokens) {
            g_last_error = "Failed to allocate memory for tokens";
            return;
        }

        // Copy tokens from vector to allocated array
        for (int i = 0; i < n_tokens; i++) {
            allocated_tokens[i] = token_vec[i];
        }

        // Return pointer and count
        *tokens = allocated_tokens;
        *count = n_tokens;

    } catch (const std::exception& e) {
        g_last_error = "Exception during tokenization: " + std::string(e.what());
        if (tokens && *tokens) {
            free(*tokens);
            *tokens = nullptr;
        }
        if (count) *count = -1;
    }
}

// Free tokens allocated by llama_wrapper_tokenize_alloc
void llama_wrapper_free_tokens(int* tokens) {
    if (tokens) {
        free(tokens);
    }
}

// A model without pooling gives an embedding per token, not one per text
static bool check_pooling(const llama_wrapper_context_t * wrapper) {
    if (llama_pooling_type(wrapper->ctx) == LLAMA_POOLING_TYPE_NONE) {
        g_last_error = "Model has no pooling, so it can't produce one embedding per text";
        return false;
    }
    return true;
}

int llama_wrapper_embeddings(void* ctx, const char* text, float* embeddings, int max_embeddings) {
    if (!ctx || !text || !embeddings) {
        g_last_error = "Invalid parameters for embeddings";
        return -1;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);
    if (!check_pooling(wrapper)) {
        return -1;
    }
    const int n_embd = llama_model_n_embd_out(wrapper->model);
    if (max_embeddings < n_embd) {
        g_last_error = "Embedding buffer is smaller than the model's output width";
        return -1;
    }

    try {
        // Clear KV cache to ensure clean state
        llama_memory_seq_rm(llama_get_memory(wrapper->ctx), 0, -1, -1);

        // Tokenize text
        std::vector<llama_token> tokens = common_tokenize(wrapper->ctx, text, true, true);

        // Models whose tokeniser adds no BOS give no tokens for empty text
        if (tokens.empty()) {
            g_last_error = "Input content cannot be empty";
            return -1;
        }

        // Evaluate tokens; all of them need output for embeddings
        common_batch batch(wrapper->ctx);
        for (size_t i = 0; i < tokens.size(); i++) {
            batch.add(tokens[i], (llama_pos)i, 0, true);
        }
        if (decode_batch(wrapper->ctx, batch) != 0) {
            g_last_error = "Failed to decode tokens for embeddings";
            return -1;
        }

        // Get embeddings from sequence 0 (works for both single and multi-sequence contexts)
        const float* embd = llama_get_embeddings_seq(wrapper->ctx, 0);
        if (!embd) {
            g_last_error = "Failed to get embeddings from context";
            return -1;
        }

        memcpy(embeddings, embd, n_embd * sizeof(float));
        return n_embd;
    } catch (const std::exception& e) {
        g_last_error = "Exception during embedding generation: " + std::string(e.what());
        return -1;
    }
}

int llama_wrapper_embeddings_batch(void* ctx, const char** texts, int n_texts, float* embeddings, int n_embd) {
    if (!ctx || !texts || !embeddings || n_texts <= 0 || n_embd <= 0) {
        g_last_error = "Invalid parameters for batch embeddings";
        return -1;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);
    if (!check_pooling(wrapper)) {
        return -1;
    }
    if (n_embd != llama_model_n_embd_out(wrapper->model)) {
        g_last_error = "Embedding width doesn't match the model's output width";
        return -1;
    }

    try {
        // Clear KV cache to ensure clean state
        llama_memory_clear(llama_get_memory(wrapper->ctx), true);

        // Tokenize all texts
        std::vector<std::vector<llama_token>> all_tokens;
        all_tokens.reserve(n_texts);

        for (int i = 0; i < n_texts; i++) {
            if (!texts[i]) {
                g_last_error = "Null text in batch at index " + std::to_string(i);
                return -1;
            }
            std::vector<llama_token> tokens = common_tokenize(wrapper->ctx, texts[i], true, true);
            if (tokens.empty()) {
                g_last_error = "Input content cannot be empty (text at index " + std::to_string(i) + ")";
                return -1;
            }
            all_tokens.push_back(std::move(tokens));
        }

        // Get batch size and max sequences
        int n_batch = llama_n_batch(wrapper->ctx);
        int n_seq_max = llama_n_seq_max(wrapper->ctx);

        common_batch batch(wrapper->ctx);

        int embeddings_stored = 0;  // Track how many embeddings we've extracted

        // Process texts in batches
        int s = 0;  // Current sequence ID in batch
        for (int k = 0; k < n_texts; k++) {
            const auto& tokens = all_tokens[k];
            int n_tokens = tokens.size();

            // Check if adding this text would exceed batch size or sequence limit
            if (batch.size() + n_tokens > n_batch || s >= n_seq_max) {
                // Decode current batch
                if (llama_process(wrapper->ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                    g_last_error = "Failed to decode batch";
                    return -1;
                }

                // Extract embeddings for all sequences in this batch
                for (int seq = 0; seq < s; seq++) {
                    const float* embd = llama_get_embeddings_seq(wrapper->ctx, seq);
                    if (!embd) {
                        g_last_error = "Failed to get embeddings for sequence " + std::to_string(seq);
                        return -1;
                    }
                    // Copy embedding to output buffer
                    memcpy(embeddings + embeddings_stored * n_embd, embd, n_embd * sizeof(float));
                    embeddings_stored++;
                }

                // Clear KV cache for processed sequences before resetting
                for (int seq = 0; seq < s; seq++) {
                    llama_memory_seq_rm(llama_get_memory(wrapper->ctx), seq, -1, -1);
                }

                // Reset for next batch
                s = 0;
                batch.clear();
            }

            // Add tokens for this text with unique seq_id
            for (int j = 0; j < n_tokens; j++) {
                // Position is relative to this sequence (starts at 0)
                // All tokens need logits for embeddings
                batch.add(tokens[j], j, s, true);
            }

            s++;  // Move to next sequence ID
        }

        // Process final batch if there are remaining sequences
        if (s > 0) {
            if (llama_process(wrapper->ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                g_last_error = "Failed to decode final batch";
                return -1;
            }

            // Extract embeddings for remaining sequences
            for (int seq = 0; seq < s; seq++) {
                const float* embd = llama_get_embeddings_seq(wrapper->ctx, seq);
                if (!embd) {
                    g_last_error = "Failed to get embeddings for final sequence " + std::to_string(seq);
                    return -1;
                }
                memcpy(embeddings + embeddings_stored * n_embd, embd, n_embd * sizeof(float));
                embeddings_stored++;
            }
        }

        // Verify we got all embeddings
        if (embeddings_stored != n_texts) {
            g_last_error = "Embedding count mismatch: expected " + std::to_string(n_texts) +
                          ", got " + std::to_string(embeddings_stored);
            return -1;
        }

        return embeddings_stored;

    } catch (const std::exception& e) {
        g_last_error = "Exception during batch embedding generation: " + std::string(e.what());
        return -1;
    }
}

int llama_wrapper_get_cached_token_count(void* ctx) {
    if (!ctx) {
        g_last_error = "Context cannot be null";
        return -1;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);
    return static_cast<int>(wrapper->cached_tokens.size());
}

// Get the chat template from model metadata
// Returns nullptr if no template is available
const char* llama_wrapper_get_chat_template(void* model) {
    if (!model) {
        return nullptr;
    }

    auto model_wrapper = static_cast<llama_wrapper_model_t*>(model);

    // Get default chat template (name = nullptr)
    const char* tmpl = llama_model_chat_template(model_wrapper->model, nullptr);

    return tmpl;  // May be nullptr if model has no template
}

void llama_wrapper_free_parsed_message(llama_wrapper_parsed_message* msg) {
    if (!msg) return;

    if (msg->content) {
        free(const_cast<char*>(msg->content));
    }
    if (msg->reasoning_content) {
        free(const_cast<char*>(msg->reasoning_content));
    }
    delete msg;
}

void* llama_wrapper_chat_templates_init(void* model, const char* template_override) {
    if (!model) return nullptr;

    auto model_wrapper = static_cast<llama_wrapper_model_t*>(model);
    std::string tmpl_override = template_override ? template_override : "";

    try {
        auto templates = common_chat_templates_init(model_wrapper->model, tmpl_override);
        return templates.release();  // Transfer ownership
    } catch (const std::exception& e) {
        g_last_error = "Failed to load chat template: " + std::string(e.what());
        return nullptr;
    }
}

void llama_wrapper_chat_templates_free(void* templates) {
    if (!templates) return;
    common_chat_templates_free(static_cast<common_chat_templates*>(templates));
}

int llama_wrapper_chat_templates_support_thinking(void* templates) {
    if (!templates) {
        g_last_error = "Templates cannot be null";
        return -1;
    }
    try {
        return common_chat_templates_support_enable_thinking(static_cast<common_chat_templates*>(templates)) ? 1 : 0;
    } catch (const std::exception& e) {
        g_last_error = "Failed to apply chat template: " + std::string(e.what());
        return -1;
    }
}

void* llama_wrapper_chat_session_new(void* model, void* templates, const char** roles, const char** contents,
                                     int n_messages, llama_wrapper_reasoning_format format,
                                     int enable_thinking) {
    if (!model || !templates || !roles || !contents || n_messages <= 0) {
        g_last_error = "Invalid parameters for chat session";
        return nullptr;
    }

    try {
        common_chat_templates_inputs inputs;
        inputs.use_jinja = true;
        inputs.add_generation_prompt = true;
        inputs.reasoning_format = static_cast<common_reasoning_format>(format);
        if (enable_thinking >= 0) {
            inputs.enable_thinking = enable_thinking != 0;
        }
        for (int i = 0; i < n_messages; i++) {
            if (!roles[i] || !contents[i]) {
                g_last_error = "Role or content cannot be null";
                return nullptr;
            }
            common_chat_msg msg;
            msg.role = roles[i];
            msg.content = contents[i];
            inputs.messages.push_back(std::move(msg));
        }

        // The vocab lets the session tokenise its grammar triggers and
        // preserved tokens. Generated text is fed back as text, which the
        // parser matches on.
        const llama_vocab* vocab = llama_model_get_vocab(static_cast<llama_wrapper_model_t*>(model)->model);
        auto* wrapper = new llama_wrapper_chat_session_t{
            common_chat_session(static_cast<common_chat_templates*>(templates), vocab, inputs),
            "",
        };
        wrapper->prompt = wrapper->session.prompt().str();
        return wrapper;
    } catch (const std::exception& e) {
        g_last_error = "Failed to apply chat template: " + std::string(e.what());
        return nullptr;
    }
}

void llama_wrapper_chat_session_free(void* session) {
    delete static_cast<llama_wrapper_chat_session_t*>(session);
}

const char* llama_wrapper_chat_session_prompt(void* session) {
    if (!session) return nullptr;
    return static_cast<llama_wrapper_chat_session_t*>(session)->prompt.c_str();
}

llama_wrapper_parsed_message* llama_wrapper_chat_session_feed(void* session, const char* text, bool final) {
    if (!session || !text) {
        g_last_error = "Invalid parameters for chat session feed";
        return nullptr;
    }

    try {
        auto& chat = static_cast<llama_wrapper_chat_session_t*>(session)->session;
        common_chat_input chunk{std::string(text)};
        const common_chat_msg& msg = final ? chat.finish(chunk) : chat.feed(chunk);

        auto* result = new llama_wrapper_parsed_message;
        result->content = strdup(msg.content.c_str());
        result->reasoning_content = msg.reasoning_content.empty()
            ? nullptr
            : strdup(msg.reasoning_content.c_str());
        return result;
    } catch (const std::exception& e) {
        g_last_error = "Failed to parse chat output: " + std::string(e.what());
        return nullptr;
    }
}

// Get model metadata string value by key
const char* llama_wrapper_model_meta_string(void* model, const char* key) {
    if (!model || !key) return nullptr;

    auto model_wrapper = static_cast<llama_wrapper_model_t*>(model);

    // Use llama.cpp's metadata API with buffer
    static char buffer[2048];  // Static buffer for metadata strings
    int32_t result = llama_model_meta_val_str(model_wrapper->model, key, buffer, sizeof(buffer));

    if (result < 0) {
        return nullptr;  // Key doesn't exist
    }

    return buffer;
}

// Get count of metadata key-value pairs
int llama_wrapper_model_meta_count(void* model) {
    if (!model) return 0;

    auto model_wrapper = static_cast<llama_wrapper_model_t*>(model);
    return llama_model_meta_count(model_wrapper->model);
}

// Get number of CUDA devices
int llama_wrapper_get_gpu_count() {
#ifdef GGML_USE_CUDA
    return ggml_backend_cuda_get_device_count();
#else
    return 0;
#endif
}

// Get GPU device information
bool llama_wrapper_get_gpu_info(int device_id, llama_wrapper_gpu_info* info) {
    if (!info) return false;

#ifdef GGML_USE_CUDA
    int count = ggml_backend_cuda_get_device_count();
    if (device_id < 0 || device_id >= count) return false;

    // Get device description
    ggml_backend_cuda_get_device_description(device_id, info->device_name, sizeof(info->device_name));
    info->device_id = device_id;

    // Get memory info
    size_t free_mem, total_mem;
    ggml_backend_cuda_get_device_memory(device_id, &free_mem, &total_mem);
    info->free_memory_mb = free_mem / (1024 * 1024);
    info->total_memory_mb = total_mem / (1024 * 1024);

    return true;
#else
    return false;
#endif
}

// Get runtime information about model and context
void llama_wrapper_get_runtime_info(void* model, void* ctx, const char* kv_cache_type, llama_wrapper_runtime_info* info) {
    if (!model || !info) return;

    auto model_wrapper = static_cast<llama_wrapper_model_t*>(model);

    // Get layer counts (llama.cpp uses singular "layer" not "layers")
    info->total_layers = llama_model_n_layer(model_wrapper->model);
    // GPU layers loaded is minimum of requested and total layers
    // (can't load more layers than the model has)
    info->gpu_layers = std::min(model_wrapper->n_gpu_layers, info->total_layers);

    if (ctx) {
        auto ctx_wrapper = static_cast<llama_wrapper_context_t*>(ctx);
        info->n_ctx = llama_n_ctx(ctx_wrapper->ctx);
        info->n_batch = llama_n_batch(ctx_wrapper->ctx);

        // Calculate KV cache size properly accounting for GQA/MQA
        // Formula: 2 * n_ctx * (head_dim * n_head_kv) * n_layers * bytes_per_element
        int n_embd = llama_model_n_embd(model_wrapper->model);
        int n_head = llama_model_n_head(model_wrapper->model);
        int n_head_kv = llama_model_n_head_kv(model_wrapper->model);
        int head_dim = n_embd / n_head;

        // Determine element size based on quantization type
        float bytes_per_element = 2.0f;  // Default f16

        if (kv_cache_type) {
            std::string cache_type(kv_cache_type);
            if (cache_type == "f16") {
                bytes_per_element = 2.0f;
            } else if (cache_type == "q8_0") {
                bytes_per_element = 1.125f;  // ~1 byte + overhead
            } else if (cache_type == "q4_0") {
                bytes_per_element = 0.625f;  // ~0.5 bytes + overhead
            }
        }

        // K and V cache: n_ctx * head_dim * n_head_kv * 2 (K+V) * n_layers * element_size
        long long cache_bytes = (long long)info->n_ctx * head_dim * n_head_kv * 2LL * info->total_layers * bytes_per_element;
        info->kv_cache_size_mb = cache_bytes / (1024 * 1024);
    } else {
        // No context - use defaults or zeros
        info->n_ctx = 0;
        info->n_batch = 0;
        info->kv_cache_size_mb = 0;
    }
}

} // extern "C"
