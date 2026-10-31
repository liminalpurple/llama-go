package llama

// Model loading options (model-level only)

// WithGPULayers sets the number of model layers to offload to GPU.
//
// By default, all layers are offloaded to GPU (-1). If GPU acceleration is
// unavailable, the library automatically falls back to CPU execution. Set to 0
// to force CPU-only execution, or specify a positive number to partially
// offload layers (useful for models larger than GPU memory).
//
// Default: -1 (offload all layers, with CPU fallback)
//
// Examples:
//
//	// Force CPU execution
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithGPULayers(0),
//	)
//
//	// Offload 35 layers to GPU, rest on CPU
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithGPULayers(35),
//	)
func WithGPULayers(n int) ModelOption {
	return func(c *modelConfig) {
		c.gpuLayers = n
	}
}

// WithMMProj loads a multimodal projector with the model, so prompts can
// include images or audio (see ChatMessage.Media and WithMedia). The
// projector is the mmproj-*.gguf file published beside a vision or audio
// model, as for llama.cpp's --mmproj.
//
// Example:
//
//	model, err := llama.LoadModel("model.gguf", llama.WithMMProj("mmproj-model-f16.gguf"))
func WithMMProj(path string) ModelOption {
	return func(c *modelConfig) {
		c.mmproj = path
	}
}

// WithImageMinTokens sets the fewest tokens an image is encoded into, as for
// llama.cpp's --image-min-tokens. Default: the projector's own setting.
func WithImageMinTokens(n int) ModelOption {
	return func(c *modelConfig) {
		c.imageMinTokens = n
	}
}

// WithImageMaxTokens sets the most tokens an image is encoded into, as for
// llama.cpp's --image-max-tokens. Larger images are scaled down to fit.
// Default: the projector's own setting.
func WithImageMaxTokens(n int) ModelOption {
	return func(c *modelConfig) {
		c.imageMaxTokens = n
	}
}

// WithMLock forces the model to stay in RAM using mlock().
//
// When enabled, prevents the operating system from swapping model data to disk.
// Useful for production environments where consistent latency is critical, but
// requires sufficient physical RAM and may require elevated privileges.
//
// Default: false (allows OS to manage memory)
//
// Example:
//
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithMLock(),
//	)
func WithMLock() ModelOption {
	return func(c *modelConfig) {
		c.mlock = true
	}
}

// WithMTPHead loads the model's built-in multi-token prediction (MTP) head,
// if it has one, so its contexts can speculate with WithMTP.
//
// Some models, such as many Qwen3.5 fine-tunes, ship an MTP head alongside
// their weights. llama.cpp skips it unless asked, as it costs extra memory.
//
// Example:
//
//	model, err := llama.LoadModel("model.gguf", llama.WithMTPHead())
//	ctx, err := model.NewContext(llama.WithMTP())
func WithMTPHead() ModelOption {
	return func(c *modelConfig) {
		c.mtpHead = true
	}
}

// WithMMap enables or disables memory-mapped file I/O for model loading.
//
// Memory mapping (mmap) allows the OS to load model data on-demand rather than
// reading the entire file upfront. This significantly reduces startup time and
// memory usage. Disable only if you encounter platform-specific issues.
//
// Default: true (enabled)
//
// Example:
//
//	// Disable mmap for compatibility
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithMMap(false),
//	)
func WithMMap(enabled bool) ModelOption {
	return func(c *modelConfig) {
		c.mmap = enabled
	}
}

// WithMainGPU sets the primary GPU device for model execution.
//
// Use this option to select a specific GPU in multi-GPU systems. The device
// string format depends on the backend (e.g. "0" for CUDA device 0). Most
// users with single-GPU systems don't need this option.
//
// Default: "" (uses default GPU)
//
// Example:
//
//	// Use second GPU
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithMainGPU("1"),
//	)
func WithMainGPU(gpu string) ModelOption {
	return func(c *modelConfig) {
		c.mainGPU = gpu
	}
}

// WithTensorSplit configures tensor distribution across multiple GPUs.
//
// Allows manual control of how model layers are distributed across GPUs in
// multi-GPU setups. The split is a list of non-negative proportions, one per
// device in order, separated by ',' or '/' as in llama.cpp's --tensor-split
// (e.g. "0.7,0.3" puts 70% on the first GPU and 30% on the second). Loading
// fails if the split is malformed or names more devices than llama.cpp
// supports. Most users should rely on automatic distribution instead.
//
// Default: "" (automatic distribution)
//
// Example:
//
//	// Distribute 60/40 across two GPUs
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithTensorSplit("0.6,0.4"),
//	)
func WithTensorSplit(split string) ModelOption {
	return func(c *modelConfig) {
		c.tensorSplit = split
	}
}

// WithTensorOverride stores the tensors whose names match a pattern on a chosen
// device, as llama.cpp's --override-tensor does.
//
// The pattern is a regular expression (ECMAScript syntax) matched against
// tensor names such as "blk.12.ffn_up_exps.weight". The device is a name from
// Devices, such as "CPU" or "CUDA0". The option can be given more than once;
// for each tensor the first matching override wins, in the order the options
// were given, and tensors matching none are placed as usual. Loading fails if
// a pattern is invalid or the device doesn't exist.
//
// Example:
//
//	// Keep the feed-forward weights of the last four layers on the CPU
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithTensorOverride(`blk\.(2[89]|3[01])\.ffn_.*`, "CPU"),
//	)
func WithTensorOverride(pattern, device string) ModelOption {
	return func(c *modelConfig) {
		c.tensorOverrides = append(c.tensorOverrides, tensorOverride{pattern: pattern, device: device})
	}
}

// WithCPUMoE keeps all of a mixture-of-experts model's expert weights in system
// memory, as llama.cpp's --cpu-moe does.
//
// Experts make up most of a MoE model's size, but each token only uses a few
// of them, so running them on the CPU costs far less speed than offloading
// fewer layers. The GPU keeps attention, the shared weights and the KV cache.
// It has no effect on models without experts.
//
// Example:
//
//	model, err := llama.LoadModel("Qwen3-30B-A3B-Q4_K_M.gguf",
//	    llama.WithCPUMoE(),
//	)
func WithCPUMoE() ModelOption {
	return func(c *modelConfig) {
		c.tensorOverrides = append(c.tensorOverrides, tensorOverride{expertLayers: -1})
	}
}

// WithNCPUMoE keeps the expert weights of a mixture-of-experts model's first n
// layers in system memory, as llama.cpp's --n-cpu-moe does, so the experts
// of the remaining layers can use whatever VRAM is left. Values of n up to 0
// change nothing.
//
// Example:
//
//	// Experts of layers 0-19 on the CPU, the rest on the GPU
//	model, err := llama.LoadModel("Qwen3-30B-A3B-Q4_K_M.gguf",
//	    llama.WithNCPUMoE(20),
//	)
func WithNCPUMoE(n int) ModelOption {
	return func(c *modelConfig) {
		if n > 0 {
			c.tensorOverrides = append(c.tensorOverrides, tensorOverride{expertLayers: n})
		}
	}
}

// WithSilentLoading disables progress output during model loading.
//
// By default, llama.cpp prints dots to stderr to indicate loading progress.
// This option suppresses that output completely, useful for clean logs in
// production environments or when progress output interferes with other
// output formatting.
//
// Note: The LLAMA_LOG environment variable controls general logging but
// does not suppress progress dots. Use this option for truly silent loading.
//
// Default: false (shows progress dots)
//
// Example:
//
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithSilentLoading(),
//	)
func WithSilentLoading() ModelOption {
	return func(c *modelConfig) {
		c.disableProgressCallback = true
	}
}

// ProgressCallback is called during model loading with progress 0.0-1.0.
// Return false to cancel loading, true to continue.
type ProgressCallback func(progress float32) bool

// WithProgressCallback sets a custom progress callback for model loading.
//
// The callback is invoked periodically during model loading with progress
// values from 0.0 (start) to 1.0 (complete). This allows implementing
// custom progress indicators, logging, or loading cancellation.
//
// The callback receives:
//   - progress: float32 from 0.0 to 1.0 indicating loading progress
//
// The callback must return:
//   - true: continue loading
//   - false: cancel loading (LoadModel will return an error)
//
// IMPORTANT: The callback is invoked from a C thread during model loading.
// Ensure any operations are thread-safe. The callback should complete
// quickly to avoid blocking the loading process.
//
// Default: nil (uses llama.cpp default dot printing)
//
// Examples:
//
//	// Simple progress indicator
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithProgressCallback(func(progress float32) bool {
//	        fmt.Printf("\rLoading: %.0f%%", progress*100)
//	        return true
//	    }),
//	)
//
//	// Cancel loading after 50%
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithProgressCallback(func(progress float32) bool {
//	        if progress > 0.5 {
//	            return false // Cancel
//	        }
//	        return true
//	    }),
//	)
func WithProgressCallback(cb ProgressCallback) ModelOption {
	return func(c *modelConfig) {
		c.progressCallback = cb
	}
}
