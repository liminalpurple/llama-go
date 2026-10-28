# Usage Guide

> **Looking for API reference?** See the
> [godoc reference](https://pkg.go.dev/github.com/tcpipuk/llama-go) for complete API documentation
> with examples.

This guide focuses on patterns, best practices, and integration examples for llama-go.

## Architecture overview

The library uses a **Model/Context separation** architecture:

- **Model**: Loads model weights and metadata (GGUF file). Thread-safe, manages vocabulary and
  chat templates.
- **Context**: Handles execution and inference. Each context has its own KV cache and state. Not
  thread-safe (use one context per goroutine).

This separation enables efficient VRAM usage - load the model once, create multiple contexts with
different configurations as needed.

## Basic patterns

### Simple text generation

The most basic pattern: load model, create context, generate text.

```go
package main

import (
    "fmt"
    llama "github.com/tcpipuk/llama-go"
)

func main() {
    // Load model weights (ModelOption: WithGPULayers, WithMLock, etc.)
    model, err := llama.LoadModel(
        "model.gguf",
        llama.WithGPULayers(-1), // -1 = offload all layers
    )
    if err != nil {
        panic(err)
    }
    defer model.Close()

    // Create execution context (ContextOption: WithContext, WithBatch, etc.)
    ctx, err := model.NewContext(
        llama.WithContext(2048),
        llama.WithThreads(8),
    )
    if err != nil {
        panic(err)
    }
    defer ctx.Close()

    // Generate text
    response, err := ctx.Generate("The capital of France is", llama.WithMaxTokens(50))
    if err != nil {
        panic(err)
    }
    fmt.Println(response)
}
```

### Chat completion

Use structured chat messages with automatic chat template formatting:

```go
// Load model and create context (same as above)
model, _ := llama.LoadModel("model.gguf", llama.WithGPULayers(-1))
defer model.Close()

ctx, _ := model.NewContext(llama.WithContext(8192))
defer ctx.Close()

// Build conversation
messages := []llama.ChatMessage{
    {Role: "system", Content: "You are a helpful assistant."},
    {Role: "user", Content: "What is the capital of France?"},
}

// Generate response
response, err := ctx.Chat(context.Background(), messages, llama.ChatOptions{
    MaxTokens: llama.Int(100),
    Temperature: llama.Float32(0.7),
})
if err != nil {
    panic(err)
}

fmt.Println(response.Content)
```

The chat template is automatically selected from the model's metadata, formatting messages
according to the model's training format (ChatML, Llama-2, etc.).

### Streaming generation

Stream tokens as they're generated using callbacks or channels:

```go
// Callback-based streaming
ctx.GenerateStream(
    "Tell me a story",
    func(token string) bool {
        fmt.Print(token)
        return true // continue generation
    },
    llama.WithMaxTokens(200),
)

// Channel-based streaming (supports context cancellation)
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

ch, errCh := ctx.GenerateChannel(ctx, "Tell me a story", llama.WithMaxTokens(200))
for {
    select {
    case token, ok := <-ch:
        if !ok {
            // Generation complete
            return
        }
        fmt.Print(token)
    case err := <-errCh:
        if err != nil {
            panic(err)
        }
        return
    case <-ctx.Done():
        fmt.Println("\nGeneration cancelled")
        return
    }
}
```

### Images and audio

Vision and audio models ship a multimodal projector (`mmproj-*.gguf`) beside the model file. Load
it with the model, then attach images or audio clips to chat messages as encoded files: PNG, JPEG,
WAV, MP3, FLAC and others are detected from their bytes.

```go
model, _ := llama.LoadModel("Qwen3.5-0.8B-Q8_0.gguf", llama.WithMMProj("mmproj-F16.gguf"))
defer model.Close()

ctx, _ := model.NewContext(llama.WithContext(4096))
defer ctx.Close()

photo, _ := os.ReadFile("photo.jpg")
response, _ := ctx.Chat(context.Background(), []llama.ChatMessage{
    {Role: "user", Content: "What's in this photo?", Media: [][]byte{photo}},
}, llama.ChatOptions{})
```

Media goes at the start of its message, in order. To place it elsewhere, put the model's
`MediaMarker()` in the content once per item. The marker is random for each process, as in
llama.cpp's server, so text from users can't place media (`LLAMA_MEDIA_MARKER` pins it). For raw
prompts, use the marker with `WithMedia`:

```go
prompt := "Compare " + model.MediaMarker() + " with " + model.MediaMarker()
text, _ := ctx.Generate(prompt, llama.WithMedia(first, second))
```

`SupportsVision()` and `SupportsAudio()` report what the projector reads. Prefix caching works
through media: a prompt that repeats an earlier image reuses it rather than encoding it again, as
media items are recognised by a hash of their bytes. `WithImageMaxTokens` caps how many tokens an
image becomes, trading detail for speed. Speculative decoding doesn't support media input yet.

### Decision models

Decision models answer typed questions with calibrated probabilities rather than generated text,
as llama.cpp's `/v1/systemone` API does: pick one of several options, answer yes or no, or rate on
a scale. They're fast and cheap to ask, which suits monitoring: is someone at the door in this
frame, which team should handle this ticket, does this page mention a given topic.

```go
model, _ := llama.LoadModel("h2o-lightning-4b-Q8_0.gguf", llama.WithMMProj("mmproj-lightning-f16.gguf"))
defer model.Close()

ctx, _ := model.NewContext(llama.WithContext(40960))
defer ctx.Close()

resp, err := ctx.Decide(context.Background(), llama.DecisionRequest{
    State: "Front door camera",
    Questions: map[string]llama.DecisionQuestion{
        "person": {Type: llama.DecisionNoul, Instructions: "Is someone at the door?"},
        "who": {Type: llama.DecisionChoice, Instructions: "Who is it?",
            Choices: map[string]any{"courier": "delivering a parcel", "visitor": "anyone else"}},
        "risk": {Type: llama.DecisionScore, Instructions: "How concerning is this?",
            Levels: []any{"normal", "unusual", "alarming"}},
    },
    Media: [][]byte{frameJPEG},
})
if resp.Answers["person"].Noul > 0.8 {
    fmt.Println(resp.Answers["who"].Choice, resp.Answers["who"].Confidence)
}
```

`Model.DecisionType()` names the model's decision type, or is empty for other models. Decisions
run llama.cpp's own decision code, so every type it supports works here, reading images and audio
where the type does. Questions about the same state decode it once, and asking about the same
state again reuses it from the prefix cache.

### Embeddings

Generate vector representations for semantic search and similarity:

```go
// Load embedding model
model, _ := llama.LoadModel("nomic-embed-text-v1.5.Q8_0.gguf")
defer model.Close()

// Create context with embeddings enabled
ctx, _ := model.NewContext(llama.WithEmbeddings())
defer ctx.Close()

// Single embedding
embedding, err := ctx.GetEmbeddings("Hello world")
if err != nil {
    panic(err)
}
fmt.Printf("Embedding dimension: %d\n", len(embedding))

// Batch embeddings (more efficient for multiple texts)
texts := []string{
    "The quick brown fox",
    "jumps over the lazy dog",
    "Machine learning is fascinating",
}
embeddings, err := ctx.GetEmbeddingsBatch(texts)
if err != nil {
    panic(err)
}
fmt.Printf("Generated %d embeddings\n", len(embeddings))
```

Some embedding models are trained on a shorter context than they can be used at. nomic-embed-text-v1.5
is trained at 2048 tokens, and its model card recommends YaRN scaling for the full 8192:

```go
ctx, err := model.NewContext(
    llama.WithEmbeddings(),
    llama.WithContext(8192),
    llama.WithRopeScaling("yarn"),
    llama.WithRopeFreqScale(0.75),
)
```

Scaling changes the embedding of every input, short ones included, so vectors produced with
and without it shouldn't be mixed in one index.

## Model vs Context

### Model responsibilities

The Model loads weights from disk and provides:

- **Weight management**: Loads GGUF file into memory (RAM/VRAM)
- **Metadata access**: Vocabulary size, context length, architecture
- **Chat templates**: Automatic message formatting for different model types
- **Context creation**: Factory for execution contexts

**Model is thread-safe** - multiple goroutines can create contexts from the same model concurrently.

### Context responsibilities

The Context handles execution and maintains:

- **KV cache**: Stores attention states for generated tokens
- **Inference state**: Current position, sampling configuration
- **Tokenization**: Convert text to tokens and back
- **Generation**: Actual LLM inference operations

**Context is NOT thread-safe** - each context should be used by a single goroutine. For concurrent
inference, create multiple contexts.

### Why separate Model and Context?

**Efficient VRAM usage**: Load model weights once (e.g. 7GB for Llama-2-7B), create multiple
contexts with different configurations. Each context only needs ~100-500MB for its KV cache.

**Flexible configurations**: Use the same model with different context sizes, batch sizes, or
thread counts:

```go
model, _ := llama.LoadModel("model.gguf", llama.WithGPULayers(-1))
defer model.Close()

// Small context for quick tokenization
tokenCtx, _ := model.NewContext(llama.WithContext(512))
defer tokenCtx.Close()

// Large context for long conversations
chatCtx, _ := model.NewContext(llama.WithContext(32768))
defer chatCtx.Close()

// Embedding context (different settings)
embedCtx, _ := model.NewContext(llama.WithEmbeddings())
defer embedCtx.Close()
```

**Clear resource ownership**: Model owns weights, Context owns execution state. Close contexts
when finished, close model when done with weights.

## Option system

Options use the functional options pattern, separated into ModelOption and ContextOption.

**Model options** (affect weight loading): `WithGPULayers`, `WithMLock`, `WithMMap`,
`WithMainGPU`, `WithTensorSplit`, `WithSilentLoading`, `WithProgressCallback`, `WithMTPHead`,
`WithMMProj`, `WithImageMinTokens`, `WithImageMaxTokens`

**Context options** (affect execution): `WithContext`, `WithBatch`, `WithUBatch`, `WithThreads`,
`WithThreadsBatch`, `WithF16Memory`, `WithEmbeddings`, `WithKVCacheType`, `WithFlashAttn`,
`WithParallel`, `WithPrefixCaching`, `WithSWAFull`, `WithDraft`, `WithMTP`, `WithMTPModel`,
`WithDraftTokens`

**RoPE options** (extend context beyond the training length): `WithRopeScaling`,
`WithRopeFreqBase`, `WithRopeFreqScale`, `WithYarnExtFactor`, `WithYarnAttnFactor`,
`WithYarnBetaFast`, `WithYarnBetaSlow`, `WithYarnOrigCtx`

The RoPE options validate their values: `NewContext` returns an error for an unknown scaling
method or an out-of-range number.

See the [option documentation](https://pkg.go.dev/github.com/tcpipuk/llama-go#pkg-functions) for
complete details, defaults, and usage examples.

## Thread safety

### Model thread safety

**Model is fully thread-safe**. Multiple goroutines can:

- Create contexts concurrently
- Access chat templates
- Query model stats
- Call any Model method

Example (safe):

```go
model, _ := llama.LoadModel("model.gguf", llama.WithGPULayers(-1))
defer model.Close()

// Spawn multiple goroutines using the same model
var wg sync.WaitGroup
for i := 0; i < 10; i++ {
    wg.Add(1)
    go func(id int) {
        defer wg.Done()

        // Each goroutine creates its own context
        ctx, _ := model.NewContext(llama.WithContext(2048))
        defer ctx.Close()

        result, _ := ctx.Generate(fmt.Sprintf("Task %d:", id))
        fmt.Printf("Worker %d: %s\n", id, result)
    }(i)
}
wg.Wait()
```

### Context thread safety

**Context is NOT thread-safe**. Each context should be used by only one goroutine at a time.

**Wrong** (race condition):

```go
ctx, _ := model.NewContext(llama.WithContext(2048))
defer ctx.Close()

// ❌ UNSAFE - two goroutines using same context
go ctx.Generate("Task 1")
go ctx.Generate("Task 2")
```

**Correct** (separate contexts):

```go
// ✅ SAFE - each goroutine has its own context
go func() {
    ctx, _ := model.NewContext(llama.WithContext(2048))
    defer ctx.Close()
    ctx.Generate("Task 1")
}()

go func() {
    ctx, _ := model.NewContext(llama.WithContext(2048))
    defer ctx.Close()
    ctx.Generate("Task 2")
}()
```

### Why contexts aren't thread-safe

Contexts maintain mutable state during inference:

- KV cache position tracking
- Token processing queues
- Sampling state

Making contexts thread-safe would require expensive locking that would hurt performance. Instead,
create cheap contexts (they share model weights) and use one per goroutine.

## Resource management

### Cleanup pattern

Always use `defer` for deterministic cleanup:

```go
// Basic pattern
model, err := llama.LoadModel("model.gguf")
if err != nil {
    return err
}
defer model.Close()

ctx, err := model.NewContext(llama.WithContext(2048))
if err != nil {
    return err
}
defer ctx.Close()

// Use model and context...
```

### What Close() does

**Model.Close()**:

- Frees model weights from RAM/VRAM
- Releases chat template cache
- Cleans up progress callbacks

**Context.Close()**:

- Frees KV cache
- Releases inference buffers
- Cleans up tokenization state

Both are idempotent (safe to call multiple times).

### Finalizers vs explicit cleanup

The library uses finalizers for safety - if you forget to call Close(), resources will eventually
be freed by the garbage collector. However:

- ✅ **Explicit Close()**: Immediate resource release, predictable memory usage
- ⚠️ **Finalizers only**: Delayed cleanup, unpredictable timing, VRAM might stay allocated

**Always prefer explicit Close() with defer** for production code.

### Handling errors during cleanup

Close() is designed to never fail in normal usage:

```go
// Simple defer is usually fine
defer model.Close()
defer ctx.Close()

// If you need to check errors:
defer func() {
    if err := ctx.Close(); err != nil {
        log.Printf("Warning: context cleanup error: %v", err)
    }
}()
```

## Advanced patterns

### Speculative decoding

Use a small draft model to propose tokens that the target model verifies in parallel. Typical
speedup: 1.5-3×.

```go
// Load both models
target, _ := llama.LoadModel("llama-2-70b.gguf", llama.WithGPULayers(-1))
defer target.Close()

draft, _ := llama.LoadModel("llama-2-7b.gguf", llama.WithGPULayers(0)) // Keep on CPU
defer draft.Close()

// A context on the target that speculates with the draft model
ctx, _ := target.NewContext(
    llama.WithContext(2048),
    llama.WithDraft(draft),
    llama.WithDraftTokens(4), // Propose up to 4 tokens per step (default 3)
)
defer ctx.Close()

// Generate, Chat and their streaming variants all speculate on this context
result, _ := ctx.Generate("Write a story about", llama.WithMaxTokens(500))

fmt.Println(result)
```

The draft model must stay open while the context uses it. Output matches the target model on its
own, though at temperature 0 a near-tie between two tokens can resolve differently, since the target
verifies several tokens per batch rather than one.

**Best results when**:

- Draft model is 5-10× smaller than target
- Models share similar vocabularies
- Generating many tokens (speedup compounds)
- Draft model runs on CPU, target on GPU

### Multi-token prediction (MTP)

Some models, such as many Qwen3.5 fine-tunes, ship a multi-token prediction head alongside their
weights. It drafts tokens from the model's own hidden states, so speculation needs no second model:

```go
model, _ := llama.LoadModel("qwen3.5-9b-mtp.gguf", llama.WithMTPHead()) // Load the head too
defer model.Close()

if !model.HasMTPHead() {
    log.Fatal("model has no MTP head")
}

ctx, _ := model.NewContext(
    llama.WithMTP(),
    llama.WithDraftTokens(2), // MTP heads usually do best at 2-3
)
defer ctx.Close()

result, _ := ctx.Generate("Write a story about", llama.WithMaxTokens(500))
```

On an RTX 3090 at temperature 0, two drafted tokens per step gave 1.3× the speed of plain
generation for Qwen3.5 0.8B, 1.45× for a Qwen3.5 9B, and 1.8× for a Qwen3.5 27B. The same
near-tie caveat applies as for draft models.

Other models, such as Gemma 4, ship the head as a separate file (`mtp-*.gguf`). Load it like any
model and pass it to `WithMTPModel`; it must stay open while the context uses it:

```go
model, _ := llama.LoadModel("gemma-4-E2B-it-Q4_K_M.gguf")
defer model.Close()
head, _ := llama.LoadModel("mtp-gemma-4-E2B-it-Q8_0.gguf")
defer head.Close()

ctx, _ := model.NewContext(llama.WithMTPModel(head), llama.WithDraftTokens(2))
defer ctx.Close()
```

On the same RTX 3090, Gemma 4 E2B with its head ran 1.3× faster at two drafted tokens.

### Multi-context usage

Efficient pattern for handling multiple concurrent requests:

```go
type InferencePool struct {
    model *llama.Model
}

func NewInferencePool(modelPath string) (*InferencePool, error) {
    model, err := llama.LoadModel(modelPath, llama.WithGPULayers(-1))
    if err != nil {
        return nil, err
    }
    return &InferencePool{model: model}, nil
}

func (p *InferencePool) Generate(prompt string) (string, error) {
    // Each request gets its own context (cheap - shares model weights)
    ctx, err := p.model.NewContext(llama.WithContext(2048))
    if err != nil {
        return "", err
    }
    defer ctx.Close()

    return ctx.Generate(prompt, llama.WithMaxTokens(100))
}

func (p *InferencePool) Close() error {
    return p.model.Close()
}
```

Usage:

```go
pool, _ := NewInferencePool("model.gguf")
defer pool.Close()

// Handle concurrent requests
var wg sync.WaitGroup
for i := 0; i < 100; i++ {
    wg.Add(1)
    go func(id int) {
        defer wg.Done()
        result, _ := pool.Generate(fmt.Sprintf("Request %d:", id))
        fmt.Println(result)
    }(i)
}
wg.Wait()
```

### Prefix caching for system prompts

Cache common prompt prefixes to avoid recomputing system prompts across thousands of generations:

```go
ctx, _ := model.NewContext(
    llama.WithContext(8192),
    llama.WithPrefixCaching(true), // Enable KV cache prefix reuse (the default)
)
defer ctx.Close()

systemPrompt := "You are a helpful coding assistant specialized in Go."

// First generation computes system prompt KV cache
ctx.Chat(context.Background(), []llama.ChatMessage{
    {Role: "system", Content: systemPrompt},
    {Role: "user", Content: "How do I read a file?"},
})

// Subsequent generations with same system prompt reuse cached KV
// (orders of magnitude faster for long system prompts)
ctx.Chat(context.Background(), []llama.ChatMessage{
    {Role: "system", Content: systemPrompt}, // Reuses cache!
    {Role: "user", Content: "How do I write a file?"},
})
```

Recurrent and hybrid models (Mamba, Qwen3.5, Nemotron-H) can't trim their memory to an arbitrary
point, and nor can sliding-window models (Gemma 2/3/4) with their default window-sized cache. For
these, the context saves a checkpoint at the end of each prompt, and reuse resumes from it only when
the next prompt continues the previous one, as in a conversation. A prompt that diverges earlier,
like the second message above, is processed again from the start. For sliding-window models,
`WithSWAFull(true)` keeps the whole context in those layers' cache so any shared prefix can be
reused, at the cost of considerably more memory at long context sizes.

### Custom sampling parameters

Fine-tune generation behaviour with per-request options:

```go
response, _ := ctx.Generate(
    "The weather today is",
    llama.WithMaxTokens(100),
    llama.WithTemperature(0.8),      // Randomness (0.0 = deterministic, 1.0+ = creative)
    llama.WithTopP(0.95),            // Nucleus sampling threshold
    llama.WithTopK(40),              // Consider top K tokens
    llama.WithRepetitionPenalty(1.1), // Penalize repetition
    llama.WithSeed(42),              // Reproducible generation
)
```

### Tokenization utilities

Access low-level tokenization for analysis or custom workflows:

```go
// Count tokens before generation (useful for API rate limits)
tokens, _ := ctx.Tokenize("This is a long prompt that might exceed limits...")
if len(tokens) > 4096 {
    return fmt.Errorf("prompt exceeds 4K token limit: %d tokens", len(tokens))
}

// Analyze vocabulary
for i, token := range tokens {
    fmt.Printf("Token %d: ID %d\n", i, token)
}
```

## Migration from old API

The Model/Context separation is a **breaking change** from the previous API. Here's how to migrate:

### Old API (pre-v2)

```go
// Old: Model did everything
model, _ := llama.NewModel("model.gguf", llama.WithF16Memory(), llama.WithContext(512))
defer model.Close()

response, _ := model.Generate("Hello", llama.WithMaxTokens(50))
```

### New API (v2+)

```go
// New: Separate Model and Context
model, _ := llama.LoadModel("model.gguf")
defer model.Close()

ctx, _ := model.NewContext(llama.WithContext(512), llama.WithF16Memory())
defer ctx.Close()

response, _ := ctx.Generate("Hello", llama.WithMaxTokens(50))
```

### Migration checklist

1. ✅ Replace `llama.NewModel()` with `llama.LoadModel()`
2. ✅ Create contexts with `model.NewContext()`
3. ✅ Move context options (WithContext, WithBatch, WithF16Memory) from LoadModel to NewContext
4. ✅ Move generation calls from `model.Generate()` to `ctx.Generate()`
5. ✅ Add `defer ctx.Close()` for each context
6. ✅ Update concurrent code: create separate contexts per goroutine instead of sharing model

### Key differences

| Old API | New API | Notes |
| --------- | --------- | ------- |
| `NewModel(path, ...opts)` | `LoadModel(path, ...modelOpts)` then `NewContext(...ctxOpts)` | Two-step initialization |
| Options mixed together | ModelOption vs ContextOption | Type-safe separation |
| `model.Generate()` | `ctx.Generate()` | Generation on context |
| Internal context pool | Explicit context creation | More control, clearer ownership |
| `model.Stats().Runtime` | N/A (removed) | Context-specific info removed from model stats |

## Performance tips

### GPU layer offloading

Use `-1` to offload all layers for maximum GPU utilization:

```go
model, _ := llama.LoadModel("model.gguf", llama.WithGPULayers(-1))
```

For mixed CPU/GPU:

```go
// Offload 32 layers to GPU, rest stays on CPU
model, _ := llama.LoadModel("model.gguf", llama.WithGPULayers(32))
```

### KV cache optimization

Use quantized KV cache to save VRAM with minimal quality loss:

```go
ctx, _ := model.NewContext(
    llama.WithContext(8192),
    llama.WithKVCacheType("q8_0"), // 8-bit quantized cache (50% VRAM reduction)
)
```

Options: `"f32"` (default), `"f16"` (50% reduction), `"q8_0"` (75% reduction), `"q4_0"` (87% reduction)

### Flash Attention

Enable Flash Attention for faster attention computation and lower memory usage:

```go
ctx, _ := model.NewContext(
    llama.WithContext(8192),
    llama.WithFlashAttn(), // Requires compatible GPU (Ampere+ for CUDA)
)
```

### Batch size tuning

Larger batches = better GPU utilization, but more VRAM:

```go
// High throughput (needs more VRAM)
ctx, _ := model.NewContext(llama.WithBatch(2048))

// Lower memory usage
ctx, _ := model.NewContext(llama.WithBatch(512))
```

### Thread configuration

For CPU inference, match your physical cores:

```go
ctx, _ := model.NewContext(
    llama.WithThreads(runtime.NumCPU()),
)
```

## Error handling

All methods return meaningful errors. Check critical operations:

```go
model, err := llama.LoadModel("model.gguf")
if err != nil {
    return fmt.Errorf("failed to load model: %w", err)
}
defer model.Close()

ctx, err := model.NewContext(llama.WithContext(8192))
if err != nil {
    return fmt.Errorf("failed to create context: %w", err)
}
defer ctx.Close()

response, err := ctx.Generate("Hello")
if err != nil {
    return fmt.Errorf("generation failed: %w", err)
}
```

Common errors:

- `"model file not found"` - Check file path
- `"failed to allocate VRAM"` - Reduce GPU layers or use smaller model
- `"context size exceeds model maximum"` - Model GGUF has built-in limits
- `"context is closed"` - Attempted to use context after Close()

## API reference

Complete API documentation is available at
[pkg.go.dev/github.com/tcpipuk/llama-go](https://pkg.go.dev/github.com/tcpipuk/llama-go), including
detailed method documentation, examples, and type definitions.

## Next steps

- See [examples/](../examples/) for working code demonstrating all patterns
- Check [building.md](building.md) for hardware acceleration options (CUDA, Metal, Vulkan, etc.)
- Read [getting-started.md](getting-started.md) for installation and setup
- Visit [pkg.go.dev](https://pkg.go.dev/github.com/tcpipuk/llama-go) for complete API reference
