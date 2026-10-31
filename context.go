package llama

import (
	gocontext "context"
	"fmt"
	"runtime"
	"runtime/cgo"
	"sync"
	"unsafe"
)

/*
#include "wrapper.h"
#include <stdlib.h>
*/
import "C"

// Context represents an execution context for inference operations.
//
// Context instances maintain their own KV cache and state, allowing independent
// inference operations. Contexts are NOT thread-safe - each context should be
// used by only one goroutine at a time. For concurrent inference, create multiple
// contexts from the same model.
//
// Multiple contexts share model weights, making concurrent inference VRAM-efficient
// (e.g., one 7GB model + 100MB per context vs 7GB per instance).
//
// Resources should be freed with Close() when finished:
//
//	ctx, _ := model.NewContext(llama.WithContext(8192))
//	defer ctx.Close()
//
// See also: Model.NewContext for creating contexts.
type Context struct {
	contextPtr unsafe.Pointer // llama_wrapper_context_t*
	model      *Model
	config     contextConfig
	mu         sync.RWMutex
	closed     bool
}

// Config types are defined in types.go

// Close frees the context and its associated resources.
//
// This method is idempotent - multiple calls are safe and subsequent calls
// return immediately without error.
//
// After Close() is called, all other methods return an error.
//
// Example:
//
//	ctx, _ := model.NewContext()
//	defer ctx.Close()
func (c *Context) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil
	}

	// Remove finaliser FIRST to prevent race with GC
	runtime.SetFinalizer(c, nil)

	// Free context
	if c.contextPtr != nil {
		C.llama_wrapper_context_free(c.contextPtr)
		c.contextPtr = nil
	}

	c.closed = true
	return nil
}

// Tokenize converts text to tokens.
//
// Tokens are integer IDs representing subword units in the model's vocabulary.
// This method is useful for advanced use cases like manual prompt construction,
// token counting, or analysis.
//
// Examples:
//
//	// Count tokens in a prompt
//	tokens, _ := ctx.Tokenize("Hello world")
//	fmt.Printf("Token count: %d\n", len(tokens))
func (c *Context) Tokenize(text string) ([]int32, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, fmt.Errorf("context is closed")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	var tokensPtr *C.int
	var count C.int

	_, errMsg := withLastError(func() struct{} {
		C.llama_wrapper_tokenize_alloc(c.contextPtr, cText, &tokensPtr, &count)
		return struct{}{}
	})

	if tokensPtr != nil {
		defer C.llama_wrapper_free_tokens(tokensPtr)
	}

	if count < 0 || tokensPtr == nil {
		return nil, fmt.Errorf("tokenisation failed: %s", errMsg)
	}

	tokens := (*[1 << 30]C.int)(unsafe.Pointer(tokensPtr))[:count:count]
	result := make([]int32, count)
	for i := 0; i < int(count); i++ {
		result[i] = int32(tokens[i])
	}

	return result, nil
}

// GetCachedTokenCount returns the number of cached tokens (for debugging/metrics).
//
// This method provides insight into prefix caching behaviour, showing how many
// tokens from previous prompts are cached.
//
// Example:
//
//	ctx.Generate("System prompt: You are helpful.\n\nUser: Hello")
//	cached, _ := ctx.GetCachedTokenCount()
//	fmt.Printf("Cached tokens: %d\n", cached)
func (c *Context) GetCachedTokenCount() (int, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return 0, fmt.Errorf("context is closed")
	}

	count, errMsg := withLastError(func() int {
		return int(C.llama_wrapper_get_cached_token_count(c.contextPtr))
	})
	if count < 0 {
		return 0, fmt.Errorf("failed to get cached token count: %s", errMsg)
	}

	return count, nil
}

// GetEmbeddings computes embeddings for the given text.
//
// Embeddings are vector representations useful for semantic search, clustering,
// or similarity tasks. The context must be created with WithEmbeddings() to use
// this method.
//
// See also: GetEmbeddingsBatch for efficient batch processing of multiple texts.
//
// Example:
//
//	ctx, _ := model.NewContext(llama.WithEmbeddings())
//	emb1, _ := ctx.GetEmbeddings("Hello world")
//	emb2, _ := ctx.GetEmbeddings("Hi there")
func (c *Context) GetEmbeddings(text string) ([]float32, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, fmt.Errorf("context is closed")
	}
	if !c.config.embeddings {
		return nil, fmt.Errorf("context was not created with WithEmbeddings")
	}

	nEmbd := int(C.llama_wrapper_model_n_embd_out(c.model.modelPtr))
	if nEmbd <= 0 {
		return nil, fmt.Errorf("invalid embedding dimension: %d", nEmbd)
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	embeddings := make([]C.float, nEmbd)

	count, errMsg := withLastError(func() C.int {
		return C.llama_wrapper_embeddings(c.contextPtr, cText, &embeddings[0], C.int(nEmbd))
	})
	if count < 0 {
		return nil, fmt.Errorf("embedding generation failed: %s", errMsg)
	}

	result := make([]float32, count)
	for i := 0; i < int(count); i++ {
		result[i] = float32(embeddings[i])
	}

	return result, nil
}

// GetEmbeddingsBatch computes embeddings for multiple texts efficiently.
//
// This method processes multiple texts in a single batch operation, which is
// significantly more efficient than calling GetEmbeddings repeatedly. Uses
// parallel sequence processing (configured via WithParallel) to maximise throughput.
//
// The context must be created with WithEmbeddings() to use this method. Batch size
// is limited by WithParallel setting (default 8 for embedding contexts).
//
// See also: GetEmbeddings for single text processing.
//
// Example:
//
//	ctx, _ := model.NewContext(llama.WithEmbeddings())
//	texts := []string{"First", "Second", "Third"}
//	embeddings, _ := ctx.GetEmbeddingsBatch(texts)
func (c *Context) GetEmbeddingsBatch(texts []string) ([][]float32, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, fmt.Errorf("context is closed")
	}
	if !c.config.embeddings {
		return nil, fmt.Errorf("context was not created with WithEmbeddings")
	}

	if len(texts) == 0 {
		return nil, fmt.Errorf("no texts provided")
	}

	// Get the width of the model's output embeddings
	nEmbd := int(C.llama_wrapper_model_n_embd_out(c.model.modelPtr))
	if nEmbd <= 0 {
		return nil, fmt.Errorf("invalid embedding dimension: %d", nEmbd)
	}

	// Convert Go strings to C strings
	cTexts := make([]*C.char, len(texts))
	for i, text := range texts {
		cTexts[i] = C.CString(text)
	}
	defer func() {
		for i := range cTexts {
			C.free(unsafe.Pointer(cTexts[i]))
		}
	}()

	outputSize := len(texts) * nEmbd
	cEmbeddings := make([]C.float, outputSize)

	count, errMsg := withLastError(func() C.int {
		return C.llama_wrapper_embeddings_batch(
			c.contextPtr,
			(**C.char)(unsafe.Pointer(&cTexts[0])),
			C.int(len(texts)),
			&cEmbeddings[0],
			C.int(nEmbd),
		)
	})

	if count < 0 {
		return nil, fmt.Errorf("batch embedding generation failed: %s", errMsg)
	}

	if int(count) != len(texts) {
		return nil, fmt.Errorf("embedding count mismatch: expected %d, got %d", len(texts), count)
	}

	result := make([][]float32, len(texts))
	for i := 0; i < len(texts); i++ {
		result[i] = make([]float32, nEmbd)
		for j := 0; j < nEmbd; j++ {
			result[i][j] = float32(cEmbeddings[i*nEmbd+j])
		}
	}

	return result, nil
}

// Generate generates text from the given prompt.
//
// This method performs synchronous text generation, returning the complete
// result when finished. The context automatically reuses KV cache entries for
// matching prompt prefixes (prefix caching), significantly improving performance
// for conversation-style usage.
//
// Thread safety: Context is NOT thread-safe. Use separate contexts for concurrent
// generation requests (create multiple contexts from the same Model).
//
// See also: GenerateStream for streaming output, Chat for structured conversations.
//
// Examples:
//
//	// Basic generation
//	result, err := ctx.Generate("Once upon a time")
//
//	// With custom parameters
//	result, err := ctx.Generate("Explain quantum physics",
//	    llama.WithMaxTokens(512),
//	    llama.WithTemperature(0.7),
//	)
func (c *Context) Generate(prompt string, opts ...GenerateOption) (string, error) {
	config := defaultGenerateConfig
	for _, opt := range opts {
		opt(&config)
	}

	return c.generateWithConfig(prompt, config, nil)
}

// GenerateStream generates text with streaming output via callback.
//
// The callback receives each generated token as it's produced. Return true to
// continue generation, or false to stop early.
//
// See also: Generate for synchronous generation, GenerateChannel for channel-based
// streaming with context cancellation support.
//
// Examples:
//
//	// Stream to stdout
//	err := ctx.GenerateStream("Tell me a story",
//	    func(token string) bool {
//	        fmt.Print(token)
//	        return true
//	    },
//	)
func (c *Context) GenerateStream(prompt string, callback func(token string) bool, opts ...GenerateOption) error {
	config := defaultGenerateConfig
	for _, opt := range opts {
		opt(&config)
	}

	_, err := c.generateWithConfig(prompt, config, callback)
	return err
}

// GenerateChannel generates text with streaming output via channel.
//
// Returns two channels: one for tokens and one for errors. The token channel
// is closed when generation completes. The error channel receives at most one
// error before closing.
//
// This method supports context cancellation for stopping generation early.
//
// See also: GenerateStream for callback-based streaming, Generate for synchronous
// generation.
//
// Example:
//
//	tokens, errs := ctx.GenerateChannel(context.Background(), "Write a story")
//	for token := range tokens {
//	    fmt.Print(token)
//	}
//	if err := <-errs; err != nil {
//	    log.Fatal(err)
//	}
func (c *Context) GenerateChannel(ctx gocontext.Context, prompt string, opts ...GenerateOption) (<-chan string, <-chan error) {
	tokenChan := make(chan string, 10)
	errChan := make(chan error, 1)

	go func() {
		defer close(tokenChan)
		defer close(errChan)

		err := c.GenerateStream(prompt, func(token string) bool {
			select {
			case <-ctx.Done():
				return false
			case tokenChan <- token:
				return true
			}
		}, opts...)

		if err != nil {
			errChan <- err
		}
	}()

	return tokenChan, errChan
}

// GenerateWithTokens generates text starting from the given tokens.
//
// This is an advanced method for cases where you've already tokenized the prompt
// or want to use cached tokens. For normal usage, use Generate() instead.
//
// Example:
//
//	tokens, _ := ctx.Tokenize("Once upon a time")
//	result, _ := ctx.GenerateWithTokens(tokens)
func (c *Context) GenerateWithTokens(tokens []int32, opts ...GenerateOption) (string, error) {
	config := defaultGenerateConfig
	for _, opt := range opts {
		opt(&config)
	}

	return c.generateWithTokensAndConfig(tokens, config, nil)
}

// GenerateWithTokensStream generates text with streaming from tokens.
//
// Combines GenerateWithTokens and GenerateStream.
//
// Example:
//
//	tokens, _ := ctx.Tokenize("Write a story")
//	err := ctx.GenerateWithTokensStream(tokens,
//	    func(token string) bool {
//	        fmt.Print(token)
//	        return true
//	    },
//	)
func (c *Context) GenerateWithTokensStream(tokens []int32, callback func(token string) bool, opts ...GenerateOption) error {
	config := defaultGenerateConfig
	for _, opt := range opts {
		opt(&config)
	}

	_, err := c.generateWithTokensAndConfig(tokens, config, callback)
	return err
}

// Chat performs conversational generation using chat messages.
//
// This method formats messages using a chat template and generates a response.
// The template can be provided in opts or will be read from the model's GGUF
// metadata. Supports 40+ template formats including ChatML, Llama-2, Llama-3,
// Mistral, Gemma, and Phi-3.
//
// See also: ChatStream for streaming responses, Generate for raw prompt completion.
//
// Example:
//
//	messages := []llama.ChatMessage{
//	    {Role: "system", Content: "You are a helpful assistant."},
//	    {Role: "user", Content: "Hello!"},
//	}
//	response, err := ctx.Chat(context.Background(), messages, llama.ChatOptions{})
func (c *Context) Chat(ctx gocontext.Context, messages []ChatMessage, opts ChatOptions) (*ChatResponse, error) {
	// Delegate to model's Chat implementation but using this context
	return c.model.chatWithContext(ctx, c, messages, opts)
}

// ChatStream performs conversational generation with streaming output.
//
// Returns channels for chat deltas and errors, similar to GenerateChannel.
// Supports context cancellation for early termination.
//
// See also: Chat for synchronous chat completion.
//
// Example:
//
//	deltas, errs := ctx.ChatStream(context.Background(), messages, llama.ChatOptions{})
//	for delta := range deltas {
//	    fmt.Print(delta.Content)
//	}
func (c *Context) ChatStream(ctx gocontext.Context, messages []ChatMessage, opts ChatOptions) (<-chan ChatDelta, <-chan error) {
	// Delegate to model's ChatStream implementation but using this context
	return c.model.chatStreamWithContext(ctx, c, messages, opts)
}

// Internal generation implementations

//export goTokenCallback
func goTokenCallback(handle C.uintptr_t, token *C.char) C.bool {
	h := cgo.Handle(handle)
	callback := h.Value().(func(string) bool)
	return C.bool(callback(C.GoString(token)))
}

// findCommonPrefix returns length of common prefix between two token slices
func findCommonPrefix(a, b []int32) int {
	commonLen := 0
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] != b[i] {
			break
		}
		commonLen++
	}
	return commonLen
}

// validate returns an error for generation options llama.cpp would ignore
func (g generateConfig) validate() error {
	// llama.cpp ignores breakers after the first LLAMA_DRY_MAX_SEQ_BREAKERS
	// distinct ones, with only a log warning
	distinct := make(map[string]struct{}, len(g.drySequenceBreakers))
	for _, breaker := range g.drySequenceBreakers {
		distinct[breaker] = struct{}{}
	}
	if limit := int(C.llama_wrapper_dry_max_seq_breakers()); len(distinct) > limit {
		return fmt.Errorf("at most %d distinct DRY sequence breakers are supported, got %d", limit, len(distinct))
	}
	return nil
}

// checkModelsOpen returns an error if the model, or the draft model this
// context speculates with, has been closed
func (c *Context) checkModelsOpen() error {
	for _, m := range []*Model{c.model, c.config.draftModel} {
		if m == nil {
			continue
		}
		m.mu.RLock()
		closed := m.closed
		m.mu.RUnlock()
		if closed {
			if m == c.model {
				return fmt.Errorf("model is closed")
			}
			return fmt.Errorf("draft model is closed")
		}
	}
	if c.model == nil {
		return fmt.Errorf("model is closed")
	}
	return nil
}

// generateWithConfig is the internal generation implementation
func (c *Context) generateWithConfig(prompt string, config generateConfig, callback func(string) bool) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return "", fmt.Errorf("context is closed")
	}

	if err := c.checkModelsOpen(); err != nil {
		return "", err
	}
	if err := config.validate(); err != nil {
		return "", err
	}

	// Convert prompt to C string
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	// Convert stop words to C array
	var cStopWords **C.char
	var stopWordsCount C.int

	if len(config.stopWords) > 0 {
		stopWordsCount = C.int(len(config.stopWords))
		cStopWordsArray := make([]*C.char, len(config.stopWords))
		for i, word := range config.stopWords {
			cStopWordsArray[i] = C.CString(word)
		}
		defer func() {
			for _, ptr := range cStopWordsArray {
				C.free(unsafe.Pointer(ptr))
			}
		}()
		cStopWords = (**C.char)(unsafe.Pointer(&cStopWordsArray[0]))
	}

	// Set up callback handle if provided
	var handle cgo.Handle
	var callbackHandle C.uintptr_t
	if callback != nil {
		handle = cgo.NewHandle(callback)
		callbackHandle = C.uintptr_t(handle)
		defer handle.Delete()
	}

	// Convert DRY sequence breakers to C array
	var cDryBreakers **C.char
	var dryBreakersCount C.int
	if len(config.drySequenceBreakers) > 0 {
		dryBreakersCount = C.int(len(config.drySequenceBreakers))
		cDryBreakersArray := make([]*C.char, len(config.drySequenceBreakers))
		for i, breaker := range config.drySequenceBreakers {
			cDryBreakersArray[i] = C.CString(breaker)
		}
		defer func() {
			for _, ptr := range cDryBreakersArray {
				C.free(unsafe.Pointer(ptr))
			}
		}()
		cDryBreakers = (**C.char)(unsafe.Pointer(&cDryBreakersArray[0]))
	}

	params := C.llama_wrapper_generate_params{
		prompt:                      cPrompt,
		max_tokens:                  C.int(config.maxTokens),
		temperature:                 C.float(config.temperature),
		top_k:                       C.int(config.topK),
		top_p:                       C.float(config.topP),
		min_p:                       C.float(config.minP),
		typ_p:                       C.float(config.typP),
		top_n_sigma:                 C.float(config.topNSigma),
		penalty_last_n:              C.int(config.penaltyLastN),
		penalty_repeat:              C.float(config.penaltyRepeat),
		penalty_freq:                C.float(config.penaltyFreq),
		penalty_present:             C.float(config.penaltyPresent),
		dry_multiplier:              C.float(config.dryMultiplier),
		dry_base:                    C.float(config.dryBase),
		dry_allowed_length:          C.int(config.dryAllowedLength),
		dry_penalty_last_n:          C.int(config.dryPenaltyLastN),
		dry_sequence_breakers:       cDryBreakers,
		dry_sequence_breakers_count: dryBreakersCount,
		dynatemp_range:              C.float(config.dynatempRange),
		dynatemp_exponent:           C.float(config.dynatempExponent),
		xtc_probability:             C.float(config.xtcProbability),
		xtc_threshold:               C.float(config.xtcThreshold),
		mirostat:                    C.int(config.mirostat),
		mirostat_tau:                C.float(config.mirostatTau),
		mirostat_eta:                C.float(config.mirostatEta),
		n_prev:                      C.int(config.nPrev),
		n_probs:                     C.int(config.nProbs),
		min_keep:                    C.int(config.minKeep),
		seed:                        C.int(config.seed),
		stop_words:                  cStopWords,
		stop_words_count:            stopWordsCount,
		callback_handle:             callbackHandle,
		ignore_eos:                  C.bool(config.ignoreEOS),
		debug:                       C.bool(config.debug),
		chat_session:                config.chatSession.cPtr(),
		reasoning_budget:            C.int(config.reasoningBudget),
		enable_prefix_caching:       C.bool(c.config.prefixCaching),
	}

	// Media goes to C as copies, as cgo can't pass Go pointers inside arrays
	if n := len(config.media); n > 0 {
		cMedia := (*[1 << 28]*C.uchar)(C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof((*C.uchar)(nil)))))[:n:n]
		cSizes := (*[1 << 28]C.size_t)(C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.size_t(0)))))[:n:n]
		defer C.free(unsafe.Pointer(&cMedia[0]))
		defer C.free(unsafe.Pointer(&cSizes[0]))
		for i, item := range config.media {
			if len(item) == 0 {
				return "", fmt.Errorf("media item %d is empty", i)
			}
			cMedia[i] = (*C.uchar)(C.CBytes(item))
			cSizes[i] = C.size_t(len(item))
			defer C.free(unsafe.Pointer(cMedia[i]))
		}
		params.media = &cMedia[0]
		params.media_sizes = &cSizes[0]
		params.n_media = C.int(n)
	}

	// Call C generation function
	cResult, errMsg := withLastError(func() *C.char {
		return C.llama_wrapper_generate(c.contextPtr, params)
	})
	if cResult == nil {
		return "", fmt.Errorf("generation failed: %s", errMsg)
	}

	result := C.GoString(cResult)
	C.llama_wrapper_free_result(cResult)

	return result, nil
}

// generateWithTokensAndConfig generates from pre-tokenized input
func (c *Context) generateWithTokensAndConfig(tokens []int32, config generateConfig, callback func(string) bool) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return "", fmt.Errorf("context is closed")
	}

	if err := c.checkModelsOpen(); err != nil {
		return "", err
	}
	if err := config.validate(); err != nil {
		return "", err
	}

	if len(tokens) == 0 {
		return "", fmt.Errorf("no tokens provided")
	}
	if len(config.media) > 0 {
		return "", fmt.Errorf("media input needs a text prompt with media markers, not tokens")
	}

	// Convert tokens to C array
	cTokens := make([]C.int, len(tokens))
	for i, token := range tokens {
		cTokens[i] = C.int(token)
	}

	// Convert stop words to C array
	var cStopWords **C.char
	var stopWordsCount C.int

	if len(config.stopWords) > 0 {
		stopWordsCount = C.int(len(config.stopWords))
		cStopWordsArray := make([]*C.char, len(config.stopWords))
		for i, word := range config.stopWords {
			cStopWordsArray[i] = C.CString(word)
		}
		defer func() {
			for _, ptr := range cStopWordsArray {
				C.free(unsafe.Pointer(ptr))
			}
		}()
		cStopWords = (**C.char)(unsafe.Pointer(&cStopWordsArray[0]))
	}

	// Set up callback handle if provided
	var handle cgo.Handle
	var callbackHandle C.uintptr_t
	if callback != nil {
		handle = cgo.NewHandle(callback)
		callbackHandle = C.uintptr_t(handle)
		defer handle.Delete()
	}

	// Convert DRY sequence breakers to C array
	var cDryBreakers **C.char
	var dryBreakersCount C.int
	if len(config.drySequenceBreakers) > 0 {
		dryBreakersCount = C.int(len(config.drySequenceBreakers))
		cDryBreakersArray := make([]*C.char, len(config.drySequenceBreakers))
		for i, breaker := range config.drySequenceBreakers {
			cDryBreakersArray[i] = C.CString(breaker)
		}
		defer func() {
			for _, ptr := range cDryBreakersArray {
				C.free(unsafe.Pointer(ptr))
			}
		}()
		cDryBreakers = (**C.char)(unsafe.Pointer(&cDryBreakersArray[0]))
	}

	params := C.llama_wrapper_generate_params{
		prompt:                      nil, // Not used for token generation
		max_tokens:                  C.int(config.maxTokens),
		temperature:                 C.float(config.temperature),
		top_k:                       C.int(config.topK),
		top_p:                       C.float(config.topP),
		min_p:                       C.float(config.minP),
		typ_p:                       C.float(config.typP),
		top_n_sigma:                 C.float(config.topNSigma),
		penalty_last_n:              C.int(config.penaltyLastN),
		penalty_repeat:              C.float(config.penaltyRepeat),
		penalty_freq:                C.float(config.penaltyFreq),
		penalty_present:             C.float(config.penaltyPresent),
		dry_multiplier:              C.float(config.dryMultiplier),
		dry_base:                    C.float(config.dryBase),
		dry_allowed_length:          C.int(config.dryAllowedLength),
		dry_penalty_last_n:          C.int(config.dryPenaltyLastN),
		dry_sequence_breakers:       cDryBreakers,
		dry_sequence_breakers_count: dryBreakersCount,
		dynatemp_range:              C.float(config.dynatempRange),
		dynatemp_exponent:           C.float(config.dynatempExponent),
		xtc_probability:             C.float(config.xtcProbability),
		xtc_threshold:               C.float(config.xtcThreshold),
		mirostat:                    C.int(config.mirostat),
		mirostat_tau:                C.float(config.mirostatTau),
		mirostat_eta:                C.float(config.mirostatEta),
		n_prev:                      C.int(config.nPrev),
		n_probs:                     C.int(config.nProbs),
		min_keep:                    C.int(config.minKeep),
		seed:                        C.int(config.seed),
		stop_words:                  cStopWords,
		stop_words_count:            stopWordsCount,
		callback_handle:             callbackHandle,
		ignore_eos:                  C.bool(config.ignoreEOS),
		debug:                       C.bool(config.debug),
		chat_session:                config.chatSession.cPtr(),
		reasoning_budget:            C.int(config.reasoningBudget),
		enable_prefix_caching:       C.bool(c.config.prefixCaching),
	}

	// Call C generation function with tokens
	cResult, errMsg := withLastError(func() *C.char {
		return C.llama_wrapper_generate_with_tokens(
			c.contextPtr,
			&cTokens[0],
			C.int(len(tokens)),
			C.int(0), // prefix_len - no prefix caching for this function
			params,
		)
	})

	if cResult == nil {
		return "", fmt.Errorf("generation with tokens failed: %s", errMsg)
	}

	result := C.GoString(cResult)
	C.llama_wrapper_free_result(cResult)

	return result, nil
}
