// Speculative example demonstrates speculative decoding for faster generation.
//
// This program loads two models - a large target model and a smaller draft
// model - and creates a context on the target that speculates with the draft.
// The draft model proposes several tokens at a time, which the target model
// verifies in one pass, so output matches the target model alone but arrives
// sooner when the draft predicts well.
//
// Usage:
//
//	speculative -target large-model.gguf -draft small-model.gguf -p "prompt"
//
// Speculative decoding works best when:
//   - The draft model is significantly smaller than the target (e.g. 1-3B vs 70B)
//   - Both models share a vocabulary and tokenisation
//   - Generation produces many tokens (speedup compounds over longer outputs)
//
// Pass -debug to see how many drafted tokens the target accepts at each step.
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"

	llama "github.com/tcpipuk/llama-go"
)

func main() {
	var (
		targetModel = flag.String("target", "./Qwen3-0.6B-Q8_0.gguf", "path to target (main) model")
		draftModel  = flag.String("draft", "./Qwen3-0.6B-Q8_0.gguf", "path to draft model")
		prompt      = flag.String("p", "The capital of France is", "prompt for generation")
		maxTokens   = flag.Int("n", 100, "maximum number of tokens to generate")
		context     = flag.Int("c", 2048, "context size")
		gpuLayers   = flag.Int("ngl", -1, "number of GPU layers (-1 for all)")
		temp        = flag.Float64("temp", 0.7, "temperature for sampling")
		draftTokens = flag.Int("draft-tokens", 16, "most tokens the draft model proposes per step")
		debug       = flag.Bool("debug", false, "enable debug output")
	)
	flag.Parse()

	// Load target model
	fmt.Printf("Loading target model: %s\n", *targetModel)
	target, err := llama.LoadModel(*targetModel,
		llama.WithGPULayers(*gpuLayers),
		llama.WithMMap(true),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading target model: %v\n", err)
		os.Exit(1)
	}
	defer func() {
		if err := target.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "Error closing target model: %v\n", err)
		}
	}()

	// Load draft model, which must stay open while the context uses it
	fmt.Printf("Loading draft model: %s\n", *draftModel)
	draft, err := llama.LoadModel(*draftModel,
		llama.WithGPULayers(*gpuLayers),
		llama.WithMMap(true),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading draft model: %v\n", err)
		os.Exit(1)
	}
	defer func() {
		if err := draft.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "Error closing draft model: %v\n", err)
		}
	}()

	// Create a target context that speculates with the draft model
	ctx, err := target.NewContext(
		llama.WithContext(*context),
		llama.WithThreads(runtime.NumCPU()),
		llama.WithF16Memory(),
		llama.WithDraft(draft),
		llama.WithDraftTokens(*draftTokens),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating context: %v\n", err)
		os.Exit(1)
	}
	defer func() {
		if err := ctx.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "Error closing context: %v\n", err)
		}
	}()

	fmt.Printf("Models loaded successfully.\n")
	fmt.Printf("Prompt: %s\n", *prompt)
	fmt.Printf("Speculating with up to %d draft tokens per step\n", *draftTokens)

	opts := []llama.GenerateOption{
		llama.WithMaxTokens(*maxTokens),
		llama.WithTemperature(float32(*temp)),
	}
	if *debug {
		opts = append(opts, llama.WithDebug())
	}

	response, err := ctx.Generate(*prompt, opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error generating text: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nResponse: %s\n", response)
}
