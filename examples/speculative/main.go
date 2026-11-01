// Speculative example demonstrates speculative decoding for faster generation.
//
// The target model speculates with a smaller draft model, its own built-in
// multi-token prediction (MTP) head, or an MTP head in a separate file, as
// Gemma 4 ships. The drafter proposes several
// tokens at a time, which the target model verifies in one pass, so output
// arrives sooner when the drafter predicts well. Pass -draft "" without -mtp
// to generate without speculation, for comparison.
//
// Usage:
//
//	speculative -target large-model.gguf -draft small-model.gguf -p "prompt"
//	speculative -target mtp-model.gguf -mtp -p "prompt"
//	speculative -target gemma-4.gguf -mtp-head mtp-gemma-4.gguf -p "prompt"
//	speculative -target mtp-model.gguf -draft "" -p "prompt"
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
	"time"

	llama "github.com/liminalpurple/llama-go"
)

func main() {
	var (
		targetModel = flag.String("target", "./Qwen3-0.6B-Q8_0.gguf", "path to target (main) model")
		draftModel  = flag.String("draft", "./Qwen3-0.6B-Q8_0.gguf", "path to draft model, or empty for none")
		mtp         = flag.Bool("mtp", false, "speculate with the target's built-in MTP head instead of a draft model")
		mtpHead     = flag.String("mtp-head", "", "path to a separate MTP head file to speculate with")
		prompt      = flag.String("p", "The capital of France is", "prompt for generation")
		maxTokens   = flag.Int("n", 100, "maximum number of tokens to generate")
		context     = flag.Int("c", 2048, "context size")
		gpuLayers   = flag.Int("ngl", -1, "number of GPU layers (-1 for all)")
		temp        = flag.Float64("temp", 0.7, "temperature for sampling")
		draftTokens = flag.Int("draft-tokens", 3, "most tokens the drafter proposes per step")
		debug       = flag.Bool("debug", false, "enable debug output")
	)
	flag.Parse()

	// Load target model, with its MTP head if speculating with it
	fmt.Printf("Loading target model: %s\n", *targetModel)
	targetOpts := []llama.ModelOption{llama.WithGPULayers(*gpuLayers), llama.WithMMap(true)}
	if *mtp {
		targetOpts = append(targetOpts, llama.WithMTPHead())
	}
	target, err := llama.LoadModel(*targetModel, targetOpts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading target model: %v\n", err)
		os.Exit(1)
	}
	defer func() {
		if err := target.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "Error closing target model: %v\n", err)
		}
	}()

	ctxOpts := []llama.ContextOption{
		llama.WithContext(*context),
		llama.WithThreads(runtime.NumCPU()),
		llama.WithF16Memory(),
		llama.WithDraftTokens(*draftTokens),
	}
	switch {
	case *mtpHead != "":
		// The head must stay open while the context uses it
		fmt.Printf("Loading MTP head: %s\n", *mtpHead)
		head, err := llama.LoadModel(*mtpHead, llama.WithGPULayers(*gpuLayers), llama.WithMMap(true))
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading MTP head: %v\n", err)
			os.Exit(1)
		}
		defer func() {
			if err := head.Close(); err != nil {
				fmt.Fprintf(os.Stderr, "Error closing MTP head: %v\n", err)
			}
		}()
		fmt.Printf("Speculating with the MTP head, up to %d tokens per step\n", *draftTokens)
		ctxOpts = append(ctxOpts, llama.WithMTPModel(head))
	case *mtp:
		fmt.Printf("Speculating with the target's MTP head, up to %d tokens per step\n", *draftTokens)
		ctxOpts = append(ctxOpts, llama.WithMTP())
	case *draftModel != "":
		// The draft model must stay open while the context uses it
		fmt.Printf("Loading draft model: %s\n", *draftModel)
		draft, err := llama.LoadModel(*draftModel, llama.WithGPULayers(*gpuLayers), llama.WithMMap(true))
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading draft model: %v\n", err)
			os.Exit(1)
		}
		defer func() {
			if err := draft.Close(); err != nil {
				fmt.Fprintf(os.Stderr, "Error closing draft model: %v\n", err)
			}
		}()
		fmt.Printf("Speculating with the draft model, up to %d tokens per step\n", *draftTokens)
		ctxOpts = append(ctxOpts, llama.WithDraft(draft))
	default:
		fmt.Printf("Generating without speculation\n")
	}

	ctx, err := target.NewContext(ctxOpts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating context: %v\n", err)
		os.Exit(1)
	}
	defer func() {
		if err := ctx.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "Error closing context: %v\n", err)
		}
	}()

	opts := []llama.GenerateOption{
		llama.WithMaxTokens(*maxTokens),
		llama.WithTemperature(float32(*temp)),
	}
	if *debug {
		opts = append(opts, llama.WithDebug())
	}

	fmt.Printf("Prompt: %s\n\n", *prompt)
	tokens := 0
	start := time.Now()
	err = ctx.GenerateStream(*prompt, func(token string) bool {
		fmt.Print(token)
		tokens++
		return true
	}, opts...)
	elapsed := time.Since(start)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nError generating text: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n\n%d tokens in %.2fs (%.1f tokens/s, including the prompt)\n",
		tokens, elapsed.Seconds(), float64(tokens)/elapsed.Seconds())
}
