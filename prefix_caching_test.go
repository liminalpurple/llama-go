package llama_test

import (
	"fmt"
	"os"
	"strings"

	llama "github.com/liminalpurple/llama-go"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Prefix Caching", Label("prefix-caching"), func() {
	var (
		model     *llama.Model
		ctx       *llama.Context
		modelPath string
	)

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_CHAT_MODEL")
		if modelPath == "" {
			Skip("TEST_CHAT_MODEL not set - skipping integration tests")
		}

		var err error
		model, err = llama.LoadModel(modelPath, llama.WithGPULayers(-1))
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		if ctx != nil {
			Expect(ctx.Close()).To(Succeed())
		}
		if model != nil {
			Expect(model.Close()).To(Succeed())
		}
	})

	Context("deterministic generation", func() {
		It("should produce identical results with prefix caching disabled", Label("integration", "gpu"), func() {
			var err error
			ctx, err = model.NewContext(
				llama.WithContext(2048),
				llama.WithPrefixCaching(false),
			)
			Expect(err).NotTo(HaveOccurred())

			seed := uint32(12345)
			prompt := "What is 2+2?"

			results := make([]string, 3)
			for i := 0; i < 3; i++ {
				result, err := ctx.Generate(prompt,
					llama.WithSeed(int(seed)),
					llama.WithMaxTokens(10),
					llama.WithIgnoreEOS(true), // Instruct models may end a raw prompt early
				)
				Expect(err).NotTo(HaveOccurred())
				Expect(result).NotTo(BeEmpty())
				results[i] = result
			}

			// All should be identical even without prefix caching (same seed)
			Expect(results[1]).To(Equal(results[0]), "Second generation should match first")
			Expect(results[2]).To(Equal(results[0]), "Third generation should match first")
		})

		It("should produce identical results regardless of prefix caching setting", Label("integration", "gpu"), func() {
			seed := uint32(12345)
			prompt := "What is 2+2?"

			// Generate with prefix caching enabled
			var err error
			ctx, err = model.NewContext(
				llama.WithContext(2048),
				llama.WithPrefixCaching(true),
			)
			Expect(err).NotTo(HaveOccurred())

			resultWithCache, err := ctx.Generate(prompt,
				llama.WithSeed(int(seed)),
				llama.WithMaxTokens(10),
				llama.WithIgnoreEOS(true), // Instruct models may end a raw prompt early
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(resultWithCache).NotTo(BeEmpty())

			Expect(ctx.Close()).To(Succeed())

			// Generate with prefix caching disabled
			ctx, err = model.NewContext(
				llama.WithContext(2048),
				llama.WithPrefixCaching(false),
			)
			Expect(err).NotTo(HaveOccurred())

			resultWithoutCache, err := ctx.Generate(prompt,
				llama.WithSeed(int(seed)),
				llama.WithMaxTokens(10),
				llama.WithIgnoreEOS(true), // Instruct models may end a raw prompt early
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(resultWithoutCache).NotTo(BeEmpty())

			// Results should be identical (same seed, deterministic sampling)
			Expect(resultWithoutCache).To(Equal(resultWithCache),
				"Results should be identical regardless of prefix caching when using same seed")
		})

		// These reuse one context so the cache is trimmed between calls. Recurrent
		// and sliding-window models can't trim arbitrarily, so a silent trim
		// failure shows up here as output that differs from a fresh context.
		// llama.cpp doesn't promise bit-identical logits across different batch
		// splits, so a prompt where two tokens are almost tied could still differ.
		Context("after reusing the cache", func() {
			first := "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"
			second := "The capital of France is Paris. The capital of Germany is Berlin. The capital of Spain is"

			generateFresh := func(prompt string) string {
				fresh, err := model.NewContext(llama.WithContext(4096), llama.WithPrefixCaching(false))
				Expect(err).NotTo(HaveOccurred())
				defer func() { Expect(fresh.Close()).To(Succeed()) }()

				result, err := fresh.Generate(prompt, llama.WithTemperature(0), llama.WithMaxTokens(20), llama.WithIgnoreEOS(true))
				Expect(err).NotTo(HaveOccurred())
				return result
			}

			BeforeEach(func() {
				var err error
				ctx, err = model.NewContext(llama.WithContext(4096), llama.WithPrefixCaching(true))
				Expect(err).NotTo(HaveOccurred())
			})

			// The shared prefix is longer than a typical sliding window (Gemma 3
			// uses 1024), and the first prompt runs well past it, beyond the
			// n_ubatch slack the windowed cache keeps, so cells the second prompt
			// needs have been overwritten by the time it's reused.
			//
			// Both contexts decode the shared prefix as exactly one batch, so
			// every cached token is computed in the same batch either way and
			// any difference comes from the cache. With a different split, the
			// prefix's last tokens share a batch with different neighbours,
			// which can tip a near-tie later on (as it does on Gemma 4).
			expectLongPrefixReuse := func(opts ...llama.ContextOption) {
				var history, detour strings.Builder
				for i := 1; i <= 120; i++ {
					fmt.Fprintf(&history, "Line %d: the lighthouse keeper logged a calm night. ", i)
				}
				for i := 1; i <= 60; i++ {
					fmt.Fprintf(&detour, "Storm note %d: waves broke over the north rocks at dawn. ", i)
				}
				longFirst := history.String() + detour.String() + "The keeper's favourite colour was"
				longSecond := history.String() + "The keeper's cat was called"

				firstTokens, err := ctx.Tokenize(longFirst)
				Expect(err).NotTo(HaveOccurred())
				secondTokens, err := ctx.Tokenize(longSecond)
				Expect(err).NotTo(HaveOccurred())
				shared := 0
				for shared < len(firstTokens) && shared < len(secondTokens) && firstTokens[shared] == secondTokens[shared] {
					shared++
				}
				contextOpts := append([]llama.ContextOption{llama.WithContext(4096), llama.WithBatch(shared)}, opts...)

				Expect(ctx.Close()).To(Succeed())
				ctx, err = model.NewContext(append(contextOpts, llama.WithPrefixCaching(true))...)
				Expect(err).NotTo(HaveOccurred())
				_, err = ctx.Generate(longFirst, llama.WithTemperature(0), llama.WithMaxTokens(20))
				Expect(err).NotTo(HaveOccurred())
				reused, err := ctx.Generate(longSecond, llama.WithTemperature(0), llama.WithMaxTokens(20), llama.WithIgnoreEOS(true))
				Expect(err).NotTo(HaveOccurred())

				fresh, err := model.NewContext(append(contextOpts, llama.WithPrefixCaching(false))...)
				Expect(err).NotTo(HaveOccurred())
				defer func() { Expect(fresh.Close()).To(Succeed()) }()
				expected, err := fresh.Generate(longSecond, llama.WithTemperature(0), llama.WithMaxTokens(20), llama.WithIgnoreEOS(true))
				Expect(err).NotTo(HaveOccurred())
				Expect(reused).To(Equal(expected))
			}

			It("should match a fresh context when a long prompt shares a prefix", Label("integration"), func() {
				expectLongPrefixReuse()
			})

			It("should match a fresh context when a long prompt shares a prefix with a full-size SWA cache", Label("integration"), func() {
				expectLongPrefixReuse(llama.WithSWAFull(true))
			})

			It("should match a fresh context when the prompt shares a prefix", Label("integration"), func() {
				_, err := ctx.Generate(first, llama.WithTemperature(0), llama.WithMaxTokens(20))
				Expect(err).NotTo(HaveOccurred())

				reused, err := ctx.Generate(second, llama.WithTemperature(0), llama.WithMaxTokens(20), llama.WithIgnoreEOS(true))
				Expect(err).NotTo(HaveOccurred())
				Expect(reused).To(Equal(generateFresh(second)))
			})

			It("should match a fresh context when the prompt repeats exactly", Label("integration"), func() {
				_, err := ctx.Generate(first, llama.WithTemperature(0), llama.WithMaxTokens(20))
				Expect(err).NotTo(HaveOccurred())

				reused, err := ctx.Generate(first, llama.WithTemperature(0), llama.WithMaxTokens(20), llama.WithIgnoreEOS(true))
				Expect(err).NotTo(HaveOccurred())
				Expect(reused).To(Equal(generateFresh(first)))
			})
		})
	})

	Context("performance", func() {
		It("should reuse cached tokens for repeated prompts", Label("integration", "gpu"), func() {
			var err error
			ctx, err = model.NewContext(
				llama.WithContext(2048),
				llama.WithPrefixCaching(true),
			)
			Expect(err).NotTo(HaveOccurred())

			prompt := "The quick brown fox"

			// First generation establishes cache
			result1, err := ctx.Generate(prompt,
				llama.WithMaxTokens(5),
				llama.WithIgnoreEOS(true), // Instruct models may end a raw prompt early
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result1).NotTo(BeEmpty())

			// Second generation should reuse cache (faster)
			result2, err := ctx.Generate(prompt,
				llama.WithMaxTokens(5),
				llama.WithIgnoreEOS(true), // Instruct models may end a raw prompt early
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result2).NotTo(BeEmpty())

			// Results may differ (no seed), but both should succeed
		})

		It("should handle partial cache hits correctly", Label("integration", "gpu"), func() {
			var err error
			ctx, err = model.NewContext(
				llama.WithContext(2048),
				llama.WithPrefixCaching(true),
			)
			Expect(err).NotTo(HaveOccurred())

			basePrompt := "The quick brown"
			extendedPrompt := "The quick brown fox"

			// Establish cache with base prompt
			_, err = ctx.Generate(basePrompt,
				llama.WithMaxTokens(3),
			)
			Expect(err).NotTo(HaveOccurred())

			// Extended prompt should reuse partial cache
			result, err := ctx.Generate(extendedPrompt,
				llama.WithMaxTokens(3),
				llama.WithIgnoreEOS(true), // Instruct models may end a raw prompt early
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeEmpty())
		})
	})

	Context("cache invalidation", func() {
		It("should not reuse cache when prefix caching is disabled", Label("integration", "gpu"), func() {
			var err error
			ctx, err = model.NewContext(
				llama.WithContext(2048),
				llama.WithPrefixCaching(true),
			)
			Expect(err).NotTo(HaveOccurred())

			prompt := "Hello world"

			// First generation with caching enabled
			_, err = ctx.Generate(prompt,
				llama.WithMaxTokens(5),
			)
			Expect(err).NotTo(HaveOccurred())

			Expect(ctx.Close()).To(Succeed())

			// Second generation with caching disabled should not reuse cache
			ctx, err = model.NewContext(
				llama.WithContext(2048),
				llama.WithPrefixCaching(false),
			)
			Expect(err).NotTo(HaveOccurred())

			result, err := ctx.Generate(prompt,
				llama.WithMaxTokens(5),
				llama.WithIgnoreEOS(true), // Instruct models may end a raw prompt early
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeEmpty())
		})

		It("should handle alternating cache settings correctly", Label("integration", "gpu"), func() {
			prompt := "Test prompt"
			seed := int(54321)

			// Generate with cache enabled
			var err error
			ctx, err = model.NewContext(
				llama.WithContext(2048),
				llama.WithPrefixCaching(true),
			)
			Expect(err).NotTo(HaveOccurred())

			result1, err := ctx.Generate(prompt,
				llama.WithSeed(seed),
				llama.WithMaxTokens(5),
				llama.WithIgnoreEOS(true), // Instruct models may end a raw prompt early
			)
			Expect(err).NotTo(HaveOccurred())

			Expect(ctx.Close()).To(Succeed())

			// Generate with cache disabled
			ctx, err = model.NewContext(
				llama.WithContext(2048),
				llama.WithPrefixCaching(false),
			)
			Expect(err).NotTo(HaveOccurred())

			result2, err := ctx.Generate(prompt,
				llama.WithSeed(seed),
				llama.WithMaxTokens(5),
				llama.WithIgnoreEOS(true), // Instruct models may end a raw prompt early
			)
			Expect(err).NotTo(HaveOccurred())

			Expect(ctx.Close()).To(Succeed())

			// Generate with cache enabled again
			ctx, err = model.NewContext(
				llama.WithContext(2048),
				llama.WithPrefixCaching(true),
			)
			Expect(err).NotTo(HaveOccurred())

			result3, err := ctx.Generate(prompt,
				llama.WithSeed(seed),
				llama.WithMaxTokens(5),
				llama.WithIgnoreEOS(true), // Instruct models may end a raw prompt early
			)
			Expect(err).NotTo(HaveOccurred())

			// All should be identical (same seed)
			Expect(result2).To(Equal(result1))
			Expect(result3).To(Equal(result1))
		})
	})

})
