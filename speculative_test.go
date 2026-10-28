package llama_test

import (
	"context"
	"os"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/tcpipuk/llama-go"
)

// The chat model drafts for itself here, so at temperature 0 the target
// accepts nearly every drafted token and output should match plain
// generation. llama.cpp doesn't promise bit-identical logits across batch
// shapes, and speculation verifies several tokens per batch where plain
// generation decodes one, so a prompt where two tokens are almost tied can
// differ. Recurrent layers such as Qwen3.5's use a different kernel for each,
// which makes this likelier there.
var _ = Describe("Speculative decoding", Label("speculative"), func() {
	const prompt = "The capital of France is Paris. The capital of Germany is"

	var (
		targetModel *llama.Model
		draftModel  *llama.Model
		ctx         *llama.Context
	)

	BeforeEach(func() {
		modelPath := os.Getenv("TEST_CHAT_MODEL")
		if modelPath == "" {
			Skip("TEST_CHAT_MODEL not set - skipping integration test")
		}

		var err error
		targetModel, err = llama.LoadModel(modelPath, llama.WithGPULayers(-1))
		Expect(err).NotTo(HaveOccurred())
		draftModel, err = llama.LoadModel(modelPath, llama.WithGPULayers(-1))
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		if ctx != nil {
			Expect(ctx.Close()).To(Succeed())
			ctx = nil
		}
		if draftModel != nil {
			Expect(draftModel.Close()).To(Succeed())
			draftModel = nil
		}
		if targetModel != nil {
			Expect(targetModel.Close()).To(Succeed())
			targetModel = nil
		}
	})

	newDraftContext := func(opts ...llama.ContextOption) *llama.Context {
		c, err := targetModel.NewContext(append([]llama.ContextOption{
			llama.WithContext(2048), llama.WithDraft(draftModel),
		}, opts...)...)
		Expect(err).NotTo(HaveOccurred())
		return c
	}

	generatePlain := func(p string, opts ...llama.GenerateOption) string {
		plain, err := targetModel.NewContext(llama.WithContext(2048))
		Expect(err).NotTo(HaveOccurred())
		defer func() { Expect(plain.Close()).To(Succeed()) }()

		result, err := plain.Generate(p, opts...)
		Expect(err).NotTo(HaveOccurred())
		return result
	}

	Context("creating a context", func() {
		It("should reject a nil draft model", Label("integration"), func() {
			_, err := targetModel.NewContext(llama.WithDraft(nil))
			Expect(err).To(MatchError(ContainSubstring("draft model cannot be nil")))
		})

		It("should reject fewer than one draft token", Label("integration"), func() {
			_, err := targetModel.NewContext(llama.WithDraft(draftModel), llama.WithDraftTokens(0))
			Expect(err).To(MatchError(ContainSubstring("draft tokens must be at least 1")))
		})

		It("should reject a closed draft model", Label("integration"), func() {
			Expect(draftModel.Close()).To(Succeed())
			_, err := targetModel.NewContext(llama.WithDraft(draftModel))
			Expect(err).To(MatchError(ContainSubstring("draft model is closed")))
		})

		It("should reject WithMTP on a model without an MTP head", Label("integration"), func() {
			withHead, err := llama.LoadModel(os.Getenv("TEST_CHAT_MODEL"), llama.WithGPULayers(-1), llama.WithMTPHead())
			Expect(err).NotTo(HaveOccurred())
			defer func() { Expect(withHead.Close()).To(Succeed()) }()
			if withHead.HasMTPHead() {
				Skip("chat model has an MTP head")
			}

			_, err = withHead.NewContext(llama.WithMTP())
			Expect(err).To(MatchError(ContainSubstring("Model has no MTP head")))
		})

		It("should reject a draft model on an embedding context", Label("integration"), func() {
			_, err := targetModel.NewContext(llama.WithDraft(draftModel), llama.WithEmbeddings())
			Expect(err).To(MatchError(ContainSubstring("single-sequence generation context")))
		})

		It("should cap draft tokens below the batch size", Label("integration"), func() {
			ctx = newDraftContext(llama.WithBatch(8), llama.WithDraftTokens(64))
			result, err := ctx.Generate(prompt, llama.WithTemperature(0), llama.WithMaxTokens(20))
			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal(generatePlain(prompt, llama.WithTemperature(0), llama.WithMaxTokens(20))))
		})
	})

	Context("generating", func() {
		It("should match generation without a draft model", Label("integration"), func() {
			ctx = newDraftContext()
			result, err := ctx.Generate(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40))
			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal(generatePlain(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40))))
		})

		It("should match generation without a draft model for one draft token", Label("integration"), func() {
			ctx = newDraftContext(llama.WithDraftTokens(1))
			result, err := ctx.Generate(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40))
			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal(generatePlain(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40))))
		})

		It("should stop after max tokens", Label("integration"), func() {
			ctx = newDraftContext()
			var tokens int
			err := ctx.GenerateStream("Count from one to one hundred: one, two, three,", func(string) bool {
				tokens++
				return true
			}, llama.WithTemperature(0), llama.WithMaxTokens(5))
			Expect(err).NotTo(HaveOccurred())
			Expect(tokens).To(Equal(5))
		})

		It("should stop at a stop word", Label("integration"), func() {
			ctx = newDraftContext()
			result, err := ctx.Generate("The quick brown fox jumps",
				llama.WithTemperature(0), llama.WithMaxTokens(100), llama.WithStopWords("over"))
			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(ContainSubstring("over"))
			Expect(len(result)).To(BeNumerically("<", 50))
		})

		It("should stop when the callback returns false", Label("integration"), func() {
			ctx = newDraftContext()
			var tokens int
			err := ctx.GenerateStream(prompt, func(string) bool {
				tokens++
				return tokens < 3
			}, llama.WithTemperature(0), llama.WithMaxTokens(40))
			Expect(err).NotTo(HaveOccurred())
			Expect(tokens).To(Equal(3))
		})

		It("should stream through a channel", Label("integration"), func() {
			ctx = newDraftContext()
			tokens, errs := ctx.GenerateChannel(context.Background(), prompt,
				llama.WithTemperature(0), llama.WithMaxTokens(40))
			var streamed strings.Builder
			for token := range tokens {
				streamed.WriteString(token)
			}
			Expect(<-errs).NotTo(HaveOccurred())
			Expect(streamed.String()).To(Equal(generatePlain(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40))))
		})

		It("should give the same output for the same seed", Label("integration"), func() {
			ctx = newDraftContext()
			first, err := ctx.Generate(prompt, llama.WithSeed(12345), llama.WithMaxTokens(40))
			Expect(err).NotTo(HaveOccurred())
			second, err := ctx.Generate(prompt, llama.WithSeed(12345), llama.WithMaxTokens(40))
			Expect(err).NotTo(HaveOccurred())
			Expect(second).To(Equal(first))
		})

		It("should match generation without a draft model after reusing the cache", Label("integration"), func() {
			ctx = newDraftContext()
			_, err := ctx.Generate(prompt+" Berlin. The capital of Italy is",
				llama.WithTemperature(0), llama.WithMaxTokens(20))
			Expect(err).NotTo(HaveOccurred())

			result, err := ctx.Generate(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40))
			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal(generatePlain(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40))))
		})

		It("should speculate in chat", Label("integration", "chat"), func() {
			ctx = newDraftContext()
			messages := []llama.ChatMessage{{Role: "user", Content: "Name three primary colours."}}
			opts := llama.ChatOptions{
				MaxTokens:      llama.Int(40),
				Temperature:    llama.Float32(0),
				EnableThinking: llama.Bool(false),
			}
			response, err := ctx.Chat(context.Background(), messages, opts)
			Expect(err).NotTo(HaveOccurred())

			prompt, err := targetModel.FormatChatPrompt(messages, opts)
			Expect(err).NotTo(HaveOccurred())
			generated, err := ctx.Generate(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40))
			Expect(err).NotTo(HaveOccurred())

			Expect(strings.TrimSpace(generated)).NotTo(BeEmpty())
			Expect(response.Content).To(ContainSubstring(strings.TrimSpace(generated)))
		})

		It("should return an error once the draft model is closed", Label("integration"), func() {
			ctx = newDraftContext()
			Expect(draftModel.Close()).To(Succeed())
			_, err := ctx.Generate(prompt, llama.WithMaxTokens(5))
			Expect(err).To(MatchError("draft model is closed"))
		})
	})
})

// TEST_MTP_MODEL names a model with a built-in MTP head. Like self-speculation
// on Qwen3.5, MTP output can differ from plain generation at near-ties, as
// verifying several tokens per batch picks between them differently, so these
// tests check MTP against itself.
var _ = Describe("MTP speculative decoding", Label("speculative", "mtp"), func() {
	const prompt = "The capital of France is Paris. The capital of Germany is"

	var (
		model *llama.Model
		ctx   *llama.Context
	)

	BeforeEach(func() {
		modelPath := os.Getenv("TEST_MTP_MODEL")
		if modelPath == "" {
			Skip("TEST_MTP_MODEL not set - skipping integration test")
		}

		var err error
		model, err = llama.LoadModel(modelPath, llama.WithGPULayers(-1), llama.WithMTPHead())
		Expect(err).NotTo(HaveOccurred())
		Expect(model.HasMTPHead()).To(BeTrue(), "TEST_MTP_MODEL must have a built-in MTP head")
	})

	AfterEach(func() {
		if ctx != nil {
			Expect(ctx.Close()).To(Succeed())
			ctx = nil
		}
		if model != nil {
			Expect(model.Close()).To(Succeed())
			model = nil
		}
	})

	newMTPContext := func(opts ...llama.ContextOption) *llama.Context {
		c, err := model.NewContext(append([]llama.ContextOption{
			llama.WithContext(2048), llama.WithMTP(),
		}, opts...)...)
		Expect(err).NotTo(HaveOccurred())
		return c
	}

	Context("creating a context", func() {
		It("should need the head to be loaded", Label("integration"), func() {
			withoutHead, err := llama.LoadModel(os.Getenv("TEST_MTP_MODEL"), llama.WithGPULayers(-1))
			Expect(err).NotTo(HaveOccurred())
			defer func() { Expect(withoutHead.Close()).To(Succeed()) }()

			_, err = withoutHead.NewContext(llama.WithMTP())
			Expect(err).To(MatchError(ContainSubstring("Load the model with its MTP head")))
		})

		It("should reject a draft model as well", Label("integration"), func() {
			_, err := model.NewContext(llama.WithMTP(), llama.WithDraft(model))
			Expect(err).To(MatchError(ContainSubstring("not both")))
		})

		It("should reject an embedding context", Label("integration"), func() {
			_, err := model.NewContext(llama.WithMTP(), llama.WithEmbeddings())
			Expect(err).To(MatchError(ContainSubstring("single-sequence generation context")))
		})
	})

	Context("generating", func() {
		It("should give the same output for the same seed", Label("integration"), func() {
			ctx = newMTPContext()
			first, err := ctx.Generate(prompt, llama.WithSeed(12345), llama.WithMaxTokens(40))
			Expect(err).NotTo(HaveOccurred())
			Expect(first).NotTo(BeEmpty())
			second, err := ctx.Generate(prompt, llama.WithSeed(12345), llama.WithMaxTokens(40))
			Expect(err).NotTo(HaveOccurred())
			Expect(second).To(Equal(first))
		})

		It("should match a fresh context after reusing the cache", Label("integration"), func() {
			ctx = newMTPContext()
			_, err := ctx.Generate(prompt+" Berlin. The capital of Italy is",
				llama.WithTemperature(0), llama.WithMaxTokens(20))
			Expect(err).NotTo(HaveOccurred())
			reused, err := ctx.Generate(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40))
			Expect(err).NotTo(HaveOccurred())

			fresh := newMTPContext()
			defer func() { Expect(fresh.Close()).To(Succeed()) }()
			expected, err := fresh.Generate(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40))
			Expect(err).NotTo(HaveOccurred())
			Expect(reused).To(Equal(expected))
		})

		It("should stop after max tokens", Label("integration"), func() {
			ctx = newMTPContext(llama.WithDraftTokens(2))
			var tokens int
			err := ctx.GenerateStream("Count from one to one hundred: one, two, three,", func(string) bool {
				tokens++
				return true
			}, llama.WithTemperature(0), llama.WithMaxTokens(7), llama.WithIgnoreEOS(true))
			Expect(err).NotTo(HaveOccurred())
			Expect(tokens).To(Equal(7))
		})

		It("should stop when the callback returns false", Label("integration"), func() {
			ctx = newMTPContext()
			var tokens int
			err := ctx.GenerateStream(prompt, func(string) bool {
				tokens++
				return tokens < 3
			}, llama.WithTemperature(0), llama.WithMaxTokens(40), llama.WithIgnoreEOS(true))
			Expect(err).NotTo(HaveOccurred())
			Expect(tokens).To(Equal(3))
		})

		It("should speculate in chat", Label("integration", "chat"), func() {
			ctx = newMTPContext()
			response, err := ctx.Chat(context.Background(),
				[]llama.ChatMessage{{Role: "user", Content: "Name three primary colours."}},
				llama.ChatOptions{MaxTokens: llama.Int(40), Temperature: llama.Float32(0), EnableThinking: llama.Bool(false)})
			Expect(err).NotTo(HaveOccurred())
			Expect(response.Content).NotTo(BeEmpty())
		})
	})
})
