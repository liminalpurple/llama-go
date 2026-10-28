package llama_test

import (
	"context"
	"os"
	"path/filepath"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/tcpipuk/llama-go"
)

// speculationPrompt continues the same way however it's computed. Speculation
// verifies several tokens per batch where plain generation decodes one, and
// with quantised weights each batch width takes a different CUDA kernel that
// rounds differently, so a prompt where two tokens are nearly tied (such as a
// list of capitals) can continue differently. Counting gives every kernel the
// same answer, so speculation must match plain generation exactly, as long as
// the model can't choose to stop: an instruct model's end-of-turn token can be
// nearly tied with carrying on, so comparisons ignore EOS.
const speculationPrompt = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10,"

// The chat model drafts for itself here, so the target accepts nearly every
// drafted token unless the test rejects them
var _ = Describe("Speculative decoding", Label("speculative"), func() {
	const prompt = speculationPrompt

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
			result, err := ctx.Generate(prompt, llama.WithTemperature(0), llama.WithMaxTokens(20), llama.WithIgnoreEOS(true))
			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal(generatePlain(prompt, llama.WithTemperature(0), llama.WithMaxTokens(20), llama.WithIgnoreEOS(true))))
		})
	})

	Context("generating", func() {
		It("should match generation without a draft model", Label("integration"), func() {
			ctx = newDraftContext()
			result, err := ctx.Generate(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40), llama.WithIgnoreEOS(true))
			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal(generatePlain(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40), llama.WithIgnoreEOS(true))))
		})

		It("should match generation without a draft model when every draft is rejected", Label("integration"), func() {
			// Rejecting every drafted token rolls back on every step
			ctx = newDraftContext(llama.WithSyntheticAcceptance(0, 0, 0))
			result, err := ctx.Generate(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40), llama.WithIgnoreEOS(true))
			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal(generatePlain(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40), llama.WithIgnoreEOS(true))))
		})

		It("should match generation without a draft model for one draft token", Label("integration"), func() {
			ctx = newDraftContext(llama.WithDraftTokens(1))
			result, err := ctx.Generate(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40), llama.WithIgnoreEOS(true))
			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal(generatePlain(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40), llama.WithIgnoreEOS(true))))
		})

		It("should stop after max tokens", Label("integration"), func() {
			ctx = newDraftContext()
			var tokens int
			err := ctx.GenerateStream("Count from one to one hundred: one, two, three,", func(string) bool {
				tokens++
				return true
			}, llama.WithTemperature(0), llama.WithMaxTokens(5), llama.WithIgnoreEOS(true))
			Expect(err).NotTo(HaveOccurred())
			Expect(tokens).To(Equal(5))
		})

		It("should stop at a stop word", Label("integration"), func() {
			ctx = newDraftContext()
			result, err := ctx.Generate("The quick brown fox jumps",
				llama.WithTemperature(0), llama.WithMaxTokens(100), llama.WithIgnoreEOS(true), llama.WithStopWords("over"))
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
			}, llama.WithTemperature(0), llama.WithMaxTokens(40), llama.WithIgnoreEOS(true))
			Expect(err).NotTo(HaveOccurred())
			Expect(tokens).To(Equal(3))
		})

		It("should stream through a channel", Label("integration"), func() {
			ctx = newDraftContext()
			tokens, errs := ctx.GenerateChannel(context.Background(), prompt,
				llama.WithTemperature(0), llama.WithMaxTokens(40), llama.WithIgnoreEOS(true))
			var streamed strings.Builder
			for token := range tokens {
				streamed.WriteString(token)
			}
			Expect(<-errs).NotTo(HaveOccurred())
			Expect(streamed.String()).To(Equal(generatePlain(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40), llama.WithIgnoreEOS(true))))
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
			_, err := ctx.Generate(prompt+" 11, 12, 13, 14, 15,",
				llama.WithTemperature(0), llama.WithMaxTokens(20), llama.WithIgnoreEOS(true))
			Expect(err).NotTo(HaveOccurred())

			result, err := ctx.Generate(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40), llama.WithIgnoreEOS(true))
			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal(generatePlain(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40), llama.WithIgnoreEOS(true))))
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

// The MTP head is a different, weaker model than its target, so it drafts
// tokens the target rejects; these tests check output against itself, and
// against plain generation when every drafted token is rejected.
func describeMTPGeneration(newContext, newPlainContext func(opts ...llama.ContextOption) *llama.Context) {
	const prompt = speculationPrompt

	Context("generating", func() {
		var ctx *llama.Context

		AfterEach(func() {
			if ctx != nil {
				Expect(ctx.Close()).To(Succeed())
				ctx = nil
			}
		})

		It("should give the same output for the same seed", Label("integration"), func() {
			ctx = newContext()
			first, err := ctx.Generate(prompt, llama.WithSeed(12345), llama.WithMaxTokens(40), llama.WithIgnoreEOS(true))
			Expect(err).NotTo(HaveOccurred())
			Expect(first).NotTo(BeEmpty())
			second, err := ctx.Generate(prompt, llama.WithSeed(12345), llama.WithMaxTokens(40), llama.WithIgnoreEOS(true))
			Expect(err).NotTo(HaveOccurred())
			Expect(second).To(Equal(first))
		})

		It("should match a fresh context after reusing the cache", Label("integration"), func() {
			ctx = newContext()
			_, err := ctx.Generate(prompt+" 11, 12, 13, 14, 15,",
				llama.WithTemperature(0), llama.WithMaxTokens(20), llama.WithIgnoreEOS(true))
			Expect(err).NotTo(HaveOccurred())
			reused, err := ctx.Generate(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40), llama.WithIgnoreEOS(true))
			Expect(err).NotTo(HaveOccurred())

			fresh := newContext()
			defer func() { Expect(fresh.Close()).To(Succeed()) }()
			expected, err := fresh.Generate(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40), llama.WithIgnoreEOS(true))
			Expect(err).NotTo(HaveOccurred())
			Expect(reused).To(Equal(expected))
		})

		It("should match plain generation when every draft is rejected", Label("integration"), func() {
			ctx = newContext(llama.WithDraftTokens(3), llama.WithSyntheticAcceptance(0, 0, 0))
			result, err := ctx.Generate(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40), llama.WithIgnoreEOS(true))
			Expect(err).NotTo(HaveOccurred())

			plain := newPlainContext()
			defer func() { Expect(plain.Close()).To(Succeed()) }()
			expected, err := plain.Generate(prompt, llama.WithTemperature(0), llama.WithMaxTokens(40), llama.WithIgnoreEOS(true))
			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal(expected))
		})

		It("should stop after max tokens", Label("integration"), func() {
			ctx = newContext(llama.WithDraftTokens(2))
			var tokens int
			err := ctx.GenerateStream("Count from one to one hundred: one, two, three,", func(string) bool {
				tokens++
				return true
			}, llama.WithTemperature(0), llama.WithMaxTokens(7), llama.WithIgnoreEOS(true))
			Expect(err).NotTo(HaveOccurred())
			Expect(tokens).To(Equal(7))
		})

		It("should stop when the callback returns false", Label("integration"), func() {
			ctx = newContext()
			var tokens int
			err := ctx.GenerateStream(prompt, func(string) bool {
				tokens++
				return tokens < 3
			}, llama.WithTemperature(0), llama.WithMaxTokens(40), llama.WithIgnoreEOS(true))
			Expect(err).NotTo(HaveOccurred())
			Expect(tokens).To(Equal(3))
		})

		It("should speculate in chat", Label("integration", "chat"), func() {
			ctx = newContext()
			response, err := ctx.Chat(context.Background(),
				[]llama.ChatMessage{{Role: "user", Content: "Name three primary colours."}},
				llama.ChatOptions{MaxTokens: llama.Int(40), Temperature: llama.Float32(0), EnableThinking: llama.Bool(false)})
			Expect(err).NotTo(HaveOccurred())
			Expect(response.Content).NotTo(BeEmpty())
		})
	})
}

// TEST_MTP_MODEL names a model with a built-in MTP head
var _ = Describe("MTP with a built-in head", Label("speculative", "mtp"), func() {
	var model *llama.Model

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
		if model != nil {
			Expect(model.Close()).To(Succeed())
			model = nil
		}
	})

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
			Expect(err).To(MatchError("use only one of WithDraft, WithMTP and WithMTPModel"))
		})

		It("should reject an embedding context", Label("integration"), func() {
			_, err := model.NewContext(llama.WithMTP(), llama.WithEmbeddings())
			Expect(err).To(MatchError(ContainSubstring("single-sequence generation context")))
		})
	})

	describeMTPGeneration(func(opts ...llama.ContextOption) *llama.Context {
		c, err := model.NewContext(append([]llama.ContextOption{llama.WithContext(2048), llama.WithMTP()}, opts...)...)
		Expect(err).NotTo(HaveOccurred())
		return c
	}, func(opts ...llama.ContextOption) *llama.Context {
		c, err := model.NewContext(append([]llama.ContextOption{llama.WithContext(2048)}, opts...)...)
		Expect(err).NotTo(HaveOccurred())
		return c
	})
})

// TEST_MTP_SIDECAR_MODEL names a model whose MTP head is a separate sidecar
// file, such as Gemma 4. As llama.cpp's downloader expects, the head is the
// mtp-*.gguf in the same folder.
var _ = Describe("MTP with a separate head", Label("speculative", "mtp"), func() {
	var model, head *llama.Model

	BeforeEach(func() {
		modelPath := os.Getenv("TEST_MTP_SIDECAR_MODEL")
		if modelPath == "" {
			Skip("TEST_MTP_SIDECAR_MODEL not set - skipping integration test")
		}
		heads, err := filepath.Glob(filepath.Join(filepath.Dir(modelPath), "mtp-*.gguf"))
		Expect(err).NotTo(HaveOccurred())
		Expect(heads).To(HaveLen(1), "expected one mtp-*.gguf beside TEST_MTP_SIDECAR_MODEL")
		headPath := heads[0]

		model, err = llama.LoadModel(modelPath, llama.WithGPULayers(-1))
		Expect(err).NotTo(HaveOccurred())
		head, err = llama.LoadModel(headPath, llama.WithGPULayers(-1))
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		if head != nil {
			Expect(head.Close()).To(Succeed())
			head = nil
		}
		if model != nil {
			Expect(model.Close()).To(Succeed())
			model = nil
		}
	})

	Context("creating a context", func() {
		It("should reject a nil head", Label("integration"), func() {
			_, err := model.NewContext(llama.WithMTPModel(nil))
			Expect(err).To(MatchError(ContainSubstring("MTP head model cannot be nil")))
		})

		It("should reject a draft model as well", Label("integration"), func() {
			_, err := model.NewContext(llama.WithMTPModel(head), llama.WithDraft(head))
			Expect(err).To(MatchError("use only one of WithDraft, WithMTP and WithMTPModel"))
		})

		It("should reject a closed head", Label("integration"), func() {
			Expect(head.Close()).To(Succeed())
			_, err := model.NewContext(llama.WithMTPModel(head))
			Expect(err).To(MatchError("draft model is closed"))
		})
	})

	describeMTPGeneration(func(opts ...llama.ContextOption) *llama.Context {
		c, err := model.NewContext(append([]llama.ContextOption{llama.WithContext(2048), llama.WithMTPModel(head)}, opts...)...)
		Expect(err).NotTo(HaveOccurred())
		return c
	}, func(opts ...llama.ContextOption) *llama.Context {
		c, err := model.NewContext(append([]llama.ContextOption{llama.WithContext(2048)}, opts...)...)
		Expect(err).NotTo(HaveOccurred())
		return c
	})
})
