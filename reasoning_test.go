package llama_test

import (
	"context"
	"os"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	llama "github.com/liminalpurple/llama-go"
)

var _ = Describe("Reasoning", func() {
	var model *llama.Model
	var ctx *llama.Context

	messages := []llama.ChatMessage{
		{Role: "user", Content: "What is 2+2? Answer in one word."},
	}

	BeforeEach(func() {
		// The chat model doubles as the reasoning model; models whose template has no
		// thinking support skip these tests
		testModelPath := os.Getenv("TEST_CHAT_MODEL")
		if testModelPath == "" {
			Skip("TEST_CHAT_MODEL environment variable not set")
		}

		var err error
		model, err = llama.LoadModel(testModelPath, llama.WithGPULayers(-1))
		Expect(err).NotTo(HaveOccurred())

		ctx, err = model.NewContext(llama.WithContext(4096))
		Expect(err).NotTo(HaveOccurred())

		supported, err := model.SupportsThinking()
		Expect(err).NotTo(HaveOccurred())
		if !supported {
			Skip("chat template doesn't support thinking")
		}
	})

	AfterEach(func() {
		if ctx != nil {
			Expect(ctx.Close()).To(Succeed())
		}
		if model != nil {
			Expect(model.Close()).To(Succeed())
		}
	})

	Describe("Parsing output", Label("integration", "chat", "reasoning"), func() {
		thinking := llama.ChatOptions{EnableThinking: llama.Bool(true)}

		// What the model would write with thinking on: some templates open
		// the thinking block at the end of the prompt, so the model's output
		// starts inside it. The sample uses <think> tags, so templates that
		// mark thinking differently, such as Gemma 4's, skip these.
		var output string
		BeforeEach(func() {
			prompt, err := model.FormatChatPrompt(messages, thinking)
			Expect(err).NotTo(HaveOccurred())
			promptOff, err := model.FormatChatPrompt(messages, llama.ChatOptions{EnableThinking: llama.Bool(false)})
			Expect(err).NotTo(HaveOccurred())

			opensThinking := strings.HasSuffix(strings.TrimSpace(prompt), "<think>")
			if !opensThinking && !strings.Contains(promptOff, "</think>") {
				Skip("chat template doesn't use <think> tags")
			}
			output = "\nTwo plus two.\n</think>\n\nFour"
			if !opensThinking {
				output = "<think>" + output
			}
		})

		It("should extract reasoning by default", func() {
			content, reasoning, err := llama.ParseChatOutput(model, messages, thinking, output)
			Expect(err).NotTo(HaveOccurred())
			Expect(strings.TrimSpace(reasoning)).To(Equal("Two plus two."))
			Expect(strings.TrimSpace(content)).To(Equal("Four"))
		})

		It("should leave reasoning inline with ReasoningFormatNone", func() {
			inline := thinking
			inline.ReasoningFormat = llama.ReasoningFormatNone
			content, reasoning, err := llama.ParseChatOutput(model, messages, inline, output)
			Expect(err).NotTo(HaveOccurred())
			Expect(reasoning).To(BeEmpty())
			Expect(content).To(ContainSubstring("<think>"))
			Expect(content).To(ContainSubstring("Four"))
		})

		It("should give the same result when fed in pieces", func() {
			var pieces []string
			for i := 0; i < len(output); i += 4 {
				pieces = append(pieces, output[i:min(i+4, len(output))])
			}
			content, reasoning, err := llama.ParseChatOutput(model, messages, thinking, pieces...)
			Expect(err).NotTo(HaveOccurred())
			Expect(strings.TrimSpace(reasoning)).To(Equal("Two plus two."))
			Expect(strings.TrimSpace(content)).To(Equal("Four"))
		})
	})

	Describe("Rendering the prompt", Label("integration", "chat", "reasoning"), func() {
		It("should render the prompt differently with thinking on and off", func() {
			// Templates mark this differently: Qwen adds an empty thinking block
			// when it's off, and Gemma 4 a flag when it's on
			on, err := model.FormatChatPrompt(messages, llama.ChatOptions{EnableThinking: llama.Bool(true)})
			Expect(err).NotTo(HaveOccurred())
			off, err := model.FormatChatPrompt(messages, llama.ChatOptions{EnableThinking: llama.Bool(false)})
			Expect(err).NotTo(HaveOccurred())
			Expect(on).NotTo(Equal(off))
		})

		It("should leave the thinking block to the model when thinking is on", func() {
			prompt, err := model.FormatChatPrompt(messages, llama.ChatOptions{EnableThinking: llama.Bool(true)})
			Expect(err).NotTo(HaveOccurred())
			Expect(prompt).NotTo(ContainSubstring("</think>"))
		})
	})

	Describe("Chat", Label("integration", "chat", "reasoning"), func() {
		opts := func(thinking bool) llama.ChatOptions {
			return llama.ChatOptions{
				MaxTokens:       llama.Int(1024),
				Temperature:     llama.Float32(0),
				EnableThinking:  llama.Bool(thinking),
				ReasoningFormat: llama.ReasoningFormatAuto,
			}
		}

		It("should separate reasoning from content when thinking", func() {
			response, err := ctx.Chat(context.Background(), messages, opts(true))
			Expect(err).NotTo(HaveOccurred())
			Expect(response.ReasoningContent).NotTo(BeEmpty())
			Expect(response.Content).NotTo(ContainSubstring("<think>"))
		})

		It("should leave the template's empty thinking block out of content by default", func() {
			response, err := ctx.Chat(context.Background(), messages, llama.ChatOptions{
				MaxTokens:      llama.Int(64),
				Temperature:    llama.Float32(0),
				EnableThinking: llama.Bool(false),
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(response.Content).NotTo(ContainSubstring("think>"))
			Expect(response.Content).NotTo(BeEmpty())
		})

		It("should return no reasoning when thinking is off", func() {
			response, err := ctx.Chat(context.Background(), messages, opts(false))
			Expect(err).NotTo(HaveOccurred())
			Expect(response.ReasoningContent).To(BeEmpty())
			Expect(response.Content).NotTo(BeEmpty())
		})

		It("should stream the same reasoning and content as Chat", func() {
			response, err := ctx.Chat(context.Background(), messages, opts(true))
			Expect(err).NotTo(HaveOccurred())

			deltas, errs := ctx.ChatStream(context.Background(), messages, opts(true))
			var content, reasoning strings.Builder
			for delta := range deltas {
				content.WriteString(delta.Content)
				reasoning.WriteString(delta.ReasoningContent)
			}
			Expect(<-errs).NotTo(HaveOccurred())

			Expect(reasoning.String()).To(Equal(response.ReasoningContent))
			Expect(content.String()).To(Equal(response.Content))
		})

		It("should cut thinking short at the reasoning budget", func() {
			budgeted := opts(true)
			budgeted.ReasoningBudget = llama.Int(16)

			response, err := ctx.Chat(context.Background(), messages, budgeted)
			Expect(err).NotTo(HaveOccurred())
			Expect(response.Content).NotTo(BeEmpty())

			tokens, err := ctx.Tokenize(response.ReasoningContent)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically("<=", 20))
		})
	})
})
