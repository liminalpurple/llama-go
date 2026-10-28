package llama_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/tcpipuk/llama-go"
)

// solidPNG encodes a square image of one colour
func solidPNG(c color.Color) []byte {
	img := image.NewRGBA(image.Rect(0, 0, 128, 128))
	for y := 0; y < 128; y++ {
		for x := 0; x < 128; x++ {
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	Expect(png.Encode(&buf, img)).To(Succeed())
	return buf.Bytes()
}

// toneWAV encodes a second of a 440 Hz tone as 16 kHz mono WAV
func toneWAV() []byte {
	const rate = 16000
	samples := make([]int16, rate)
	for i := range samples {
		samples[i] = int16(8000 * math.Sin(2*math.Pi*440*float64(i)/rate))
	}
	var buf bytes.Buffer
	write := func(v any) { Expect(binary.Write(&buf, binary.LittleEndian, v)).To(Succeed()) }
	buf.WriteString("RIFF")
	write(uint32(36 + 2*len(samples)))
	buf.WriteString("WAVEfmt ")
	write(uint32(16))
	write(uint16(1)) // PCM
	write(uint16(1)) // mono
	write(uint32(rate))
	write(uint32(rate * 2))
	write(uint16(2))
	write(uint16(16))
	buf.WriteString("data")
	write(uint32(2 * len(samples)))
	write(samples)
	return buf.Bytes()
}

// loadWithMMProj loads the model a variable names, with the mmproj-*.gguf in
// its folder, as llama.cpp places a projector beside its model
func loadWithMMProj(variable string) *llama.Model {
	modelPath := os.Getenv(variable)
	if modelPath == "" {
		Skip(variable + " not set - skipping integration test")
	}
	projectors, err := filepath.Glob(filepath.Join(filepath.Dir(modelPath), "mmproj-*.gguf"))
	Expect(err).NotTo(HaveOccurred())
	Expect(projectors).To(HaveLen(1), "expected one mmproj-*.gguf beside "+variable)

	model, err := llama.LoadModel(modelPath, llama.WithGPULayers(-1), llama.WithMMProj(projectors[0]))
	Expect(err).NotTo(HaveOccurred())
	return model
}

var _ = Describe("Media input", Label("media"), func() {
	var (
		model *llama.Model
		ctx   *llama.Context
		red   = solidPNG(color.RGBA{255, 0, 0, 255})
		blue  = solidPNG(color.RGBA{0, 0, 255, 255})
		opts  = llama.ChatOptions{MaxTokens: llama.Int(20), Temperature: llama.Float32(0), EnableThinking: llama.Bool(false)}
	)

	ask := func(c *llama.Context, content string, media ...[]byte) string {
		response, err := c.Chat(context.Background(), []llama.ChatMessage{{Role: "user", Content: content, Media: media}}, opts)
		Expect(err).NotTo(HaveOccurred())
		return strings.ToLower(response.Content)
	}

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

	// TEST_VISION_MODEL names a vision model with its mmproj-*.gguf beside it
	Context("with images", func() {
		BeforeEach(func() {
			model = loadWithMMProj("TEST_VISION_MODEL")
			Expect(model.SupportsVision()).To(BeTrue())
			var err error
			ctx, err = model.NewContext(llama.WithContext(4096))
			Expect(err).NotTo(HaveOccurred())
		})

		It("should see an image", Label("integration"), func() {
			Expect(ask(ctx, "What colour is this image? Answer in one word.", red)).To(ContainSubstring("red"))
		})

		It("should see several images in order", Label("integration"), func() {
			answer := ask(ctx, "What colours are these two images, in order? Answer briefly.", red, blue)
			Expect(answer).To(ContainSubstring("red"))
			Expect(answer).To(ContainSubstring("blue"))
			Expect(strings.Index(answer, "red")).To(BeNumerically("<", strings.Index(answer, "blue")))
		})

		It("should place images where the content puts the marker", Label("integration"), func() {
			marker := model.MediaMarker()
			answer := ask(ctx, "Image A: "+marker+"\nImage B: "+marker+"\nWhat colour is image B? Answer in one word.", red, blue)
			Expect(answer).To(ContainSubstring("blue"))
		})

		It("should reuse a cached image, and not mistake a different one for it", Label("integration"), func() {
			const question = "What colour is this image? Answer in one word."
			Expect(ask(ctx, question, red)).To(ContainSubstring("red"))
			Expect(ask(ctx, question, red)).To(ContainSubstring("red"))
			Expect(ask(ctx, question, blue)).To(ContainSubstring("blue"))
		})

		It("should generate from a raw prompt with WithMedia", Label("integration"), func() {
			messages := []llama.ChatMessage{{Role: "user", Content: "What colour is this image? Answer in one word.", Media: [][]byte{red}}}
			prompt, err := model.FormatChatPrompt(messages, opts)
			Expect(err).NotTo(HaveOccurred())
			Expect(prompt).To(ContainSubstring(model.MediaMarker()))

			raw, err := ctx.Generate(prompt, llama.WithMedia(red), llama.WithTemperature(0), llama.WithMaxTokens(20))
			Expect(err).NotTo(HaveOccurred())
			Expect(strings.ToLower(raw)).To(ContainSubstring("red"))
		})

		It("should use a random marker rather than mtmd's fixed default", Label("integration"), func() {
			Expect(model.MediaMarker()).NotTo(BeEmpty())
			Expect(model.MediaMarker()).NotTo(Equal("<__media__>"))
		})

		It("should reject a marker count that doesn't match the media", Label("integration"), func() {
			_, err := ctx.Chat(context.Background(), []llama.ChatMessage{{
				Role: "user", Content: "Only one marker: " + model.MediaMarker(), Media: [][]byte{red, blue},
			}}, opts)
			Expect(err).To(MatchError(ContainSubstring("places 1 media items but has 2")))

			_, err = ctx.Generate("No marker here", llama.WithMedia(red))
			Expect(err).To(MatchError(ContainSubstring("media markers (0) don't match the number of media items (1)")))
		})

		It("should reject media it can't read", Label("integration"), func() {
			_, err := ctx.Generate(model.MediaMarker(), llama.WithMedia([]byte("not an image")))
			Expect(err).To(MatchError(ContainSubstring("not a supported image or audio format")))

			_, err = ctx.Generate(model.MediaMarker(), llama.WithMedia([]byte{}))
			Expect(err).To(MatchError(ContainSubstring("media item 0 is empty")))
		})

		It("should reject media with token input", Label("integration"), func() {
			tokens, err := ctx.Tokenize("Hello")
			Expect(err).NotTo(HaveOccurred())
			_, err = ctx.GenerateWithTokens(tokens, llama.WithMedia(red))
			Expect(err).To(MatchError(ContainSubstring("not tokens")))
		})

		It("should reject media while speculating", Label("integration", "speculative"), func() {
			draft, err := llama.LoadModel(os.Getenv("TEST_VISION_MODEL"), llama.WithGPULayers(-1))
			Expect(err).NotTo(HaveOccurred())
			defer func() { Expect(draft.Close()).To(Succeed()) }()
			speculating, err := model.NewContext(llama.WithContext(2048), llama.WithDraft(draft))
			Expect(err).NotTo(HaveOccurred())
			defer func() { Expect(speculating.Close()).To(Succeed()) }()

			_, err = speculating.Generate(model.MediaMarker(), llama.WithMedia(red))
			Expect(err).To(MatchError(ContainSubstring("doesn't support media input yet")))
		})
	})

	Context("without a projector", func() {
		It("should report no media support and reject media", Label("integration"), func() {
			modelPath := os.Getenv("TEST_CHAT_MODEL")
			if modelPath == "" {
				Skip("TEST_CHAT_MODEL not set - skipping integration test")
			}
			var err error
			model, err = llama.LoadModel(modelPath, llama.WithGPULayers(-1))
			Expect(err).NotTo(HaveOccurred())
			Expect(model.SupportsVision()).To(BeFalse())
			Expect(model.SupportsAudio()).To(BeFalse())
			Expect(model.MediaMarker()).To(BeEmpty())

			ctx, err = model.NewContext(llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			_, err = ctx.Chat(context.Background(), []llama.ChatMessage{{Role: "user", Content: "Hi", Media: [][]byte{red}}}, opts)
			Expect(err).To(MatchError(ContainSubstring("WithMMProj")))
			_, err = ctx.Generate("Hi", llama.WithMedia(red))
			Expect(err).To(MatchError(ContainSubstring("WithMMProj")))
		})
	})

	// TEST_AUDIO_MODEL names an audio model with its mmproj-*.gguf beside it
	Context("with audio", func() {
		It("should hear an audio clip", Label("integration"), func() {
			model = loadWithMMProj("TEST_AUDIO_MODEL")
			Expect(model.SupportsAudio()).To(BeTrue())
			var err error
			ctx, err = model.NewContext(llama.WithContext(4096))
			Expect(err).NotTo(HaveOccurred())

			answer := ask(ctx, "Describe this audio clip in a few words.", toneWAV())
			Expect(answer).NotTo(BeEmpty())
		})
	})
})
