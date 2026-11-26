package llama_test

import (
	"image/color"
	"math"
	"os"

	"github.com/liminalpurple/llama-go"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// cosine returns the cosine similarity of two vectors
func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

var _ = Describe("Media embeddings", Label("media"), func() {
	Context("without a projector", func() {
		It("should say the model needs WithMMProj", Label("integration"), func() {
			modelPath := os.Getenv("TEST_EMBEDDING_MODEL")
			if modelPath == "" {
				Skip("TEST_EMBEDDING_MODEL not set - skipping integration test")
			}
			model, err := llama.LoadModel(modelPath, llama.WithGPULayers(-1))
			Expect(err).NotTo(HaveOccurred())
			defer func() { Expect(model.Close()).To(Succeed()) }()
			ctx, err := model.NewContext(llama.WithContext(2048), llama.WithEmbeddings())
			Expect(err).NotTo(HaveOccurred())
			defer func() { Expect(ctx.Close()).To(Succeed()) }()

			_, err = ctx.GetEmbeddingsWithMedia("", [][]byte{solidPNG(color.RGBA{255, 0, 0, 255})})
			Expect(err).To(MatchError(ContainSubstring("WithMMProj")))
		})
	})

	Context("with a multimodal embedding model", Ordered, func() {
		var (
			model *llama.Model
			ctx   *llama.Context
			paper []byte
			red   = solidPNG(color.RGBA{255, 0, 0, 255})
			blue  = solidPNG(color.RGBA{0, 0, 255, 255})
		)

		BeforeAll(func() {
			model = loadWithMMProj("TEST_MEDIA_EMBEDDING_MODEL")
			var err error
			ctx, err = model.NewContext(llama.WithContext(4096), llama.WithBatch(4096), llama.WithEmbeddings())
			Expect(err).NotTo(HaveOccurred())
			paper, err = os.ReadFile("llama.cpp/tools/mtmd/test-1.jpeg")
			Expect(err).NotTo(HaveOccurred())
		})

		AfterAll(func() {
			if ctx != nil {
				Expect(ctx.Close()).To(Succeed())
			}
			if model != nil {
				Expect(model.Close()).To(Succeed())
			}
		})

		embed := func(text string, media ...[]byte) []float32 {
			embedding, err := ctx.GetEmbeddingsWithMedia(text, media)
			Expect(err).NotTo(HaveOccurred())
			return embedding
		}

		It("should give an image an embedding as wide as a text's", Label("integration"), func() {
			text, err := ctx.GetEmbeddings("a red square")
			Expect(err).NotTo(HaveOccurred())
			Expect(embed("", red)).To(HaveLen(len(text)))
		})

		It("should place an image nearer text that describes it", Label("integration"), func() {
			image := embed("", paper)
			match := embed("a newspaper front page with a headline about men walking on the moon")
			other := embed("a bowl of fruit on a kitchen table")
			Expect(cosine(image, match)).To(BeNumerically(">", cosine(image, other)))

			redImage, blueImage := embed("", red), embed("", blue)
			redText, blueText := embed("a plain red square"), embed("a plain blue square")
			Expect(cosine(redImage, redText)).To(BeNumerically(">", cosine(redImage, blueText)))
			Expect(cosine(blueImage, blueText)).To(BeNumerically(">", cosine(blueImage, redText)))
		})

		It("should put media at the start unless the text places it", Label("integration"), func() {
			marker := model.MediaMarker()
			Expect(embed("a photo", red)).To(Equal(embed(marker+"a photo", red)))
			Expect(embed("a photo", red)).NotTo(Equal(embed("a photo"+marker, red)))
		})

		It("should read every media item in an input", Label("integration"), func() {
			// Each image is encoded before the input is decoded, so a mix of
			// images must not come out as the last one repeated
			Expect(embed("", red, blue)).NotTo(Equal(embed("", blue, blue)))
		})

		It("should give each input the same embedding in a batch as alone", Label("integration"), func() {
			inputs := []llama.EmbeddingInput{
				{Media: [][]byte{paper}},
				{Text: "a plain red square"},
				{Text: "red and blue", Media: [][]byte{red, blue}},
			}
			batch, err := ctx.GetEmbeddingsBatchWithMedia(inputs)
			Expect(err).NotTo(HaveOccurred())
			Expect(batch).To(HaveLen(len(inputs)))
			for i, input := range inputs {
				Expect(cosine(batch[i], embed(input.Text, input.Media...))).To(BeNumerically(">", 0.999), "input %d", i)
			}
		})

		It("should reject text that places a different number of items", Label("integration"), func() {
			marker := model.MediaMarker()
			_, err := ctx.GetEmbeddingsWithMedia(marker+marker, [][]byte{red})
			Expect(err).To(MatchError("input 0 places 2 media items but has 1"))
		})

		It("should reject an empty media item", Label("integration"), func() {
			_, err := ctx.GetEmbeddingsBatchWithMedia([]llama.EmbeddingInput{{Text: "fine"}, {Media: [][]byte{{}}}})
			Expect(err).To(MatchError("input 1 media item 0 is empty"))
		})

		It("should reject a context created without WithEmbeddings", Label("integration"), func() {
			plain, err := model.NewContext(llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer func() { Expect(plain.Close()).To(Succeed()) }()
			_, err = plain.GetEmbeddingsWithMedia("", [][]byte{red})
			Expect(err).To(MatchError("context was not created with WithEmbeddings"))
		})
	})
})
