package llama_test

import (
	"context"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/liminalpurple/llama-go"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// The example request from H2O Lightning's model card, covering every
// question type
var decisionRequest = llama.DecisionRequest{
	State: map[string]any{"customer": "I was charged twice for order 1182 and nobody has replied.", "plan": "pro"},
	Questions: map[string]llama.DecisionQuestion{
		"route": {Type: llama.DecisionChoice, Instructions: "Which team should handle this?",
			Choices: map[string]any{"billing": "payments and refunds", "shipping": "delivery problems", "technical": "app or website errors"}},
		"angry":   {Type: llama.DecisionNoul, Instructions: "Is the customer upset?"},
		"urgency": {Type: llama.DecisionScore, Instructions: "How urgent is this?", Levels: []any{"can wait", "this week", "today", "right now"}},
	},
}

// expectWellFormed checks each answer has the shape its question type gives,
// with probabilities that sum to 1
func expectWellFormed(req llama.DecisionRequest, resp *llama.DecisionResponse) {
	Expect(resp.Answers).To(HaveLen(len(req.Questions)))
	Expect(resp.InputTokens).To(BeNumerically(">", 0))
	for id, q := range req.Questions {
		answer, ok := resp.Answers[id]
		Expect(ok).To(BeTrue(), "no answer for "+id)
		Expect(answer.Type).To(Equal(q.Type))
		sum := 0.0
		for _, p := range answer.Probabilities {
			sum += p
		}
		switch q.Type {
		case llama.DecisionChoice:
			Expect(q.Choices).To(HaveKey(answer.Choice))
			Expect(answer.Probabilities).To(HaveLen(len(q.Choices)))
			Expect(sum).To(BeNumerically("~", 1, 1e-6))
		case llama.DecisionNoul:
			Expect(answer.Noul).To(BeNumerically(">=", 0))
			Expect(answer.Noul).To(BeNumerically("<=", 1))
		case llama.DecisionScore:
			Expect(answer.Score).To(BeNumerically(">=", 0))
			Expect(answer.Score).To(BeNumerically("<=", len(q.Levels)-1))
			Expect(answer.Legend).To(HaveLen(len(q.Levels)))
			Expect(sum).To(BeNumerically("~", 1, 1e-6))
		}
	}
}

// expectSimilarAnswers checks two responses agree to within tolerance
func expectSimilarAnswers(a, b *llama.DecisionResponse, tolerance float64) {
	Expect(b.Answers).To(HaveLen(len(a.Answers)))
	for id, x := range a.Answers {
		y := b.Answers[id]
		Expect(math.Abs(y.Noul-x.Noul)).To(BeNumerically("<=", tolerance), id)
		Expect(math.Abs(y.Score-x.Score)).To(BeNumerically("<=", tolerance), id)
		for k, p := range x.Probabilities {
			Expect(math.Abs(y.Probabilities[k]-p)).To(BeNumerically("<=", tolerance), id+" "+k)
		}
	}
}

// TEST_DECISION_MODELS lists decision models, comma-separated. A model with
// an mmproj-*.gguf beside it is also asked about an image.
var _ = Describe("Decision models", Label("decision"), func() {
	var paths []string
	for _, path := range strings.Split(os.Getenv("TEST_DECISION_MODELS"), ",") {
		if path = strings.TrimSpace(path); path != "" {
			paths = append(paths, path)
		}
	}
	if len(paths) == 0 {
		It("should answer typed questions", func() {
			Skip("TEST_DECISION_MODELS not set - skipping integration test")
		})
	}

	for _, path := range paths {
		Context(filepath.Base(path), func() {
			var (
				model  *llama.Model
				ctx    *llama.Context
				mmproj string
			)

			BeforeEach(func() {
				projectors, err := filepath.Glob(filepath.Join(filepath.Dir(path), "mmproj-*.gguf"))
				Expect(err).NotTo(HaveOccurred())
				opts := []llama.ModelOption{llama.WithGPULayers(-1)}
				mmproj = ""
				if len(projectors) == 1 {
					mmproj = projectors[0]
					opts = append(opts, llama.WithMMProj(mmproj))
				}
				model, err = llama.LoadModel(path, opts...)
				Expect(err).NotTo(HaveOccurred())
				ctx, err = model.NewContext(llama.WithContext(8192))
				Expect(err).NotTo(HaveOccurred())
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

			It("should report its decision type", Label("integration"), func() {
				Expect(model.DecisionType()).NotTo(BeEmpty())
			})

			It("should answer every question type", Label("integration"), func() {
				resp, err := ctx.Decide(context.Background(), decisionRequest)
				Expect(err).NotTo(HaveOccurred())
				expectWellFormed(decisionRequest, resp)
			})

			It("should give the same answers when reusing a state from the prefix cache", Label("integration"), func() {
				// The questions share the state, decoded once and kept for each
				// question, so asking again reuses it with the same arithmetic
				first, err := ctx.Decide(context.Background(), decisionRequest)
				Expect(err).NotTo(HaveOccurred())
				reused, err := ctx.Decide(context.Background(), decisionRequest)
				Expect(err).NotTo(HaveOccurred())
				expectSimilarAnswers(first, reused, 0)

				// Without the cache the prompt is decoded in different batches,
				// which recurrent models round slightly differently (by up to
				// about 0.013 for Granite Switch at Q4_K_M), but a stale or
				// misplaced state is far off or fails outright
				fresh, err := model.NewContext(llama.WithContext(8192), llama.WithPrefixCaching(false))
				Expect(err).NotTo(HaveOccurred())
				defer func() { Expect(fresh.Close()).To(Succeed()) }()
				uncached, err := fresh.Decide(context.Background(), decisionRequest)
				Expect(err).NotTo(HaveOccurred())
				expectSimilarAnswers(uncached, reused, 0.05)
			})

			It("should answer about an image", Label("integration"), func() {
				if mmproj == "" {
					Skip("no mmproj-*.gguf beside the model")
				}
				req := llama.DecisionRequest{
					State: "A photo from a camera.",
					Questions: map[string]llama.DecisionQuestion{
						"colour": {Type: llama.DecisionChoice, Instructions: "What colour is the image?",
							Choices: map[string]any{"red": nil, "green": nil, "blue": nil}},
					},
					Media: [][]byte{solidPNG(color.RGBA{255, 0, 0, 255})},
				}
				resp, err := ctx.Decide(context.Background(), req)
				Expect(err).NotTo(HaveOccurred())
				expectWellFormed(req, resp)
			})

			It("should answer about an audio clip", Label("integration"), func() {
				if !model.SupportsAudio() {
					Skip("the model's projector doesn't read audio")
				}
				req := llama.DecisionRequest{
					State: "A recording from a microphone.",
					Questions: map[string]llama.DecisionQuestion{
						"sound": {Type: llama.DecisionNoul, Instructions: "Can a sound be heard in the recording?"},
					},
					Media: [][]byte{toneWAV()},
				}
				resp, err := ctx.Decide(context.Background(), req)
				Expect(err).NotTo(HaveOccurred())
				expectWellFormed(req, resp)
			})

			It("should reject requests llama.cpp would", Label("integration"), func() {
				_, err := ctx.Decide(context.Background(), llama.DecisionRequest{State: "Hello"})
				Expect(err).To(MatchError(ContainSubstring(`"questions" must be a non-empty object`)))

				_, err = ctx.Decide(context.Background(), llama.DecisionRequest{
					State:     "Hello",
					Questions: map[string]llama.DecisionQuestion{"q": {Type: "guess", Instructions: "Pick one"}},
				})
				Expect(err).To(HaveOccurred())

				withMedia := decisionRequest
				withMedia.Media = [][]byte{{}}
				_, err = ctx.Decide(context.Background(), withMedia)
				Expect(err).To(MatchError(ContainSubstring("media item 0 is empty")))
			})
		})
	}

	Context("with a model that isn't one", func() {
		It("should report no decision type and refuse to decide", Label("integration"), func() {
			modelPath := os.Getenv("TEST_CHAT_MODEL")
			if modelPath == "" {
				Skip("TEST_CHAT_MODEL not set - skipping integration test")
			}
			model, err := llama.LoadModel(modelPath, llama.WithGPULayers(-1))
			Expect(err).NotTo(HaveOccurred())
			defer func() { Expect(model.Close()).To(Succeed()) }()
			Expect(model.DecisionType()).To(BeEmpty())

			ctx, err := model.NewContext(llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer func() { Expect(ctx.Close()).To(Succeed()) }()
			_, err = ctx.Decide(context.Background(), decisionRequest)
			Expect(err).To(MatchError(ContainSubstring("not a decision model")))
		})
	})

	// TEST_DECISION_VISION_MODEL names a decision model that can see, with its
	// mmproj-*.gguf beside it. The test models above only check the answers are
	// well-formed; this one checks they're right, which fails if the image is
	// ignored or misplaced.
	Context("reading images", func() {
		It("should answer correctly and confidently", Label("integration"), func() {
			model := loadWithMMProj("TEST_DECISION_VISION_MODEL")
			defer func() { Expect(model.Close()).To(Succeed()) }()
			ctx, err := model.NewContext(llama.WithContext(8192))
			Expect(err).NotTo(HaveOccurred())
			defer func() { Expect(ctx.Close()).To(Succeed()) }()

			// mtmd's own test image, the 1969 New York Times front page
			paper, err := os.ReadFile("llama.cpp/tools/mtmd/test-1.jpeg")
			Expect(err).NotTo(HaveOccurred())
			colour := llama.DecisionQuestion{Type: llama.DecisionChoice, Instructions: "What colour is the image?",
				Choices: map[string]any{"red": nil, "green": nil, "blue": nil}}

			for _, tc := range []struct {
				name     string
				media    []byte
				question llama.DecisionQuestion
				want     string
			}{
				{"document", paper, llama.DecisionQuestion{Type: llama.DecisionChoice, Instructions: "What kind of document is this?",
					Choices: map[string]any{"newspaper": nil, "restaurant menu": nil, "shopping receipt": nil, "handwritten letter": nil}}, "newspaper"},
				{"headline", paper, llama.DecisionQuestion{Type: llama.DecisionChoice, Instructions: "Where does the headline say men walked?",
					Choices: map[string]any{"moon": nil, "beach": nil, "mountain": nil, "city": nil}}, "moon"},
				// Answering both from the text alone is impossible
				{"red", solidPNG(color.RGBA{255, 0, 0, 255}), colour, "red"},
				{"blue", solidPNG(color.RGBA{0, 0, 255, 255}), colour, "blue"},
			} {
				resp, err := ctx.Decide(context.Background(), llama.DecisionRequest{
					State:     "An image.",
					Questions: map[string]llama.DecisionQuestion{"q": tc.question},
					Media:     [][]byte{tc.media},
				})
				Expect(err).NotTo(HaveOccurred(), tc.name)
				answer := resp.Answers["q"]
				Expect(answer.Choice).To(Equal(tc.want), tc.name)
				Expect(answer.Probabilities[tc.want]).To(BeNumerically(">", 0.5), tc.name)
			}
		})
	})
})
