package llama

/*
#include "wrapper.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"strings"
	"unsafe"
)

// EmbeddingInput is one input to GetEmbeddingsBatchWithMedia: text and media
// embedded together into a single vector.
type EmbeddingInput struct {
	// Text is passed to the model as it is, so add any prompt the model
	// expects (e.g. EmbeddingGemma's "task: search result | query: ").
	// It may be empty when the input is media alone.
	Text string

	// Media holds images or audio clips, as encoded files (PNG, JPEG, WAV,
	// MP3, FLAC and others, detected from their bytes). Each item goes at the
	// start of the input, unless Text places them with the model's
	// MediaMarker, once per item in order.
	Media [][]byte
}

// GetEmbeddingsWithMedia computes one embedding for text and media together,
// for multimodal embedding models such as EmbeddingGemma 2.
//
// The model must be loaded with WithMMProj and the context created with
// WithEmbeddings. Each media item goes at the start of the input, unless text
// places them with the model's MediaMarker, once per item in order. Text is
// passed to the model as it is and may be empty for media alone.
//
// See also: GetEmbeddingsBatchWithMedia for several inputs at once, and
// GetEmbeddings for text alone.
//
// Example:
//
//	model, _ := llama.LoadModel("embeddinggemma-2-Q8_0.gguf",
//	    llama.WithMMProj("mmproj-embeddinggemma-2-Q8_0.gguf"))
//	ctx, _ := model.NewContext(llama.WithEmbeddings())
//	image, _ := os.ReadFile("cat.jpg")
//	embedding, _ := ctx.GetEmbeddingsWithMedia("", [][]byte{image})
func (c *Context) GetEmbeddingsWithMedia(text string, media [][]byte) ([]float32, error) {
	embeddings, err := c.GetEmbeddingsBatchWithMedia([]EmbeddingInput{{Text: text, Media: media}})
	if err != nil {
		return nil, err
	}
	return embeddings[0], nil
}

// GetEmbeddingsBatchWithMedia computes one embedding for each input's text
// and media, processing the inputs together as GetEmbeddingsBatch does.
//
// The model must be loaded with WithMMProj and the context created with
// WithEmbeddings. Each input must fit in one batch (WithBatch, and WithUBatch
// for models with non-causal attention, which read an input in one pass), and
// an image can take hundreds of tokens.
//
// See also: GetEmbeddingsWithMedia for a single input.
//
// Example:
//
//	embeddings, _ := ctx.GetEmbeddingsBatchWithMedia([]llama.EmbeddingInput{
//	    {Media: [][]byte{photo}},
//	    {Text: "a cat asleep on a sofa"},
//	})
func (c *Context) GetEmbeddingsBatchWithMedia(inputs []EmbeddingInput) ([][]float32, error) {
	// Embedding replaces what the context holds in memory, so it takes the
	// context to itself as generation does
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, fmt.Errorf("context is closed")
	}
	if !c.config.embeddings {
		return nil, fmt.Errorf("context was not created with WithEmbeddings")
	}
	if len(inputs) == 0 {
		return nil, fmt.Errorf("no inputs provided")
	}

	marker := c.model.MediaMarker()
	if marker == "" {
		return nil, fmt.Errorf("media embeddings need the model loaded with a multimodal projector (WithMMProj)")
	}

	nEmbd := int(C.llama_wrapper_model_n_embd_out(c.model.modelPtr))
	if nEmbd <= 0 {
		return nil, fmt.Errorf("invalid embedding dimension: %d", nEmbd)
	}

	// Place each input's media with the marker, at the start unless its text
	// already places them
	cTexts := make([]*C.char, len(inputs))
	cCounts := make([]C.int, len(inputs))
	var media [][]byte
	defer func() {
		for _, t := range cTexts {
			C.free(unsafe.Pointer(t))
		}
	}()
	for i, input := range inputs {
		text := input.Text
		switch placed := strings.Count(text, marker); placed {
		case 0:
			text = strings.Repeat(marker, len(input.Media)) + text
		case len(input.Media):
		default:
			return nil, fmt.Errorf("input %d places %d media items but has %d", i, placed, len(input.Media))
		}
		for j, item := range input.Media {
			if len(item) == 0 {
				return nil, fmt.Errorf("input %d media item %d is empty", i, j)
			}
		}
		cTexts[i] = C.CString(text)
		cCounts[i] = C.int(len(input.Media))
		media = append(media, input.Media...)
	}

	// Media goes to C as copies, as cgo can't pass Go pointers inside arrays
	var cMediaPtr **C.uchar
	var cSizesPtr *C.size_t
	if n := len(media); n > 0 {
		cMedia := (*[1 << 28]*C.uchar)(C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof((*C.uchar)(nil)))))[:n:n]
		cSizes := (*[1 << 28]C.size_t)(C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.size_t(0)))))[:n:n]
		defer C.free(unsafe.Pointer(&cMedia[0]))
		defer C.free(unsafe.Pointer(&cSizes[0]))
		for i, item := range media {
			cMedia[i] = (*C.uchar)(C.CBytes(item))
			cSizes[i] = C.size_t(len(item))
		}
		defer func() {
			for _, p := range cMedia {
				C.free(unsafe.Pointer(p))
			}
		}()
		cMediaPtr = &cMedia[0]
		cSizesPtr = &cSizes[0]
	}

	cEmbeddings := make([]C.float, len(inputs)*nEmbd)
	count, errMsg := withLastError(func() C.int {
		return C.llama_wrapper_embeddings_media_batch(
			c.contextPtr,
			(**C.char)(unsafe.Pointer(&cTexts[0])),
			C.int(len(inputs)),
			cMediaPtr,
			cSizesPtr,
			&cCounts[0],
			&cEmbeddings[0],
			C.int(nEmbd),
		)
	})
	if count < 0 {
		return nil, fmt.Errorf("media embedding generation failed: %s", errMsg)
	}
	if int(count) != len(inputs) {
		return nil, fmt.Errorf("embedding count mismatch: expected %d, got %d", len(inputs), count)
	}

	result := make([][]float32, len(inputs))
	for i := range result {
		result[i] = make([]float32, nEmbd)
		for j := 0; j < nEmbd; j++ {
			result[i][j] = float32(cEmbeddings[i*nEmbd+j])
		}
	}
	return result, nil
}
