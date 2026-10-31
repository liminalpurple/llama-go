package llama

/*
#include "wrapper.h"
#include <stdlib.h>
*/
import "C"

import (
	gocontext "context"
	"encoding/json"
	"fmt"
	"unsafe"
)

// DecisionQuestionType is the kind of answer a decision question asks for.
type DecisionQuestionType string

const (
	// DecisionChoice picks one of several named options.
	DecisionChoice DecisionQuestionType = "choice"
	// DecisionNoul answers yes or no, as the probability of yes.
	DecisionNoul DecisionQuestionType = "noul"
	// DecisionScore rates on an ordered scale of levels.
	DecisionScore DecisionQuestionType = "score"
)

// DecisionQuestion is one typed question for a decision model.
//
// Instructions and the descriptions of choices and levels are usually text,
// but can be anything JSON-encodable, as llama.cpp accepts.
type DecisionQuestion struct {
	Type         DecisionQuestionType
	Instructions any
	// Choices names the options of a DecisionChoice question, each with a
	// description, or nil for none.
	Choices map[string]any
	// Levels describes the levels of a DecisionScore question, lowest first.
	Levels []any
}

// DecisionRequest asks a decision model typed questions about a state.
type DecisionRequest struct {
	// State is what the questions are about: text, or anything JSON-encodable.
	State any
	// Questions maps an id of your choosing to each question.
	Questions map[string]DecisionQuestion
	// Media holds images or audio clips the questions are about, as encoded
	// files. It needs a model loaded with WithMMProj whose decision type
	// reads media.
	Media [][]byte
}

// DecisionAnswer is a decision model's answer to one question, with
// calibrated probabilities rather than generated text.
type DecisionAnswer struct {
	Type DecisionQuestionType `json:"type"`
	// Choice is the most likely option of a DecisionChoice question.
	Choice string `json:"choice,omitempty"`
	// Noul is the probability of yes, for a DecisionNoul question.
	Noul float64 `json:"noul,omitempty"`
	// Score is the expected level of a DecisionScore question, from 0 for
	// the first level, with Legend naming each level.
	Score  float64           `json:"score,omitempty"`
	Legend map[string]string `json:"legend,omitempty"`
	// Probabilities gives each option or level its probability, for
	// DecisionChoice and DecisionScore questions.
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	// Confidence is how sure the model is of a DecisionChoice or
	// DecisionScore answer, from 0 to 1.
	Confidence float64 `json:"confidence,omitempty"`
}

// DecisionResponse holds a decision model's answers, by question id.
type DecisionResponse struct {
	Answers     map[string]DecisionAnswer
	InputTokens int
}

// DecisionType returns the model's decision type, such as "h2o-lightning" or
// "kev", or "" if it isn't a decision model. Decision models answer typed
// questions with Context.Decide.
func (m *Model) DecisionType() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.closed {
		return ""
	}
	if t := C.llama_wrapper_model_decision_type(m.modelPtr); t != nil {
		return C.GoString(t)
	}
	return ""
}

// Decide asks a decision model typed questions about a state, as
// llama.cpp's /v1/systemone API does, and returns calibrated answers.
//
// Questions about the same state reuse it from the prefix cache, so asking
// several at once costs little more than asking one.
//
// Example:
//
//	resp, err := ctx.Decide(context.Background(), llama.DecisionRequest{
//	    State: "The customer was charged twice for order 1182.",
//	    Questions: map[string]llama.DecisionQuestion{
//	        "angry": {Type: llama.DecisionNoul, Instructions: "Is the customer upset?"},
//	        "route": {Type: llama.DecisionChoice, Instructions: "Which team should handle this?",
//	            Choices: map[string]any{"billing": "payments and refunds", "shipping": "delivery problems"}},
//	    },
//	})
//	fmt.Println(resp.Answers["route"].Choice, resp.Answers["angry"].Noul)
func (c *Context) Decide(ctx gocontext.Context, req DecisionRequest) (*DecisionResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	body, err := decisionBody(req)
	if err != nil {
		return nil, err
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil, fmt.Errorf("context is closed")
	}
	if err := c.checkModelsOpen(); err != nil {
		return nil, err
	}

	cBody := C.CString(string(body))
	defer C.free(unsafe.Pointer(cBody))

	// Media goes to C as copies, as cgo can't pass Go pointers inside arrays
	var cMediaPtr **C.uchar
	var cSizesPtr *C.size_t
	if n := len(req.Media); n > 0 {
		cMedia := (*[1 << 28]*C.uchar)(C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof((*C.uchar)(nil)))))[:n:n]
		cSizes := (*[1 << 28]C.size_t)(C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.size_t(0)))))[:n:n]
		defer C.free(unsafe.Pointer(&cMedia[0]))
		defer C.free(unsafe.Pointer(&cSizes[0]))
		for i, item := range req.Media {
			if len(item) == 0 {
				return nil, fmt.Errorf("media item %d is empty", i)
			}
			cMedia[i] = (*C.uchar)(C.CBytes(item))
			cSizes[i] = C.size_t(len(item))
			defer C.free(unsafe.Pointer(cMedia[i]))
		}
		cMediaPtr = &cMedia[0]
		cSizesPtr = &cSizes[0]
	}

	cResult, errMsg := withLastError(func() *C.char {
		return C.llama_wrapper_decide(c.contextPtr, cBody, cMediaPtr, cSizesPtr, C.int(len(req.Media)),
			C.bool(c.config.prefixCaching))
	})
	if cResult == nil {
		return nil, fmt.Errorf("decision failed: %s", errMsg)
	}
	result := C.GoString(cResult)
	C.llama_wrapper_free_result(cResult)

	var parsed struct {
		Answers map[string]DecisionAnswer `json:"answers"`
		Usage   struct {
			InputTokens int `json:"input_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal([]byte(result), &parsed); err != nil {
		return nil, fmt.Errorf("failed to read decision result: %w", err)
	}
	return &DecisionResponse{Answers: parsed.Answers, InputTokens: parsed.Usage.InputTokens}, nil
}

// decisionBody encodes a request as the JSON body of llama.cpp's
// /v1/systemone API, with media passed separately.
func decisionBody(req DecisionRequest) ([]byte, error) {
	questions := make(map[string]map[string]any, len(req.Questions))
	for id, q := range req.Questions {
		question := map[string]any{"type": q.Type}
		if q.Instructions != nil {
			question["instructions"] = q.Instructions
		}
		switch {
		case q.Choices != nil:
			question["criteria"] = q.Choices
		case q.Levels != nil:
			question["criteria"] = q.Levels
		}
		questions[id] = question
	}
	body, err := json.Marshal(map[string]any{"state": req.State, "questions": questions})
	if err != nil {
		return nil, fmt.Errorf("failed to encode decision request: %w", err)
	}
	return body, nil
}
