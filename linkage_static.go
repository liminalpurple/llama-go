//go:build !shared_lib
// +build !shared_lib

// Static-mode CGO link flags.
//
// This is the default build mode for llama-go. The Makefile copies the static
// archives (libbinding.a, libllama-common.a, libllama-common-base.a, libllama.a,
// libggml*.a) into the workspace root and the linker pulls all symbols into
// the consuming binary. No shared libraries are required at runtime, so the
// resulting Go binary is self-contained for the llama.cpp side.
//
// To opt into shared mode instead, build with: go build -tags shared_lib
// (and produce the workspace with: BUILD_LINKAGE=shared make libbinding.a).
//
// -Wl,--start-group/--end-group is used because libllama-common, libllama, and
// the various libggml-* archives have cross-references that the linker can't
// resolve in a single pass. Apple's ld64 has no group flags (it re-searches
// archives automatically), no libgomp, and builds the Metal + Accelerate/BLAS
// backends by default, so darwin gets its own LDFLAGS line.
//
// The archives are found through LIBRARY_PATH, not -L./: since Go 1.27.4 the
// go command walks every local -L directory for GCC spec files, and the
// repository root holds llama.cpp's RPM .spec files.

package llama

/*
#cgo CFLAGS: -I./ -I./llama.cpp -I./llama.cpp/include -I./llama.cpp/ggml/include -I./llama.cpp/common -I./llama.cpp/tools/mtmd -I./llama.cpp/vendor -I./cgo_headers -I./cgo_headers/llama.cpp -I./cgo_headers/llama.cpp/include -I./cgo_headers/llama.cpp/ggml/include -I./cgo_headers/llama.cpp/common -I./cgo_headers/llama.cpp/tools/mtmd -I./cgo_headers/llama.cpp/thirdparty
#cgo CXXFLAGS: -std=c++17 -I./ -I./llama.cpp -I./llama.cpp/include -I./llama.cpp/ggml/include -I./llama.cpp/common -I./llama.cpp/tools/mtmd -I./llama.cpp/vendor -I./cgo_headers -I./cgo_headers/llama.cpp -I./cgo_headers/llama.cpp/include -I./cgo_headers/llama.cpp/ggml/include -I./cgo_headers/llama.cpp/common -I./cgo_headers/llama.cpp/tools/mtmd -I./cgo_headers/llama.cpp/thirdparty
#cgo !darwin LDFLAGS: -Wl,--start-group -lbinding -lserver-context -lmtmd -lvendor-hash -lllama-common -lllama-common-base -lcpp-httplib -lllama -lggml-cpu -lggml -lggml-base -Wl,--end-group -lstdc++ -lm -lgomp
#cgo darwin LDFLAGS: -lbinding -lserver-context -lmtmd -lvendor-hash -lllama-common -lllama-common-base -lcpp-httplib -lllama -lggml-cpu -lggml -lggml-base -lggml-metal -lggml-blas -lstdc++ -lm -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
*/
import "C"
