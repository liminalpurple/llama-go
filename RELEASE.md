# Release process

Each llama-go release pins the llama.cpp submodule to one upstream build and is tagged after that
build passes the full test suite, so a tag tells you exactly which llama.cpp features you get.

## Versioning

Tags follow `llama.cpp-{upstream-tag}`, e.g. `llama.cpp-b11982` is tested against llama.cpp tag
`b11982`.

## Where the work happens

Everything builds in `git.cisnt.uk/morgan/llama-go:build-cuda`, which already sets
`BUILD_TYPE=cublas`, `CUDA_ARCHITECTURES=86` and the include and library paths, and routes
llama.cpp's compiles through sccache. Mount the host's `/var/cache/sccache` so the cache survives
between containers: a rebuild after `make clean` then takes minutes rather than the 20-25 of a cold
build.

Testing happens in two places. Locally you build and run the suite while fixing whatever the
upstream bump broke, which is the fast loop. Before tagging, the **Release Validation** workflow
repeats it in CI from a clean checkout, in both static and shared linkage, so the tag rests on a run
anyone can see.

## Release workflow

### 1. Choose the upstream build

Pick a tag from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) and note how
far it is from the current pin:

```bash
git -C llama.cpp describe --tags                        # current pin
git -C llama.cpp fetch --tags origin
git -C llama.cpp rev-list --count <current>..<target>   # builds in between
```

### 2. Clean and move the submodule

```bash
docker run --rm -v $(pwd):/workspace -w /workspace git.cisnt.uk/morgan/llama-go:build-cuda \
  "make clean"
ls -la *.a *.so 2>/dev/null   # should print nothing

git -C llama.cpp checkout <target>
git -C llama.cpp describe --tags   # should print the exact tag
git -C llama.cpp status            # should be clean
git add llama.cpp
```

`make clean` also deletes `cgo_headers/`. The next build regenerates it from the submodule.

### 3. Read what changed upstream

Start the build (step 4) in the background and read the upstream diff while it runs, so you
already know what moved when the compiler complains. The wrapper includes these headers, and
changes to them are what matter:

```bash
git -C llama.cpp diff <current>..<target> -- include/llama.h common/common.h common/chat.h \
  common/sampling.h common/speculative.h tools/mtmd/mtmd.h tools/mtmd/mtmd-helper.h \
  tools/server/server-decision.h
```

The generation loop in `wrapper.cpp` follows the server's, so it's worth a look at server changes to
speculation and checkpoints too, even when nothing breaks the build:

```bash
git -C llama.cpp log --oneline <current>..<target> -- common/speculative.cpp \
  tools/server/server-context.cpp
```

A behaviour change worth following gets its own `fix(...)` commit, separate from the submodule
bump.

### 4. Build

```bash
docker run --rm -v $(pwd):/workspace -w /workspace \
  -v /var/cache/sccache:/var/cache/sccache git.cisnt.uk/morgan/llama-go:build-cuda \
  "make libbinding.a && sccache --show-stats"
```

The build regenerates `cgo_headers/`, so `git status` will show additions, deletions and edits
there. They're expected, and they must go in the same commit as the submodule pointer: the Go
module ships these headers so cgo can compile `wrapper.cpp` from the module proxy, which doesn't
carry the submodule.

When the wrapper stops compiling, see how the server and examples now call the changed API and do
the same.

### 5. Test locally

The test models and where they come from are listed in `.forgejo/actions/test-models/action.yml`,
which CI uses too; the commands below use its folder layout. Integration tests skip when their
model variable is unset, so you can run a subset while iterating.

```bash
docker run --rm --gpus all --cpus=8 -v $(pwd):/workspace -w /workspace \
  -e TEST_CHAT_MODEL=Qwen3-0.6B-Q8_0.gguf \
  -e TEST_EMBEDDING_MODEL=Qwen3-Embedding-0.6B-Q8_0.gguf \
  -e TEST_MTP_MODEL=Qwen3.5-0.8B/Qwen3.5-0.8B-Q8_0.gguf \
  -e TEST_VISION_MODEL=Qwen3.5-0.8B/Qwen3.5-0.8B-Q8_0.gguf \
  -e TEST_AUDIO_MODEL=gemma-4-E2B-it/gemma-4-E2B-it-Q4_K_M.gguf \
  -e TEST_MTP_SIDECAR_MODEL=gemma-4-E2B-it/gemma-4-E2B-it-Q4_K_M.gguf \
  -e TEST_MOE_MODEL=stories15M_MOE/stories15M_MOE-F16.gguf \
  git.cisnt.uk/morgan/llama-go:build-cuda \
  "go run github.com/onsi/ginkgo/v2/ginkgo -tags cublas --flake-attempts 3 ./..."
```

Generation, caching and speculation behave differently across model families, so repeat the chat
tests with Gemma 4 as the chat model (sliding-window memory, and an instruct model that ends raw
prompts early):

```bash
docker run --rm --gpus all --cpus=8 -v $(pwd):/workspace -w /workspace \
  -e TEST_CHAT_MODEL=gemma-4-E2B-it/gemma-4-E2B-it-Q4_K_M.gguf \
  git.cisnt.uk/morgan/llama-go:build-cuda \
  "go run github.com/onsi/ginkgo/v2/ginkgo -tags cublas --flake-attempts 3 ./..."
```

Then check the example still generates sensible text:

```bash
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace git.cisnt.uk/morgan/llama-go:build-cuda \
  "go run -tags cublas ./examples/simple -m Qwen3-0.6B-Q8_0.gguf -p 'Hello world' -n 50"
```

### 6. Commit

Commit any behaviour fixes first, then the bump itself with its regenerated headers:

```bash
git add llama.cpp cgo_headers
git diff --cached --stat   # the submodule pointer plus cgo_headers churn
git submodule status       # no + prefix, so no "-dirty" pointer
```

The message says what moved and what had to change for it:

```text
feat(deps): update llama.cpp to b6709

Update llama.cpp from b6615 to b6709 (94 builds). Initialise the new
no_host field in llama_model_params, which upstream added without a
default.

Upstream release: https://github.com/ggml-org/llama.cpp/releases/tag/b6709
```

### 7. Push and validate in CI

Push `main`. CI runs the static suite and republishes the `libs-cuda` image. Then run **Release
Validation** from the Actions tab, or through the API:

```bash
curl -s -X POST -H "Authorization: token $TOKEN" -H "Content-Type: application/json" \
  -d '{"ref":"main"}' \
  https://git.cisnt.uk/api/v1/repos/morgan/llama-go/actions/workflows/release-validation.yml/dispatches
```

It builds and tests both linkage modes with every test model, repeats the chat tests with Gemma 4,
and runs the example. CI, Release Validation and the CUDA image share a concurrency group, so a run
waits for the one before it rather than cancelling it.

### 8. Tag

Once Release Validation passes, tag the commit it tested. Look at the previous tag first
(`git tag -n99 llama.cpp-<previous>`) and follow its shape: llama-go's own changes since then, in a
few bullets, then a line or two on what upstream brought.

```bash
git tag -a llama.cpp-<target> -m "$(cat <<'EOF'
Update to llama.cpp <target>

Upstream release: https://github.com/ggml-org/llama.cpp/releases/tag/<target>

Major features since llama.cpp-<previous>:
- A llama-go feature, with a brief description
- Fixes and improvements
- Test and CI changes

This release includes llama.cpp improvements: <n> builds between <previous>
and <target>, with <highlights>.
EOF
)"
git push origin llama.cpp-<target>
```

## When tests fail

Check for resource pressure before reading code. A test that doesn't set a context size gets the
model's native one, often 40K or more, and the VRAM adds up across a run, so crashes or OOM errors
late in the suite usually mean a test is missing `WithContext(2048)`. `nvidia-smi` shows driver
trouble.

When the failure is a real API change, `git -C llama.cpp diff <current>..<target>` is the reliable
way in, read alongside upstream's own callers. The usual culprits are new struct fields needing
initialisation, stricter validation, and changed cleanup requirements.

Some differences look like bugs and aren't. With quantised weights, speculative output can differ
from plain generation at temperature 0 when two tokens are nearly tied, because CUDA picks a
different kernel for each batch width. The speculative tests account for this by using prompts
every kernel continues the same way.

## Pre-built libraries

Most backends need their own build: CPU-only, CUDA, ROCm, Metal, Vulkan and the rest each need
different SDKs, flags and runtime libraries, so applications build the variant they need as part of
their own pipeline.

The one exception is CUDA in Docker. `git.cisnt.uk/morgan/llama-go:libs-cuda` holds static
archives for CUDA architecture 86 and the headers cgo needs, so a downstream image can
`COPY --from=` them rather than rebuilding llama.cpp.

## Checklist

- [ ] Submodule at the exact target tag, with no local changes
- [ ] `cgo_headers/` regenerated and committed with the submodule pointer
- [ ] Behaviour changes from upstream's server followed in their own commits
- [ ] Local suite passes, with Qwen and with Gemma 4 as the chat model
- [ ] Example generates sensible text
- [ ] CI passes on `main` and `libs-cuda` republishes
- [ ] Release Validation passes in both linkage modes
- [ ] Annotated tag on the validated commit, pushed
