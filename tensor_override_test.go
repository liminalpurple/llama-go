package llama_test

import (
	"os"

	llama "github.com/liminalpurple/llama-go"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Devices", Label("devices"), func() {
	It("should list the CPU as a device that stores tensors", func() {
		devices := llama.Devices()
		Expect(devices).NotTo(BeEmpty())

		var cpu *llama.Device
		for i := range devices {
			if devices[i].Type == llama.DeviceCPU {
				cpu = &devices[i]
			}
		}
		Expect(cpu).NotTo(BeNil(), "devices: %+v", devices)
		Expect(cpu.Name).To(Equal("CPU"))
		Expect(cpu.StoresTensors).To(BeTrue())
	})

	It("should list a GPU with its memory", Label("gpu"), func() {
		var gpu *llama.Device
		for _, d := range llama.Devices() {
			if d.Type == llama.DeviceGPU {
				gpu = &d
				break
			}
		}
		Expect(gpu).NotTo(BeNil())
		Expect(gpu.Name).NotTo(BeEmpty())
		Expect(gpu.Description).NotTo(BeEmpty())
		Expect(gpu.StoresTensors).To(BeTrue())
		Expect(gpu.TotalMemory).To(BeNumerically(">", 0))
		Expect(gpu.FreeMemory).To(BeNumerically(">", 0))
		// With unified memory, CUDA reports the host's available memory as
		// free, which can exceed the device's own total
		if os.Getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") == "" {
			Expect(gpu.FreeMemory).To(BeNumerically("<=", gpu.TotalMemory))
		}
	})
})

var _ = Describe("Tensor overrides", Label("tensor-overrides"), func() {
	var modelPath string
	var model *llama.Model

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_MOE_MODEL")
		if modelPath == "" {
			Skip("TEST_MOE_MODEL not set - skipping tensor override tests")
		}
	})

	AfterEach(func() {
		if model != nil {
			Expect(model.Close()).To(Succeed())
			model = nil
		}
	})

	generate := func(opts ...llama.ModelOption) string {
		var err error
		model, err = llama.LoadModel(modelPath, opts...)
		Expect(err).NotTo(HaveOccurred())
		// q8_0, the default cache type, stores blocks of 32, which its 48-wide heads cannot hold
		ctx, err := model.NewContext(llama.WithContext(256), llama.WithKVCacheType("f16"))
		Expect(err).NotTo(HaveOccurred())
		defer func() { Expect(ctx.Close()).To(Succeed()) }()
		result, err := ctx.Generate("Once upon a time",
			llama.WithMaxTokens(16),
			llama.WithTemperature(0),
			llama.WithIgnoreEOS(true),
		)
		Expect(err).NotTo(HaveOccurred())
		return result
	}

	It("should generate with every expert on the CPU", Label("integration"), func() {
		Expect(generate(llama.WithCPUMoE())).NotTo(BeEmpty())
	})

	It("should generate with the first layers' experts on the CPU", Label("integration"), func() {
		Expect(generate(llama.WithNCPUMoE(2))).NotTo(BeEmpty())
	})

	It("should generate with a pattern sent to a listed device", Label("integration"), func() {
		// A GPU when there is one, so the override moves tensors off the CPU
		target := "CPU"
		for _, d := range llama.Devices() {
			if d.StoresTensors && d.Type != llama.DeviceCPU {
				target = d.Name
				break
			}
		}
		Expect(generate(llama.WithGPULayers(0), llama.WithTensorOverride(`\.ffn_.*`, target))).NotTo(BeEmpty())
	})

	It("should do nothing for no layers", Label("integration"), func() {
		Expect(generate(llama.WithNCPUMoE(0))).NotTo(BeEmpty())
	})

	It("should name the available devices when one doesn't exist", Label("integration"), func() {
		m, err := llama.LoadModel(modelPath, llama.WithTensorOverride(`\.ffn_.*`, "NOPE0"))
		Expect(err).To(HaveOccurred())
		Expect(m).To(BeNil())
		Expect(err.Error()).To(ContainSubstring(`Unknown tensor override device "NOPE0"`))
		Expect(err.Error()).To(ContainSubstring("CPU"))
	})

	It("should reject an invalid pattern", Label("integration"), func() {
		m, err := llama.LoadModel(modelPath, llama.WithTensorOverride(`blk\.(`, "CPU"))
		Expect(err).To(HaveOccurred())
		Expect(m).To(BeNil())
		Expect(err.Error()).To(ContainSubstring("Invalid tensor override pattern"))
	})
})
