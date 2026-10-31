package llama

/*
#include "wrapper.h"
*/
import "C"

// DeviceType is the kind of hardware a Device is.
type DeviceType int

// Device types, as llama.cpp's backends report them.
const (
	DeviceCPU           DeviceType = iota // CPU, using system memory
	DeviceGPU                             // GPU with its own memory
	DeviceIntegratedGPU                   // GPU sharing system memory
	DeviceAccelerator                     // Accelerator used alongside the CPU, such as BLAS
	DeviceMeta                            // Several devices combined for tensor parallelism
)

// String returns the type's name.
func (t DeviceType) String() string {
	switch t {
	case DeviceCPU:
		return "CPU"
	case DeviceGPU:
		return "GPU"
	case DeviceIntegratedGPU:
		return "integrated GPU"
	case DeviceAccelerator:
		return "accelerator"
	case DeviceMeta:
		return "meta"
	}
	return "unknown"
}

// Device is a compute device available to llama.cpp.
type Device struct {
	Name        string     // Name to give WithTensorOverride (e.g. "CPU", "CUDA0")
	Description string     // Hardware description (e.g. "NVIDIA GeForce RTX 3090")
	Type        DeviceType // Kind of device
	// Whether tensors can be stored on the device; an accelerator may only compute
	// on tensors held by the CPU
	StoresTensors bool
	FreeMemory    uint64 // Free memory in bytes, as the backend reports it
	TotalMemory   uint64 // Total memory in bytes
}

// Devices lists the compute devices this build of llama.cpp can use, on any
// backend, in llama.cpp's order. Use it to find the names WithTensorOverride
// accepts before loading a model.
//
// Example:
//
//	for _, d := range llama.Devices() {
//	    fmt.Printf("%s: %s, %d MiB free\n", d.Name, d.Description, d.FreeMemory>>20)
//	}
func Devices() []Device {
	n := int(C.llama_wrapper_device_count())
	devices := make([]Device, 0, n)
	for i := range n {
		var info C.llama_wrapper_device_info
		if !C.llama_wrapper_device_info_get(C.int(i), &info) {
			continue
		}
		devices = append(devices, Device{
			Name:          C.GoString(&info.name[0]),
			Description:   C.GoString(&info.description[0]),
			Type:          DeviceType(info._type),
			StoresTensors: bool(info.has_buffer),
			FreeMemory:    uint64(info.free_memory),
			TotalMemory:   uint64(info.total_memory),
		})
	}
	return devices
}
