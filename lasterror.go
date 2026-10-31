package llama

/*
#include "wrapper.h"
*/
import "C"

import "runtime"

// withLastError runs a call into the wrapper and returns its result with the message the
// wrapper set if it failed. The wrapper keeps that message per thread, and Go may move a
// goroutine to another thread between two cgo calls, so the call and the read share one
// locked thread. Every call whose failure is reported through llama_wrapper_last_error goes
// through here.
func withLastError[T any](call func() T) (T, string) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	C.llama_wrapper_clear_last_error()
	result := call()
	return result, C.GoString(C.llama_wrapper_last_error())
}
