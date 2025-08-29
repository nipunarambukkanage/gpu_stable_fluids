#include "gpu_fluids/trace_recorder.hpp"

#include <filesystem>
#include <iostream>
#include <string>

namespace {

int fail(const char* message) {
  std::cerr << "native_trace_recorder_contract: " << message << '\n';
  return 1;
}

}  // namespace

int main() {
  gpu_fluids::TraceRecorder trace;
  trace.beginFrame(7);
  trace.begin("solver", "cuda");
  trace.instant("barrier", "cuda");
  trace.end("solver", "cuda");
  trace.endFrame();
  if (trace.frameCount() != 1 || trace.events().size() != 2 || trace.frameActive()) {
    return fail("trace recorder did not close and retain frame events");
  }
  if (trace.events().front().frameIndex != 7 || trace.events().front().name.empty()) {
    return fail("trace event metadata is incomplete");
  }

  const std::filesystem::path output =
      std::filesystem::temp_directory_path() / "gpu-stable-fluids-trace-contract.json";
  trace.writeChromeTrace(output);
  if (!std::filesystem::exists(output)) {
    return fail("Chrome trace was not written");
  }
  std::filesystem::remove(output);
  trace.clear();
  if (!trace.events().empty() || trace.frameCount() != 0) {
    return fail("trace clear did not reset recorder state");
  }
  std::cout << "native_trace_recorder_contract: passed\n";
  return 0;
}
