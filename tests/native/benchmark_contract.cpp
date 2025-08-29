#include "gpu_fluids/benchmark.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

int fail(const char* message) {
  std::cerr << "native_benchmark_contract: " << message << '\n';
  return 1;
}

template <typename Callable>
bool throwsInvalidArgument(Callable&& callable) {
  try {
    callable();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

}  // namespace

int main() {
  gpu_fluids::BenchmarkLedger ledger(3);
  ledger.record("advection", 2.0, 1'000'000);
  ledger.record("pressure\"jacobi", 4.0, 500'000);
  ledger.record("render", 1.0, 250'000);

  const gpu_fluids::BenchmarkSummary summary = ledger.summarize();
  if (summary.samples != 3 || summary.bytesTransferred != 1'750'000 ||
      std::abs(summary.totalMilliseconds - 7.0) > 1.0e-9 ||
      std::abs(summary.averageMilliseconds - (7.0 / 3.0)) > 1.0e-9 ||
      std::abs(summary.p95Milliseconds - 4.0) > 1.0e-9 ||
      std::abs(summary.effectiveBandwidthGBPerSecond - 0.25) > 1.0e-9) {
    return fail("benchmark summary statistics are incorrect");
  }

  const std::string json = ledger.toJson();
  if (json.find("\"samples\": 3") == std::string::npos ||
      json.find("\"p95Milliseconds\": 4.000000") == std::string::npos ||
      json.find("pressure\\\"jacobi") == std::string::npos) {
    return fail("benchmark JSON is incomplete or incorrectly escaped");
  }

  if (!throwsInvalidArgument([&] { ledger.record("", 1.0); }) ||
      !throwsInvalidArgument([&] {
        ledger.record("invalid", std::numeric_limits<double>::quiet_NaN());
      }) ||
      !throwsInvalidArgument([&] { ledger.record("invalid", -1.0); })) {
    return fail("invalid benchmark samples were accepted");
  }

  ledger.clear();
  const gpu_fluids::BenchmarkSummary empty = ledger.summarize();
  if (!ledger.empty() || empty.samples != 0 || empty.totalMilliseconds != 0.0 ||
      empty.effectiveBandwidthGBPerSecond != 0.0) {
    return fail("benchmark reset did not restore an empty ledger");
  }

  std::cout << "native_benchmark_contract: passed\n";
  return 0;
}
