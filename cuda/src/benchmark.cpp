#include "gpu_fluids/benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace gpu_fluids {
namespace {

constexpr double kBytesPerGigabyte = 1.0e9;

double percentile95(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const auto rank = static_cast<std::size_t>(std::ceil(static_cast<double>(values.size()) * 0.95));
  const std::size_t index = rank == 0 ? 0 : std::min(values.size() - 1, rank - 1);
  return values[index];
}

}  // namespace

BenchmarkLedger::BenchmarkLedger(std::size_t expectedSamples) {
  reserve(expectedSamples);
}

void BenchmarkLedger::reserve(std::size_t expectedSamples) {
  samples_.reserve(expectedSamples);
}

void BenchmarkLedger::record(std::string stage, double milliseconds, std::size_t bytesTransferred) {
  if (stage.empty()) {
    throw std::invalid_argument("benchmark stage names cannot be empty");
  }
  if (!std::isfinite(milliseconds) || milliseconds < 0.0) {
    throw std::invalid_argument("benchmark duration must be finite and non-negative");
  }
  samples_.push_back({std::move(stage), milliseconds, bytesTransferred});
}

void BenchmarkLedger::clear() noexcept {
  samples_.clear();
}

BenchmarkSummary BenchmarkLedger::summarize() const {
  BenchmarkSummary summary;
  summary.samples = samples_.size();
  std::vector<double> durations;
  durations.reserve(samples_.size());
  for (const BenchmarkSample& sample : samples_) {
    summary.bytesTransferred += sample.bytesTransferred;
    summary.totalMilliseconds += sample.milliseconds;
    durations.push_back(sample.milliseconds);
  }
  if (summary.samples == 0) {
    return summary;
  }
  summary.averageMilliseconds = summary.totalMilliseconds / static_cast<double>(summary.samples);
  summary.p95Milliseconds = percentile95(std::move(durations));
  if (summary.totalMilliseconds > 0.0) {
    summary.effectiveBandwidthGBPerSecond =
        static_cast<double>(summary.bytesTransferred) /
        (summary.totalMilliseconds * 1.0e-3) / kBytesPerGigabyte;
  }
  return summary;
}

std::string BenchmarkLedger::escape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 8);
  for (const char character : value) {
    switch (character) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += character;
        break;
    }
  }
  return escaped;
}

std::string BenchmarkLedger::toJson() const {
  const BenchmarkSummary summary = summarize();
  std::ostringstream json;
  json << std::fixed << std::setprecision(6);
  json << "{\n  \"samples\": " << summary.samples
       << ",\n  \"bytesTransferred\": " << summary.bytesTransferred
       << ",\n  \"totalMilliseconds\": " << summary.totalMilliseconds
       << ",\n  \"averageMilliseconds\": " << summary.averageMilliseconds
       << ",\n  \"p95Milliseconds\": " << summary.p95Milliseconds
       << ",\n  \"effectiveBandwidthGBPerSecond\": "
       << summary.effectiveBandwidthGBPerSecond << ",\n  \"stages\": [\n";
  for (std::size_t index = 0; index < samples_.size(); ++index) {
    const BenchmarkSample& sample = samples_[index];
    json << "    {\"stage\": \"" << escape(sample.stage)
         << "\", \"milliseconds\": " << sample.milliseconds
         << ", \"bytesTransferred\": " << sample.bytesTransferred << "}";
    json << (index + 1 == samples_.size() ? "\n" : ",\n");
  }
  json << "  ]\n}";
  return json.str();
}

}  // namespace gpu_fluids
