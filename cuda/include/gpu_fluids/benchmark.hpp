#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace gpu_fluids {

struct BenchmarkSample final {
  std::string stage;
  double milliseconds = 0.0;
  std::size_t bytesTransferred = 0;
};

struct BenchmarkSummary final {
  std::size_t samples = 0;
  std::size_t bytesTransferred = 0;
  double totalMilliseconds = 0.0;
  double averageMilliseconds = 0.0;
  double p95Milliseconds = 0.0;
  double effectiveBandwidthGBPerSecond = 0.0;
};

class BenchmarkLedger final {
 public:
  explicit BenchmarkLedger(std::size_t expectedSamples = 0);

  void reserve(std::size_t expectedSamples);
  void record(std::string stage, double milliseconds, std::size_t bytesTransferred = 0);
  void clear() noexcept;

  [[nodiscard]] bool empty() const noexcept { return samples_.empty(); }
  [[nodiscard]] const std::vector<BenchmarkSample>& samples() const noexcept { return samples_; }
  [[nodiscard]] BenchmarkSummary summarize() const;
  [[nodiscard]] std::string toJson() const;

 private:
  static std::string escape(const std::string& value);

  std::vector<BenchmarkSample> samples_;
};

}  // namespace gpu_fluids
