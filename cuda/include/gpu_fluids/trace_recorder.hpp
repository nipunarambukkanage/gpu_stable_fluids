#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace gpu_fluids {

struct TraceEvent {
  std::string name;
  std::string category;
  std::uint64_t timestampMicroseconds = 0;
  std::uint64_t durationMicroseconds = 0;
  std::uint64_t frameIndex = 0;
  std::uint32_t processId = 1;
  std::uint32_t threadId = 1;
};

class TraceRecorder final {
 public:
  using Clock = std::chrono::steady_clock;

  void clear() noexcept;
  void beginFrame(std::uint64_t frameIndex);
  void endFrame();
  void begin(std::string name, std::string category);
  void end(std::string name, std::string category);
  void instant(std::string name, std::string category);

  [[nodiscard]] bool frameActive() const noexcept { return frameActive_; }
  [[nodiscard]] const std::vector<TraceEvent>& events() const noexcept { return events_; }
  [[nodiscard]] std::uint64_t frameCount() const noexcept { return frameCount_; }

  void writeChromeTrace(const std::filesystem::path& path) const;

 private:
  struct OpenEvent {
    std::string name;
    std::string category;
    Clock::time_point start;
    std::uint64_t frameIndex = 0;
  };

  [[nodiscard]] static std::uint64_t timestampMicroseconds(Clock::time_point time) noexcept;
  [[nodiscard]] static std::string escape(const std::string& value);
  [[nodiscard]] std::size_t findOpenEvent(const std::string& name,
                                           const std::string& category) const noexcept;

  std::vector<TraceEvent> events_;
  std::vector<OpenEvent> openEvents_;
  Clock::time_point epoch_ = Clock::now();
  std::uint64_t currentFrame_ = 0;
  std::uint64_t frameCount_ = 0;
  bool frameActive_ = false;
};

}  // namespace gpu_fluids
