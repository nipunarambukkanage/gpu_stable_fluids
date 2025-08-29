#include "gpu_fluids/trace_recorder.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace gpu_fluids {

void TraceRecorder::clear() noexcept {
  events_.clear();
  openEvents_.clear();
  currentFrame_ = 0;
  frameCount_ = 0;
  frameActive_ = false;
  epoch_ = Clock::now();
}

void TraceRecorder::beginFrame(std::uint64_t frameIndex) {
  if (frameActive_) {
    endFrame();
  }
  currentFrame_ = frameIndex;
  frameActive_ = true;
}

void TraceRecorder::endFrame() {
  if (!frameActive_) {
    return;
  }
  while (!openEvents_.empty()) {
    end(openEvents_.back().name, openEvents_.back().category);
  }
  ++frameCount_;
  frameActive_ = false;
}

void TraceRecorder::begin(std::string name, std::string category) {
  if (!frameActive_ || name.empty() || category.empty()) {
    return;
  }
  openEvents_.push_back({std::move(name), std::move(category), Clock::now(), currentFrame_});
}

void TraceRecorder::end(std::string name, std::string category) {
  const auto openIndex = findOpenEvent(name, category);
  if (openIndex == openEvents_.size()) {
    return;
  }
  const OpenEvent open = std::move(openEvents_[openIndex]);
  openEvents_.erase(openEvents_.begin() + static_cast<std::ptrdiff_t>(openIndex));
  const auto now = Clock::now();
  const auto timestamp = timestampMicroseconds(open.start);
  const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - open.start).count();
  events_.push_back({open.name, open.category, timestamp, static_cast<std::uint64_t>(std::max<std::int64_t>(0, duration)),
                     open.frameIndex, 1, 1});
}

void TraceRecorder::instant(std::string name, std::string category) {
  if (!frameActive_ || name.empty() || category.empty()) {
    return;
  }
  events_.push_back({std::move(name), std::move(category), timestampMicroseconds(Clock::now()), 0, currentFrame_, 1, 1});
}

std::uint64_t TraceRecorder::timestampMicroseconds(Clock::time_point time) noexcept {
  const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(time - epoch_).count();
  return static_cast<std::uint64_t>(std::max<std::int64_t>(0, elapsed));
}

std::size_t TraceRecorder::findOpenEvent(const std::string& name,
                                         const std::string& category) const noexcept {
  for (std::size_t index = openEvents_.size(); index > 0; --index) {
    const auto& event = openEvents_[index - 1];
    if (event.name == name && event.category == category) {
      return index - 1;
    }
  }
  return openEvents_.size();
}

std::string TraceRecorder::escape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 4);
  for (const char character : value) {
    if (character == '\\' || character == '"') {
      escaped.push_back('\\');
    }
    if (character == '\n') {
      escaped += "\\n";
    } else {
      escaped.push_back(character);
    }
  }
  return escaped;
}

void TraceRecorder::writeChromeTrace(const std::filesystem::path& path) const {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("Could not open Chrome trace output: " + path.string());
  }
  output << "{\"traceEvents\":[";
  for (std::size_t index = 0; index < events_.size(); ++index) {
    const auto& event = events_[index];
    output << "{\"name\":\"" << escape(event.name) << "\",\"cat\":\"" << escape(event.category)
            << "\",\"ph\":\"X\",\"ts\":" << event.timestampMicroseconds
            << ",\"dur\":" << event.durationMicroseconds << ",\"pid\":" << event.processId
            << ",\"tid\":" << event.threadId << ",\"args\":{\"frame\":" << event.frameIndex << "}}";
    if (index + 1 != events_.size()) {
      output << ',';
    }
  }
  output << "]}\n";
  if (!output) {
    throw std::runtime_error("Could not write Chrome trace output: " + path.string());
  }
}

}  // namespace gpu_fluids
