#pragma once

#include "gpu_fluids/native_runtime.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace gpu_fluids {

class ExperimentManifest final {
 public:
  void setString(std::string key, std::string value);
  void setInteger(std::string key, long long value);
  void setNumber(std::string key, double value);
  void setBoolean(std::string key, bool value);
  void setStringArray(std::string key, std::vector<std::string> values);

  [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }
  [[nodiscard]] std::string toJson() const;
  void write(const std::filesystem::path& path) const;

 private:
  struct Entry {
    std::string key;
    std::string jsonValue;
  };

  static std::string escape(const std::string& value);
  void setRaw(std::string key, std::string jsonValue);

  std::vector<Entry> entries_;
};

ExperimentManifest makeRuntimeManifest(const RuntimeConfig& config,
                                       const RuntimeReport& report);

}  // namespace gpu_fluids
