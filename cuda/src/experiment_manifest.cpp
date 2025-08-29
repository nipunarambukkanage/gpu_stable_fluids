#include "gpu_fluids/experiment_manifest.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace gpu_fluids {
namespace {

std::string pathString(const std::filesystem::path& path) {
  return path.generic_string();
}

}  // namespace

void ExperimentManifest::setRaw(std::string key, std::string jsonValue) {
  if (key.empty()) {
    throw std::invalid_argument("experiment manifest keys cannot be empty");
  }
  for (Entry& entry : entries_) {
    if (entry.key == key) {
      entry.jsonValue = std::move(jsonValue);
      return;
    }
  }
  entries_.push_back({std::move(key), std::move(jsonValue)});
}

void ExperimentManifest::setString(std::string key, std::string value) {
  setRaw(std::move(key), "\"" + escape(value) + "\"");
}

void ExperimentManifest::setInteger(std::string key, long long value) {
  setRaw(std::move(key), std::to_string(value));
}

void ExperimentManifest::setNumber(std::string key, double value) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument("experiment manifest numbers must be finite");
  }
  std::ostringstream number;
  number << std::fixed << std::setprecision(6) << value;
  setRaw(std::move(key), number.str());
}

void ExperimentManifest::setBoolean(std::string key, bool value) {
  setRaw(std::move(key), value ? "true" : "false");
}

void ExperimentManifest::setStringArray(std::string key, std::vector<std::string> values) {
  std::ostringstream array;
  array << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    array << '"' << escape(values[index]) << '"';
    if (index + 1 != values.size()) {
      array << ", ";
    }
  }
  array << ']';
  setRaw(std::move(key), array.str());
}

std::string ExperimentManifest::escape(const std::string& value) {
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

std::string ExperimentManifest::toJson() const {
  std::ostringstream json;
  json << "{\n";
  for (std::size_t index = 0; index < entries_.size(); ++index) {
    const Entry& entry = entries_[index];
    json << "  \"" << escape(entry.key) << "\": " << entry.jsonValue;
    json << (index + 1 == entries_.size() ? "\n" : ",\n");
  }
  json << "}\n";
  return json.str();
}

void ExperimentManifest::write(const std::filesystem::path& path) const {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("Could not open experiment manifest: " + path.string());
  }
  output << toJson();
  if (!output) {
    throw std::runtime_error("Could not write experiment manifest: " + path.string());
  }
}

ExperimentManifest makeRuntimeManifest(const RuntimeConfig& config, const RuntimeReport& report) {
  ExperimentManifest manifest;
  manifest.setString("schema", "gpu-stable-fluids/native-experiment/v1");
  manifest.setString("backend", report.backend);
  manifest.setString("executionModel", "persistent-state-fixed-timestep");
  manifest.setInteger("width", config.width);
  manifest.setInteger("height", config.height);
  manifest.setInteger("frameLimit", config.frameLimit);
  manifest.setInteger("exportEvery", config.exportEvery);
  manifest.setInteger("pressureIterations", config.pressureIterations);
  manifest.setNumber("fixedDeltaTime", config.fixedDeltaTime);
  manifest.setBoolean("exportFrames", config.exportFrames);
  manifest.setBoolean("writeTelemetry", config.writeTelemetry);
  manifest.setInteger("framesSimulated", static_cast<long long>(report.framesSimulated));
  manifest.setInteger("framesExported", static_cast<long long>(report.framesExported));
  manifest.setInteger("commandsApplied", static_cast<long long>(report.commandsApplied));
  manifest.setInteger("validationChecks", static_cast<long long>(report.validationChecks));
  manifest.setInteger("validationFailures", static_cast<long long>(report.validationFailures));
  manifest.setNumber("maximumObservedSpeed", report.maximumObservedSpeed);
  manifest.setNumber("finalDyeEnergy", report.finalDyeEnergy);
  manifest.setNumber("averageFrameMilliseconds", report.telemetry.averageFrameMilliseconds);
  manifest.setNumber("p95FrameMilliseconds", report.telemetry.p95FrameMilliseconds);

  std::vector<std::string> exportedPaths;
  exportedPaths.reserve(report.exports.size());
  for (const ExportedFrame& frame : report.exports) {
    exportedPaths.push_back(pathString(frame.path));
  }
  manifest.setStringArray("exportedFrames", std::move(exportedPaths));
  return manifest;
}

}  // namespace gpu_fluids
