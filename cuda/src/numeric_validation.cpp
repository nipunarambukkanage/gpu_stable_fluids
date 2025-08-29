#include "gpu_fluids/numeric_validation.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace gpu_fluids {
namespace {

const char* severityName(ValidationSeverity severity) noexcept {
  switch (severity) {
    case ValidationSeverity::Info:
      return "info";
    case ValidationSeverity::Warning:
      return "warning";
    case ValidationSeverity::Error:
      return "error";
  }
  return "unknown";
}

std::string jsonEscape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
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

}  // namespace

void ValidationReport::add(ValidationSeverity severity, std::string code, std::string message) {
  ++checkCount_;
  if (severity == ValidationSeverity::Error) {
    ++errorCount_;
  }
  issues_.push_back({severity, std::move(code), std::move(message)});
}

void ValidationReport::info(std::string code, std::string message) {
  add(ValidationSeverity::Info, std::move(code), std::move(message));
}

void ValidationReport::warning(std::string code, std::string message) {
  add(ValidationSeverity::Warning, std::move(code), std::move(message));
}

void ValidationReport::error(std::string code, std::string message) {
  add(ValidationSeverity::Error, std::move(code), std::move(message));
}

std::string ValidationReport::summary() const {
  std::ostringstream value;
  value << (valid() ? "valid" : "invalid") << ", checks=" << checkCount_
        << ", errors=" << errorCount_ << ", issues=" << issues_.size();
  return value.str();
}

std::string ValidationReport::toJson() const {
  std::ostringstream value;
  value << "{\"valid\": " << (valid() ? "true" : "false")
        << ", \"checks\": " << checkCount_ << ", \"errors\": " << errorCount_
        << ", \"issues\": [";
  for (std::size_t index = 0; index < issues_.size(); ++index) {
    const auto& issue = issues_[index];
    value << "{\"severity\": \"" << severityName(issue.severity)
          << "\", \"code\": \"" << jsonEscape(issue.code)
          << "\", \"message\": \"" << jsonEscape(issue.message) << "\"}";
    if (index + 1 != issues_.size()) {
      value << ", ";
    }
  }
  value << "]}";
  return value.str();
}

ValidationReport validateFiniteField(const std::vector<float>& field,
                                     const char* fieldName,
                                     float maximumAbsoluteValue) {
  if (fieldName == nullptr || *fieldName == '\0') {
    throw std::invalid_argument("finite-field validation requires a field name");
  }
  if (!std::isfinite(maximumAbsoluteValue) || maximumAbsoluteValue <= 0.0F) {
    throw std::invalid_argument("finite-field validation requires a positive finite limit");
  }

  ValidationReport report;
  report.info("field-size", std::string(fieldName) + " contains " + std::to_string(field.size()) + " values");
  for (std::size_t index = 0; index < field.size(); ++index) {
    const float value = field[index];
    if (!std::isfinite(value)) {
      report.error("non-finite", std::string(fieldName) + " contains a non-finite value at index " + std::to_string(index));
      break;
    }
    if (std::abs(value) > maximumAbsoluteValue) {
      report.warning("range", std::string(fieldName) + " exceeds the configured range at index " + std::to_string(index));
      break;
    }
  }
  if (field.empty()) {
    report.warning("empty", std::string(fieldName) + " is empty");
  }
  return report;
}

ValidationReport validateRgbaFrame(const std::vector<std::uint8_t>& rgba, int width, int height) {
  ValidationReport report;
  if (width <= 0 || height <= 0) {
    report.error("dimensions", "RGBA frame dimensions must be positive");
    return report;
  }
  const std::size_t expectedBytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4U;
  if (rgba.size() != expectedBytes) {
    report.error("byte-count", "RGBA frame byte count does not match its dimensions");
    return report;
  }
  report.info("byte-count", "RGBA frame byte count matches dimensions");
  std::size_t transparentPixels = 0;
  for (std::size_t pixel = 0; pixel < rgba.size(); pixel += 4) {
    if (rgba[pixel + 3] != 255) {
      ++transparentPixels;
    }
  }
  if (transparentPixels == 0) {
    report.info("alpha", "all rendered pixels are opaque");
  } else {
    report.warning("alpha", std::to_string(transparentPixels) + " pixels have non-opaque alpha");
  }
  return report;
}

ValidationReport validateSimulationDiagnostics(float maxSpeed,
                                               float dyeEnergy,
                                               std::size_t expectedCellCount) {
  ValidationReport report;
  if (!std::isfinite(maxSpeed) || maxSpeed < 0.0F) {
    report.error("max-speed", "maximum speed is not finite and non-negative");
  } else {
    report.info("max-speed", "maximum speed is finite");
  }
  if (!std::isfinite(dyeEnergy) || dyeEnergy < 0.0F) {
    report.error("dye-energy", "dye energy is not finite and non-negative");
  } else {
    report.info("dye-energy", "dye energy is finite");
  }
  if (expectedCellCount == 0) {
    report.error("cell-count", "expected cell count must be non-zero");
  } else {
    report.info("cell-count", "expected cell count is non-zero");
  }
  if (maxSpeed > std::numeric_limits<float>::max() * 0.25F) {
    report.warning("speed-range", "maximum speed is close to float overflow");
  }
  return report;
}

}  // namespace gpu_fluids
