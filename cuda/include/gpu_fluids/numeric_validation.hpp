#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gpu_fluids {

enum class ValidationSeverity : std::uint8_t {
  Info,
  Warning,
  Error,
};

struct ValidationIssue {
  ValidationSeverity severity = ValidationSeverity::Info;
  std::string code;
  std::string message;
};

class ValidationReport final {
 public:
  void add(ValidationSeverity severity, std::string code, std::string message);
  void info(std::string code, std::string message);
  void warning(std::string code, std::string message);
  void error(std::string code, std::string message);

  [[nodiscard]] bool valid() const noexcept { return errorCount_ == 0; }
  [[nodiscard]] std::size_t checkCount() const noexcept { return checkCount_; }
  [[nodiscard]] std::size_t errorCount() const noexcept { return errorCount_; }
  [[nodiscard]] const std::vector<ValidationIssue>& issues() const noexcept { return issues_; }
  [[nodiscard]] std::string toJson() const;
  [[nodiscard]] std::string summary() const;

 private:
  std::vector<ValidationIssue> issues_;
  std::size_t checkCount_ = 0;
  std::size_t errorCount_ = 0;
};

ValidationReport validateFiniteField(const std::vector<float>& field,
                                     const char* fieldName,
                                     float maximumAbsoluteValue);
ValidationReport validateRgbaFrame(const std::vector<std::uint8_t>& rgba,
                                   int width,
                                   int height);
ValidationReport validateSimulationDiagnostics(float maxSpeed,
                                               float dyeEnergy,
                                               std::size_t expectedCellCount);

}  // namespace gpu_fluids
