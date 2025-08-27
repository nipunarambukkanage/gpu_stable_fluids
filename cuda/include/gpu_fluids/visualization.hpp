#pragma once

#include <cstdint>
#include <filesystem>

namespace gpu_fluids {

void writePpm(const std::filesystem::path& path, int width, int height, const std::uint8_t* rgba);

}  // namespace gpu_fluids
