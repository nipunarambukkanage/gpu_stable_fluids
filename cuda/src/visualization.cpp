#include "gpu_fluids/visualization.hpp"

#include <fstream>
#include <stdexcept>

namespace gpu_fluids {

void writePpm(const std::filesystem::path& path, int width, int height, const std::uint8_t* rgba) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("Could not open frame output: " + path.string());
  }
  output << "P6\n" << width << " " << height << "\n255\n";
  for (int pixel = 0; pixel < width * height; ++pixel) {
    output.put(static_cast<char>(rgba[pixel * 4]));
    output.put(static_cast<char>(rgba[pixel * 4 + 1]));
    output.put(static_cast<char>(rgba[pixel * 4 + 2]));
  }
  if (!output) {
    throw std::runtime_error("Could not write frame output: " + path.string());
  }
}

}  // namespace gpu_fluids
