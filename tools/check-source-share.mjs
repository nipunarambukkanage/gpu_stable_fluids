import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { extname } from "node:path";

const nativeExtensions = new Set([".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp"]);
const codeExtensions = new Set([...nativeExtensions, ".css", ".html", ".js", ".mjs", ".ts", ".wgsl"]);

const trackedFiles = execFileSync("git", ["ls-files", "-z"], { encoding: "utf8" })
  .split("\0")
  .filter(Boolean);

const lineCount = (filePath) => {
  const text = readFileSync(filePath, "utf8");
  return text.length === 0 ? 0 : text.split(/\r?\n/).length - (text.endsWith("\n") || text.endsWith("\r") ? 1 : 0);
};

let nativeLines = 0;
let codeLines = 0;
let codeFiles = 0;
for (const filePath of trackedFiles) {
  const extension = extname(filePath).toLowerCase();
  if (!codeExtensions.has(extension)) continue;
  const lines = lineCount(filePath);
  codeLines += lines;
  codeFiles += 1;
  if (nativeExtensions.has(extension)) nativeLines += lines;
}

const share = codeLines === 0 ? 0 : (nativeLines / codeLines) * 100;
console.log(`native source: ${nativeLines} lines`);
console.log(`tracked code: ${codeLines} lines across ${codeFiles} files`);
console.log(`native C++/CUDA share: ${share.toFixed(2)}%`);

if (nativeLines * 2 <= codeLines) {
  console.error("source-share: failed; native C++/CUDA code must exceed 50% of tracked code lines");
  process.exitCode = 1;
} else {
  console.log("source-share: passed; native C++/CUDA remains the majority implementation");
}
