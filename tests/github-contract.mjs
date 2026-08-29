import { existsSync, readFileSync } from "node:fs";

const requiredFiles = [
  ".github/dependabot.yml",
  ".github/workflows/codeql.yml",
  ".github/workflows/native-cuda.yml",
  ".github/workflows/quality.yml",
  ".github/ISSUE_TEMPLATE/bug_report.yml",
  ".github/ISSUE_TEMPLATE/feature_request.yml",
  ".github/ISSUE_TEMPLATE/config.yml",
  ".github/pull_request_template.md",
  "tools/check-source-share.mjs"
];

const failures = [];
const assert = (condition, message) => {
  if (!condition) failures.push(message);
};
const read = (filePath) => readFileSync(filePath, "utf8");

for (const filePath of requiredFiles) {
  assert(existsSync(filePath), `missing GitHub quality file: ${filePath}`);
}

const quality = read(".github/workflows/quality.yml");
const codeql = read(".github/workflows/codeql.yml");
const native = read(".github/workflows/native-cuda.yml");
const dependabot = read(".github/dependabot.yml");
const pullRequest = read(".github/pull_request_template.md");
const sourceShare = read("tools/check-source-share.mjs");
const readme = read("README.md");

assert(quality.includes("npm run check") && quality.includes("node-version: 22"), "quality workflow must run repository checks on Node 22");
assert(quality.includes("cancel-in-progress: true"), "quality workflow must cancel superseded runs");
assert(codeql.includes("github/codeql-action/init@v3") && codeql.includes("github/codeql-action/analyze@v3"), "CodeQL workflow must use the supported CodeQL actions");
assert(codeql.includes("javascript-typescript") && codeql.includes("security-events: write"), "CodeQL workflow must configure JavaScript analysis and security-event permissions");
assert(native.includes("cmake --preset cuda-release") && native.includes("ctest --preset cuda-release"), "native workflow must configure, build, and test the CUDA preset");
assert(native.includes("upload-artifact@v4") && native.includes("fluid_cuda_demo.exe") && native.includes("--frames 2") && native.includes("--output build\\ci-smoke"), "native workflow must retain a valid small smoke run and diagnostics artifact");
assert(dependabot.includes("package-ecosystem: npm") && dependabot.includes("interval: monthly"), "Dependabot must maintain the npm toolchain monthly");
assert(pullRequest.includes("npm run check") && pullRequest.includes("Host/device transfers"), "PR template must require validation and GPU ownership review");
assert(sourceShare.includes("nativeLines * 2 <= codeLines"), "source-share contract must enforce a strict native-code majority");
assert(readme.includes("actions/workflows/quality.yml/badge.svg") && readme.includes("actions/workflows/codeql.yml/badge.svg"), "README must expose actionable GitHub workflow badges");

if (failures.length > 0) {
  console.error("github-contract: failed");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exitCode = 1;
} else {
  console.log("github-contract: passed");
  console.log("- CI, CodeQL, Dependabot, issue forms, and PR review contracts found");
}
