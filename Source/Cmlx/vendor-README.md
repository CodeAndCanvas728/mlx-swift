# vendored code

Local copies of code that cannot be pulled as part of the swiftpm build.

## json

This is https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz

## metal-cpp

This comes from https://developer.apple.com/metal/cpp/ specifically:

- https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18-beta.zip

Note that `metal-cpp.patch` has been applied to the contents of that zip.

## fmt

This is https://github.com/fmtlib/fmt.git tag 12.1.0

## mlx and mlx-c (SharpAI fork)

`mlx/` and `mlx-c/` are not submodules in this fork: they are plain directories holding the
upstream release that `ml-explore/mlx-swift` pins (see `UPSTREAM_MERGE_PLAN.md` for the exact tag and
commit) plus the SharpAI patches (SSD expert streaming, TurboKV, fast loaders). `mlx-generated/`
is produced by `tools/update-mlx.sh`.
