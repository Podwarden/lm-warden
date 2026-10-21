// swift-tools-version: 5.10
// LM Warden macOS app — Swift package skeleton (Task 6).
//
// This package will grow into the native macOS shell (SwiftUI) and the
// MLX/Metal engines described in docs/superpowers/specs/2026-09-09-macos-metal-design.md.
// For now it is a minimal, building, tested skeleton: an executable target
// `LMWarden` (the binary name Task 7's DMG script consumes), with an
// `AppVersion` helper, and an XCTest target.
//
// `swift-tools-version: 5.10` keeps the M1 builder's Swift 6.1 able to
// build this (the dev Mac runs Swift 6.2). XCTest works on both.
import PackageDescription

// Sparkle 2.x (P8 sparkle-embed): the auto-updater framework, linked here and
// shipped in Contents/Frameworks. PIN (decision recorded in
// scripts/macos/embed-sparkle.sh): the upstream SwiftPM binary artifact of the
// latest stable release, by exact URL + checksum (the sha256 of the zip; it is
// the value upstream's own Package.swift at tag 2.10.0 and GitHub's release
// asset digest both publish). SwiftPM refuses a download whose sha256 differs.
// embed-sparkle.sh carries the SAME version/URL/sha256 (tests keep them equal)
// and places the framework in the .app from its own verified copy of the zip.
let sparkleVersion = "2.10.0"
let sparkleURL = "https://github.com/sparkle-project/Sparkle/releases/download/\(sparkleVersion)/Sparkle-for-Swift-Package-Manager.zip"
let sparkleChecksum = "17e28312b8e18ab7cdbbe09a6fb28cc55a5479ec6c371dbc07cdecd2a14fd959"

let package = Package(
    name: "LMWarden",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "LMWarden",
            dependencies: ["Sparkle"],
            path: "Sources/LMWarden",
            // The bundled app loads Sparkle from Contents/Frameworks: the
            // binary links @rpath/Sparkle.framework/Versions/B/Sparkle, and
            // this rpath resolves it inside the bundle. (SwiftPM's own
            // @loader_path rpath keeps `swift run` / `swift test` finding the
            // copy in .build/<config>/.)
            linkerSettings: [
                .unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@executable_path/../Frameworks"]),
            ]
        ),
        .binaryTarget(
            name: "Sparkle",
            url: sparkleURL,
            checksum: sparkleChecksum
        ),
        .testTarget(
            name: "LMWardenTests",
            dependencies: ["LMWarden"],
            path: "Tests/LMWardenTests"
        ),
    ]
)
