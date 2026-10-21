// LMWarden — entry point for the native macOS app shell.
//
// `LaunchMode.resolve` picks what this launch means (planner iter 42):
// - `.gui`: no arguments inside LMWarden.app — the menu bar app
//   (NSApplication, `.accessory` policy, `ShellAppDelegate`).
// - `.version`: no arguments outside a bundle (e.g. `swift run LMWarden`, a
//   smoke test that the package builds and links) or `--version`.
// - `.headless`: `--headless --python <abs> [--app-support <abs>]` boots
//   `AppController` with the live environment and stops cleanly on
//   SIGINT/SIGTERM.
// - `.updateProbe`: `--update-probe` (inside the .app) checks the bundle's
//   Sparkle feed once, prints the outcome and exits (UpdateProbe).
import AppKit
import Foundation

let headlessUsage = "usage: LMWarden [--version | --update-probe | --headless --python <abs> [--app-support <abs>]]"
let headlessArguments = Array(CommandLine.arguments.dropFirst())

/// The installed signal sources. Held for the life of the process so neither
/// source is released (and cancelled) once `installSignal` returns.
///
/// `nonisolated(unsafe)`: top-level globals are main-actor isolated under
/// strict concurrency, but `installSignal` (nonisolated) appends to this. It
/// is only ever touched from `runHeadless` on the main thread, before
/// `dispatchMain()`, and never read, so there is no concurrent access.
nonisolated(unsafe) var headlessSignalSources: [DispatchSourceSignal] = []

/// The GUI app delegate. `NSApplication.delegate` is weak, so this global
/// keeps the delegate (and the coordinator it owns) alive for the process
/// lifetime.
var shellAppDelegate: ShellAppDelegate?

do {
    switch try LaunchMode.resolve(arguments: headlessArguments, bundlePath: Bundle.main.bundlePath) {
    case .version:
        print("LMWarden \(AppVersion.current)")
        exit(0)
    case .headless(let options):
        runHeadless(options: options)
    case .gui:
        runGUI()
    case .updateProbe:
        MainActor.assumeIsolated {
            UpdateProbe.run()
        }
    }
} catch {
    FileHandle.standardError.write(Data("LMWarden: \(error)\n\(headlessUsage)\n".utf8))
    exit(64)
}

/// Start the menu bar app on the main thread and never return.
private func runGUI() {
    MainActor.assumeIsolated {
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        let delegate = ShellAppDelegate()
        shellAppDelegate = delegate
        app.delegate = delegate
        app.run()
    }
    exit(0)
}

/// Install the signal handlers, boot the live server for `options`, and hand
/// control to the main run loop. `exit(1)` on a boot failure or when the
/// supervisor gives up, `exit(0)` after a clean signal-driven stop.
private func runHeadless(options: HeadlessOptions) {
    let controller = AppController(
        environment: .live(
            pythonPath: options.pythonPath,
            applicationSupport: options.applicationSupport ?? AppPaths.defaultApplicationSupport(),
            // Run from inside the .app, headless serves the bundle's UI too;
            // run from .build/release there is none and nil is passed.
            webuiDir: ShellAppDelegate.bundledWebUIDir()
        )
    )

    // Suppress the default disposition BEFORE the child is launched, so a
    // signal arriving during boot cannot tear the process down and leave the
    // child behind. The sources deliver on the main queue, which only runs
    // once `dispatchMain()` is reached, so a signal during boot is handled
    // after it.
    signal(SIGINT, SIG_IGN)
    signal(SIGTERM, SIG_IGN)

    var stopping = false
    func handleSignal() {
        // Stop once: a second signal (or SIGINT then SIGTERM) changes nothing.
        guard !stopping else {
            return
        }
        stopping = true
        controller.stop()
        // Give the supervisor its stop grace period (plus 1 s) before the
        // process exits.
        DispatchQueue.main.asyncAfter(deadline: .now() + ServerSupervisor.defaultStopGracePeriod + 1) {
            exit(0)
        }
    }
    installSignal(SIGINT, handleSignal)
    installSignal(SIGTERM, handleSignal)

    do {
        let url = try controller.boot()
        print("LMWarden headless: \(url.absoluteString)")
        // Flush now: stdout to a pipe or file is block-buffered, and a
        // supervisor of this process reads the URL line while it runs.
        fflush(stdout)
    } catch {
        FileHandle.standardError.write(Data("LMWarden headless: boot failed: \(error)\n".utf8))
        exit(1)
    }

    // The supervisor gives up after too many consecutive short exits; nothing
    // would restart it, so exit non-zero instead of idling forever.
    controller.supervisor?.onStateChange = { state in
        if case .failed(let failures) = state, !stopping {
            FileHandle.standardError.write(
                Data("LMWarden headless: server failed after \(failures) consecutive exits\n".utf8)
            )
            exit(1)
        }
    }

    dispatchMain()
}

/// Install a `DispatchSource` signal handler for `sig` on the main queue and
/// keep the source alive. The handler runs on the main queue, so it
/// serialises with the boot and the stop's grace-period closure.
private func installSignal(_ sig: Int32, _ handler: @escaping () -> Void) {
    let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    source.setEventHandler { handler() }
    source.resume()
    headlessSignalSources.append(source)
}
