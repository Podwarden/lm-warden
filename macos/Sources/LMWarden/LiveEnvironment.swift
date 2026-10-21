// LiveEnvironment — the production wiring of the AppEnvironment composition
// root (Metal design s2/s4, non-UI).
//
// `AppEnvironment.live` is the single place that picks the real dependencies:
// the system Keychain, a loopback PortSelector, `SystemProcessLauncher` with a
// `HealthProbe` on the port the spec passes to uvicorn, and `livePrepare`
// (AppPaths.prepare + the models-dir markers). Everything is injectable so a
// test can observe the wiring without touching the real Keychain, binding a
// real port, or launching a real process.
import Foundation

extension AppEnvironment {
    /// The production environment: system Keychain, loopback port probe,
    /// `livePrepare`, and a `ServerSupervisor` backed by
    /// `SystemProcessLauncher` + `HealthProbe` on the spec's port.
    ///
    /// `processInfo` is injected so a test can assert the parent PID and base
    /// environment come from the real process without forking. `schedule`
    /// defaults to the main queue; a test injects a recording closure so the
    /// supervisor's first health poll is observable without a run loop.
    static func live(
        pythonPath: String,
        applicationSupport: String = AppPaths.defaultApplicationSupport(),
        processInfo: ProcessInfo = .processInfo,
        excludeModelsFromBackup: Bool = true,
        webuiDir: String? = nil,
        schedule: @escaping (TimeInterval, @escaping () -> Void) -> Void = AppEnvironment.mainQueueSchedule
    ) -> AppEnvironment {
        AppEnvironment(
            pythonPath: pythonPath,
            applicationSupport: applicationSupport,
            parentPID: processInfo.processIdentifier,
            baseEnvironment: processInfo.environment,
            store: KeychainStore(backend: SystemKeychainBackend()),
            ports: PortSelector(),
            prepare: { paths in try AppEnvironment.livePrepare(paths, excludeFromBackup: excludeModelsFromBackup) },
            makeSupervisor: { spec in
                ServerSupervisor(
                    spec: spec,
                    launcher: SystemProcessLauncher(),
                    probe: HealthProbe(port: port(of: spec) ?? 0).check,
                    schedule: schedule
                )
            },
            webuiDir: webuiDir
        )
    }

    /// Schedules `work` on the main queue after `delay` seconds. The
    /// production `ServerSupervisor` calls this for every health poll and
    /// backoff; the caller (the app) serialises those on the main queue.
    static func mainQueueSchedule(_ delay: TimeInterval, _ work: @escaping () -> Void) {
        DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: work)
    }

    /// The uvicorn port the spec was built with: the integer following the
    /// first "--port" argument. Nil when the flag is absent, is the last
    /// argument, or its value is not an integer.
    static func port(of spec: ServerLaunchSpec) -> Int? {
        let args = spec.arguments
        guard let flagIndex = args.firstIndex(of: "--port") else {
            return nil
        }
        let valueIndex = flagIndex + 1
        guard valueIndex < args.count else {
            return nil
        }
        return Int(args[valueIndex])
    }

    /// The one-argument convenience wrapper: the models dir is excluded from
    /// Time Machine backups (today's behaviour), so existing callers are
    /// unchanged.
    static func livePrepare(_ paths: AppPaths) throws {
        try livePrepare(paths, excludeFromBackup: true)
    }

    /// Creates the data and models directories (errors propagate), then drops
    /// the models-dir markers. Marker failures are best-effort: a models dir on
    /// a network/exFAT volume is allowed and may refuse the backup key, so a
    /// marker failure never fails boot. The `exclude` closure is called once
    /// with the models dir and `excludeFromBackup`: true sets the Time Machine
    /// exclusion, false clears one set on an earlier launch, so the setting
    /// always matches what Time Machine does.
    static func livePrepare(
        _ paths: AppPaths,
        excludeFromBackup: Bool,
        touch: (String) throws -> Void = ModelDirMarkers.systemTouch,
        exclude: (String, Bool) throws -> Void = AppEnvironment.systemSetExcludedFromBackup
    ) throws {
        try AppPaths.prepare(paths)
        try? ModelDirMarkers.apply(
            modelsDir: paths.modelsDir,
            touch: touch,
            excludeFromBackup: { path in try exclude(path, excludeFromBackup) }
        )
    }

    /// Sets NSURLIsExcludedFromBackupKey on the directory URL to `excluded`:
    /// true makes Time Machine skip it, false includes it again.
    static func systemSetExcludedFromBackup(_ path: String, _ excluded: Bool) throws {
        var url = URL(fileURLWithPath: path)
        var values = URLResourceValues()
        values.isExcludedFromBackup = excluded
        try url.setResourceValues(values)
    }
}
