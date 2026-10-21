// SystemSignals — turn macOS thermal state and power-source changes into
// ShellEvents for NotificationPolicy.
//
// The Python server has no thermal or power signal, so the shell reads them
// itself. `SystemSignals` is the pure mapping (unit-tested);
// `SystemSignalObserver` is the thin glue around
// ProcessInfo.thermalStateDidChangeNotification and the IOKit power-source
// run-loop source (not unit-tested: it needs real system notifications).
//
// The observer forwards every reading; NotificationPolicy does the edge
// detection, hysteresis and rate limiting. start() also posts one snapshot of
// the current thermal and power state so the policy starts from a real
// baseline rather than its defaults.
import Foundation
import IOKit.ps

enum SystemSignals {
    /// Map ProcessInfo's thermal state one to one; unknown future cases are
    /// treated as nominal.
    static func thermal(_ s: ProcessInfo.ThermalState) -> ShellThermal {
        switch s {
        case .nominal: return .nominal
        case .fair: return .fair
        case .serious: return .serious
        case .critical: return .critical
        @unknown default: return .nominal
        }
    }

    /// The exact value IOPSGetProvidingPowerSourceType returns on battery
    /// (kIOPMBatteryPowerKey == kIOPSBatteryPowerValue == "Battery Power").
    static let batteryPowerValue = "Battery Power"

    /// True only for exactly "Battery Power"; "AC Power", "UPS Power", nil and
    /// anything else are not battery.
    static func onBattery(powerSourceState: String?) -> Bool {
        powerSourceState == batteryPowerValue
    }

    /// Read the providing power source type from IOKit. nil when no snapshot
    /// is available (e.g. a desktop Mac reporting nothing).
    static func currentPowerSourceType() -> String? {
        // Copy rule: IOPSCopyPowerSourcesInfo returns a +1 reference.
        guard let snapshot = IOPSCopyPowerSourcesInfo()?.takeRetainedValue() else {
            return nil
        }
        // Get rule: the returned string is owned by the snapshot (+0).
        guard let type = IOPSGetProvidingPowerSourceType(snapshot)?.takeUnretainedValue() else {
            return nil
        }
        return type as String
    }
}

/// Observes thermal and power-source changes and posts ShellEvents on the
/// main queue. Call start() and stop() on the main thread; both are
/// idempotent, and deinit stops, so the IOKit callback can never outlive the
/// observer.
final class SystemSignalObserver {
    private let onEvent: (ShellEvent) -> Void
    private let modelLoaded: () -> Bool
    private var thermalToken: NSObjectProtocol?
    private var powerSource: CFRunLoopSource?

    init(onEvent: @escaping (ShellEvent) -> Void, modelLoaded: @escaping () -> Bool) {
        self.onEvent = onEvent
        self.modelLoaded = modelLoaded
    }

    deinit {
        stop()
    }

    func start() {
        guard thermalToken == nil, powerSource == nil else { return }

        thermalToken = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.postThermal()
        }

        // The context is unretained: the observer owns the source and removes
        // and invalidates it in stop()/deinit before it can go away.
        let context = Unmanaged.passUnretained(self).toOpaque()
        // Create rule: IOPSNotificationCreateRunLoopSource returns a +1
        // reference, which ARC now owns via `powerSource`.
        if let source = IOPSNotificationCreateRunLoopSource(systemSignalPowerCallback, context)?
            .takeRetainedValue() {
            CFRunLoopAddSource(CFRunLoopGetMain(), source, .defaultMode)
            powerSource = source
        }

        postThermal()
        postPower()
    }

    func stop() {
        if let token = thermalToken {
            NotificationCenter.default.removeObserver(token)
            thermalToken = nil
        }
        if let source = powerSource {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .defaultMode)
            CFRunLoopSourceInvalidate(source)
            powerSource = nil // drops the +1 reference taken in start()
        }
    }

    private func postThermal() {
        let level = SystemSignals.thermal(ProcessInfo.processInfo.thermalState)
        deliver(.thermal(level))
    }

    fileprivate func postPower() {
        let onBattery = SystemSignals.onBattery(powerSourceState: SystemSignals.currentPowerSourceType())
        deliver(.power(onBattery: onBattery, modelLoaded: modelLoaded()))
    }

    private func deliver(_ event: ShellEvent) {
        if Thread.isMainThread {
            onEvent(event)
        } else {
            DispatchQueue.main.async { [weak self] in self?.onEvent(event) }
        }
    }
}

/// C callback for the IOKit power-source run-loop source. It runs on the main
/// run loop, where the source is installed; the observer is alive because
/// stop() removes the source before the observer is released.
private func systemSignalPowerCallback(_ context: UnsafeMutableRawPointer?) {
    guard let context else { return }
    let observer = Unmanaged<SystemSignalObserver>.fromOpaque(context).takeUnretainedValue()
    observer.postPower()
}
