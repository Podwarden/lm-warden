// Wizard step <-> route mapping.
//
// Server steps (app/setup/state_machine.py) use snake_case ("hf_token");
// the Next.js route segments use kebab-case ("hf-token"). Keep this map in
// sync with STEPS in app/setup/state_machine.py.
export const STEP_TO_PATH: Record<string, string> = {
  welcome: '/setup/welcome',
  gpus: '/setup/gpus',
  hf_token: '/setup/hf-token',
  admin: '/setup/admin',
  done: '/setup/done',
};

// Unknown/malformed step falls back to the wizard entry point.
export function pathForStep(step: unknown): string {
  return (typeof step === 'string' && STEP_TO_PATH[step]) || '/setup/welcome';
}
