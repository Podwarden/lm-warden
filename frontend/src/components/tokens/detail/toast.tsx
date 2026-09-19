"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

export interface ToastApi {
  message: string;
  visible: boolean;
  show: (msg: string) => void;
}

/** The mockup's toast: one message at a time, 2200 ms. */
export function useToast(): ToastApi {
  const [message, setMessage] = useState("");
  const [visible, setVisible] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const show = useCallback((msg: string) => {
    setMessage(msg);
    setVisible(true);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setVisible(false), 2200);
  }, []);
  useEffect(() => () => clearTimeout(timer.current), []);
  return { message, visible, show };
}

export function Toast({ api, bottomPx = 20 }: { api: ToastApi; bottomPx?: number }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className={cn(
        "pointer-events-none fixed left-1/2 z-30 -translate-x-1/2 rounded-md border border-chat-rule bg-chat-surface-2 px-4 py-[9px] text-[13px] text-chat-fg transition-[opacity,transform] duration-[180ms] ease-[ease] motion-reduce:transition-none",
        api.visible ? "translate-y-0 opacity-100" : "translate-y-5 opacity-0",
      )}
      style={{ bottom: bottomPx }}
    >
      {api.message}
    </div>
  );
}
