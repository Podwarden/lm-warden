"use client";

import { useMemo } from "react";
import { AnsiUp } from "ansi_up";

interface AnsiLogProps {
  text: string;
  className?: string;
}

export function AnsiLog({ text, className }: AnsiLogProps) {
  const html = useMemo(() => {
    const ansi = new AnsiUp();
    return ansi.ansi_to_html(text);
  }, [text]);

  return (
    <pre
      className={className}
      dangerouslySetInnerHTML={{ __html: html }}
      style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}
    />
  );
}
