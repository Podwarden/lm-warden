"use client";

import { forwardRef } from "react";
import { cn } from "@/lib/utils";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "secondary" | "outline" | "ghost" | "destructive";
  size?: "sm" | "md" | "lg";
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { className, variant = "default", size = "md", disabled, ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900 disabled:pointer-events-none disabled:opacity-50",
        {
          "bg-emerald-600 text-white hover:bg-emerald-500": variant === "default",
          "bg-slate-700 text-slate-100 hover:bg-slate-600": variant === "secondary",
          "border border-slate-600 bg-transparent hover:bg-slate-800": variant === "outline",
          "hover:bg-slate-800 hover:text-slate-100": variant === "ghost",
          "bg-red-600 text-white hover:bg-red-500": variant === "destructive",
        },
        {
          "h-8 px-3 text-xs": size === "sm",
          "h-9 px-4 text-sm": size === "md",
          "h-10 px-6 text-sm": size === "lg",
        },
        className
      )}
      disabled={disabled}
      {...props}
    />
  );
});

Button.displayName = "Button";
