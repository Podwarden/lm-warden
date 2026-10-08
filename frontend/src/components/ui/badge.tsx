import { cn } from "@/lib/utils";

export function Badge({
  className,
  variant = "default",
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & {
  variant?: "default" | "success" | "warning" | "error" | "info";
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium",
        {
          "bg-slate-700 text-slate-200": variant === "default",
          "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-300": variant === "success",
          "bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-300": variant === "warning",
          "bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300": variant === "error",
          "bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300": variant === "info",
        },
        className
      )}
      {...props}
    />
  );
}
