// Pure data->pixel scale + tick helpers, extracted from the SpaghettiChart
// idiom in App.tsx so the flexplot renderer stays a thin, testable mapper.
// The server precomputes all geometry; these only place it on a pixel canvas.

export type Scale = (v: number) => number;

/** Linear scale mapping [d0, d1] onto pixel range [p0, p1]. */
export function makeLinear(d0: number, d1: number, p0: number, p1: number): Scale {
  const span = d1 - d0 || 1;
  return (v: number) => p0 + ((v - d0) / span) * (p1 - p0);
}

/**
 * Log10 scale over [d0, d1] onto [p0, p1]. Non-positive inputs clamp to the
 * bottom of the axis (concentrations are strictly positive on a log plot).
 */
export function makeLog(d0: number, d1: number, p0: number, p1: number): Scale {
  const lo = Math.log10(Math.max(d0, 1e-12));
  const hi = Math.log10(Math.max(d1, d0 * 1.0001, 1e-12));
  const span = hi - lo || 1;
  return (v: number) => (v > 0 ? p0 + ((Math.log10(v) - lo) / span) * (p1 - p0) : p0);
}

/** Even linear ticks across [d0, d1] (default 5). */
export function linearTicks(d0: number, d1: number, count = 5): number[] {
  const step = (d1 - d0) / (count - 1);
  return Array.from({ length: count }, (_, i) => d0 + i * step);
}

/** Decade ticks (…, 1, 10, 100, …) within [d0, d1] for a log axis. */
export function decadeTicks(d0: number, d1: number): number[] {
  const lo = Math.floor(Math.log10(Math.max(d0, 1e-12)));
  const hi = Math.ceil(Math.log10(Math.max(d1, 1e-12)));
  const out: number[] = [];
  for (let k = lo; k <= hi; k++) {
    const v = Math.pow(10, k);
    if (v >= d0 * 0.5 && v <= d1 * 2) out.push(v);
  }
  return out;
}

/** Compact numeric tick label matching the SpaghettiChart convention. */
export function fmtTick(v: number): string {
  const a = Math.abs(v);
  if (a >= 1000) return `${(v / 1000).toFixed(a >= 10000 ? 0 : 1)}k`;
  if (a >= 1) return String(Math.round(v * 10) / 10);
  if (a === 0) return '0';
  return v.toPrecision(2);
}
