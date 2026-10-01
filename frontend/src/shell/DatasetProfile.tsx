/** Stat strip for the dataset profile. App passes only the stats that exist
 *  in PharmState — no dash placeholders, no invented units. */
export function DatasetProfile({ items }: { items: { label: string; value: string }[] }) {
  return (
    <dl className="stat-strip">
      {items.map(it => (
        <div key={it.label} className="stat">
          <dt>{it.label}</dt>
          <dd>{it.value}</dd>
        </div>
      ))}
    </dl>
  );
}
