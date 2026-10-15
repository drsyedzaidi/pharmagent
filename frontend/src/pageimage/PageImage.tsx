import './pageimage.css';
import { bboxCssPercent, bboxInFrame } from './pageImageLoader.ts';
import type { NormBBox, PageImageLocator, PageImageState } from './types.ts';
import { usePageImage } from './usePageImage.ts';

/** Pure view over a load state. Exported for tests and for hosts that already
 *  hold the state. `bbox` is in page-normalized coordinates; when the image was
 *  fetched with a `crop`, pass the same `crop` so the overlay lines up. */
export function PageImageView({ state, bbox, crop, alt }: {
  state: PageImageState;
  bbox?: NormBBox;
  crop?: NormBBox;
  alt: string;
}) {
  switch (state.status) {
    case 'empty':
      return <div className="page-image page-image--note" role="status">No page selected</div>;
    case 'loading':
      return <div className="page-image page-image--note" role="status" aria-busy="true">Loading page…</div>;
    case 'error':
      return <div className="page-image page-image--note page-image--error" role="alert">{state.message}</div>;
    case 'ready': {
      const box = bboxInFrame(bbox, crop);
      return (
        <div className="page-image">
          <img className="page-image__img" src={state.src} alt={alt} draggable={false} />
          {box && (
            <div
              className="page-image__bbox"
              data-testid="page-image-bbox"
              role="img"
              aria-label="Highlighted region"
              style={bboxCssPercent(box)}
            />
          )}
        </div>
      );
    }
  }
}

/** Rendered page of a stored source PDF, with an optional highlighted region. */
export function PageImage({ locator, bbox, crop, scale, alt = 'Source document page' }: {
  locator: PageImageLocator | null;
  bbox?: NormBBox;
  crop?: NormBBox;
  scale?: number;
  alt?: string;
}) {
  const state = usePageImage(locator, { scale, crop });
  return <PageImageView state={state} bbox={bbox} crop={crop} alt={alt} />;
}
