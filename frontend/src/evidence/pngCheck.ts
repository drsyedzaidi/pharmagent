/** A page image body must be a PNG that the browser can decode. A gateway page, an SPA fallback or a
 *  truncated file would otherwise load as a blank box with an outline drawn on it. React-free. */
const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

/** Throws unless *blob* starts with the PNG signature and (where the browser can) decodes. */
export async function assertPageImage(blob: Blob): Promise<void> {
  const head = new Uint8Array(await blob.slice(0, PNG_SIGNATURE.length).arrayBuffer());
  if (head.length < PNG_SIGNATURE.length || !PNG_SIGNATURE.every((b, i) => head[i] === b)) {
    throw new Error('the server did not return a PNG image');
  }
  if (typeof createImageBitmap !== 'function') return;
  try {
    const bitmap = await createImageBitmap(blob);
    bitmap.close();
  } catch {
    throw new Error('the page image could not be decoded');
  }
}
