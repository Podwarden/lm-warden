-- Add GPU model name to gpu_samples so the stats UI can label gauges.
-- New column is nullable; the sampler populates it on every write, but rows
-- written by pre-0013 builds will stay NULL and the API simply returns null
-- (the UI is expected to fall back to "GPU N").
ALTER TABLE gpu_samples ADD COLUMN name TEXT;
