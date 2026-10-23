// Reads a newline-delimited JSON stream, calling onMessage for each object as soon as its line is complete.
// A line may span several chunks or share one; the last line may lack a trailing newline.
// Return true from onMessage to stop reading early.
export async function readNdjson(
  body: ReadableStream<Uint8Array>,
  onMessage: (data: Record<string, unknown>) => boolean | void,
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const lines = buffer.split('\n');
      buffer = done ? '' : lines.pop() ?? '';

      for (const line of lines) {
        if (!line.trim()) continue;
        let data: Record<string, unknown>;
        try {
          data = JSON.parse(line);
        } catch {
          console.warn('Skipping malformed NDJSON line:', line);
          continue;
        }
        if (onMessage(data)) return;
      }
      if (done) return;
    }
  } finally {
    reader.cancel().catch(() => {}); // frees the connection when we stop before the stream ends
  }
}
