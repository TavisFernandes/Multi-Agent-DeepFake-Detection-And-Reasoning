// Run: node src/lib/ndjson.test.ts   (Node 22.18+ runs TypeScript directly)
import assert from 'node:assert/strict';
import { readNdjson } from './ndjson.ts';

function streamOf(chunks: string[]) {
  const enc = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(c) {
      for (const s of chunks) c.enqueue(enc.encode(s));
      c.close();
    },
  });
}

async function collect(chunks: string[], stopOn?: string) {
  const seen: unknown[] = [];
  await readNdjson(streamOf(chunks), (d) => {
    seen.push(d.type);
    return d.type === stopOn;
  });
  return seen;
}

// One line split across three chunks, two lines in one chunk, last line without a newline.
assert.deepEqual(
  await collect(['{"type":"agent_', 'result","n":1', '}\n{"type":"final_verdict"}\n{"ty', 'pe":"done"}']),
  ['agent_result', 'final_verdict', 'done'],
);
// Blank lines and malformed lines are skipped, not fatal.
assert.deepEqual(await collect(['\n{"type":"a"}\n\nnot json\n{"type":"b"}\n']), ['a', 'b']);
// A multi-byte character split across chunks decodes correctly.
const bytes = new TextEncoder().encode('{"type":"é"}\n');
const split = new ReadableStream<Uint8Array>({
  start(c) { c.enqueue(bytes.slice(0, 10)); c.enqueue(bytes.slice(10)); c.close(); },
});
const got: unknown[] = [];
await readNdjson(split, (d) => { got.push(d.type); });
assert.deepEqual(got, ['é']);
// Returning true stops early and ignores the rest.
assert.deepEqual(await collect(['{"type":"done"}\n{"type":"late"}\n'], 'done'), ['done']);

console.log('ndjson: all checks passed');
