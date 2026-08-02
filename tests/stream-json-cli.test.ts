/**
 * Tests for stream-json-cli.ts — subprocess lifecycle management.
 *
 * Uses real child_process.spawn (echo + node -e scripts) to produce
 * controllable NDJSON output without mocking.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  iterStreamJsonEvents,
  TextAssembler,
  classifyCursorAgentAssistantEvent,
  extractCursorAgentReasoningDelta,
} from "../src/providers/stream-json-cli.js";

test("iterStreamJsonEvents yields parsed NDJSON events from subprocess stdout", async () => {
  const events: Record<string, unknown>[] = [];
  // Use setTimeout to separate writes into different ticks, avoiding
  // the readline once('line') race when both lines arrive in one chunk.
  const lines = [
    `process.stdout.write('{"type":"test","n":1}\\n');`,
    `setTimeout(() => { process.stdout.write('{"type":"test","n":2}\\n'); }, 10);`,
  ].join("");
  for await (const evt of iterStreamJsonEvents({
    cmd: ["node", "-e", lines],
    timeoutMs: 5000,
  })) {
    events.push(evt);
  }
  assert.equal(events.length, 2);
  assert.equal(events[0].type, "test");
  assert.equal(events[0].n, 1);
  assert.equal(events[1].type, "test");
  assert.equal(events[1].n, 2);
});

test("iterStreamJsonEvents handles killOnResult", async () => {
  const events: Record<string, unknown>[] = [];
  const script = [
    `process.stdout.write('{"type":"assistant","message":{"content":"hello"}}\\n');`,
    `setTimeout(() => { process.stdout.write('{"type":"result","result":"done"}\\n'); }, 10);`,
    // Would output after result but process should be killed before this
    `setTimeout(() => process.stdout.write('{"type":"after","x":1}\\n'), 15000);`,
  ].join(" ");
  for await (const evt of iterStreamJsonEvents({
    cmd: ["node", "-e", script],
    timeoutMs: 5000,
    killOnResult: true,
  })) {
    events.push(evt);
  }
  assert.equal(events.length, 2);
  assert.equal(events[0].type, "assistant");
  assert.equal(events[1].type, "result");
});

test("iterStreamJsonEvents propagates subprocess non-zero exit as error", async () => {
  try {
    for await (const _evt of iterStreamJsonEvents({
      cmd: ["node", "-e", `process.stderr.write('something broke'); process.exit(1);`],
      timeoutMs: 5000,
    })) {
      // should throw before yielding anything useful
    }
    assert.fail("Expected error was not thrown");
  } catch (e) {
    const msg = String(e);
    assert.ok(
      msg.includes("something broke") || msg.includes("subprocess failed") || msg.includes("1"),
      `error message should indicate subprocess failure, got: ${msg}`,
    );
  }
});

test("iterStreamJsonEvents with AbortSignal kills subprocess early", async () => {
  const ac = new AbortController();
  const events: Record<string, unknown>[] = [];

  const script = [
    `process.stdout.write('{"type":"start"}\\n');`,
    // Sleep 30 seconds — abort should kill before this
    `setTimeout(() => process.stdout.write('{"type":"never"}\\n'), 30000);`,
  ].join(" ");

  const iter = iterStreamJsonEvents({
    cmd: ["node", "-e", script],
    timeoutMs: 3000,
    signal: ac.signal,
  });

  // Abort after a short delay
  setTimeout(() => ac.abort(), 50);

  try {
    for await (const evt of iter) {
      events.push(evt);
    }
  } catch (_) {
    // Expected — generator should throw after abort kills the process
  }

  const types = events.map((e) => e.type);
  assert.ok(types.includes("start"), "should have received start event");
  assert.ok(!types.includes("never"), "should NOT have received never event");
});

test("TextAssembler produces clean deltas from partial/full text", () => {
  const a = new TextAssembler();
  assert.equal(a.feed("Hello"), "Hello");
  assert.equal(a.text, "Hello");
  assert.equal(a.feed("Hello World"), " World");
  assert.equal(a.text, "Hello World");
  assert.equal(a.feed("New"), "New");
  assert.equal(a.text, "Hello WorldNew");
  assert.equal(a.feed(""), "");
  assert.equal(a.text, "Hello WorldNew");
});

test("TextAssembler.feed never drops a short chunk that coincidentally starts like the accumulated text", () => {
  // feed() is only for genuine incremental deltas (each carries its own
  // timestamp_ms upstream) — it must never guess "this looks like a recap"
  // from text shape alone. A short unrelated chunk that happens to start
  // with the same characters as the (much longer) accumulated text is
  // common in natural language ("de", "la", "el", ...) and must be
  // appended in full, not dropped.
  const a = new TextAssembler();
  a.feed("tu compañero de laboratorio con delirios de grandeza técnica y sarcasmo fino");
  assert.equal(a.feed("tuañero"), "tuañero");
  assert.equal(
    a.text,
    "tu compañero de laboratorio con delirios de grandeza técnica y sarcasmo finotuañero"
  );
});

test("classifyCursorAgentAssistantEvent marks events with timestamp_ms as partial, others as final", () => {
  const mkEvt = (text: string, withTimestamp: boolean) => ({
    type: "assistant",
    message: { role: "assistant", content: [{ type: "text", text }] },
    ...(withTimestamp ? { timestamp_ms: "1785635558972" } : {}),
  });

  // True partial-output deltas each carry timestamp_ms — narration, not the answer.
  assert.deepEqual(classifyCursorAgentAssistantEvent(mkEvt("Un pod de Kuber", true)), {
    text: "Un pod de Kuber",
    isFinal: false,
  });
  // The one final canonical event lacks timestamp_ms — this is the real answer,
  // taken verbatim, with no comparison against what was streamed as narration.
  assert.deepEqual(
    classifyCursorAgentAssistantEvent(mkEvt("Un pod de Kubernetes es la unidad mínima.", false)),
    { text: "Un pod de Kubernetes es la unidad mínima.", isFinal: true }
  );
  assert.equal(classifyCursorAgentAssistantEvent({ type: "thinking", text: "no" }), null);
  assert.equal(classifyCursorAgentAssistantEvent({ type: "assistant", message: null }), null);
});

test("extractCursorAgentReasoningDelta extracts thinking deltas and ignores the completed marker", () => {
  const a = new TextAssembler();
  assert.equal(
    extractCursorAgentReasoningDelta({ type: "thinking", subtype: "delta", text: "Pensando..." }, a),
    "Pensando..."
  );
  assert.equal(extractCursorAgentReasoningDelta({ type: "thinking", subtype: "completed" }, a), "");
  assert.equal(extractCursorAgentReasoningDelta({ type: "assistant", text: "no" }, a), "");
});

test("extractCursorAgentReasoningDelta dedupes a repeated thinking chunk instead of doubling it", () => {
  // Regression: long thinking traces were observed to repeat/garble
  // fragments when concatenated blindly (no assembler at all).
  const a = new TextAssembler();
  assert.equal(
    extractCursorAgentReasoningDelta({ type: "thinking", subtype: "delta", text: "Comprobando permisos" }, a),
    "Comprobando permisos"
  );
  assert.equal(
    extractCursorAgentReasoningDelta({ type: "thinking", subtype: "delta", text: "Comprobando permisos" }, a),
    "",
    "an exact repeat of the same chunk must not be re-emitted"
  );
  assert.equal(a.text, "Comprobando permisos");
});
