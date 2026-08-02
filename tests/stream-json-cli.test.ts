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
  extractCursorAgentDelta,
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

test("TextAssembler.reconcileFinal ignores a trailing full-text recap instead of duplicating it", () => {
  // Reproduces a real cursor-agent --stream-partial-output capture: true
  // incremental chunks (fed via feed()), then one final canonical event
  // restating everything from the start (that event lacks timestamp_ms,
  // which is why extractCursorAgentDelta routes it to reconcileFinal
  // instead of feed() — see the dispatch test below).
  const a = new TextAssembler();
  const chunks = [
    "Un pod de Kubernetes es la unidad mínima que el clúster realmente m",
    "ueve: uno o más contenedores que viajan juntos, comparten red y almacenamiento, y viven (",
    "y mueren) como un solo equipo. Si un contenedor es un m",
    "úsico, el pod es la banda: no despliegas al baterista solo y",
    " esperas que suene bien.",
  ];
  let assembled = "";
  for (const c of chunks) {
    assembled += a.feed(c);
  }
  assert.equal(assembled, chunks.join(""), "incremental chunks must assemble cleanly");

  const recap = chunks.join(""); // the exact same full text, resent from scratch
  assert.equal(a.reconcileFinal(recap), "", "the recap must not be re-emitted as a new delta");
  assert.equal(a.text, chunks.join(""), "assembled text must stay as-is, not duplicated");
});

test("TextAssembler.reconcileFinal emits only the genuine new tail when the final event extends what was streamed", () => {
  const a = new TextAssembler();
  a.feed("Parte uno.");
  assert.equal(a.reconcileFinal("Parte uno. Parte dos."), " Parte dos.");
  assert.equal(a.text, "Parte uno. Parte dos.");
});

test("TextAssembler.reconcileFinal trusts a diverging final text without re-emitting mismatched content", () => {
  const a = new TextAssembler();
  a.feed("Borrador inicial");
  // The canonical final text doesn't extend what was streamed at all —
  // trust it for accounting, but don't try to patch up what already went out.
  assert.equal(a.reconcileFinal("Texto final completamente distinto"), "");
  assert.equal(a.text, "Texto final completamente distinto");
});

test("extractCursorAgentDelta routes events with timestamp_ms through feed(), others through reconcileFinal", () => {
  const a = new TextAssembler();
  const mkEvt = (text: string, withTimestamp: boolean) => ({
    type: "assistant",
    message: { role: "assistant", content: [{ type: "text", text }] },
    ...(withTimestamp ? { timestamp_ms: "1785635558972" } : {}),
  });

  assert.equal(extractCursorAgentDelta(mkEvt("Hola", true), a), "Hola");
  assert.equal(extractCursorAgentDelta(mkEvt("Hola mundo", true), a), " mundo");
  // Final canonical event (no timestamp_ms) recaps the same text — must not duplicate.
  assert.equal(extractCursorAgentDelta(mkEvt("Hola mundo", false), a), "");
  assert.equal(a.text, "Hola mundo");
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
