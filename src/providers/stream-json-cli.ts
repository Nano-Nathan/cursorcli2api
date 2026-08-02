/**
 * Runs CLI subprocesses and yields NDJSON events from stdout.
 * Ported from Python stream_json_cli.py.
 */

import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import type { Readable } from 'node:stream';

/**
 * Normalize message content to plain text (OpenAI-style parts).
 */
function normalizeMessageContent(content: unknown): string {
  if (content == null) return '';
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const part of content) {
      if (part && typeof part === 'object' && 'type' in part && part.type === 'text' && typeof part.text === 'string') {
        parts.push(part.text);
      }
    }
    return parts.join('');
  }
  if (typeof content === 'object' && 'type' in content && (content as { type: string }).type === 'text' && 'text' in content && typeof (content as { text: string }).text === 'string') {
    return (content as { text: string }).text;
  }
  return String(content);
}

/**
 * Handles mixed partial/full text streams.
 * Turns mixed streams into clean deltas (and a final assembled text).
 */
export class TextAssembler {
  text = '';

  feed(incoming: string): string {
    const s = incoming ?? '';
    if (!s) return '';
    if (s === this.text) return '';
    if (s.startsWith(this.text)) {
      const delta = s.slice(this.text.length);
      this.text = s;
      return delta;
    }
    // Genuinely new content that doesn't extend what we have.
    this.text += s;
    return s;
  }

  /**
   * Reconciles a canonical, complete message against whatever has been fed
   * so far via feed(). cursor-agent's --stream-partial-output sends true
   * incremental deltas throughout (each carrying a `timestamp_ms` field),
   * then one final "assistant" event that restates the *entire* response
   * from the start rather than extending it (that final event lacks
   * `timestamp_ms` — see extractCursorAgentDelta). Treating that final
   * event as just another feed() call duplicates the opening of the reply;
   * this instead trusts it as ground truth and only emits the part that
   * hasn't already gone out.
   */
  reconcileFinal(full: string): string {
    const s = full ?? '';
    if (!s) return '';
    if (s === this.text) return '';
    if (s.startsWith(this.text)) {
      const tail = s.slice(this.text.length);
      this.text = s;
      return tail;
    }
    // Diverges from what was already streamed — trust the canonical text
    // for accounting, but don't try to patch up what the client already saw.
    this.text = s;
    return '';
  }
}

export interface IterStreamJsonEventsOptions {
  cmd: string[];
  env?: Record<string, string>;
  /** Per-line idle timeout (resets on each line). */
  timeoutMs?: number;
  /** Absolute wall-clock timeout for the entire subprocess. */
  totalTimeoutMs?: number;
  /** Kill the subprocess after seeing a "result" event (for CLIs that don't exit on their own). */
  killOnResult?: boolean;
  /** Data to write to the subprocess stdin before closing it. */
  stdinData?: string | null;
  /** AbortSignal to cancel the subprocess (e.g. client disconnect). */
  signal?: AbortSignal;
  eventCallback?: (evt: Record<string, unknown>) => void;
  stderrCallback?: (line: string) => void;
}

/**
 * Async generator that spawns a subprocess, reads stdout line-by-line as NDJSON,
 * drains stderr, and yields parsed JSON events.
 */
export async function* iterStreamJsonEvents(
  opts: IterStreamJsonEventsOptions
): AsyncGenerator<Record<string, unknown>> {
  const {
    cmd,
    env,
    timeoutMs = 60_000,
    totalTimeoutMs,
    killOnResult = false,
    stdinData = null,
    signal,
    eventCallback,
    stderrCallback,
  } = opts;

  const proc = spawn(cmd[0], cmd.slice(1), {
    stdio: [stdinData != null ? 'pipe' : 'ignore', 'pipe', 'pipe'],
    env: { ...process.env, ...env },
  });

  if (stdinData != null && proc.stdin) {
    proc.stdin.write(stdinData);
    proc.stdin.end();
  }

  // AbortSignal: check immediately-aborted, defer listener until rl is created
  if (signal) {
    if (signal.aborted) {
      proc.kill("SIGKILL");
      throw new Error("aborted");
    }
  }

  const stderrBuf: Buffer[] = [];
  let lastHint: string | null = null;
  let totalTimedOut = false;
  const totalTimer = totalTimeoutMs
    ? setTimeout(() => {
        totalTimedOut = true;
        proc.kill("SIGKILL");
      }, totalTimeoutMs)
    : null;

  const drainStderr = (): Promise<void> =>
    new Promise((resolve) => {
      if (!proc.stderr) {
        resolve();
        return;
      }
      let textBuf = '';
      proc.stderr.on('data', (chunk: Buffer) => {
        stderrBuf.push(chunk);
        if (stderrCallback) {
          textBuf += chunk.toString('utf8');
          const lines = textBuf.split('\n');
          if (!textBuf.endsWith('\n')) {
            textBuf = lines.pop() ?? '';
          } else {
            textBuf = '';
          }
          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed) stderrCallback(trimmed);
          }
        }
      });
      proc.stderr.on('end', () => {
        if (stderrCallback && textBuf.trim()) {
          for (const line of textBuf.split('\n')) {
            const trimmed = line.trim();
            if (trimmed) stderrCallback(trimmed);
          }
        }
        resolve();
      });
    });

  const drainPromise = drainStderr();

  let rl: ReturnType<typeof createInterface> | null = null;

  try {
    if (!proc.stdout) {
      throw new Error('subprocess stdout not available');
    }

    rl = createInterface({
      input: proc.stdout as Readable,
      crlfDelay: Infinity,
    });

    // Register abort listener now that rl is assigned
    if (signal && !signal.aborted) {
      const onAbort = () => {
        rl?.close();
        proc.kill("SIGKILL");
      };
      signal.addEventListener("abort", onAbort, { once: true });
    }

    const iface = rl; // non-null: assigned above, only reached inside try block

    const nextLine = (): Promise<string | null> =>
      new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          iface.removeListener('line', onLine);
          iface.removeListener('close', onClose);
          proc.kill("SIGKILL");
          reject(new Error(`subprocess timeout after ${timeoutMs}ms`));
        }, timeoutMs);

        const onLine = (line: string) => {
          clearTimeout(timer);
          iface.removeListener('line', onLine);
          iface.removeListener('close', onClose);
          resolve(line);
        };

        const onClose = () => {
          clearTimeout(timer);
          iface.removeListener('line', onLine);
          iface.removeListener('close', onClose);
          resolve(null);
        };

        iface.once('line', onLine);
        iface.once('close', onClose);
      });

    while (true) {
      if (totalTimedOut) {
        await drainPromise;
        throw new Error(`subprocess total timeout after ${totalTimeoutMs}ms`);
      }

      let line: string | null;
      try {
        line = await nextLine();
      } catch (err) {
        await drainPromise;
        throw err;
      }

      if (line === null) break;

      const raw = line.trim();
      if (!raw) continue;

      let evt: Record<string, unknown>;
      try {
        evt = JSON.parse(raw) as Record<string, unknown>;
      } catch {
        continue;
      }

      if (evt.type === 'result' && typeof evt.result === 'string' && evt.result) {
        lastHint = String(evt.result).trim() || lastHint;
      }
      if (evt.type === 'error' && typeof evt.message === 'string' && evt.message) {
        lastHint = String(evt.message).trim() || lastHint;
      }

      if (eventCallback) eventCallback(evt);
      yield evt;

      if (killOnResult && evt.type === 'result') {
        proc.kill("SIGKILL");
        return;
      }
    }

    await drainPromise;

    await new Promise<void>((resolve, reject) => {
      const finish = (code: number | null) => {
        if (code !== 0) {
          const msg = Buffer.concat(stderrBuf).toString('utf8').trim();
          const exitInfo = code != null ? `${code}` : `signal ${proc.signalCode ?? "unknown"}`;
          reject(new Error(msg || lastHint || `subprocess failed: ${exitInfo}`));
        } else {
          resolve();
        }
      };
      if (proc.exitCode != null) {
        finish(proc.exitCode);
      } else {
        proc.once('exit', finish);
      }
    });
  } finally {
    if (totalTimer) clearTimeout(totalTimer);
    rl?.close();
    // exitCode is null for signal-killed processes, so also check signalCode
    if (proc.exitCode === null && proc.signalCode === null) {
      proc.kill("SIGKILL");
      await new Promise<void>((r) => proc.once("exit", () => r()));
    }
  }
}

/**
 * Extract text delta from cursor-agent events.
 *
 * cursor-agent tags every true partial-output "assistant" event with a
 * `timestamp_ms` field; the one final "assistant" event that restates the
 * complete response from scratch omits it. That presence/absence is a
 * structural signal (not a text-similarity guess) for which event is a
 * live delta to append vs. the canonical final message to reconcile against.
 */
export function extractCursorAgentDelta(
  evt: Record<string, unknown>,
  assembler: TextAssembler
): string {
  if (evt.type !== 'assistant') return '';
  const message = evt.message;
  if (!message || typeof message !== 'object') return '';
  const content = (message as Record<string, unknown>).content;
  const incoming = normalizeMessageContent(content);
  const isPartialDelta = typeof evt.timestamp_ms !== 'undefined';
  return isPartialDelta ? assembler.feed(incoming) : assembler.reconcileFinal(incoming);
}

/**
 * Extract live "thinking" text from cursor-agent's --stream-partial-output
 * events (type "thinking", subtype "delta"). The subtype "completed" event
 * carries no text and just marks the end of the thinking phase. Routed
 * through a TextAssembler (like assistant deltas) because long thinking
 * traces have been observed to repeat/garble fragments when concatenated
 * blindly; feed() is a no-op for genuinely new, non-overlapping chunks so
 * this is safe even if that turns out not to be needed for a given turn.
 */
export function extractCursorAgentReasoningDelta(
  evt: Record<string, unknown>,
  assembler: TextAssembler
): string {
  if (evt.type !== 'thinking') return '';
  if (evt.subtype !== 'delta') return '';
  const text = evt.text;
  if (typeof text !== 'string') return '';
  return assembler.feed(text);
}

/**
 * Extract text delta from claude events.
 */
export function extractClaudeDelta(
  evt: Record<string, unknown>,
  assembler: TextAssembler
): string {
  if (evt.type !== 'assistant') return '';
  const message = evt.message;
  if (!message || typeof message !== 'object') return '';
  const content = (message as Record<string, unknown>).content;
  const incoming = normalizeMessageContent(content);
  return assembler.feed(incoming);
}

/**
 * Extract text delta from gemini events.
 */
export function extractGeminiDelta(
  evt: Record<string, unknown>,
  assembler: TextAssembler
): string {
  if (evt.type !== 'message') return '';
  if (evt.role !== 'assistant') return '';
  const content = evt.content;
  const incoming = normalizeMessageContent(content);
  return assembler.feed(incoming);
}

/**
 * Extract usage from claude result events.
 */
export function extractUsageFromClaudeResult(
  evt: Record<string, unknown>
): { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null {
  if (evt.type !== 'result') return null;
  const usage = evt.usage;
  if (!usage || typeof usage !== 'object') return null;
  const u = usage as Record<string, unknown>;
  const inTokens = Math.floor(Number(u.input_tokens) || 0);
  const outTokens = Math.floor(Number(u.output_tokens) || 0);
  return {
    prompt_tokens: inTokens,
    completion_tokens: outTokens,
    total_tokens: inTokens + outTokens,
  };
}

/**
 * Extract usage from gemini result events.
 */
export function extractUsageFromGeminiResult(
  evt: Record<string, unknown>
): { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null {
  if (evt.type !== 'result') return null;
  const stats = evt.stats;
  if (!stats || typeof stats !== 'object') return null;
  const s = stats as Record<string, unknown>;
  const inTokens = Math.floor(Number(s.input_tokens) || 0);
  const outTokens = Math.floor(Number(s.output_tokens) || 0);
  const total = Math.floor(Number(s.total_tokens) || inTokens + outTokens);
  return {
    prompt_tokens: inTokens,
    completion_tokens: outTokens,
    total_tokens: total,
  };
}
