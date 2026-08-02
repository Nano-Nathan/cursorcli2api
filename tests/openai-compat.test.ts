import test from "node:test";
import assert from "node:assert/strict";

import {
  compatChatRequestToChatRequest,
  normalizeToolingRequest,
  parseToolCallResponse,
  responsesRequestToChatRequest,
  type ChatCompletionRequest,
} from "../src/lib/openai-compat.js";

test("normalizeToolingRequest mirrors top-level tool fields into model_extra", () => {
  const request = normalizeToolingRequest({
    model: "claude:sonnet",
    stream: false,
    max_tokens: null,
    tools: [
      {
        type: "function",
        function: {
          name: "get_weather",
          description: "Get weather",
          parameters: { type: "object", properties: { city: { type: "string" } } },
        },
      },
    ],
    tool_choice: "auto",
    parallel_tool_calls: false,
    messages: [
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "get_weather", arguments: "{\"city\":\"Beijing\"}" },
          },
        ],
      },
      {
        role: "tool",
        tool_call_id: "call_1",
        content: "sunny",
      },
    ],
  } as ChatCompletionRequest & Record<string, unknown>);

  const extra = (request as Record<string, unknown>).model_extra as Record<string, unknown>;
  assert.deepEqual(extra.tools, (request as Record<string, unknown>).tools);
  assert.equal(extra.tool_choice, "auto");
  assert.equal(extra.parallel_tool_calls, false);

  const assistantExtra = ((request.messages[0] as Record<string, unknown>).model_extra ?? {}) as Record<string, unknown>;
  const toolExtra = ((request.messages[1] as Record<string, unknown>).model_extra ?? {}) as Record<string, unknown>;
  assert.deepEqual(assistantExtra.tool_calls, (request.messages[0] as Record<string, unknown>).tool_calls);
  assert.equal(toolExtra.tool_call_id, "call_1");
});

test("normalizeToolingRequest preserves existing model_extra precedence", () => {
  const topLevelTools = [{ type: "function", function: { name: "wrong" } }];
  const modelExtraTools = [{ type: "function", function: { name: "right" } }];

  const request = normalizeToolingRequest({
    model: "codex:auto",
    stream: false,
    max_tokens: null,
    tools: topLevelTools,
    messages: [
      {
        role: "assistant",
        content: "",
        tool_calls: [{ id: "call_1", type: "function", function: { name: "wrong", arguments: "{}" } }],
        model_extra: {
          tool_calls: [{ id: "call_1", type: "function", function: { name: "right", arguments: "{}" } }],
        },
      },
    ],
    model_extra: {
      tools: modelExtraTools,
      tool_choice: "required",
    },
  } as ChatCompletionRequest & Record<string, unknown>);

  const extra = (request as Record<string, unknown>).model_extra as Record<string, unknown>;
  assert.deepEqual(extra.tools, modelExtraTools);
  assert.equal(extra.tool_choice, "required");

  const assistantExtra = ((request.messages[0] as Record<string, unknown>).model_extra ?? {}) as Record<string, unknown>;
  assert.deepEqual(assistantExtra.tool_calls, [{ id: "call_1", type: "function", function: { name: "right", arguments: "{}" } }]);
});

test("compat and responses request conversion keep top-level tool fields available in model_extra", () => {
  const compatRequest = compatChatRequestToChatRequest({
    model: "claude:sonnet",
    stream: false,
    messages: [{ role: "user", content: "hello" }],
    tools: [{ type: "function", function: { name: "lookup" } }],
    tool_choice: { type: "function", function: { name: "lookup" } },
  } as ChatCompletionRequest & Record<string, unknown>);

  const compatExtra = (compatRequest as Record<string, unknown>).model_extra as Record<string, unknown>;
  assert.deepEqual(compatExtra.tools, [{ type: "function", function: { name: "lookup" } }]);
  assert.deepEqual(compatExtra.tool_choice, { type: "function", function: { name: "lookup" } });

  const responsesRequest = responsesRequestToChatRequest({
    model: "gemini:flash",
    input: [{ role: "user", content: [{ type: "input_text", text: "hi" }] }],
    stream: false,
    tools: [{ type: "function", function: { name: "lookup" } }],
    parallel_tool_calls: true,
  } as Record<string, unknown>);

  const responsesExtra = (responsesRequest as Record<string, unknown>).model_extra as Record<string, unknown>;
  assert.deepEqual(responsesExtra.tools, [{ type: "function", function: { name: "lookup" } }]);
  assert.equal(responsesExtra.parallel_tool_calls, true);
});

test("parseToolCallResponse repairs an unescaped literal newline left inside a JSON string value", () => {
  const marker = "___TOOL_CALL___";
  // Simulates the real failure: a long multi-line shell/heredoc command where
  // the model forgot to escape a line break as \n inside the JSON string.
  const badJson = `{"name": "exec", "arguments": {"command": "echo start\nprint(1)\nprint(2)"}}`;
  const raw = `Voy a ejecutar esto.\n${marker}\n${badJson}\n${marker}`;

  const { text, toolCalls } = parseToolCallResponse(raw);

  assert.ok(toolCalls, "expected the malformed block to be repaired and parsed");
  assert.equal(toolCalls?.length, 1);
  assert.equal(toolCalls?.[0].function.name, "exec");
  const args = JSON.parse(toolCalls![0].function.arguments) as { command: string };
  assert.equal(args.command, "echo start\nprint(1)\nprint(2)");
  assert.ok(!text.includes(marker), "marker should not leak into the remaining/visible text");
});

test("parseToolCallResponse still returns null (unmodified text) for genuinely invalid JSON", () => {
  const marker = "___TOOL_CALL___";
  const raw = `${marker}\n{"name": "exec", "arguments": {oops this is not json}}\n${marker}`;

  const { text, toolCalls } = parseToolCallResponse(raw);

  assert.equal(toolCalls, null);
  assert.equal(text, raw);
});

test("parseToolCallResponse repairs a lone backslash left un-doubled inside a regex argument", () => {
  const marker = "___TOOL_CALL___";
  // Simulates the real failure: a grep/sed regex like `sessions\.reset`
  // where the model forgot to double the backslash for JSON (\\.).
  const badJson = `{"name":"exec","arguments":{"command":"grep -n -E 'sessions\\.(reset|new)' file.md"}}`;
  const raw = `Voy a revisar la doc.\n${marker}\n${badJson}\n${marker}`;

  const { text, toolCalls } = parseToolCallResponse(raw);

  assert.ok(toolCalls, "expected the malformed block to be repaired and parsed");
  assert.equal(toolCalls?.length, 1);
  assert.equal(toolCalls?.[0].function.name, "exec");
  const args = JSON.parse(toolCalls![0].function.arguments) as { command: string };
  assert.equal(args.command, "grep -n -E 'sessions\\.(reset|new)' file.md");
  assert.ok(!text.includes(marker), "marker should not leak into the remaining/visible text");
});

test("parseToolCallResponse leaves already-valid escaped backslashes untouched", () => {
  const marker = "___TOOL_CALL___";
  const goodJson = `{"name":"exec","arguments":{"command":"grep -n -E 'sessions\\\\.(reset|new)' file.md"}}`;
  const raw = `${marker}\n${goodJson}\n${marker}`;

  const { toolCalls } = parseToolCallResponse(raw);

  assert.ok(toolCalls);
  const args = JSON.parse(toolCalls![0].function.arguments) as { command: string };
  assert.equal(args.command, "grep -n -E 'sessions\\.(reset|new)' file.md");
});

test("parseToolCallResponse drops an immediate duplicate of the same tool call", () => {
  const marker = "___TOOL_CALL___";
  // Observed live: Cursor Agent's stream sometimes re-emits the exact same
  // marker block twice back-to-back for one logical action.
  const call = `{"name":"exec","arguments":{"command":"echo hi"}}`;
  const raw = `${marker}\n${call}\n${marker}${marker}\n${call}\n${marker}`;

  const { toolCalls } = parseToolCallResponse(raw);

  assert.ok(toolCalls);
  assert.equal(toolCalls?.length, 1, "the immediate duplicate should be dropped");
});

test("parseToolCallResponse keeps distinct (non-duplicate) consecutive tool calls", () => {
  const marker = "___TOOL_CALL___";
  const callA = `{"name":"exec","arguments":{"command":"echo a"}}`;
  const callB = `{"name":"exec","arguments":{"command":"echo b"}}`;
  const raw = `${marker}\n${callA}\n${marker}${marker}\n${callB}\n${marker}`;

  const { toolCalls } = parseToolCallResponse(raw);

  assert.ok(toolCalls);
  assert.equal(toolCalls?.length, 2, "distinct calls must not be deduplicated");
  assert.equal(JSON.parse(toolCalls![0].function.arguments).command, "echo a");
  assert.equal(JSON.parse(toolCalls![1].function.arguments).command, "echo b");
});
