/* Slash command palette — resolves a composer message like ``/analyze``
 * into either a rewritten chat message (sent through the normal
 * composer submit path, same as clicking a starter-prompt chip) or a
 * UI action to run instead of sending anything.
 *
 * Deliberately DOM-free and self-contained (mirrors markdown.js's
 * architecture) so command RESOLUTION is unit-testable through node
 * without a browser — see tests/test_slash_commands.py. app.js wires
 * the 'ui' outcomes to the actual panel-opening functions, since
 * those depend on the DOM this file intentionally knows nothing about.
 */
(function () {
  // Commands that resolve to a chat message. Each is a real message
  // sent through the normal composer path — no special "run this
  // playbook" codepath that could drift from what a researcher typing
  // the same words would get, matching the existing Analyze-button /
  // starter-prompt convention already documented in app.js.
  const CHAT_COMMANDS = {
    analyze: (
      'Analyze this dataset end to end: profile it, find what stands '
      + 'out, verify and stress-test the findings that matter, and '
      + 'tell me the things most worth knowing.'
    ),
    verify: (
      'Show the deterministic verification results (confidence, '
      + 'causality, and any warnings) for the most recent finding, in '
      + 'plain language.'
    ),
    challenge: (
      'Challenge the most recent finding: run alternative '
      + 'specifications and report plainly whether it holds up.'
    ),
    chart: 'Create a chart that best shows the most recent finding.',
    connect: 'Help me connect a database or external data source to this session.',
  };

  // Commands that open an existing panel instead of sending a
  // message. app.js maps each name to the real DOM-touching function
  // (openDataPanel / openExport / openLedger) — this module only
  // ever returns the NAME, never calls anything itself.
  const UI_COMMAND_NAMES = new Set(['profile', 'report', 'privacy']);

  const ALL_COMMAND_NAMES = Object.freeze(
    Object.keys(CHAT_COMMANDS).concat(Array.from(UI_COMMAND_NAMES)).sort()
  );

  /* Parse ``rawText`` (the trimmed composer value) as a slash command.
   *
   * Returns:
   *   - ``null`` when ``rawText`` isn't a recognised slash command —
   *     the caller should send it as an ordinary message unchanged.
   *   - ``{kind: 'chat', text}`` — the message to actually send. Any
   *     text typed after the command name is appended to the
   *     canned phrase (so ``/chart the top 5 regions`` still reads
   *     as a normal, specific request).
   *   - ``{kind: 'ui', name}`` — a panel-opening action to run
   *     instead of sending anything.
   */
  function resolveSlashCommand(rawText) {
    if (typeof rawText !== 'string') return null;
    const m = rawText.match(/^\/(\w+)(?:\s+([\s\S]*))?$/);
    if (!m) return null;
    const name = m[1].toLowerCase();
    const rest = (m[2] || '').trim();

    if (Object.prototype.hasOwnProperty.call(CHAT_COMMANDS, name)) {
      const base = CHAT_COMMANDS[name];
      return { kind: 'chat', text: rest ? base + '\n\n' + rest : base };
    }
    if (UI_COMMAND_NAMES.has(name)) {
      return { kind: 'ui', name: name };
    }
    return null;
  }

  window.SiftSlashCommands = {
    resolveSlashCommand: resolveSlashCommand,
    ALL_COMMAND_NAMES: ALL_COMMAND_NAMES,
  };
})();
