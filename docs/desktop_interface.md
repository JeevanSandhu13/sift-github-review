# Sift desktop interface architecture

## Product direction

Sift should feel like a serious research instrument, not a themed terminal.
The reference image is useful for its clarity, dense information hierarchy,
visible state, restrained colour, and keyboard-friendly character. The desktop
application will borrow those qualities without copying the ASCII logo, neon
green treatment, box-drawing decoration, or permanently dense dashboard.

The interface is a conventional desktop application with a light
research-console influence:

- native operating-system window chrome and native file/folder dialogs;
- one familiar session sidebar, one primary work area, and one composer;
- system UI typography for reading and navigation;
- monospace typography only for paths, code, data fields, counts, model names,
  audit records, and keyboard shortcuts;
- a muted green accent for active/verified states, never as the only state cue;
- square-to-softly-rounded panels, quiet one-pixel borders, and minimal shadow;
- system light/dark preference with an explicit in-app override;
- no faux command prompt, scanlines, glow, animated grids, or decorative sci-fi.

## Product truth and trust language

The UI must distinguish local computation from model disclosure precisely.
“Raw data stays local” is appropriate for Sift's guarded analysis path. It must
not be expanded into “nothing leaves this device,” because a researcher can
permit disclosure-controlled summaries, schema, prompts, and selected
attachments to reach their chosen model provider.

The persistent shell will use the compact label **Local workspace**. Its help
text will explain: raw datasets are processed locally; only the information
allowed by the active permission tier is sent to the selected model. The data
profile and privacy ledger remain the detailed verification surfaces.

Credential wording must be platform-neutral: secrets are stored in the
operating system's protected credential store (Keychain on macOS, Credential
Manager on Windows, and Secret Service/keyring on supported Linux desktops).

## Information architecture

### 1. First launch and provider setup

- A compact Sift wordmark and one-sentence product promise.
- Provider rows for Anthropic, OpenAI, Google, and OpenAI-compatible endpoints.
- Each row shows configuration state, key input, save/remove actions, and a
  plain-language billing note.
- The continue action remains disabled until at least one provider is usable.
- Keyboard focus starts at the first incomplete credential field.
- No provider logo is required for comprehension; provider names remain text.

### 2. Workspace selection

- Primary drop target for supported files.
- Equal access to native **Choose files** and **Choose folder** actions.
- A synthetic sample-data path for safe evaluation.
- Recent sessions presented as a conventional list, not a dashboard grid.
- Provider management remains reachable without leaving the application.

### 3. Research workspace

- Left: resizable/collapsible session navigation.
- Top: application identity, working-directory breadcrumb, local-workspace
  assurance, and infrequent workspace tools.
- Centre: readable transcript and analysis outputs, capped to a comfortable
  line length while tables and code keep their own horizontal scroll areas.
- Bottom: prompt composer, attachment control, data-permission control, model
  selector, and context usage.
- Modal inspectors: dataset profile, privacy ledger, evidence, exports,
  checkpoints, feedback, and shortcuts. These remain separate because they
  contain detailed or infrequent work and should not crowd the main workspace.

## Responsive desktop behaviour

Sift is a desktop application, but it must remain usable when tiled, zoomed, or
run on a small laptop.

- **Wide (1200 px and above):** full session sidebar, full tool labels, centred
  research transcript.
- **Standard (900–1199 px):** narrower sidebar and tighter toolbar spacing;
  secondary labels can truncate with accessible names intact.
- **Compact (below 900 px):** the session rail collapses; toolbar controls wrap
  or reduce to essential labels; dialogs use nearly the full viewport.
- **Reflow/zoom:** ordinary content reflows without two-dimensional scrolling.
  Real data tables and code blocks may scroll inside their own labelled region.
- The native window starts at 1180 × 780 and supports a practical minimum of
  880 × 600. The content CSS continues to reflow below that effective width for
  browser zoom and accessibility tools.

## Cross-platform application contract

The same tested HTML/CSS/JavaScript interface runs inside pywebview, while the
window, web engine, dialogs, credential storage, packaging, signing, and runtime
sandbox remain native to each platform.

### macOS

- WKWebView in a normal titled, resizable AppKit window.
- System font stack headed by San Francisco; SF Mono/Menlo for technical text.
- Command-key labels in the shortcuts panel.
- `.app` bundle, application icon, hardened runtime, notarization, and DMG flow
  continue through the existing release pipeline.

### Windows

- WebView2/Edge Chromium in a normal resizable Win32 window.
- Segoe UI Variable/Segoe UI shell typography and Consolas for technical text.
- Ctrl-key labels in the shortcuts panel.
- One-directory signed application and archive continue through the existing
  Windows release pipeline. WebView2 availability remains part of qualification.

### Linux

- Qt WebEngine inside a conventional Qt window; no custom client-side titlebar.
- Desktop/system UI font, then Noto Sans; system monospace for technical text.
- Ctrl-key labels and Linux desktop theme/high-contrast support.
- The existing PyInstaller archive remains portable across the documented
  distribution baseline; application-menu/desktop-file packaging is a later
  release-delivery layer, not a second UI implementation.

## Accessibility contract

- Every action is keyboard reachable in a logical order.
- Focus is always visible with at least a two-CSS-pixel indicator and sufficient
  contrast against adjacent colours.
- Interactive targets are at least 28 × 28 CSS pixels; primary controls are
  generally 32–36 pixels high.
- Normal text targets WCAG AA 4.5:1 contrast; large text targets 3:1.
- Active, warning, error, verified, and busy states use text or shape in addition
  to colour.
- Reduced-motion disables non-essential animation and smooth scrolling.
- Forced-colours/high-contrast mode retains borders, focus, and control identity.
- Dialogs expose dialog semantics, descriptive names, close actions, and focus
  return. The existing keyboard escape handling is retained and strengthened.
- The transcript stays selectable. Generated code and data tables remain
  copyable and independently scrollable.

## Security and performance contract

- Keep the JS-to-Python bridge; do not add a localhost REST API or a remote asset
  dependency.
- Keep normal native window chrome; no custom draggable-titlebar code.
- Run the webview in private storage mode explicitly and production debug mode
  off.
- Bundle every font/script/style locally and preserve cache-busted asset loading.
- Add no frontend framework, icon package, analytics SDK, or telemetry.
- Use CSS custom properties and a small platform-detection script only; the new
  visual layer should not add meaningful startup or runtime cost.
- Never put credentials, raw records, connection strings, or private paths into
  browser storage, data attributes, CSS, or diagnostics.

## Maintenance workflow

1. Correct platform-specific and overbroad privacy copy.
2. Add semantic landmarks, a skip link, accurate local-workspace assurance, and
   platform-aware shortcut labels.
3. Add a final, isolated desktop-shell stylesheet that replaces the visual
   system while leaving the mature behaviour layer intact.
4. Set explicit native window dimensions, minimum size, background, selectable
   text, zoom support, and private webview storage.
5. Add automated desktop-GUI contract tests for semantics, platform labels,
   accessibility modes, responsive breakpoints, native-window options, and
   cross-platform asset bundling.
6. Run the full relevant Python and static contract suite.
7. Render the real interface with a local bridge fixture at wide, standard, and
   compact sizes; inspect light, dark, reduced-motion, and keyboard focus states.
8. Fix every issue found, repeat the tests and visual pass, and record the native
   qualification boundary: this machine can validate macOS directly, while
   signed Windows/Linux binaries still require their corresponding build hosts.

## Acceptance criteria

- Existing Sift workflows and bridge method contracts remain unchanged.
- The interface looks like one calm desktop product on all three platforms.
- Terminal influence is visible in technical text and state treatment, not in
  decoration.
- No clipped primary action at 880 × 600 or equivalent zoomed viewport.
- All primary workflows are keyboard operable with visible focus.
- Light, dark, high-contrast, and reduced-motion modes remain legible.
- No new network request occurs before the researcher explicitly invokes a
  provider or external link.
- New assets are included automatically by the cross-platform PyInstaller spec.
- Automated tests pass and the locally available native application launches.

## Research basis

This plan reconciles Apple Human Interface Guidelines for layout, sidebars, and
typography; Microsoft guidance for keyboard interaction, navigation, typography,
and accessibility; GNOME guidance for accessibility, adaptive layout, styling,
and typography; WCAG 2.2 guidance for contrast, focus appearance, target size,
and reflow; and pywebview's documented native-window, bridge, private-mode, and
security architecture.
