# Desktop interface

Sift uses one shared interface across macOS, Windows, and Linux. The product
should feel like a research application: calm, legible, and explicit about
what is happening. Its restrained console influence appears in technical text
and state indicators, not in decorative terminal effects.

This document records the interface contract. It is a reference for changes to
`src/sift/web/`, the desktop bridge, and native packaging.

## Design principles

- Use the operating system's normal window frame and file dialogs.
- Keep one session list, one primary workspace, and one prompt composer.
- Use system UI type for reading and navigation. Reserve monospace type for
  paths, code, fields, counts, model names, audit records, and shortcuts.
- Use a muted green accent for active and verified states, always paired with
  text or shape.
- Prefer quiet borders and modest corner radii to heavy shadows or decoration.
- Follow the system light or dark preference, with an explicit in-app choice.
- Do not use faux prompts, scanlines, glow, animated grids, ASCII branding, or
  other “hacker” styling.

## Trust language

The interface distinguishes local processing from model disclosure.

**Raw data stays local** is accurate for Sift's guarded analysis path.
**Nothing leaves this device** is not accurate: the researcher can allow
messages, schema, sanitized summaries, and approved figures to reach a chosen
model provider.

The persistent status label is **Local workspace**. Its supporting text
explains that raw datasets are processed locally and only information allowed
by the active permission level is sent to the selected model. The dataset
profile, result inspector, and disclosure ledger provide the detailed record.

Credential copy remains platform-neutral. It says that secrets are stored in
the operating system's protected credential store, with Keychain, Credential
Manager, and Secret Service named where platform-specific help is useful.

## Main workflows

### Connect a model

The first-run view presents supported providers as a simple list. Each row has
configuration state, the required credential or endpoint fields, save and
remove actions, and a short billing explanation.

The primary continue action remains unavailable until at least one provider is
usable. Keyboard focus begins at the first incomplete field. Provider names
remain visible as text even when a logo is present.

### Choose data

The landing view provides:

- one clear drop target for supported files;
- equally visible **Choose files** and **Choose folder** actions;
- **Try Sift with sample data** for a safe first session;
- recent sessions in a conventional list;
- access to provider management and the data-source catalog.

Database, warehouse, cloud, and research-service connectors explain what
credential and permission they need before opening a connection. The user
selects the object or read-only result; the model is not given account-browsing
access.

### Work in a session

The research workspace has four stable regions:

1. A resizable session sidebar.
2. A top bar with application identity, workspace path, local-workspace status,
   and infrequent workspace actions.
3. A readable transcript containing messages, plans, code, results, warnings,
   and verification.
4. A composer containing attachments, permission level, model selection,
   context usage, and the send action.

Detailed or occasional material opens in focused inspectors rather than
crowding the transcript. These inspectors include the dataset profile,
evidence, disclosure ledger, exports, checkpoints, settings, and keyboard
shortcuts.

## Content rules

- Plans identify pending, active, completed, and blocked work in text.
- A result is called verified only when the corresponding deterministic check
  ran.
- Warnings and missing diagnostics appear before interpretive prose.
- Generated code remains selectable and copyable.
- Wide tables and code blocks scroll within their own labelled region.
- Raw local output and model-visible sanitized output are not presented as the
  same object.
- Errors state what failed, whether local data was affected, and the next safe
  action. They do not expose credentials, connection strings, raw records, or
  unrestricted paths.

## Window sizes and reflow

The native window opens at 1180 × 780 and supports a practical minimum of
880 × 600.

| Effective width | Interface behavior |
| --- | --- |
| 1200 px and above | Full session sidebar, full toolbar labels, centred transcript |
| 900–1199 px | Narrower sidebar and tighter toolbar spacing |
| Below 900 px | Collapsed session rail, essential toolbar labels, near-full-width dialogs |

Ordinary content reflows without page-level horizontal scrolling. Data tables
and code may scroll inside their own containers. The same rules apply when
browser zoom or an accessibility tool reduces the effective viewport.

## Platform behavior

The HTML, CSS, JavaScript, and Python bridge remain shared. Native integration
is platform specific.

| Platform | Web engine and window | Platform details |
| --- | --- | --- |
| macOS | WKWebView in a normal AppKit window | San Francisco system stack, SF Mono or Menlo for technical text, Command-key shortcuts, signed and notarized app bundle |
| Windows | WebView2 in a normal Win32 window | Segoe UI stack, Consolas for technical text, Ctrl-key shortcuts, per-user installer and portable archive |
| Linux | Qt WebEngine in a conventional Qt window | Desktop font stack with Noto Sans fallback, system monospace, Ctrl-key shortcuts, application-menu entry from the per-user installer |

The Windows beta is currently unsigned; that packaging status does not change
the interface contract.

## Accessibility

- Every action is keyboard reachable in a logical order.
- Focus remains visible with a two-CSS-pixel indicator and sufficient contrast
  against adjacent colours.
- Interactive targets are at least 28 × 28 CSS pixels; primary controls are
  generally 32–36 pixels high.
- Normal text meets WCAG AA 4.5:1 contrast and large text meets 3:1.
- Active, warning, error, verified, and busy states never rely on colour alone.
- Reduced-motion mode removes non-essential animation and smooth scrolling.
- Forced-colours and high-contrast modes preserve borders, focus, and control
  identity.
- Dialogs expose names, descriptions, close actions, and focus return.
- The transcript, code, and tables remain selectable.
- A skip link moves keyboard users directly to the research workspace.

## Security and performance

- The interface uses the in-process pywebview bridge. It does not open a
  localhost API server.
- Scripts, styles, fonts, and icons are bundled locally. No remote asset is
  required to render the application.
- Production debug mode is off and webview storage is private.
- No analytics SDK, telemetry library, frontend framework, or remote font is
  loaded.
- Credentials, raw records, connection strings, and private paths never enter
  browser storage, DOM data attributes, CSS, or model-visible diagnostics.
- Asset names are cache-busted during packaging, and all platforms use the
  same generated asset inventory.

## Changing the interface

An interface change is complete when:

1. existing bridge methods and privacy wording remain accurate;
2. keyboard navigation, visible focus, light and dark themes, reduced motion,
   and high-contrast behavior have been checked;
3. the layout has been exercised at wide, standard, compact, and zoomed sizes;
4. no remote request occurs before a user explicitly invokes a provider,
   connector, update check, or external link;
5. cross-platform asset-bundling and desktop contract tests pass;
6. the locally available native application launches with the packaged assets.

Native window, renderer, installer, and confinement behavior must still be
qualified on the target operating system.
