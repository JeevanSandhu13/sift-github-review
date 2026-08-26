/* Sift web UI — frontend JS.
 *
 * Communicates with the Python backend via pywebview's bridge:
 *   window.pywebview.api.<method>(...)    — JS → Python
 *   Python calls `window.sift_event({type, ...payload})` via
 *     webview.evaluate_js to push events back to us.
 *
 * This file deliberately avoids a framework. The chat state is the
 * DOM; each event appends a message node to #messages. Swap for a
 * real framework later if we want components and state management.
 */

const landingEl = document.getElementById('landing');
const chatEl = document.getElementById('chat');
const authEl = document.getElementById('auth');
const authStatusEl = document.getElementById('auth-status');
const authContinueBtn = document.getElementById('auth-continue-btn');
const authContinueHint = document.getElementById('auth-continue-hint');
const dropZone = document.getElementById('drop-zone');
const chooseFilesBtn = document.getElementById('choose-files-btn');
const chooseFolderBtn = document.getElementById('choose-folder-btn');
const landingStatus = document.getElementById('landing-status');
const messagesEl = document.getElementById('messages');
const form = document.getElementById('compose-form');
const input = document.getElementById('compose-input');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const cwdEl = document.getElementById('cwd-display');
const contextChip = document.getElementById('context-chip');
const doctorBannerEl = document.getElementById('doctor-banner');
const doctorBannerTextEl = document.getElementById('doctor-banner-text');
const doctorBannerDismissBtn = document.getElementById('doctor-banner-dismiss');
const skipLinkEl = document.getElementById('skip-link');
const updatesOverlay = document.getElementById('updates-overlay');
const updatesCloseBtn = document.getElementById('updates-close');
const updatesCheckBtn = document.getElementById('updates-check');
const updatesDownloadBtn = document.getElementById('updates-download');
const updatesVersionEl = document.getElementById('updates-version');
const updatesChannelEl = document.getElementById('updates-channel');
const updatesStatusEl = document.getElementById('updates-status');
const updatesPathEl = document.getElementById('updates-path');

// Human-facing native integration labels. desktop-platform.js normalizes the
// host renderer before this file loads; the values affect copy only, never a
// security or execution decision.
const desktopPlatform = document.documentElement.dataset.platform || 'unknown';
const nativeFileManager = desktopPlatform === 'macos'
  ? 'Finder'
  : desktopPlatform === 'windows'
    ? 'File Explorer'
    : 'your file manager';
const protectedCredentialStore = desktopPlatform === 'macos'
  ? 'Keychain'
  : desktopPlatform === 'windows'
    ? 'Credential Manager'
    : 'credential store';

function setSkipTarget(target, label) {
  if (!skipLinkEl) return;
  skipLinkEl.href = target;
  skipLinkEl.textContent = label;
}

let updatesReturnFocus = null;

function renderUpdateResult(result) {
  const safe = result || {};
  updatesStatusEl.textContent = safe.reason || '';
  updatesStatusEl.classList.toggle('error', safe.ok === false);
  updatesPathEl.classList.add('hidden');
  updatesPathEl.textContent = '';
  updatesDownloadBtn.classList.add('hidden');
  if (safe.status === 'current') {
    updatesStatusEl.textContent = `Sift ${safe.version} is current.`;
  } else if (safe.status === 'available') {
    updatesStatusEl.textContent = `Sift ${safe.version} is available. The installer has not been downloaded.`;
    updatesDownloadBtn.classList.remove('hidden');
  } else if (safe.status === 'ready') {
    updatesStatusEl.textContent = `Sift ${safe.version} was downloaded and cryptographically verified. Quit Sift before running the native installer.`;
    updatesPathEl.textContent = `Installer: ${safe.installer}`;
    updatesPathEl.classList.remove('hidden');
  }
}

async function openUpdates(event) {
  updatesReturnFocus = event && event.currentTarget ? event.currentTarget : document.activeElement;
  updatesOverlay.classList.remove('hidden');
  updatesStatusEl.textContent = 'Loading local update configuration…';
  updatesPathEl.classList.add('hidden');
  updatesDownloadBtn.classList.add('hidden');
  updatesCheckBtn.disabled = true;
  updatesCloseBtn.focus();
  try {
    const api = window.pywebview && window.pywebview.api;
    if (!api || typeof api.update_configuration !== 'function') {
      renderUpdateResult({ok: false, reason: 'Restart Sift to use the update panel.'});
      return;
    }
    const config = await api.update_configuration();
    updatesVersionEl.textContent = config.installed_version || 'Unknown';
    updatesChannelEl.textContent = config.channel || 'Not configured';
    updatesCheckBtn.disabled = config.configured !== true;
    renderUpdateResult({
      ok: config.ok && config.configured,
      reason: config.configured ? 'No network request has been made.' : config.reason,
    });
  } catch (err) {
    renderUpdateResult({ok: false, reason: `Update configuration could not be read: ${err && err.message ? err.message : err}`});
  }
}

function closeUpdates() {
  updatesOverlay.classList.add('hidden');
  if (updatesReturnFocus && typeof updatesReturnFocus.focus === 'function') updatesReturnFocus.focus();
}

async function runUpdateAction(download) {
  updatesCheckBtn.disabled = true;
  updatesDownloadBtn.disabled = true;
  updatesStatusEl.textContent = download
    ? 'Downloading and verifying the native installer and SBOM…'
    : 'Checking the signed release channel…';
  try {
    const result = await window.pywebview.api.check_for_updates(download);
    renderUpdateResult(result);
  } catch (err) {
    renderUpdateResult({ok: false, reason: `The update check was blocked: ${err && err.message ? err.message : err}`});
  } finally {
    updatesCheckBtn.disabled = false;
    updatesDownloadBtn.disabled = false;
  }
}

document.querySelectorAll('[data-open-updates]').forEach((button) => {
  button.addEventListener('click', openUpdates);
});
if (updatesCloseBtn) updatesCloseBtn.addEventListener('click', closeUpdates);
if (updatesCheckBtn) updatesCheckBtn.addEventListener('click', () => runUpdateAction(false));
if (updatesDownloadBtn) updatesDownloadBtn.addEventListener('click', () => runUpdateAction(true));
if (updatesOverlay) updatesOverlay.addEventListener('click', (event) => {
  if (event.target === updatesOverlay) closeUpdates();
});

// Topbar pill — click to reveal the session folder in Finder. The
// pill always shows the abbreviated cwd path (formatCwd) so the
// researcher can see at a glance WHERE on disk Sift is writing
// scripts, logs, generated data, and plots; clicking opens that
// folder so they can inspect outputs in the OS file manager
// directly. Renaming a session lives only on the sidebar's pencil
// button — the topbar used to double as a rename trigger, but
// because it shared display state with the auto-derived title,
// renames flowed one way (topbar→sidebar) and not the other
// (sidebar→topbar), which was confusing. Splitting the two
// concerns — sidebar = name, topbar = path — fixes the asymmetry.
if (cwdEl) {
  cwdEl.title = `Open this session's folder in ${nativeFileManager}`;
  cwdEl.setAttribute('role', 'button');
  cwdEl.setAttribute('tabindex', '0');
  const openSessionFolder = async () => {
    if (!currentCwd) return;
    if (!window.pywebview || !window.pywebview.api) return;
    if (typeof window.pywebview.api.open_path !== 'function') {
      toast('Restart Sift to enable folder reveal.', 'info');
      return;
    }
    try {
      const res = await window.pywebview.api.open_path(currentCwd);
      if (!res || !res.ok) {
        const reason = (res && res.reason) || 'unknown';
        toast('Could not open folder: ' + reason, 'error');
      }
    } catch (err) {
      console.warn('open_path failed', err);
      toast(
        'Could not open folder: '
        + (err && err.message ? err.message : err),
        'error',
      );
    }
  };
  cwdEl.addEventListener('click', openSessionFolder);
  cwdEl.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    e.preventDefault();
    openSessionFolder();
  });
}

// Shared helper: makes a non-<button> element (image thumbnail, etc.)
// that already has a 'click' listener for opening something (image
// lightbox, file reveal, ...) reachable and activatable from the
// keyboard, mirroring the role="button" + tabindex="0" + Enter/Space
// pattern used by the topbar cwd pill above. Reuses the existing
// click listener via el.click() rather than duplicating handler
// logic at each call site.
function makeKeyboardClickable(el, label) {
  if (!el) return;
  el.setAttribute('role', 'button');
  el.setAttribute('tabindex', '0');
  if (label) el.setAttribute('aria-label', label);
  el.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    e.preventDefault();
    el.click();
  });
}

// Context-window ceiling for the chip's ratio display. Updated
// whenever the researcher picks a model (see updateModelChip) from
// the catalog's per-model ``context_window``. The starting value
// matches the default model and is replaced when a model is selected.
const DEFAULT_CONTEXT_WINDOW = 1_000_000;
let contextWindow = DEFAULT_CONTEXT_WINDOW;
// Last solid count returned by ``count_next_context``. The chip
// keeps showing this until a new count lands — no estimates, no
// projections from pending messages, no provider ``turn_done``
// echoes. Triggers (turn complete, rewind success, session
// open/switch, attachment add/remove) call ``triggerContextRecount``
// which fades the chip and updates this when the backend returns.
let lastContextCount = null;     // {tokens, exact, ceiling}
let contextCountRequestId = 0;   // monotonic counter for stale-response rejection

// (No high-water clamp on the context chip. Earlier we Math.max'd
// each turn_done's reported usage against a per-cwd watermark so the
// chip "only grew." That hid measurement asymmetries between
// providers — Anthropic's MAX-across-rounds inflation got pinned in
// place forever, and OpenAI's silent truncation never visibly shrank
// the chip even though it was happening. The chip now reflects
// exactly what the last completed turn reported. The conversation
// chain grows monotonically on its own, so under normal use the
// chip will only rise. If it drops, that's a real signal — a
// reconnect that lost in-context state, a model swap that opened a
// fresh session, a truncation event — and surfacing it is the
// point.)

// ---- multi-session focus state -------------------------------------------
// The bridge runs every session as its own SessionRunner: turns in
// session A keep streaming after the researcher clicks B in the
// sidebar. The frontend tracks two pieces of state to render that
// honestly:
//
// - ``currentCwd``: the path of the session the user is currently
//   looking at. Set by ``showChat`` on first load and by
//   ``switchSession`` afterwards. Events that don't match this cwd
//   are background activity: their busy-state still updates the
//   sidebar dot, but they don't render into the focused
//   transcript.
// - ``busySessions``: a Set of cwd paths that have a turn currently
//   in flight. A focus-matched event flips the composer Send/Stop
//   icons; a background-matched event flips only the sidebar dot.
//   We add to the Set when a session's send is queued and remove
//   when a terminal event (turn_done / turn_error / auth_failure)
//   arrives for that session.
let currentCwd = null;
const busySessions = new Set();
// Environment-health banner: "not now" dismissal for this page
// session only (see ``refreshDoctorBanner`` below). Not persisted
// across a relaunch -- a blocked runtime is worth re-surfacing next
// time Sift starts, since silently staying dismissed forever would
// recreate exactly the "script fails, no explanation" gap the
// banner exists to close.
let doctorBannerDismissed = false;

// Sessions whose in-flight turn the researcher just hit Stop on.
// Cancellation is async — the bridge has to propagate it through the
// asyncio task and the provider stream, which can leave tokens or
// tool calls already in transit. Without this set, those latent
// events keep painting into the transcript even though the
// researcher signalled "stop". We add the cwd on Stop click and
// remove it on the next terminal event for that session OR on the
// next user send (whichever comes first, in case a terminal never
// arrives because the SDK closed without one). While a cwd is in
// the set, ``sift_event`` drops non-terminal events for that
// session.
// Cancelled-turn drop list. The runner stamps every event with a
// per-turn id; when Stop fires we add the in-flight id here and
// keep dropping any late events that carry it. Crucially, this set
// is NEVER cleared when a new message starts — that was the prior
// bug: ``cancelledCwds.delete(currentCwd)`` on send let late events
// from the previous (cancelled) turn slip through after the new
// turn started, because the suppression key was a cwd rather than
// a turn id. Each new turn gets a fresh id, so the new turn's
// events naturally pass the filter without anyone having to clear
// state. The backend dispatcher applies the same drop authoritatively
// (see ``_dispatch_event`` in ui.py) — this set is best-effort
// defense in depth.
//
// Bounded growth: oldest ids are evicted past ~CANCELLED_TURN_ID_CAP
// entries. After eviction a stray late event can render, but by
// then the turn is far enough back in history that a brief flash
// before the staleness sweep is acceptable.
const CANCELLED_TURN_ID_CAP = 256;
const cancelledTurnIds = new Set();
const cancelledTurnIdOrder = [];

function markTurnCancelled(turnId) {
  if (!turnId) return;
  if (cancelledTurnIds.has(turnId)) return;
  cancelledTurnIds.add(turnId);
  cancelledTurnIdOrder.push(turnId);
  while (cancelledTurnIdOrder.length > CANCELLED_TURN_ID_CAP) {
    const evicted = cancelledTurnIdOrder.shift();
    cancelledTurnIds.delete(evicted);
  }
}

// Persisted model choice — survives restarts. Applied on boot after
// (Model preference used to live in localStorage as a global default.
// It now lives per-session in ``.sift/session_state.json`` and is
// restored by the backend on session open — see ``_set_cwd`` /
// ``_restore_session_model_preference`` in ui.py.)

// ----- theme toggle -------------------------------------------------------
// Temporary light/dark override. When the user hasn't clicked the
// toggle, the CSS `prefers-color-scheme` media query picks the theme
// from the OS. Once they click, we set `data-theme` on <html> and
// persist to localStorage so the choice survives restarts. Clearing
// the entry (via the same toggle or dev tools) returns to OS
// follow-mode.
const THEME_STORAGE_KEY = 'sift.theme';
const themeToggleBtn = document.getElementById('theme-toggle');
const themeToggleIcon = document.getElementById('theme-toggle-icon');

function currentTheme() {
  // Effective theme (light | dark): whatever data-theme says, or
  // the OS preference if no override is set.
  const forced = document.documentElement.getAttribute('data-theme');
  if (forced === 'light' || forced === 'dark') return forced;
  return window.matchMedia('(prefers-color-scheme: dark)').matches
    ? 'dark' : 'light';
}

function renderThemeIcon() {
  if (!themeToggleIcon) return;
  // Glyph shows the theme you'll jump TO on click, matching the
  // affordance researchers expect from every other theme toggle.
  themeToggleIcon.textContent = currentTheme() === 'dark' ? '☀' : '☾';
  if (themeToggleBtn) {
    const dark = currentTheme() === 'dark';
    themeToggleBtn.setAttribute('aria-pressed', String(dark));
    themeToggleBtn.setAttribute(
      'aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme'
    );
    themeToggleBtn.title = dark ? 'Switch to light theme' : 'Switch to dark theme';
  }
}

function applyStoredTheme() {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === 'light' || stored === 'dark') {
      document.documentElement.setAttribute('data-theme', stored);
    }
  } catch (_) { /* localStorage blocked — fall back to OS default */ }
  renderThemeIcon();
}

if (themeToggleBtn) {
  themeToggleBtn.addEventListener('click', () => {
    const next = currentTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem(THEME_STORAGE_KEY, next); } catch (_) {}
    renderThemeIcon();
  });
}

// OS theme change while Sift is open: keep the icon in sync
// (only matters when no explicit override is set).
window.matchMedia('(prefers-color-scheme: dark)').addEventListener(
  'change', renderThemeIcon
);

applyStoredTheme();

// Depth tiers — kept in sync with sift/policy.py::VALID_DEPTHS and
// with the get_schema tool help. Labels are researcher-facing plain
// English using an additive "+ X" convention: each tier adds what
// the previous tier sees, so the ladder reads naturally top to
// bottom. No "(default)" marker here — the current tier is implied
// by the <select>'s own selected-state, and the backend default
// (DEFAULT_MAX_DEPTH in policy.py) decides which one that is.
const DEPTH_TIERS = [
  { value: 'names_only',                 label: 'Variable names only' },
  { value: 'names_types',                label: '+ types' },
  { value: 'names_types_labels',         label: '+ labels / value labels' },
  { value: 'names_types_labels_summary', label: '+ NA count / distinct count' },
];

// ----- view routing ------------------------------------------------------

// Display labels for provider ids. The auth bridge speaks lowercase
// ids ("anthropic", "openai"); UI copy should never leak those raw.
// Falls back to capitalising the id if a new provider is added before
// this map catches up.
const PROVIDER_LABELS = {
  anthropic: 'Anthropic',
  openai: 'OpenAI',
  gemini: 'Gemini',
  openai_compatible: 'Custom (OpenAI-compatible)',
};
function providerLabel(id) {
  if (!id) return '';
  if (PROVIDER_LABELS[id]) return PROVIDER_LABELS[id];
  return id.charAt(0).toUpperCase() + id.slice(1);
}

function showAuth(authPayload) {
  /* Reveal the auth screen and render per-provider rows from the
   * payload returned by ``ui_ready`` / ``auth_status``. Called on
   * first launch (no provider configured) and any time the
   * researcher clicks "back to auth" from the landing screen. */
  if (!authEl) return;
  authEl.classList.remove('hidden');
  if (landingEl) landingEl.classList.add('hidden');
  if (chatEl) chatEl.classList.add('hidden');
  setSkipTarget('#auth-content', 'Skip to provider setup');
  if (authStatusEl) {
    authStatusEl.textContent = '';
    authStatusEl.className = 'auth-status';
  }
  renderAuthScreen(authPayload);
  // Drop focus into the first unconfigured input so a first-launch
  // researcher can paste straight away without tabbing to find the
  // field. If every provider is already authed (e.g., they hit
  // "Manage providers" from landing just to peek), skip — stealing
  // focus to a field they don't need to touch is noisy.
  const firstEmpty = authEl.querySelector(
    '.auth-provider:not(.configured) [data-role="key-input"]'
  );
  if (firstEmpty) {
    // requestAnimationFrame so focus lands after the panel has
    // un-hidden; focusing a still-hidden element is a no-op.
    requestAnimationFrame(() => firstEmpty.focus());
  }
}

function renderAuthScreen(authPayload) {
  /* Update each provider row's status badge and Forget button based
   * on the auth_status payload. Continue button is enabled iff at
   * least one provider is configured. */
  if (!authEl) return;
  const status = (authPayload && authPayload.providers) || {};
  const rows = authEl.querySelectorAll('.auth-provider');
  rows.forEach((row) => {
    const provider = row.dataset.provider;
    const info = status[provider] || {};
    const statusEl = row.querySelector('[data-role="status"]');
    const forgetBtn = row.querySelector('[data-role="forget-btn"]');
    if (info.configured) {
      row.classList.add('configured');
      if (statusEl) {
        statusEl.classList.add('ok');
        statusEl.textContent = info.method === 'endpoint'
          ? 'Endpoint configured'
          : 'API key stored';
      }
    } else {
      row.classList.remove('configured');
      if (statusEl) {
        statusEl.classList.remove('ok');
        // Tri-state. ``keyring_unavailable`` is distinct from ``missing``:
        // a locked or denied macOS Keychain prompt previously rendered
        // identically to "no credential here", so a researcher might
        // re-paste a key they already had (or dismiss the prompt
        // thinking the credential was gone when it was still stored).
        // Show a different message so they retry the prompt instead.
        statusEl.textContent = info.readiness === 'blocked_by_policy'
          ? 'Blocked by organization policy'
          : (info.readiness === 'needs_configuration'
              ? 'Endpoint settings incomplete'
              : (info.status === 'keyring_unavailable'
                  ? `${protectedCredentialStore} unavailable — could not check`
                  : 'Not configured'));
      }
    }
    if (forgetBtn) {
      forgetBtn.disabled = !info.has_keyring_entry;
    }
  });
  const anyAuthed = !!(authPayload && authPayload.any_authed);
  if (authContinueBtn) {
    authContinueBtn.disabled = !anyAuthed;
  }
  if (authContinueHint) {
    // Swap between the "why is Continue gray?" prompt and a quiet
    // ready-state confirmation. Keeping the element rendered (rather
    // than show/hide) prevents the footer from jumping vertically
    // when the researcher's first Save flips the state.
    authContinueHint.textContent = anyAuthed
      ? 'Ready when you are.'
      : 'Save at least one provider to continue.';
    authContinueHint.classList.toggle('ready', anyAuthed);
  }
}

function setAuthStatus(text, kind) {
  if (!authStatusEl) return;
  authStatusEl.textContent = text || '';
  authStatusEl.className = 'auth-status' + (kind ? ' ' + kind : '');
}

async function loadAuthStatus() {
  if (!window.pywebview || !window.pywebview.api) return null;
  if (typeof window.pywebview.api.auth_status !== 'function') return null;
  try {
    return await window.pywebview.api.auth_status();
  } catch (err) {
    console.warn('auth_status failed', err);
    return null;
  }
}

if (authEl) {
  // Save / Forget per row.
  authEl.querySelectorAll('.auth-provider').forEach((row) => {
    const provider = row.dataset.provider;
    const input = row.querySelector('[data-role="key-input"]');
    const saveBtn = row.querySelector('[data-role="save-btn"]');
    const forgetBtn = row.querySelector('[data-role="forget-btn"]');

    if (saveBtn && input) {
      saveBtn.addEventListener('click', async () => {
        if (!window.pywebview || !window.pywebview.api) return;
        const key = (input.value || '').trim();
        if (!key) {
          setAuthStatus('Paste an API key first.', 'error');
          return;
        }
        saveBtn.disabled = true;
        try {
          const res = await window.pywebview.api.save_credential(provider, key);
          if (res && res.ok) {
            input.value = '';
            setAuthStatus(`${providerLabel(provider)} key saved.`, 'ok');
            renderAuthScreen(res.auth);
          } else {
            const reason = (res && res.reason) || 'unknown error';
            setAuthStatus(`Save failed: ${reason}`, 'error');
          }
        } catch (err) {
          setAuthStatus('Save failed: ' + err, 'error');
        } finally {
          saveBtn.disabled = false;
        }
      });
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') saveBtn.click();
      });
    }

    if (forgetBtn) {
      forgetBtn.addEventListener('click', async () => {
        if (!window.pywebview || !window.pywebview.api) return;
        if (typeof window.pywebview.api.delete_credential !== 'function') return;
        forgetBtn.disabled = true;
        try {
          const res = await window.pywebview.api.delete_credential(provider);
          if (res && res.ok) {
            setAuthStatus(`${providerLabel(provider)} credential removed.`, 'ok');
            renderAuthScreen(res.auth);
          } else {
            setAuthStatus('Forget failed: ' + ((res && res.reason) || ''), 'error');
          }
        } catch (err) {
          setAuthStatus('Forget failed: ' + err, 'error');
        }
      });
    }
  });

  // Continue → land on the data picker (or chat if cwd already set).
  if (authContinueBtn) {
    authContinueBtn.addEventListener('click', async () => {
      if (!window.pywebview || !window.pywebview.api) return;
      try {
        const state = await window.pywebview.api.ui_ready();
        if (state && state.state === 'ready') {
          showChat(state);
          refreshDoctorBanner();
        } else {
          showLanding();
        }
      } catch (err) {
        console.error('ui_ready failed after auth', err);
        showLanding();
      }
    });
  }
}

// "Manage providers" link on the landing card — explicit way to
// reach the auth screen when the researcher already has one configured
// provider but wants to add, replace, or remove another. The
// auth screen only appears automatically when no provider is
// configured at all.
async function openAuthScreen() {
  const auth = await loadAuthStatus();
  showAuth(auth || { providers: {}, any_authed: false });
}

const manageProvidersBtn = document.getElementById('manage-providers-btn');
if (manageProvidersBtn) {
  manageProvidersBtn.addEventListener('click', openAuthScreen);
}

// Inline "create an API key" links inside each provider's help text.
// WKWebView would otherwise navigate the whole webview to the provider
// console, blowing away the auth state. Intercept the click and hand
// the URL to the OS browser via the allowlisted bridge.
if (authEl) {
  authEl.addEventListener('click', (e) => {
    const link = e.target.closest('.auth-help-link');
    if (!link) return;
    e.preventDefault();
    const url = link.getAttribute('data-external-url');
    if (url) openExternal(url);
  });
}

function showLanding() {
  if (authEl) authEl.classList.add('hidden');
  landingEl.classList.remove('hidden');
  chatEl.classList.add('hidden');
  setSkipTarget('#landing-content', 'Skip to workspace selection');
  // Reset landing-side UI so re-entering from "New session" feels
  // fresh: re-enable both buttons, clear any leftover status text,
  // drop the drag-over highlight. Without this, a researcher who
  // cancelled a previous file-picker or came back from chat sees
  // greyed-out buttons and thinks the page is frozen.
  setLandingBusy(false, '');
  dropZone.classList.remove('dragover');
  // Show past sessions underneath the upload area so a researcher
  // can jump straight back into a prior working directory.
  loadLandingSessions();
}

async function loadLandingSessions() {
  const container = document.getElementById('landing-sessions');
  const listEl = document.getElementById('landing-sessions-list');
  if (!container || !listEl) return;
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.list_sessions !== 'function') return;
  try {
    const res = await window.pywebview.api.list_sessions();
    if (!res || !res.ok) return;
    const sessions = res.sessions || [];
    if (sessions.length === 0) {
      container.classList.add('hidden');
      return;
    }
    container.classList.remove('hidden');
    listEl.innerHTML = '';
    // Cap to the 5 most recent so the landing page doesn't become
    // a wall of text. Full list is always available in the sidebar
    // once the researcher is in a session.
    sessions.slice(0, 5).forEach((s) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'landing-session';
      btn.title = s.path;

      // Title = the resolved session title (custom name if set, else
      // the dataset filename, else "<first> +N"). Renders single-line
      // with the relative-age label pinned to the right.
      const title = document.createElement('span');
      title.className = 'landing-session-title';
      title.textContent = s.title || (s.datasets.length
        ? s.datasets.join(', ')
        : '(no data files)');
      btn.appendChild(title);

      const age = document.createElement('span');
      age.className = 'landing-session-age';
      age.textContent = formatSessionAge(s.timestamp);
      btn.appendChild(age);

      btn.addEventListener('click', () => switchSession(s.path, false));
      listEl.appendChild(btn);
    });
  } catch (_) { /* silent — landing shouldn't block on history */ }
}

function showChat(payload) {
  /* Reveal the chat view and populate it from a ready payload.
   * Called both on initial startup (when the backend already has a
   * cwd) and after a session switch, so it has to reset any prior
   * transcript rather than just updating pieces in place. */
  if (authEl) authEl.classList.add('hidden');
  landingEl.classList.add('hidden');
  chatEl.classList.remove('hidden');
  setSkipTarget('#messages', 'Skip to research workspace');

  // Reset the transcript to a fresh welcome placeholder.
  // replayHistory() below will clear and repopulate this if the
  // target session has persisted events on disk.
  messagesEl.innerHTML = '';
  const welcomeMsg = document.createElement('div');
  // .welcome-greeting is a permanent class (not toggled with
  // welcome-only) so the horizontal centering — full chat-area
  // width with body justify-content:center — applies even AFTER
  // the first message arrives. Without this, removing welcome-only
  // dropped the welcome back into the 960px column cap, which
  // sits left-of-center on a wide window because the sidebar
  // eats space on the left side.
  welcomeMsg.className = 'message system welcome-greeting';
  const welcomeBody = document.createElement('div');
  welcomeBody.className = 'message-body';
  welcomeBody.id = 'welcome';
  welcomeBody.textContent = payload.greeting || 'Ready.';
  welcomeMsg.appendChild(welcomeBody);
  const analyzeAction = buildAnalyzeAction(payload);
  if (analyzeAction) welcomeMsg.appendChild(analyzeAction);
  const starters = buildStarterPrompts(payload);
  if (starters) welcomeMsg.appendChild(starters);
  messagesEl.appendChild(welcomeMsg);
  setWelcomeOnlyMode(true);

  // Topbar shows the abbreviated session path (``~/.sift-sessions/
  // <id>``) as a clickable handle to "reveal in Finder". The full
  // path stays in the title-attribute tooltip; the pill's click
  // handler hands the path to ``open_path`` so the researcher can
  // inspect Sift's scripts / logs / generated data on disk. The
  // session's friendly name (custom_name or auto-derived) lives on
  // the sidebar row, not here — see the cwdEl click-handler comment
  // above for the rationale.
  cwdEl.textContent = formatCwd(payload.cwd || '');
  cwdEl.title = payload.cwd
    ? `${payload.cwd}. Click to open in ${nativeFileManager}.`
    : '';

  // Mark this session as the focused one. Subsequent
  // ``window.sift_event`` callbacks compare incoming
  // ``session_cwd`` against this — events for other sessions are
  // background activity and don't render into the visible
  // transcript (but still update the sidebar busy dot).
  currentCwd = payload.cwd || null;
  // Drop staged composer state so attachments don't leak across
  // sessions: an image staged in A but never sent must NOT ride
  // along with the next message in B. (Without this, the JS
  // ``stagedImages`` array stays populated through the focus
  // switch and the next form submit in B would inline A's
  // attachment into B's prompt.) Also revoke object URLs so the
  // blobs aren't pinned in memory.
  if (stagedImages.length > 0) {
    stagedImages.forEach((img) => {
      if (img && img.url) URL.revokeObjectURL(img.url);
    });
    stagedImages.length = 0;
  }
  stagedDataNotices.length = 0;
  renderAttachments();
  // Each session has its own files; bust the cache so the next
  // "@" doesn't offer rows from the previous session.
  invalidateMentionCache();
  if (typeof closeMentionPopup === 'function') closeMentionPopup();
  // Sync the composer state to whether THIS session is currently
  // busy. Switching to a session that's mid-turn shows Stop +
  // loading indicator immediately; switching to an idle session
  // shows Send.
  syncComposerToFocus();

  updatePolicyChip(payload.policy);
  loadSessions();
  loadModels();
  refreshCheckpointsChip();
  // Hide the chip until the first turn_done arrives — without a
  // measurement we have nothing honest to display. Reset the
  // last-rendered count too so a fresh session doesn't re-render
  // the previous session's number against the new model's window.
  // Drop the previous session's count so we don't briefly render
  // it against the new session's eventual count. ``replayHistory``
  // triggers a recount once the new history loads.
  lastContextCount = null;
  if (contextChip) {
    contextChip.classList.add('hidden');
    contextChip.classList.remove('stale');
  }

  // Pass the cwd we just committed to so replay can abandon if the
  // user has already switched again by the time the history fetch
  // resolves. Without the guard, two quick A→B switches race the
  // get_chat_history responses and a late-arriving A history paints
  // into B's transcript.
  replayHistory(currentCwd);
  rotatePlaceholder();
  input.focus();
}

// Turns that yielded no visible reply artifacts. Keep them around
// just long enough for the researcher to notice the failure, then
// sweep them the next time a new message is sent so the transcript
// doesn't fill with dead-end bubbles.
let activeLiveTurn = null;
let replayTailTurn = null;
let staleTranscriptTurns = [];

function dropNodes(nodes) {
  (nodes || []).forEach((node) => {
    if (node && typeof node.remove === 'function') node.remove();
  });
}

function sweepStaleTranscriptTurns() {
  staleTranscriptTurns.forEach((turn) => dropNodes(turn));
  staleTranscriptTurns = [];
}

function queueDisposableTurn(nodes) {
  const kept = (nodes || []).filter(Boolean);
  if (kept.length > 0) staleTranscriptTurns.push(kept);
}

async function replayHistory(expectedCwd) {
  /* ``expectedCwd`` pins which session this replay belongs to. After
   * the ``get_chat_history`` await resolves, we compare against the
   * live ``currentCwd`` and abandon if it changed — that means the
   * researcher switched again while our fetch was in flight, and a
   * fresher ``replayHistory`` call is already (or about to be)
   * painting the new session's transcript. Without this guard, a
   * late-arriving response wipes ``messagesEl`` and paints the
   * previous session's history into the now-focused session's view
   * (the response carries whichever cwd was active at call time;
   * pywebview RPC ordering doesn't preserve issue order).
   *
   * Callers that don't pass an explicit cwd default to the live
   * ``currentCwd`` — same effect as the pre-guard behaviour, plus
   * the guard is a no-op (expected matches live).
   */
  if (expectedCwd === undefined) expectedCwd = currentCwd;
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.get_chat_history !== 'function') return;
  try {
    const res = await window.pywebview.api.get_chat_history();
    if (expectedCwd !== currentCwd) return;
    if (!res || !res.ok) return;
    const events = res.events || [];
    // Always wipe the existing transcript before painting from
    // the persisted log — even when the log is empty. The empty
    // case fires after a rewind that dropped the very first
    // message: the backend has truncated chat_history.jsonl, but
    // an early ``return`` here would leave the old DOM in place,
    // so ``runEditedMessage``'s subsequent ``appendUser`` would
    // stack the revised bubble on top of stale messages from the
    // dropped branch. Same reasoning for ``setWelcomeOnlyMode`` —
    // the welcome "system" line should also be cleared so the
    // post-rewind state starts empty, and the next live turn
    // toggles welcome mode off naturally.
    messagesEl.innerHTML = '';
    setWelcomeOnlyMode(false);
    if (events.length === 0) return;
    try {
      events.forEach((evt) => replayEvent(evt));
    } finally {
      if (replayTailTurn && !replayTailTurn.hasVisibleReply) {
        queueDisposableTurn(replayTailTurn.nodes);
      }
      replayTailTurn = null;
    }
    scrollToBottom();
    // Restore the context chip from the LAST persisted ``turn_done``,
    // if there is one. Each ``turn_done`` carries the provider's
    // authoritative token counts (``post_turn_tokens`` since the
    // canonical-contract commit; sum of the granular fields for
    // older sessions persisted before that field existed). This is
    // honest data, not a chars/4 estimate. Without it, a session
    // with valid prior measurements would show no context pressure
    // across reload / session-switch / model-swap until another turn
    // completed — which can be a long wait on idle resumes.
    let lastTurnDone = null;
    for (let i = events.length - 1; i >= 0; i--) {
      if (events[i].type === 'turn_done') {
        lastTurnDone = events[i];
        break;
      }
    }
    // Trigger a fresh pre-flight count after replay. Don't seed the
    // chip from ``lastTurnDone`` usage fields — that mixes
    // post-billing reality into a chip that's supposed to predict
    // the NEXT request's headroom, which was the source of the
    // fluctuation this refactor cleans up.
    triggerContextRecount('replay');
    // Re-anchor the loading indicator if this session is still in
    // flight. ``showChat → syncComposerToFocus`` already added it
    // for a busy session, but the ``messagesEl.innerHTML = ''`` we
    // just ran wiped that node. Without this, a refresh / focus
    // switch into a session mid-turn loses the cat + label until
    // the next event arrives. Check ``busySessions`` (not
    // ``turnInFlight``) because turn_done may have landed mid-
    // replay and flipped the focused-session bit off.
    if (currentCwd && busySessions.has(currentCwd)) {
      showLoadingIndicator();
    }
  } catch (err) {
    console.warn('get_chat_history failed', err);
  }
}

function replayEvent(evt) {
  /* Shared handler for replayed events. Live events go through
   * window.sift_event; replay feeds the same records back
   * through the same append* helpers so rendering is identical.
   * Named deliberately to avoid shadowing window.dispatchEvent,
   * which is a DOM method and was my first attempt. */
  switch (evt.type) {
    case 'user_message': {
      if (replayTailTurn && !replayTailTurn.hasVisibleReply) {
        dropNodes(replayTailTurn.nodes);
      }
      // ``attachments`` is a list of script filenames (the backend
      // persists the names that travelled with this message).
      const att = Array.isArray(evt.attachments) ? evt.attachments : [];
      // ``images`` is the persisted ``[{data, mime}]`` list. Convert
      // to the data-URL shape ``appendUser`` expects so the bubble
      // shows the actual thumbnails on reload, not a phantom "you
      // sent an image" with no preview. Older histories carry only
      // ``image_count`` (no bytes) — fall through to the no-images
      // path; better a bare text bubble than a broken thumb.
      const imgs = Array.isArray(evt.images)
        ? evt.images
            .filter((i) => i && typeof i.data === 'string')
            .map((i) => ({
              url: `data:${i.mime || 'image/png'};base64,${i.data}`,
              mime: i.mime || 'image/png',
            }))
        : [];
      const userEl = appendUser(evt.text || '', att, imgs);
      replayTailTurn = { nodes: [userEl], hasVisibleReply: false };
      break;
    }
    case 'assistant_text':
      if (replayTailTurn) replayTailTurn.hasVisibleReply = true;
      appendAssistant(evt.text || '');
      break;
    case 'assistant_thinking':
      if (replayTailTurn) replayTailTurn.hasVisibleReply = true;
      appendThinking(evt.text || '');
      break;
    case 'tool_call': {
      const card = appendToolCall(evt);
      if (card && replayTailTurn) replayTailTurn.hasVisibleReply = true;
      break;
    }
    case 'tool_result': {
      const card = appendToolResult(evt);
      if (card && replayTailTurn) replayTailTurn.hasVisibleReply = true;
      break;
    }
  }
}

function formatCwd(raw) {
  /* Abbreviate the working-directory path for display in the topbar
   * chip. The full path —
   * ``/Users/you/.sift-sessions/20260422T160059Z_f13630f4`` —
   * overflows the chip and the timestamp at the tail is the part
   * the researcher actually wants to see (which session). We
   * collapse ``/Users/<user>`` to ``~`` and cap at 40 chars with a
   * left-side ellipsis, preserving the tail. Full path remains
   * available via the ``title`` tooltip.
   */
  if (!raw) return '';
  let p = raw.replace(/^\/Users\/[^/]+/, '~');
  const MAX = 40;
  if (p.length <= MAX) return p;
  return '…' + p.slice(-(MAX - 1));
}

// ----- landing: file picker / folder picker / drag-drop -----------------

// Auto-dismiss handle shared by setLandingBusy and setLandingError.
// Lives at module scope so a busy / new-error transition can cancel
// the prior error's pending clear and avoid wiping a still-relevant
// message mid-action.
let landingErrorTimer = null;

const trySampleBtn = document.getElementById('try-sample-btn');

if (trySampleBtn) {
  trySampleBtn.addEventListener('click', async () => {
    /* Routes through the same staging path as a real upload, so what
     * the evaluator exercises is the real pipeline — not a demo mode
     * that behaves differently from the thing they're evaluating. */
    if (!window.pywebview || !window.pywebview.api) {
      setLandingError('Restart Sift to load the sample dataset.');
      return;
    }
    setLandingBusy(true, 'Generating sample data…');
    trySampleBtn.disabled = true;
    try {
      const res = await window.pywebview.api.start_sample_session();
      await handleSessionResult(res);
    } catch (e) {
      setLandingError('Could not start the sample session: ' + e);
    } finally {
      trySampleBtn.disabled = false;
    }
  });
}

function buildAnalyzeAction(payload) {
  /* The single-button entry point for "just look at my data and tell
   * me what's in it" — no prompt to write, no menu to pick from.
   * Deliberately separate from ``buildStarterPrompts`` below (which
   * stays as narrower, alternative starting points) rather than
   * folded into that list: this is meant to read as THE default
   * action for an empty session, not one option among several, so it
   * gets its own visual weight (a filled primary button, not a chip)
   * and sits above the chips.
   *
   * Like the starter chips, this is a real message sent through the
   * normal composer submit path — not a special "run analysis"
   * codepath that could drift from what a researcher typing the same
   * words would get. The autonomous-analysis playbook in the system
   * prompt is what actually does the work (profile, hypothesize,
   * model, diagnose, challenge, synthesize); this button only saves
   * someone the trouble of typing the trigger phrase.
   *
   * Same dataset-presence gate as the starter chips: with nothing
   * loaded, there is nothing to analyze.
   */
  const datasets = (payload && payload.policy && payload.policy.datasets)
    ? payload.policy.datasets : [];
  if (!datasets.length) return null;

  const wrap = document.createElement('div');
  wrap.className = 'analyze-action';
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'analyze-action-btn';
  btn.textContent = datasets.length === 1
    ? 'Analyze this dataset'
    : 'Analyze these datasets';
  btn.addEventListener('click', () => {
    input.value = (
      'Analyze this dataset end to end: profile it, find what stands '
      + 'out, verify and stress-test the findings that matter, and '
      + 'tell me the things most worth knowing.'
    );
    autosize();
    form.dispatchEvent(new Event('submit', { cancelable: true }));
  });
  wrap.appendChild(btn);
  const hint = document.createElement('div');
  hint.className = 'analyze-action-hint';
  hint.textContent = 'Sift profiles the data, runs and checks its own '
    + 'analysis, and shows its work — nothing leaves this machine but '
    + 'the sanitized results.';
  wrap.appendChild(hint);
  return wrap;
}

function buildStarterPrompts(payload) {
  /* Opening suggestions for an empty session.
   *
   * These are real messages: clicking one puts the text in the
   * composer and submits it through the normal path, so there is no
   * second send codepath that could drift from the real one. They are
   * not progress theatre and they do not claim anything has happened.
   *
   * Shown only when the session actually has data files — suggesting
   * "profile this dataset" with nothing loaded would be a dead end.
   * The blank composer is a bad first screen for someone who does not
   * yet know what the tool can be asked. */
  const datasets = (payload && payload.policy && payload.policy.datasets)
    ? payload.policy.datasets : [];
  if (!datasets.length) return null;

  const prompts = [
    'Profile this dataset and tell me what stands out.',
    'What questions could this data answer well, and which would it answer badly?',
    'Find the strongest relationships here, then check whether they hold up.',
  ];

  const wrap = document.createElement('div');
  wrap.className = 'starter-prompts';
  prompts.forEach((text) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'starter-chip';
    chip.textContent = text;
    chip.addEventListener('click', () => {
      input.value = text;
      autosize();
      // Reuse the composer's own submit handler rather than
      // reimplementing send: queueing, attachments, image staging and
      // in-flight handling all live there.
      form.dispatchEvent(new Event('submit', { cancelable: true }));
    });
    wrap.appendChild(chip);
  });
  return wrap;
}

function setLandingBusy(busy, msg) {
  if (landingErrorTimer) {
    clearTimeout(landingErrorTimer);
    landingErrorTimer = null;
  }
  chooseFilesBtn.disabled = busy;
  chooseFolderBtn.disabled = busy;
  if (trySampleBtn) trySampleBtn.disabled = busy;
  landingStatus.classList.remove('error');
  landingStatus.textContent = msg || '';
}

function setLandingError(msg, { autoDismissMs } = {}) {
  if (landingErrorTimer) {
    clearTimeout(landingErrorTimer);
    landingErrorTimer = null;
  }
  chooseFilesBtn.disabled = false;
  chooseFolderBtn.disabled = false;
  if (trySampleBtn) trySampleBtn.disabled = false;
  landingStatus.classList.add('error');
  landingStatus.textContent = msg;
  if (autoDismissMs) {
    landingErrorTimer = setTimeout(() => {
      landingStatus.classList.remove('error');
      landingStatus.textContent = '';
      landingErrorTimer = null;
    }, autoDismissMs);
  }
}

async function handleSessionResult(result) {
  if (!result) {
    setLandingError('no response from the backend');
    return;
  }
  if (!result.ok) {
    const reason = result.reason || 'unknown';
    if (reason === 'cancelled') {
      // Researcher cancelled the dialog — no noise, just clear status.
      setLandingBusy(false, '');
    } else {
      setLandingError(reason);
    }
    return;
  }
  showChat(result);
}

chooseFilesBtn.addEventListener('click', async () => {
  if (!window.pywebview || !window.pywebview.api) return;
  setLandingBusy(true, 'Opening file picker…');
  try {
    const result = await window.pywebview.api.choose_files();
    await handleSessionResult(result);
  } catch (err) {
    setLandingError('failed: ' + err);
  }
});

chooseFolderBtn.addEventListener('click', async () => {
  if (!window.pywebview || !window.pywebview.api) return;
  setLandingBusy(true, 'Opening folder picker…');
  try {
    const result = await window.pywebview.api.choose_folder();
    await handleSessionResult(result);
  } catch (err) {
    setLandingError('failed: ' + err);
  }
});

// Drag-drop. Visual state on the drop zone; actual handling on the whole
// landing area so a near-miss still works.
['dragenter', 'dragover'].forEach((name) => {
  landingEl.addEventListener(name, (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });
});
['dragleave', 'drop'].forEach((name) => {
  landingEl.addEventListener(name, (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
  });
});

// Keep every data-file drop surface on one allowlist. This mirrors the
// backend's schema.DATA_EXTENSIONS; a regression test keeps the two in sync.
const DATA_FILE_EXTS = new Set([
  'csv', 'tsv', 'dta', 'rds', 'rda', 'rdata', 'parquet', 'feather', 'arrow', 'ipc', 'orc',
  'json', 'jsonl', 'ndjson', 'sav', 'zsav', 'por', 'sas7bdat', 'xpt', 'xlsx', 'xls', 'ods',
  'gz', 'zip', 'avro', 'xml', 'dbf', 'h5', 'hdf5', 'nc', 'netcdf', 'mat',
  'fits', 'fit', 'fts', 'geojson', 'gpkg', 'shp', 'tif', 'tiff', 'vrt',
  'vcf', 'bcf', 'bed', 'nii', 'dcm', 'fhir',
]);

landingEl.addEventListener('drop', async (e) => {
  e.preventDefault();
  const dt = e.dataTransfer;
  if (!dt || !dt.files || dt.files.length === 0) return;
  const files = Array.from(dt.files);
  // Only data files we recognize. Anything else gets rejected up-front
  // so researchers see the reason in the UI rather than a confused
  // session with non-data files in it.
  const accepted = files.filter((f) => DATA_FILE_EXTS.has(fileExt(f)));
  const rejected = files.length - accepted.length;
  if (accepted.length === 0) {
    setLandingError(
      'Drop a supported data file (CSV, Stata, R, Parquet/Arrow, '
      + 'SPSS/SAS, JSON/JSON Lines, or Excel/OpenDocument). Other types are ignored.'
    );
    return;
  }
  // Size gate: reject before FileReader runs. Without this, a
  // researcher who drops a multi-GB .dta sees the app freeze for
  // tens of seconds (and possibly OOM) before the Python side
  // returns "too large." The native picker is one click away and
  // copies straight from disk.
  const oversize = accepted.find((f) => f.size > MAX_DRAG_DROP_BYTES);
  if (oversize) {
    setLandingError(
      formatDragDropOversizeReason(oversize, 'Use Choose files instead.'),
      { autoDismissMs: 4000 },
    );
    return;
  }
  // Aggregate cap: each file passes the per-file cap (above), but
  // the loop below accumulates every file's base64 string in
  // memory before calling upload_files. Two 800 MB files would
  // each pass the per-file cap yet hold ~2.1 GB of base64 in the JS
  // heap concurrently, freezing or crashing the page. Bound the
  // total too. Same threshold as per-file so the user sees a
  // consistent rule: "the drag-drop path can move up to 1 GB at
  // a time, regardless of how many files."
  const aggregateBytes = accepted.reduce((s, f) => s + f.size, 0);
  if (aggregateBytes > MAX_DRAG_DROP_BYTES) {
    setLandingError(
      'Total drop too large. Use Choose files instead.',
      { autoDismissMs: 4000 },
    );
    return;
  }
  try {
    // Read serially with a progress message so large drops don't
    // look frozen. readAsDataURL loads the whole file into memory —
    // size gated above, so the worst case here is one ~1 GB read.
    const payload = [];
    for (let i = 0; i < accepted.length; i++) {
      const file = accepted[i];
      const sizeMb = Math.round(file.size / (1024 * 1024));
      setLandingBusy(
        true,
        `Reading (${i + 1}/${accepted.length}) ${file.name}` +
          (sizeMb > 0 ? ` (${sizeMb} MB)…` : '…')
      );
      payload.push(await readFileAsBase64(file));
    }
    setLandingBusy(
      true,
      `Staging ${accepted.length} file${accepted.length === 1 ? '' : 's'}…`
    );
    const result = await window.pywebview.api.upload_files(payload);
    if (result && result.ok && rejected > 0) {
      // Will switch views — the note on rejected-types is just
      // a courtesy; no need to block.
      console.info(`${rejected} non-data file(s) ignored.`);
    }
    await handleSessionResult(result);
  } catch (err) {
    setLandingError('upload failed: ' + err);
  }
});

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      // result is a data URL; base64 content is after the first comma.
      const dataUrl = reader.result;
      const comma = dataUrl.indexOf(',');
      resolve({
        name: file.name,
        content: comma >= 0 ? dataUrl.substring(comma + 1) : dataUrl,
      });
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

// ----- send / receive -----------------------------------------------------

// Turn lifecycle: the bridge's send_message is fire-and-forget (it
// queues the turn on the asyncio worker and returns immediately),
// so we cannot tie the active-turn state to the await
// on that call. Instead we latch `turnInFlight` to true on submit
// and clear it when a terminal event arrives (turn_done /
// turn_error / auth_failure). That keeps the Send button disabled
// — and blocks Enter-to-send — while the prior turn is running,
// so a quick tester can't pipeline prompts that interleave in the
// transcript.

let turnInFlight = false;

// Staged images for the next message. Populated by drop / paste,
// cleared on send. Each entry: { data: base64String, mime: string,
// url: objectURL (for thumbnail preview) }.
const stagedImages = [];
const attachmentsEl = document.getElementById('compose-attachments');
const ALLOWED_IMAGE_MIMES = new Set([
  'image/png', 'image/jpeg', 'image/webp', 'image/gif',
]);
const MAX_IMAGE_BYTES = 5 * 1024 * 1024;  // 5 MB per image (Anthropic limit ballpark)

// Drag-drop / paste cap on data and script files. The Python backend
// enforces the same cap (``_DRAG_DROP_MAX_BYTES`` in ui.py); both
// sides must move together. The FileReader → base64 → bridge → decode
// chain peaks at roughly 3–4× the file size in memory, so 1 GB peaks
// around 3–4 GB across the JS heap and the bridge transfer. That's
// fine on a 16 GB Mac and tight (but workable) on 8 GB during the
// transfer window — researchers on entry-tier hardware with multi-GB
// datasets should prefer the native file picker (Choose Files… /
// + button), which uses ``shutil.copy2`` and avoids the in-memory
// round-trip entirely. Above this cap, drag-drop is rejected with a
// clear hint pointing at that path. Picking the same cap on both
// sides means rejecting BEFORE allocating multiple GB — rejecting
// after the bridge transfer defeats the whole point of a cap.
const MAX_DRAG_DROP_BYTES = 1024 * 1024 * 1024;

function formatDragDropOversizeReason(file, hint) {
  return `${file.name} is too large for drag and drop. ${hint}`;
}

function renderAttachments() {
  if (!attachmentsEl) return;
  attachmentsEl.innerHTML = '';
  if (stagedImages.length === 0 && stagedDataNotices.length === 0) {
    attachmentsEl.classList.add('hidden');
    return;
  }
  attachmentsEl.classList.remove('hidden');
  stagedImages.forEach((img, idx) => {
    const wrap = document.createElement('div');
    wrap.className = 'compose-attachment';
    const thumb = document.createElement('img');
    thumb.src = img.url;
    thumb.alt = 'Staged image ' + (idx + 1);
    thumb.title = 'Click to view full size';
    thumb.style.cursor = 'zoom-in';
    thumb.addEventListener('click', () => showImageLightbox(img.url));
    makeKeyboardClickable(thumb, 'View staged image ' + (idx + 1) + ' full size');
    wrap.appendChild(thumb);
    const rm = document.createElement('button');
    rm.type = 'button';
    rm.className = 'compose-attachment-remove';
    rm.setAttribute('aria-label', 'Remove');
    rm.textContent = '×';
    rm.addEventListener('click', (e) => {
      // Stop propagation so the underlying thumbnail click
      // doesn't also fire the lightbox.
      e.stopPropagation();
      URL.revokeObjectURL(img.url);
      stagedImages.splice(idx, 1);
      renderAttachments();
    });
    wrap.appendChild(rm);
    attachmentsEl.appendChild(wrap);
  });
  // Named-chip notices for data/script files the researcher just
  // added. Data files are visual "yes that landed" receipts; script
  // files (.py / .do / .r / .rmd) ALSO travel with the next message
  // as a context block (see _pending_script_attachments in ui.py),
  // so the chip tooltip names that distinction.
  stagedDataNotices.forEach((name, idx) => {
    const chip = document.createElement('div');
    chip.className = 'compose-attachment compose-attachment-file';
    const ext = (name.split('.').pop() || '').toLowerCase();
    const isScript = ['py', 'do', 'r', 'rmd'].includes(ext);
    if (isScript) {
      chip.classList.add('compose-attachment-script');
      chip.title = name + '. Saved in this session and sent with your next message.';
    } else {
      chip.title = name + '. Copied into the session.';
    }
    const label = document.createElement('span');
    label.className = 'compose-attachment-filename';
    label.textContent = name;
    chip.appendChild(label);
    const rm = document.createElement('button');
    rm.type = 'button';
    rm.className = 'compose-attachment-remove';
    rm.setAttribute('aria-label', 'Dismiss');
    rm.textContent = '×';
    rm.addEventListener('click', async () => {
      // Splice the JS-side notice immediately for a responsive UI.
      const removed = stagedDataNotices.splice(idx, 1)[0];
      renderAttachments();
      // Scripts ride the next message as inline context (the
      // backend stages them in _pending_script_attachments). The
      // chip × must also call the bridge to unstage there —
      // without this, the file silently rides along after the
      // researcher dismissed it. The on-disk copy is untouched.
      if (
        isScript
        && window.pywebview
        && window.pywebview.api
        && typeof window.pywebview.api.unstage_attachment === 'function'
      ) {
        try {
          await window.pywebview.api.unstage_attachment(removed);
        } catch (err) {
          console.warn('unstage_attachment failed', err);
        }
        // The next request just shrunk — let the chip reflect it
        // immediately rather than waiting for the next turn.
        triggerContextRecount('attachment-remove');
      }
    });
    chip.appendChild(rm);
    attachmentsEl.appendChild(chip);
  });
}

async function stageImageFile(file) {
  if (!ALLOWED_IMAGE_MIMES.has(file.type)) {
    appendError('Only PNG, JPEG, WebP, and GIF images are supported.');
    return false;
  }
  if (file.size > MAX_IMAGE_BYTES) {
    appendError('Image too large. Max 5 MB per file.');
    return false;
  }
  const data = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const url = reader.result;
      const comma = url.indexOf(',');
      resolve(comma >= 0 ? url.slice(comma + 1) : url);
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
  stagedImages.push({
    data,
    mime: file.type,
    url: URL.createObjectURL(file),
  });
  renderAttachments();
  return true;
}

// Extensions we accept on composer drop/paste alongside images.
// Same set the + button's picker accepts — data files, R/Stata/
// Python scripts, Stata graphs, logs, and R Markdown. All copied
// into the session cwd so the model can reference them.
const COMPOSER_DATA_EXTS = new Set([
  ...DATA_FILE_EXTS,
  'do', 'r', 'py', 'ipynb',
  'gph',
  'log', 'smcl',
  'rmd',
]);

function fileExt(file) {
  const parts = (file.name || '').split('.');
  if (parts.length < 2) return '';
  return parts.pop().toLowerCase();
}

function acceptedByComposer(file) {
  if (ALLOWED_IMAGE_MIMES.has(file.type)) return true;
  return COMPOSER_DATA_EXTS.has(fileExt(file));
}

// Stage a non-image data/script file by shipping it to the backend,
// which copies it into the session cwd. Shows a named chip in the
// attachment bar as confirmation. Errors go into the chat transcript.
async function stageDataFile(file) {
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.add_files_from_blobs !== 'function') {
    appendError('Restart Sift to drop files into the chat.');
    return;
  }
  // Size gate (see MAX_DRAG_DROP_BYTES at the top of this section).
  // The composer drop / paste path also goes through FileReader →
  // base64 → bridge → decode, so a multi-GB drop here would freeze
  // the chat the same way it freezes the landing page. The "+"
  // button next to the composer uses the native picker and has no
  // size limit.
  if (file.size > MAX_DRAG_DROP_BYTES) {
    appendError(
      formatDragDropOversizeReason(file, 'Use the + button instead.'),
    );
    return;
  }
  const data = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const url = reader.result;
      const comma = url.indexOf(',');
      resolve(comma >= 0 ? url.slice(comma + 1) : url);
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
  try {
    const res = await window.pywebview.api.add_files_from_blobs([
      { name: file.name, content: data, mime: file.type || '' },
    ]);
    if (!res || !res.ok) {
      appendError(friendlyAddFilesError(res && res.reason ? res.reason : 'unknown'));
      return;
    }
    if (addStagedDataNotices(res.added || [])) {
      renderAttachments();
    }
    if (res.skipped && res.skipped.length > 0) {
      appendError('Skipped: ' + res.skipped.join(', '));
    }
    if (res.skipped_existing && res.skipped_existing.length > 0) {
      // Already-in-session collisions are surfaced as their own line so
      // it's clear nothing was overwritten. Researcher can rename
      // upstream and try again.
      appendError(
        'Already in this session, not overwritten: '
        + res.skipped_existing.join(', ')
      );
    }
    if (res.policy) updatePolicyChip(res.policy);
    // Topbar pill shows the cwd path now (not the auto-derived
    // session title), so add_files no longer needs to repaint it —
    // the path doesn't change on upload. The new dataset name
    // surfaces in the sidebar row + Permission chip already.
    refreshFilesChip();
    if (typeof loadSessions === 'function') loadSessions();
    // Script attachments inflate the next request — recount so the
    // chip reflects them. The bridge reads
    // ``runner.pending_script_attachments`` directly.
    triggerContextRecount('attachment-add');
  } catch (err) {
    appendError(friendlyAddFilesError(err && err.message ? err.message : String(err)));
  }
}

// Named-chip notices for data/script files the researcher just
// added via composer drop/paste. Pure visual confirmation — the
// file is already on disk. Cleared on send or when the researcher
// removes the chip.
const stagedDataNotices = [];

function addStagedDataNotices(names) {
  /* Mirror backend-staged data/script files into the composer's
   * receipt chips. Deduplicate by basename so the native Add Files
   * button, drag/drop, and Files-popup attach all keep one visual
   * chip per staged file.
   */
  let changed = false;
  (names || []).forEach((name) => {
    if (!name || stagedDataNotices.includes(name)) return;
    stagedDataNotices.push(name);
    changed = true;
  });
  return changed;
}

// Drop handling on the compose form. Accepts images (staged as
// vision attachments) and data/script files (.csv, .dta, .rds,
// .do, .r, .log, .smcl, .gph, .rmd — copied into the session cwd).
// Landing's drop-zone handles fresh-session files on the landing
// screen; this handler is for mid-session additions.
if (form) {
  ['dragenter', 'dragover'].forEach((name) => {
    form.addEventListener(name, (e) => {
      if (!e.dataTransfer) return;
      // Check items (not files) on dragover — files is empty during
      // drag on most browsers for security reasons. items gives us
      // the MIME type but not the filename, so we only highlight
      // when at least one clearly-usable file is being dragged.
      const items = Array.from(e.dataTransfer.items || []);
      const hasUsable = items.some((it) =>
        it.kind === 'file' && ALLOWED_IMAGE_MIMES.has(it.type)
      );
      if (!hasUsable && items.length === 0) return;
      // Even if we can't confirm a usable file (items lacks ext
      // info for non-image drops), allow the drop — we'll filter
      // on the drop event where filenames are available.
      e.preventDefault();
      form.classList.add('drop-target');
    });
  });
  form.addEventListener('dragleave', (e) => {
    if (e.target === form) form.classList.remove('drop-target');
  });
  form.addEventListener('drop', async (e) => {
    const allFiles = Array.from(e.dataTransfer?.files || []);
    const usable = allFiles.filter(acceptedByComposer);
    if (usable.length === 0) return;
    e.preventDefault();
    form.classList.remove('drop-target');
    const skipped = allFiles.filter((f) => !acceptedByComposer(f));
    if (skipped.length > 0) {
      appendError(
        'Skipped: ' + skipped.map((f) => f.name).join(', ') +
        '. Only supported images, datasets, scripts, graphs, logs, and R Markdown files can be dropped here.'
      );
    }
    for (const file of usable) {
      if (ALLOWED_IMAGE_MIMES.has(file.type)) {
        // Stage for one-turn vision AND persist to the session cwd
        // so the model can @-mention or read_attached_file the
        // image on later turns. Earlier behavior was vision-only,
        // which made dropped images one-shot while native "+ Add
        // Files" persisted them — confusing inconsistency.
        //
        // Critical: only persist via stageDataFile WHEN the image
        // passed the vision cap. Earlier code unconditionally ran
        // both, so a 100 MB screenshot rejected by the 5 MB vision
        // cap STILL got FileReader-base64'd and bridge-sent under
        // the 1 GB drag-drop cap, freezing the UI on the very
        // payload the image cap was meant to refuse. Treat the
        // image cap as the floor for both paths.
        const accepted = await stageImageFile(file);
        if (accepted) {
          await stageDataFile(file);
        }
      } else {
        await stageDataFile(file);
      }
    }
    input.focus();
  });
}

// Cmd-V / Ctrl-V pasting into the composer. Images stage as vision
// attachments; data/script files (if pasted from Finder) land in
// the session cwd just like a drop.
if (input) {
  input.addEventListener('paste', async (e) => {
    const items = Array.from(e.clipboardData?.items || []);
    const fileItems = items.filter((it) => it.kind === 'file');
    if (fileItems.length === 0) return;
    const usable = [];
    for (const it of fileItems) {
      const f = it.getAsFile();
      if (f && acceptedByComposer(f)) usable.push(f);
    }
    if (usable.length === 0) return;
    e.preventDefault();
    for (const f of usable) {
      if (ALLOWED_IMAGE_MIMES.has(f.type)) {
        // Stage for vision AND persist — same dual-tracking as the
        // drop handler. The image-cap-rejection short-circuit also
        // applies here: an oversize pasted screenshot must not slip
        // through the data path's larger cap. See the drop handler
        // above for the full rationale.
        const accepted = await stageImageFile(f);
        if (accepted) {
          await stageDataFile(f);
        }
      } else {
        await stageDataFile(f);
      }
    }
  });
}

// ---- @-mention dropdown for session files --------------------------------
//
// When the researcher types "@" the composer offers a filtered list of
// every file already in this session: scripts, datasets, plots, logs.
// Selecting a row stages the file via attach_session_file (the same
// bridge endpoint the Files panel uses) and inserts "@<filename>" at
// the caret. The transcript chip + composer chip then track the file
// the same way a drag/drop attachment would.
let mentionFiles = null;
let mentionFilesFresh = false;
const mentionPopup = document.createElement('div');
mentionPopup.id = 'mention-popup';
mentionPopup.className = 'mention-popup hidden';
// ARIA listbox pattern: the popup is a listbox of selectable files,
// each row an option, and the researcher's focus never actually
// leaves the composer textarea -- selection is tracked via
// aria-activedescendant on ``input`` instead (set/cleared in
// renderMentionPopup/closeMentionPopup below), which is the standard
// way to expose a "type in one field, select from a floating list"
// widget without moving DOM focus off the textarea mid-typing.
mentionPopup.setAttribute('role', 'listbox');
mentionPopup.setAttribute('aria-label', 'Mentionable files in this session');
document.body.appendChild(mentionPopup);
let mentionState = null;
if (input) {
  input.setAttribute('aria-autocomplete', 'list');
  input.setAttribute('aria-controls', 'mention-popup');
  input.setAttribute('aria-expanded', 'false');
}

function invalidateMentionCache() {
  mentionFilesFresh = false;
}

async function ensureMentionFiles() {
  if (mentionFilesFresh && Array.isArray(mentionFiles)) return mentionFiles;
  if (!window.pywebview || !window.pywebview.api) return [];
  if (typeof window.pywebview.api.list_mentionable_files !== 'function') return [];
  try {
    const res = await window.pywebview.api.list_mentionable_files();
    if (res && res.ok && Array.isArray(res.files)) {
      mentionFiles = res.files;
      mentionFilesFresh = true;
      return mentionFiles;
    }
  } catch (err) {
    console.warn('list_mentionable_files failed', err);
  }
  mentionFiles = [];
  mentionFilesFresh = true;
  return mentionFiles;
}

function detectMentionTrigger() {
  if (!input) return null;
  const value = input.value;
  const caret = input.selectionStart;
  if (caret == null || caret !== input.selectionEnd) return null;
  let i = caret - 1;
  let scanned = 0;
  while (i >= 0) {
    const c = value[i];
    if (c === '@') {
      if (i === 0 || /\s/.test(value[i - 1])) {
        return {
          startIdx: i,
          endIdx: caret,
          query: value.slice(i + 1, caret).toLowerCase(),
        };
      }
      return null;
    }
    if (/\s/.test(c)) return null;
    scanned += 1;
    if (scanned > 64) return null;
    i -= 1;
  }
  return null;
}

function filterMentionFiles(files, query) {
  if (!query) return files.slice(0, 20);
  const scored = [];
  for (const f of files) {
    const lname = (f.name || '').toLowerCase();
    const idx = lname.indexOf(query);
    if (idx === -1) continue;
    let score = 10;
    if (idx === 0) score = 100;
    else {
      const prev = lname[idx - 1];
      if (prev === '.' || prev === '_' || prev === '-') score = 50;
    }
    score -= idx;
    scored.push({ score, f });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, 20).map((s) => s.f);
}

function mentionKindIcon(kind) {
  switch (kind) {
    case 'script': return 'S';
    case 'data':   return 'D';
    case 'graph':  return 'G';
    case 'log':    return 'L';
    default:       return '·';
  }
}

function mentionFormatBytes(n) {
  if (n == null) return '';
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return Math.round(n / 1024) + ' KB';
  return (n / (1024 * 1024)).toFixed(1) + ' MB';
}

function escapeMentionHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => (
    c === '&' ? '&amp;' :
    c === '<' ? '&lt;' :
    c === '>' ? '&gt;' :
    c === '"' ? '&quot;' : '&#39;'
  ));
}

function renderMentionPopup() {
  if (!mentionState) {
    mentionPopup.classList.add('hidden');
    if (input) {
      input.setAttribute('aria-expanded', 'false');
      input.removeAttribute('aria-activedescendant');
    }
    return;
  }
  const { items, selected } = mentionState;
  mentionPopup.innerHTML = '';
  if (!items || items.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'mention-empty';
    // A screen reader user gets no aria-activedescendant to land on
    // when there are zero options (nothing to point the composer's
    // aria-activedescendant at) -- role="status" + aria-live makes
    // "No matching files" actually announced instead of silently
    // rendered. Deliberately scoped to just this element, not the
    // whole listbox: the normal case (matches present) is already
    // communicated via aria-activedescendant on each keystroke, and
    // wrapping the whole popup in a live region would re-announce
    // the entire file list on every render instead.
    empty.setAttribute('role', 'status');
    empty.setAttribute('aria-live', 'polite');
    empty.textContent = 'No matching files in this session.';
    mentionPopup.appendChild(empty);
    if (input) input.removeAttribute('aria-activedescendant');
  } else {
    const list = document.createElement('div');
    list.className = 'mention-list';
    items.forEach((f, idx) => {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'mention-row' + (idx === selected ? ' selected' : '');
      row.dataset.idx = String(idx);
      // ARIA listbox option: id is what the composer's
      // aria-activedescendant below points at, so a screen reader
      // announces the highlighted file as arrow keys move through
      // the list without focus ever leaving the textarea.
      row.id = 'mention-option-' + idx;
      row.setAttribute('role', 'option');
      row.setAttribute('aria-selected', idx === selected ? 'true' : 'false');
      row.innerHTML = (
        '<span class="mention-icon mention-icon-' + escapeMentionHtml(f.kind || '') + '" '
          + 'aria-hidden="true">' + escapeMentionHtml(mentionKindIcon(f.kind)) + '</span>'
        + '<span class="mention-name">' + escapeMentionHtml(f.name) + '</span>'
        + '<span class="mention-meta">' + escapeMentionHtml(f.kind || '')
          + (f.size != null ? ' · ' + escapeMentionHtml(mentionFormatBytes(f.size)) : '')
          + '</span>'
      );
      row.addEventListener('mousedown', (e) => {
        e.preventDefault();
        selectMention(idx);
      });
      row.addEventListener('mouseenter', () => {
        if (!mentionState) return;
        mentionState.selected = idx;
        list.querySelectorAll('.mention-row').forEach((el, i) => {
          el.classList.toggle('selected', i === idx);
          el.setAttribute('aria-selected', i === idx ? 'true' : 'false');
        });
        if (input) input.setAttribute('aria-activedescendant', row.id);
      });
      list.appendChild(row);
    });
    mentionPopup.appendChild(list);
    const hint = document.createElement('div');
    hint.className = 'mention-hint';
    hint.textContent = '↑↓ navigate · ↵ insert · esc dismiss';
    mentionPopup.appendChild(hint);
    // Keep the selected row in view when arrow-key navigation
    // moves past the visible window. ``.mention-popup`` is the
    // scroll container (max-height 280px, overflow-y: auto), and
    // without this the highlighted row slides off the top/bottom
    // as the user holds ↓ — the visual selection just disappears.
    // ``block: 'nearest'`` scrolls only when the row is actually
    // off-screen, so rows that are already in view don't bounce.
    const selectedRow = list.querySelector('.mention-row.selected');
    if (selectedRow && typeof selectedRow.scrollIntoView === 'function') {
      selectedRow.scrollIntoView({ block: 'nearest' });
    }
    if (input) {
      input.setAttribute(
        'aria-activedescendant', 'mention-option-' + selected,
      );
    }
  }
  mentionPopup.classList.remove('hidden');
  if (input) input.setAttribute('aria-expanded', 'true');
  positionMentionPopup();
}

function positionMentionPopup() {
  if (!input) return;
  const rect = input.getBoundingClientRect();
  const width = Math.min(Math.max(rect.width, 320), 480);
  mentionPopup.style.width = width + 'px';
  mentionPopup.style.left = rect.left + 'px';
  requestAnimationFrame(() => {
    const popupH = mentionPopup.offsetHeight;
    const top = Math.max(8, rect.top - popupH - 6);
    mentionPopup.style.top = top + 'px';
  });
}

function closeMentionPopup() {
  mentionState = null;
  mentionPopup.classList.add('hidden');
  if (input) {
    input.setAttribute('aria-expanded', 'false');
    input.removeAttribute('aria-activedescendant');
  }
}

async function refreshMentionState() {
  const trigger = detectMentionTrigger();
  if (!trigger) {
    closeMentionPopup();
    return;
  }
  const files = await ensureMentionFiles();
  if (files.length === 0) {
    closeMentionPopup();
    return;
  }
  const items = filterMentionFiles(files, trigger.query);
  mentionState = {
    startIdx: trigger.startIdx,
    endIdx: trigger.endIdx,
    query: trigger.query,
    items,
    selected: 0,
  };
  renderMentionPopup();
}

async function selectMention(idx) {
  if (!mentionState) return;
  const item = mentionState.items[idx];
  if (!item) return;
  const before = input.value.slice(0, mentionState.startIdx);
  const after = input.value.slice(mentionState.endIdx);
  const token = '@' + item.name;
  input.value = before + token + after;
  const caret = before.length + token.length;
  input.selectionStart = input.selectionEnd = caret;
  autosize();
  closeMentionPopup();
  // Pass the row's path through to the bridge so basename
  // collisions (helper plots in different run dirs commonly
  // produce ``coefficients.png`` or ``marginal_effects.png``)
  // resolve to the exact row the researcher clicked, not
  // whichever copy ``iterdir`` returns first.
  await stageMentionedFile(item.name, item.path);
  input.focus();
}

async function stageMentionedFile(name, path) {
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.attach_session_file !== 'function') return;
  try {
    const res = await window.pywebview.api.attach_session_file(name, path);
    if (!res || !res.ok) {
      const reason = (res && res.reason) || 'unknown';
      toast('Could not attach: ' + reason, 'error');
      return;
    }
    if (!res.already_attached) {
      if (addStagedDataNotices([res.name || name])) renderAttachments();
    }
  } catch (err) {
    console.warn('attach_session_file failed', err);
  }
}

if (input) {
  input.addEventListener('input', () => { refreshMentionState(); });
  input.addEventListener('click', () => { refreshMentionState(); });
  input.addEventListener('keyup', (e) => {
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight'
        || e.key === 'Home' || e.key === 'End') {
      refreshMentionState();
    }
  });
  input.addEventListener('keydown', (e) => {
    if (!mentionState || mentionPopup.classList.contains('hidden')) return;
    const items = mentionState.items || [];
    const len = Math.max(items.length, 1);
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      mentionState.selected = (mentionState.selected + 1) % len;
      renderMentionPopup();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      mentionState.selected = (mentionState.selected - 1 + len) % len;
      renderMentionPopup();
    } else if (e.key === 'Enter' || e.key === 'Tab') {
      if (items.length > 0) {
        e.preventDefault();
        e.stopImmediatePropagation();
        selectMention(mentionState.selected);
      } else {
        closeMentionPopup();
      }
    } else if (e.key === 'Escape') {
      e.preventDefault();
      e.stopImmediatePropagation();
      closeMentionPopup();
    }
  });
  input.addEventListener('blur', () => {
    setTimeout(closeMentionPopup, 120);
  });
}
window.addEventListener('resize', () => {
  if (mentionState) positionMentionPopup();
});

// ---- hard reload (Cmd/Ctrl+Shift+R) --------------------------------------
//
// In-app reload (Cmd+R) re-fetches the SAME ``index.bust-<id>.html``
// URL the bridge wrote at startup, so ``style.css?v=<old-id>`` and
// ``app.js?v=<old-id>`` hit WKWebView's persistent disk cache —
// edits made while sift is running don't show up. Hard reload calls
// the bridge's ``hard_reload``, which recomputes the build-id from
// the current asset mtimes, writes a fresh ``.bust-*.html``, and
// navigates the window to its file:// URL. New URL → cache miss →
// fresh fetch → CSS/JS edits visible without quitting sift.
//
// Bound to Cmd+Shift+R (macOS convention) AND Ctrl+Shift+R (so a
// keyboard with no Cmd, or future non-mac builds, keep working).
// Both browsers and editors use this combo for "force-reload, ignore
// cache," which is exactly the semantics here.
document.addEventListener('keydown', (e) => {
  if (!e.shiftKey) return;
  if (!(e.metaKey || e.ctrlKey)) return;
  if (e.key !== 'R' && e.key !== 'r') return;
  e.preventDefault();
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.hard_reload !== 'function') return;
  // Fire-and-forget. The bridge calls load_url which navigates the
  // window away from the current page, so any logging here would be
  // racing against the navigation. The new page picks up logging on
  // its own once it loads.
  window.pywebview.api.hard_reload();
});

// ---- send-while-busy queue -----------------------------------------------
//
// When a turn is already in flight, the user can still type and Send.
// We render the user bubble immediately with a "queued" pill, snapshot
// the staged attachments, and fire the actual ``send_message`` only
// when the current turn's terminal event arrives. The runner's
// per-session ``_send_lock`` already serialises sends on the Python
// side; this queue is purely about (a) capturing the right
// attachments / images at submit time (so they go with the right
// message rather than getting folded into whichever turn is running)
// and (b) giving the researcher visible feedback that their follow-up
// landed.
//
// Stop drains the queue: cancelled queued bubbles get marked
// ``.not-sent`` and stay visible (with reduced opacity) so the
// researcher can see what they asked but didn't ship.
const pendingByCwd = new Map();

function pendingFor(cwd) {
  let q = pendingByCwd.get(cwd);
  if (!q) { q = []; pendingByCwd.set(cwd, q); }
  return q;
}

async function fireQueuedMessage(cwd, item) {
  item.userEl.classList.remove('queued');
  // ``activeLiveTurn`` is the focused-session global the Stop button
  // and ``assistant_*`` event handlers read. A queued message can
  // fire AFTER the researcher has switched sessions — if we
  // unconditionally wrote the global, a background flush would
  // steal Stop / hasVisibleReply / disposable-turn cleanup from the
  // focused session's live turn (clicking Stop would cancel the
  // wrong turn; the focused turn's "model replied" flag would land
  // on the background turn's tracking object). The local handle
  // still exists so this function's own try/catch can attach error
  // nodes to the queued user bubble; we just don't promote it to
  // the focused-only global when the queue isn't for the focus.
  const isFocused = (cwd === currentCwd);
  const localTurn = { id: null, nodes: [item.userEl], hasVisibleReply: false };
  if (isFocused) activeLiveTurn = localTurn;
  // Use the explicit-target send variants. The plain ``send_message``
  // routes to the bridge's ``self.cwd`` (focused session); a queue
  // can flush AFTER the user has switched sessions, so falling back
  // to the focused cwd would persist / execute the queued message
  // against the WRONG session. The ``_to_session`` variants route to
  // the runner whose cwd matches ``cwd``, regardless of focus.
  //
  // ``item.attachmentToken`` is the per-queued-message attachment
  // snapshot the bridge took at queue time. The token-bearing send
  // variants restore that exact snapshot into runner.pending_* just
  // before sending, so two queued messages with different staged
  // scripts each fire with their OWN attachments rather than racing
  // for whatever happens to be in the global pending lists.
  const api = window.pywebview.api;
  const supportsTokenedSend = (
    typeof api.send_message_with_token_to_session === 'function'
    && typeof api.send_message_with_images_and_token_to_session === 'function'
  );
  const supportsTargeted = (
    typeof api.send_message_to_session === 'function'
    && typeof api.send_message_with_images_to_session === 'function'
  );
  try {
    let turnId = null;
    const token = item.attachmentToken || '';
    if (item.images.length > 0) {
      const payload = item.images.map((img) => ({ data: img.data, mime: img.mime }));
      if (supportsTokenedSend && token) {
        turnId = await api.send_message_with_images_and_token_to_session(
          cwd, item.text, payload, token,
        );
      } else if (supportsTargeted) {
        turnId = await api.send_message_with_images_to_session(cwd, item.text, payload);
      } else if (typeof api.send_message_with_images === 'function') {
        // Older bridge: fall back to focused-cwd send. Cross-session
        // mix-up risk remains until the new APIs ship; the explicit
        // path above is the durable fix.
        turnId = await api.send_message_with_images(item.text, payload);
      } else {
        // The error bubble is a focused-transcript surface; only
        // surface it (and queue disposable nodes against the focused
        // global) when this flush is for the focused session.
        if (isFocused) {
          const errEl = appendError('Restart Sift to send images.');
          localTurn.nodes.push(errEl);
          queueDisposableTurn(localTurn.nodes);
          activeLiveTurn = null;
        }
        setSending(false, cwd);
        return;
      }
    } else if (supportsTokenedSend && token) {
      turnId = await api.send_message_with_token_to_session(cwd, item.text, token);
    } else if (supportsTargeted) {
      turnId = await api.send_message_to_session(cwd, item.text);
    } else {
      turnId = await api.send_message(item.text);
    }
    localTurn.id = turnId;
  } catch (err) {
    // Same focused-only routing as the no-image-support branch above.
    if (isFocused) {
      const errEl = appendError('send failed: ' + err);
      localTurn.nodes.push(errEl);
      queueDisposableTurn(localTurn.nodes);
      activeLiveTurn = null;
    }
    setSending(false, cwd);
  }
}

function flushPendingFor(cwd) {
  const q = pendingByCwd.get(cwd);
  if (!q || q.length === 0) return false;
  const next = q.shift();
  setSending(true, cwd);
  Promise.resolve().then(() => fireQueuedMessage(cwd, next));
  return true;
}

function drainPendingFor(cwd) {
  const q = pendingByCwd.get(cwd);
  if (!q || q.length === 0) return 0;
  const drained = q.length;
  const api = window.pywebview && window.pywebview.api;
  for (const item of q) {
    item.userEl.classList.remove('queued');
    item.userEl.classList.add('not-sent');
    item.userEl.title = 'Stopped before this message could send.';
    // Drop each cancelled item's frozen attachment snapshot from
    // the runner — without this, a Stop+resend cycle would leak
    // the snapshots and any attachments restored later by a
    // misrouted token would carry stale state.
    if (
      item.attachmentToken
      && api
      && typeof api.discard_pending_attachments_token === 'function'
    ) {
      try {
        api.discard_pending_attachments_token(cwd, item.attachmentToken);
      } catch (err) {
        console.warn('discard_pending_attachments_token failed', err);
      }
    }
  }
  q.length = 0;
  return drained;
}

// ----- slash command palette ----------------------------------------------
//
// Resolution logic lives in slash_commands.js (DOM-free, unit-tested
// through node — see tests/test_slash_commands.py). This object maps
// each 'ui'-kind command name to the actual panel-opening function,
// which DOES depend on the DOM and so can't live in that file.
// openDataPanel / openExport / openLedger are all defined further
// down in this same script; referencing them here is safe because
// this object is only ever consulted from inside the submit handler,
// well after the whole script (including those definitions) has run.
const SLASH_UI_ACTIONS = {
  profile: () => openDataPanel(),
  report: () => openExport(),
  privacy: () => openLedger(),
};

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const rawInputText = input.value.trim();
  const slash = window.SiftSlashCommands
    ? window.SiftSlashCommands.resolveSlashCommand(rawInputText)
    : null;
  if (slash && slash.kind === 'ui') {
    // A panel-opening command never sends a message — clear the
    // composer and run the action, same as clicking the equivalent
    // topbar chip would.
    input.value = '';
    autosize();
    const action = SLASH_UI_ACTIONS[slash.name];
    if (action) action();
    return;
  }
  const text = slash ? slash.text : rawInputText;
  const images = stagedImages.slice();  // snapshot
  // Send is allowed when ANY of {text, image, attached data file} is
  // present. The earlier guard ignored ``stagedDataNotices`` so a
  // researcher who attached a .do file and pressed Send without
  // typing got silent nothing — looked like file calling was
  // broken. With a script chip in the composer the model gets the
  // attachment as context and can proceed (the chip name itself is
  // implicit "run / inspect this").
  if (!text && images.length === 0 && stagedDataNotices.length === 0) return;
  if (!window.pywebview || !window.pywebview.api) {
    appendSystem('backend not ready yet; try again');
    return;
  }
  // Snapshot which staged-data notices are travelling with THIS
  // message (only script files actually ride along as inline context;
  // pure data files are session-resident and discoverable via
  // get_schema). Rendered as transcript chips below the user bubble
  // so the upload stays visible after send instead of disappearing
  // with the composer chip.
  const SCRIPT_EXTS_RE = /\.(py|do|r|rmd)$/i;
  const messageAttachments = stagedDataNotices.filter(
    (n) => SCRIPT_EXTS_RE.test(n)
  );
  // Snapshot image thumbnails for the user bubble. Use base64
  // data URLs (not blob:), because the form submit immediately
  // revokes the blob URLs — keeping the blob would leave the
  // bubble's thumbnails dead. data: URLs are self-contained, so
  // they survive the blob revoke at the bottom of this handler.
  const messageImages = images.map((img) => ({
    url: dataUrlFromBase64(img.data, img.mime),
    mime: img.mime,
  }));
  sweepStaleTranscriptTurns();
  const userEl = appendUser(
    text || '(image only)', messageAttachments, messageImages
  );
  input.value = '';
  closeMentionPopup();
  autosize();
  rotatePlaceholder();
  // Clear staged images from the UI immediately — the snapshot
  // carries them to the backend. Data-file notices are receipts
  // only, no payload to send; clear them at the same point so the
  // composer returns to a clean state after each turn.
  stagedImages.forEach((img) => URL.revokeObjectURL(img.url));
  stagedImages.length = 0;
  stagedDataNotices.length = 0;
  renderAttachments();

  // If a turn is already running on this session, queue the new
  // message instead of firing it. The terminal-event handler
  // (turn_done / turn_error / auth_failure) drains the queue, so
  // by the time the running turn finishes the next one fires
  // automatically. The user bubble is already in the transcript;
  // we tag it ``.queued`` so the researcher can see what's
  // pending.
  if (turnInFlight) {
    userEl.classList.add('queued');
    userEl.title = 'Queued. Will send when the current turn finishes.';
    // Freeze the runner's current pending_* lists into a per-message
    // snapshot. The bridge clears the runner's pending state, so any
    // attachments the user stages NEXT (for a later queued message)
    // land in a fresh slot. When this queued item fires below, the
    // backend will restore THIS snapshot into pending_* — closing
    // the race where the second queued send used to consume the
    // first message's script chip.
    let attachmentToken = '';
    const api = window.pywebview && window.pywebview.api;
    if (api && typeof api.freeze_pending_attachments === 'function') {
      try {
        const t = await api.freeze_pending_attachments(currentCwd);
        if (typeof t === 'string') attachmentToken = t;
      } catch (err) {
        // If freezing fails for any reason, fall through to the
        // older fire-without-token path. The race is back, but the
        // message at least sends — silent failure here would be
        // worse for the user.
        console.warn('freeze_pending_attachments failed', err);
      }
    }
    pendingFor(currentCwd).push({
      text,
      images: images.map((img) => ({ data: img.data, mime: img.mime })),
      attachments: messageAttachments,
      attachmentToken,
      userEl,
    });
    return;
  }

  // No "clear cancelled state" step: each turn has its own id, and
  // the new turn's id won't be in ``cancelledTurnIds``. Late events
  // from the previously-cancelled turn keep getting dropped because
  // they carry the OLD id; new events flow because they carry the
  // NEW id. That's the whole point of the turn-identity rewrite.
  activeLiveTurn = { id: null, nodes: [userEl], hasVisibleReply: false };
  setSending(true);
  try {
    // If images are attached, use the richer send method. The
    // simpler string send stays as the fast path for text-only.
    let turnId = null;
    if (images.length > 0 && typeof window.pywebview.api.send_message_with_images === 'function') {
      const payload = images.map((img) => ({ data: img.data, mime: img.mime }));
      turnId = await window.pywebview.api.send_message_with_images(text, payload);
    } else if (images.length > 0) {
      const errEl = appendError('Restart Sift to send images.');
      if (activeLiveTurn && !activeLiveTurn.hasVisibleReply) {
        activeLiveTurn.nodes.push(errEl);
        queueDisposableTurn(activeLiveTurn.nodes);
      }
      activeLiveTurn = null;
      setSending(false);
      return;
    } else {
      turnId = await window.pywebview.api.send_message(text);
    }
    // Capture the turn id the bridge assigned. Stop reads this to
    // mark the turn cancelled. If the await resolved with no id
    // (unexpected: the bridge returns null only on early failure
    // paths that already dispatched a turn_error), leave id null
    // and let the turn_error event clean up.
    if (activeLiveTurn) activeLiveTurn.id = turnId;
    // Don't clear setSending here — the await resolves as soon as
    // the turn is QUEUED on the Python side, not when it finishes.
    // The turn_done / turn_error / auth_failure event handler
    // below is what flips the button back on.
  } catch (err) {
    const errEl = appendError('send failed: ' + err);
    if (activeLiveTurn && !activeLiveTurn.hasVisibleReply) {
      activeLiveTurn.nodes.push(errEl);
      queueDisposableTurn(activeLiveTurn.nodes);
    }
    activeLiveTurn = null;
    setSending(false);
  }
});

// Shift-Enter inserts a newline; plain Enter sends. Sending while a
// turn is already in flight is allowed: the submit handler queues
// the new message and the terminal-event handler drains the queue.
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    form.dispatchEvent(new Event('submit'));
  }
});

// Auto-grow textarea.
input.addEventListener('input', autosize);
function autosize() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 220) + 'px';
}

// Rotating task prompts. Keep these concise and professionally useful: the
// composer should suggest valid research moves, not compete with the work.
const PLACEHOLDERS = [
  'Ask Sift about this dataset',
  'Describe your research question',
  'Profile the variables and data quality',
  'Check missingness and unusual values',
  'Define the outcome and treatment',
  'State the hypothesis to test',
  'Build a reproducible analysis',
  'Explain this result',
  'Compare groups or models',
  'Review the analysis assumptions',
  'Create a publication-ready table',
  'What should we investigate next?',
];

function rotatePlaceholder() {
  if (!input) return;
  const next = PLACEHOLDERS[Math.floor(Math.random() * PLACEHOLDERS.length)];
  input.setAttribute('placeholder', next);
}
rotatePlaceholder();

function setSending(sending, cwd) {
  // ``cwd`` defaults to the focused session — this is what
  // happens when send_message is called from the form submit.
  // Background events (terminal events for non-focused sessions)
  // pass an explicit cwd so the busy state lands on the right row.
  const target = cwd || currentCwd;
  if (target) {
    setSessionBusy(target, sending);
    // No cancelled-state clearing here. Each turn has its own id
    // and the new turn's events naturally pass the
    // ``cancelledTurnIds`` filter without needing a per-cwd reset
    // (that reset was the prior leak: it allowed late events from
    // the previous, cancelled turn to render after a fresh send
    // started). The cancelled set is only mutated by Stop, never
    // by Send.
  }
  // Composer state mirrors the FOCUSED session only.
  if (target && target !== currentCwd) return;
  turnInFlight = sending;
  // Toggle the Send / Stop icons rather than disabling the Send
  // button. During a turn, Stop replaces Send in the same spot so
  // the composer footprint doesn't reflow.
  if (sending) {
    sendBtn.classList.add('hidden');
    stopBtn.classList.remove('hidden');
    showLoadingIndicator();
  } else {
    stopBtn.classList.add('hidden');
    sendBtn.classList.remove('hidden');
    hideLoadingIndicator();
  }
  input.setAttribute('aria-busy', sending ? 'true' : 'false');
}

function setSessionBusy(cwd, busy) {
  /* Track which sessions have a turn in flight so the sidebar can
   * show a small "still working" dot on background sessions. The
   * dot is purely informational — clicking the row still works as
   * a normal focus switch and the in-flight turn keeps streaming
   * regardless. */
  if (!cwd) return;
  if (busy) busySessions.add(cwd);
  else busySessions.delete(cwd);
  // Update the sidebar row in place. Falls through if the row
  // isn't currently rendered (e.g., scrolled off in a long list);
  // the next loadSessions() will re-render with the right state.
  const rows = document.querySelectorAll('.session-item');
  rows.forEach((row) => {
    if (row.dataset && row.dataset.path === cwd) {
      row.classList.toggle('busy', busy);
    }
  });
}

function syncComposerToFocus() {
  /* Called on every focus switch. Sets the composer state to match
   * whether the now-focused session has a turn in flight. Without
   * this, switching from a busy session to an idle one would leave
   * Stop on screen, and switching back to a busy session would
   * show Send (the wrong control). */
  const focusedBusy = currentCwd && busySessions.has(currentCwd);
  setSending(!!focusedBusy, currentCwd);
}

// Rotation of vague, slightly silly labels for the loading indicator.
// Picked at random each time the indicator shows; keeps a long wait
// from feeling monotonous. Kept short so the line doesn't reflow on
// narrow windows. Explicit "doing X to Y" labels are avoided — the
// researcher doesn't need to know whether we're peeking at a schema
// or running a script; they just need to know we're still at it.
const LOADING_LABELS = [
  'facticulating',
  'triangulating',
  'cogitating',
  'ruminating',
  'percolating',
  'synthesizing',
  'noodling',
  'pondering',
  'mulling',
  'musing',
  'deliberating',
  'contemplating',
  'chewing on it',
  'connecting dots',
  'chasing threads',
  'conjuring',
  'marinating',
  'consulting the oracle',
  'squinting at it',
  'untangling',
  // Data-massage flavor — same gerund shape, vaguely statistical
  // verbs that sound like they describe what's happening to the
  // numbers without committing to any specific tool call. Bare
  // verb when the verb stands alone; object retained only where
  // the noun IS the joke (e.g. "reticulating splines").
  'massaging',
  'crunching',
  'wrangling',
  'kneading',
  'sifting',
  'whisking',
  'polishing',
  'coaxing',
  'shuffling',
  'tightening',
  'auditing',
  'interrogating',
  'sweeping',
  'tuning',
  'weighing',
];

function showLoadingIndicator() {
  /* Append a small animated indicator at the bottom of the transcript
   * while the assistant is working. Removed on any terminal event (turn_done /
   * turn_error / auth_failure). Idempotent — multiple calls in a
   * row don't stack extra spinners. */
  if (document.getElementById('loading-indicator')) return;
  const label = LOADING_LABELS[
    Math.floor(Math.random() * LOADING_LABELS.length)
  ];
  const el = document.createElement('div');
  el.id = 'loading-indicator';
  el.className = 'loading-indicator';
  el.setAttribute('aria-label', 'Sift is ' + label);
  // A lightweight CSS-only indicator keeps startup and analysis status
  // provider-neutral, offline, and consistent across native renderers.
  el.innerHTML =
    '<span class="thinking-dots" aria-hidden="true">' +
    '<span></span><span></span><span></span></span>' +
    '<span class="loading-text">' + label + '</span>';
  messagesEl.appendChild(el);
  scrollToBottom();
}

function hideLoadingIndicator() {
  const el = document.getElementById('loading-indicator');
  if (el) el.remove();
}

// Stop button — asks the bridge to cancel the in-flight turn AND
// always returns the UI to a clean state. The bridge cancels the
// asyncio task and tears down the SDK client so no half-finished
// request leaks into the next turn. We used to wait for the
// turn_error event to clear setSending, but the provider stream can
// in rare cases close without yielding any terminal event (network
// blip, SDK internal hiccup); the JS then stayed stuck on "sending"
// forever and the bridge said "no turn in flight." Hard-resetting
// here means Stop is always a reliable recovery button: researchers
// can always get the composer back.
if (stopBtn) {
  stopBtn.addEventListener('click', async () => {
    if (!window.pywebview || !window.pywebview.api) return;
    stopBtn.disabled = true;
    // Mark this turn's events as suppressed BEFORE awaiting the
    // bridge cancel. Two ids go into ``cancelledTurnIds``, belt and
    // suspenders:
    //   1. ``activeLiveTurn?.id`` — the id we captured when the send
    //      Promise resolved. Available unless Stop fires in the
    //      tiny window between Send and the await returning.
    //   2. The id the bridge returns from ``interrupt_turn``. The
    //      bridge knows which turn it just cancelled and surfaces
    //      the id explicitly. Covers the race above.
    // Either path alone would usually be enough; together they
    // guarantee the JS-side filter has the cancelled id no matter
    // when Stop fires relative to the send Promise.
    if (activeLiveTurn) markTurnCancelled(activeLiveTurn.id);
    // The terminal turn_error that the runner emits is now also
    // dropped (see ``sift_event`` for why), so handle the activeLiveTurn
    // cleanup the terminal handler used to do. If the cancelled
    // turn produced no visible reply (the model hadn't written
    // anything when Stop fired), queue the user-bubble nodes for
    // the staleness sweep so they don't accumulate.
    if (activeLiveTurn && !activeLiveTurn.hasVisibleReply) {
      queueDisposableTurn(activeLiveTurn.nodes);
    }
    activeLiveTurn = null;
    // Stop = "stop everything for this session": cancel the
    // running turn AND drain any queued follow-ups. Cancelled
    // queued messages get marked ``.not-sent`` so the researcher
    // can see what they typed but didn't ship; silently removing
    // their text would be hostile UX.
    drainPendingFor(currentCwd);
    if (currentCwd) {
      triggerContextRecount('stop');
    }
    // Visible acknowledgement: the cancellation cascades through
    // the SDK + the subprocess kill, which can take a beat. Without
    // this toast, a researcher who pressed Stop and immediately
    // resent the same prompt would see the new turn queue behind
    // the cancellation cleanup and assume "Stop did nothing".
    toast('Stopping…', 'info');
    try {
      const res = await window.pywebview.api.interrupt_turn();
      // Belt-and-suspenders: if the bridge tells us which turn id
      // it cancelled, add that to the drop set too. Covers the
      // case where ``activeLiveTurn`` was null at click time
      // (e.g., Stop fired between fireQueuedMessage clearing the
      // turn and the next message starting).
      if (res && res.turn_id) markTurnCancelled(res.turn_id);
    } catch (_) {
      // swallow: the bridge may have nothing to cancel; we still
      // want to clear the UI state below.
    } finally {
      // Always restore the composer regardless of what the bridge
      // said. If a terminal event arrives after this, it's a no-op
      // (setSending(false) is idempotent); if none arrives, the
      // researcher isn't trapped.
      setSending(false);
      stopBtn.disabled = false;
    }
  });
}

// ----- Python → JS event handler -----------------------------------------

window.sift_event = function (evt) {
  // evt is a plain object; {type} + type-specific fields.
  // Every event from a runner carries ``session_cwd``: the cwd of
  // the runner that emitted it. ``ready`` and ``policy_updated``
  // are bridge-level events without a session_cwd — those always
  // apply to the focused session and pass through.
  const evtCwd = evt.session_cwd;
  const isFocused = !evtCwd || (currentCwd && evtCwd === currentCwd);

  // Drop late events from any turn the researcher cancelled.
  //
  // Why drop ALL events from the cancelled turn (not "all except the
  // terminal"): cancellation is async at multiple levels — the
  // asyncio task gets a cancel signal, the provider stream still
  // has buffered tokens, and pywebview's evaluate_js queue has a
  // tail of events from before the cancel. Earlier behavior keyed
  // suppression on the cwd, then cleared it the moment a new
  // message started — late events from the previous turn then
  // slipped through and rendered after the new turn began. The
  // turn-id key fixes that: each turn has a unique id, the
  // cancelled set is never cleared by Send, and a new turn's
  // events naturally pass because they carry a fresh id.
  //
  // The backend dispatcher (``_dispatch_event`` in ui.py) applies
  // the same drop authoritatively — this filter is best-effort
  // defense in depth. If both layers ever disagree, the backend
  // wins (events that don't reach JS were never persisted).
  if (evt.turn_id && cancelledTurnIds.has(evt.turn_id)) {
    return;
  }

  switch (evt.type) {
    case 'ready':
      showChat(evt);
      break;
    case 'assistant_text':
      if (!isFocused) return;
      if (activeLiveTurn) activeLiveTurn.hasVisibleReply = true;
      appendAssistant(evt.text);
      break;
    case 'assistant_thinking':
      if (!isFocused) return;
      if (activeLiveTurn) activeLiveTurn.hasVisibleReply = true;
      appendThinking(evt.text);
      break;
    case 'tool_call': {
      if (!isFocused) return;
      const card = appendToolCall(evt);
      if (card && activeLiveTurn) activeLiveTurn.hasVisibleReply = true;
      break;
    }
    case 'tool_result': {
      if (!isFocused) return;
      const card = appendToolResult(evt);
      if (card && activeLiveTurn) activeLiveTurn.hasVisibleReply = true;
      // Refresh the Files panel only when the tool that just finished
      // could plausibly have written to disk. ``run_dir`` is populated
      // by the provider layer iff this result came from
      // ``submit_script`` / ``submit_script_file`` — the only tools
      // that stage a script, log, or plot. Plumbing tools
      // (``get_schema``, ``expand_result``, ``list_results``,
      // ``request_data``) leave it ``null`` and don't need a refresh.
      //
      // The cost matters: ``list_session_files`` walks every run
      // dir under ``.sift/runs``, stats each ``script.{do,R,py}``,
      // recurses into every ``_sift_plots/`` subdir, and base64-
      // encodes thumbnails up to 3 MB each. In a long session that
      // adds up fast, and firing it on every plumbing tool turned
      // the chat loop into a steady drip of heavy IPC.
      if (evt.run_dir) refreshFilesChip();
      break;
    }
    case 'turn_done':
      // Terminal event: clear busy state for THIS session
      // (whether focused or background) and, if focused, refresh
      // the composer + context chip.
      //
      // The chip tracks "context occupied AFTER this turn" — i.e.,
      // the prompt this turn loaded PLUS the response that just
      // landed. Each provider computes ``post_turn_tokens`` from its
      // own usage fields (Anthropic sums input + cache_read +
      // cache_creation + output; OpenAI sums input + output because
      // its input_tokens already covers the cached prefix), so the
      // chip just renders that number. Older sessions persisted
      // before this field existed fall back to the legacy sum below.
      if (isFocused) {
        // Refresh the chip from the pre-flight counter — single
        // source of truth. The provider's ``post_turn_tokens`` is
        // post-billing reality, useful for diagnostics but not the
        // right number for "what's the next request going to weigh"
        // (which is what the chip predicts). Logged below for any
        // debug consumer that still wants the post-hoc value.
        if (typeof evt.post_turn_tokens === 'number') {
          // Diagnostic only; not on the chip.
        }
        triggerContextRecount('turn_done');
        refreshLedgerChip();
        refreshCheckpointsChip();
        if (activeLiveTurn && !activeLiveTurn.hasVisibleReply) {
          queueDisposableTurn(activeLiveTurn.nodes);
        }
        activeLiveTurn = null;
      }
      // ``flushPendingFor`` returns true iff a queued message just
      // fired. In that case ``setSending(true, evtCwd)`` was
      // re-asserted inside, so we leave the composer in the busy
      // state and DON'T flip back to Send.
      if (!flushPendingFor(evtCwd)) {
        setSending(false, evtCwd);
      }
      // hideLoadingIndicator() (inside setSending(false)) removed
      // the cat above the new assistant reply, so the reply just
      // shifted up by ~80 px. Re-apply the top-anchor so the
      // researcher still lands on the answer's first line.
      reapplyAssistantTopAnchor();
      // Refresh the sidebar so the just-active session bubbles to
      // the top — list_sessions sorts by chat_history.jsonl mtime,
      // which the persist path bumped during this turn.
      if (typeof loadSessions === 'function') loadSessions();
      break;
    case 'auth_failure':
      // Auth failures matter cross-session: even a background
      // turn that hits an auth error should drop its busy dot.
      // Render the error bubble only into the focused transcript
      // (the message is in the persisted log; switching to that
      // session will replay it). Drain any queued follow-ups for
      // this session: if auth is broken, queueing them up to fail
      // one after another is just noise.
      if (isFocused) {
        const errEl = appendError('Auth failure: ' + (evt.reason || 'unknown'));
        if (activeLiveTurn && !activeLiveTurn.hasVisibleReply) {
          activeLiveTurn.nodes.push(errEl);
          queueDisposableTurn(activeLiveTurn.nodes);
        }
        activeLiveTurn = null;
      }
      drainPendingFor(evtCwd);
      setSending(false, evtCwd);
      if (evtCwd === currentCwd) triggerContextRecount('turn_settled');
      break;
    case 'turn_error':
      if (isFocused) {
        const errEl = appendError(evt.message || 'unknown error');
        if (activeLiveTurn && !activeLiveTurn.hasVisibleReply) {
          activeLiveTurn.nodes.push(errEl);
          queueDisposableTurn(activeLiveTurn.nodes);
        }
        activeLiveTurn = null;
      }
      // For ordinary turn errors (e.g., model returned a tool-use
      // error), drain the queue: the researcher's follow-ups were
      // probably reasoning-conditioned on the previous turn
      // succeeding, so firing them blindly is worse than asking
      // them to retry.
      drainPendingFor(evtCwd);
      setSending(false, evtCwd);
      if (evtCwd === currentCwd) triggerContextRecount('turn_settled');
      break;
    case 'policy_updated':
      updatePolicyChip(evt.policy);
      break;
    case 'install_confirmation_request':
      // Hard consent gate for the install_packages tool. The Python
      // handler is blocked awaiting a response keyed by ``evt.token``;
      // a missing emitter would have made it deny immediately, so the
      // request only arrives here when a real install is pending.
      // Show the modal regardless of focused vs background cwd: the
      // install affects the researcher's machine globally, so the
      // confirmation belongs in front of them right now.
      showInstallConfirmationModal(evt);
      break;
    default:
      console.warn('unknown event type', evt);
  }
};

// ----- message rendering --------------------------------------------------

function appendUser(text, attachments, images) {
  /* ``attachments`` is an optional array of filenames that traveled
   * with this message (e.g. dragged-in .py / .do scripts).
   * ``images`` is an optional array of ``{url, mime}`` data-URL
   * thumbnails. Both render in the transcript so the upload event
   * is visible permanently — not just as ephemeral composer chips
   * that clear on send. */
  return append('user', text, /*markdown=*/ false, attachments || [], images || []);
}

function appendAssistant(text) {
  return append('assistant', text || '', /*markdown=*/ true);
}

function appendThinking(text) {
  /* Render a provider-supplied reasoning trace as a collapsible card so the
   * researcher can inspect it without the trace
   * dominating the transcript. Starts collapsed; clicking the
   * header expands. Mirrors the submit_script card shape so the
   * interaction model is consistent. */
  setWelcomeOnlyMode(false);
  const card = document.createElement('div');
  card.className = 'thinking-card collapsed';

  const header = document.createElement('div');
  header.className = 'thinking-header';
  const arrow = document.createElement('span');
  arrow.className = 'thinking-arrow';
  arrow.textContent = '▼';
  const label = document.createElement('span');
  label.className = 'thinking-label';
  label.textContent = 'Thinking';
  header.appendChild(arrow);
  header.appendChild(label);

  const body = document.createElement('div');
  body.className = 'thinking-body';
  body.textContent = text;

  header.addEventListener('click', () => card.classList.toggle('collapsed'));

  card.appendChild(header);
  card.appendChild(body);
  messagesEl.appendChild(card);
  scrollToBottom();
  return card;
}

function appendSystem(text) {
  return append('system', text);
}

function appendError(text) {
  return append('error', text);
}

function append(kind, text, markdown, attachments, images) {
  setWelcomeOnlyMode(false);
  const wrapper = document.createElement('div');
  wrapper.className = 'message ' + kind;
  // Stash the original (pre-render) text on the wrapper so the
  // per-bubble copy button can hand the markdown source — not the
  // rendered HTML — to the clipboard. Researchers paste these into
  // other chats / editors and want the literal text they sent or
  // received, not a transformed version with HTML entities decoded
  // / list bullets converted to glyphs / etc. Stored as a property
  // (not a dataset attribute) to avoid a large stringify on every
  // long assistant turn.
  wrapper.__siftRawText = text || '';
  // Image thumbnails render ABOVE the bubble — same vertical order
  // they appeared in the composer, easier to scan. Clicking opens
  // the full-resolution image in a new browser tab so the
  // researcher can inspect details (axis labels, fine print, etc.)
  // that don't survive the chat-width thumbnail size.
  if (images && images.length > 0) {
    const row = document.createElement('div');
    row.className = 'message-images';
    images.forEach((img, idx) => {
      const thumb = document.createElement('img');
      thumb.src = img.url || '';
      thumb.alt = `Attached image ${idx + 1}`;
      thumb.className = 'message-image-thumb';
      thumb.title = 'Click to view full size';
      thumb.addEventListener('click', () => {
        if (img.url) showImageLightbox(img.url);
      });
      makeKeyboardClickable(thumb, 'View attached image ' + (idx + 1) + ' full size');
      row.appendChild(thumb);
    });
    wrapper.appendChild(row);
  }
  const body = document.createElement('div');
  body.className = 'message-body';
  if (markdown && window.SiftMarkdown) {
    body.innerHTML = window.SiftMarkdown.render(text);
  } else {
    body.textContent = text;
  }
  wrapper.appendChild(body);
  // Copy button — only on user and assistant bubbles. System / error
  // / thinking messages don't get one: system + error are rare status
  // lines (clipboard isn't useful), and thinking traces are already
  // in a collapsible card with their own controls. The button hides
  // by default and reveals on bubble hover via the ``.message-actions``
  // / ``.message:hover .message-actions`` rules in style.css; on
  // touch / no-hover devices the action row is always visible (a
  // matching ``@media (hover: none)`` rule keeps it on those clients).
  if (kind === 'user' || kind === 'assistant') {
    const actions = document.createElement('div');
    actions.className = 'message-actions';
    const copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.className = 'message-action-btn message-copy-btn';
    copyBtn.title = 'Copy message to clipboard';
    copyBtn.setAttribute('aria-label', 'Copy message');
    // Inline SVG so the icon doesn't depend on a font load. Same
    // copy glyph as the Files-panel action; see ``iconSvg``.
    copyBtn.innerHTML = iconSvg('copy');
    copyBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      copyMessageBubble(wrapper, copyBtn);
      // Drop focus so the actions row hides when the cursor leaves
      // the message — without this, the button keeps focus and any
      // ``:focus-within``-style rule would pin the row visible after
      // a mouse click.
      copyBtn.blur();
    });
    actions.appendChild(copyBtn);
    // Edit button — user bubbles only. Provenance boundary: editing
    // an assistant bubble would let the researcher put words in
    // the model's mouth, then have the chat continue from there as
    // if the model had said them. That's a shape we deliberately
    // refuse. User-message edit is the supported "revisit and try
    // a different question" affordance; the rewind path truncates
    // the chat at THIS user message, hides results from the dropped
    // branch in the store, and re-fires the bubble's new text as a
    // fresh turn with a warm-start replay over the truncated log.
    if (kind === 'user') {
      const editBtn = document.createElement('button');
      editBtn.type = 'button';
      editBtn.className = 'message-action-btn message-edit-btn';
      editBtn.title = 'Edit and re-run from this message';
      editBtn.setAttribute('aria-label', 'Edit message');
      // Pencil glyph — same stroke style as the copy icon for
      // visual rhythm in the actions row.
      editBtn.innerHTML = iconSvg('edit');
      editBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        enterEditMode(wrapper);
      });
      actions.appendChild(editBtn);
    }
    wrapper.appendChild(actions);
  }
  // Attachment chips render as a small row beneath the user bubble
  // (only shown when the caller passes a non-empty list). This is
  // the "I uploaded a script and Sift can see it" affordance the
  // composer chip can't be (composer chips clear on send).
  if (attachments && attachments.length > 0) {
    const row = document.createElement('div');
    row.className = 'message-attachments';
    attachments.forEach((name) => {
      const chip = document.createElement('span');
      chip.className = 'message-attachment-chip';
      chip.textContent = '📎 ' + name;
      chip.title = name + '. Sent with this message.';
      row.appendChild(chip);
    });
    wrapper.appendChild(row);
  }
  messagesEl.appendChild(wrapper);
  if (kind === 'assistant') {
    // Anchor on the user's own preceding message so the researcher
    // sees what they asked at the top of the viewport, then the
    // reply right below. Scrolling to the assistant bubble's top
    // landed them on the second line of the answer and dropped
    // their own question off-screen — confusing for "what did I
    // just ask?" review. Falls back to top-anchoring the reply
    // itself if no preceding user message exists (the first turn
    // of a session that opens with an assistant greeting). Other
    // message kinds (user, system, error) still pin to the
    // bottom: a user message pairs with the empty composer below
    // it, and system / error notices are usually short status
    // lines.
    //
    // Remember the anchor element so the turn_done handler can
    // re-apply the scroll AFTER the loading indicator is removed.
    // Without that re-anchor, hiding the cat shifts everything up
    // by ~80px and the chosen anchor scrolls off the top.
    const anchor = findPrecedingUserMessage(wrapper) || wrapper;
    pendingAssistantTopAnchor = anchor;
    scrollMessageToTop(anchor);
  } else {
    if (kind === 'user') pendingAssistantTopAnchor = null;
    scrollToBottom();
  }
  return wrapper;
}

function findPrecedingUserMessage(wrapper) {
  /* Walk backwards through ``messagesEl`` siblings looking for the
   * nearest user bubble that DROVE this reply. Skip ``.queued`` and
   * ``.not-sent`` user bubbles: those sit in the DOM ahead of the
   * reply (queued ones haven't fired yet; cancelled ones never will),
   * so DOM proximity would otherwise mis-anchor the reply under a
   * later prompt that the model never saw. Example: A is in flight,
   * the researcher queues B, A's reply arrives — without the filter
   * the reply would anchor under B's prompt, not A's.
   *
   * Returns ``null`` when nothing matches (rare — only at session
   * start before the researcher has typed anything, or when the
   * provider emits an assistant message without a preceding user
   * turn).
   */
  let prev = wrapper.previousElementSibling;
  while (prev) {
    if (
      prev.classList
      && prev.classList.contains('user')
      && !prev.classList.contains('queued')
      && !prev.classList.contains('not-sent')
    ) {
      return prev;
    }
    prev = prev.previousElementSibling;
  }
  return null;
}

// Set when we top-align a new assistant reply; the turn_done handler
// re-applies the same top-align after hideLoadingIndicator() removes
// the cat (which would otherwise shift the reply up off the viewport).
let pendingAssistantTopAnchor = null;

function reapplyAssistantTopAnchor() {
  if (!pendingAssistantTopAnchor) return;
  const wrapper = pendingAssistantTopAnchor;
  pendingAssistantTopAnchor = null;
  // Defer one frame so layout has settled after the indicator removal.
  requestAnimationFrame(() => scrollMessageToTop(wrapper));
}

function scrollMessageToTop(wrapper) {
  /* Scroll ``messagesEl`` so ``wrapper``'s top edge sits near the
   * top of the visible chat region. Uses offset arithmetic instead
   * of ``scrollIntoView`` because the messages container is the
   * scroll parent (not the document) and ``scrollIntoView`` on a
   * descendant can scroll the WHOLE page in some WebKit builds.
   *
   * A small breathing-room offset (16 px) keeps the message from
   * kissing the topbar; if the wrapper sits very close to the
   * bottom (short tail) the clamp prevents an over-scroll that
   * would leave the wrapper not actually at the top.
   */
  if (!messagesEl || !wrapper) return;
  const breathingRoom = 16;
  const target = Math.max(0, wrapper.offsetTop - breathingRoom);
  const max = Math.max(0, messagesEl.scrollHeight - messagesEl.clientHeight);
  messagesEl.scrollTop = Math.min(target, max);
}


function renderAnalysisPlan(steps) {
  /* Live analysis-plan checklist. The model replaces the whole plan
   * on every update_analysis_plan call; we keep ONE card and update
   * it in place so the transcript doesn't fill with stale copies.
   * The card is created at the point in the conversation where the
   * plan first appeared. Titles arrive backend-sanitized; they are
   * still inserted via textContent, never innerHTML. */
  if (!Array.isArray(steps) || steps.length === 0) return;
  let card = document.getElementById('analysis-plan-card');
  if (!card) {
    card = document.createElement('div');
    card.id = 'analysis-plan-card';
    card.className = 'tool-card analysis-plan-card';
    const header = document.createElement('div');
    header.className = 'tool-header';
    header.innerHTML = '<span class="tool-name">Analysis plan</span>';
    const body = document.createElement('div');
    body.className = 'tool-body plan-body';
    header.addEventListener('click', () =>
      card.classList.toggle('collapsed'));
    card.appendChild(header);
    card.appendChild(body);
    messagesEl.appendChild(card);
  }
  const body = card.querySelector('.plan-body');
  body.innerHTML = '';
  const icons = { done: '✓', active: '●', pending: '○', skipped: '–' };
  steps.slice(0, 20).forEach((s) => {
    if (!s || typeof s !== 'object') return;
    const row = document.createElement('div');
    const status = icons[s.status] !== undefined ? s.status : 'pending';
    row.className = 'plan-step plan-step-' + status;
    const mark = document.createElement('span');
    mark.className = 'plan-mark';
    mark.textContent = icons[status];
    const title = document.createElement('span');
    title.className = 'plan-title';
    title.textContent = String(s.title || '');
    row.appendChild(mark);
    row.appendChild(title);
    body.appendChild(row);
  });
  scrollToBottom();
}

function appendToolCall(evt) {
  // Only ``submit_script`` and ``submit_script_file`` render cards.
  // ``get_schema``, ``request_data``, ``expand_result``,
  // ``list_results`` etc. happen silently — they're plumbing, not
  // results the researcher reads. The assistant summarizes whatever
  // matters from them in the chat text that follows.
  setWelcomeOnlyMode(false);
  const shortName = shortenToolName(evt.name);
  if (shortName === 'update_analysis_plan') {
    renderAnalysisPlan(evt.input && evt.input.steps);
    return;
  }
  const isSubmitScript = shortName === 'submit_script';
  const isSubmitScriptFile = shortName === 'submit_script_file';
  if (!isSubmitScript && !isSubmitScriptFile) return;

  const card = document.createElement('div');
  card.dataset.callId = evt.call_id;
  card.className = 'tool-card';

  const header = document.createElement('div');
  header.className = 'tool-header';
  const arrow = document.createElement('span');
  arrow.className = 'tool-arrow';
  arrow.textContent = '▼';
  const title = document.createElement('span');
  title.innerHTML =
    '<span class="tool-name">' + shortName + '</span>' +
    ' <span class="tool-status">running…</span>';
  header.appendChild(arrow);
  header.appendChild(title);

  const body = document.createElement('div');
  body.className = 'tool-body';

  const input = evt.input || {};
  if (isSubmitScript) {
    // Render the language + code + label prominently, not as JSON
    // stringification. Researcher sees the actual script.
    const langText = (input.language || '').toString().toUpperCase();
    if (input.label) {
      const label = document.createElement('div');
      label.className = 'tool-label';
      label.textContent = input.label;
      body.appendChild(label);
    }
    const pre = document.createElement('pre');
    pre.className = 'tool-code lang-' + (input.language || 'text').toLowerCase();
    // Small language badge inside the code block so the researcher
    // knows whether this is R or Stata at a glance.
    const badge = document.createElement('span');
    badge.className = 'tool-lang-badge';
    badge.textContent = langText || 'script';
    pre.appendChild(badge);
    const codeEl = document.createElement('code');
    codeEl.textContent = input.code || '';
    pre.appendChild(codeEl);
    body.appendChild(pre);
    const sources = Array.isArray(input.source_datasets) && input.source_datasets.length
      ? input.source_datasets : (input.source_dataset ? [input.source_dataset] : []);
    if (sources.length) {
      const src = document.createElement('div');
      src.className = 'tool-source';
      src.textContent = 'source' + (sources.length > 1 ? 's: ' : ': ') + sources.join(' + ');
      body.appendChild(src);
    }
  } else if (isSubmitScriptFile) {
    // The script bytes don't ride in the tool input — the file is
    // already on disk in cwd. Render filename + label + language so
    // the researcher recognises which attachment is being run.
    const langText = (input.language || '').toString().toUpperCase();
    if (input.label) {
      const label = document.createElement('div');
      label.className = 'tool-label';
      label.textContent = input.label;
      body.appendChild(label);
    }
    const pre = document.createElement('pre');
    pre.className = 'tool-code lang-' + (input.language || 'text').toLowerCase();
    const badge = document.createElement('span');
    badge.className = 'tool-lang-badge';
    badge.textContent = langText || 'script';
    pre.appendChild(badge);
    const fileEl = document.createElement('code');
    fileEl.textContent = 'running ' + (input.name || '(unnamed file)');
    pre.appendChild(fileEl);
    body.appendChild(pre);
    const sources = Array.isArray(input.source_datasets) && input.source_datasets.length
      ? input.source_datasets : (input.source_dataset ? [input.source_dataset] : []);
    if (sources.length) {
      const src = document.createElement('div');
      src.className = 'tool-source';
      src.textContent = 'source' + (sources.length > 1 ? 's: ' : ': ') + sources.join(' + ');
      body.appendChild(src);
    }
  }

  header.addEventListener('click', () => card.classList.toggle('collapsed'));

  card.appendChild(header);
  card.appendChild(body);
  messagesEl.appendChild(card);
  scrollToBottom();
  return card;
}

function appendToolResult(evt) {
  setWelcomeOnlyMode(false);
  // Non-submit_script tool calls don't create cards in
  // appendToolCall, so there's nothing to update here — silent pass.
  const existingCard = [...messagesEl.querySelectorAll('.tool-card')]
    .find((c) => c.dataset.callId === evt.call_id);
  if (!existingCard) return;

  if (evt.is_error) existingCard.classList.add('error');
  const statusEl = existingCard.querySelector('.tool-status');
  if (statusEl) statusEl.textContent = evt.is_error ? 'error' : 'done';
  const body = existingCard.querySelector('.tool-body');

  // submit_script only: show the native R/Stata result (post-
  // preamble) inline, plus the Open-in / Show-folder buttons.
  // Errors are not surfaced on the card — the assistant reply
  // explains what went wrong. The sanitized payload is not shown;
  // researchers who want it can ask the assistant.
  renderScriptResultInline(body, evt);
  scrollToBottom();
  return existingCard;
}

// Marker emitted by the Stata preamble in executor.py. Everything
// above this line in stdout is plumbing (adopath, cd, comments the
// researcher didn't write); everything below is the researcher's
// actual script output — the regression table, summary stats, etc.
// R has no equivalent marker because ``source(sift.R)`` runs
// silently, so R stdout is already clean.
const STATA_PREAMBLE_MARKER =
  'Sift preamble above; researcher code below';

function stripPreamble(stdout, _language) {
  /* Split on the Stata preamble marker and return everything after
   * it. The ``_language`` argument is kept for signature stability,
   * but the decision is marker-based, not language-based: error
   * paths in tools.py (execution_failed / rejected_by_sanitizer)
   * don't set ``_language``, so gating on language==='Stata' would
   * leak the preamble on every failed run. Since the marker is only
   * injected into Stata stdout, R output is unaffected either way. */
  if (!stdout) return '';
  const idx = stdout.indexOf(STATA_PREAMBLE_MARKER);
  if (idx < 0) return stdout;
  // Marker sits inside a `.*! ----- ... -----` comment line. Jump
  // past the end of that line so the caller sees clean output from
  // the first researcher command onwards.
  const eol = stdout.indexOf('\n', idx);
  return eol < 0 ? '' : stdout.slice(eol + 1);
}

function renderScriptResultInline(body, evt) {
  /* Appends to the submit_script tool-body:
   *   1. Sift-rendered canonical result tables, when the tool
   *      result envelope carries them (one per ok-status entry's
   *      ``markdown`` field). Product output, not model prose —
   *      the same payload renders identically across recalls.
   *   2. Native script output (post-preamble) — inline, visible.
   *      The Stata regression table, the R summary, whatever the
   *      script actually printed.
   *   3. Action buttons row: [Open in R/Stata] [Show folder].
   *
   * The model still interprets the result in chat; table SHAPE
   * (column choice, p-value column, precision) is enforced here.
   */
  renderCanonicalResultTables(body, evt);

  const nativeStdout = stripPreamble(evt.raw_stdout || '', evt.language).trim();
  if (nativeStdout) {
    // Collapsed by default. The script + the canonical regression
    // tables above already tell the researcher what they need; the
    // raw R/Stata/Python log is for "let me audit" moments. Same
    // disclosure pattern as the multi-result panel.
    const details = document.createElement('details');
    details.className = 'tool-output-collapsed';
    const summary = document.createElement('summary');
    summary.className = 'tool-output-summary';
    const lang = evt.language || 'script';
    summary.textContent = `${lang} output (click to expand)`;
    details.appendChild(summary);
    const pre = document.createElement('pre');
    pre.className = 'tool-output';
    pre.textContent = nativeStdout;
    details.appendChild(pre);
    body.appendChild(details);
  }

  // Plot-helper diagnostic — surfaced when a helper was clearly
  // called (the run-dir has a ``_sift_plots/`` subdir + stderr
  // mentions ``sift.plot_*``) but no plot files actually landed.
  // Without this note, the researcher sees an empty thumbnail row
  // and has no signal about why. Most common cause: matplotlib
  // not installed in the Python environment.
  if (evt.plot_diagnostic) {
    const note = document.createElement('div');
    note.className = 'tool-plot-diagnostic';
    note.textContent = evt.plot_diagnostic;
    body.appendChild(note);
  }

  // Inline plot thumbnails — every .png the script wrote into its
  // run dir (including those produced by `graph export` in Stata,
  // `ggsave` in R, `plt.savefig` in Python). These are the
  // RESEARCHER's view; the model only ever sees plots that came
  // through the manifest-allowlist gate in the runner.
  if (evt.plots && Array.isArray(evt.plots) && evt.plots.length > 0) {
    const grid = document.createElement('div');
    grid.className = 'tool-plots';
    evt.plots.forEach((plot) => {
      const tile = document.createElement('div');
      tile.className = 'tool-plot-tile';
      tile.title = plot.name;
      if (plot.data) {
        const img = document.createElement('img');
        img.alt = plot.name;
        img.src = `data:${plot.mime || 'image/png'};base64,${plot.data}`;
        img.addEventListener('click', () => showImageLightbox(img.src));
        makeKeyboardClickable(img, 'View plot ' + (plot.name || '') + ' full size');
        tile.appendChild(img);
      } else {
        // Above the inline byte cap — render a placeholder with the
        // file size so the researcher knows it exists, plus an
        // "Open" button that hands off to the OS image viewer.
        const placeholder = document.createElement('div');
        placeholder.className = 'tool-plot-placeholder';
        placeholder.textContent = formatBytes(plot.size || 0);
        tile.appendChild(placeholder);
        if (plot.path && window.pywebview && window.pywebview.api &&
            typeof window.pywebview.api.open_path === 'function') {
          tile.style.cursor = 'pointer';
          tile.addEventListener('click', () => {
            window.pywebview.api.open_path(plot.path);
          });
          makeKeyboardClickable(tile, 'Open ' + (plot.name || 'plot') + ' in image viewer');
        }
      }
      const caption = document.createElement('div');
      caption.className = 'tool-plot-caption';
      caption.textContent = plot.name;
      tile.appendChild(caption);
      grid.appendChild(tile);
    });
    body.appendChild(grid);
  }

  if (evt.run_dir) {
    const actions = document.createElement('div');
    actions.className = 'tool-actions';

    const lang = evt.language;  // "R" | "Stata" | "Python" | undefined
    const scriptFile =
      lang === 'Stata' ? 'script.do'
        : lang === 'Python' ? 'script.py'
        : 'script.R';
    const openInLabel =
      lang === 'Stata' ? 'Open in Stata'
        : lang === 'R' ? 'Open in R'
        : lang === 'Python' ? 'Open in Python'
        : 'Open script';
    const openMode =
      lang === 'Stata' ? 'run_stata'
        : lang === 'R' ? 'run_r'
        : lang === 'Python' ? 'run_python'
        : null;

    const openScriptBtn = document.createElement('button');
    openScriptBtn.type = 'button';
    openScriptBtn.className = 'tool-action';
    openScriptBtn.textContent = openInLabel;
    openScriptBtn.title =
      lang === 'Stata'
        ? 'Launch Stata with the script loaded.'
        : lang === 'R'
        ? 'Launch RStudio with the script loaded.'
        : lang === 'Python'
        ? 'Open the Python script in an editor without running it.'
        : 'Open the script in its default app.';
    openScriptBtn.addEventListener('click', () => {
      const primary = evt.run_dir + '/' + scriptFile;
      const fallback = evt.run_dir + '/' + (scriptFile === 'script.R' ? 'script.do' : 'script.R');
      openInNativeApp(primary, openScriptBtn, fallback, openMode);
    });
    actions.appendChild(openScriptBtn);

    const openFolderBtn = document.createElement('button');
    openFolderBtn.type = 'button';
    openFolderBtn.className = 'tool-action';
    openFolderBtn.textContent = 'Show folder';
    openFolderBtn.title = `Reveal the run directory in ${nativeFileManager}.`;
    openFolderBtn.addEventListener('click', () =>
      openInNativeApp(evt.run_dir, openFolderBtn, null, null)
    );
    actions.appendChild(openFolderBtn);

    body.appendChild(actions);
  }
}

function _parseToolResultPayload(evt) {
  if (!evt || !evt.text) return null;
  try {
    return JSON.parse(evt.text);
  } catch (_) {
    return null;
  }
}

function renderCanonicalResultTables(body, evt) {
  /* Append the per-result canonical tables to the tool-card body.
   *
   * Single-result run: render the one panel inline (it IS the
   * reading surface; nothing to compare against). Multi-result
   * run (N >= 2): collapse the panels into a ``<details>``
   * element closed by default, with a summary line naming the
   * count and id range. The model is now expected to call
   * ``compose_results`` and drop the comparison table into its
   * reply — that becomes the primary reading surface. The
   * collapsed panels stay accessible for audit (one click to
   * expand) without dominating the transcript with N stacked
   * tables.
   *
   * Same data source for both: each entry's ``markdown`` field,
   * canonical render from the sanitized payload. */
  const payload = _parseToolResultPayload(evt);
  const results = payload && Array.isArray(payload.results)
    ? payload.results
    : [];
  const rendered = results.filter((r) =>
    r && r.status === 'ok' && typeof r.markdown === 'string' && r.markdown.trim()
  );
  if (rendered.length === 0) return;

  const panel = document.createElement('div');
  panel.className = 'result-panel';

  function renderOneSection(r, idx) {
    const section = document.createElement('section');
    section.className = 'result-markdown';
    const header = document.createElement('div');
    header.className = 'result-header';
    header.textContent = r.label || r.result_id || ('Result ' + (idx + 1));
    section.appendChild(header);
    const tableWrap = document.createElement('div');
    tableWrap.className = 'result-markdown-body';
    if (window.SiftMarkdown) {
      tableWrap.innerHTML = window.SiftMarkdown.render(r.markdown);
    } else {
      tableWrap.textContent = r.markdown;
    }
    section.appendChild(tableWrap);
    if (r.result_id) {
      const actions = document.createElement('div');
      actions.className = 'result-actions';
      const evidenceBtn = document.createElement('button');
      evidenceBtn.type = 'button';
      evidenceBtn.className = 'result-challenge-btn';
      evidenceBtn.textContent = 'View evidence';
      evidenceBtn.title = 'Dataset, sample size, verification, and '
        + 'generated code behind this result.';
      evidenceBtn.addEventListener('click', () => {
        openEvidencePanel(r.result_id, evidenceBtn);
      });
      actions.appendChild(evidenceBtn);
      actions.appendChild(buildChallengeButton(r));
      section.appendChild(actions);
    } else {
      section.appendChild(buildChallengeButton(r));
    }
    return section;
  }

  function buildChallengeButton(r) {
    /* Sends a real message asking Sift to re-check this one result
     * under alternative specifications — same submit path as the
     * starter chips / Analyze action, not a special codepath. The
     * ROBUST/FRAGILE verdict that comes back is computed by
     * ``challenge_summary`` (server-side, from the batch's actual
     * shared coefficients), not asserted by the model; see the
     * badge rendered from ``payload.challenge_summary`` below. */
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'result-challenge-btn';
    btn.textContent = 'Challenge this finding';
    btn.title = 'Ask Sift to re-estimate this under alternative '
      + 'specifications and report whether it holds up.';
    btn.addEventListener('click', () => {
      const ref = r.label || r.result_id || 'the result above';
      input.value =
        'Challenge the finding "' + ref + '": re-estimate it under '
        + 'at least three reasonable alternative specifications '
        + '(different controls, outlier trimming, an alternate '
        + 'estimator, robust or clustered standard errors, or a '
        + 'placebo check where it applies) in one script, with the '
        + 'original specification as the first result. Then tell me '
        + 'plainly whether the finding holds up.';
      autosize();
      form.dispatchEvent(new Event('submit', { cancelable: true }));
    });
    return btn;
  }

  const challenge = payload && payload.challenge_summary;
  if (challenge && (challenge.verdict === 'ROBUST' || challenge.verdict === 'FRAGILE')) {
    const badge = document.createElement('div');
    badge.className = 'challenge-badge challenge-' + challenge.verdict.toLowerCase();
    badge.textContent = challenge.verdict + ' — ' + challenge.agreeing + ' of '
      + challenge.total + ' alternative specification'
      + (challenge.total === 1 ? '' : 's') + ' agree';
    badge.title = challenge.note || '';
    panel.appendChild(badge);
  }

  if (rendered.length === 1) {
    panel.appendChild(renderOneSection(rendered[0], 0));
    body.appendChild(panel);
    return;
  }

  // Multi-result: collapsed by default. Summary line names what's
  // inside (count + id range) so the audit affordance is visible
  // even when collapsed; one click to expand for the per-result
  // detail. The composite table from ``compose_results`` lives in
  // the model's reply, not here.
  const details = document.createElement('details');
  details.className = 'result-panel-collapsed';
  const summary = document.createElement('summary');
  summary.className = 'result-panel-summary';
  const firstId = rendered[0].result_id || '?';
  const lastId = rendered[rendered.length - 1].result_id || '?';
  const idRange = firstId === lastId
    ? firstId
    : `${firstId}–${lastId}`;
  summary.textContent =
    `${rendered.length} regressions stored: ${idRange} (click to expand)`;
  details.appendChild(summary);
  rendered.forEach((r, idx) => {
    details.appendChild(renderOneSection(r, idx));
  });
  panel.appendChild(details);
  body.appendChild(panel);
}

async function openInNativeApp(path, btn, fallback, mode) {
  /* Ask the Python bridge to hand the path to the native OS opener. ``mode``
   * selects a safe Stata/R/Python-specific launch where available.
   * ``fallback`` is a path to try if the primary fails. */
  if (!window.pywebview || !window.pywebview.api) return;
  const originalText = btn && btn.textContent;
  if (btn) { btn.disabled = true; btn.textContent = 'Opening…'; }
  try {
    const result = await window.pywebview.api.open_path(path, mode || null);
    if (!result || !result.ok) {
      if (fallback) {
        const f = await window.pywebview.api.open_path(fallback, mode || null);
        if (f && f.ok) {
          if (btn) {
            btn.textContent = 'Opened';
            setTimeout(() => { btn.textContent = originalText; }, 1500);
          }
          return;
        }
      }
      if (btn) btn.textContent = result && result.reason
        ? 'Error: ' + result.reason
        : 'Failed';
      setTimeout(() => {
        if (btn) btn.textContent = originalText;
      }, 3000);
    } else {
      if (btn) {
        btn.textContent = 'Opened';
        setTimeout(() => { btn.textContent = originalText; }, 1500);
      }
    }
  } catch (e) {
    if (btn) btn.textContent = 'Failed';
    setTimeout(() => { if (btn) btn.textContent = originalText; }, 2000);
  } finally {
    if (btn) btn.disabled = false;
  }
}

function shortenToolName(name) {
  // mcp__sift__submit_script → submit_script
  const parts = name.split('__');
  return parts[parts.length - 1] || name;
}

function prettyJson(text) {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch (_) {
    return text;
  }
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// "Scroll to latest" floating button. Anchored above the composer
// in the markup; visibility tracks whether the transcript is near
// its bottom.
const scrollToBottomBtn = document.getElementById('scroll-to-bottom');
const SCROLL_TO_BOTTOM_THRESHOLD = 100;

function updateScrollToBottomVisibility() {
  if (!scrollToBottomBtn || !messagesEl) return;
  // Welcome screen has no transcript to scroll through, so the
  // button would just be visual noise above the composer.
  if (messagesEl.classList.contains('welcome-only')) {
    scrollToBottomBtn.classList.add('hidden');
    return;
  }
  const distanceFromBottom = (
    messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight
  );
  const nearBottom = distanceFromBottom < SCROLL_TO_BOTTOM_THRESHOLD;
  scrollToBottomBtn.classList.toggle('hidden', nearBottom);
}

if (messagesEl) {
  messagesEl.addEventListener('scroll', updateScrollToBottomVisibility);
  // Container resizes on sidebar drag, file panel toggle, etc. —
  // the "near bottom" check depends on clientHeight, so we re-poll
  // on resize too.
  window.addEventListener('resize', updateScrollToBottomVisibility);
}

if (scrollToBottomBtn) {
  scrollToBottomBtn.addEventListener('click', () => {
    // Smooth scroll matches user expectation that this is a
    // navigation gesture, not a snap. ``scrollToBottom()`` (the
    // existing helper) is left as the immediate snap that other
    // call sites use after appending content.
    if (typeof messagesEl.scrollTo === 'function') {
      messagesEl.scrollTo({
        top: messagesEl.scrollHeight,
        behavior: 'smooth',
      });
    } else {
      scrollToBottom();
    }
  });
}

function setWelcomeOnlyMode(enabled) {
  if (!messagesEl) return;
  messagesEl.classList.toggle('welcome-only', !!enabled);
  updateScrollToBottomVisibility();
}

// ----- policy chip + popup (next to composer) ----------------------------

const policyChip = document.getElementById('policy-chip');
const policyChipLabel = document.getElementById('policy-chip-label');
const policyPopup = document.getElementById('policy-popup');
let policyPopupBuiltFor = null;  // cached copy so we don't rebuild needlessly

function renderContextChip() {
  /* Paint the chip from ``lastContextCount`` (the last solid count
   * returned by the backend). Stable single-source-of-truth render:
   * no projections, no pending-message arithmetic, no provider
   * ``turn_done`` echoes. The number changes only when a recount
   * response lands — see ``triggerContextRecount`` for the trigger
   * list (turn complete, rewind success, session switch, attachment
   * add/remove).
   *
   * The chip text itself carries NO ``~`` prefix even when
   * ``exact=False`` (chars/3.5 fallback today — replaced by
   * tiktoken / count_tokens once those paths land). The
   * approximate-vs-exact distinction lives in the chip's tooltip
   * instead: an exact count shows just ``"N tokens"``, an
   * approximate count shows ``"N tokens · local estimate, expect
   * a small gap from the provider's billed count"``. This is the
   * only place in the UI that should display a context-size
   * number; ad-hoc estimates elsewhere were the bug we're fixing.
   */
  if (!contextChip) return;
  if (!lastContextCount) {
    contextChip.classList.add('hidden');
    return;
  }
  const { tokens, exact, ceiling } = lastContextCount;
  const fmt = (n) => (n >= 10_000 ? (n / 1000).toFixed(1) + 'k' : n.toString());
  const ceilingLabel = ceiling >= 1_000_000
    ? (ceiling / 1_000_000) + 'M'
    : (ceiling / 1000) + 'k';
  const rawPct = Math.round((tokens / ceiling) * 100);
  // No ``~`` prefix on the chip text. While ``exact`` is False on
  // every render today (the chars/3.5 fallback is the only path
  // wired), a prefix that's always present conveys no information
  // and reads as visual noise. The tooltip still spells out that
  // the value is an approximation. When exact tokenization lands
  // (tiktoken for OpenAI, count_tokens API for Claude), the
  // tooltip wording switches but the chip text stays clean.
  contextChip.textContent =
    `Context ${fmt(tokens)} / ${ceilingLabel} (${rawPct}%)`;
  // Tooltip: precise count + a one-clause honesty caveat when the
  // number is local-approximate. Researchers know how tokens work;
  // they don't need a trigger list. The precise count is the only
  // value-add over the chip text itself ("47.2k" → "47,234").
  contextChip.title = exact
    ? `${tokens.toLocaleString()} tokens`
    : (
        `${tokens.toLocaleString()} tokens · local estimate, ` +
        `expect a small gap from the provider's billed count`
      );
  contextChip.classList.remove('hidden');
  contextChip.classList.remove('warn', 'danger', 'over');
  if (rawPct >= 100) contextChip.classList.add('over');
  else if (rawPct >= 90) contextChip.classList.add('danger');
  else if (rawPct >= 70) contextChip.classList.add('warn');
}

async function triggerContextRecount(reason) {
  /* Refresh the context chip from the backend's pre-flight counter.
   * Call sites: turn complete, rewind success, session open/switch,
   * attachment add/remove. ``reason`` rides through to the console
   * for debugging only — the chip itself shows no "updating..." text.
   *
   * Visual contract: fade the chip immediately so the researcher
   * sees their action acknowledged, even before the backend
   * returns. The fade clears when the matching count response
   * lands. A newer recount (different ``request_id``) supersedes
   * an in-flight one — the older response, when it eventually
   * arrives, fails the id check and is dropped, leaving the chip
   * faded until the newer response lands.
   */
  if (!contextChip) return;
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.count_next_context !== 'function') return;

  contextCountRequestId += 1;
  const id = contextCountRequestId;
  contextChip.classList.add('stale');

  // Pull whatever the composer currently shows. Empty when the chip
  // is fired between turns. The composer element is ``input`` per
  // ``getElementById('compose-input')`` at module top — earlier
  // drafts of this code referenced ``composerEl`` which was never
  // declared, throwing a silent ReferenceError inside this async
  // function and leaving the chip stuck hidden forever.
  const draftText = (input && input.value) || '';
  const nImages = pendingComposerImageCount();
  const nAttachments = pendingComposerScriptCount();

  let res;
  try {
    res = await window.pywebview.api.count_next_context(
      draftText, nImages, nAttachments, id,
    );
  } catch (err) {
    // Network / bridge failure. Leave the chip faded but stop
    // claiming "updating" — the next trigger will retry. The
    // researcher still sees the last solid value, just dimmed.
    console.warn('count_next_context failed', err);
    return;
  }
  if (!res || !res.ok) return;
  if (res.request_id !== contextCountRequestId) {
    // Stale response — newer recount in flight. Drop silently.
    return;
  }
  lastContextCount = {
    tokens: res.tokens,
    exact: !!res.exact,
    ceiling: res.ceiling,
  };
  contextChip.classList.remove('stale');
  renderContextChip();
}

function pendingComposerImageCount() {
  // Pending images attached to the next send (drag-drop / paste).
  // Module-local state, mirrors what ``handleSubmit`` will pack.
  return Array.isArray(stagedImages) ? stagedImages.length : 0;
}

function pendingComposerScriptCount() {
  // Script attachments live on the bridge runner, not in JS state.
  // The bridge's ``count_next_context`` reads
  // ``runner.pending_script_attachments`` directly and overrides
  // whatever JS passes here with the authoritative count + content
  // bytes — so 0 from this side just means "let the backend tell
  // us." Keeping the call site lets a future per-tab JS-only
  // staging list slot in without touching the recount path.
  return 0;
}

function updatePolicyChip(policy) {
  /* Refresh the compact chip label + the per-dataset dropdowns
   * inside the popup. Called on session start and after any policy
   * mutation. The topbar Files chip is refreshed from a separate
   * ``list_session_files`` endpoint that includes scripts and
   * graphs too — the policy payload covers data files only because
   * the SDC layer's permission tiers are dataset-specific. */
  if (!policy || !policy.datasets || policy.datasets.length === 0) {
    policyChip.classList.add('hidden');
  } else {
    policyChip.classList.remove('hidden');
    policyChipLabel.textContent = compactPolicyLabel(policy);
    policyPopupBuiltFor = policy;
    setKnownDatasets(policy);
    policyPopup.innerHTML = '';
    policyPopup.appendChild(buildPolicyPopup(policy));
  }
  refreshFilesChip();
}

// ----- topbar files chip ------------------------------------------------
//
// "Files" lives in the topbar's centered cluster, next to the
// session-title pill. Read-only — permission-tier editing stays in
// the bottom Permission chip. The chip exists so a researcher can
// answer "did my upload land?" at a glance, including scripts and
// graphs that don't show up in the data-only Permission panel.

const filesChip = document.getElementById('files-chip');
const filesChipLabel = document.getElementById('files-chip-label');
const filesPopup = document.getElementById('files-popup');

const FILES_KIND_LABELS = {
  data: 'Data',
  script: 'Scripts',
  graph: 'Graphs',
  log: 'Logs',
};

async function refreshFilesChip() {
  /* Pull the full session file list from the bridge and re-render
   * the chip + popup. Called from updatePolicyChip and from any
   * other event that might have changed cwd contents (drag-drop,
   * dialog upload, session switch).
   *
   * Note: Data files are filtered out of this popup — they're
   * already shown in the bottom Permission chip with their depth
   * tier. The Files popup is the surface for the OTHER session
   * artifacts (scripts, graphs, logs) that have no home elsewhere.
   */
  // The Files panel and the @-mention dropdown share the same
  // source of truth (session-resident files). Whenever this chip
  // refreshes, bust the mention cache so the next "@" pull
  // re-fetches.
  invalidateMentionCache();
  if (!filesChip) return;
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.list_session_files !== 'function') return;
  let res;
  try {
    res = await window.pywebview.api.list_session_files();
  } catch (err) {
    console.warn('list_session_files failed', err);
    return;
  }
  const allFiles = (res && res.files) || [];
  // Drop the data group — it already shows in the Permission chip
  // alongside each file's schema-depth tier. Showing the same names
  // twice (once here, once there) just creates two surfaces that
  // can drift.
  const files = allFiles.filter((f) => f.kind !== 'data');
  if (files.length === 0) {
    filesChip.classList.add('hidden');
    if (filesPopup) filesPopup.classList.add('hidden');
    filesChip.classList.remove('open');
    filesChip.setAttribute('aria-expanded', 'false');
    return;
  }
  filesChip.classList.remove('hidden');
  if (filesChipLabel) {
    filesChipLabel.textContent = `Files · ${files.length}`;
  }
  if (filesPopup) {
    filesPopup.innerHTML = '';
    const wrap = document.createElement('div');
    const header = document.createElement('div');
    header.className = 'policy-popup-header';
    header.innerHTML =
      '<strong>Files</strong>. Scripts, graphs, and logs from this '
      + 'session.';
    wrap.appendChild(header);
    // Group by kind so scripts / graphs / logs land in their own
    // sections — same shape the model picker uses for providers.
    const byKind = new Map();
    files.forEach((f) => {
      const k = f.kind || 'script';
      if (!byKind.has(k)) byKind.set(k, []);
      byKind.get(k).push(f);
    });
    // Render order: graphs first (the visual outputs the
    // researcher iterates on), then scripts (sometimes attached
    // mid-chat), then logs (rarely interacted with). Previously
    // scripts came first, which buried plots below text-only
    // rows even though plots are the most-clicked kind.
    ['graph', 'script', 'log'].forEach((kind) => {
      const rows = byKind.get(kind);
      if (!rows || rows.length === 0) return;
      const groupHeader = document.createElement('div');
      groupHeader.className = 'model-group-header';
      groupHeader.textContent = FILES_KIND_LABELS[kind] || kind;
      wrap.appendChild(groupHeader);
      rows.forEach((f) => {
        wrap.appendChild(buildFilesRow(kind, f));
      });
    });
    filesPopup.appendChild(wrap);
  }
}

function buildFilesRow(kind, f) {
  /* Build one row in the Files popup. Layout: [primary action]
   * [title / thumbnail] [delete]. Every actionable kind uses
   * "copy to clipboard" — the verb stays consistent across rows
   * so a researcher can grab any output and paste it into another
   * chat or an external editor without context-switching to the
   * folder. The clipboard payload varies with the file kind:
   *   - graph (image with thumbnail data): copy as image
   *   - graph (no thumbnail data, e.g. .gph): open externally
   *     (no meaningful clipboard representation for a Stata-
   *     binary plot file)
   *   - script: copy file text content
   *   - log:    copy file text content
   * Delete is always on the right.
   *
   * Why copy instead of "send to next message" for scripts:
   * scripts already get attached via the @-mention dropdown and
   * via drag-drop, so the panel button doesn't need to duplicate
   * that path. Copy-to-clipboard solves the OTHER common case —
   * the researcher wants to take a do-file Sift wrote and use it
   * elsewhere, or save it to their own filesystem.
   */
  const row = document.createElement('div');
  row.className = 'files-row files-row-actionable';
  row.dataset.kind = kind;
  row.dataset.path = f.path || '';
  row.title = f.path || f.name;

  // Left action (kind-specific).
  const leftAction = document.createElement('button');
  leftAction.type = 'button';
  leftAction.className = 'files-row-action files-row-action-left';
  let primaryConfigured = false;
  if (kind === 'graph' && f.data) {
    leftAction.title = 'Copy image to clipboard';
    leftAction.setAttribute('aria-label', 'Copy image');
    leftAction.innerHTML = iconSvg('copy');
    leftAction.addEventListener('click', (ev) => {
      ev.stopPropagation();
      copyImageToClipboard(f.data, f.mime || 'image/png', f.name);
    });
    primaryConfigured = true;
  } else if (kind === 'graph' && f.path) {
    leftAction.title = 'Open in default viewer';
    leftAction.setAttribute('aria-label', 'Open');
    leftAction.innerHTML = iconSvg('openExternal');
    leftAction.addEventListener('click', (ev) => {
      ev.stopPropagation();
      if (window.pywebview && window.pywebview.api &&
          typeof window.pywebview.api.open_path === 'function') {
        window.pywebview.api.open_path(f.path);
      }
    });
    primaryConfigured = true;
  } else if (kind === 'script' || kind === 'log') {
    leftAction.title = 'Copy file contents to clipboard';
    leftAction.setAttribute('aria-label', 'Copy contents');
    leftAction.innerHTML = iconSvg('copy');
    leftAction.addEventListener('click', (ev) => {
      ev.stopPropagation();
      // Pass the full path (not just f.name) so the bridge gate
      // can verify membership in the panel listing by absolute
      // path. The Files panel surfaces researcher-uploaded
      // scripts and logs (run-dir scripts are hidden by design
      // — they already render on the result card).
      copySessionFileText(f.path || f.name, f.name);
    });
    primaryConfigured = true;
  }
  if (primaryConfigured) {
    row.appendChild(leftAction);
  } else {
    // Spacer keeps the title aligned with rows that DO have a
    // left action — visual alignment beats squeezing an extra
    // pixel of horizontal space.
    const spacer = document.createElement('span');
    spacer.className = 'files-row-action-spacer';
    row.appendChild(spacer);
  }

  // Center: title + (for image rows) the thumbnail.
  const center = document.createElement('div');
  center.className = 'files-row-center';
  if (kind === 'graph' && f.data) {
    const thumb = document.createElement('img');
    thumb.className = 'files-row-thumb';
    thumb.alt = f.name;
    thumb.src = `data:${f.mime || 'image/png'};base64,${f.data}`;
    thumb.addEventListener('click', (ev) => {
      ev.stopPropagation();
      showImageLightbox(thumb.src);
    });
    makeKeyboardClickable(thumb, 'View ' + (f.name || 'image') + ' full size');
    center.appendChild(thumb);
  }
  const caption = document.createElement('div');
  caption.className = 'files-row-caption';
  caption.textContent = f.name;
  center.appendChild(caption);
  row.appendChild(center);

  // Right action: delete. Always present so every file in the
  // panel can be removed with one click. Matches the session-list
  // delete affordance — a plain ``×`` glyph rather than an icon —
  // so the visual vocabulary stays consistent across delete
  // surfaces.
  const deleteBtn = document.createElement('button');
  deleteBtn.type = 'button';
  deleteBtn.className = 'files-row-action files-row-action-right files-row-action-danger';
  deleteBtn.title = 'Delete file';
  deleteBtn.setAttribute('aria-label', 'Delete file');
  deleteBtn.textContent = '×';
  deleteBtn.addEventListener('click', (ev) => {
    ev.stopPropagation();
    deleteSessionFile(f.path || f.name, f.name);
  });
  row.appendChild(deleteBtn);
  return row;
}


// Inline SVG icons. Small, monochrome — color is set via CSS so
// the icon picks up the row's hover/focus colors. Single registry
// so the same glyph reads identically wherever it appears: the
// chat-bubble copy button and the Files-panel copy action are the
// SAME copy icon, the bubble's transient post-copy checkmark is the
// SAME check icon, etc. Add a new key here rather than inlining
// another <svg> string at a call site.
const ICON_SVG = {
  copy: (
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" ' +
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
    'stroke-linejoin="round" aria-hidden="true">' +
    '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>' +
    '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>' +
    '</svg>'
  ),
  edit: (
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" ' +
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
    'stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M12 20h9"></path>' +
    '<path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>' +
    '</svg>'
  ),
  // Heavier stroke (2.5 vs 2) so the transient post-copy checkmark
  // reads as confirmation rather than just another monochrome stroke.
  check: (
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" ' +
    'stroke="currentColor" stroke-width="2.5" stroke-linecap="round" ' +
    'stroke-linejoin="round" aria-hidden="true">' +
    '<polyline points="20 6 9 17 4 12"></polyline>' +
    '</svg>'
  ),
  // 16×16 viewBox for this one — the original "external link" glyph
  // was authored at 16px and the path coordinates reflect that.
  openExternal: (
    '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">' +
    '<path fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" ' +
    'd="M9 2h5v5M14 2L7 9M3 4h3M3 4v9h9v-3"/>' +
    '</svg>'
  ),
  // Pin glyphs — minimalist circle-head + needle. ``pin`` is the
  // outlined "click to pin" affordance (hollow head). ``pinFilled``
  // shares the same silhouette with a solid head so the engaged
  // state reads at a glance without needing an accent color: the
  // filled head plus the row's position at the top of the list is
  // enough to spot pinned sessions.
  pin: (
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" ' +
    'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" ' +
    'stroke-linejoin="round" aria-hidden="true">' +
    '<circle cx="12" cy="8" r="3.5"></circle>' +
    '<line x1="12" y1="11.5" x2="12" y2="20"></line>' +
    '</svg>'
  ),
  pinFilled: (
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" ' +
    'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" ' +
    'stroke-linejoin="round" aria-hidden="true">' +
    '<circle cx="12" cy="8" r="3.5" fill="currentColor"></circle>' +
    '<line x1="12" y1="11.5" x2="12" y2="20"></line>' +
    '</svg>'
  ),
};

function iconSvg(name) {
  return ICON_SVG[name] || '';
}

async function copyImageToClipboard(base64Data, mime, name) {
  /* Copy a thumbnail image to the system clipboard via the
   * Clipboard API. WKWebView supports ClipboardItem; if for any
   * reason it doesn't (older macOS), we fall back to a toast
   * pointing at "Show folder" so the researcher isn't stuck.
   */
  try {
    if (!navigator.clipboard || typeof ClipboardItem === 'undefined') {
      toast('Clipboard API unavailable in this WebView; use Show folder.', 'info');
      return;
    }
    const binary = atob(base64Data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const blob = new Blob([bytes], { type: mime || 'image/png' });
    await navigator.clipboard.write([
      new ClipboardItem({ [blob.type]: blob }),
    ]);
    toast('Copied ' + (name || 'image') + ' to clipboard.', 'success');
  } catch (err) {
    console.warn('copyImageToClipboard failed', err);
    toast('Copy failed: ' + (err && err.message ? err.message : err), 'error');
  }
}

// ----- edit & rewind ----------------------------------------------------
//
// Researcher clicks the pencil on a user bubble → that bubble flips into
// edit mode (textarea + Run / Cancel). Run calls
// ``window.pywebview.api.rewind_to(turn_index)`` on the bridge — which
// truncates ``chat_history.jsonl`` at that user message, hides every
// dropped result row in the store, clears the runner's pending
// attachments, and resets the provider session. JS then calls
// ``replayHistory()`` to repaint the trimmed transcript and ``send_message``
// to fire the revised text as a fresh turn — events stream in normally
// on top.
//
// One bubble may be in edit mode at a time; entering edit on a second
// bubble cancels the first. Edit is refused while the session is busy
// (``busySessions.has(currentCwd)``); the researcher must Stop the
// running turn first, which the toast spells out.

let activeEditWrapper = null;

function enterEditMode(wrapper) {
  if (!wrapper) return;
  if (currentCwd && busySessions.has(currentCwd)) {
    toast(
      'Stop the running turn first, then click edit again.',
      'info',
    );
    return;
  }
  // Cancel any other in-progress edit. Two open editors would let a
  // researcher accidentally rewind through both, with the second
  // landing on a transcript that already started shifting under the
  // first.
  if (activeEditWrapper && activeEditWrapper !== wrapper) {
    cancelEditMode(activeEditWrapper);
  }
  if (wrapper.classList.contains('editing')) return;
  activeEditWrapper = wrapper;
  wrapper.classList.add('editing');

  const body = wrapper.querySelector('.message-body');
  if (!body) return;
  // Stash the original DOM so Cancel restores exactly what was
  // there (rendered markdown, attachments hidden by their own row,
  // etc.). textContent + innerHTML aren't enough because some
  // bubbles carry image rows above and chip rows below — those
  // siblings stay put; only the body swaps.
  wrapper.__siftEditOriginalBody = body;

  const editor = document.createElement('div');
  editor.className = 'message-edit-mode';
  const textarea = document.createElement('textarea');
  textarea.className = 'message-edit-textarea';
  textarea.value = wrapper.__siftRawText || '';
  textarea.rows = Math.min(12, Math.max(2, textarea.value.split('\n').length + 1));
  textarea.addEventListener('keydown', (e) => {
    // Submit on Enter (no modifier) and on Cmd/Ctrl+Enter — same
    // affordances as the composer. Shift+Enter inserts a newline.
    const isEnter = e.key === 'Enter';
    const submitPlain = isEnter && !e.shiftKey && !e.metaKey && !e.ctrlKey;
    const submitModifier = isEnter && (e.metaKey || e.ctrlKey) && !e.shiftKey;
    if (submitPlain || submitModifier) {
      e.preventDefault();
      runEditedMessage(wrapper);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      cancelEditMode(wrapper);
    }
  });

  const buttonRow = document.createElement('div');
  buttonRow.className = 'message-edit-buttons';
  const cancelBtn = document.createElement('button');
  cancelBtn.type = 'button';
  cancelBtn.className = 'message-edit-btn-cancel';
  cancelBtn.textContent = 'Cancel';
  cancelBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    cancelEditMode(wrapper);
  });
  const runBtn = document.createElement('button');
  runBtn.type = 'button';
  runBtn.className = 'message-edit-btn-run';
  runBtn.textContent = 'Run';
  runBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    runEditedMessage(wrapper);
  });
  buttonRow.appendChild(cancelBtn);
  buttonRow.appendChild(runBtn);

  editor.appendChild(textarea);
  editor.appendChild(buttonRow);

  // Replace the body in place with the editor; Cancel reverses this.
  body.replaceWith(editor);
  wrapper.__siftEditEditor = editor;
  textarea.focus();
  // Place the cursor at the end so the researcher can extend
  // immediately rather than overwriting.
  textarea.setSelectionRange(textarea.value.length, textarea.value.length);
}

function cancelEditMode(wrapper) {
  if (!wrapper || !wrapper.classList.contains('editing')) return;
  const editor = wrapper.__siftEditEditor;
  const original = wrapper.__siftEditOriginalBody;
  if (editor && original) {
    editor.replaceWith(original);
  }
  wrapper.__siftEditEditor = null;
  wrapper.__siftEditOriginalBody = null;
  wrapper.classList.remove('editing');
  if (activeEditWrapper === wrapper) activeEditWrapper = null;
}

function userMessageIndex(wrapper) {
  /* The bubble's 0-based position among ``.message.user`` bubbles in
   * DOM order — this matches the index of the corresponding
   * ``user_message`` event in ``chat_history.jsonl`` because the
   * persistence path appends one record per appended user bubble.
   * No counter to maintain; we just count siblings.
   */
  const all = Array.from(messagesEl.querySelectorAll('.message.user'));
  return all.indexOf(wrapper);
}

async function runEditedMessage(wrapper) {
  if (!wrapper) return;
  const editor = wrapper.__siftEditEditor;
  if (!editor) return;
  const textarea = editor.querySelector('.message-edit-textarea');
  if (!textarea) return;
  const newText = textarea.value.trim();
  if (!newText) {
    toast('Edited message is empty. Type something or click Cancel.', 'info');
    return;
  }
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.rewind_to !== 'function') {
    toast('Restart Sift to enable message edit.', 'info');
    return;
  }

  const turnIndex = userMessageIndex(wrapper);
  if (turnIndex < 0) {
    toast('Could not locate this message in the transcript.', 'error');
    return;
  }

  // Edit-and-rerun on a message that originally carried attachments
  // or images would silently drop them: the rewind path sends
  // ``newText`` only, and the bridge has no way to reconstitute
  // image bytes (they were never persisted, by design) or guarantee
  // the named scripts still exist on disk. The resulting "rerun"
  // is a different request than the original, while the original's
  // attachment evidence has just been truncated out of history.
  // Force the researcher to acknowledge the drop before proceeding;
  // the alternative ("Cancel and resend a new message") gives them
  // a clean way to redo it with fresh attachments.
  const attachmentChips = wrapper.querySelectorAll(
    '.message-attachment-chip',
  );
  const imageThumbs = wrapper.querySelectorAll('.message-image-thumb');
  if (attachmentChips.length > 0 || imageThumbs.length > 0) {
    const parts = [];
    if (attachmentChips.length > 0) {
      parts.push(
        attachmentChips.length === 1
          ? '1 attached file'
          : attachmentChips.length + ' attached files',
      );
    }
    if (imageThumbs.length > 0) {
      parts.push(
        imageThumbs.length === 1
          ? '1 image'
          : imageThumbs.length + ' images',
      );
    }
    const msg = (
      'This message originally included ' + parts.join(' and ') + '.\n\n'
      + 'Editing and re-running will send the new text WITHOUT those '
      + 'attachments. The rerun is therefore a different request than '
      + 'the original. To keep the attachments, cancel here and resend '
      + 'a new message instead.\n\n'
      + 'Continue without attachments?'
    );
    if (!window.confirm(msg)) {
      return;
    }
  }

  // Disable the editor's buttons so a double-click doesn't fire
  // the rewind twice. The composer / send button stays as-is —
  // the next user_message is what actually re-fires the turn, and
  // we want it to come from the live send path with the same
  // codepath any other message would.
  const buttons = editor.querySelectorAll('button');
  buttons.forEach((b) => { b.disabled = true; });

  let res;
  try {
    res = await window.pywebview.api.rewind_to(turnIndex);
  } catch (err) {
    console.warn('rewind_to failed', err);
    toast(
      'Rewind failed: ' + (err && err.message ? err.message : err),
      'error',
    );
    buttons.forEach((b) => { b.disabled = false; });
    return;
  }
  if (!res || !res.ok) {
    const reason = (res && res.reason) || 'unknown';
    toast('Rewind refused: ' + reason, 'error');
    buttons.forEach((b) => { b.disabled = false; });
    return;
  }

  // Rewind committed. Repaint the transcript from the truncated log
  // (which no longer includes this user message or anything after
  // it), THEN render the new user bubble, THEN fire the send. The
  // bubble has to be rendered JS-side here for the same reason the
  // composer does it on submit: the bridge persists ``user_message``
  // to ``chat_history.jsonl`` but does NOT dispatch a live
  // ``user_message`` event back to JS — there's no event to render
  // off of, so the composer always paints its own bubble. Without
  // this explicit ``appendUser`` call, the rewind path would leave
  // the researcher staring at a transcript that ends one message
  // before the one they just submitted, with no bubble for the
  // edit until ``assistant_text`` started streaming.
  activeEditWrapper = null;
  await replayHistory();

  // Render the new user bubble + busy state, mirroring the
  // composer's submit handler. ``activeLiveTurn`` carries the
  // bubble nodes so disposable-turn cleanup recognises it.
  const userEl = appendUser(newText, [], []);
  activeLiveTurn = { id: null, nodes: [userEl], hasVisibleReply: false };
  setSending(true);

  if (typeof window.pywebview.api.send_message === 'function') {
    try {
      const turnId = await window.pywebview.api.send_message(newText);
      // Stop reads this id to mark the turn cancelled. Same
      // capture pattern as the composer's submit handler.
      if (activeLiveTurn && typeof turnId === 'string' && turnId) {
        activeLiveTurn.id = turnId;
      }
    } catch (err) {
      console.warn('send_message after rewind failed', err);
      toast(
        'Rewind succeeded but send failed: '
        + (err && err.message ? err.message : err),
        'error',
      );
      // Roll back the busy state — the send didn't take, so the
      // composer should be ready to accept another attempt.
      setSending(false);
      activeLiveTurn = null;
      return;
    }
  }

  if (res.hidden_count > 0) {
    toast(
      'Edited message. ' + res.hidden_count + ' prior result'
      + (res.hidden_count === 1 ? '' : 's')
      + ' hidden from model context.',
      'success',
    );
  }
}


async function copyMessageBubble(wrapper, btnEl) {
  /* Copy a chat bubble's text to the clipboard. Source is the
   * pre-render markdown / raw text the bubble was created with —
   * stored on ``wrapper.__siftRawText`` by ``append()``. We
   * deliberately don't read ``message-body.textContent`` because the
   * markdown renderer transforms inline math, smartens punctuation,
   * and collapses adjacent whitespace on lists, so a round-trip
   * through the DOM differs from the literal turn the model
   * generated or the user typed. Researchers pasting into another
   * chat or an editor want the source they sent / received.
   *
   * Visual feedback on success: swap the icon to a checkmark for
   * 1.4 s. The toast bus handles failure cases with the same
   * semantics as the Files-panel copy button.
   */
  if (!wrapper) return;
  const raw = wrapper.__siftRawText || '';
  if (!raw) {
    toast('Nothing to copy from this message.', 'info');
    return;
  }
  try {
    if (!navigator.clipboard || typeof navigator.clipboard.writeText !== 'function') {
      toast('Clipboard API unavailable in this WebView.', 'info');
      return;
    }
    await navigator.clipboard.writeText(raw);
  } catch (err) {
    console.warn('copyMessageBubble failed', err);
    toast('Copy failed: ' + (err && err.message ? err.message : err), 'error');
    return;
  }
  // Inline visual confirmation — a 1.4 s state swap on the button —
  // is less disruptive than a toast on every copy. The researcher
  // gets the receipt right where their cursor is. Toast is reserved
  // for failures, where attention DOES need to leave the bubble.
  if (btnEl) {
    const original = btnEl.innerHTML;
    btnEl.innerHTML = iconSvg('check');
    btnEl.classList.add('copied');
    setTimeout(() => {
      btnEl.innerHTML = original;
      btnEl.classList.remove('copied');
    }, 1400);
  }
}


async function copySessionFileText(path, displayName) {
  /* Pull a script's or log's text content from the bridge and
   * write it to the system clipboard so the researcher can paste
   * it into another chat or an external editor. Sister of
   * copyImageToClipboard for non-image kinds.
   *
   * Takes the full ``path`` (not just a name) so the bridge can
   * verify the row in the panel listing — the bridge enforces
   * the gate against the Files-panel enumeration, so passing a
   * disk path the panel never lists comes back as a refusal.
   * ``displayName`` is the row label shown in the toast.
   *
   * The bridge enforces the size cap and the script/log
   * extension allowlist; on the JS side we just relay the result
   * to the clipboard and surface the bridge's "reason" verbatim
   * if the read failed (so an over-the-cap log produces the
   * researcher-actionable hint about opening the folder, not a
   * generic "copy failed").
   */
  if (!path) return;
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.read_session_file_text !== 'function') {
    toast('Restart Sift to enable copy-from-Files.', 'info');
    return;
  }
  let res;
  try {
    res = await window.pywebview.api.read_session_file_text(path);
  } catch (err) {
    console.warn('read_session_file_text failed', err);
    toast(
      'Copy failed: ' + (err && err.message ? err.message : err),
      'error',
    );
    return;
  }
  if (!res || !res.ok) {
    toast('Copy failed: ' + ((res && res.reason) || 'unknown'), 'error');
    return;
  }
  try {
    if (!navigator.clipboard || typeof navigator.clipboard.writeText !== 'function') {
      toast('Clipboard API unavailable in this WebView.', 'info');
      return;
    }
    await navigator.clipboard.writeText(res.text || '');
    toast('Copied ' + (displayName || res.name || path) + ' to clipboard.', 'success');
  } catch (err) {
    console.warn('clipboard.writeText failed', err);
    toast('Copy failed: ' + (err && err.message ? err.message : err), 'error');
  }
}


async function deleteSessionFile(path, displayName) {
  /* Delete a file via the bridge. Confirms first because
   * unlinking is irreversible and the Files panel doesn't have
   * an undo. Refreshes the panel + composer chips on success
   * so the row vanishes immediately.
   */
  if (!path) return;
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.delete_session_file !== 'function') {
    toast('Restart Sift to enable file deletion.', 'info');
    return;
  }
  const label = displayName || path;
  const ok = window.confirm(`Delete ${label}?\n\nThis cannot be undone.`);
  if (!ok) return;
  try {
    const res = await window.pywebview.api.delete_session_file(path);
    if (!res || !res.ok) {
      const reason = (res && res.reason) || 'unknown';
      toast('Could not delete: ' + reason, 'error');
      return;
    }
    toast('Deleted ' + (res.name || label) + '.', 'success');
    refreshFilesChip();
    // Drop matching composer chips before re-rendering. ``res.unstaged``
    // is the authoritative list of staged names the backend just
    // dropped from the runner's pending_* lists — for run-dir scripts
    // it carries the label-derived display name (``linear_regression.py``)
    // that the chip shows, not the on-disk basename (``script.py``)
    // in ``res.name``. Splicing by both keeps the chip row honest
    // for older bridges that don't populate ``unstaged``.
    const dropped = Array.isArray(res.unstaged) ? res.unstaged.slice() : [];
    if (res.name) dropped.push(res.name);
    if (dropped.length > 0) {
      for (let i = stagedDataNotices.length - 1; i >= 0; i--) {
        if (dropped.includes(stagedDataNotices[i])) {
          stagedDataNotices.splice(i, 1);
        }
      }
    }
    renderAttachments();
  } catch (err) {
    console.warn('delete_session_file failed', err);
    toast('Could not delete: ' + (err && err.message ? err.message : err), 'error');
  }
}


if (filesChip && filesPopup) {
  filesChip.addEventListener('click', (e) => {
    e.stopPropagation();
    const isOpen = !filesPopup.classList.contains('hidden');
    filesPopup.classList.toggle('hidden');
    filesChip.classList.toggle('open', !isOpen);
    filesChip.setAttribute('aria-expanded', String(!isOpen));
  });
  document.addEventListener('click', (e) => {
    if (filesPopup.classList.contains('hidden')) return;
    if (filesPopup.contains(e.target) || filesChip.contains(e.target)) return;
    filesPopup.classList.add('hidden');
    filesChip.classList.remove('open');
    filesChip.setAttribute('aria-expanded', 'false');
  });
}

function compactPolicyLabel(policy) {
  // Chip label. "Permission" reads more clearly than the internal
  // "Policy" term — it's what the selected model is *permitted* to see, not a
  // legal/admin policy. Count only surfaces when something's been
  // customized; at default we keep the chip to a single word.
  const customized = policy.datasets.filter((d) => d.explicit).length;
  if (customized === 0) return 'Permission';
  return `Permission · ${customized} custom`;
}

function buildPolicyPopup(policy) {
  /* Custom popup — each dataset gets a tier-selector built from
   * buttons instead of <select>. The native dropdown ignores most
   * CSS (it uses OS chrome), so the old version went off-theme in
   * dark mode. Button rows follow the Sift palette and are
   * accessible via keyboard and screen readers. */
  const wrapper = document.createElement('div');
  const header = document.createElement('div');
  header.className = 'policy-popup-header';
  // Compact but informative: names the control, the unit it acts on,
  // and the one-way semantic (ceiling, not target). Drops the
  // "default: …" crutch. The active tier is visible in the row
  // selection itself. Period separator matches the Files / Model
  // popups so the three top-bar chips read identically.
  header.innerHTML =
    '<strong>Permission</strong>. Ceiling on variable details ' +
    'Sift sees per dataset. It can ask for less, never more.';
  wrapper.appendChild(header);

  policy.datasets.forEach((d) => {
    const group = document.createElement('div');
    group.className = 'policy-dataset';

    const name = document.createElement('div');
    name.className = 'policy-dataset-name';
    name.textContent = d.name;
    group.appendChild(name);

    const options = document.createElement('div');
    options.className = 'policy-options';

    DEPTH_TIERS.forEach((tier) => {
      const opt = document.createElement('button');
      opt.type = 'button';
      opt.className = 'policy-option';
      opt.dataset.value = tier.value;
      if (tier.value === d.ceiling) opt.classList.add('selected');

      const check = document.createElement('span');
      check.className = 'policy-option-check';
      check.setAttribute('aria-hidden', 'true');
      check.textContent = tier.value === d.ceiling ? '●' : '○';
      opt.appendChild(check);

      const label = document.createElement('span');
      label.className = 'policy-option-label';
      label.textContent = tier.label;
      opt.appendChild(label);

      opt.addEventListener('click', async () => {
        if (opt.classList.contains('selected')) return;
        const prev = group.querySelector('.policy-option.selected');
        // Optimistic UI: flip selection immediately; revert on error.
        options.querySelectorAll('.policy-option').forEach((el) => {
          el.classList.remove('selected');
          const c = el.querySelector('.policy-option-check');
          if (c) c.textContent = '○';
        });
        opt.classList.add('selected');
        const c = opt.querySelector('.policy-option-check');
        if (c) c.textContent = '●';

        try {
          const result = await window.pywebview.api.set_dataset_policy(
            d.name, tier.value
          );
          if (!result || !result.ok) {
            // Revert on failure.
            opt.classList.remove('selected');
            if (c) c.textContent = '○';
            if (prev) {
              prev.classList.add('selected');
              const pc = prev.querySelector('.policy-option-check');
              if (pc) pc.textContent = '●';
            }
            const err = document.createElement('div');
            err.className = 'policy-row-err';
            err.textContent = result && result.reason ? result.reason : 'failed';
            group.appendChild(err);
            setTimeout(() => err.remove(), 4000);
          } else if (result.policy) {
            updatePolicyChip(result.policy);
          }
        } catch (e) {
          opt.classList.remove('selected');
          if (c) c.textContent = '○';
          if (prev) {
            prev.classList.add('selected');
            const pc = prev.querySelector('.policy-option-check');
            if (pc) pc.textContent = '●';
          }
        }
      });
      options.appendChild(opt);
    });

    group.appendChild(options);
    wrapper.appendChild(group);
  });

  return wrapper;
}

// Chip click toggles the popup; click outside closes it.
policyChip.addEventListener('click', (e) => {
  e.stopPropagation();
  const isOpen = !policyPopup.classList.contains('hidden');
  policyPopup.classList.toggle('hidden');
  policyChip.classList.toggle('open', !isOpen);
  policyChip.setAttribute('aria-expanded', String(!isOpen));
});
document.addEventListener('click', (e) => {
  if (policyPopup.classList.contains('hidden')) return;
  if (policyPopup.contains(e.target) || policyChip.contains(e.target)) return;
  policyPopup.classList.add('hidden');
  policyChip.classList.remove('open');
  policyChip.setAttribute('aria-expanded', 'false');
});

// ----- model picker ------------------------------------------------------

const modelChip = document.getElementById('model-chip');
const modelChipLabel = document.getElementById('model-chip-label');
const modelPopup = document.getElementById('model-popup');
let availableModels = [];      // cached from list_models
let currentModelId = null;     // the backend's authoritative selection
// Effort ladders are PER PROVIDER — Anthropic offers max, OpenAI
// stops at xhigh — so the bar is rebuilt from the selected model's
// provider on every render. effortsByProvider holds all of them so a
// model switch repaints without another round-trip.
let effortsByProvider = {};    // {provider: [{id,label}]}
let currentEffortId = null;    // the backend's authoritative effort level
let defaultEffortId = null;    // catalog default (dotted in the bar)

async function loadModels() {
  /* Fetch the model catalog from the backend and render the popup.
   *
   * The backend's ``current`` is authoritative — it reflects the
   * per-session choice restored from ``.sift/session_state.json`` if
   * the researcher already used a particular model in this session,
   * or the global default otherwise. We used to override it with a
   * localStorage value here, but that turned a per-session feature
   * into a global one and silently swapped models on session open.
   */
  if (!modelChip || !window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.list_models !== 'function') return;
  try {
    const res = await window.pywebview.api.list_models();
    if (!res || !res.ok) return;
    availableModels = res.models || [];
    currentModelId = res.current;
    effortsByProvider = res.efforts_by_provider || {};
    currentEffortId = res.current_effort || null;
    defaultEffortId = res.default_effort || null;
    renderModelChip();
  } catch (err) {
    console.warn('list_models failed', err);
  }
}

function renderModelChip() {
  if (!modelChip || !modelChipLabel) return;
  const info = availableModels.find((m) => m.id === currentModelId);
  // "Sonnet 5 · xhigh" — the effort rides the chip as a dim mono
  // suffix so the current level is visible without opening the
  // popup. Rendered as two spans so the suffix can be
  // styled independently; the label span is rebuilt each time.
  modelChipLabel.textContent = '';
  const nameSpan = document.createElement('span');
  nameSpan.textContent = info ? info.label : 'Model';
  modelChipLabel.appendChild(nameSpan);
  if (currentEffortId) {
    const effSpan = document.createElement('span');
    effSpan.className = 'model-chip-effort';
    effSpan.textContent = ' · ' + currentEffortId;
    modelChipLabel.appendChild(effSpan);
  }
  // Keep the context-chip ceiling in sync with the selected model
  // so the X / Y ratio reflects that model's actual window. If a
  // turn_done already painted the chip against the default window
  // (the loadModels / first-turn race in showChat), re-render so the
  // ratio reflects the now-known ceiling.
  if (info && info.context_window) {
    const changed = contextWindow !== info.context_window;
    contextWindow = info.context_window;
    // Window changed — re-render against the new denominator if we
    // have a solid count cached. No backend recount: same-tokenizer
    // model swap doesn't change the numerator, and cross-tokenizer
    // swap only matters once a turn fires under the new model
    // (which then triggers a recount on its own).
    if (changed && lastContextCount) {
      lastContextCount = { ...lastContextCount, ceiling: contextWindow };
      renderContextChip();
    }
  }
  renderModelPopup();
}

function renderModelPopup() {
  if (!modelPopup) return;
  modelPopup.innerHTML = '';
  const wrap = document.createElement('div');
  const header = document.createElement('div');
  header.className = 'policy-popup-header';
  header.innerHTML = '<strong>Model</strong>. The active model for this session.';
  wrap.appendChild(header);

  // Group models by provider so the picker reads as
  //   Anthropic
  //     Sonnet 5 (1M)
  //     Opus 5 (1M)
  //     Fable 5 (1M)
  //   OpenAI
  //     GPT-5.6 Terra (1.05M)
  //     GPT-5.6 Sol (1.05M)
  // Models for un-authed providers stay in the list but render
  // disabled with a "Configure auth" hint so the researcher can see
  // the option exists without being able to silently pick it.
  const byProvider = new Map();
  availableModels.forEach((m) => {
    const p = m.provider || 'anthropic';
    if (!byProvider.has(p)) byProvider.set(p, []);
    byProvider.get(p).push(m);
  });

  const providerOrder = ['anthropic', 'openai', 'gemini', 'openai_compatible'];
  providerOrder.forEach((p) => {
    const models = byProvider.get(p);
    if (!models || models.length === 0) return;
    const groupHeader = document.createElement('div');
    groupHeader.className = 'model-group-header';
    groupHeader.textContent = providerLabel(p);
    wrap.appendChild(groupHeader);

    models.forEach((m) => {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'model-option';
      if (m.id === currentModelId) row.classList.add('active');
      const available = m.available !== false;
      if (!available) {
        row.classList.add('unconfigured');
      }

      // Hover tooltip — only for the auth state. Researchers can
      // open the per-row pricing link for live pricing detail rather
      // than relying on canned text that goes stale.
      if (!available) {
        row.dataset.tooltip = 'Click to configure';
      } else {
        delete row.dataset.tooltip;
      }

      const name = document.createElement('span');
      name.className = 'model-option-name';
      name.textContent = m.label;
      row.appendChild(name);

      // Right-side cluster: context window + per-row pricing link.
      // Wrapped so the link sits inside the picker row's hover area
      // but stops click propagation so it doesn't trigger a model
      // switch on the way out.
      const right = document.createElement('span');
      right.className = 'model-option-right';

      const ctx = document.createElement('span');
      ctx.className = 'model-option-ctx';
      ctx.textContent = formatContextWindow(m.context_window);
      right.appendChild(ctx);

      if (m.pricing_url) {
        const priceLink = document.createElement('button');
        priceLink.type = 'button';
        priceLink.className = 'model-option-price';
        priceLink.title = 'View pricing';
        priceLink.textContent = '$';
        priceLink.addEventListener('click', (e) => {
          e.stopPropagation();
          openExternal(m.pricing_url);
        });
        right.appendChild(priceLink);
      }
      row.appendChild(right);

      row.addEventListener('click', () => {
        modelPopup.classList.add('hidden');
        modelChip.classList.remove('open');
        modelChip.setAttribute('aria-expanded', 'false');
        if (available) {
          setModel(m.id, /*silent=*/false);
        } else {
          // Clicking an un-authed model jumps to the auth screen so
          // the researcher can paste a key for that provider without
          // having to bounce through landing first.
          openAuthScreen();
        }
      });
      wrap.appendChild(row);
    });
  });

  // Effort bar embedded under the model list so "which model" and "how hard it
  // thinks" live in one place. The ladder belongs to the SELECTED
  // model's provider (Anthropic goes to max, OpenAI stops at xhigh),
  // so it's looked up per render and changes when the model does.
  // Segmented control; the active level is filled, and the default
  // carries a small dot so a researcher who wandered off it can find
  // the way back.
  const currentInfo = availableModels.find((m) => m.id === currentModelId);
  const currentProvider = (currentInfo && currentInfo.provider) || 'anthropic';
  const availableEfforts = effortsByProvider[currentProvider] || [];
  if (availableEfforts.length > 0) {
    const effHeader = document.createElement('div');
    effHeader.className = 'model-group-header';
    effHeader.textContent = 'Effort';
    wrap.appendChild(effHeader);

    const seg = document.createElement('div');
    seg.className = 'effort-seg';
    seg.setAttribute('role', 'radiogroup');
    seg.setAttribute('aria-label', 'Reasoning effort');
    availableEfforts.forEach((e) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'effort-seg-btn';
      btn.dataset.effort = e.id;
      btn.setAttribute('role', 'radio');
      const isActive = e.id === currentEffortId;
      btn.setAttribute('aria-checked', isActive ? 'true' : 'false');
      if (isActive) btn.classList.add('active');
      if (e.id === defaultEffortId) btn.classList.add('is-default');
      btn.textContent = e.id;
      btn.title = e.label + (e.id === defaultEffortId ? ' (default)' : '');
      btn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        if (e.id === currentEffortId) return;
        setEffort(e.id);
      });
      seg.appendChild(btn);
    });
    wrap.appendChild(seg);
  }

  modelPopup.appendChild(wrap);
}

function formatContextWindow(n) {
  if (!n) return '';
  if (n >= 1_000_000) {
    // 1.05M reads better than 1.0500000M; one decimal place is plenty.
    const m = n / 1_000_000;
    return (Math.round(m * 100) / 100) + 'M ctx';
  }
  return (n / 1000) + 'k ctx';
}

function showInstallConfirmationModal(evt) {
  /* Hard consent gate for the ``install_packages`` tool. The Python
   * handler is awaiting a response keyed by ``evt.token``; the
   * researcher's click on Approve / Deny calls
   * ``respond_install_confirmation`` to release that future. If the
   * researcher closes the modal without clicking either button
   * (Esc / overlay-click), we send an explicit deny so the handler
   * doesn't sit on its 5-minute timeout.
   *
   * Multiple concurrent requests are stacked: each call creates its
   * own overlay (token-keyed) so an early decision can't be applied
   * to a later request by accident. */
  if (!evt || !evt.token) return;

  const overlay = document.createElement('div');
  overlay.className = 'install-confirmation-overlay';
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-modal', 'true');
  overlay.setAttribute('aria-label', 'Confirm package install');
  overlay.tabIndex = -1;

  const card = document.createElement('div');
  card.className = 'install-confirmation-card';

  const title = document.createElement('div');
  title.className = 'install-confirmation-title';
  const actionLabel = (evt.action === 'remove')
    ? 'Remove'
    : (evt.action === 'reinstall') ? 'Reinstall' : 'Install';
  title.textContent = actionLabel + ' ' + (evt.language || '') + ' packages?';

  // Which session is asking. Sift can run several sessions'
  // background turns concurrently (switching sidebar focus doesn't
  // stop the one left behind), so a request from a session the
  // researcher ISN'T currently looking at must say so plainly --
  // without this, an install prompt could be mistaken for one
  // belonging to whatever session happens to be on screen right
  // now. ``evt.session_title``/``evt.cwd`` are populated by the
  // bridge; falls back to a generic note if somehow both are empty
  // (an older bridge build, or title resolution failed server-side).
  const sessionLine = document.createElement('div');
  sessionLine.className = 'install-confirmation-session';
  const sessionLabel = evt.session_title || evt.cwd || '(unknown session)';
  const isBackgroundSession = !!(evt.cwd && evt.cwd !== currentCwd);
  sessionLine.textContent = (
    (isBackgroundSession ? 'Requested by background session: ' : 'Session: ')
    + sessionLabel
  );
  if (isBackgroundSession) {
    sessionLine.classList.add('install-confirmation-session-background');
  }
  if (evt.cwd) sessionLine.title = evt.cwd;

  const desc = document.createElement('div');
  desc.className = 'install-confirmation-desc';
  desc.textContent = (
    'Sift wants to ' + actionLabel.toLowerCase() + ' the following '
    + 'packages on this machine. This runs the language’s package '
    + 'manager outside the sandbox and writes to your user library.'
  );

  const pkgList = document.createElement('ul');
  pkgList.className = 'install-confirmation-pkgs';
  (evt.packages || []).slice(0, 50).forEach((name) => {
    const li = document.createElement('li');
    li.textContent = String(name);
    pkgList.appendChild(li);
  });
  if ((evt.packages || []).length > 50) {
    const more = document.createElement('li');
    more.className = 'install-confirmation-more';
    more.textContent = '… and ' + ((evt.packages.length - 50)) + ' more';
    pkgList.appendChild(more);
  }

  const actions = document.createElement('div');
  actions.className = 'install-confirmation-actions';
  const denyBtn = document.createElement('button');
  denyBtn.type = 'button';
  denyBtn.className = 'install-confirmation-deny';
  denyBtn.textContent = 'Deny';
  const approveBtn = document.createElement('button');
  approveBtn.type = 'button';
  approveBtn.className = 'install-confirmation-approve';
  approveBtn.textContent = actionLabel;
  actions.appendChild(denyBtn);
  actions.appendChild(approveBtn);

  card.appendChild(title);
  card.appendChild(sessionLine);
  card.appendChild(desc);
  card.appendChild(pkgList);
  card.appendChild(actions);
  overlay.appendChild(card);

  let resolved = false;
  const respond = async (approved) => {
    if (resolved) return;
    resolved = true;
    overlay.remove();
    document.removeEventListener('keydown', onKey);
    if (!window.pywebview || !window.pywebview.api) return;
    if (typeof window.pywebview.api.respond_install_confirmation !== 'function') {
      // Older bridge build without the new method. The Python
      // handler will time out; nothing we can do from JS to fix.
      return;
    }
    try {
      await window.pywebview.api.respond_install_confirmation(evt.token, !!approved);
      // An approved install can change runtime health (a missing
      // package that was gating a "warning" status, or a fresh
      // ``pip install`` fixing a "blocked" one) -- refresh the
      // banner rather than leaving a stale warning up, or missing a
      // newly-introduced one, until the researcher happens to
      // restart Sift.
      if (approved) refreshDoctorBanner();
    } catch (err) {
      console.warn('respond_install_confirmation failed', err);
    }
  };

  approveBtn.addEventListener('click', () => respond(true));
  denyBtn.addEventListener('click', () => respond(false));
  // Esc / overlay-click default to deny — closing without an explicit
  // approve is the safe interpretation, and Python is blocked on the
  // future, so silently dismissing would leave it waiting.
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) respond(false);
  });
  const onKey = (e) => {
    if (e.key === 'Escape') respond(false);
    // Enter MUST NOT unconditionally approve. The modal focuses
    // Deny by default (see ``denyBtn.focus()`` below), and an
    // unguarded ``Enter → respond(true)`` contradicted that — a
    // researcher hitting Enter on the focused Deny button got an
    // Approve anyway, which is the opposite of what every native
    // dialog convention promises. Only approve if Approve actually
    // has keyboard focus (the researcher Tabbed to it deliberately).
    // Native button activation on the focused element handles the
    // visual case correctly without this branch; the explicit gate
    // is here so a focus change that happens between events still
    // routes correctly.
    if (e.key === 'Enter' && document.activeElement === approveBtn) {
      respond(true);
    }
  };
  document.addEventListener('keydown', onKey);
  document.body.appendChild(overlay);
  // Focus Deny by default so an absent-minded Enter denies (the
  // button's native Enter handler clicks it); the researcher must
  // Tab to Approve before Enter approves.
  denyBtn.focus();
}

function showImageLightbox(url) {
  /* Open an in-page overlay showing ``url`` at full resolution.
   * Used for image attachments — the bridge's ``open_external`` is
   * allowlisted to specific HTTPS pricing pages and won't accept
   * blob: URLs anyway. Click anywhere or press Esc to close. */
  if (!url) return;
  const overlay = document.createElement('div');
  overlay.className = 'image-lightbox';
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-modal', 'true');
  overlay.setAttribute('aria-label', 'Enlarged image');
  overlay.tabIndex = -1;
  const big = document.createElement('img');
  big.src = url;
  big.alt = 'Enlarged image';
  big.className = 'image-lightbox-img';
  overlay.appendChild(big);
  const close = () => {
    overlay.remove();
    document.removeEventListener('keydown', onKey);
  };
  const onKey = (e) => {
    if (e.key === 'Escape') close();
  };
  overlay.addEventListener('click', close);
  document.addEventListener('keydown', onKey);
  document.body.appendChild(overlay);
  // Focus so Esc works without an extra click first.
  overlay.focus();
}

async function openExternal(url) {
  /* Hand a URL to the OS default browser via the bridge. The bridge
   * allowlists URLs against the known pricing pages so this can't be
   * coerced into navigating to attacker-controlled sites by a stray
   * tool result. */
  if (!url) return;
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.open_external !== 'function') return;
  try {
    const res = await window.pywebview.api.open_external(url);
    if (res && !res.ok) {
      console.warn('open_external rejected:', res.reason);
    }
  } catch (err) {
    console.warn('open_external failed', err);
  }
}

async function setModel(modelId, silent) {
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.set_model !== 'function') {
    toast('Restart Sift to enable model switching.', 'error', 'model');
    return;
  }
  try {
    const res = await window.pywebview.api.set_model(modelId);
    if (!res || !res.ok) {
      if (!silent) {
        const reason = res && res.reason ? res.reason : 'unknown';
        toast('Model switch failed: ' + reason, 'error', 'model');
      }
      return;
    }
    currentModelId = modelId;
    // A cross-provider switch can clamp the effort onto the new
    // provider's ladder (Anthropic max -> OpenAI xhigh). The backend
    // is authoritative about where it landed, so adopt what it
    // reports rather than assuming the level carried over.
    const clamped = res.effort && res.effort !== currentEffortId;
    if (res.effort) currentEffortId = res.effort;
    renderModelChip();
    if (!silent && !res.unchanged) {
      const tail = clamped
        ? ' Effort moved to ' + res.effort + ' (highest this provider offers).'
        : '';
      toast('Model switched to ' + (res.label || modelId) + '. Takes effect on the next message.' + tail, 'success', 'model');
    }
  } catch (err) {
    console.warn('set_model failed', err);
    if (!silent) toast('Model switch failed: ' + err, 'error', 'model');
  }
}

async function setEffort(effortId) {
  /* Switch the focused session's reasoning effort. Mirrors setModel:
   * the backend is authoritative, so we only repaint after it says ok.
   * The popup stays open — effort is a dial researchers nudge and
   * compare, not a one-shot pick like the model — and the segmented
   * control re-renders in place. Anthropic sessions re-warm on the
   * next message (the Agent SDK takes effort at launch only), so the
   * toast says so when the backend flags it. */
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.set_effort !== 'function') {
    toast('Restart Sift to enable effort switching.', 'error', 'model');
    return;
  }
  try {
    const res = await window.pywebview.api.set_effort(effortId);
    if (!res || !res.ok) {
      const reason = res && res.reason ? res.reason : 'unknown';
      toast('Effort switch failed: ' + reason, 'error', 'model');
      return;
    }
    currentEffortId = effortId;
    renderModelChip();
    if (!res.unchanged) {
      const tail = res.conversation_rewarmed
        ? ' Session re-warms on the next message.'
        : ' Takes effect on the next message.';
      toast('Effort set to ' + (res.label || effortId) + '.' + tail, 'success', 'model');
    }
  } catch (err) {
    console.warn('set_effort failed', err);
    toast('Effort switch failed: ' + err, 'error', 'model');
  }
}

if (modelChip && modelPopup) {
  modelChip.addEventListener('click', (e) => {
    e.stopPropagation();
    const isOpen = !modelPopup.classList.contains('hidden');
    modelPopup.classList.toggle('hidden');
    modelChip.classList.toggle('open', !isOpen);
    modelChip.setAttribute('aria-expanded', String(!isOpen));
  });
  document.addEventListener('click', (e) => {
    if (modelPopup.classList.contains('hidden')) return;
    if (modelPopup.contains(e.target) || modelChip.contains(e.target)) return;
    modelPopup.classList.add('hidden');
    modelChip.classList.remove('open');
    modelChip.setAttribute('aria-expanded', 'false');
  });
}

// ----- session browser sidebar -------------------------------------------

const sidebarEl = document.getElementById('sidebar');
const sidebarListEl = document.getElementById('sidebar-list');
const sidebarToggleBtn = document.getElementById('sidebar-toggle');
const newSessionBtn = document.getElementById('new-session-btn');
const sidebarResizeEl = document.getElementById('sidebar-resize');
const SIDEBAR_COLLAPSED_KEY = 'sift.sidebarCollapsed';
const SIDEBAR_WIDTH_KEY = 'sift.sidebarWidth';
const SIDEBAR_MIN_WIDTH = 220;
const SIDEBAR_MAX_WIDTH = 520;

function setSidebarWidth(px, persist) {
  /* Write the width through a CSS custom property so flex-basis,
   * width, and max-width all update in one place (see .sidebar in
   * style.css). Clamped so a wild drag can't eat the chat column
   * or shrink the rail below usable size. */
  const clamped = Math.max(SIDEBAR_MIN_WIDTH, Math.min(SIDEBAR_MAX_WIDTH, px));
  if (sidebarEl) {
    sidebarEl.style.setProperty('--sidebar-width', clamped + 'px');
  }
  if (persist) {
    try { localStorage.setItem(SIDEBAR_WIDTH_KEY, String(clamped)); }
    catch (_) {}
  }
  return clamped;
}

// Restore width from a prior session before the first paint.
try {
  const stored = parseInt(localStorage.getItem(SIDEBAR_WIDTH_KEY) || '', 10);
  if (!Number.isNaN(stored)) setSidebarWidth(stored, /*persist=*/false);
} catch (_) {}

// Drag-to-resize. The handle is a 6px strip glued to the right edge.
// mousemove/up live on document so fast drags don't lose the pointer.
if (sidebarResizeEl) {
  let dragging = false;
  const onMouseMove = (e) => {
    if (!dragging) return;
    // Mouse X from the left edge of the viewport is the new width
    // (sidebar is pinned to the left). Don't persist mid-drag; wait
    // for mouseup so we only write once per gesture.
    setSidebarWidth(e.clientX, /*persist=*/false);
  };
  const onMouseUp = () => {
    if (!dragging) return;
    dragging = false;
    sidebarResizeEl.classList.remove('dragging');
    document.body.style.userSelect = '';
    document.body.style.cursor = '';
    // Persist the final width once.
    const w = parseInt(getComputedStyle(sidebarEl).width, 10);
    if (!Number.isNaN(w)) {
      try { localStorage.setItem(SIDEBAR_WIDTH_KEY, String(w)); } catch (_) {}
    }
  };
  sidebarResizeEl.addEventListener('mousedown', (e) => {
    e.preventDefault();
    dragging = true;
    sidebarResizeEl.classList.add('dragging');
    // Suppress text-selection flashing while dragging, and keep the
    // resize cursor even when the pointer strays onto other elements.
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'ew-resize';
  });
  document.addEventListener('mousemove', onMouseMove);
  document.addEventListener('mouseup', onMouseUp);
}

async function loadSessions() {
  /* Fetch the session list from the backend and render it. Called
   * on chat-view entry and after a session switch.
   *
   * If the bridge method is missing (running against an older
   * backend that hasn't been restarted since switch_session /
   * list_sessions were added), render an explanatory empty state
   * instead of a silently-blank sidebar.
   */
  if (!sidebarListEl) return;
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.list_sessions !== 'function') {
    renderSidebarMessage('Restart Sift to see past sessions.');
    return;
  }
  try {
    const res = await window.pywebview.api.list_sessions();
    if (!res || !res.ok) {
      renderSidebarMessage('Could not load sessions.');
      return;
    }
    renderSessions(res.sessions || [], res.current || null);
  } catch (err) {
    console.warn('list_sessions failed', err);
    renderSidebarMessage('Could not load sessions.');
  }
}

function renderSidebarMessage(text) {
  if (!sidebarListEl) return;
  sidebarListEl.innerHTML = '';
  const note = document.createElement('div');
  note.className = 'sidebar-list-empty';
  note.textContent = text;
  sidebarListEl.appendChild(note);
}

function renderSessions(sessions, currentPath) {
  sidebarListEl.innerHTML = '';
  if (sessions.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'sidebar-list-empty';
    empty.textContent = 'No past sessions yet.';
    sidebarListEl.appendChild(empty);
    return;
  }
  sessions.forEach((s) => {
    // Each row: a pin toggle on the far left, the clickable session
    // body in the middle (switches cwd), rename + trash on the right.
    // Every action button is a sibling of the session-item button
    // rather than a nested child — putting interactives inside the
    // <button> is invalid nesting AND would make their clicks bubble
    // into the switch handler.
    const row = document.createElement('div');
    row.className = 'session-row';
    if (s.path === currentPath) row.classList.add('active');
    if (s.pinned) row.classList.add('pinned');
    row.title = s.path;

    // Pin toggle — leftmost. Two visually distinct icons make pinned
    // vs unpinned readable at a glance without reading the title
    // attribute, and the row is also tagged ``.pinned`` so we can
    // also hint with a subtle background/weight change in CSS.
    const pinBtn = document.createElement('button');
    pinBtn.type = 'button';
    pinBtn.className = 'session-pin';
    if (s.pinned) pinBtn.classList.add('is-pinned');
    pinBtn.setAttribute(
      'aria-label', s.pinned ? 'Unpin session' : 'Pin session to top'
    );
    pinBtn.title = s.pinned ? 'Unpin from top' : 'Pin to top';
    pinBtn.innerHTML = iconSvg(s.pinned ? 'pinFilled' : 'pin');
    pinBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      toggleSessionPinned(s.path, !s.pinned);
    });
    row.appendChild(pinBtn);

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'session-item';
    // ``data-path`` is read by setSessionBusy() to flip the busy
    // dot on this row when a turn starts/ends — including for
    // background sessions the user isn't currently looking at.
    btn.dataset.path = s.path;
    if (busySessions.has(s.path)) btn.classList.add('busy');
    // Single-line layout: [title …] [age | busy-dot]
    // ``s.title`` already resolves to custom_name if the researcher
    // set one, otherwise to the dataset filename ("<first> +N" for
    // multi-file sessions), otherwise to a session-stamp fallback.
    // The busy dot lives in the same right-side slot as the age and
    // CSS flips visibility — pulse replaces "1d" while a turn runs.
    const titleEl = document.createElement('span');
    titleEl.className = 'session-title';
    titleEl.textContent = s.title || (s.datasets.length
      ? s.datasets.join(', ')
      : '(no data files)');
    btn.appendChild(titleEl);

    const ageEl = document.createElement('span');
    ageEl.className = 'session-age';
    ageEl.textContent = formatSessionAge(s.timestamp);
    btn.appendChild(ageEl);

    const dot = document.createElement('span');
    dot.className = 'session-busy-dot';
    dot.setAttribute('aria-hidden', 'true');
    btn.appendChild(dot);

    btn.addEventListener('click', () =>
      switchSession(s.path, s.path === currentPath)
    );
    row.appendChild(btn);

    // Folder-backed sessions (opened via the folder picker) are
    // not stored under ~/.sift-sessions/, so the backend's
    // delete_session and set_session_name reject them outright
    // (they require parent == SESSIONS_ROOT). Offering the
    // buttons here would prompt the researcher with a
    // destructive confirm or an editable name, then fail on
    // commit. Hide them so the controls match what the backend
    // will accept; the researcher manages their own project
    // dir for those sessions.
    if (s.kind !== 'folder') {
      const rename = document.createElement('button');
      rename.type = 'button';
      rename.className = 'session-rename';
      rename.setAttribute('aria-label', 'Rename session');
      rename.title = 'Rename this session';
      rename.textContent = '✎';
      rename.addEventListener('click', (e) => {
        e.stopPropagation();
        // Putting an <input> inside the <button.session-item>
        // would be invalid nesting (interactive in interactive),
        // so swap the whole button for an edit container. On
        // commit/cancel, beginRenameSession's loadSessions()
        // refresh redraws the row from server state.
        const editor = document.createElement('div');
        editor.className = 'session-item session-item-editing';
        editor.textContent = s.custom_name || s.title || '';
        row.replaceChild(editor, btn);
        beginRenameSession(editor, s.path);
      });
      row.appendChild(rename);

      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'session-delete';
      del.setAttribute('aria-label', 'Delete session');
      del.title = 'Delete this session (data + history)';
      del.textContent = '×';
      del.addEventListener('click', (e) => {
        e.stopPropagation();
        deleteSession(s, s.path === currentPath);
      });
      row.appendChild(del);
    } else {
      // Folder-backed row: no rename/delete (those mutate a real
      // project directory, see the comment above), but the
      // sidebar still needs SOME way to tidy up a folder the
      // researcher is done with -- otherwise every folder ever
      // opened via the picker stays pinned in the sidebar forever.
      // "Remove from list" only drops the recent-folders registry
      // entry (``forget_external_session`` -> ``external_sessions.
      // forget``); it never touches the folder or its ``.sift``
      // state, so this is safe even for the currently-focused
      // session.
      const forgetBtn = document.createElement('button');
      forgetBtn.type = 'button';
      forgetBtn.className = 'session-delete';
      forgetBtn.setAttribute('aria-label', 'Remove from sidebar list');
      forgetBtn.title = "Remove from this list (doesn't delete your files)";
      forgetBtn.textContent = '×';
      forgetBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        forgetExternalSession(s, s.path === currentPath);
      });
      row.appendChild(forgetBtn);
    }

    sidebarListEl.appendChild(row);
  });
}

function formatBytes(n) {
  if (!n) return '0 B';
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(0) + ' KB';
  if (n < 1024 * 1024 * 1024) return (n / (1024 * 1024)).toFixed(1) + ' MB';
  return (n / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
}

async function toggleSessionPinned(path, pinned) {
  /* Flip a session's pin-to-top flag through the bridge. The bridge
   * stamps ``pinned_at`` on the pin transition so the sort within
   * the pinned group surfaces the most-recently-pinned row first,
   * matching the researcher's most likely "I just pinned this, where
   * did it go" expectation.
   *
   * On success we re-fetch the session list rather than mutating the
   * row in place: a pin reorders the panel, which is easier to do
   * correctly from a fresh list_sessions render than by splicing the
   * existing DOM. The round-trip is cheap (list_sessions is a
   * directory walk).
   */
  if (!path) return;
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.set_session_pinned !== 'function') {
    toast('Restart Sift to enable session pinning.', 'info');
    return;
  }
  try {
    const res = await window.pywebview.api.set_session_pinned(path, pinned);
    if (!res || !res.ok) {
      toast(
        'Pin failed: ' + ((res && res.reason) || 'unknown'),
        'error',
      );
      return;
    }
  } catch (err) {
    console.warn('set_session_pinned failed', err);
    toast('Pin failed: ' + (err && err.message ? err.message : err), 'error');
    return;
  }
  if (typeof loadSessions === 'function') loadSessions();
}

async function deleteSession(s, isCurrent) {
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.delete_session !== 'function') {
    toast('Restart Sift to enable session deletion.', 'error');
    return;
  }
  const when = formatSessionWhen(s.timestamp);
  const sizeText = typeof s.size === 'number' ? ' (' + formatBytes(s.size) + ')' : '';
  // Active-session deletes get a more explicit prompt — the
  // researcher is wiping the chat they're currently looking at,
  // not a stale one in the sidebar. The default prompt covers
  // the data-loss content; the prefix makes the "you're in this
  // one right now" angle unambiguous.
  const headline = isCurrent
    ? `Delete the session you're currently in (from ${when})${sizeText}?`
    : `Delete session from ${when}${sizeText}?`;
  const ok = window.confirm(
    `${headline}\n\nThis removes the data copies, run logs, results.db, and chat history. Cannot be undone.`
  );
  if (!ok) return;
  try {
    const res = await window.pywebview.api.delete_session(s.path);
    if (!res || !res.ok) {
      const reason = res && res.reason ? res.reason : 'unknown';
      toast('Delete failed: ' + reason, 'error');
      return;
    }
    toast('Session deleted.', 'success');
    // Active-session delete: bridge dropped self.cwd; the page
    // must navigate to the landing screen to match. Without this
    // the chat surface still shows the (now stale) transcript and
    // the next send_message would crash on a missing cwd.
    if (res.was_active) {
      showLanding();
    } else {
      loadSessions();
    }
  } catch (err) {
    console.warn('delete_session failed', err);
    toast('Delete failed: ' + err, 'error');
  }
}

async function forgetExternalSession(s, isCurrent) {
  /* Folder-backed sessions' counterpart to deleteSession(): removes
   * the recent-folders registry entry only (nothing on disk is
   * touched -- see forget_external_session's docstring). No confirm
   * dialog, unlike delete: this is reversible in the sense that
   * re-opening the same folder via "Choose folder" re-registers it
   * immediately, and unlike delete it destroys nothing. */
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.forget_external_session !== 'function') {
    toast('Restart Sift to enable removing folders from the list.', 'error');
    return;
  }
  try {
    const res = await window.pywebview.api.forget_external_session(s.path);
    if (!res || !res.ok) {
      const reason = res && res.reason ? res.reason : 'unknown';
      toast('Could not remove from list: ' + reason, 'error');
      return;
    }
    // Unlike deleteSession, the active session is untouched here --
    // isCurrent only changes the toast wording (reassurance that
    // nothing closed), never navigation. The row just disappears
    // from the next loadSessions() paint.
    toast(
      isCurrent
        ? "Removed from sidebar list. You're still in this session."
        : 'Removed from sidebar list.',
      'success',
    );
    loadSessions();
  } catch (err) {
    console.warn('forget_external_session failed', err);
    toast('Could not remove from list: ' + err, 'error');
  }
}

function beginRenameSession(targetEl, path) {
  /* Swap the static label for an inline <input>, focused and
   * pre-filled with the current text. Save on Enter or blur,
   * cancel on Escape. Used by the topbar pill and by sidebar
   * rows — the call site passes whichever element should be
   * replaced for the duration of the edit.
   *
   * Empty / whitespace-only saves are intentional: they clear the
   * custom name and revert to the auto-derived label (single
   * dataset filename, "<first> +N", or session timestamp).
   */
  if (!targetEl || !path) return;
  if (targetEl.classList.contains('editing')) return;
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.set_session_name !== 'function') {
    toast('Restart Sift to enable session renaming.', 'info');
    return;
  }
  const original = targetEl.textContent || '';
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'session-rename-input';
  input.value = original;
  input.maxLength = 120;
  input.setAttribute('aria-label', 'Session name');
  // Cache layout context so we can put the static text back.
  const placeholder = document.createElement('span');
  placeholder.className = 'session-rename-placeholder';
  targetEl.classList.add('editing');
  targetEl.replaceChildren(placeholder, input);
  input.focus();
  input.select();

  let done = false;
  const restore = (text) => {
    if (done) return;
    done = true;
    targetEl.classList.remove('editing');
    targetEl.replaceChildren();
    targetEl.textContent = text;
    // Sidebar rows enter edit mode by swapping the row's button for
    // a stand-in edit container — restoring that container's text
    // alone won't bring back the rename/delete buttons. Re-rendering
    // the sidebar from server state is harmless for the topbar case
    // and necessary for the sidebar case.
    if (typeof loadSessions === 'function') loadSessions();
  };
  const commit = async () => {
    if (done) return;
    const next = input.value;
    // Block accidental double-fires (blur after Enter).
    done = true;
    targetEl.classList.remove('editing');
    targetEl.replaceChildren();
    // Show what the user typed immediately so the edit feels
    // local; the bridge call below confirms or corrects it.
    targetEl.textContent = next.trim() || original;
    try {
      const res = await window.pywebview.api.set_session_name(path, next);
      if (!res || !res.ok) {
        targetEl.textContent = original;
        toast(
          'Could not rename: ' + ((res && res.reason) || 'unknown'),
          'error',
        );
        return;
      }
      // Server's resolved title is authoritative — it falls back
      // to the auto-derived label when the input was empty.
      if (res.title) targetEl.textContent = res.title;
      // Sidebar rows also display the title; refresh so they
      // stay in sync.
      if (typeof loadSessions === 'function') loadSessions();
    } catch (err) {
      console.warn('set_session_name failed', err);
      targetEl.textContent = original;
      toast('Rename failed: ' + (err && err.message ? err.message : err), 'error');
    }
  };

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      commit();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      restore(original);
    }
  });
  input.addEventListener('blur', () => {
    // Don't commit if Escape already restored.
    if (!done) commit();
  });
  // Stop click bubbling so a sidebar-row rename doesn't also
  // trigger the row's switch-session handler.
  input.addEventListener('click', (e) => e.stopPropagation());
}

function formatSessionWhen(epochSeconds) {
  /* Human-friendly timestamp for a past session.
   * Today: "3:14 PM"
   * This year: "Apr 22, 3:14 PM"
   * Older: "Apr 22, 2025"
   */
  if (!epochSeconds) return '—';
  const d = new Date(epochSeconds * 1000);
  const now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  const sameYear = d.getFullYear() === now.getFullYear();
  const time = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  const date = d.toLocaleDateString([], { month: 'short', day: 'numeric' });
  if (sameDay) return time;
  if (sameYear) return `${date}, ${time}`;
  return `${date}, ${d.getFullYear()}`;
}

function formatSessionAge(epochSeconds) {
  /* Compact "time since inception" label for the session list.
   * < 1 min : "now"
   * < 1 h   : "Xm"
   * < 1 d   : "Xh"
   * < 30 d  : "Xd"
   * else    : "Xmo"
   */
  if (!epochSeconds) return '';
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - epochSeconds));
  if (secs < 60) return 'now';
  if (secs < 3600) return Math.floor(secs / 60) + 'm';
  if (secs < 86400) return Math.floor(secs / 3600) + 'h';
  if (secs < 86400 * 30) return Math.floor(secs / 86400) + 'd';
  return Math.floor(secs / (86400 * 30)) + 'mo';
}

async function switchSession(path, isCurrent) {
  if (isCurrent) return;  // already on it — click is a no-op
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.switch_session !== 'function') {
    // Bridge method isn't registered — almost always because the
    // backend process hasn't been restarted since this code landed.
    toast('Restart Sift to enable session switching.', 'info');
    return;
  }
  // Clear the LEAVING session's backend pending lists. ``showChat``
  // wipes the JS-side staged composer state (image thumbs, data
  // notices, mention chips) on the way in to the new session — but
  // the runner's ``pending_*`` lists are per-cwd and survive a focus
  // switch. Without this call, a script attachment / @-mention
  // staged in A but never sent rides invisibly with the next plain
  // message in A: the UI shows no chip, the backend silently
  // inlines the file. Best-effort: a missing bridge method or a
  // failing RPC just means we keep the prior behaviour, not a
  // hard error.
  const leavingCwd = currentCwd;
  if (
    leavingCwd
    && leavingCwd !== path
    && typeof window.pywebview.api.clear_pending_for_session === 'function'
  ) {
    try {
      await window.pywebview.api.clear_pending_for_session(leavingCwd);
    } catch (err) {
      console.warn('clear_pending_for_session failed', err);
    }
  }
  try {
    const res = await window.pywebview.api.switch_session(path);
    if (!res || !res.ok) {
      const reason = res && res.reason ? res.reason : 'unknown';
      toast('Session switch failed: ' + reason, 'error');
      return;
    }
    // showChat handles everything: hide landing, reveal chat, reset
    // transcript, refresh cwd pill + policy + sidebar + model chip,
    // hide the context chip, replay persisted history.
    showChat(res);
  } catch (err) {
    console.warn('switch_session failed', err);
    toast('Session switch failed: ' + (err && err.message ? err.message : err), 'error');
  }
}

// Arrow-key navigation inside the session list. When focus is on a
// session row, up/down moves between rows, Enter opens it, and
// Delete / Backspace deletes it (same flow as clicking the × icon).
if (sidebarListEl) {
  sidebarListEl.addEventListener('keydown', (e) => {
    const focused = document.activeElement;
    if (!focused || !sidebarListEl.contains(focused)) return;
    // Don't hijack typing keys when the user is inside the inline
    // rename input — Backspace would otherwise trigger the row's
    // delete confirm instead of erasing a character.
    if (
      focused.tagName === 'INPUT'
      || focused.tagName === 'TEXTAREA'
      || focused.isContentEditable
    ) {
      return;
    }
    const buttons = Array.from(
      sidebarListEl.querySelectorAll('.session-item')
    );
    if (buttons.length === 0) return;
    // Walk up from whatever inner element has focus to the button.
    const currentBtn = focused.closest('.session-item');
    const idx = buttons.indexOf(currentBtn);

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      const next = buttons[Math.min(idx + 1, buttons.length - 1)] || buttons[0];
      if (next) next.focus();
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      const prev = buttons[Math.max(idx - 1, 0)] || buttons[0];
      if (prev) prev.focus();
      return;
    }
    if (e.key === 'Delete' || e.key === 'Backspace') {
      if (!currentBtn) return;
      e.preventDefault();
      // Walk up to the .session-row, then click the × sibling so we
      // go through the same confirm+toast path as the mouse.
      const row = currentBtn.closest('.session-row');
      const del = row ? row.querySelector('.session-delete') : null;
      if (del) del.click();
    }
  });
}

if (sidebarToggleBtn) {
  const renderSidebarToggle = () => {
    const collapsed = sidebarEl.classList.contains('collapsed');
    sidebarToggleBtn.setAttribute('aria-expanded', String(!collapsed));
    sidebarToggleBtn.setAttribute(
      'aria-label', collapsed ? 'Expand sidebar' : 'Collapse sidebar'
    );
    sidebarToggleBtn.title = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
    sidebarToggleBtn.textContent = collapsed ? '›' : '‹';
  };
  sidebarToggleBtn.addEventListener('click', () => {
    const collapsed = sidebarEl.classList.toggle('collapsed');
    try { localStorage.setItem(SIDEBAR_COLLAPSED_KEY, collapsed ? '1' : '0'); }
    catch (_) {}
    renderSidebarToggle();
  });
  // Restore collapsed state on load.
  try {
    if (localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === '1') {
      sidebarEl.classList.add('collapsed');
    }
  } catch (_) {}
  renderSidebarToggle();
}

if (newSessionBtn) {
  newSessionBtn.addEventListener('click', () => {
    // Kick the user back to the landing screen so they can drop or
    // pick files for a brand-new session. The existing cwd stays
    // set on the backend until they finish staging, which is fine —
    // ui_ready will update on the next showChat call.
    showLanding();
  });
}

// ----- add files to current session --------------------------------------
// Lets a researcher drop more data into the active session without
// starting over. Uses the same native file-picker as landing, then
// copies into the current cwd on the backend.

const addFilesBtn = document.getElementById('add-files-btn');
if (addFilesBtn) {
  addFilesBtn.addEventListener('click', async () => {
    if (!window.pywebview || !window.pywebview.api) return;
    if (typeof window.pywebview.api.add_files !== 'function') {
      toast('Restart Sift to enable mid-session file adding.', 'info');
      return;
    }
    addFilesBtn.disabled = true;
    try {
      const res = await window.pywebview.api.add_files();
      if (!res || !res.ok) {
        const reason = res && res.reason ? res.reason : 'unknown';
        if (reason !== 'cancelled') {
          // Errors go into the main chat transcript, not a toast.
          // The researcher is looking at the conversation; a
          // message in-flow is clearer than a floating bubble.
          appendError(friendlyAddFilesError(reason));
        }
        return;
      }
      const added = res.added || [];
      const images = res.images || [];
      const skipped = res.skipped || [];

      // Stage any images the researcher picked as attachments on
      // the composer's next message. Images don't go into the
      // session cwd — they travel with the outgoing user message
      // so a vision-capable model can inspect them.
      images.forEach((img) => {
        const url = dataUrlFromBase64(img.data, img.mime);
        stagedImages.push({ data: img.data, mime: img.mime, url });
      });
      const addedNotices = addStagedDataNotices(added);
      if (images.length > 0 || addedNotices) renderAttachments();

      // Summary toast: describe what happened in one line.
      const parts = [];
      if (added.length === 1) parts.push('Added ' + added[0]);
      else if (added.length > 1) parts.push('Added ' + added.length + ' files');
      if (images.length === 1) parts.push('attached 1 image');
      else if (images.length > 1) parts.push('attached ' + images.length + ' images');
      if (parts.length > 0) {
        toast(parts.join(' · '), 'success');
      }
      if (skipped.length > 0) {
        toast('Skipped: ' + skipped.join(', '), 'info');
      }
      const skippedExisting = res.skipped_existing || [];
      if (skippedExisting.length > 0) {
        // Distinct toast colour so the researcher sees this is "we
        // didn't overwrite", not "we couldn't read these".
        toast(
          'Already in this session, not overwritten: '
          + skippedExisting.join(', '),
          'info'
        );
      }

      // Refresh permission chip (new data files get default
      // policy) and the Files panel; sidebar size also updates. The
      // topbar pill shows the cwd path now, not the auto-derived
      // session title — uploads don't change cwd, so the pill
      // doesn't need a repaint.
      if (res.policy) updatePolicyChip(res.policy);
      refreshFilesChip();
      loadSessions();
      // Script attachments inflate the next request — recount so the
      // chip reflects them.
      triggerContextRecount('attachment-add');
    } catch (err) {
      console.warn('add_files failed', err);
      appendError(friendlyAddFilesError(err && err.message ? err.message : String(err)));
    } finally {
      addFilesBtn.disabled = false;
    }
  });
}

// Turn the raw error reason from the Python bridge into something
// a researcher can act on. We strip the pywebview "dialog error:"
// prefix and the regex-complaint tail, which never help the user —
// if they see "not a valid file filter" they can't fix that.
function friendlyAddFilesError(raw) {
  const s = String(raw || 'unknown error').trim();
  if (/not a valid file filter/i.test(s)) {
    return "Couldn't open the file picker. Try restarting Sift.";
  }
  if (/no active session/i.test(s)) {
    return 'Start a session first. Drop files or pick a folder from the landing screen.';
  }
  if (/window not ready/i.test(s)) {
    return 'Sift is still starting up. Try again in a moment.';
  }
  // Strip the "dialog error:" prefix pywebview bubbles up.
  const cleaned = s.replace(/^dialog error:\s*/i, '');
  return "Couldn't add files: " + cleaned;
}

function dataUrlFromBase64(b64, mime) {
  // Reconstruct a data URL for <img src=...>. Browsers accept
  // either inline data URLs or blob URLs; data URL is simplest
  // when we already have the base64 string on hand.
  return `data:${mime || 'image/png'};base64,${b64}`;
}

// ----- toasts -------------------------------------------------------------
// Transient notifications that belong OUTSIDE the chat transcript:
// session switched, delete failed, model switched, "restart to
// enable X", etc. Auto-dismiss after 4 seconds, click to dismiss
// early. Chat-flow errors (turn_error, auth_failure) still go
// through appendError() so they land in the transcript alongside
// the turn they belong to.

// Single centered status line above the composer. Replaces the
// older anchored-bubble toasts that popped up in different spots
// depending on which chip fired them — researchers read those as
// "random popups". Every transient notice now lands in the same
// strip as plain dim text and auto-clears after 4s. The `anchor`
// argument is accepted for backwards-compat but ignored; everything
// flows through this one channel.
const statusLineEl = document.getElementById('status-line');
let statusClearTimer = null;

function toast(message, kind /*, anchor */) {
  if (!statusLineEl) { console.log('[status]', kind, message); return; }
  // Rebuild classes so kind-tinting doesn't accumulate across notices.
  statusLineEl.className = 'status-line visible ' + (kind || 'info');
  statusLineEl.textContent = message;
  if (statusClearTimer) {
    clearTimeout(statusClearTimer);
    statusClearTimer = null;
  }
  statusClearTimer = setTimeout(() => {
    statusLineEl.classList.remove('visible');
    // Leave text in place during the fade — the next notice
    // replaces it anyway, and empty content after a fade looks
    // abrupt.
    statusClearTimer = null;
  }, 4000);
}

// Kept so callers that imported this helper don't break; with the
// single-line design there's no bubble to dismiss.
function dismissToast() { /* no-op */ }

// ----- keyboard shortcuts ------------------------------------------------

const shortcutsOverlay = document.getElementById('shortcuts-overlay');
const shortcutsCloseBtn = document.getElementById('shortcuts-close');

let shortcutsReturnFocusEl = null;
function openShortcuts() {
  if (!shortcutsOverlay) return;
  // Reachable from anywhere via the global '?' shortcut (not just a
  // dedicated chip/button), so — like the evidence panel — capture
  // whatever had focus at open time rather than assuming a fixed
  // trigger element to return to.
  shortcutsReturnFocusEl = document.activeElement;
  shortcutsOverlay.classList.remove('hidden');
  if (shortcutsCloseBtn) shortcutsCloseBtn.focus();
}
function closeShortcuts() {
  if (shortcutsOverlay) shortcutsOverlay.classList.add('hidden');
  if (shortcutsReturnFocusEl && document.contains(shortcutsReturnFocusEl)) {
    shortcutsReturnFocusEl.focus();
  }
  shortcutsReturnFocusEl = null;
}
if (shortcutsCloseBtn) shortcutsCloseBtn.addEventListener('click', closeShortcuts);
if (shortcutsOverlay) {
  shortcutsOverlay.addEventListener('click', (e) => {
    if (e.target === shortcutsOverlay) closeShortcuts();
  });
}

// ----- dataset profile panel ---------------------------------------------
const dataChip = document.getElementById('data-chip');
const dataOverlay = document.getElementById('data-overlay');
const dataCloseBtn = document.getElementById('data-close');
const dataTabsEl = document.getElementById('data-tabs');
const dataSummaryEl = document.getElementById('data-summary');
const dataTableWrap = document.getElementById('data-table-wrap');

// WAI-ARIA tab pattern: role="tab" implies Left/Right/Home/End arrow-key
// navigation between tabs within a role="tablist" container, in addition
// to the plain Tab+Enter/Space that native <button> elements already give
// for free. Wired once per tablist container via event delegation so it
// keeps working across renderDatasetTabs() re-rendering its children.
// Shared by the Data panel tabs and the Privacy Inspector tabs,
// which use the same .data-tab / role="tab" markup pattern.
function wireTabListKeyboardNav(container) {
  if (!container) return;
  container.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight'
        && e.key !== 'Home' && e.key !== 'End') return;
    const tabs = Array.from(container.querySelectorAll('[role="tab"]'));
    if (!tabs.length) return;
    const currentIndex = tabs.indexOf(document.activeElement);
    if (currentIndex === -1) return;
    e.preventDefault();
    let nextIndex;
    if (e.key === 'ArrowRight') nextIndex = (currentIndex + 1) % tabs.length;
    else if (e.key === 'ArrowLeft') nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
    else if (e.key === 'Home') nextIndex = 0;
    else nextIndex = tabs.length - 1;
    tabs[nextIndex].focus();
    tabs[nextIndex].click();
  });
}
wireTabListKeyboardNav(dataTabsEl);

let knownDatasetNames = [];

function setKnownDatasets(policy) {
  /* The Permission payload already enumerates the session's data
   * files; reuse it rather than adding a second listing endpoint that
   * could drift from it. */
  const names = (policy && policy.datasets ? policy.datasets : [])
    .map((d) => (typeof d === 'string' ? d : d && d.name))
    .filter(Boolean);
  knownDatasetNames = names;
  if (dataChip) dataChip.classList.toggle('hidden', names.length === 0);
}

function openDataPanel() {
  if (!dataOverlay) return;
  dataOverlay.classList.remove('hidden');
  if (dataCloseBtn) dataCloseBtn.focus();
  renderDatasetTabs();
  if (knownDatasetNames.length) loadDatasetProfile(knownDatasetNames[0]);
}

function renderLinkage() {
  /* Join-key diagnostics between the session's datasets. Merge errors
   * — silently dropped unmatched records, many-to-many fan-out — are
   * the class that most often invalidates an otherwise correct
   * analysis, and they are invisible after the fact. */
  dataSummaryEl.textContent = 'Checking how these datasets link…';
  dataTableWrap.innerHTML = '';
  window.pywebview.api
    .get_linkage_report()
    .then((r) => {
      if (!r || !r.ok) {
        dataSummaryEl.textContent =
          'Could not check linkage: ' + ((r && r.reason) || 'unknown');
        return;
      }
      const pairs = r.pairs || [];
      dataSummaryEl.textContent = pairs.length
        ? pairs.length + ' dataset pair(s) share a plausible join key'
        : 'No shared join keys found between the datasets in this session.';
      const blocks = pairs.map((pair) => {
        const rows = pair.keys.map((k) => {
          const warns = (k.warnings || []).map(
            (w) => '<div class="link-warn">⚠ ' + escapeHtml(w) + '</div>'
          ).join('');
          return '<tr><td><code>' + escapeHtml(k.key) + '</code>' + warns +
            '</td><td>' + escapeHtml(k.relationship) + '</td><td>' +
            escapeHtml(k.left_match_pct) + '% / ' +
            escapeHtml(k.right_match_pct) + '%</td><td>' +
            escapeHtml(k.matched_keys) + '</td></tr>';
        }).join('');
        return '<p class="link-pair">' + escapeHtml(pair.left) + ' ↔ ' +
          escapeHtml(pair.right) + '</p>' +
          '<table class="ledger-table"><thead><tr><th>Key</th>' +
          '<th>Relationship</th><th>Matched (L/R)</th><th>Keys</th>' +
          '</tr></thead><tbody>' + rows + '</tbody></table>';
      }).join('');
      const skipped = (r.skipped || []).map(
        (s) => '<div class="link-warn">' + escapeHtml(s.dataset) + ': ' +
          escapeHtml(s.reason) + '</div>').join('');
      dataTableWrap.innerHTML = blocks + skipped;
    })
    .catch((e) => {
      dataSummaryEl.textContent = 'Could not check linkage: ' + e;
    });
}

function renderDatasetTabs(active) {
  if (!dataTabsEl) return;
  dataTabsEl.innerHTML = '';
  knownDatasetNames.forEach((name, i) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    const isActive = active === name || (!active && i === 0);
    btn.className = 'data-tab' + (isActive ? ' active' : '');
    btn.setAttribute('role', 'tab');
    btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
    btn.textContent = name;
    btn.addEventListener('click', () => {
      renderDatasetTabs(name);
      loadDatasetProfile(name);
    });
    dataTabsEl.appendChild(btn);
  });
  // Links tab: only meaningful with two or more datasets.
  if (knownDatasetNames.length > 1) {
    const linkBtn = document.createElement('button');
    linkBtn.type = 'button';
    linkBtn.className = 'data-tab' + (active === '__links__' ? ' active' : '');
    linkBtn.setAttribute('role', 'tab');
    linkBtn.setAttribute('aria-selected',
      active === '__links__' ? 'true' : 'false');
    linkBtn.textContent = 'Links';
    linkBtn.addEventListener('click', () => {
      renderDatasetTabs('__links__');
      renderLinkage();
    });
    dataTabsEl.appendChild(linkBtn);
  }
}

function loadDatasetProfile(name) {
  if (!window.pywebview || !window.pywebview.api) return;
  dataSummaryEl.textContent = 'Profiling ' + name + '…';
  dataTableWrap.innerHTML = '';
  window.pywebview.api
    .get_dataset_profile(name)
    .then((p) => {
      if (!p || !p.ok) {
        dataSummaryEl.textContent =
          'Could not profile this file: ' + ((p && p.reason) || 'unknown');
        return;
      }
      // Treat every profile field as optional at this renderer boundary.
      // Older sessions, an interrupted profiling worker, or a version-skewed
      // bridge can legitimately return a partial payload. A missing number
      // must never become an internal ``undefined.toLocaleString`` error in
      // the researcher-facing panel.
      const rows = Number(p.rows);
      const columns = Number(p.columns);
      const missingPct = Number(p.missing_pct);
      const bits = [];
      if (Number.isFinite(rows)) {
        const rowLabel = p.rows_exact ? rows.toLocaleString() : '~' + rows.toLocaleString();
        bits.push(rowLabel + ' rows');
      }
      if (Number.isFinite(columns)) bits.push(columns.toLocaleString() + ' variables');
      if (Number.isFinite(missingPct)) bits.push(missingPct + '% missing');
      if (typeof p.size_display === 'string' && p.size_display.trim()) {
        bits.push(p.size_display.trim());
      }
      if (p.duplicate_rows !== null && p.duplicate_rows !== undefined) {
        bits.push(p.duplicate_rows + ' duplicate rows');
      }
      const rowsProfiled = Number(p.rows_profiled);
      if (p.sampled && Number.isFinite(rowsProfiled)) {
        bits.push('profiled from the first ' +
          rowsProfiled.toLocaleString() + ' rows (large file)');
      }
      dataSummaryEl.textContent = bits.length
        ? bits.join(' · ')
        : 'Profile details are not available for this file yet.';
      if (Array.isArray(p.available_sheets) && p.available_sheets.length > 1) {
        // Multi-sheet .xlsx workbook: let the researcher pick which
        // worksheet this profile (and the model's own get_schema
        // view) uses. Selecting a sheet SAVES the choice immediately
        // (SiftBridge.set_dataset_excel_sheet) and reprofiles — no
        // separate "preview, then confirm" step, since re-picking is
        // just as cheap as picking the first time.
        const sheetRow = document.createElement('div');
        sheetRow.className = 'data-sheet-picker';
        const sheetLabel = document.createElement('label');
        sheetLabel.textContent = 'Worksheet: ';
        sheetLabel.htmlFor = 'data-sheet-select';
        const sheetSelect = document.createElement('select');
        sheetSelect.id = 'data-sheet-select';
        p.available_sheets.forEach((sheetName) => {
          const opt = document.createElement('option');
          opt.value = sheetName;
          opt.textContent = sheetName;
          if (sheetName === p.sheet_read
              || (p.sheet_read === 0 && sheetName === p.available_sheets[0])) {
            opt.selected = true;
          }
          sheetSelect.appendChild(opt);
        });
        sheetSelect.addEventListener('change', () => {
          window.pywebview.api
            .set_dataset_excel_sheet(name, sheetSelect.value)
            .then(() => loadDatasetProfile(name))
            .catch((e) => toast('Could not switch worksheet: ' + e, 'error'));
        });
        sheetLabel.appendChild(sheetSelect);
        sheetRow.appendChild(sheetLabel);
        dataSummaryEl.appendChild(sheetRow);
      }
      if (Array.isArray(p.survey_design_columns) && p.survey_design_columns.length) {
        const note = document.createElement('div');
        note.className = 'data-design-note';
        note.textContent =
          'Survey design variables detected (' +
          p.survey_design_columns
            .map((d) => d.name + ': ' + d.role).join(', ') +
          '). Estimates should use design-aware methods.';
        dataSummaryEl.appendChild(note);
      }
      if (Array.isArray(p.likely_target_candidates) && p.likely_target_candidates.length) {
        const note = document.createElement('div');
        note.className = 'data-target-note';
        const names = p.likely_target_candidates.map((c) => c.name).join(', ');
        note.textContent =
          'Possible outcome variable' +
          (p.likely_target_candidates.length === 1 ? '' : 's') + ': ' +
          names + ' — a heuristic guess based on name and column shape, ' +
          'not a claim about your research question.';
        note.title = p.likely_target_candidates
          .map((c) => c.name + ': ' + c.reasons.join('; '))
          .join('\n');
        dataSummaryEl.appendChild(note);
      }

      renderDatasetHealth(p);

      const targetNames = new Set(
        (p.likely_target_candidates || []).map((c) => c.name));
      const variableRows = (Array.isArray(p.variables) ? p.variables : []).map((v) => {
        const range = (v.min !== undefined && v.max !== undefined)
          ? escapeHtml(v.min) + ' – ' + escapeHtml(v.max) : '';
        let flag = v.flag
          ? '<span class="data-flag">' + escapeHtml(v.flag) + '</span>' : '';
        if (v.survey_design_role) {
          flag += '<span class="data-flag data-flag-design">' +
            escapeHtml(v.survey_design_role) + '</span>';
        }
        if (targetNames.has(v.name)) {
          flag += '<span class="data-flag data-flag-target"' +
            ' title="Heuristic guess — see the note above the table">' +
            'possible outcome</span>';
        }
        if (v.possible_missing_code !== undefined) {
          flag += '<span class="data-flag">' +
            escapeHtml(v.possible_missing_code) +
            ' may mean missing</span>';
        }
        // Semantic type rides alongside the pandas dtype rather than
        // replacing it -- dtype is the ground truth (what the file
        // actually stores), semantic_type is the best-effort "what
        // this probably MEANS" guess layered on top, and conflating
        // them would make the guess look like a fact.
        const semantic = v.semantic_type
          ? ' <span class="data-semantic-type">' +
            escapeHtml(v.semantic_type) + '</span>' : '';
        return '<tr><td>' + escapeHtml(v.name) + ' ' + flag + '</td><td>' +
          escapeHtml(v.dtype) + semantic + '</td><td>' +
          escapeHtml(v.missing_pct) +
          '%</td><td>' + escapeHtml(v.distinct == null ? '' : v.distinct) +
          '</td><td>' + range + '</td></tr>';
      }).join('');
      dataTableWrap.innerHTML =
        '<table class="ledger-table"><thead><tr><th>Variable</th>' +
        '<th>Type</th><th>Missing</th><th>Distinct</th><th>Range</th>' +
        '</tr></thead><tbody>' + variableRows + '</tbody></table>';
    })
    .catch(() => {
      dataSummaryEl.textContent =
        'Could not profile this file. Close and reopen Data, then try again.';
    });
}

function renderDatasetHealth(p) {
  /* Health summary above the variable table: a deterministic score
   * (see ``dataset_profile._compute_health`` — every point deducted
   * traces to a real computed check, nothing here is a model
   * opinion) plus the issue list it was built from. Absent entirely
   * on a failed profile; ``p.health`` only exists when ``p.ok``. */
  const existing = document.getElementById('data-health');
  if (existing) existing.remove();
  if (!p.health) return;

  const wrap = document.createElement('div');
  wrap.id = 'data-health';
  wrap.className = 'data-health';

  const score = p.health.score;
  const band = score >= 85 ? 'good' : (score >= 60 ? 'fair' : 'poor');
  const scoreRow = document.createElement('div');
  scoreRow.className = 'data-health-score-row';
  const badge = document.createElement('span');
  badge.className = 'data-health-badge data-health-' + band;
  badge.textContent = score + ' / 100';
  scoreRow.appendChild(badge);
  const label = document.createElement('span');
  label.className = 'data-health-label';
  label.textContent = (p.health.issues || []).length
    ? 'Data health — ' + p.health.issues.length +
      (p.health.issues.length === 1 ? ' thing worth a look' : ' things worth a look')
    : 'Data health — nothing flagged';
  scoreRow.appendChild(label);
  wrap.appendChild(scoreRow);

  if ((p.health.issues || []).length) {
    const list = document.createElement('ul');
    list.className = 'data-health-issues';
    p.health.issues.forEach((issue) => {
      const li = document.createElement('li');
      li.className = 'data-health-issue data-health-issue-' + issue.severity;
      li.textContent = issue.message;
      list.appendChild(li);
    });
    wrap.appendChild(list);

    const discussBtn = document.createElement('button');
    discussBtn.type = 'button';
    discussBtn.className = 'data-health-discuss-btn';
    discussBtn.textContent = 'Discuss these with Sift';
    discussBtn.title =
      'Sends a message asking Sift to walk through the flagged issues '
      + 'and propose how to handle them — nothing is changed '
      + 'automatically.';
    discussBtn.addEventListener('click', () => {
      const bullets = p.health.issues
        .map((i) => '- ' + i.message + (i.columns.length
          ? ' (' + i.columns.join(', ') + ')' : ''))
        .join('\n');
      input.value =
        'The data health panel for ' + p.name + ' flagged this:\n\n'
        + bullets
        + '\n\nWalk me through what\'s worth addressing and how you '
        + 'would handle each one. Don\'t change anything until I say so.';
      autosize();
      closeDataPanel();
      form.dispatchEvent(new Event('submit', { cancelable: true }));
    });
    wrap.appendChild(discussBtn);
  }

  dataSummaryEl.insertAdjacentElement('afterend', wrap);
}

function closeDataPanel() {
  if (dataOverlay) dataOverlay.classList.add('hidden');
  if (dataChip) dataChip.focus();
}

if (dataChip) dataChip.addEventListener('click', openDataPanel);
if (dataCloseBtn) dataCloseBtn.addEventListener('click', closeDataPanel);
if (dataOverlay) {
  dataOverlay.addEventListener('click', (e) => {
    if (e.target === dataOverlay) closeDataPanel();
  });
}

// ----- evidence panel -----------------------------------------------------
const evidenceOverlay = document.getElementById('evidence-overlay');
const evidenceCloseBtn = document.getElementById('evidence-close');
const evidenceContentEl = document.getElementById('evidence-content');
let evidenceReturnFocusEl = null;

function openEvidencePanel(resultId, triggerEl) {
  if (!evidenceOverlay || !window.pywebview || !window.pywebview.api) return;
  evidenceReturnFocusEl = triggerEl || document.activeElement;
  evidenceOverlay.classList.remove('hidden');
  if (evidenceContentEl) evidenceContentEl.textContent = 'Loading…';
  if (evidenceCloseBtn) evidenceCloseBtn.focus();
  window.pywebview.api
    .get_result_evidence(resultId)
    .then(renderEvidence)
    .catch((e) => {
      if (evidenceContentEl) {
        evidenceContentEl.textContent = 'Could not load evidence: ' + e;
      }
    });
}

function closeEvidencePanel() {
  if (evidenceOverlay) evidenceOverlay.classList.add('hidden');
  if (evidenceReturnFocusEl && document.contains(evidenceReturnFocusEl)) {
    evidenceReturnFocusEl.focus();
  }
  evidenceReturnFocusEl = null;
}

function renderEvidence(ev) {
  if (!evidenceContentEl) return;
  evidenceContentEl.innerHTML = '';
  if (!ev || !ev.ok) {
    const msg = document.createElement('div');
    msg.className = 'data-summary';
    msg.textContent = (ev && ev.reason)
      ? 'Could not load this result: ' + ev.reason
      : 'Could not load this result.';
    evidenceContentEl.appendChild(msg);
    return;
  }

  const meta = document.createElement('div');
  meta.className = 'evidence-meta';
  const bits = [ev.analysis_type];
  const evidenceSources = Array.isArray(ev.source_datasets) && ev.source_datasets.length
    ? ev.source_datasets : (ev.source_dataset ? [ev.source_dataset] : []);
  if (evidenceSources.length) bits.push('dataset' + (evidenceSources.length > 1 ? 's: ' : ': ') + evidenceSources.join(' + '));
  if (ev.n !== undefined && ev.n !== null) bits.push('n = ' + ev.n);
  if (ev.created_at) bits.push(new Date(ev.created_at).toLocaleString());
  meta.textContent = (ev.label || ev.result_id) + ' · ' + bits.join(' · ');
  evidenceContentEl.appendChild(meta);

  if (ev.challenge_summary) {
    const cs = ev.challenge_summary;
    const badge = document.createElement('div');
    badge.className = 'challenge-badge challenge-' + cs.verdict.toLowerCase();
    badge.style.marginTop = '10px';
    badge.textContent = cs.verdict + ' — ' + cs.agreeing + ' of ' + cs.total
      + ' alternative specification' + (cs.total === 1 ? '' : 's') + ' agree'
      + (ev.is_challenge_baseline === false ? ' (this is one alternative)' : '');
    evidenceContentEl.appendChild(badge);
  }

  if (ev.markdown && window.SiftMarkdown) {
    const tableWrap = document.createElement('div');
    tableWrap.className = 'result-markdown-body';
    tableWrap.style.marginTop = '12px';
    tableWrap.innerHTML = window.SiftMarkdown.render(ev.markdown);
    evidenceContentEl.appendChild(tableWrap);
  }

  if (ev.verification && Array.isArray(ev.verification.checks) &&
      ev.verification.checks.length) {
    const list = document.createElement('ul');
    list.className = 'data-health-issues';
    list.style.marginTop = '12px';
    ev.verification.checks.forEach((c) => {
      const li = document.createElement('li');
      li.className = 'data-health-issue data-health-issue-' +
        (c.status === 'warn' ? 'warn' : 'info');
      li.textContent = c.detail;
      list.appendChild(li);
    });
    evidenceContentEl.appendChild(list);
  }

  if (ev.script_code) {
    const details = document.createElement('details');
    details.className = 'tool-output-collapsed';
    details.style.marginTop = '12px';
    const summary = document.createElement('summary');
    summary.className = 'tool-output-summary';
    summary.textContent = 'Generated code (click to expand)';
    details.appendChild(summary);
    const pre = document.createElement('pre');
    pre.className = 'tool-output';
    pre.textContent = ev.script_code;
    details.appendChild(pre);
    evidenceContentEl.appendChild(details);
  }

  const privacy = document.createElement('p');
  privacy.className = 'feedback-note';
  privacy.style.marginTop = '12px';
  privacy.textContent = ev.privacy_note || '';
  evidenceContentEl.appendChild(privacy);
}

if (evidenceCloseBtn) evidenceCloseBtn.addEventListener('click', closeEvidencePanel);
if (evidenceOverlay) {
  evidenceOverlay.addEventListener('click', (e) => {
    if (e.target === evidenceOverlay) closeEvidencePanel();
  });
}

// Event delegation: any inline citation marker the model emits
// (rendered by markdown.js as ``<button class="evidence-cite"
// data-result-id="...">``) opens the panel for that result, from
// anywhere in the transcript, without each render site needing its
// own listener.
if (messagesEl) {
  messagesEl.addEventListener('click', (e) => {
    const cite = e.target.closest('.evidence-cite');
    if (cite && cite.dataset.resultId) {
      openEvidencePanel(cite.dataset.resultId, cite);
    }
  });
}

// ----- export modal ------------------------------------------------------
const exportChip = document.getElementById('export-chip');
const exportOverlay = document.getElementById('export-overlay');
const exportCloseBtn = document.getElementById('export-close');
const exportReplicationBtn = document.getElementById('export-replication');
const exportDisclosureBtn = document.getElementById('export-disclosure');
const exportReportBtn = document.getElementById('export-report');
const exportReportPdfBtn = document.getElementById('export-report-pdf');
const exportReportPptxBtn = document.getElementById('export-report-pptx');
const exportCodebookBtn = document.getElementById('export-codebook');
const exportResultEl = document.getElementById('export-result');

function openExport() {
  if (!exportOverlay) return;
  if (exportResultEl) exportResultEl.textContent = '';
  exportOverlay.classList.remove('hidden');
  if (exportCloseBtn) exportCloseBtn.focus();
}

function closeExport() {
  if (exportOverlay) exportOverlay.classList.add('hidden');
  if (exportChip) exportChip.focus();
}

function runExport(apiCall, busyLabel) {
  /* Shared handler for both export buttons. Disables the buttons for
   * the duration so a double-click can't launch two concurrent
   * exports into two timestamped directories. */
  if (!window.pywebview || !window.pywebview.api) return;
  const buttons = [exportReplicationBtn, exportDisclosureBtn, exportReportBtn,
    exportReportPdfBtn, exportReportPptxBtn, exportCodebookBtn];
  buttons.forEach((b) => { if (b) b.disabled = true; });
  if (exportResultEl) exportResultEl.textContent = busyLabel;
  apiCall()
    .then((res) => {
      if (!exportResultEl) return;
      if (res && res.ok) {
        const where = res.display_path || '';
        exportResultEl.textContent = 'Written to ' + where;
        toast('Export complete', 'info');
      } else {
        exportResultEl.textContent =
          'Export failed: ' + ((res && res.reason) || 'unknown error');
      }
    })
    .catch((e) => {
      if (exportResultEl) {
        exportResultEl.textContent = 'Export failed: ' + e;
      }
    })
    .finally(() => {
      buttons.forEach((b) => { if (b) b.disabled = false; });
      refreshFilesChip();
    });
}

if (exportChip) exportChip.addEventListener('click', openExport);
if (exportCloseBtn) exportCloseBtn.addEventListener('click', closeExport);
if (exportOverlay) {
  exportOverlay.addEventListener('click', (e) => {
    if (e.target === exportOverlay) closeExport();
  });
}
if (exportReplicationBtn) {
  exportReplicationBtn.addEventListener('click', () =>
    runExport(
      () => window.pywebview.api.export_replication_package(),
      'Building replication package…',
    ));
}
if (exportCodebookBtn) {
  exportCodebookBtn.addEventListener('click', () =>
    runExport(
      () => window.pywebview.api.export_codebook(),
      'Building codebook…',
    ));
}
if (exportReportBtn) {
  exportReportBtn.addEventListener('click', () =>
    runExport(
      () => window.pywebview.api.export_analysis_report(),
      'Building analysis report…',
    ));
}
if (exportReportPdfBtn) {
  exportReportPdfBtn.addEventListener('click', () =>
    runExport(
      () => window.pywebview.api.export_analysis_report_pdf(),
      'Building PDF report…',
    ));
}
if (exportReportPptxBtn) {
  exportReportPptxBtn.addEventListener('click', () =>
    runExport(
      () => window.pywebview.api.export_analysis_report_pptx(),
      'Building slide deck…',
    ));
}
if (exportDisclosureBtn) {
  exportDisclosureBtn.addEventListener('click', () =>
    runExport(
      () => window.pywebview.api.export_disclosure_report(),
      'Writing disclosure report…',
    ));
}

// ----- analysis checkpoints -----------------------------------------------
//
// A checkpoint is a non-destructive bookmark: create_checkpoint just
// records the current turn_index + visible result_ids, touching
// nothing else. Restore delegates to the existing rewind_to bridge
// method (same as editing a past message), so it inherits that
// method's full safety behaviour. Compare reads only already-model-
// visible result metadata -- no new privacy-boundary crossing.
const checkpointsChip = document.getElementById('checkpoints-chip');
const checkpointsChipLabel = document.getElementById('checkpoints-chip-label');
const checkpointsOverlay = document.getElementById('checkpoints-overlay');
const checkpointsCloseBtn = document.getElementById('checkpoints-close');
const checkpointsLabelInput = document.getElementById('checkpoints-label-input');
const checkpointsCreateBtn = document.getElementById('checkpoints-create-btn');
const checkpointsCreateResultEl = document.getElementById('checkpoints-create-result');
const checkpointsListEl = document.getElementById('checkpoints-list');
const checkpointsCompareBtn = document.getElementById('checkpoints-compare-btn');
const checkpointsCompareHintEl = document.getElementById('checkpoints-compare-hint');
const checkpointsCompareResultEl = document.getElementById('checkpoints-compare-result');
let checkpointsReturnFocusEl = null;
// Up to two ids the researcher has checked for comparison. A plain
// array (not a Set) keeps insertion order so "first checked = A,
// second checked = B" reads naturally in the compare output.
let checkpointsSelectedForCompare = [];

function formatCheckpointTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleString();
}

async function refreshCheckpointsChip() {
  if (!checkpointsChip || !window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.list_checkpoints !== 'function') return;
  let res;
  try {
    res = await window.pywebview.api.list_checkpoints();
  } catch (err) {
    console.warn('list_checkpoints failed', err);
    return;
  }
  const cps = (res && res.checkpoints) || [];
  if (!currentCwd) {
    checkpointsChip.classList.add('hidden');
    return;
  }
  checkpointsChip.classList.remove('hidden');
  if (checkpointsChipLabel) {
    checkpointsChipLabel.textContent =
      cps.length > 0 ? `Checkpoints · ${cps.length}` : 'Checkpoints';
  }
}

function updateCheckpointsCompareState() {
  const n = checkpointsSelectedForCompare.length;
  if (checkpointsCompareBtn) checkpointsCompareBtn.disabled = n !== 2;
  if (checkpointsCompareHintEl) {
    checkpointsCompareHintEl.textContent =
      n === 2
        ? 'Ready to compare.'
        : n === 1
          ? 'Pick one more checkpoint to compare.'
          : 'Pick two checkpoints below to compare.';
  }
}

function renderCheckpointsList(cps) {
  if (!checkpointsListEl) return;
  checkpointsListEl.innerHTML = '';
  checkpointsSelectedForCompare = checkpointsSelectedForCompare.filter(
    (id) => cps.some((c) => c.id === id),
  );
  updateCheckpointsCompareState();
  if (cps.length === 0) {
    const empty = document.createElement('p');
    empty.className = 'feedback-note';
    empty.textContent = 'No checkpoints yet in this session.';
    checkpointsListEl.appendChild(empty);
    return;
  }
  cps.forEach((cp) => {
    const row = document.createElement('div');
    row.className = 'checkpoint-row';

    const checkboxLabel = document.createElement('label');
    checkboxLabel.className = 'checkpoint-select';
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = checkpointsSelectedForCompare.includes(cp.id);
    checkbox.setAttribute('aria-label', `Select "${cp.label}" for comparison`);
    checkbox.addEventListener('change', () => {
      if (checkbox.checked) {
        if (checkpointsSelectedForCompare.length >= 2) {
          // Drop the oldest selection so the researcher can always
          // just check a new box rather than having to uncheck one
          // first -- max-2 acts as a rolling window, not a hard stop.
          const dropped = checkpointsSelectedForCompare.shift();
          const droppedBox = checkpointsListEl.querySelector(
            `input[data-checkpoint-id="${CSS.escape(dropped)}"]`,
          );
          if (droppedBox) droppedBox.checked = false;
        }
        checkpointsSelectedForCompare.push(cp.id);
      } else {
        checkpointsSelectedForCompare = checkpointsSelectedForCompare.filter(
          (id) => id !== cp.id,
        );
      }
      updateCheckpointsCompareState();
      if (checkpointsCompareResultEl) checkpointsCompareResultEl.innerHTML = '';
    });
    checkbox.dataset.checkpointId = cp.id;
    checkboxLabel.appendChild(checkbox);
    row.appendChild(checkboxLabel);

    const info = document.createElement('div');
    info.className = 'checkpoint-info';
    const labelEl = document.createElement('div');
    labelEl.className = 'checkpoint-label';
    labelEl.textContent = cp.label;
    const metaEl = document.createElement('div');
    metaEl.className = 'checkpoint-meta';
    const resultsText = cp.result_count === 1
      ? '1 result' : `${cp.result_count} results`;
    metaEl.textContent = `${resultsText} · ${formatCheckpointTime(cp.created_at)}`;
    info.appendChild(labelEl);
    info.appendChild(metaEl);
    row.appendChild(info);

    const actions = document.createElement('div');
    actions.className = 'checkpoint-actions';

    const restoreBtn = document.createElement('button');
    restoreBtn.type = 'button';
    restoreBtn.className = 'feedback-button';
    restoreBtn.textContent = 'Restore';
    restoreBtn.addEventListener('click', () => restoreCheckpoint(cp));
    actions.appendChild(restoreBtn);

    const deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.className = 'feedback-button';
    deleteBtn.textContent = 'Delete';
    deleteBtn.addEventListener('click', () => deleteCheckpointRow(cp));
    actions.appendChild(deleteBtn);

    row.appendChild(actions);
    checkpointsListEl.appendChild(row);
  });
}

async function loadCheckpointsList() {
  if (!checkpointsListEl || !window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.list_checkpoints !== 'function') return;
  let res;
  try {
    res = await window.pywebview.api.list_checkpoints();
  } catch (err) {
    checkpointsListEl.textContent = 'Could not load checkpoints: ' + err;
    return;
  }
  if (!res || !res.ok) {
    checkpointsListEl.textContent =
      'Could not load checkpoints: ' + ((res && res.reason) || 'unknown error');
    return;
  }
  renderCheckpointsList(res.checkpoints || []);
}

function openCheckpoints(triggerEl) {
  if (!checkpointsOverlay) return;
  checkpointsReturnFocusEl = triggerEl || document.activeElement;
  checkpointsOverlay.classList.remove('hidden');
  if (checkpointsCreateResultEl) checkpointsCreateResultEl.textContent = '';
  if (checkpointsCompareResultEl) checkpointsCompareResultEl.innerHTML = '';
  checkpointsSelectedForCompare = [];
  if (checkpointsLabelInput) {
    checkpointsLabelInput.value = '';
    checkpointsLabelInput.focus();
  } else if (checkpointsCloseBtn) {
    checkpointsCloseBtn.focus();
  }
  loadCheckpointsList();
}

function closeCheckpoints() {
  if (checkpointsOverlay) checkpointsOverlay.classList.add('hidden');
  if (checkpointsReturnFocusEl && document.contains(checkpointsReturnFocusEl)) {
    checkpointsReturnFocusEl.focus();
  }
  checkpointsReturnFocusEl = null;
}

async function createCheckpointFromInput() {
  if (!window.pywebview || !window.pywebview.api) return;
  const label = checkpointsLabelInput ? checkpointsLabelInput.value : '';
  if (!label || !label.trim()) {
    if (checkpointsCreateResultEl) {
      checkpointsCreateResultEl.textContent = 'Enter a label first.';
    }
    return;
  }
  if (checkpointsCreateBtn) checkpointsCreateBtn.disabled = true;
  if (checkpointsCreateResultEl) checkpointsCreateResultEl.textContent = 'Saving…';
  try {
    const res = await window.pywebview.api.create_checkpoint(label);
    if (!res || !res.ok) {
      const reason = (res && res.reason) || 'unknown error';
      if (checkpointsCreateResultEl) {
        checkpointsCreateResultEl.textContent = 'Could not save checkpoint: ' + reason;
      }
      return;
    }
    if (checkpointsCreateResultEl) checkpointsCreateResultEl.textContent = 'Saved.';
    if (checkpointsLabelInput) checkpointsLabelInput.value = '';
    await loadCheckpointsList();
    refreshCheckpointsChip();
  } catch (err) {
    if (checkpointsCreateResultEl) {
      checkpointsCreateResultEl.textContent = 'Could not save checkpoint: ' + err;
    }
  } finally {
    if (checkpointsCreateBtn) checkpointsCreateBtn.disabled = false;
  }
}

async function deleteCheckpointRow(cp) {
  if (!window.pywebview || !window.pywebview.api) return;
  const ok = window.confirm(`Delete checkpoint "${cp.label}"?`);
  if (!ok) return;
  try {
    const res = await window.pywebview.api.delete_checkpoint(cp.id);
    if (!res || !res.ok) {
      toast('Could not delete checkpoint: ' + ((res && res.reason) || 'unknown'), 'error');
      return;
    }
    await loadCheckpointsList();
    refreshCheckpointsChip();
  } catch (err) {
    toast('Could not delete checkpoint: ' + err, 'error');
  }
}

async function restoreCheckpoint(cp) {
  if (!window.pywebview || !window.pywebview.api) return;
  const ok = window.confirm(
    `Restore "${cp.label}"?\n\nThis rewinds the session to that point -- `
    + 'anything after it is hidden from the model (branching from here), '
    + 'the same as editing an earlier message.',
  );
  if (!ok) return;
  try {
    const res = await window.pywebview.api.restore_checkpoint(cp.id);
    if (!res || !res.ok) {
      toast('Could not restore checkpoint: ' + ((res && res.reason) || 'unknown'), 'error');
      return;
    }
    await replayHistory();
    await loadCheckpointsList();
    refreshCheckpointsChip();
    toast(`Restored "${cp.label}".`, 'success');
    closeCheckpoints();
  } catch (err) {
    toast('Could not restore checkpoint: ' + err, 'error');
  }
}

function renderCheckpointCompareGroup(title, items) {
  const wrap = document.createElement('div');
  wrap.className = 'checkpoint-compare-group';
  const h = document.createElement('div');
  h.className = 'checkpoint-compare-group-title';
  h.textContent = `${title} (${items.length})`;
  wrap.appendChild(h);
  if (items.length === 0) {
    const none = document.createElement('div');
    none.className = 'checkpoint-compare-empty';
    none.textContent = 'none';
    wrap.appendChild(none);
    return wrap;
  }
  const ul = document.createElement('ul');
  ul.className = 'checkpoint-compare-list';
  items.forEach((r) => {
    const li = document.createElement('li');
    const resultSources = Array.isArray(r.source_datasets) && r.source_datasets.length
      ? r.source_datasets : (r.source_dataset ? [r.source_dataset] : []);
    const dataset = resultSources.length ? ` · ${escapeHtml(resultSources.join(' + '))}` : '';
    li.innerHTML =
      `<strong>${escapeHtml(r.analysis_type)}</strong> — `
      + `${escapeHtml(r.label)}${dataset}`;
    ul.appendChild(li);
  });
  wrap.appendChild(ul);
  return wrap;
}

function renderCheckpointTally(title, tally) {
  const wrap = document.createElement('div');
  wrap.className = 'checkpoint-compare-tally';
  const h = document.createElement('span');
  h.className = 'checkpoint-compare-tally-title';
  h.textContent = title + ': ';
  wrap.appendChild(h);
  const entries = Object.entries(tally || {});
  if (entries.length === 0) {
    const span = document.createElement('span');
    span.textContent = 'no results';
    wrap.appendChild(span);
    return wrap;
  }
  wrap.appendChild(document.createTextNode(
    entries.map(([type, n]) => `${type} × ${n}`).join(', '),
  ));
  return wrap;
}

async function runCheckpointsCompare() {
  if (!window.pywebview || !window.pywebview.api) return;
  if (checkpointsSelectedForCompare.length !== 2) return;
  const [idA, idB] = checkpointsSelectedForCompare;
  if (checkpointsCompareBtn) checkpointsCompareBtn.disabled = true;
  if (checkpointsCompareResultEl) {
    checkpointsCompareResultEl.textContent = 'Comparing…';
  }
  try {
    const res = await window.pywebview.api.compare_checkpoints(idA, idB);
    if (!res || !res.ok) {
      if (checkpointsCompareResultEl) {
        checkpointsCompareResultEl.textContent =
          'Could not compare: ' + ((res && res.reason) || 'unknown error');
      }
      return;
    }
    if (!checkpointsCompareResultEl) return;
    checkpointsCompareResultEl.innerHTML = '';
    const summary = document.createElement('p');
    summary.className = 'feedback-note';
    summary.textContent =
      `Comparing "${res.checkpoint_a.label}" against "${res.checkpoint_b.label}".`;
    checkpointsCompareResultEl.appendChild(summary);
    checkpointsCompareResultEl.appendChild(
      renderCheckpointTally(res.checkpoint_a.label, res.tally_a),
    );
    checkpointsCompareResultEl.appendChild(
      renderCheckpointTally(res.checkpoint_b.label, res.tally_b),
    );
    checkpointsCompareResultEl.appendChild(
      renderCheckpointCompareGroup(`Only in "${res.checkpoint_a.label}"`, res.only_in_a),
    );
    checkpointsCompareResultEl.appendChild(
      renderCheckpointCompareGroup(`Only in "${res.checkpoint_b.label}"`, res.only_in_b),
    );
    checkpointsCompareResultEl.appendChild(
      renderCheckpointCompareGroup('In both', res.common),
    );
  } catch (err) {
    if (checkpointsCompareResultEl) {
      checkpointsCompareResultEl.textContent = 'Could not compare: ' + err;
    }
  } finally {
    updateCheckpointsCompareState();
  }
}

if (checkpointsChip) checkpointsChip.addEventListener('click', () => openCheckpoints(checkpointsChip));
if (checkpointsCloseBtn) checkpointsCloseBtn.addEventListener('click', closeCheckpoints);
if (checkpointsOverlay) {
  checkpointsOverlay.addEventListener('click', (e) => {
    if (e.target === checkpointsOverlay) closeCheckpoints();
  });
}
if (checkpointsCreateBtn) checkpointsCreateBtn.addEventListener('click', createCheckpointFromInput);
if (checkpointsLabelInput) {
  checkpointsLabelInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      createCheckpointFromInput();
    }
  });
}
if (checkpointsCompareBtn) checkpointsCompareBtn.addEventListener('click', runCheckpointsCompare);

// ----- privacy ledger modal ----------------------------------------------
const ledgerChip = document.getElementById('ledger-chip');
const ledgerOverlay = document.getElementById('ledger-overlay');
const ledgerCloseBtn = document.getElementById('ledger-close');
const ledgerSummaryEl = document.getElementById('ledger-summary');
const ledgerTableWrap = document.getElementById('ledger-table-wrap');

function escapeHtml(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function refreshLedgerChip() {
  /* Chip shows the running disclosure count for the focused session.
   * Hidden on the landing screen (no session) and when the bridge
   * isn't up yet. Count comes from the append-only ledger file. */
  if (!ledgerChip || !window.pywebview || !window.pywebview.api) return;
  if (!window.pywebview.api.get_release_ledger) return;
  window.pywebview.api
    .get_release_ledger(1)
    .then((res) => {
      if (!res || typeof res.count !== 'number') return;
      if (res.count > 0) {
        ledgerChip.classList.remove('hidden');
        if (exportChip) exportChip.classList.remove('hidden');
        document.getElementById('ledger-chip-label').textContent =
          'Ledger · ' + res.count;
      }
    })
    .catch(() => {});
}

// ----- Privacy Inspector tabs: Budgets / Patterns ------------------------
//
// Both panels are read-only advisory surfaces over data the backend
// already computes each time the panel opens (privacy_budget.py /
// differential_privacy.py / query_fingerprint.py) -- no state is
// held here beyond "which tab is active", matching the Data panel's
// existing tab pattern (renderDatasetTabs / loadDatasetProfile).

const privacyTabLedgerBtn = document.getElementById('privacy-tab-ledger');
const privacyTabBudgetsBtn = document.getElementById('privacy-tab-budgets');
const privacyTabPatternsBtn = document.getElementById('privacy-tab-patterns');
const privacyPanelLedger = document.getElementById('privacy-panel-ledger');
const privacyPanelBudgets = document.getElementById('privacy-panel-budgets');
const privacyPanelPatterns = document.getElementById('privacy-panel-patterns');
const privacyBudgetsWrap = document.getElementById('privacy-budgets-wrap');
const privacyPatternsWrap = document.getElementById('privacy-patterns-wrap');
wireTabListKeyboardNav(document.getElementById('privacy-tabs'));

function setPrivacyTab(tab) {
  const tabs = [
    [privacyTabLedgerBtn, privacyPanelLedger, 'ledger'],
    [privacyTabBudgetsBtn, privacyPanelBudgets, 'budgets'],
    [privacyTabPatternsBtn, privacyPanelPatterns, 'patterns'],
  ];
  tabs.forEach(([btn, panel, name]) => {
    const active = name === tab;
    if (btn) {
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-selected', active ? 'true' : 'false');
    }
    if (panel) panel.classList.toggle('hidden', !active);
  });
  if (tab === 'budgets') renderPrivacyBudgets();
  if (tab === 'patterns') renderPrivacyPatterns();
}

if (privacyTabLedgerBtn) {
  privacyTabLedgerBtn.addEventListener('click', () => setPrivacyTab('ledger'));
}
if (privacyTabBudgetsBtn) {
  privacyTabBudgetsBtn.addEventListener('click', () => setPrivacyTab('budgets'));
}
if (privacyTabPatternsBtn) {
  privacyTabPatternsBtn.addEventListener('click', () => setPrivacyTab('patterns'));
}

function tierBadge(tierLabel) {
  const cls = tierLabel === 'strict' ? 'ledger-chain-bad'
    : tierLabel === 'elevated' ? 'link-warn' : 'ledger-chain-ok';
  return '<span class="' + cls + '">' + escapeHtml(tierLabel) + '</span>';
}

function renderPrivacyBudgets() {
  if (!privacyBudgetsWrap) return;
  if (!window.pywebview || !window.pywebview.api) return;
  privacyBudgetsWrap.innerHTML = '<p class="feedback-note">Loading…</p>';
  const suppression = window.pywebview.api.get_privacy_budget_status
    ? window.pywebview.api.get_privacy_budget_status()
    : Promise.resolve({ datasets: [] });
  const epsilon = window.pywebview.api.get_epsilon_budget_status
    ? window.pywebview.api.get_epsilon_budget_status()
    : Promise.resolve({ datasets: [] });
  Promise.all([suppression, epsilon])
    .then(([sup, eps]) => {
      const supRows = (sup.datasets || []).map((d) => {
        const usage = d.unbounded ? 'unbounded' :
          d.consumed + ' / ' + d.budget;
        return '<tr><td>' + escapeHtml(d.name) + '</td><td>' +
          escapeHtml(d.privacy_profile) + '</td><td>' +
          escapeHtml(usage) + '</td><td>' +
          tierBadge(d.tier_label) + '</td></tr>';
      }).join('');
      const supTable = '<table class="ledger-table"><thead><tr>' +
        '<th>Dataset</th><th>Profile</th><th>Granted releases</th>' +
        '<th>Suppression tier</th></tr></thead><tbody>' +
        (supRows || '<tr><td colspan="4">No datasets tracked yet.</td></tr>') +
        '</tbody></table>';

      const epsRows = (eps.datasets || []).map((d) => {
        const usage = d.unbounded ? 'unbounded' :
          d.spent.toFixed(3) + ' / ' + d.cap;
        return '<tr><td>' + escapeHtml(d.name) + '</td><td>' +
          escapeHtml(d.privacy_profile) + '</td><td>' +
          escapeHtml(d.dp_epsilon) + '</td><td>' +
          escapeHtml(usage) + '</td></tr>';
      }).join('');
      const epsTable = eps.datasets && eps.datasets.length
        ? '<table class="ledger-table"><thead><tr>' +
          '<th>Dataset</th><th>Profile</th><th>Per-query epsilon</th>' +
          '<th>Session spend / cap</th></tr></thead><tbody>' +
          epsRows + '</tbody></table>'
        : '<p class="feedback-note">No dataset has differential privacy ' +
          'enabled. It is opt-in only -- nothing here happens by default.</p>';

      privacyBudgetsWrap.innerHTML =
        '<h4 class="privacy-subhead">Adaptive suppression</h4>' + supTable +
        '<h4 class="privacy-subhead">Differential privacy (noisy_count)</h4>' +
        epsTable;
    })
    .catch((e) => {
      privacyBudgetsWrap.innerHTML =
        '<p class="feedback-note">Could not load budget status: ' +
        escapeHtml(String(e)) + '</p>';
    });
}

function renderPrivacyPatterns() {
  if (!privacyPatternsWrap) return;
  if (!window.pywebview || !window.pywebview.api) return;
  if (!window.pywebview.api.get_query_fingerprint_report) {
    privacyPatternsWrap.innerHTML = '';
    return;
  }
  privacyPatternsWrap.innerHTML = '<p class="feedback-note">Loading…</p>';
  window.pywebview.api
    .get_query_fingerprint_report()
    .then((r) => {
      if (!r || r.is_empty) {
        privacyPatternsWrap.innerHTML =
          '<p class="feedback-note">No repeated-query, combined-release, ' +
          'or differencing patterns detected yet this session.</p>';
        return;
      }
      const sections = [];
      if ((r.repeated_queries || []).length) {
        const rows = r.repeated_queries.map((f) =>
          '<tr><td>' + escapeHtml(f.dataset) + '</td><td>' +
          escapeHtml(f.variable) + '</td><td>' + escapeHtml(f.count) +
          '</td><td>' + escapeHtml((f.request_types || []).join(', ')) +
          '</td></tr>').join('');
        sections.push('<h4 class="privacy-subhead">Repeated queries</h4>' +
          '<table class="ledger-table"><thead><tr><th>Dataset</th>' +
          '<th>Variable</th><th>Calls</th><th>Request types</th>' +
          '</tr></thead><tbody>' + rows + '</tbody></table>');
      }
      if ((r.combined_release_variables || []).length) {
        const rows = r.combined_release_variables.map((f) =>
          '<tr><td>' + escapeHtml(f.dataset) + '</td><td>' +
          escapeHtml(f.variable) + '</td><td>' +
          escapeHtml((f.request_types || []).join(', ')) +
          '</td></tr>').join('');
        sections.push('<h4 class="privacy-subhead">Combined releases ' +
          '(many fact types, one variable)</h4>' +
          '<table class="ledger-table"><thead><tr><th>Dataset</th>' +
          '<th>Variable</th><th>Request types</th></tr></thead><tbody>' +
          rows + '</tbody></table>');
      }
      if ((r.differencing_candidates || []).length) {
        const rows = r.differencing_candidates.map((f) =>
          '<tr><td>' + escapeHtml(f.dataset) + '</td><td>' +
          escapeHtml(f.analysis_type) + '</td><td>' +
          escapeHtml(f.observation_count) + '</td><td>' +
          escapeHtml((f.distinct_n_values || []).join(', ')) +
          '</td></tr>').join('');
        sections.push('<h4 class="privacy-subhead">Differencing candidates ' +
          '(N drift on the same analysis)</h4>' +
          '<table class="ledger-table"><thead><tr><th>Dataset</th>' +
          '<th>Analysis type</th><th>Observations</th>' +
          '<th>Distinct N values</th></tr></thead><tbody>' +
          rows + '</tbody></table>');
      }
      privacyPatternsWrap.innerHTML = sections.join('');
    })
    .catch((e) => {
      privacyPatternsWrap.innerHTML =
        '<p class="feedback-note">Could not load pattern report: ' +
        escapeHtml(String(e)) + '</p>';
    });
}

function openLedger() {
  if (!ledgerOverlay) return;
  ledgerOverlay.classList.remove('hidden');
  if (ledgerCloseBtn) ledgerCloseBtn.focus();
  setPrivacyTab('ledger');
  if (!window.pywebview || !window.pywebview.api) return;
  window.pywebview.api
    .get_release_ledger(200)
    .then((res) => {
      if (!res) return;
      const okBadge = res.chain_ok
        ? '<span class="ledger-chain-ok">chain verified</span>'
        : '<span class="ledger-chain-bad">chain INCONSISTENT — ' +
          escapeHtml(res.detail) + '</span>';
      ledgerSummaryEl.innerHTML =
        res.count + ' disclosure' + (res.count === 1 ? '' : 's') +
        ' sent to the model this session · ' + okBadge +
        '. Recorded locally in an append-only, hash-chained log.';
      renderUsageLine();
      const rows = (res.records || []).map((r) => {
        const facts = r.facts || {};
        const args = r.args || {};
        const factSources = Array.isArray(facts.source_datasets) ? facts.source_datasets.join(' + ') : '';
        const what = args.dataset || args.result_id || factSources || facts.source_dataset ||
          (r.extra && r.extra.filename) || '';
        const n = facts.n != null ? facts.n : '';
        const ts = (r.ts || '').replace('T', ' ').replace('+00:00', ' UTC');
        return '<tr><td>' + escapeHtml(ts) + '</td><td>' +
          escapeHtml(r.tool || r.kind) + '</td><td>' +
          escapeHtml(what) + '</td><td>' + escapeHtml(n) + '</td><td>' +
          escapeHtml((r.response_sha256 || '').slice(0, 12)) + '</td></tr>';
      }).join('');
      ledgerTableWrap.innerHTML =
        '<table class="ledger-table"><thead><tr>' +
        '<th>Time</th><th>Operation</th><th>Subject</th><th>N</th>' +
        '<th>Payload hash</th></tr></thead><tbody>' +
        (rows || '<tr><td colspan="5">No disclosures recorded yet.</td></tr>') +
        '</tbody></table>';
    })
    .catch(() => {});
}

function renderUsageLine() {
  /* Session token + spend accounting, appended to the ledger panel.
   *
   * Presentation rule: a provider-reported cost is a measurement and
   * is shown as such; a rate-table figure is an estimate and is
   * always labelled, with the date the rates were recorded. When no
   * rate is known the cost is omitted entirely — never rendered as
   * $0.00, which would read as "free". */
  if (!window.pywebview || !window.pywebview.api) return;
  if (!window.pywebview.api.get_usage_summary) return;
  window.pywebview.api
    .get_usage_summary()
    .then((u) => {
      if (!u || !ledgerSummaryEl) return;
      const el = document.createElement('div');
      el.className = 'usage-line';
      const bits = [
        u.turns + ' turn' + (u.turns === 1 ? '' : 's'),
        (u.total_tokens || 0).toLocaleString() + ' tokens',
      ];
      if (typeof u.reported_cost_usd === 'number') {
        bits.push('$' + u.reported_cost_usd.toFixed(2) +
          ' billed (reported by provider)');
      } else if (typeof u.estimated_cost_usd === 'number') {
        bits.push('~$' + u.estimated_cost_usd.toFixed(2) +
          ' estimated' + (u.complete ? '' : ', partial') +
          ' at ' + escapeHtml(u.rates_as_of) + ' list rates');
      }
      el.textContent = bits.join(' · ');
      ledgerSummaryEl.appendChild(el);
    })
    .catch(() => {});
}

function closeLedger() {
  if (ledgerOverlay) ledgerOverlay.classList.add('hidden');
  if (ledgerChip) ledgerChip.focus();
}

if (ledgerChip) ledgerChip.addEventListener('click', openLedger);
if (ledgerCloseBtn) ledgerCloseBtn.addEventListener('click', closeLedger);
if (ledgerOverlay) {
  ledgerOverlay.addEventListener('click', (e) => {
    if (e.target === ledgerOverlay) closeLedger();
  });
}

function activeModal() {
  const dialogs = Array.from(
    document.querySelectorAll('[role="dialog"][aria-modal="true"]')
  );
  return dialogs.reverse().find((dialog) =>
    !dialog.classList.contains('hidden')
    && getComputedStyle(dialog).display !== 'none'
  ) || null;
}

function trapModalTab(event, modal) {
  if (!modal || event.key !== 'Tab') return false;
  const focusable = Array.from(modal.querySelectorAll(
    'button:not([disabled]), a[href], input:not([disabled]), '
    + 'textarea:not([disabled]), select:not([disabled]), '
    + '[tabindex]:not([tabindex="-1"])'
  )).filter((element) => {
    const style = getComputedStyle(element);
    return style.display !== 'none' && style.visibility !== 'hidden';
  });
  if (focusable.length === 0) {
    event.preventDefault();
    modal.focus();
    return true;
  }
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
    return true;
  }
  if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
    return true;
  }
  if (!modal.contains(document.activeElement)) {
    event.preventDefault();
    first.focus();
    return true;
  }
  return false;
}

function isTypingInField(el) {
  if (!el) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || el.isContentEditable;
}

document.addEventListener('keydown', (e) => {
  const cmd = e.metaKey || e.ctrlKey;
  const modal = activeModal();

  if (trapModalTab(e, modal)) return;

  // Cmd-R: reload. Works from anywhere including the composer.
  if (cmd && e.key.toLowerCase() === 'r' && !e.shiftKey && !e.altKey) {
    e.preventDefault();
    location.reload();
    return;
  }

  // Cmd-K: focus composer.
  if (cmd && e.key.toLowerCase() === 'k' && !e.shiftKey && !e.altKey) {
    if (modal) {
      e.preventDefault();
      return;
    }
    e.preventDefault();
    if (input) { input.focus(); input.select?.(); }
    return;
  }

  // Escape: close the shortcuts overlay or any open popup.
  if (e.key === 'Escape') {
    if (updatesOverlay && !updatesOverlay.classList.contains('hidden')) {
      closeUpdates();
      return;
    }
    if (evidenceOverlay && !evidenceOverlay.classList.contains('hidden')) {
      closeEvidencePanel();
      return;
    }
    if (ledgerOverlay && !ledgerOverlay.classList.contains('hidden')) {
      closeLedger();
      return;
    }
    if (exportOverlay && !exportOverlay.classList.contains('hidden')) {
      closeExport();
      return;
    }
    if (checkpointsOverlay && !checkpointsOverlay.classList.contains('hidden')) {
      closeCheckpoints();
      return;
    }
    if (dataOverlay && !dataOverlay.classList.contains('hidden')) {
      closeDataPanel();
      return;
    }
    if (shortcutsOverlay && !shortcutsOverlay.classList.contains('hidden')) {
      closeShortcuts();
      return;
    }
    if (policyPopup && !policyPopup.classList.contains('hidden')) {
      policyPopup.classList.add('hidden');
      if (policyChip) {
        policyChip.classList.remove('open');
        policyChip.setAttribute('aria-expanded', 'false');
      }
      return;
    }
    if (filesPopup && !filesPopup.classList.contains('hidden')) {
      filesPopup.classList.add('hidden');
      if (filesChip) {
        filesChip.classList.remove('open');
        filesChip.setAttribute('aria-expanded', 'false');
      }
      return;
    }
    if (modelPopup && !modelPopup.classList.contains('hidden')) {
      modelPopup.classList.add('hidden');
      if (modelChip) {
        modelChip.classList.remove('open');
        modelChip.setAttribute('aria-expanded', 'false');
      }
      return;
    }
  }

  // `?` opens the shortcuts overlay, but only when not typing in
  // the composer (otherwise a literal question mark never makes it
  // into the prompt).
  if (e.key === '?' && !isTypingInField(document.activeElement)) {
    if (modal) {
      e.preventDefault();
      return;
    }
    e.preventDefault();
    openShortcuts();
    return;
  }
});

// Signal to the Python side that we're ready to receive events. pywebview
// sets window.pywebview once its bridge is ready; until then we wait.
function whenReady(fn) {
  if (window.pywebview && window.pywebview.api) return fn();
  window.addEventListener('pywebviewready', fn, { once: true });
}

// Rebuild ``busySessions`` from the bridge's view of which runners
// have a turn in flight. Called on every page boot — both initial
// load and after a hard reload (Cmd+Shift+R) — because
// ``busySessions`` is JS module-scope state that gets wiped on
// every navigation. Without this, refreshing during a turn drops
// the loading indicator AND the sidebar busy dot even though the
// backend is still streaming events.
//
// Logs to the console so the researcher can verify the seed path
// Environment health banner. Calls ``doctor_report()`` (see
// ``ui.py`` -- the Python-side method existed for a while with no
// caller; this is the wiring its own docstring describes: "the UI
// can render the same checks as a banner"). Shows a dismissible
// notice when a runtime the researcher might reach for is blocked
// (script submission in that language will fail outright) or merely
// degraded (runs, but a feature like plotting silently drops).
//
// Called once at startup (both branches that land on showChat) and
// again after an install_packages approval resolves -- NOT on every
// session switch, since runtime health is a machine-wide property,
// not a per-session one, and re-probing on every sidebar click would
// be both wasteful and naggy.
async function refreshDoctorBanner() {
  if (!doctorBannerEl || !window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.doctor_report !== 'function') {
    // Older bridge build without the method; nothing to show.
    return;
  }
  let report;
  try {
    report = await window.pywebview.api.doctor_report();
  } catch (err) {
    console.warn('doctor_report failed', err);
    return;
  }
  if (!report || !Array.isArray(report.runtimes)) return;

  // ``run_doctor``'s own docstring is explicit: "``blocked`` on the
  // returned report means script execution will fail. The CLI maps
  // this to exit code 1; the UI banner maps it to disabling the
  // chat input." The banner text alone doesn't satisfy that -- a
  // researcher could read "Environment issue" and still type and
  // submit a script that is GUARANTEED to fail (no sandbox backend,
  // or every language blocked), reintroducing in softer form the
  // exact "confidently wrong, no actionable stop" failure mode this
  // whole feature exists to close. Applied unconditionally, ahead of
  // the dismiss-driven early return below, so dismissing the BANNER
  // (a "stop showing me this text" action) never re-enables the
  // composer -- only an actual re-check that comes back healthy does.
  applyDoctorBlockedComposerState(!!report.blocked);

  // An optional language that is simply not installed is not an app
  // problem. In particular, opening a Stata .dta file uses Sift's bundled
  // reader and does not require a paid Stata installation. Show unavailable
  // languages only when *every* language is unavailable and the report is
  // therefore blocked overall. Present-but-broken runtimes and degraded
  // features remain visible.
  const unhealthy = report.runtimes.filter((r) => r && (
    r.status === 'blocked'
    || r.status === 'warning'
    || (report.blocked && r.status === 'unavailable')
  ));
  if (unhealthy.length === 0 || doctorBannerDismissed) {
    doctorBannerEl.classList.add('hidden');
    return;
  }

  const parts = unhealthy.map((r) => {
    const advice = (r.advice || [])[0];
    return advice
      ? `${r.runtime}: ${advice}`
      : `${r.runtime}: ${r.detail || 'unavailable'}`;
  });
  doctorBannerTextEl.textContent =
    (report.blocked ? 'Environment issue — ' : 'Environment note — ')
    + parts.join('  •  ');
  doctorBannerEl.classList.toggle('doctor-banner-blocked', !!report.blocked);
  doctorBannerEl.classList.remove('hidden');
}

function applyDoctorBlockedComposerState(blocked) {
  // Disables the actual textarea + send button (not just hiding
  // them, since the busy/idle send<->stop swap already uses
  // classList visibility toggling elsewhere -- .disabled composes
  // safely with that regardless of which button is currently
  // visible). Idempotent: safe to call every refreshDoctorBanner
  // tick even when the blocked state hasn't changed.
  if (input) {
    input.disabled = blocked;
    input.title = blocked
      ? 'Script execution is blocked in this environment -- see the banner above.'
      : '';
  }
  if (sendBtn) sendBtn.disabled = blocked;
}

if (doctorBannerDismissBtn) {
  doctorBannerDismissBtn.addEventListener('click', () => {
    doctorBannerDismissed = true;
    if (doctorBannerEl) doctorBannerEl.classList.add('hidden');
  });
}

// fired (DevTools → Console). If you see "seed: API method missing"
// after a refresh, sift was started with an older Python image
// that didn't have ``list_busy_sessions`` — fully quit and relaunch.
async function seedBusySessions() {
  if (!window.pywebview || !window.pywebview.api) {
    console.log('[sift] seed: pywebview bridge not ready yet');
    return;
  }
  if (typeof window.pywebview.api.list_busy_sessions !== 'function') {
    console.log(
      '[sift] seed: API method missing — bridge predates ' +
      'list_busy_sessions; fully restart sift to pick it up'
    );
    return;
  }
  try {
    const res = await window.pywebview.api.list_busy_sessions();
    if (!res || !res.ok || !Array.isArray(res.cwds)) {
      console.log('[sift] seed: bridge returned unexpected shape', res);
      return;
    }
    busySessions.clear();
    res.cwds.forEach((cwd) => busySessions.add(cwd));
    console.log(
      `[sift] seed: ${res.cwds.length} busy session(s) — `,
      res.cwds,
    );
  } catch (err) {
    console.warn('[sift] seed: list_busy_sessions failed', err);
  }
}

whenReady(async () => {
  // Three-stage state machine on startup:
  //   needs_auth     → researcher hasn't configured any provider yet.
  //                    Show the auth screen first so the chat can't
  //                    silently fail with "no API key" later.
  //   needs_session  → auth is good but no working directory chosen.
  //                    Land on the drop / choose-files screen.
  //   ready          → both done; jump into chat.
  // Any exception from ui_ready falls back to showLanding() — better
  // to be too permissive than to wedge the page on a hard error.
  try {
    const state = await window.pywebview.api.ui_ready();
    // Seed the busy-set BEFORE branching into showChat / showLanding.
    // showChat → syncComposerToFocus reads busySessions to decide
    // whether to show the loading indicator + Stop button, and
    // loadSessions reads it to decide whether to paint the sidebar
    // dot. Both fire inside showChat, so the seed has to land first
    // or the first paint shows an idle UI for a busy session.
    await seedBusySessions();
    if (state && state.state === 'needs_auth') {
      showAuth(state.auth);
    } else if (state && state.state === 'ready') {
      showChat(state);
      refreshDoctorBanner();
    } else {
      showLanding();
    }
  } catch (err) {
    console.error('ui_ready failed', err);
    showLanding();
  }
});
