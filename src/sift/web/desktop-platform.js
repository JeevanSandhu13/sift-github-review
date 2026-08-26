/* Small, dependency-free platform adapter for the shared desktop shell.
 *
 * Sift intentionally keeps one frontend across macOS, Windows, and Linux.
 * Platform detection is used only for native typography and shortcut labels;
 * no product behaviour or security decision depends on user-agent text.
 */
(function initializeDesktopPlatform() {
  'use strict';

  function normalizePlatform(value) {
    const raw = String(value || '').toLowerCase();
    if (raw.includes('mac') || raw.includes('darwin') || raw.includes('cocoa')) {
      return 'macos';
    }
    if (raw.includes('win') || raw.includes('edgechromium') || raw.includes('mshtml')) {
      return 'windows';
    }
    if (raw.includes('linux') || raw.includes('qt') || raw.includes('gtk')) {
      return 'linux';
    }
    return 'unknown';
  }

  function browserPlatform() {
    const modern = navigator.userAgentData && navigator.userAgentData.platform;
    return normalizePlatform(modern || navigator.platform || navigator.userAgent);
  }

  function applyPlatform(value) {
    const platform = normalizePlatform(value);
    const resolved = platform === 'unknown' ? browserPlatform() : platform;
    document.documentElement.dataset.platform = resolved;
    const modifier = resolved === 'macos' ? '⌘' : 'Ctrl';
    document.querySelectorAll('[data-shortcut-mod]').forEach((element) => {
      element.textContent = modifier;
    });
  }

  applyPlatform(browserPlatform());

  // pywebview's renderer name is more authoritative than the browser user
  // agent, which can be deliberately generic. Reconcile once its bridge is
  // available, while retaining the browser-derived fallback for static QA.
  window.addEventListener('pywebviewready', () => {
    if (window.pywebview && window.pywebview.platform) {
      applyPlatform(window.pywebview.platform);
    }
  }, { once: true });
})();
