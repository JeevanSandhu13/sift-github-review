/* Browser-only bridge fixture used for visual QA of the production web shell.
 * The release bundle never loads this file; the local preview server injects it
 * ahead of app.js while testing responsive and accessibility states.
 */
(function installPreviewBridge() {
  const now = Math.floor(Date.now() / 1000);
  const sessions = [
    {
      path: '/Users/research/.sift-sessions/clinical-outcomes',
      title: 'Clinical outcomes',
      datasets: ['outcomes.parquet'],
      timestamp: now - 420,
      kind: 'managed',
      pinned: true,
      size: 12800000,
    },
    {
      path: '/Users/research/.sift-sessions/survey-wave-4',
      title: 'Survey wave 4',
      datasets: ['responses.csv'],
      timestamp: now - 86400,
      kind: 'managed',
      pinned: false,
      size: 3200000,
    },
    {
      path: '/Users/research/projects/replication-study',
      title: 'Replication study',
      datasets: ['panel.dta'],
      timestamp: now - 86400 * 5,
      kind: 'folder',
      pinned: false,
      size: 74000000,
    },
  ];

  const policy = {
    datasets: [{
      name: 'outcomes.parquet',
      depth: 'names_types_labels_summary',
      max_depth: 'names_types_labels_summary',
    }],
  };

  const view = new URLSearchParams(location.search).get('preview') || 'chat';
  const auth = {
    any_authed: false,
    providers: {
      anthropic: { configured: false, status: 'missing' },
      openai: { configured: false, status: 'missing' },
      gemini: { configured: false, status: 'missing' },
      openai_compatible: { configured: false, status: 'missing' },
    },
  };

  const specific = {
    ui_ready: async () => view === 'auth'
      ? { state: 'needs_auth', auth }
      : view === 'landing'
        ? { state: 'needs_session' }
        : {
            state: 'ready',
            cwd: sessions[0].path,
            greeting: 'Ready to examine outcomes.parquet.',
            policy,
            datasets: ['outcomes.parquet'],
          },
    auth_status: async () => auth,
    list_busy_sessions: async () => ({ ok: true, cwds: [] }),
    list_sessions: async () => ({
      ok: true,
      current: sessions[0].path,
      sessions,
    }),
    get_chat_history: async () => ({
      ok: true,
      events: [
        {
          type: 'user_message',
          text: 'Profile the outcome measures and flag anything that could invalidate the analysis.',
        },
        {
          type: 'assistant_text',
          text: 'I checked schema, missingness, ranges, and identifier risk. The outcome fields are usable, but treatment_group has 3.8% missing values concentrated in two sites. I would test whether that pattern is related to assignment before estimating effects.\n\nNext I can run the missingness comparison and produce a reproducible table.',
        },
      ],
    }),
    list_models: async () => ({
      ok: true,
      current: 'anthropic:sonnet',
      current_effort: 'high',
      default_effort: 'high',
      efforts_by_provider: {
        anthropic: [{ id: 'medium' }, { id: 'high' }, { id: 'max' }],
      },
      models: [{
        id: 'anthropic:sonnet',
        label: 'Sonnet',
        provider: 'anthropic',
        available: true,
        context_window: 1000000,
      }],
    }),
    list_session_files: async () => ({
      ok: true,
      files: [{ name: 'outcomes.parquet', kind: 'data', size: 12800000 }],
    }),
    get_dataset_profile: async () => ({
      ok: true,
      rows: 84291,
      rows_exact: true,
      rows_profiled: 84291,
      sampled: false,
      columns: 6,
      missing_pct: 1.3,
      duplicate_rows: 0,
      size_display: '12.8 MB',
      health: { score: 88, issues: [] },
      variables: [
        { name: 'participant_id', dtype: 'string', semantic_type: 'identifier', missing_pct: 0, distinct: 84291, flag: 'likely identifier' },
        { name: 'treatment_group', dtype: 'category', semantic_type: 'categorical', missing_pct: 3.8, distinct: 3 },
        { name: 'outcome_score', dtype: 'float64', semantic_type: 'continuous', missing_pct: 1.1, distinct: 4182, min: 4.2, max: 98.7 },
        { name: 'followup_days', dtype: 'int64', semantic_type: 'duration', missing_pct: 0, distinct: 366, min: 0, max: 365 },
      ],
      likely_target_candidates: [{ name: 'outcome_score', reasons: ['outcome-like name'] }],
      survey_design_columns: [],
      available_sheets: [],
    }),
    list_checkpoints: async () => ({ ok: true, checkpoints: [] }),
    count_next_context: async () => ({
      ok: true,
      tokens: 3240,
      exact: true,
      ceiling: 1000000,
    }),
    doctor_report: async () => ({ blocked: false, runtimes: [] }),
  };

  const api = new Proxy(specific, {
    get(target, property) {
      if (property in target) return target[property];
      return async () => ({ ok: true });
    },
  });

  window.pywebview = { platform: 'cocoa', api };
})();
