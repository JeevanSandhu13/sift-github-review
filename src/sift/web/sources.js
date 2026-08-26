/* Unified data-source hub and first-run walkthrough.
 *
 * This file only orchestrates researcher-triggered bridge calls. It never
 * receives stored credentials back from Python, never enumerates a remote
 * account, and never exposes a connector to the model tool surface.
 */

(() => {
  'use strict';

  const byId = (id) => document.getElementById(id);
  const overlay = byId('sources-overlay');
  const configEl = byId('source-config');
  const gridEl = byId('source-grid');
  const localPanel = byId('source-panel-local');
  const catalogPanel = byId('source-panel-catalog');
  const statusEl = byId('sources-status');
  const walkthrough = byId('walkthrough-overlay');
  const walkthroughContent = byId('walkthrough-content');
  const walkthroughProgress = walkthrough?.querySelector('.walkthrough-progress');
  const WALKTHROUGH_KEY = 'sift.walkthrough.v1';

  let catalog = null;
  let activeTab = 'local';
  let walkthroughStep = 0;
  let pendingUseCase = null;

  const BRANDS = {
    sqlite: ['SQLite', '#eaf3f8', '#0f80cc', 'SQ'],
    duckdb: ['DuckDB', '#fff6bf', '#272727', 'DB'],
    postgresql: ['PostgreSQL', '#e6eef6', '#336791', 'PG'],
    mysql: ['MySQL', '#e6f3f5', '#00758f', 'my'],
    mariadb: ['MariaDB', '#ece7e1', '#003545', 'Ma'],
    mssql: ['SQL Server', '#f9e9e9', '#cc2927', 'MS'],
    oracle: ['Oracle', '#feeaea', '#c74634', 'O'],
    snowflake: ['Snowflake', '#e8f8fd', '#29b5e8', '❄'],
    bigquery: ['BigQuery', '#e9f0ff', '#4285f4', 'BQ'],
    redshift: ['Redshift', '#eee9fb', '#8c4fff', 'RS'],
    databricks: ['Databricks', '#feeceb', '#ff3621', '▰'],
    s3: ['Amazon S3', '#fff0dc', '#e15d15', 'S3'],
    gcs: ['Google Cloud Storage', '#eaf1ff', '#4285f4', 'GC'],
    azure_blob: ['Azure Blob Storage', '#e5f4fb', '#0078d4', 'Az'],
    https: ['Secure web link', '#e8f2ed', '#276f50', '↗'],
    sftp: ['SFTP', '#eceef1', '#45515b', 'SF'],
    zotero: ['Zotero', '#fdebec', '#cc2936', 'Z'],
    osf: ['OSF', '#edf1f2', '#263947', 'OSF'],
    dataverse: ['Dataverse', '#edf5fb', '#3b7ea1', 'DV'],
    zenodo: ['Zenodo', '#e9f2fb', '#0b5f9d', 'Ze'],
    figshare: ['Figshare', '#f6e9f3', '#9c2472', 'fi'],
    dryad: ['Dryad', '#eef7e7', '#6c9f35', 'Dr'],
    google_drive: ['Google Drive', '#edf5ee', '#0f9d58', '△'],
    onedrive: ['OneDrive', '#e9f3fb', '#0078d4', '☁'],
    sharepoint: ['SharePoint', '#e6f3f2', '#038387', 'SP'],
    box: ['Box', '#e9f2ff', '#0061d5', 'box'],
    dropbox: ['Dropbox', '#eaf0ff', '#0061ff', '◆'],
    redcap: ['REDCap', '#fae9e8', '#a9161c', 'RC'],
    qualtrics: ['Qualtrics', '#f0ebfb', '#6f2c91', 'Q'],
    kobotoolbox: ['KoboToolbox', '#eaf4fc', '#2095f2', 'Ko'],
    openclinica: ['OpenClinica', '#ecf5ed', '#258044', 'OC'],
  };

  const CONNECTOR_COPY = {
    sqlite: 'Open a local SQLite research database.',
    duckdb: 'Query a local DuckDB analytical workspace.',
    postgresql: 'Import a bounded query from PostgreSQL.',
    mysql: 'Import from a MySQL application or research database.',
    mariadb: 'Import from MariaDB with verified transport.',
    mssql: 'Connect to Microsoft SQL Server or Azure SQL.',
    oracle: 'Import from Oracle using its configured security.',
    snowflake: 'Check cost and materialize a warehouse query.',
    bigquery: 'Dry-run byte cost before importing a query.',
    redshift: 'Import from an Amazon Redshift warehouse.',
    databricks: 'Import through a SQL warehouse and catalog.',
    s3: 'Copy one exact object using your AWS identity.',
    gcs: 'Copy one exact object using your Google Cloud identity.',
    azure_blob: 'Import one blob with a scoped SAS credential.',
    https: 'Download a dataset from one verified HTTPS address.',
    sftp: 'Import one remote file over verified SFTP.',
    zotero: 'Use selected items from a local Zotero JSON export.',
    osf: 'Import one selected file and version from an OSF project.',
    dataverse: 'Import one file from an exact dataset version.',
    zenodo: 'Import one file from a published Zenodo record.',
    figshare: 'Import one file from a specific article version.',
    dryad: 'Import an exact Dryad file with provenance.',
    google_drive: 'Import one Google Drive file by its file ID.',
    onedrive: 'Import one file from a selected Microsoft drive.',
    sharepoint: 'Import one file from a SharePoint document library.',
    box: 'Import one selected Box file without account browsing.',
    dropbox: 'Import one Dropbox file at a known revision.',
    redcap: 'Export one project or report with a scoped token.',
    qualtrics: 'Create and import one survey response export.',
    kobotoolbox: 'Import a configured export from one Kobo project.',
    openclinica: 'Import one completed extract from a study.',
  };

  const CATALOG_TEXT = {
    database: ['Databases', 'Run a read-only query, check it, then copy only its result into Sift.'],
    cloud: ['Cloud storage', 'Provide one exact object address. Sift never lists a bucket or drive.'],
    research: ['Research services', 'Import a known record, file, survey, or extract without account-wide browsing.'],
  };

  const USE_CASES = {
    understand: {
      label: 'Understand a new dataset',
      prompt: 'Profile these datasets before analysis. Explain the schema, missingness, duplicates, unusual values, likely identifiers, and the most important data-quality risks. Then recommend a defensible next step.',
    },
    verify: {
      label: 'Verify an analysis',
      prompt: 'Help me verify an existing analysis. First identify the estimand, assumptions, variables, and expected result. Then reproduce the result, check implementation choices, and run the most important sensitivity analyses.',
    },
    compare: {
      label: 'Compare groups or outcomes',
      prompt: 'Help me compare the relevant groups or outcomes. Inspect the data first, recommend an appropriate design and statistical test, check its assumptions, and report effect sizes and uncertainty alongside significance.',
    },
    report: {
      label: 'Prepare a research output',
      prompt: 'Help me produce a defensible research output from these data. Build evidence-linked tables and figures, state assumptions and limitations, verify every reported number, and prepare reproducible export materials.',
    },
  };

  const DATABASES = ['sqlite', 'duckdb', 'postgresql', 'mysql', 'mariadb', 'mssql', 'oracle', 'snowflake', 'bigquery', 'redshift', 'databricks'];
  const CLOUD = ['s3', 'gcs', 'azure_blob', 'https', 'sftp'];
  const RESEARCH = ['zotero', 'osf', 'dataverse', 'zenodo', 'figshare', 'dryad', 'google_drive', 'onedrive', 'sharepoint', 'box', 'dropbox', 'redcap', 'qualtrics', 'kobotoolbox', 'openclinica'];
  const CONNECTOR_GROUPS = {
    database: [
      ['On this computer', ['sqlite', 'duckdb']],
      ['Database servers', ['postgresql', 'mysql', 'mariadb', 'mssql', 'oracle']],
      ['Cloud warehouses', ['snowflake', 'bigquery', 'redshift', 'databricks']],
    ],
    cloud: [['Object and file storage', CLOUD]],
    research: [
      ['Reference and open repositories', ['zotero', 'osf', 'dataverse', 'zenodo', 'figshare', 'dryad']],
      ['Institutional drives', ['google_drive', 'onedrive', 'sharepoint', 'box', 'dropbox']],
      ['Surveys and clinical data', ['redcap', 'qualtrics', 'kobotoolbox', 'openclinica']],
    ],
  };

  const DATABASE_HINTS = {
    sqlite: 'sqlite:////absolute/path/research.db',
    duckdb: 'duckdb:////absolute/path/research.duckdb',
    postgresql: 'postgresql://user:password@host:5432/database?sslmode=verify-full',
    mysql: 'mysql://user:password@host:3306/database',
    mariadb: 'mariadb://user:password@host:3306/database',
    mssql: 'mssql://user:password@host:1433/database?Encrypt=yes',
    oracle: 'oracle://user:password@host:1521/?service_name=service',
    snowflake: 'snowflake://user:password@account/database/schema?warehouse=WH',
    bigquery: 'bigquery://project/dataset',
    redshift: 'redshift://user:password@host:5439/database?sslmode=verify-full',
    databricks: 'databricks://token:…@server-hostname/http/path',
  };

  const RESEARCH_FIELDS = {
    osf: [['node_id', 'OSF project / node ID'], ['file_id', 'File ID'], ['filename', 'Local filename'], ['version_id', 'Version ID (optional)']],
    dataverse: [['base_url', 'Dataverse server URL'], ['persistent_id', 'Dataset persistent ID'], ['dataset_version', 'Dataset version'], ['file_id', 'File ID'], ['filename', 'Local filename']],
    zenodo: [['record_id', 'Record ID'], ['file_id', 'File ID'], ['filename', 'Local filename']],
    figshare: [['article_id', 'Article ID'], ['file_id', 'File ID'], ['filename', 'Local filename'], ['article_version', 'Article version (optional)']],
    dryad: [['artifact_id', 'Dataset DOI / artifact ID'], ['file_id', 'File ID'], ['filename', 'Local filename'], ['metadata_url', 'Metadata URL'], ['download_url', 'Exact download URL'], ['revision', 'Revision (optional)']],
    google_drive: [['file_id', 'Google Drive file ID'], ['filename', 'Local filename']],
    onedrive: [['drive_id', 'Drive ID'], ['file_id', 'File ID'], ['filename', 'Local filename']],
    sharepoint: [['drive_id', 'Drive ID'], ['file_id', 'File ID'], ['filename', 'Local filename']],
    box: [['file_id', 'Box file ID'], ['filename', 'Local filename']],
    dropbox: [['file_id', 'Dropbox path / file ID'], ['filename', 'Local filename'], ['revision', 'Revision']],
    redcap: [['api_url', 'REDCap API URL'], ['project_id', 'Project ID'], ['filename', 'Local filename'], ['report_id', 'Report ID (optional)']],
    qualtrics: [['datacenter', 'Qualtrics data center'], ['survey_id', 'Survey ID'], ['filename', 'Local filename']],
    kobotoolbox: [['server_url', 'Kobo server URL'], ['asset_uid', 'Project asset UID'], ['export_settings_uid', 'Export settings UID'], ['filename', 'Local filename']],
    openclinica: [['server_url', 'OpenClinica server URL'], ['study_id', 'Study ID'], ['job_execution_id', 'Extract job ID'], ['filename', 'Local filename']],
  };

  const RESEARCH_GUIDANCE = {
    zotero: 'Export the selected Zotero items as JSON, then paste the item keys shown in that export.',
    osf: 'Open the project and file in OSF. The node and file identifiers are available in the file URL or API details.',
    dataverse: 'Use the persistent dataset identifier, exact version, and file ID shown by your Dataverse installation.',
    zenodo: 'Open the published record and use its record number plus the exact file ID.',
    figshare: 'Open the article and selected file. Use the article, version, and file identifiers from its details.',
    dryad: 'Use the dataset DOI and the exact file download details for the revision you intend to analyze.',
    google_drive: 'Use the file ID from the selected file’s link. Sift does not request permission to list the rest of your Drive.',
    onedrive: 'Use the drive and file identifiers supplied by Microsoft Graph or your institution’s administrator.',
    sharepoint: 'Use the drive and file identifiers for the exact document in the selected SharePoint library.',
    box: 'Use the file ID from the selected file’s Box link or details panel.',
    dropbox: 'Use the exact Dropbox path or file ID and, when available, its revision.',
    redcap: 'Use the API address and identifiers provided by your REDCap project. Prefer a token limited to the required report.',
    qualtrics: 'Use the data-center and survey identifiers from the selected Qualtrics project.',
    kobotoolbox: 'Use the asset and export-setting identifiers from the selected KoboToolbox project.',
    openclinica: 'Use the study and completed extract-job identifiers provided by OpenClinica.',
  };

  const WALKTHROUGH = [
    {
      kicker: '1 · Choose a source',
      title: 'Bring the data to a local workspace',
      body: 'Start with files, a project folder, an explicitly selected database query, or one exact object from a connected service.',
      points: ['Sift never grants the model a way to browse your accounts.', 'Remote data is materialized locally with provenance and integrity details.', 'Use synthetic sample data to learn the workflow safely.'],
    },
    {
      kicker: '2 · Set the boundary',
      title: 'Privacy is a permission, not a slogan',
      body: 'Raw observations remain on this computer. Your active permission tier determines which checked, minimized results may enter model context.',
      points: ['Credentials live in the operating system credential vault.', 'Generated analysis code has no network access.', 'Images leave only when you explicitly attach them.'],
    },
    {
      kicker: '3 · Bring your own model',
      title: 'You control the model account',
      body: 'Connect an API account from Anthropic, OpenAI, Google, or a supported managed/local endpoint. Sift does not include or resell model access.',
      points: ['Provider billing and data terms remain with your account.', 'Sift sends prompts and only disclosure-approved tool results.', 'Local endpoints stay local only when their address is truly loopback.'],
    },
    {
      kicker: '4 · Analyze',
      title: 'The model reasons; Sift performs the work',
      body: 'Profile unfamiliar data, reproduce an existing result, compare groups, fit models, analyze surveys, or prepare a research output in plain language.',
      points: ['Start broad with schema, missingness, duplicates, and data-quality risks.', 'Then move to design, assumptions, effect sizes, uncertainty, and sensitivity checks.', 'Read-only database extraction is researcher-triggered; local statistical output is checked before model disclosure.'],
    },
    {
      kicker: '5 · Verify and export',
      title: 'Keep a defensible research record',
      body: 'Inspect evidence, assumptions, verification results, checkpoints, and the disclosure ledger before using a finding.',
      points: ['Exports include provenance instead of just polished prose.', 'Checkpoints make consequential choices resumable and reviewable.', 'The ledger records what was disclosed—not your raw dataset.'],
    },
  ];

  function api() {
    if (!window.pywebview?.api) throw new Error('Sift is still starting. Try again in a moment.');
    return window.pywebview.api;
  }

  function setStatus(message = '', kind = '') {
    if (!statusEl) return;
    statusEl.textContent = message;
    statusEl.className = `sources-status${kind ? ` ${kind}` : ''}`;
  }

  function logo(id, large = false) {
    const [label, bg, fg, mark] = BRANDS[id] || [id, '#eef1ed', '#536059', id.slice(0, 2).toUpperCase()];
    const node = document.createElement('span');
    node.className = 'connector-logo';
    node.style.setProperty('--brand-bg', bg);
    node.style.setProperty('--brand-fg', fg);
    node.textContent = mark;
    node.title = label;
    node.setAttribute('aria-hidden', 'true');
    if (large) node.style.flexBasis = '40px';
    return node;
  }

  function readinessFor(id) {
    return (catalog?.readiness || []).find((item) => item.integration_id === id) || {};
  }

  function manifestFor(kind, id) {
    const directKey = kind === 'database' ? 'databases' : kind === 'cloud' ? 'cloud_source_adapters' : 'research_service_adapters';
    return (catalog?.[directKey] || []).find((item) => item.id === id)
      || (catalog?.contracts || []).find((item) => item.id === id && item.kind === (kind === 'cloud' ? 'object_storage' : kind))
      || {};
  }

  function connectorCard(id, kind) {
    const [fallback] = BRANDS[id] || [id];
    const manifest = manifestFor(kind, id);
    const ready = readinessFor(id);
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'connector-card';
    button.dataset.connector = id;
    button.appendChild(logo(id));

    const copy = document.createElement('span');
    copy.className = 'connector-copy';
    const label = document.createElement('span');
    label.className = 'connector-label';
    label.textContent = manifest.label || fallback;
    copy.appendChild(label);
    const meta = document.createElement('span');
    meta.className = 'connector-meta';
    meta.textContent = CONNECTOR_COPY[id] || 'Import an explicit selection into the local workspace.';
    copy.appendChild(meta);
    const state = document.createElement('span');
    state.className = `connector-state ${ready.ready ? 'ready' : (ready.state || '')}`;
    const driverUnavailable = ready.diagnostics?.driver_installed === false;
    state.textContent = driverUnavailable ? 'Not installed' : ready.ready ? 'Available' : 'Set up';
    copy.appendChild(state);
    button.appendChild(copy);
    button.addEventListener('click', () => {
      try {
        renderConfig(kind, id);
      } catch (error) {
        configEl?.classList.add('hidden');
        gridEl?.classList.remove('hidden');
        catalogPanel?.querySelector('.source-catalog-toolbar')?.classList.remove('hidden');
        setStatus(`Could not open ${manifest.label || fallback}: ${error?.message || String(error)}`, 'error');
      }
    });
    return button;
  }

  async function loadCatalog() {
    if (catalog) return catalog;
    const response = await api().list_integrations();
    if (!response?.ok) throw new Error(response?.reason || 'Could not load connectors.');
    catalog = response;
    return catalog;
  }

  async function openSources(tab = 'local') {
    overlay?.classList.remove('hidden');
    setStatus('');
    const context = byId('source-use-case-context');
    if (context) {
      const selected = pendingUseCase ? USE_CASES[pendingUseCase] : null;
      context.textContent = selected
        ? `Starting workflow: ${selected.label}. Choose where the data comes from; Sift will prepare the first question for you.`
        : '';
      context.classList.toggle('hidden', !selected);
    }
    switchTab(tab);
    try {
      await loadCatalog();
      if (tab !== 'local') renderGrid(tab);
    } catch (error) {
      setStatus(error.message || String(error), 'error');
    }
    byId('sources-close')?.focus();
  }

  function closeSources() {
    overlay?.classList.add('hidden');
    configEl?.classList.add('hidden');
  }

  function switchTab(tab) {
    activeTab = tab;
    document.querySelectorAll('.source-tab').forEach((button) => {
      const selected = button.dataset.sourceTab === tab;
      button.classList.toggle('active', selected);
      button.setAttribute('aria-selected', String(selected));
    });
    localPanel?.classList.toggle('hidden', tab !== 'local');
    catalogPanel?.classList.toggle('hidden', tab === 'local');
    configEl?.classList.add('hidden');
    gridEl?.classList.remove('hidden');
    catalogPanel?.querySelector('.source-catalog-toolbar')?.classList.remove('hidden');
    byId('source-empty')?.classList.add('hidden');
    const search = byId('source-search-input');
    if (search) search.value = '';
    if (tab !== 'local' && catalog) renderGrid(tab);
  }

  function renderGrid(kind, query = '') {
    if (!gridEl) return;
    gridEl.replaceChildren();
    const ids = kind === 'database' ? DATABASES : kind === 'cloud' ? CLOUD : RESEARCH;
    const normalized = query.trim().toLowerCase();
    const visible = ids.filter((id) => {
      const label = BRANDS[id]?.[0] || id;
      return !normalized || `${label} ${CONNECTOR_COPY[id] || ''}`.toLowerCase().includes(normalized);
    });
    (CONNECTOR_GROUPS[kind] || [['Connectors', ids]]).forEach(([groupLabel, groupIds]) => {
      const groupVisible = groupIds.filter((id) => visible.includes(id));
      if (!groupVisible.length) return;
      const heading = document.createElement('div');
      heading.className = 'source-group-heading';
      heading.textContent = groupLabel;
      gridEl.appendChild(heading);
      groupVisible.forEach((id) => gridEl.appendChild(connectorCard(id, kind)));
    });
    const [title, description] = CATALOG_TEXT[kind] || ['Choose a connector', ''];
    if (byId('source-catalog-title')) byId('source-catalog-title').textContent = `${title} · ${visible.length}`;
    if (byId('source-catalog-description')) byId('source-catalog-description').textContent = description;
    byId('source-empty')?.classList.toggle('hidden', visible.length !== 0);
  }

  function field(name, label, options = {}) {
    const wrap = document.createElement('div');
    wrap.className = `source-field${options.wide ? ' wide' : ''}`;
    const labelEl = document.createElement('label');
    labelEl.htmlFor = `source-field-${name}`;
    labelEl.textContent = label;
    const input = options.textarea ? document.createElement('textarea') : document.createElement('input');
    input.id = `source-field-${name}`;
    input.name = name;
    if (!options.textarea) input.type = options.secret ? 'password' : 'text';
    input.autocomplete = 'off';
    input.spellcheck = false;
    if (options.placeholder) input.placeholder = options.placeholder;
    if (options.value) input.value = options.value;
    wrap.appendChild(labelEl);
    if (options.secret) {
      const secretWrap = document.createElement('div');
      secretWrap.className = 'source-secret-wrap';
      const reveal = document.createElement('button');
      reveal.type = 'button';
      reveal.className = 'source-secret-toggle';
      reveal.textContent = 'Show';
      reveal.setAttribute('aria-label', `Show ${label.toLowerCase()}`);
      reveal.addEventListener('click', () => {
        const showing = input.type === 'text';
        input.type = showing ? 'password' : 'text';
        reveal.textContent = showing ? 'Show' : 'Hide';
        reveal.setAttribute('aria-label', `${showing ? 'Show' : 'Hide'} ${label.toLowerCase()}`);
      });
      secretWrap.append(input, reveal);
      wrap.appendChild(secretWrap);
    } else {
      wrap.appendChild(input);
    }
    if (options.hint) {
      const hint = document.createElement('div');
      hint.className = 'source-field-hint';
      hint.textContent = options.hint;
      wrap.appendChild(hint);
    }
    return wrap;
  }

  function action(label, className, handler) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `source-action${className ? ` ${className}` : ''}`;
    button.textContent = label;
    button.addEventListener('click', async () => {
      button.disabled = true;
      try {
        await handler();
      } catch (error) {
        setStatus(error?.message || String(error), 'error');
      } finally {
        button.disabled = false;
      }
    });
    return button;
  }

  function formValues(form) {
    return Object.fromEntries([...form.querySelectorAll('[name]')].map((input) => [input.name, input.value.trim()]));
  }

  function configHeading(id, subtitle) {
    const heading = document.createElement('div');
    heading.className = 'source-config-heading';
    heading.appendChild(logo(id, true));
    const copy = document.createElement('div');
    const h = document.createElement('h3');
    h.textContent = BRANDS[id]?.[0] || id;
    const p = document.createElement('p');
    p.textContent = subtitle;
    copy.append(h, p);
    heading.appendChild(copy);
    return heading;
  }

  function renderConfig(kind, id) {
    if (!configEl) return;
    configEl.replaceChildren();
    configEl.classList.remove('hidden');
    gridEl?.classList.add('hidden');
    catalogPanel?.querySelector('.source-catalog-toolbar')?.classList.add('hidden');
    byId('source-empty')?.classList.add('hidden');
    const back = document.createElement('button');
    back.type = 'button';
    back.className = 'source-config-back';
    back.textContent = `← All ${CATALOG_TEXT[kind]?.[0]?.toLowerCase() || 'connectors'}`;
    back.addEventListener('click', () => {
      configEl.classList.add('hidden');
      gridEl?.classList.remove('hidden');
      catalogPanel?.querySelector('.source-catalog-toolbar')?.classList.remove('hidden');
      setStatus('');
    });
    configEl.appendChild(back);
    if (kind === 'database') renderDatabaseConfig(id);
    else if (kind === 'cloud') renderCloudConfig(id);
    else renderResearchConfig(id);
    configEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  async function ensureWorkspace() {
    if (typeof currentCwd !== 'undefined' && currentCwd) return;
    const result = await api().start_empty_session();
    if (!result?.ok) throw new Error(result?.reason || 'Could not create a local workspace.');
    if (typeof showChat === 'function') showChat(result);
  }

  async function refreshWorkspace() {
    const ready = await api().ui_ready();
    if (ready?.ok && typeof showChat === 'function') showChat(ready);
    if (typeof loadSessions === 'function') loadSessions();
  }

  function applyPendingUseCase() {
    const selected = pendingUseCase ? USE_CASES[pendingUseCase] : null;
    if (!selected || typeof input === 'undefined' || !input) return;
    input.value = selected.prompt;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.focus();
    if (typeof toast === 'function') toast(`${selected.label} is ready to review`, 'success');
    pendingUseCase = null;
  }

  function chooseUseCase(id) {
    if (!USE_CASES[id]) return;
    pendingUseCase = id;
    openSources('local');
  }

  function requireValues(values, names) {
    const missing = names.filter((name) => !values[name]);
    if (missing.length) throw new Error(`Complete the required fields: ${missing.join(', ')}.`);
  }

  function renderDatabaseConfig(id) {
    const ready = readinessFor(id);
    const manifest = manifestFor('database', id);
    const localDatabase = id === 'sqlite' || id === 'duckdb';
    configEl.appendChild(configHeading(id, id === 'sqlite' || id === 'duckdb'
      ? 'Choose a local database file, check a read-only query, then copy its result into this workspace.'
      : 'Use a connection string supplied by your database administrator. Sift stores it in the OS vault and runs only read-only SQL.'));
    const form = document.createElement('form');
    form.className = 'source-form-grid';
    form.addEventListener('submit', (event) => event.preventDefault());
    form.append(
      field('profile', 'Saved connection name', { value: `${id}-research`, hint: 'A name used only on this computer.' }),
      field('connection', 'Connection string', { secret: true, placeholder: DATABASE_HINTS[id] || '', hint: localDatabase ? 'Choose the database file below and Sift will fill this safely.' : 'Paste the value from your database administrator. It is stored in the protected credential vault.' }),
      field('sql', 'Query to import', { textarea: true, wide: true, placeholder: 'SELECT * FROM schema.table', hint: 'Only SELECT, WITH, or VALUES queries are accepted. Sift rejects statements that can change data.' }),
      field('dataset_name', 'Name for the local copy', { value: `${id}_extract.parquet` }),
      field('row_limit', 'Safety limit in rows', { value: '0', hint: 'Leave 0 to use Sift’s protected default.' }),
    );
    configEl.appendChild(form);
    const trust = document.createElement('div');
    trust.className = 'source-trust';
    const driver = ready.diagnostics?.driver_installed === false
      ? `This development build does not include the ${manifest.install_extra || id} driver. Full release builds verify it before packaging.`
      : 'This connector is available in the current build.';
    trust.textContent = `${driver} Sift tests and imports only when you ask; the model never receives database credentials or permission to browse the database.`;
    configEl.appendChild(trust);
    const actions = document.createElement('div');
    actions.className = 'source-actions';
    if (localDatabase) {
      actions.appendChild(action('Choose database file', '', async () => {
        setStatus('Opening the system file picker…');
        const result = await api().choose_database_file(id);
        if (!result?.ok) {
          if (result?.reason === 'cancelled') { setStatus('No file selected.'); return; }
          throw new Error(result?.reason || 'Could not select database file.');
        }
        form.querySelector('[name="connection"]').value = result.connection;
        setStatus(`${result.display} selected. Test the connection next.`, 'success');
      }));
    }
    actions.append(
      action('1 · Test connection', '', async () => {
        setStatus('Saving the connection securely…');
        const values = formValues(form);
        requireValues(values, ['profile', 'connection']);
        const saved = await api().save_database_profile(values.profile, values.connection);
        if (!saved?.ok) throw new Error(saved?.reason || 'Could not save connection.');
        await ensureWorkspace();
        const result = await api().test_database_profile_connection(values.profile);
        if (!result?.ok) throw new Error(result?.reason || 'Connection test failed.');
        form.querySelector('[name="connection"]').value = '';
        setStatus('Connection saved in the OS vault and tested successfully.', 'success');
      }),
      action('2 · Check query', '', async () => {
        setStatus('Checking the query without importing rows…');
        const values = formValues(form);
        requireValues(values, ['profile', 'sql']);
        await ensureWorkspace();
        const result = await api().preview_database_profile_query(values.profile, values.sql);
        if (!result?.ok) throw new Error(result?.reason || 'Query preview failed.');
        const estimate = result.estimated_rows != null ? ` Estimated rows: ${result.estimated_rows}.` : '';
        setStatus(`Query is read-only and passed preview.${estimate}`, 'success');
      }),
      action('3 · Import data', 'primary', async () => {
        setStatus('Running the read-only extract locally…');
        const values = formValues(form);
        requireValues(values, ['profile', 'sql', 'dataset_name']);
        await ensureWorkspace();
        const result = await api().run_database_profile_extract(values.profile, values.sql, values.dataset_name, Number(values.row_limit || 0));
        if (!result?.ok) throw new Error(result?.reason || 'Database import failed.');
        await refreshWorkspace();
        applyPendingUseCase();
        setStatus(`Imported ${result.dataset}${result.rows != null ? ` (${result.rows.toLocaleString()} rows)` : ''}.`, 'success');
        if (typeof toast === 'function') toast(`${result.dataset} added`, 'success');
      }),
    );
    configEl.appendChild(actions);
  }

  function cloudCredentialKind(id) {
    return id === 'azure_blob' ? 'azure_sas' : id === 'https' ? 'https_bearer' : id === 'sftp' ? 'sftp_key' : '';
  }

  function renderCloudConfig(id) {
    const uriHints = { s3: 's3://bucket/path/data.parquet', gcs: 'gs://bucket/path/data.csv', azure_blob: 'az://container/path/data.parquet', https: 'https://example.org/data.csv', sftp: 'sftp://host/path/data.csv' };
    configEl.appendChild(configHeading(id, 'Sift downloads one exact object after you name it. It does not list or browse your bucket, drive, or server.'));
    const form = document.createElement('form');
    form.className = 'source-form-grid';
    form.addEventListener('submit', (event) => event.preventDefault());
    form.append(
      field('uri', 'Object address', { wide: true, placeholder: uriHints[id], hint: 'Copy the exact address from the service. Sift will not scan the surrounding bucket, drive, or folder.' }),
      field('dataset_name', 'Name for the local copy (optional)', { placeholder: 'Use source filename' }),
    );
    const credentialKind = cloudCredentialKind(id);
    if (credentialKind) {
      form.append(
        field('profile', 'Saved access name', { value: `${id}-research`, hint: 'A label used only on this computer.' }),
        field('secret', id === 'sftp' ? 'Private key contents' : 'Token / SAS credential', { secret: true, hint: 'Saved directly to the OS credential vault.' }),
      );
    }
    configEl.appendChild(form);
    const trust = document.createElement('div');
    trust.className = 'source-trust';
    trust.textContent = credentialKind
      ? 'Saving a credential is optional only when the endpoint is public. The secret never enters the web page again after saving.'
      : 'Authentication uses the operating system or cloud SDK default identity chain. Sift does not ask you to paste long-lived cloud keys.';
    configEl.appendChild(trust);
    const actions = document.createElement('div');
    actions.className = 'source-actions';
    actions.appendChild(action('Import this object', 'primary', async () => {
      setStatus('Fetching the selected object into the local workspace…');
      const values = formValues(form);
      requireValues(values, ['uri']);
      if (credentialKind && values.secret) {
        requireValues(values, ['profile']);
        const saved = await api().save_remote_source_credential(values.profile, credentialKind, values.secret);
        if (!saved?.ok) throw new Error(saved?.reason || 'Could not save credential.');
        form.querySelector('[name="secret"]').value = '';
      }
      await ensureWorkspace();
      const result = await api().import_cloud_dataset(values.uri, values.dataset_name, values.profile || '');
      if (!result?.ok) throw new Error(result?.reason || 'Cloud import failed.');
      await refreshWorkspace();
      applyPendingUseCase();
      setStatus(`Imported ${result.dataset}; integrity fingerprint recorded.`, 'success');
      if (typeof toast === 'function') toast(`${result.dataset} added`, 'success');
    }));
    configEl.appendChild(actions);
  }

  function renderResearchConfig(id) {
    configEl.appendChild(configHeading(id, id === 'zotero'
      ? 'Import an explicit selection from a local Zotero JSON export. Sift does not synchronize or browse your library.'
      : 'Enter the exact record and file identifiers. Sift never requests permission to browse the whole account.'));
    const guidance = document.createElement('div');
    guidance.className = 'source-setup-help';
    guidance.textContent = RESEARCH_GUIDANCE[id] || 'Use the identifiers for the exact record or file you intend to analyze.';
    configEl.appendChild(guidance);
    const form = document.createElement('form');
    form.className = 'source-form-grid';
    form.addEventListener('submit', (event) => event.preventDefault());
    if (id === 'zotero') {
      form.append(
        field('exported_items', 'Zotero JSON export path', { wide: true, placeholder: '/path/to/My Library.json' }),
        field('item_keys', 'Selected Zotero item keys', { wide: true, placeholder: 'ABCD1234, EFGH5678' }),
        field('attachment_paths', 'Selected attachment paths (optional)', { wide: true, placeholder: '/path/article.pdf, /path/appendix.pdf' }),
      );
    } else {
      (RESEARCH_FIELDS[id] || []).forEach(([name, label]) => form.appendChild(field(name, label, { wide: ['base_url', 'metadata_url', 'download_url', 'api_url', 'server_url'].includes(name) })));
      const publicRepository = ['osf', 'dataverse', 'zenodo', 'figshare', 'dryad'].includes(id);
      form.append(
        field('profile', 'Saved access name', { value: publicRepository ? '' : `${id}-research`, hint: 'Leave blank for public records; required for private sources.' }),
        field('secret', 'Scoped access token (optional)', { secret: true, hint: 'Stored in the OS credential vault before import.' }),
      );
    }
    configEl.appendChild(form);
    const trust = document.createElement('div');
    trust.className = 'source-trust';
    trust.textContent = 'Only the object you identify is materialized. The service may retain access logs under your account policy; its credential is never available to generated code or the model.';
    configEl.appendChild(trust);
    const actions = document.createElement('div');
    actions.className = 'source-actions';
    actions.appendChild(action('Import selected data', 'primary', async () => {
      setStatus('Importing the explicit selection…');
      const values = formValues(form);
      await ensureWorkspace();
      let result;
      if (id === 'zotero') {
        requireValues(values, ['exported_items', 'item_keys']);
        const keys = values.item_keys.split(',').map((value) => value.trim()).filter(Boolean);
        const paths = values.attachment_paths.split(',').map((value) => value.trim()).filter(Boolean);
        result = await api().import_local_zotero_selection(values.exported_items, keys, paths);
      } else {
        const required = (RESEARCH_FIELDS[id] || []).filter(([, label]) => !label.includes('(optional)')).map(([name]) => name);
        requireValues(values, required);
        if (values.secret) {
          requireValues(values, ['profile']);
          const saved = await api().save_remote_source_credential(values.profile, 'research_token', values.secret);
          if (!saved?.ok) throw new Error(saved?.reason || 'Could not save credential.');
          form.querySelector('[name="secret"]').value = '';
        }
        const selection = {};
        [...form.querySelectorAll('[name]')].forEach((input) => {
          if (input.name !== 'secret' && input.value.trim()) selection[input.name === 'profile' ? 'credential_profile' : input.name] = input.value.trim();
        });
        result = await api().import_research_service_selection(id, selection);
      }
      if (!result?.ok) throw new Error(result?.reason || 'Research-service import failed.');
      await refreshWorkspace();
      applyPendingUseCase();
      setStatus(`Imported ${result.dataset}; source and revision details recorded.`, 'success');
      if (typeof toast === 'function') toast(`${result.dataset} added`, 'success');
    }));
    configEl.appendChild(actions);
  }

  async function runLocal(method) {
    setStatus('Opening the system picker…');
    try {
      const result = await api()[method]();
      if (!result?.ok) {
        if (result?.reason === 'cancelled') setStatus('No selection made.');
        else throw new Error(result?.reason || 'Could not open that source.');
        return;
      }
      closeSources();
      if (typeof showChat === 'function') showChat(result);
      if (typeof loadSessions === 'function') loadSessions();
      applyPendingUseCase();
    } catch (error) {
      setStatus(error.message || String(error), 'error');
    }
  }

  function renderWalkthrough() {
    const item = WALKTHROUGH[walkthroughStep];
    walkthroughContent.replaceChildren();
    const kicker = document.createElement('div');
    kicker.className = 'walkthrough-kicker';
    kicker.textContent = item.kicker;
    const title = document.createElement('h2');
    title.textContent = item.title;
    const body = document.createElement('p');
    body.textContent = item.body;
    const points = document.createElement('div');
    points.className = 'walkthrough-points';
    item.points.forEach((point, index) => {
      const row = document.createElement('div');
      row.className = 'walkthrough-point';
      const mark = document.createElement('div');
      mark.className = 'walkthrough-point-mark';
      mark.textContent = String(index + 1);
      const copy = document.createElement('div');
      copy.textContent = point;
      row.append(mark, copy);
      points.appendChild(row);
    });
    walkthroughContent.append(kicker, title, body, points);
    walkthroughProgress.replaceChildren(...WALKTHROUGH.map((_, index) => {
      const bar = document.createElement('span');
      bar.classList.toggle('active', index <= walkthroughStep);
      return bar;
    }));
    byId('walkthrough-back').disabled = walkthroughStep === 0;
    byId('walkthrough-next').textContent = walkthroughStep === WALKTHROUGH.length - 1 ? 'Done' : 'Next';
    byId('walkthrough-count').textContent = `${walkthroughStep + 1} of ${WALKTHROUGH.length}`;
  }

  function openWalkthrough() {
    walkthroughStep = 0;
    renderWalkthrough();
    walkthrough.classList.remove('hidden');
    byId('walkthrough-next')?.focus();
  }

  function closeWalkthrough(remember = true) {
    walkthrough?.classList.add('hidden');
    if (remember) {
      try { localStorage.setItem(WALKTHROUGH_KEY, 'seen'); } catch (_) { /* storage may be disabled */ }
    }
  }

  function bind() {
    byId('landing-sources-btn')?.addEventListener('click', () => openSources('database'));
    byId('sources-chip')?.addEventListener('click', () => openSources('local'));
    byId('landing-how-btn')?.addEventListener('click', openWalkthrough);
    byId('how-sift-works-btn')?.addEventListener('click', openWalkthrough);
    byId('sources-close')?.addEventListener('click', closeSources);
    byId('walkthrough-close')?.addEventListener('click', () => closeWalkthrough(true));
    byId('source-local-files')?.addEventListener('click', () => runLocal(typeof currentCwd !== 'undefined' && currentCwd ? 'add_files' : 'choose_files'));
    byId('source-local-folder')?.addEventListener('click', () => runLocal('choose_folder'));
    byId('source-sample')?.addEventListener('click', () => runLocal('start_sample_session'));
    document.querySelectorAll('[data-use-case]').forEach((button) => {
      button.addEventListener('click', () => chooseUseCase(button.dataset.useCase));
    });
    byId('source-search-input')?.addEventListener('input', (event) => {
      if (activeTab !== 'local') renderGrid(activeTab, event.target.value);
    });
    document.querySelectorAll('.source-tab').forEach((button) => button.addEventListener('click', async () => {
      switchTab(button.dataset.sourceTab);
      if (button.dataset.sourceTab !== 'local') {
        try { await loadCatalog(); renderGrid(button.dataset.sourceTab); }
        catch (error) { setStatus(error.message || String(error), 'error'); }
      }
    }));
    byId('walkthrough-back')?.addEventListener('click', () => {
      if (walkthroughStep > 0) { walkthroughStep -= 1; renderWalkthrough(); }
    });
    byId('walkthrough-next')?.addEventListener('click', () => {
      if (walkthroughStep < WALKTHROUGH.length - 1) { walkthroughStep += 1; renderWalkthrough(); }
      else closeWalkthrough(true);
    });
    overlay?.addEventListener('click', (event) => { if (event.target === overlay) closeSources(); });
    walkthrough?.addEventListener('click', (event) => { if (event.target === walkthrough) closeWalkthrough(true); });
    document.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape') return;
      if (!overlay?.classList.contains('hidden')) { closeSources(); event.stopImmediatePropagation(); }
      else if (!walkthrough?.classList.contains('hidden')) { closeWalkthrough(true); event.stopImmediatePropagation(); }
    }, true);

    /* A first-run guide appears once, after the provider screen has resolved.
       It never interrupts credential entry and remains available from both
       landing and workspace headers afterward. */
    if (typeof whenReady === 'function') whenReady(() => {
      window.setTimeout(() => {
        let seen = true;
        try { seen = localStorage.getItem(WALKTHROUGH_KEY) === 'seen'; } catch (_) { /* fail quiet */ }
        const authVisible = !byId('auth')?.classList.contains('hidden');
        if (!seen && !authVisible) openWalkthrough();
      }, 350);
    });
  }

  function guardAsyncErrors() {
    window.addEventListener('unhandledrejection', (event) => {
      if (!overlay?.classList.contains('hidden')) {
        setStatus(event.reason?.message || 'The connector operation could not be completed.', 'error');
        event.preventDefault();
      }
    });
  }

  bind();
  guardAsyncErrors();
})();
