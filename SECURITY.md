# Security policy

Sift is designed to keep raw research data and local computation outside a
model provider's direct reach. Its security contract is deliberately narrower
than a claim that no information can ever leave the computer.

Generated analysis runs without network access under native operating-system
confinement. Model-visible results must pass registered shape validation and
statistical disclosure controls. Credentials remain in the operating system's
protected credential service.

## Supported versions

| Version | Security fixes |
| --- | --- |
| 0.1.x beta | Yes |
| Earlier development builds | No |

This table will be updated when Sift reaches a stable release.

## Report a vulnerability

Please do not open a public issue for a suspected vulnerability.

Use GitHub's private vulnerability reporting from the repository's
**Security** tab, or email
[js.sandhu@mail.utoronto.ca](mailto:js.sandhu@mail.utoronto.ca).

Include:

- the Sift version and operating system;
- a concise description of the issue and its expected impact;
- the smallest safe reproduction you can provide;
- the affected area, such as tools, confinement, result sanitization,
  credentials, connectors, updates, or packaging;
- whether exploitation requires a deliberately malicious provider or script.

Do not send real research data, API keys, database credentials, or
institutional secrets. A synthetic reproduction is strongly preferred.

We aim to acknowledge a complete report within seven days. The response will
prioritize containment and a tested fix, followed by coordinated disclosure
when appropriate.

## Security boundary

The model receives only the researcher's messages, schema and summaries
allowed by the active permission level, sanitized statistical results,
approved aggregate figures, and redacted execution errors.

The model is not given a general shell, filesystem browser, database
connection, or web-search tool. Generated scripts run with network access
denied and a narrow filesystem allowlist. Sift refuses to run them when the
required confinement backend or its live check is unavailable.

Reports are particularly important when they show that:

- raw values, free text, or low-count cells can cross the result sanitizer;
- generated code can escape confinement or gain filesystem or network access;
- a provider can invoke a tool that is not part of Sift's declared interface;
- credentials can reach model context, generated code, logs, or plaintext
  settings;
- an installer, update, signature, or release manifest can be substituted or
  accepted without the required verification;
- an institutional policy can be ignored or made to fail open.

## Documented limitation

Generated code currently shares an interpreter with Sift's runtime helpers.
Per-run integrity framing rejects missing or stale envelopes and many direct
bypasses, but it does not prove the semantic meaning of a calculation. A
deliberately malicious provider could try to fabricate a permitted-looking
aggregate or misuse result metadata to encode information.

Sift is therefore not a defence against a model provider intentionally trying
to extract the dataset through generated code. Researchers with that threat
model must use a fully local model in the same trusted environment or keep
generated-code execution disabled.

Other expected behavior that is not, by itself, a vulnerability:

- a researcher can disclose information they type into the conversation;
- repeated approved queries may reveal more information over time; the
  disclosure ledger records this activity but is not a differential-privacy
  budget;
- package installation can contact external repositories only after explicit
  user approval;
- an unsigned Windows beta can trigger or be blocked by Windows security
  controls;
- upstream dependency issues that cannot affect Sift's boundary may have no
  practical impact on the application.

For the full design, see [Architecture](docs/architecture.md) and the
machine-readable [threat model](docs/security_threat_model.json).

## Deployment responsibility

Sift does not by itself establish compliance with privacy law, an ethics
approval, a data-use agreement, or an institutional security standard.
Organizations remain responsible for approving the selected data, model
provider, endpoint, disclosure policy, and deployment environment.
