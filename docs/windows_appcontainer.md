# Windows AppContainer backend

## Architecture

`src/sift/win_appcontainer.py` implements AppContainer confinement through a
LowBox token passed to `CreateProcess` with `STARTUPINFOEX`. A Job Object adds
resource limits and process-tree cleanup. `executor.run_script` dispatches to
this backend on Windows; `env_detect.py` and `doctor.py` report its readiness.

Windows-specific correctness is qualified by the native Windows workflow in
`.github/workflows/windows-11-native-qualification.yml`. A source review or a
non-Windows unit test is not a substitute for that release gate.

The backend does not trust API availability alone. At startup,
`probe_appcontainer_health()` launches a throwaway AppContainer and verifies
that it cannot read outside its grant set or open a network connection. If
either check fails or is inconclusive, `run_script` refuses to execute
generated code.

## What the existing backends guarantee

Each platform backend enforces four things around a script subprocess:

1. **No network** — the subprocess has no interface to bind or connect
   from at all (Linux: a fresh network namespace via `--unshare-all`;
   macOS: `(deny network*)` in the SBPL profile).
2. **Filesystem confinement** — read-only access to the system trees an
   interpreter needs (stdlib, shared libs, an R/Python install's own
   package directory), read-write access to exactly the researcher's cwd
   and the run's scratch directory, and a private-state carve-out so
   `.sift/` (session state, the release ledger, prior run logs) is
   invisible even though it lives inside the writable cwd.
3. **Process isolation** — a script cannot enumerate or signal any
   process outside its own sandbox (Linux: PID namespace; macOS: SBPL's
   process model).
4. **Resource limits** — CPU time, memory, process count, and per-file
   size caps (`resource_limits_preexec`'s `RLIMIT_*` calls on both
   platforms, layered underneath the namespace/profile confinement).

A Windows backend has to replicate all four with OS-enforced guarantees,
not merely "best-effort" ones — the whole reason Sift refuses outright
on an unsupported platform is that a script with unrestricted local file
access could smuggle data out through any surviving channel, and a
sandbox that only *looks* like it confines the process is worse than an
honest refusal.

## Windows primitives surveyed

| Mechanism | What it gives you | Fit for Sift |
|---|---|---|
| **AppContainer** (`CreateAppContainerProfile`, LowBox tokens, capability SIDs) | Real OS-enforced confinement: a process gets a per-container SID, and the kernel denies access to any filesystem path, registry key, or network capability that wasn't explicitly granted to that SID. This is the mechanism Windows Sandbox, MSIX-packaged apps, and modern browser sandboxes (Edge, Chrome on Windows) are built on. | **Best fit.** Filesystem confinement is ACL-grant-based rather than mount-based, but the end guarantee is the same shape as bwrap's allowlist: nothing is readable or writable unless explicitly granted. Network is denied by default — a container with no `internetClient`/`internetClientServer` capability SID literally cannot open a socket, which is a stronger and simpler guarantee than a firewall rule. |
| **Job Objects** (`CreateJobObject`, `JOBOBJECT_EXTENDED_LIMIT_INFORMATION`) | Per-job CPU time caps (`JOB_OBJECT_LIMIT_JOB_TIME`), memory caps (`JOB_OBJECT_LIMIT_JOB_MEMORY`), and active-process-count caps (`JOB_OBJECT_LIMIT_ACTIVE_PROCESS`) — plus "kill all on job close," which replaces bwrap's `--die-with-parent`. | **Directly maps to `resource_limits_preexec`.** This is the easy, well-trodden half of the work — `pywin32`'s `win32job` module covers the whole API, and it needs no elevated privilege. |
| **Restricted tokens** (`CreateRestrictedToken`) | Strips privileges and can deny specific SIDs. Predates AppContainer; this is roughly what pre-2012 sandboxes (old Chrome, old Adobe Reader) used. | **Superseded.** Coarser than AppContainer's capability model and more error-prone to get right (it's an SID *denylist*, not an allowlist — a missed SID is a silent hole, the same class of bug bwrap's `--remount-ro /` comment describes discovering empirically on Linux). Not recommended as the primary mechanism for new work. |
| **Windows Sandbox** (full disposable VM, `Windows Sandbox` optional feature + `.wsb` config) | Maximum isolation — a genuinely separate lightweight VM per run. | **Wrong shape for this product.** Requires Windows 10/11 **Pro or Enterprise** (unavailable on Home, which a meaningful fraction of individual researchers run), requires enabling an optional Windows feature via an admin prompt and often a reboot before first use, and each sandboxed run boots a VM rather than spawning a subprocess — orders of magnitude slower than the sub-second bwrap/sandbox-exec launch Sift's UX depends on for the "run this script" loop. File sharing into the VM is also configured through a static `.wsb` XML mapped-folder list, which doesn't compose cleanly with the executor's per-run scratch-directory model (a fresh `run_dir` every script execution). Worth revisiting later as an optional "maximum isolation" tier, not as the default backend. |
| **WSL2** (ship a Linux userspace, reuse the existing bwrap backend inside it) | Reuses tested code entirely. | **Rejected as the default path.** WSL2 is an opt-in Windows feature with its own enablement friction (similar to Windows Sandbox), and routing a Windows researcher's *native* Python/R/Stata installs and their data through a Linux VM's filesystem translation layer is a materially different (and worse) support story than confining the researcher's actual Windows processes directly. Could be offered as a fallback for researchers who already have WSL2 set up, but shouldn't be the primary design. |

**Recommendation:** AppContainer (via a LowBox token passed to
`CreateProcess`'s `STARTUPINFOEX` / `PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES`)
for filesystem + network confinement, layered with a Job Object for
resource limits and process-count caps. This is the same tier of
assurance as bwrap/sandbox-exec (OS-enforced, not advisory) and the same
general shape: two independent, composable primitives rather than one
monolithic profile.

## Concrete mapping (bwrap/sandbox-exec → Windows)

| Current guarantee | Windows equivalent |
|---|---|
| `--unshare-all` (no network namespace) | AppContainer with no `internetClient`/`internetClientServer` capability SID granted |
| Read-only system trees (`/usr`, `/lib`, …) | `icacls`/`SetNamedSecurityInfo` granting the AppContainer SID **read+execute** ACEs on the interpreter's install directory and its dependent DLL paths |
| Read-write cwd + run_dir | ACEs granting the AppContainer SID **read+write** on exactly those two paths, added before spawn and — importantly — removed after the run completes (ACL grants are persistent filesystem state, unlike a mount namespace that vanishes when the sandbox process exits; leaking a stale grant is a real cleanup-correctness risk this design has to get right) |
| `.sift/` private-state carve-out | Before the inheritable workspace ACE is applied, `.sift/` and `.sift/runs/` are marked DACL-protected and receive only non-inheriting traversal permission for the ephemeral AppContainer SID. The workspace ALLOW therefore cannot propagate into private state; only the exact current run receives a recursive read/write ACE. Cleanup restores both the original DACL and its original protected/unprotected inheritance state. |
| PID/IPC namespace isolation | Job Object's own process accounting scopes what the sandboxed process can see/signal; not identical to a Linux PID namespace but comparable in practice since AppContainer processes already can't open handles to processes outside their integrity/container boundary without an explicit capability |
| CPU/memory/process `RLIMIT_*` (`resource_limits_preexec`) | `JOBOBJECT_EXTENDED_LIMIT_INFORMATION` (`JOB_OBJECT_LIMIT_JOB_TIME`, `JOB_OBJECT_LIMIT_JOB_MEMORY`, `JOB_OBJECT_LIMIT_ACTIVE_PROCESS`) |
| `RLIMIT_FSIZE` single-file ceiling | Windows has no Job Object equivalent. `WritableFileSizeMonitor` takes a pre-launch baseline and polls both effective writable scopes (workspace excluding `.sift`, plus the explicitly exposed current run directory), terminating the complete Job Object on an oversized new/modified file and failing closed if any scope cannot be scanned responsively. This is intentionally described as parent-side polling, not exact `RLIMIT_FSIZE` equivalence: a write can overshoot until the next scan, and a temporary file created and removed wholly between scans cannot be recovered from a metadata walk. Poll delay adapts to scan cost (50–500 ms), overlapping roots are walked once, and a scan taking over one second is refused rather than allowed to consume the workspace continuously while providing a misleadingly slow guard. |
| Aggregate many-small-file exhaustion | The same parent monitor issues one constant-size free-space query per distinct filesystem backing the effective writable scopes at each poll. It refuses to start below `SIFT_SCRIPT_MIN_FREE_DISK_BYTES`, terminates the complete Job Object if free space crosses that reserve, and fails closed if capacity cannot be measured. This does not impose a fixed aggregate-output quota; it preserves a machine-level safety margin even when no individual file reaches `RLIMIT_FSIZE`. |
| `--die-with-parent` | Job Object's `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` |
| `find_sandbox_exec()` / `find_bwrap()` health probe | A `find_appcontainer_support()` probe: confirm `CreateAppContainerProfile` succeeds for a throwaway profile, a trivial process launches inside it with the intended capability set, and the deny/allow ACL model behaves as expected on **this specific machine** (this class of platform-health probe already exists for both current backends specifically because "the API exists" and "the API works on this machine" are different questions — see `_probe_sandbox_health`'s docstring) |

## Security rationale

Everything past the design-mapping table above is genuine Win32 systems
programming: `CreateAppContainerProfile`, building a `SECURITY_CAPABILITIES`
struct, threading it through `STARTUPINFOEX` and `UpdateProcThreadAttribute`,
and getting the ACL grant/revoke sequencing exactly right (a revoke that
fails to run after a crashed script would leak filesystem access to the
next process that happens to run under the same AppContainer SID). These
guarantees cannot be established by non-Windows tests because they depend on
Windows kernel behavior and concrete `ctypes` bindings.

Writing that code blind — struct layouts and API call sequences composed
from documentation and never actually executed — carries a specific,
serious risk: a subtly wrong `SECURITY_CAPABILITIES` struct, or a missed
`SetNamedSecurityInfoW` call, would not necessarily raise an obvious
error. It could produce a process that *looks* sandboxed (constructed
with the right API calls, launched successfully) while actually running
with full access — a silent, false-confidence failure mode that is
categorically worse than an explicit refusal.

Sift therefore fails closed rather than trusting API presence.
`probe_appcontainer_health()` is a mandatory gate, checked fresh
every session via `env_detect.appcontainer_probe_result()`, that actually
launches a throwaway AppContainer process on the researcher's real
machine and empirically confirms — not merely assumes — that a denied
file read and a denied network connect both actually get denied. Only
if that probe passes does `run_script` treat the backend as available;
otherwise it refuses exactly like "no backend installed" would. This
means a failed probe prevents activation rather than creating silent confidence.
The probe does not prove the complete implementation correct, so every release
must also pass the native Windows qualification workflow.

## Implementation map

1. **`src/sift/win_appcontainer.py`** — pure planning functions (`plan_acl_grants`,
   `plan_capability_sids`, `plan_job_limits`) plus the portable
   `WritableFileSizeMonitor` (unit-tested on every
   platform, mirroring how `_bwrap_argv` is tested without bwrap
   actually running) plus the Windows-only application layer
   (`create_appcontainer_profile`, `grant_acl`, `create_job_object`,
   `spawn_in_appcontainer`, the `AppContainerRun` context manager, and
   `probe_appcontainer_health`).
2. **`env_detect.py`** — `find_appcontainer_support()` (cheap "does the
   API surface exist" check, Windows 8+) and
   `appcontainer_probe_result()` (cached accessor for the live probe),
   plus a new `Environment.appcontainer_support` field and a
   `has_sandbox_backend()` win32 branch that checks BOTH gates, not just
   API presence.
3. **`executor.py`** — `run_script`'s preflight now has a real win32
   branch (two-gate: API-surface-present, then live-probe-passed) instead
   of an unconditional refusal; the confinement-wrapper build site
   branches on platform (Windows applies confinement via `CreateProcess`
   flags, not an argv-prefixed wrapper binary, unlike macOS/Linux); the
   `subprocess.Popen` call site now spawns through
   `AppContainerRun.__enter__()` on win32, returning an
   `AppContainerProcess` that implements the same
   `.pid`/`.communicate(timeout=)`/`.kill()`/`.returncode` surface so the
   rest of `run_script` (result parsing, stderr splitting, etc.) needed
   no Windows-specific duplicate; and the timeout handler's process-group
   kill (`os.killpg`/`os.getpgid`, which don't exist on Windows) now
   branches to `AppContainerProcess.kill()` (Job Object termination —
   reaches every process in the job, a strictly stronger guarantee than
   `killpg` for parallel/multiprocessing workers).
4. **`doctor.py`** — a new `_appcontainer_report()` with the same
   three-way split as the two gates above (not present / present-but-
   probe-failed / ok), wired into `_sandbox_report`'s win32 branch and
   the `_SANDBOX_ROW_NAMES` gating tuple.
5. **`ui.py`** — the CLI's sandbox-backend warning names "AppContainer"
   on win32 instead of falling through to the generic "a sandbox
   backend" string.
6. **Tests** — `tests/test_win_appcontainer.py` covers pure planning,
   writable-scope/file-size-monitor invariants, every
   pure planning function, ACL grant ordering/masking invariants, and
   the off-Windows refusal path for every OS-calling function).
   `tests/test_executor_sandbox.py` and `tests/test_doctor.py` cover absent,
   failed-probe, and verified-probe branches. Native integration tests provide
   the Windows API and kernel evidence.

## Native qualification checklist

The self-hosted Windows 11 workflow must pass before a Windows artifact is
released. It must establish all of the following on a clean x64 client host:

1. The complete test suite passes and every `ctypes` binding loads correctly.
2. The live probe confirms allowed writes, outside/private read denials,
   network denial, file-size enforcement, and the aggregate disk reserve.
3. Real subprocess integration tests confirm denied reads, denied writes, and
   denied network calls inside AppContainer.
4. Crash, cancellation, and timeout tests leave no stale ACL grants or orphaned
   AppContainer profiles.
5. The bundled Python runtime resolves and runs inside the installed layout;
   optional R and Stata installations work when present.

The workflow is defined in
`.github/workflows/windows-11-native-qualification.yml`. A release artifact
without current native evidence is not Windows-qualified.
