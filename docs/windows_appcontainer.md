# Windows AppContainer confinement

Sift runs generated analysis code inside a Windows AppContainer. The backend is
implemented in `src/sift/win_appcontainer.py` and is used by
`executor.run_script` on Windows.

AppContainer provides the filesystem and network boundary. A Job Object adds
resource limits, cancellation, and process-tree cleanup. Sift performs a live
denial probe before enabling generated-code execution; the presence of the
Windows APIs alone is not treated as evidence that confinement works.

## Security properties

The Windows backend is responsible for the same outcome as the macOS and Linux
backends:

- generated code has no network capability;
- only reviewed runtime files are readable;
- the workspace and current run receive the minimum required write access;
- private `.sift` state remains inaccessible;
- memory, CPU time, process count, file size, and disk reserve are bounded;
- cancellation and timeouts terminate the complete process tree;
- temporary access-control entries and AppContainer profiles are removed.

If setup, launch, monitoring, or cleanup is incomplete, the run fails closed.

## AppContainer process

Sift creates a LowBox token and passes its security capabilities to
`CreateProcess` through `STARTUPINFOEX`. The process receives no
`internetClient` or `internetClientServer` capability, so outbound and
listening sockets are unavailable.

Filesystem access is granted to the ephemeral AppContainer SID with explicit
access-control entries:

- read and execute for the selected interpreter and required libraries;
- read and write for the active workspace and run directory;
- traversal, but not content access, through protected private-state
  directories;
- no grant for unrelated user files, credentials, or system locations.

The workspace grant is temporary filesystem state. Sift records the original
access-control descriptors, applies the minimum grants before launch, and
restores the original descriptors during cleanup. Cleanup is required after a
successful run, failure, cancellation, timeout, or child-process crash.

## Private session state

The workspace may contain a `.sift` directory holding chat history,
provenance, prior results, the disclosure ledger, and run records. An
inheritable workspace grant must not flow into that directory.

Before the workspace grant is applied, Sift protects the `.sift` and
`.sift/runs` DACLs and adds only the traversal needed to reach the exact
current run. The current run receives its own recursive read/write grant.
Cleanup restores both the original DACL and its original inheritance state.

## Job Object and resource limits

Every generated process is assigned to a Job Object configured with:

- `JOB_OBJECT_LIMIT_JOB_TIME`;
- `JOB_OBJECT_LIMIT_JOB_MEMORY`;
- `JOB_OBJECT_LIMIT_ACTIVE_PROCESS`;
- `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`.

Closing or terminating the job reaches children created by parallel and
multiprocessing workloads. This is the Windows equivalent of terminating the
complete process group on macOS or Linux.

Windows has no direct Job Object equivalent to POSIX `RLIMIT_FSIZE`.
`WritableFileSizeMonitor` therefore records a pre-launch baseline and scans
the effective writable scopes while the job is active. It terminates the job
when a new or modified file exceeds the configured limit.

The same monitor checks free space once per filesystem. A run is refused when
available capacity is already below `SIFT_SCRIPT_MIN_FREE_DISK_BYTES`, and
the job is terminated if it crosses that reserve. A scan that cannot complete
within the reviewed responsiveness limit fails the run rather than silently
providing a weak guard.

## Launch lifecycle

Each run follows this order:

1. Confirm that the AppContainer API surface is present.
2. Require a successful live health probe for the current session.
3. Resolve and validate the runtime, workspace, and private-state paths.
4. Create an ephemeral AppContainer profile.
5. Save the original access-control descriptors.
6. Apply the reviewed read, write, and traversal grants.
7. Create the Job Object and resource monitors.
8. Launch the process with the LowBox token and sanitized environment.
9. Collect bounded output and framed result files.
10. Terminate the job on timeout, cancellation, or monitor failure.
11. Restore access-control descriptors and remove the profile.
12. Refuse the result if cleanup cannot be confirmed.

## Live health probe

`probe_appcontainer_health()` launches a throwaway container on the
researcher's machine and checks both a denied file read and a denied network
connection. `env_detect.appcontainer_probe_result()` exposes the result to
the executor, platform check, and diagnostics.

The probe has three possible outcomes:

- API unavailable;
- API available but the live denial check failed or was inconclusive;
- API available and the denial check passed.

Only the third outcome enables generated-code execution. The probe is a
machine-health gate, not a complete proof of implementation correctness.

## Implementation map

| Area | Location |
| --- | --- |
| AppContainer profiles, ACL planning, Job Objects, monitors, launch, and cleanup | `src/sift/win_appcontainer.py` |
| Host capability and live-probe detection | `src/sift/env_detect.py` |
| Language execution and timeout handling | `src/sift/executor.py` |
| Packaged platform report | `src/sift/platform_support.py` |
| AppContainer unit and process-contract tests | `tests/test_win_appcontainer.py`, `tests/test_win_appcontainer_communicate.py` |
| Executor and diagnostic integration tests | `tests/test_executor_sandbox.py`, `tests/test_platform_support.py` |
| Native Windows release lane | `.github/workflows/windows-11-native-qualification.yml` |

## Qualification

Portable tests can validate planning, structure layouts, ACL ordering,
monitoring logic, cleanup state machines, and off-Windows refusal. They cannot
establish Windows kernel behavior.

Before Sift describes a Windows artifact as native-qualified, the self-hosted
Windows 11 x64 workflow must establish that:

1. the full test suite and Windows `ctypes` bindings pass;
2. the live probe confirms allowed writes, outside and private read denials,
   and network denial;
3. file-size enforcement and the disk reserve terminate the full job;
4. real subprocess tests confirm read, write, and network restrictions;
5. crash, cancellation, and timeout paths leave no stale ACL or profile;
6. the installed application launches with its bundled runtime;
7. optional R and Stata execution works when those products are present.

Windows Server CI remains useful compatibility evidence but is not a substitute
for the Windows 11 client lane. A beta without current native evidence must be
labelled as such and must not be described as Windows-qualified.

## Known differences from POSIX confinement

- The file-size limit is enforced by parent-side polling rather than a kernel
  `RLIMIT_FSIZE`. A write may briefly exceed the limit before the next scan,
  and a temporary file created and deleted entirely between scans may not be
  observed.
- A Job Object is not a Linux PID namespace. It provides process accounting,
  limits, and full-tree termination, while AppContainer access checks prevent
  ordinary cross-boundary process access.
- AppContainer filesystem grants are persistent until restored, unlike a mount
  namespace that disappears with the process. Cleanup verification is
  therefore part of the security boundary.

These differences are documented rather than described as exact equivalence.
