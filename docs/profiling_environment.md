# CPU profiling environment

Captured on 2026-07-12 before Phase 7 profiling. The machine-readable event probe is generated under the ignored `profiles/` tree by `scripts/probe_perf_events.py`.

## Host and permissions

| Property | Observed value |
| --- | --- |
| CPU | AMD Ryzen 7 5800H with Radeon Graphics |
| Logical CPUs | 16 (`nproc`: 16) |
| Physical topology | 1 socket, 8 cores/socket, 2 threads/core |
| WSL2 kernel | Linux 6.18.33.2-microsoft-standard-WSL2, x86_64, PREEMPT_DYNAMIC |
| perf | 7.0.6 (`/usr/bin/perf`) |
| `perf_event_paranoid` | 2 |
| `kptr_restrict` | 1 |
| User | unprivileged local user |
| Affinity | CPUs 0-15 |
| Memory | 7.5 GiB total, 6.7 GiB available; 2.0 GiB swap |
| Governor | unavailable: WSL2 exposed no `cpu0/cpufreq` files |

CPU flags: `fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx mmxext fxsr fxsr_opt sse sse2 ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology tsc_reliable nonstop_tsc cpuid extd_apicid tsc_known_freq pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext perfctr_core ssbd ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves clzero xsaveerptr arat npt nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vmsave_vmload umip vaes vpclmulqdq rdpid fsrm`.

## Build and symbols

The verified `third_party/llama.cpp/build-release/bin/llama-bench` is a non-stripped PIE with `.symtab`, built as `Release` using `-O3 -DNDEBUG`, `GGML_CPU=ON`, `GGML_NATIVE=ON`, and `GGML_CPU_REPACK=ON`. No `.debug_*` sections were present. Neither the cache nor ELF evidence shows frame pointers enabled; Release defaults therefore provide no frame-pointer guarantee. DWARF call graphs are selected for recording. A separate build is unnecessary unless smoke-profile symbol resolution proves insufficient; the verified Release build is not replaced.

llama.cpp source commit: `e3546c7948e3af463d0b401e6421d5a4c2faf565`.

## perf event capability

Each event was probed independently with `perf stat -x, -e EVENT -- true`.

| Classification | Events |
| --- | --- |
| Supported | `task-clock`, `cpu-clock`, `cycles`, `instructions`, `branches`, `branch-misses`, `cache-references`, `cache-misses`, `context-switches`, `cpu-migrations`, `page-faults`, `minor-faults`, `major-faults`, `stalled-cycles-frontend`, `L1-dcache-loads`, `L1-dcache-load-misses` |
| Unsupported | `stalled-cycles-backend`, `LLC-loads`, `LLC-load-misses` |
| Permission denied | none during the probe |
| Unavailable in WSL2 | none separately classified; unsupported PMU aliases above may reflect WSL2 PMU exposure |
| Noisy or unreliable | none in the short capability probe; support does not guarantee stable multiplexed counts |

Exact unsupported diagnostics were “No supported events found” and “event is not supported.” The runner re-probes events individually and excludes unsupported events from a combined measurement while preserving every diagnostic. System-wide settings were not changed.

## Reproduction commands

```text
uname -a
lscpu
grep -m1 '^flags' /proc/cpuinfo
nproc
perf --version
cat /proc/sys/kernel/perf_event_paranoid
cat /proc/sys/kernel/kptr_restrict
free -h
taskset -pc $$
.venv/bin/python scripts/probe_perf_events.py --json profiles/perf-event-probe.json --markdown profiles/perf-event-probe.md
```
