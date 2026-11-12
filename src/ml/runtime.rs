//! ONNX Runtime initialization and management

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use ort::execution_providers::{ExecutionProvider, ExecutionProviderDispatch};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

/// Latch: only log the provider-probe result the first time a session is
/// built. Subsequent sessions reuse whichever provider won without spamming.
static EP_LOGGED: AtomicBool = AtomicBool::new(false);

/// Successfully registered session providers. This is explicitly NOT a
/// claim about GPU node placement; different models can use different providers.
static ACTIVE_PROVIDER: Mutex<&'static str> = Mutex::new("Not initialized");
#[cfg(target_os = "windows")]
static WINDOWS_RUNTIME_LIBS: OnceLock<Vec<libloading::Library>> = OnceLock::new();
static ORT_INITIALIZED: OnceLock<()> = OnceLock::new();
static ORT_INIT_LOCK: Mutex<()> = Mutex::new(());

/// User-facing label for the active execution provider, e.g. "DirectML",
/// "CPU", or a qualified GPU-registration label. Uninitialized until a session succeeds.
pub fn active_execution_provider() -> &'static str {
    *ACTIVE_PROVIDER.lock().unwrap_or_else(|e| e.into_inner())
}

/// Platform-specific ONNX Runtime library name
#[cfg(target_os = "windows")]
const ORT_LIB_NAME: &str = "onnxruntime.dll";
#[cfg(target_os = "macos")]
const ORT_LIB_NAME: &str = "libonnxruntime.dylib";
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
const ORT_LIB_NAME: &str = "libonnxruntime.so";

/// One accelerated session for a model, shared by every face worker in a run.
///
/// Building a GPU session per worker caused cold-start and VRAM blowups, and
/// DirectML does not allow concurrent `Run` calls on one session anyway, so
/// workers serialize only the inference call here and keep decoding,
/// pre-processing and post-processing in parallel.
///
/// The first inference error disables it for the rest of the run. Workers then
/// see an error from [`SharedGpuSession::run`] and move to their own CPU
/// session through their existing one-shot CPU fallback.
pub struct SharedGpuSession {
    session: Mutex<Session>,
    label: &'static str,
    disabled: AtomicBool,
}

impl std::fmt::Debug for SharedGpuSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedGpuSession")
            .field("provider", &self.label)
            .field("disabled", &self.disabled.load(Ordering::Relaxed))
            .finish()
    }
}

impl SharedGpuSession {
    /// `None` when no accelerated provider can host this model.
    pub fn try_new(path: &Path) -> Option<std::sync::Arc<Self>> {
        let (session, label) = OnnxRuntime::load_gpu_model(path)?;
        Some(std::sync::Arc::new(Self {
            session: Mutex::new(session),
            label,
            disabled: AtomicBool::new(false),
        }))
    }

    pub fn provider(&self) -> &'static str {
        self.label
    }

    pub fn is_disabled(&self) -> bool {
        self.disabled.load(Ordering::Relaxed)
    }

    /// Run `f` with exclusive access to the session. Any error (or a panic
    /// that poisoned the lock) disables the session for everyone.
    pub fn run<R>(&self, f: impl FnOnce(&mut Session) -> ort::Result<R>) -> ort::Result<R> {
        if self.is_disabled() {
            return Err(ort::Error::new(
                "GPU inference disabled after an earlier failure",
            ));
        }
        let mut guard = match self.session.lock() {
            Ok(g) => g,
            Err(_) => {
                self.disable("session lock poisoned by a panic");
                return Err(ort::Error::new("GPU session poisoned"));
            }
        };
        let result = f(&mut guard);
        if let Err(error) = &result {
            self.disable(&error.to_string());
        }
        result
    }

    fn disable(&self, reason: &str) {
        if !self.disabled.swap(true, Ordering::Relaxed) {
            tracing::warn!(provider = self.label, %reason,
                "GPU inference failed; workers will continue on CPU");
        }
    }
}

/// A model session that prefers the GPU and degrades to the CPU on its own.
///
/// This is the one place the GPU-then-CPU policy lives, so every ONNX model
/// (face detection, face embedding, visual search, ...) behaves the same way:
///
/// * built on a [`SharedGpuSession`] when one is available, else on its own CPU
///   session;
/// * if a GPU-backed run fails, it loads its own CPU session once and retries;
/// * a CPU-backed run that fails just returns the error.
pub struct ManagedSession {
    gpu: Option<std::sync::Arc<SharedGpuSession>>,
    cpu: Option<Session>,
    path: PathBuf,
    cpu_threads: usize,
    cpu_fallback_attempted: bool,
}

impl ManagedSession {
    /// A private CPU session.
    pub fn cpu(path: &Path, cpu_threads: usize) -> ort::Result<Self> {
        Ok(Self {
            gpu: None,
            cpu: Some(OnnxRuntime::load_cpu_model_with_threads(path, cpu_threads)?),
            path: path.to_path_buf(),
            cpu_threads,
            cpu_fallback_attempted: false,
        })
    }

    /// A handle on a session shared with other workers. No per-handle session is
    /// built; `cpu_threads` is only used if this handle has to fall back to CPU.
    pub fn shared(gpu: std::sync::Arc<SharedGpuSession>, path: &Path, cpu_threads: usize) -> Self {
        Self {
            gpu: Some(gpu),
            cpu: None,
            path: path.to_path_buf(),
            cpu_threads,
            cpu_fallback_attempted: false,
        }
    }

    /// For a model with a single owner: the GPU when `use_gpu` is set and a
    /// provider starts, else a private CPU session.
    pub fn auto(path: &Path, use_gpu: bool, cpu_threads: usize) -> ort::Result<Self> {
        if use_gpu {
            if let Some(gpu) = SharedGpuSession::try_new(path) {
                return Ok(Self::shared(gpu, path, cpu_threads));
            }
        }
        Self::cpu(path, cpu_threads)
    }

    /// Run `f` against the session, moving to the CPU once if a GPU run fails.
    /// `f` may be called twice, so it must not consume its inputs.
    pub fn run<R>(&mut self, mut f: impl FnMut(&mut Session) -> ort::Result<R>) -> ort::Result<R> {
        match self.run_once(&mut f) {
            Err(error) if self.gpu.is_some() && !self.cpu_fallback_attempted => {
                tracing::warn!(model = %self.path.display(), %error,
                    "Inference failed on the GPU; retrying once on CPU");
                self.cpu_fallback_attempted = true;
                // Drop our handle on the shared session; it is already disabled.
                self.gpu = None;
                self.cpu = None; // Release memory before building the fallback.
                self.cpu = Some(OnnxRuntime::load_cpu_model(&self.path, self.cpu_threads)?);
                self.run_once(&mut f)
            }
            other => other,
        }
    }

    fn run_once<R>(
        &mut self,
        f: &mut impl FnMut(&mut Session) -> ort::Result<R>,
    ) -> ort::Result<R> {
        if let Some(gpu) = self.gpu.clone() {
            return gpu.run(|session| f(session));
        }
        match self.cpu.as_mut() {
            Some(session) => f(session),
            None => Err(ort::Error::new("model session unavailable")),
        }
    }
}

/// ONNX Runtime environment wrapper
///
/// Manages the global ONNX Runtime environment and provides
/// helper methods for loading models.
///
/// Uses `load-dynamic` feature: the runtime library is loaded at runtime.
/// The library is resolved in this order:
/// 1. `ORT_DYLIB_PATH` environment variable (if set)
/// 2. `libs/onnxruntime/<LIB>` relative to the executable
/// 3. `libs/onnxruntime/<LIB>` relative to the current working directory
pub struct OnnxRuntime;

impl OnnxRuntime {
    fn usable_runtime_file(path: &Path) -> bool {
        std::fs::metadata(path)
            .is_ok_and(|metadata| metadata.is_file() && metadata.len() >= 1024 * 1024)
    }

    fn runtime_file_name_matches(name: &str) -> bool {
        #[cfg(target_os = "macos")]
        {
            name.starts_with("libonnxruntime") && name.ends_with(".dylib")
        }
        #[cfg(not(target_os = "macos"))]
        {
            name.starts_with(ORT_LIB_NAME)
        }
    }

    fn find_runtime_in_dir(dir: &Path) -> Option<PathBuf> {
        let direct = dir.join(ORT_LIB_NAME);
        if Self::usable_runtime_file(&direct) {
            return Some(direct);
        }

        // Also check for versioned variants (e.g. libonnxruntime.so.1.23.0)
        let entries = std::fs::read_dir(dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if Self::runtime_file_name_matches(name) && Self::usable_runtime_file(&path) {
                    return Some(path);
                }
            }
        }

        None
    }

    /// Resolve the path to the ONNX Runtime shared library.
    ///
    /// Checks ORT_DYLIB_PATH env var first, then looks in libs/onnxruntime/
    /// relative to the executable and current working directory.
    fn resolve_dylib_path() -> Option<PathBuf> {
        // 1. Check ORT_DYLIB_PATH environment variable
        if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
            let p = PathBuf::from(&path);
            if p.exists() {
                tracing::info!("Using ONNX Runtime from ORT_DYLIB_PATH: {}", p.display());
                return Some(p);
            }
            tracing::warn!("ORT_DYLIB_PATH set but file not found: {}", path);
        }

        // 1b. Check installed optional asset-pack roots. The asset pack
        // stores runtimes under platform subdirectories; bootstrap owns
        // that layout so health checks and dynamic loading agree.
        if let Some(candidate) = crate::bootstrap::onnx_runtime_path() {
            tracing::info!(
                "Using ONNX Runtime from optional asset-pack path: {}",
                candidate.display()
            );
            return Some(candidate);
        }

        let rel_dir = Path::new("libs").join("onnxruntime");

        // 2. Relative to the executable. Also tries two levels up
        // because `cargo tauri dev` runs the binary from
        // `target/debug/`, and the dev-tree `libs/onnxruntime/` lives
        // at the workspace root (target/debug/../.. = workspace).
        if let Ok(exe) = std::env::current_exe() {
            if let Some(exe_dir) = exe.parent() {
                for base in [exe_dir.to_path_buf(), exe_dir.join("..").join("..")] {
                    let candidate_dir = base.join(&rel_dir);
                    if let Some(candidate) = Self::find_runtime_in_dir(&candidate_dir) {
                        tracing::info!(
                            "Using ONNX Runtime from exe-relative path: {}",
                            candidate.display()
                        );
                        return Some(candidate);
                    }
                }
            }
        }

        // 3. Relative to the current working directory. Also walks one
        // level up because `cargo tauri dev` sets CWD to src-tauri/,
        // not the workspace root.
        if let Ok(cwd) = std::env::current_dir() {
            let mut bases: Vec<PathBuf> = vec![cwd.clone()];
            if let Some(parent) = cwd.parent() {
                bases.push(parent.to_path_buf());
            }
            for base in bases {
                let candidate_dir = base.join(&rel_dir);
                if let Some(candidate) = Self::find_runtime_in_dir(&candidate_dir) {
                    tracing::info!(
                        "Using ONNX Runtime from cwd-relative path: {}",
                        candidate.display()
                    );
                    return Some(candidate);
                }
            }
        }

        None
    }

    /// Initialize the ONNX Runtime global environment.
    ///
    /// Dynamically loads libonnxruntime.so at runtime. The library is searched
    /// in the order described on [`OnnxRuntime`].
    ///
    /// This should be called once at application startup, before creating any sessions.
    pub fn init() -> ort::Result<Self> {
        if ORT_INITIALIZED.get().is_some() {
            return Ok(Self);
        }

        let _guard = ORT_INIT_LOCK.lock().map_err(|_| {
            ort::Error::new("ONNX Runtime initialization lock is poisoned".to_string())
        })?;
        if ORT_INITIALIZED.get().is_some() {
            return Ok(Self);
        }

        if let Some(dylib_path) = Self::resolve_dylib_path() {
            #[cfg(target_os = "windows")]
            if WINDOWS_RUNTIME_LIBS.get().is_none() {
                use libloading::os::windows::{
                    Library, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
                };
                let root = dylib_path
                    .parent()
                    .ok_or_else(|| ort::Error::new("Runtime path has no parent"))?;
                let mut libraries = Vec::new();
                for dependency in [
                    "DirectML.dll",
                    "onnxruntime_providers_shared.dll",
                    ORT_LIB_NAME,
                ] {
                    let path = root.join(dependency);
                    if !path.is_file() {
                        continue;
                    } // Standard CPU installations remain supported.
                      // SAFETY: load only runtime assets beside the selected ORT DLL.
                      // Keep handles alive for all sessions; dependency search is restricted
                      // to that directory and Windows' safe default directories.
                    let library = unsafe {
                        Library::load_with_flags(
                            &path,
                            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
                        )
                    }
                    .map_err(|e| {
                        ort::Error::new(format!("Failed loading {}: {e}", path.display()))
                    })?;
                    libraries.push(library.into());
                }
                let _ = WINDOWS_RUNTIME_LIBS.set(libraries);
            }
            ort::init_from(&dylib_path)?.commit();
            let _ = ORT_INITIALIZED.set(());
            tracing::info!(
                "ONNX Runtime initialized (dynamic) from: {}",
                dylib_path.display()
            );
        } else {
            return Err(ort::Error::new(
                format!(
                    "ONNX Runtime library not found. Set ORT_DYLIB_PATH or place {} (1.23.x) in libs/onnxruntime/",
                    ORT_LIB_NAME
                ),
            ));
        }
        Ok(Self)
    }

    /// Accelerated execution providers for this platform, in priority order.
    /// Never includes CPU.
    fn gpu_providers() -> Vec<(&'static str, ExecutionProviderDispatch)> {
        vec![
            #[cfg(target_os = "windows")]
            (
                "DirectML registered (placement unverified)",
                ort::execution_providers::DirectMLExecutionProvider::default().build(),
            ),
            #[cfg(target_os = "linux")]
            (
                "CUDA registered (placement unverified)",
                ort::execution_providers::CUDAExecutionProvider::default().build(),
            ),
            #[cfg(target_os = "linux")]
            (
                "OpenVINO registered (placement unverified)",
                ort::execution_providers::OpenVINO::default()
                    .with_device_type("GPU")
                    .build(),
            ),
            #[cfg(target_os = "macos")]
            (
                "CoreML registered (placement unverified)",
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
            ),
        ]
    }

    /// Build one session with exactly one provider. Registration failure is
    /// an error (`error_on_failure`), never a silent CPU placement.
    fn build_session(
        path: &Path,
        label: &str,
        provider: ExecutionProviderDispatch,
        intra_threads: usize,
    ) -> ort::Result<Session> {
        let builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(intra_threads)?
            .with_parallel_execution(false)?
            .with_config_entry("session.intra_op.allow_spinning", "0")?
            .with_config_entry("session.inter_op.allow_spinning", "0")?;
        let builder = if label.starts_with("DirectML") {
            builder.with_memory_pattern(false)?
        } else {
            builder
        };
        builder
            .with_execution_providers([provider.error_on_failure()])?
            .commit_from_file(path)
    }

    /// Record which provider a session ended up on, for the status label.
    fn record_provider(label: &'static str) {
        let mut active = ACTIVE_PROVIDER.lock().unwrap_or_else(|e| e.into_inner());
        *active = if *active == "Not initialized" || *active == label {
            label
        } else {
            "Mixed model providers (see logs; placement unverified)"
        };
    }

    /// Try to build a session on an accelerated provider ONLY. Returns `None`
    /// when this platform has no accelerator or every one fails to initialize
    /// (missing driver, CPU-only ONNX Runtime build, unsupported ops, ...).
    /// There is deliberately no CPU tail here: the caller must know it did not
    /// get a GPU so it can pick its own CPU strategy.
    pub fn load_gpu_model(path: &Path) -> Option<(Session, &'static str)> {
        if !EP_LOGGED.swap(true, Ordering::Relaxed) {
            Self::probe_and_log_providers();
        }
        for (label, provider) in Self::gpu_providers() {
            match Self::build_session(path, label, provider, 2) {
                Ok(session) => {
                    Self::record_provider(label);
                    tracing::info!(model = %path.display(), provider = label, "GPU inference session initialized");
                    return Some((session, label));
                }
                Err(error) => {
                    tracing::warn!(model = %path.display(), provider = label, %error,
                        "GPU provider initialization failed; trying next");
                }
            }
        }
        None
    }

    /// A one-way CPU fallback after an inference-time provider/device failure.
    pub fn load_cpu_model(path: &Path, intra_threads: usize) -> ort::Result<Session> {
        let session = Self::load_cpu_model_with_threads(path, intra_threads)?;
        *ACTIVE_PROVIDER.lock().unwrap_or_else(|e| e.into_inner()) =
            "CPU fallback used (see per-model logs)";
        tracing::warn!(model = %path.display(), "Inference moved to CPU after device/provider failure");
        Ok(session)
    }

    /// Build a CPU-only session with an explicit thread budget. This is the CPU
    /// path for every model: used when the GPU is off or unavailable, and as the
    /// fallback after a GPU failure. Callers must never build one *GPU* session
    /// per worker (severe cold-start and VRAM regressions); they share a
    /// [`SharedGpuSession`].
    pub fn load_cpu_model_with_threads(path: &Path, intra_threads: usize) -> ort::Result<Session> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(intra_threads.max(1))?
            .with_parallel_execution(false)?
            .with_config_entry("session.intra_op.allow_spinning", "0")?
            .with_config_entry("session.inter_op.allow_spinning", "0")?
            .with_execution_providers([ort::execution_providers::CPUExecutionProvider::default()
                .build()
                .error_on_failure()])?
            .commit_from_file(path)?;
        let mut active = ACTIVE_PROVIDER.lock().unwrap_or_else(|e| e.into_inner());
        *active = if *active == "Not initialized" || *active == "CPU" {
            "CPU"
        } else {
            "Mixed model providers (see logs; placement unverified)"
        };
        tracing::info!(model = %path.display(), threads = intra_threads.max(1), "CPU inference session initialized");
        Ok(session)
    }

    /// Load a latency-sensitive CPU model without the expensive offline-style
    /// graph rewrites used by long-running batch inference. This is the right
    /// tradeoff for semantic text search: the session must become available
    /// quickly, while each interaction runs only one tiny text input.
    pub fn load_cpu_model_for_interactive(path: &Path) -> ort::Result<Session> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_intra_threads(1)?
            .with_parallel_execution(false)?
            .with_config_entry("session.intra_op.allow_spinning", "0")?
            .with_config_entry("session.inter_op.allow_spinning", "0")?
            .with_execution_providers([ort::execution_providers::CPUExecutionProvider::default()
                .build()
                .error_on_failure()])?
            .commit_from_file(path)?;
        tracing::info!(model = %path.display(), "Interactive CPU inference session initialized");
        Ok(session)
    }

    /// One-shot probe: log which execution providers are usable on this
    /// machine, and say plainly which of them this build will actually use.
    ///
    /// This only reports which providers are present. It does NOT claim a
    /// model actually runs on one: sessions log their own registered provider,
    /// and node placement is unverified (unsupported ops can stay on CPU).
    fn probe_and_log_providers() {
        let mut accelerated: Vec<&'static str> = Vec::new();

        #[cfg(target_os = "windows")]
        {
            let ep = ort::execution_providers::DirectMLExecutionProvider::default();
            if ep.is_available().unwrap_or(false) {
                accelerated.push("DirectML");
            }
        }

        #[cfg(target_os = "linux")]
        {
            let ep = ort::execution_providers::CUDAExecutionProvider::default();
            if ep.is_available().unwrap_or(false) {
                accelerated.push("CUDA");
            }

            let ep = ort::execution_providers::OpenVINO::default();
            if ep.is_available().unwrap_or(false) {
                accelerated.push("OpenVINO");
            }
        }

        #[cfg(target_os = "macos")]
        {
            let ep = ort::execution_providers::CoreMLExecutionProvider::default();
            if ep.is_available().unwrap_or(false) {
                accelerated.push("CoreML");
            }
        }

        tracing::info!(
            accelerated = %if accelerated.is_empty() { "none".to_string() } else { accelerated.join(", ") },
            "Execution providers present. Face detection/embedding and semantic search try \
             an accelerated provider first and fall back to CPU."
        );
    }
}
