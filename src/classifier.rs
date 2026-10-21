// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use log::{info, debug, warn, error};
use indicatif::{ProgressBar, ProgressStyle};

// Import from the commit-classifier library
use commit_classifier::{
    HuggingFaceEmbeddings,
    ChromaStore,
    Metadata,
    Llm,
    CommitFeatures,
    ProviderResult,
    PredictionResult,
};

// ModelState from commit-classifier
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ModelState {
    pub persist_directory: Option<PathBuf>,
}

// Full backport classifier with embeddings functionality
pub struct BackportClassifier {
    pub embeddings: HuggingFaceEmbeddings,
    pub vectorstore: Option<ChromaStore>,
    pub persist_directory: Option<PathBuf>,
    pub repo_path: Option<PathBuf>,
    pub llm_providers: Vec<String>,
    pub llm_configs: HashMap<String, serde_json::Value>,
    pub llms: Option<HashMap<String, Box<dyn Llm>>>,
}

impl BackportClassifier {
    pub fn new(
        llm_providers: Vec<String>,
        llm_configs: Option<HashMap<String, serde_json::Value>>,
        persist_directory: Option<&Path>,
        repo_path: Option<&Path>
    ) -> Self {
        let default_configs = Self::create_default_llm_configs();

        let final_configs = if let Some(user_configs) = llm_configs {
            let mut merged = default_configs.clone();
            for (k, v) in user_configs {
                merged.insert(k, v);
            }
            merged
        } else {
            default_configs
        };

        BackportClassifier {
            embeddings: HuggingFaceEmbeddings::new(),
            vectorstore: None,
            persist_directory: persist_directory.map(std::path::Path::to_path_buf),
            repo_path: repo_path.map(std::path::Path::to_path_buf),
            llm_providers,
            llm_configs: final_configs,
            llms: None,
        }
    }

    // Helper function to create default LLM configurations
    fn create_default_llm_configs() -> HashMap<String, serde_json::Value> {
        let mut configs = HashMap::new();

        // Claude default config
        configs.insert("claude".to_string(), serde_json::json!({
            "model": "claude-3-7-sonnet-20250219",
            "temperature": 0,
            "max_tokens": 16000,
            "debug_logging": false
        }));

        // NVIDIA default config
        configs.insert("nvidia".to_string(), serde_json::json!({
            "model_type": "deepseek",
            "temperature": 0.6,
            "max_tokens": 16384
        }));

        // OpenAI default config
        configs.insert("openai".to_string(), serde_json::json!({
            "model_type": "gpt4o",
            "temperature": 0.2,
            "max_tokens": 16384,
            "debug_logging": false
        }));

        // Ollama default config
        configs.insert("ollama".to_string(), serde_json::json!({
            "model_type": "llama3",
            "temperature": 0.1,
            "max_tokens": 16384,
            "api_host": "http://localhost:11434",
            "debug_logging": false
        }));

        configs
    }

    /// Creates a new `BackportClassifier` instance optimized for processing prompts
    /// This is a lighter version that doesn't require a vectorstore
    pub fn new_for_prompt(
        llm_providers: Vec<String>,
        llm_configs: Option<HashMap<String, serde_json::Value>>,
        repo_path: Option<&Path>
    ) -> Self {
        // Initialize with the standard constructor but no persist_directory
        Self::new(llm_providers, llm_configs, None, repo_path)
    }

    #[allow(clippy::too_many_lines)]
    pub fn train(&mut self, commit_dataset: &[CommitFeatures]) -> Result<(), String> {
        const BATCH_SIZE: usize = 1000;

        info!("Training classifier with {} commits", commit_dataset.len());

        // Process dataset into texts and metadata for vectorstore
        let mut texts = Vec::new();
        let mut metadatas = Vec::new();

        let max_chunk_length = 100_000; // Characters

        let pb = ProgressBar::new(commit_dataset.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap());

        for commit in commit_dataset {
            let message_lines: Vec<&str> = commit.message.trim().lines().collect();
            let subject = (*message_lines.first().unwrap_or(&"")).to_string();
            let message_body = if message_lines.len() > 1 {
                message_lines[1..].join("\n")
            } else {
                String::new()
            };

            // Build rich context for each commit
            let mut commit_context = Vec::new();

            commit_context.push(format!("Subject: {subject}"));
            commit_context.push(format!("Commit Message:\n{message_body}"));

            // Add code changes
            let diff_text = if commit.diff.len() > max_chunk_length {
                let truncated = commit.diff.chars().take(max_chunk_length).collect::<String>();
                warn!("Truncating large diff for commit {} (size: {})", commit.sha, commit.diff.len());
                format!("{truncated}\n[... truncated due to size ...]")
            } else {
                commit.diff.clone()
            };

            commit_context.push(format!("Changes:\n{diff_text}"));

            // Add files changed
            let files_text = if commit.files_changed.is_empty() {
                "Files: [none]".to_string()
            } else {
                let displayed_files = if commit.files_changed.len() > 5 {
                    let joined = commit.files_changed[0..5].join(", ");
                    format!("{} and {} more", joined, commit.files_changed.len() - 5)
                } else {
                    commit.files_changed.join(", ")
                };
                format!("Files: {displayed_files}")
            };

            commit_context.push(files_text);

            // Mark backport status prominently
            let backport_status = if commit.was_selected == Some(true) { "YES" } else { "NO" };
            let text = format!("[Backport Status: {}]\n\n{}", backport_status, commit_context.join("\n\n"));

            // Create metadata
            let mut metadata = HashMap::new();
            metadata.insert("was_selected".to_string(), backport_status.to_string());
            metadata.insert("sha".to_string(), commit.sha.clone());
            metadata.insert("date".to_string(), commit.date.to_rfc3339());

            // Add file paths joined with commas (limited to 10)
            if !commit.files_changed.is_empty() {
                let file_list = commit.files_changed.iter()
                    .take(10)
                    .cloned()
                    .collect::<Vec<String>>()
                    .join(",");
                metadata.insert("files".to_string(), file_list);
            }

            // Add text and metadata to vectors
            texts.push(text);

            // Convert HashMap to Metadata
            let mut metadata_obj = Metadata::default();
            for (k, v) in metadata {
                metadata_obj.extra.insert(k, serde_json::Value::String(v));
            }
            metadatas.push(metadata_obj);

            pb.inc(1);
        }

        pb.finish_with_message("Processed commit dataset");

        // Initialize embeddings
        if let Err(e) = self.embeddings.initialize() {
            return Err(format!("Failed to initialize embeddings: {e}"));
        }

        // Process in reasonable batches
        let mut start_idx = 0;

        if self.vectorstore.is_none() {
            // Create initial vectorstore with first batch
            let batch_size = std::cmp::min(BATCH_SIZE, texts.len());
            let batch_texts = texts[..batch_size].to_vec();
            let batch_metadatas = metadatas[..batch_size].to_vec();

            info!("Creating vectorstore with initial {batch_size} items");

            self.vectorstore = match ChromaStore::from_texts(
                &batch_texts,
                &mut self.embeddings,
                Some(&batch_metadatas.iter().map(|m| {
                    let mut map = HashMap::new();
                    for (k, v) in &m.extra {
                        if let serde_json::Value::String(s) = v {
                            map.insert(k.clone(), s.clone());
                        }
                    }
                    map
                }).collect::<Vec<_>>()),
                self.persist_directory.as_deref()
            ) {
                Ok(vs) => Some(vs),
                Err(e) => return Err(format!("Failed to create vectorstore: {e}")),
            };

            start_idx = batch_size;
        }

        // Add remaining batches
        if start_idx < texts.len() {
            let pb = ProgressBar::new((texts.len() - start_idx) as u64);
            pb.set_style(ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap());

            for batch_start in (start_idx..texts.len()).step_by(BATCH_SIZE) {
                let batch_end = std::cmp::min(batch_start + BATCH_SIZE, texts.len());
                let batch_size = batch_end - batch_start;

                let batch_texts = texts[batch_start..batch_end].to_vec();
                let batch_metadatas = metadatas[batch_start..batch_end].to_vec();

                if let Some(vs) = &mut self.vectorstore
                    && let Err(e) = vs.add_texts(batch_texts, Some(batch_metadatas))
                {
                    warn!("Failed to add batch to vectorstore: {e}");
                }

                pb.inc(batch_size as u64);
            }

            pb.finish_with_message("Added all batches to vectorstore");
        }

        // Count backported vs non-backported
        let backported_count = commit_dataset.iter()
            .filter(|c| c.was_selected == Some(true))
            .count();

        info!("Training complete - indexed {} commits ({} backported, {} non-backported)",
              commit_dataset.len(), backported_count, commit_dataset.len() - backported_count);

        Ok(())
    }

    pub fn save_model(&self, model_path: &Path) -> Result<(), String> {
        // Save the vectorstore first if it exists
        if let Some(vectorstore) = &self.vectorstore
            && let Some(persist_dir) = &self.persist_directory
        {
            debug!("Saving vectorstore to {}", persist_dir.display());
            if let Err(e) = vectorstore.persist(persist_dir) {
                return Err(format!("Failed to save vectorstore: {e}"));
            }
        }

        // Create a simplified model state for serialization
        let model_state = ModelState {
            persist_directory: self.persist_directory.clone(),
        };

        // Serialize and save
        match serde_json::to_string_pretty(&model_state) {
            Ok(json) => match std::fs::write(model_path, json) {
                Ok(()) => Ok(()),
                Err(e) => Err(format!("Failed to write model file: {e}")),
            },
            Err(e) => Err(format!("Failed to serialize model state: {e}")),
        }
    }

    pub fn load_model(model_path: &Path) -> Result<Self, String> {
        // Load the model state from file
        let model_state: ModelState = match std::fs::read_to_string(model_path) {
            Ok(contents) => match serde_json::from_str(&contents) {
                Ok(state) => state,
                Err(e) => return Err(format!("Failed to parse model state: {e}")),
            },
            Err(e) => return Err(format!("Failed to read model file: {e}")),
        };

        // Set persist directory from model state
        let persist_dir = match &model_state.persist_directory {
            Some(dir) => {
                if !dir.exists() {
                    return Err(format!("Persist directory does not exist: {}", dir.display()));
                }
                Some(dir.as_path())
            },
            None => None,
        };

        // Create classifier
        let mut classifier = BackportClassifier::new(
            vec!["claude".to_string()],
            None,
            persist_dir,
            None,
        );

        // Try to load the vectorstore if persist directory is set
        if let Some(persist_dir) = persist_dir {
            // No need to initialize embeddings here - the vectorstore load will initialize it if needed

            // Load the vectorstore
            info!("Loading vectorstore from {}", persist_dir.display());
            match ChromaStore::load(persist_dir, classifier.embeddings.clone()) {
                Ok(vs) => classifier.vectorstore = Some(vs),
                Err(e) => return Err(format!("Failed to load vectorstore: {e}")),
            }
        }

        Ok(classifier)
    }

    pub fn format_commit_info(commit_info: &CommitFeatures) -> String {
        let mut commit_parts = Vec::new();

        // Extract subject (first line of commit message)
        let message_lines: Vec<&str> = commit_info.message.lines().collect();
        let subject = (*message_lines.first().unwrap_or(&"No subject")).to_string();

        commit_parts.push(format!("Subject: {subject}"));
        commit_parts.push(format!("Commit Message:\n{}", commit_info.message));

        // Add code changes
        if !commit_info.diff.is_empty() {
            commit_parts.push(format!("Changes:\n{}", commit_info.diff));
        }

        // Add files changed
        if !commit_info.files_changed.is_empty() {
            let mut files_text = commit_info.files_changed.iter()
                .take(10)
                .map(std::string::ToString::to_string)
                .collect::<Vec<String>>()
                .join(", ");

            if commit_info.files_changed.len() > 10 {
                use std::fmt::Write;
                write!(files_text, " and {} more", commit_info.files_changed.len() - 10).unwrap();
            }

            commit_parts.push(format!("Files Changed: {files_text}"));
        }

        commit_parts.join("\n\n")
    }

    pub fn construct_prompt(commit_text: &str, similar_commits: &[(String, bool)]) -> String {
        // Template for the prompt
        let prompt_template = r#"You are a Linux kernel expert analyzing commits to determine if they should be backported to stable kernel trees.

CRITICAL: NEVER use "git log --all" or any git command with the --all flag. It searches hundreds of branches and takes hours. Use "git log" without --all, or specify a branch like "git log master".

CRITICAL: EVERY COMMIT YOU ANALYZE IS A CANDIDATE - NOT YET SELECTED:
You are evaluating whether this commit SHOULD be backported. It has NOT been selected yet.
- IGNORE "[Upstream commit ...]" markers - these are added during candidate preparation
- IGNORE "Signed-off-by: Sasha Levin" - this is added during candidate preparation
- IGNORE any evidence that suggests the commit "is already selected" or "has been prepared for stable"
- These markers do NOT mean the commit was selected - they are part of the evaluation pipeline
- Your job is to make the YES/NO decision based purely on TECHNICAL MERIT of the fix itself
- Do NOT say "this is already selected" or "this has been prepared for backport" - that's circular reasoning

CRITICAL: DECIDE FOR THE LOCALLY CHECKED-OUT TREE ONLY:

You are running inside a git checkout of one specific kernel tree. Your verdict is
about THAT tree and no other one.
- Establish which tree you are in before you decide: "git describe HEAD",
  "make kernelversion", or the Makefile VERSION/PATCHLEVEL/SUBLEVEL lines
- Then answer exactly one question: should this commit be backported to THIS tree?
- Check the code, the prerequisites and the context in THIS checkout - not in some
  other stable series you remember

Do NOT split your answer across kernel versions. "YES for 6.12.y and 6.18.y, not
applicable to 6.6.y" is not a usable answer, because only one tree is being decided.
If the code this commit touches is missing from the local tree, if the fix needs
commits that are not in it, or if the bug was introduced after this tree branched,
then the answer is NO.

UNDERSTANDING STABLE KERNEL TREES:

The Linux kernel stable trees (also known as -stable or LTS trees) are long-term supported versions that receive critical bug fixes
and security updates after their initial release. Examples include 6.6.y, 6.1.y, 5.15.y, etc. These stable kernels are used by:
- Enterprise distributions (RHEL, SLES, Ubuntu LTS)
- Embedded systems and IoT devices
- Android devices
- Systems that cannot afford the risk of frequent major kernel updates

The goal of backporting is to bring important fixes from the mainline development tree to these stable versions, allowing users
to benefit from bug fixes and security improvements without upgrading to a newer kernel version that might introduce compatibility
issues or require extensive testing.

STABLE KERNEL RULES (from Documentation/process/stable-kernel-rules.rst):

A patch must meet these criteria to be backported to stable:
1. It must be obviously correct and tested
2. It must fix a real bug
3. It must fix an important issue (security bug, data corruption, serious crash, deadlock, etc.)
4. It must be small and contained (no large refactoring or architectural changes)
5. It must NOT introduce new features or new APIs
6. It must apply cleanly to the stable tree or come with a backport

The stable tree maintainers are conservative by necessity - backporting the wrong patch can break systems in production.
A regression in stable is much more serious than a regression in mainline because stable users expect reliability.

IMPORTANT EXCEPTIONS (things that ARE allowed in stable despite being "additions"):

1. NEW DEVICE IDs (Very Common):
   - Adding PCI IDs, USB IDs, ACPI IDs, etc. to existing drivers
   - These are trivial one-line additions that enable hardware support
   - Example: "drm/i915: Add new device ID for Raptor Lake"
   - Rule: The driver must already exist in stable; only the ID is new

2. QUIRKS and WORKAROUNDS:
   - Hardware-specific quirks for broken/buggy devices
   - Suspend/resume quirks, USB quirks for non-compliant devices, PCI quirks for misconfigured hardware
   - Example: "USB: quirks: Add USB_QUIRK_DELAY_INIT for device 0x1234:0x5678"
   - These fix real-world hardware issues even though they add code

3. DEVICE TREE (DT) UPDATES:
   - DT binding additions for existing hardware, DT fixes for incorrect hardware descriptions
   - Enabling already-supported hardware on new boards
   - Usually safe because they only affect specific ARM/embedded platforms

4. BUILD FIXES:
   - Fixes for compilation errors or warnings, Kconfig dependency fixes, include file fixes
   - These are critical for users who need to build the kernel

5. DOCUMENTATION and COMMENT FIXES:
   - Fixing incorrect documentation, updating obsolete comments
   - Zero risk of runtime regression

WHEN EXCEPTIONS DON'T APPLY:
- Adding a new driver entirely (almost never stable material)
- Adding new userspace-visible APIs or syscalls (never stable material)
- Adding new module parameters that change behavior (risky)
- Enabling experimental or staging features (not for stable)

HARDWARE QUIRK DETAILS:

Hardware quirks are common in stable and deserve special attention:

1. USB QUIRKS (drivers/usb/core/quirks.c):
   - USB_QUIRK_NO_AUTOSUSPEND - device breaks on suspend
   - USB_QUIRK_RESET_RESUME - needs reset after resume
   - USB_QUIRK_DELAY_INIT - needs delay during enumeration
   - USB_QUIRK_NO_BOS - hangs when BOS descriptor requested
   - Format: { USB_DEVICE(vendor, product), .driver_info = QUIRK }

2. PCI QUIRKS (drivers/pci/quirks.c):
   - DECLARE_PCI_FIXUP_* macros for device-specific fixes
   - Power management, BAR sizing, interrupt routing fixes

3. INPUT DEVICE QUIRKS (drivers/input/serio/i8042-x86ia64io.h):
   - Keyboard/touchpad initialization quirks
   - DMI-based matching for specific laptops
   - SERIO_QUIRK_NOMUX, SERIO_QUIRK_RESET_ALWAYS, etc.

4. AUDIO CODEC QUIRKS (sound/pci/hda/patch_realtek.c):
   - SND_PCI_QUIRK entries for specific laptop models
   - Pin configuration, headphone detection fixes

5. SFP/NETWORK QUIRKS:
   - SFP_QUIRK_* for optical modules with broken behavior
   - Vendor/model string matching

These are all YES for stable - they fix real hardware for real users.

BUG PATTERN RECOGNITION:

These are the actual bug categories that get backported to stable (from analysis of 200+ stable commits).
Learn to recognize them even when the commit message doesn't explicitly say "fix":

1. USE-AFTER-FREE (UAF) - 4-6% of stable commits:
   - Object freed while still referenced elsewhere
   - Code patterns: kfree() moved after last use, adding refcount ops (get/put), NULL checks after free
   - Adding synchronize_rcu(), rcu_read_lock() around access
   - Keywords: "use after free", "UAF", "freed while", "still in use", "dangling pointer"
   - Example: ksmbd UAF in session handling (commit 1067361a1cc6a)

2. MEMORY LEAKS / RESOURCE LEAKS - 10-12% of stable commits:
   - Missing free/release on error paths or normal exit paths
   - Code patterns: Adding kfree/put_device/release in error handling, adding "goto cleanup"
   - Often labeled as "cleanup" but actually fixes real leaks
   - Common in USB (URB leaks), OF node references, driver probe paths
   - Example: 4 identical CAN USB driver URB leak fixes (adec5e1f9c99f and 3 others)

3. RACE CONDITIONS - 8-9% of stable commits:
   - Concurrent access without proper synchronization
   - Code patterns: Adding locks (spin_lock, mutex_lock, down_read/write)
   - Adding memory barriers (smp_mb, smp_wmb, smp_rmb)
   - WRITE_ONCE/READ_ONCE additions (KCSAN-detected data races)
   - Often in registration/unregistration paths, list operations, state machines
   - Example: NFSD registration vs exports_proc race (8120e420013d9)

4. DOUBLE-FREE - 3% of stable commits:
   - Same object freed twice
   - Code patterns: Setting pointer to NULL after free, removing duplicate free calls
   - Common with for_each_*() macros that call put internally, then goto cleanup also calls put
   - Example: rockchip USB2 PHY double-free in probe (ebae26dd15140)

5. NULL POINTER DEREFERENCES - 2-4% of stable commits:
   - Accessing memory through NULL pointer
   - Code patterns: Adding NULL checks before dereference, checking return values
   - Often in error paths where allocation failed but code continues

6. BUFFER OVERFLOWS / OUT-OF-BOUNDS - 1-2% of stable commits:
   - Reading/writing beyond allocated bounds
   - Code patterns: Array bounds checks, size validation, fixing size vs count confusion
   - Example: ALSA scarlett2 buffer overflow - checked total size instead of element size (91a756d22f048)

7. REFERENCE COUNTING BUGS - Common cause of UAF:
   - Imbalanced get/put operations on reference-counted objects
   - Code patterns: Adding missing put_*(), get_*(), *_get(), *_put() calls
   - Functions: kobject_put, kref_put, dev_put, fput, dput, iput, of_node_put
   - Missing put in error path = leak; extra put = UAF when refcount hits zero

8. UNINITIALIZED DATA - 3-4% of stable commits:
   - Using variables before initialization
   - Code patterns: Adding initializers, kmem_cache_alloc -> kmem_cache_zalloc
   - Detected by KMSAN - these reports indicate real bugs
   - Example: ntfs3 uninitialized memory use (7d52c592cf53f)

9. DEADLOCKS / LOCK ORDERING - 1-2% of stable commits:
   - Circular lock dependencies or self-deadlock
   - Code patterns: Changing lock acquisition order, expanding lock scope
   - Example: hugetlb migration lock ordering fix (526394af4e8ad)

10. ENDIANNESS / TYPE BUGS - 1-2% of stable commits:
    - Wrong byte order or type size assumptions
    - Code patterns: Adding cpu_to_le/be conversions, fixing GENMASK values
    - Common in IIO sensors, network drivers

SYZBOT AND FUZZER-FOUND BUGS:

"Reported-by: syzbot" indicates a bug found by Google's kernel fuzzer. These are almost always
backport-worthy because:
- Syzbot only reports REPRODUCIBLE issues with a concrete trigger
- The bug is reachable from syscalls (not theoretical)
- Usually security-relevant (exploitable from userspace)

Syzbot indicators:
- "Reported-by: syzbot+<hash>@syzkaller.appspotmail.com"
- "Link:" to syzkaller.appspot.com bug reports
- Crash types mentioned: KASAN, KMSAN, KCSAN, UBSAN, BUG, WARNING

SANITIZER REPORTS (KASAN/KMSAN/KCSAN/UBSAN):
- KASAN: Memory safety (UAF, out-of-bounds) - HIGH priority
- KMSAN: Uninitialized memory use - HIGH priority
- KCSAN: Data race detection - MEDIUM-HIGH priority
- UBSAN: Undefined behavior (overflows, shifts) - MEDIUM priority

Commits fixing sanitizer-reported issues are strong YES candidates.

BACKPORT SIGNALS:

IMPORTANT: The commits you are reviewing do NOT have "Cc: stable@vger.kernel.org" or "Fixes:" tags - that's precisely
WHY they need manual review. If they had those tags, they would be automatically selected without review. Therefore:
- The ABSENCE of these tags is NOT a negative signal - it's expected for every commit you review
- Do NOT use "lacks Fixes: tag" or "lacks Cc: stable tag" as a reason to vote NO
- Your decision must be based purely on technical analysis of the patch itself

STRONG YES signals:
- Fixes crashes, panics, oopses, kernel BUGs, WARN_ON triggers, soft/hard lockups
- Fixes data corruption, filesystem issues, security vulnerabilities (CVE)
- Fixes deadlocks, livelocks, RCU stalls, hung tasks
- Fixes memory leaks in commonly-used code paths (not one-time init leaks)
- Fixes use-after-free, double-free, null pointer dereference
- Fixes race conditions with concrete triggers (syzbot, user reports, code analysis)
- Fixes reference counting bugs (missing get/put operations)
- Fixes buffer overflows, out-of-bounds access, integer overflows
- Fixes uninitialized data use (KMSAN reports)
- Small, surgical fix with clear scope; already tested in mainline
- Adds device IDs to existing drivers (PCI, USB, ACPI, I2C IDs)
- Adds hardware quirks/workarounds for broken devices (USB quirks, PCI quirks, codec quirks)
- Build fixes that prevent compilation
- Device tree updates for existing hardware
- Reported by syzbot/syzkaller (fuzzer-found, reproducible bugs)
- Has "Reported-by:" from multiple users (indicates real-world impact)
- Fixes flagged by KASAN, KMSAN, KCSAN, UBSAN sanitizers

STRONG NO signals:
- Adds new features that aren't fixing real hardware issues
- Adds new APIs, syscalls, or changes userspace-visible behavior
- Adds entirely new drivers (not device IDs to existing drivers)
- Large refactoring or code reorganization (>100 lines across many files) without fixing a bug
- Changes that require other commits to work (check for "patch X/Y" in subject)
- Optimization-only changes that don't fix user-visible bugs or prevent lockups
- Changes to new code that doesn't exist in stable trees (new subsystems added after branch point)
- Code cleanup that doesn't fix an actual bug (pure style, whitespace, renaming)
- Changes to staging drivers (usually not stable material)
- Adds new module parameters or sysfs attributes that change behavior
- "Preparation" commits that just refactor for a future fix (no standalone value)
- Reverts of commits not in stable (nothing to revert there)

IMPORTANT NUANCES:
- "Theoretical" races ARE backport-worthy if: the race window is realistic AND
  consequences are severe (crash, corruption, security). Don't dismiss races just
  because no one reported them - they may be hard to reproduce.
- Performance fixes ARE backport-worthy if they prevent: soft lockups, hard lockups,
  RCU stalls, or system hangs. These are stability issues, not optional optimizations.
- "Cleanup" that adds error handling IS often a bug fix (fixing resource leaks).

RECOGNIZING FIXES DISGUISED AS CLEANUP/IMPROVEMENT:

Many real bug fixes are described as "cleanup", "improvement", or "simplify" but actually fix bugs.
Dig deeper when you see these patterns:

1. ERROR PATH CHANGES:
   - "Clean up error handling" - Often fixes resource leaks on error paths
   - Adding goto labels for cleanup - Usually fixing missing cleanup
   - Adding rollback/undo operations - Fixing incomplete error recovery
   - Question: Does this ensure ALL resources are freed on ALL error paths?

2. INITIALIZATION CHANGES:
   - "Initialize variable X" - Fixes uninitialized variable use
   - "Zero-initialize structure" - Prevents info leaks or undefined behavior
   - kmem_cache_alloc -> kmem_cache_zalloc - Fixing KMSAN-detected bug

3. SYNCHRONIZATION CHANGES:
   - "Clean up locking" - Often fixes races or deadlocks
   - Moving lock scope - Fixing a race window
   - Adding memory barriers - Fixing data races

4. REFERENCE COUNTING CHANGES:
   - "Balance refcount" - Fixes leak (missing put) or UAF (extra put)
   - Adding *_put() call - This IS a bug fix

5. RETURN VALUE CHECKING:
   - "Add missing check for X" - Fixes NULL deref, overflow, or other bug
   - "Handle error from Y" - Fixing ignored error that causes problems

RULE OF THUMB: If the "cleanup" prevents ANY of these, it's a bug fix worth backporting:
- Memory/resource leak
- NULL pointer dereference
- Use-after-free
- Race condition
- Data corruption
- System hang

BORDERLINE CASES (need deeper analysis):
- Performance fixes that address user complaints
- Fixes to rarely-used code paths
- Fixes that partially solve a problem (incomplete fixes)
- Changes that are large but fix critical issues
- Reverts: Check if the original commit is in stable. If yes, revert is likely needed.
  If no, the revert has nothing to revert in stable (NO).
- Commits that fix issues introduced by OTHER stable commits (stable-to-stable fixes):
  These are important - check if the broken commit went to stable.
- Incomplete fix series: If commit says "patch 2/3", the fix may need other patches.
  Check if it's self-contained or needs dependencies.

ANALYSIS WORKFLOW - MANDATORY INVESTIGATION CHECKLIST:

You MUST work through EVERY step below in order. Do NOT skip steps. For each step,
record your findings as instructed. If a step is genuinely not applicable (e.g., no
Fixes: tag to follow), state why and move on. If a tool call fails, note the failure
and what you could not verify.

This is a deep forensic investigation. You are not skimming - you are drilling down
into every available data source to build a complete picture before making your decision.
Spare no effort. Use every tool at your disposal. Leave no stone unturned.

===============================================================
PHASE 1: COMMIT MESSAGE FORENSICS
===============================================================

Step 1.1: PARSE THE SUBJECT LINE
  - Extract the subsystem prefix (e.g., "net: tcp:", "drm/i915:", "mm/slub:")
  - Identify the action verb: "fix", "add", "remove", "refactor", "clean up",
    "improve", "handle", "prevent", "ensure", "avoid", "correct"
  - Note: verbs like "handle", "prevent", "ensure", "avoid", "correct" almost
    always indicate a bug fix even if "fix" is not used
  Record: [subsystem] [action verb] [one-line summary of claimed intent]

Step 1.2: PARSE ALL COMMIT MESSAGE TAGS
  Extract and record EVERY tag present. Each tag type carries specific information:

  a) Fixes: <sha> ("commit title")
     - This tells you EXACTLY which commit introduced the bug
     - You MUST look up this commit later (Step 3.2)
     - Note: absence of Fixes: tag is EXPECTED (that's why this commit needs review)

  b) Reported-by: Name <email>
     - Single reporter = someone hit this bug
     - Multiple reporters = widespread real-world issue (strong YES signal)
     - "syzbot+<hash>@syzkaller.appspotmail.com" = fuzzer-found, always significant
     - "kernel test robot" = build/static analysis issue

  c) Tested-by: Name <email>
     - The fix has been verified by someone other than the author
     - Strong quality signal

  d) Reviewed-by: Name <email>
     - A developer has reviewed the code (check if they are the subsystem maintainer)

  e) Acked-by: Name <email>
     - A maintainer has acknowledged the patch

  f) Link: URL
     - lore.kernel.org links = mailing list discussion (investigate in Phase 4)
     - syzkaller.appspot.com links = syzbot bug report (investigate in Phase 4)
     - bugzilla.kernel.org links = user-filed bug report
     - github.com links = may reference related work

  g) Cc: stable@vger.kernel.org
     - Explicit stable nomination (note: absence is expected and NOT a negative signal)

  h) Signed-off-by: (ignore Sasha Levin's SOB - it's added by the pipeline)
     - The author's SOB tells you who wrote the fix
     - Other SOBs show the patch's journey through maintainer trees

  Record: Complete list of all tags with their values. Flag any notable patterns
  (multiple Reported-by, syzbot, etc.)

Step 1.3: ANALYZE THE COMMIT BODY TEXT
  - Read every sentence of the commit body carefully
  - Identify: What bug is described? What symptom does the user see?
  - Look for: stack traces, error messages, reproduction steps, crash descriptions
  - Look for: "can cause", "may lead to", "results in", "triggers" - these describe
    the failure mode
  - Look for references to other commits, patches, or discussions
  - Look for explicit mention of kernel versions affected
  - Distinguish between: the author explaining the bug mechanism vs. describing
    the fix approach
  Record: [Bug description as understood from message] [Symptom/failure mode]
  [Any version information] [Author's explanation of root cause]

Step 1.4: DETECT HIDDEN BUG FIXES
  Many commits are bug fixes disguised with non-fix language:
  - "Clean up error handling" -> often fixes resource leaks on error paths
  - "Improve locking" -> often fixes race conditions
  - "Simplify X" -> often fixes complex bug-prone logic
  - "Handle X properly" -> almost certainly fixing incorrect handling
  - "Ensure X" -> adding a missing safety check
  - "Initialize X" -> fixing uninitialized variable use
  - "Balance refcount" -> fixing leak (missing put) or UAF (extra put)
  - "Add missing check" -> fixing NULL deref, overflow, or other crash
  Record: [Is this commit likely a hidden bug fix? Why or why not?]

===============================================================
PHASE 2: DIFF ANALYSIS - LINE BY LINE
===============================================================

Step 2.1: INVENTORY THE CHANGES
  - List every file modified and whether lines were added, removed, or changed
  - Count total lines added/removed (rough scope metric)
  - Identify which functions are modified (look at @@ hunk headers)
  - Classify the change: single-file surgical fix, multi-file refactor, or
    cross-subsystem change
  Record: [Files changed with +/- counts] [Functions modified] [Scope classification]

Step 2.2: UNDERSTAND THE CODE FLOW CHANGE
  For each hunk in the diff:
  - What was the code doing BEFORE this change?
  - What does the code do AFTER this change?
  - What specific execution path is affected?
  - Is this changing a normal path, an error path, or an initialization path?
  Record: [For each hunk: before behavior -> after behavior]

Step 2.3: IDENTIFY THE BUG MECHANISM
  Based on the diff, classify what kind of bug this fixes (if any):

  a) Error path / resource leak fixes:
     - New goto labels, new cleanup code before returns
     - Added kfree/put_device/release/of_node_put on error paths
     - "goto err" additions, new error handling blocks

  b) Synchronization / race condition fixes:
     - Added locks: spin_lock*, mutex_lock*, down_read/write*, raw_spin_lock*
     - Memory barriers: smp_mb, smp_wmb, smp_rmb, smp_store_release, smp_load_acquire
     - RCU: rcu_read_lock, synchronize_rcu, rcu_dereference, call_rcu
     - Atomics: READ_ONCE, WRITE_ONCE, atomic_*, xchg, cmpxchg

  c) Reference counting fixes:
     - get_device/put_device, kobject_get/put, kref_get/put
     - of_node_get/put, fget/fput, dget/dput, iget/iput
     - try_module_get/module_put
     - Check: is a put being ADDED (leak fix) or REMOVED (double-free fix)?

  d) Memory safety fixes:
     - NULL checks before dereference
     - Bounds checks before array access (ARRAY_SIZE, min_t, max_t)
     - Size validation before copy/allocation
     - Moving kfree() after last use (UAF fix)
     - Adding synchronize_*() before free (RCU UAF fix)

  e) Initialization fixes:
     - Adding initializers, zeroing memory
     - kmem_cache_alloc -> kmem_cache_zalloc
     - memset additions

  f) Type / endianness fixes:
     - cpu_to_le/be conversions
     - GENMASK fixes
     - Type cast corrections

  g) Logic / correctness fixes:
     - Conditional expression changes (wrong operator, inverted condition)
     - Off-by-one fixes
     - Missing break in switch
     - Wrong variable used

  h) Hardware workarounds:
     - Device ID additions (PCI, USB, ACPI, I2C, SPI, OF)
     - Quirk table entries
     - DMI match table entries
     - Timing/delay adjustments for hardware

  Record: [Bug category] [Specific mechanism: what was broken and how the fix works]

Step 2.4: ASSESS THE FIX QUALITY
  - Is the fix obviously correct? Can you verify it by reading the code?
  - Is it minimal / surgical, or does it include unrelated changes?
  - Could the fix itself introduce a regression? (e.g., does adding a lock
    risk deadlock? does a NULL check change the return path?)
  - Are there any red flags: large scope, touching multiple subsystems,
    changing public APIs?
  Record: [Fix quality assessment] [Any regression risk from the fix itself]

===============================================================
PHASE 3: GIT HISTORY INVESTIGATION
===============================================================

Use git commands to investigate the history. REMEMBER: Never use --all flag.

Step 3.1: BLAME THE CHANGED LINES
  Run git blame on the file(s) modified to understand the history of the
  code being changed:
  - git blame <file> -- for lines around the changed area
  - When was the buggy code introduced? By which commit?
  - Has this code been stable for a long time, or was it recently added?
  - A bug in code present since v4.x is more important than a bug in code
    added last month (more stable trees affected)
  Record: [Commit that introduced the buggy code] [Kernel version it first appeared]

Step 3.2: FOLLOW THE FIXES: TAG (if present)
  If the commit message has a "Fixes:" tag:
  - git show <fixes_sha> - read the commit that introduced the bug
  - Understand what the original commit did and why it was buggy
  - git log --oneline <fixes_sha>..HEAD -- <files> - check if there were
    intermediate fixes for the same issue
  - Determine: does the Fixes: target exist in stable trees?
  Record: [Original buggy commit details] [Whether it exists in stable trees]
  [Any intermediate related fixes]

Step 3.3: CHECK FILE HISTORY FOR RELATED CHANGES
  - git log --oneline -20 -- <modified_files> - recent changes to the same files
  - Look for: related fixes, prerequisite refactoring, or patch series
  - Are there other commits that are prerequisites for this fix?
  - Is this part of a series (check for "patch X/Y" or sequential commits
    by the same author touching the same code)?
  Record: [Related recent commits] [Any prerequisites identified]
  [Is this standalone or part of a series?]

Step 3.4: CHECK THE AUTHOR'S OTHER COMMITS
  - git log --oneline --author="<author>" -10 -- <subsystem_path>
  - Is the author the subsystem maintainer or a regular contributor?
  - Did the author submit other related fixes around the same time?
  - A fix from the subsystem maintainer carries more weight
  Record: [Author's relationship to the subsystem] [Related author commits]

Step 3.5: CHECK FOR DEPENDENT/PREREQUISITE COMMITS
  - Does this fix reference other commits that must be applied first?
  - Does the diff assume code structures or functions that may not exist
    in stable trees?
  - git log --oneline --grep="<function_name>" -- to find related commits
  Record: [Dependencies found, if any] [Can this apply standalone?]

===============================================================
PHASE 4: MAILING LIST AND EXTERNAL RESEARCH
===============================================================

Use "b4 dig" and WebFetch to investigate mailing list history. This phase is
critical for borderline cases. Do not skip it.

TOOL: b4 dig - FINDING THE ORIGINAL PATCH SUBMISSION

"b4 dig" is the primary tool for tracing a commit back to its original mailing
list submission on lore.kernel.org. Given a commit hash, it matches by patch-id,
author, and subject to find the exact thread where the patch was submitted and
reviewed. This is MORE RELIABLE than manual keyword searches on lore.

Usage:
  b4 dig -c <commit_hash>
    Finds the original patch submission and returns the lore.kernel.org URL
    for the matching thread. This is the fastest way to find a patch's
    discussion thread. Use this FIRST before falling back to manual searches.

  b4 dig -c <commit_hash> -w
    Same as above, but also shows the full list of people who were originally
    included on the patch (To/Cc recipients). This tells you which maintainers,
    reviewers, and mailing lists were involved in the review. Useful for
    understanding who vetted the patch and whether the right people reviewed it.

  b4 dig -c <commit_hash> -a
    Shows ALL revisions of the patch series (v1, v2, v3, etc.), not just the
    latest matching version. This is critical for understanding how the patch
    evolved during review - were there significant changes between versions?
    Did a reviewer raise concerns that were addressed in a later version?
    Each revision is listed with its date, subject, and lore URL.

  b4 dig -c <commit_hash> -m /tmp/thread.mbox
    Saves the entire matched thread (patch + all replies) to an mbox file.
    You can then read the mbox to see the full review discussion, including
    reviewer comments, NAKs, Acked-by responses, and stable nominations.

Notes on b4 dig:
  - It searches lore.kernel.org automatically - no need to construct URLs manually
  - It matches by patch-id (computed from the diff), so it finds the right thread
    even if the subject line was slightly different during submission
  - If b4 dig fails to find a match (some very old or unusual patches), fall back
    to manual lore.kernel.org searches as described below
  - Use -C (--no-cache) if you suspect stale cached results

Step 4.1: FIND THE ORIGINAL PATCH DISCUSSION
  First, use b4 dig to find the original submission:
  - b4 dig -c <commit_hash>
  - If b4 dig finds a match, use the returned lore URL to fetch the discussion:
    WebFetch <lore_url> to read the thread
  - If b4 dig does not find a match, fall back to manual search:
    WebFetch https://lore.kernel.org/all/?q=<commit_subject_keywords>
    Try the commit subject (minus subsystem prefix) as search terms
    Try the author's name + key function names

  Then investigate the patch evolution:
  - b4 dig -c <commit_hash> -a  to see all series revisions
  - Check: Is the committed version the latest revision, or was there a newer
    version that was NOT applied? Multiple revisions suggest review feedback
    that may be relevant to stability assessment.

  In the discussion, look for:
  - Was this patch explicitly nominated for stable by a reviewer?
    (Look for "Cc: stable" suggestions in review comments)
  - Were there any NAKs or objections?
  - Did reviewers request changes that were made in later versions?
  - Is this the latest version of the patch, or was there a v2/v3/v4?
  - Did the maintainer add any comments about stable suitability?
  - Did any reviewer flag risks or concerns?
  Record: [Lore discussion URL from b4 dig] [Series revisions found]
  [Key reviewer feedback] [Any stable nominations by reviewers]
  [Any NAKs or concerns]

Step 4.2: CHECK WHO REVIEWED THE PATCH
  Use b4 dig -w to see the original recipients:
  - b4 dig -c <commit_hash> -w
  - Were the appropriate subsystem maintainers included?
  - Were relevant mailing lists CC'd?
  - A patch reviewed by the subsystem maintainer carries more weight
  Record: [Key reviewers/maintainers from b4 dig -w]

Step 4.3: SEARCH FOR THE BUG REPORT (if applicable)
  If there are Link: tags or Reported-by: tags:
  - Follow Link: URLs using WebFetch
  - For syzbot links: check the crash type, reproducer, and affected versions
  - For bugzilla links: check the severity, number of duplicates, and user reports
  - Search lore for the original bug report email:
    WebFetch https://lore.kernel.org/all/?q=<bug_description_keywords>
  Record: [Bug report details] [Severity from reporter's perspective]
  [How many users affected]

Step 4.4: CHECK FOR RELATED PATCHES AND SERIES
  - If b4 dig -a showed this is part of a multi-patch series, investigate:
    Are the other patches in the series also needed for stable?
  - Search lore for other patches in the same series:
    WebFetch https://lore.kernel.org/all/?q=<author_name>+<subsystem>
  - Is this a standalone fix or does it depend on other patches in the series?
  Record: [Related patches found] [Series context if applicable]

Step 4.5: CHECK STABLE MAILING LIST HISTORY
  - Search for whether this fix or the underlying bug has been discussed
    in the context of stable:
    WebFetch https://lore.kernel.org/stable/?q=<search_terms>
  - Check if someone already requested this for stable
  - Check if there's a known reason it was NOT selected for stable
  Record: [Any stable-specific discussion found]

===============================================================
PHASE 5: CODE SEMANTIC ANALYSIS
===============================================================

Use semcode MCP tools if available (mcp__semcode__*), otherwise use git grep
and manual analysis. This phase maps the impact surface of the change.

Step 5.1: IDENTIFY KEY FUNCTIONS IN THE DIFF
  - If semcode available: diff_functions to extract all function names modified
  - Otherwise: read the @@ hunk headers to identify modified functions
  - For each modified function, proceed with steps 5.2-5.4
  Record: [List of functions modified or added in the diff]

Step 5.2: TRACE CALLERS (Who calls the affected code?)
  - If semcode available: find_callers for each key function modified
  - Otherwise: git grep "<function_name>" to find call sites
  - This tells you the IMPACT SURFACE: how many code paths are affected?
  - Pay special attention to: is this called from interrupt context?
    From syscall handlers? From hot paths? From module init/cleanup?
  - A fix to a function called from 50 places has higher impact than
    one called from 1 place
  Record: [For each function: list of callers and calling context]

Step 5.3: TRACE CALLEES (What does the affected code call?)
  - If semcode available: find_calls for each key function
  - Otherwise: read the function body in context
  - This helps understand the function's purpose and side effects
  - Look for: memory allocation calls, lock operations, I/O operations
  Record: [Key callees that help understand the function's role]

Step 5.4: FOLLOW THE CALL CHAIN (for bug reachability)
  - If semcode available: find_callchain for critical functions
  - Otherwise: manually trace the most important call path
  - This answers: "Can a user actually trigger this code path?"
  - Code reachable from syscalls or common device operations = HIGH priority
  - Code only reachable from obscure config options = LOWER priority
  Record: [Call chain summary] [Is the buggy path reachable from userspace?]

Step 5.5: SEARCH FOR SIMILAR PATTERNS
  - If semcode available: grep_functions for similar code patterns
  - Otherwise: git grep for the bug pattern in nearby code
  - Are there other instances of the same bug? (suggests systematic issue)
  - Was the same fix applied to sibling functions/drivers?
  Record: [Similar patterns found, if any]

===============================================================
PHASE 6: CROSS-REFERENCING AGAINST THE LOCAL TREE
===============================================================

Every step here is about the tree you have checked out. Inspect it directly rather
than reasoning from memory about what some stable series contains.

Step 6.1: DOES THE BUGGY CODE EXIST IN THIS TREE?
  - Check whether the file and the specific buggy code exist in this checkout
  - git log --oneline -- <file> - see what changed in this tree
  - If the buggy code was introduced after this tree branched, the fix is
    irrelevant here (the bug does not exist in it) - that is a NO
  - If the code has been in the tree for many releases, the fix is more valuable
  Record: [Does the buggy code exist in this tree? Where?]
  [Approximate kernel version the bug was introduced]

Step 6.2: CHECK FOR BACKPORT COMPLICATIONS
  - Has the file been significantly refactored between mainline and this tree?
  - Would this patch apply cleanly here, or would it need modification?
  - Are there conflicting changes in this tree?
  - git log --oneline -10 -- <file> - see recent churn in the file
  Record: [Expected backport difficulty: clean apply / minor conflicts / needs rework]

Step 6.3: CHECK IF RELATED FIXES ARE ALREADY HERE
  - Has a different fix for the same bug already been applied to this tree?
  - Is there an earlier version of this fix already in it?
  - git log --oneline --grep="<bug_description>"
  Record: [Any related fixes already present?]

===============================================================
PHASE 7: SUBSYSTEM AND MAINTAINER CONTEXT
===============================================================

Step 7.1: IDENTIFY THE SUBSYSTEM AND ITS CRITICALITY
  - From the file paths, determine the kernel subsystem:
    mm/ = memory management, fs/ = filesystems, net/ = networking,
    drivers/ = device drivers (note the sub-category), kernel/ = core,
    arch/ = architecture-specific, security/ = security modules,
    sound/ = audio, crypto/ = cryptography, block/ = block layer
  - Criticality:
    - CORE (mm, kernel, fs/vfs, net/core, block) = affects all users
    - IMPORTANT (specific filesystems, network protocols, common drivers) = affects many users
    - PERIPHERAL (obscure drivers, arch-specific, staging) = affects fewer users
  Record: [Subsystem] [Criticality level: CORE/IMPORTANT/PERIPHERAL]

Step 7.2: ASSESS SUBSYSTEM ACTIVITY
  - git log --oneline -20 -- <subsystem_path> - is this actively developed?
  - Active subsystem = more likely the fix is for a recently introduced bug
  - Mature/stable subsystem = bug has been present longer, fix is more important
  Record: [Subsystem activity level]

===============================================================
PHASE 8: IMPACT AND RISK ASSESSMENT
===============================================================

Step 8.1: DETERMINE WHO IS AFFECTED
  - Core kernel paths (mm, scheduler, VFS) -> ALL users affected
  - Filesystem-specific -> users of that filesystem
  - Driver-specific -> users of that hardware
  - Architecture-specific -> users of that platform
  - Config-dependent -> only if CONFIG_X is set
  Record: [Affected user population: universal / filesystem-specific /
  driver-specific / config-specific / platform-specific]

Step 8.2: DETERMINE THE TRIGGER CONDITIONS
  - How common is the trigger? Every boot? Every file operation? Only under
    memory pressure? Only during error recovery? Only with specific hardware?
  - Can an unprivileged user trigger it? (security relevance)
  - Is it a timing-dependent race? (hard to reproduce but real)
  Record: [Trigger conditions] [How common/likely is triggering]
  [Can unprivileged users trigger it?]

Step 8.3: DETERMINE THE FAILURE MODE SEVERITY
  When the bug triggers, what happens?
  - Kernel panic / BUG() / oops -> CRITICAL (system crash, data loss risk)
  - Deadlock / hang / soft lockup -> CRITICAL (system unusable)
  - Data corruption (filesystem, memory) -> CRITICAL (silent data loss)
  - Security vulnerability (privilege escalation, info leak) -> CRITICAL
  - Memory leak in hot path -> HIGH (eventual OOM, system degradation)
  - Use-after-free -> HIGH (crash or security, unpredictable)
  - WARN_ON / error message only -> MEDIUM (no functional impact)
  - Performance degradation -> MEDIUM (if severe) to LOW
  - Incorrect error code returned -> LOW to MEDIUM
  - Cosmetic / logging issue -> LOW
  Record: [Failure mode] [Severity: CRITICAL/HIGH/MEDIUM/LOW]

Step 8.4: CALCULATE RISK-BENEFIT RATIO
  BENEFIT: How much does backporting this fix help stable users?
  - Prevents crashes in common paths = very high benefit
  - Fixes security hole exploitable from userspace = very high benefit
  - Fixes hardware enablement for popular devices = high benefit
  - Fixes rare race in obscure driver = low benefit

  RISK: How much could backporting this fix hurt stable users?
  - 1-5 lines, obviously correct = very low risk
  - 5-20 lines, well-contained = low risk
  - 20-50 lines, touches error paths only = low-medium risk
  - Touches locking, memory management, or hot paths = medium risk
  - >100 lines or cross-subsystem = high risk
  - Changes public APIs or data structures = very high risk
  Record: [Benefit level] [Risk level] [Ratio assessment]

===============================================================
PHASE 9: FINAL SYNTHESIS
===============================================================

Step 9.1: COMPILE THE EVIDENCE
  Review all findings from Phases 1-8. List:
  - ALL evidence FOR backporting (bug severity, user impact, fix quality,
    reviewer endorsements, syzbot reports, etc.)
  - ALL evidence AGAINST backporting (scope too large, dependencies,
    code not in stable, regression risk, etc.)
  - ALL UNRESOLVED QUESTIONS (things you could not verify)

Step 9.2: APPLY THE STABLE RULES CHECKLIST
  Check each criterion from Documentation/process/stable-kernel-rules.rst.
  Answer each one PASS or FAIL - never "yes" or "no", so that YES and NO stay
  reserved for your final verdict and cannot be mistaken for it:
  1. Obviously correct and tested? [PASS/FAIL with evidence]
  2. Fixes a real bug that affects users? [PASS/FAIL with evidence]
  3. Important issue (crash, security, corruption, deadlock)? [PASS/FAIL - what severity?]
  4. Small and contained? [PASS/FAIL - line count and scope]
  5. No new features or APIs? [PASS/FAIL]
  6. Can apply to the local tree? [PASS/FAIL/UNKNOWN]

Step 9.3: CHECK FOR EXCEPTION CATEGORIES
  Does this fall into an exception category that automatically qualifies?
  - Device ID addition to existing driver
  - Hardware quirk/workaround
  - Device tree update for existing hardware
  - Build fix
  - Documentation fix
  Record: [Exception category if applicable]

Step 9.4: MAKE YOUR DECISION
  Based on ALL the evidence gathered:
  - If the evidence clearly supports backporting: YES
  - If the evidence clearly argues against: NO
  - If borderline: weigh benefit vs. risk, erring on the side of
    including fixes that prevent crashes, security issues, or data corruption

AVOIDING FALSE NEGATIVES (Missing Real Bugs):

The goal is to find bugs that SHOULD be backported. Be vigilant for:

1. SUBTLE COMMIT MESSAGES:
   - "Improve X" - might fix a race or leak
   - "Clean up X" - might fix error handling bugs
   - "Simplify X" - might fix complex bug-prone code
   - "Handle X properly" - almost certainly a bug fix
   - "Ensure X" - likely adding missing safety check

2. SMALL BUT CRITICAL FIXES:
   - One-line fixes can prevent crashes (NULL check, bounds check)
   - Adding a single lock/unlock can fix a data race
   - Adding one 'goto err' can fix a resource leak
   - Single put_*() call addition fixes refcount bug
   - Don't dismiss small patches - they're often the most important

3. DRIVER FIXES MATTER:
   - Even "obscure" drivers have real users who need them
   - Suspend/resume fixes prevent data loss on laptops
   - Probe/remove fixes prevent use-after-free on module unload
   - Hardware quirks make specific devices work at all

4. SUBSYSTEM IMPORTANCE:
   - Memory management (mm/) - affects everyone
   - Filesystems (fs/) - data corruption = data loss
   - Networking (net/) - security exposure, connectivity
   - Block layer - data integrity
   - USB/PCI core - hardware access

5. WHEN UNCERTAIN ABOUT A FIX:
   Ask: "If this bug triggers, what happens to the user?"
   - Kernel crash/panic = YES
   - Data corruption = YES
   - Security vulnerability = YES
   - System hang/lockup = YES
   - Resource leak in hot path = YES (eventual OOM)
   - Minor issue in rare config = Probably NO

6. TRUST INDICATORS:
   - Multiple "Reported-by:" tags = multiple people hit it = real bug
   - "Link:" to bug reports/mailing list = documented issue
   - Detailed commit message explaining mechanism = author understood the bug
   - Stack traces in commit message = reproducible crash
   - Author is subsystem maintainer = they know what matters

New Commit to analyze:
{commit_info}

Based on your analysis, should this commit be backported to stable kernel trees?

========================================
RESPONSE FORMAT REQUIREMENTS:
========================================

CRITICAL: VERIFY YOUR CLAIMS:
- Every factual claim you make in your analysis MUST be verified against the actual code,
  git history, or mailing list discussions. Do NOT speculate or assume.
- If you claim a function can race, verify the call paths. If you claim code is reachable,
  trace the callers. If you claim a bug has user impact, find evidence.
- If you cannot verify a claim, explicitly state it is unverified and do not let unverified
  claims drive your final YES/NO decision.
- Do NOT fabricate commit SHAs, function behaviors, or mailing list discussions.

Your response must include:
1. A complete walkthrough of ALL 9 phases above, with "Record:" entries for each step.
   Do not skip phases. If a phase is not applicable, state why in one line and move on.
2. Detailed explanation of:
   - What problem the commit solves and why it matters to stable users
   - Whether it meets stable kernel rules (obviously correct, fixes real bug, small scope, no new features)
   - Risk vs benefit trade-off
   - Any concerns (dependencies, backport adjustments needed, affected versions)
3. A "Verification" section listing every verification action you performed and its result,
   organized by phase, e.g.:
   Verification:
   - [Phase 1] Parsed tags: found Reported-by: syzbot, Link: to syzkaller report
   - [Phase 2] Diff analysis: 3 lines added to error path in foo_probe(), adds missing kfree()
   - [Phase 3] git blame: buggy code introduced in commit abc123 (v5.15), present in this tree
   - [Phase 3] git show <fixes_sha>: confirmed original commit is in this tree
   - [Phase 4] b4 dig -c <sha>: found original submission at <lore_URL>
   - [Phase 4] b4 dig -a: patch went through v1->v3, applied version is latest
   - [Phase 4] b4 dig -w: subsystem maintainer was CC'd and reviewed the patch
   - [Phase 4] lore discussion: reviewer suggested Cc: stable
   - [Phase 5] find_callers: foo_probe() called during device enumeration, common path
   - [Phase 6] git describe HEAD: local tree is v6.12.y; buggy code present in it
   - [Phase 8] Failure mode: NULL deref -> kernel oops on device probe, severity CRITICAL
   - UNVERIFIED: Could not determine if stable 5.10.y has divergent code in this file
   Every claim in your analysis must have a corresponding entry here. If you made a claim
   without verification, list it as UNVERIFIED and do not let unverified claims drive your
   final YES/NO decision.

FINAL LINE - Your decision:
The VERY LAST LINE of your response must be EXACTLY **YES** or **NO** (nothing else after it).

The verdict line stands alone. Do NOT write it as part of a sentence
("the recommendation remains **NO** for stable backport") and do NOT label it
("**Verdict: NO**"). Do NOT add a caveat, summary, or any other text after it.

Run every command in the foreground and wait for its output. Do NOT put a command
in the background or start anything whose result arrives later: when a background
task reports back after you have answered, its text gets appended to your response
and destroys the final line. Before writing the verdict, confirm nothing is still
running and you have nothing left to add.

Example endings:
...The fix is small, surgical, and meets all stable kernel criteria.

**YES**

...This adds new functionality that is not appropriate for stable trees.

**NO**

REMEMBER: Last line MUST be exactly **YES** or **NO** - nothing else!"#;

        // Replace placeholder in the template
        let _ = similar_commits; // unused now that historical context is removed
        prompt_template.replace("{commit_info}", commit_text)
    }

    pub fn find_similar_commits(&self, commit_text: &str, k: usize) -> Vec<(String, bool)> {
        // If no vectorstore is available, return empty results
        if self.vectorstore.is_none() {
            warn!("No vectorstore available for similarity search");
            return Vec::new();
        }

        // Check if embeddings are initialized
        if !self.embeddings.is_initialized() {
            error!("Embeddings model not initialized");
            return Vec::new();
        }

        // Get the vectorstore
        let vectorstore = self.vectorstore.as_ref().unwrap();

        // Perform similarity search
        match vectorstore.similarity_search(commit_text, k) {
            Ok(scored_docs) => {
                scored_docs.into_iter()
                    .map(|scored_doc| {
                        let doc_text = scored_doc.document.text;
                        let was_backported = scored_doc.document.metadata.extra.get("was_selected")
                            .and_then(|v| {
                                if let serde_json::Value::String(s) = v {
                                    Some(s == "YES")
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(false);
                        (doc_text, was_backported)
                    })
                    .collect()
            },
            Err(e) => {
                error!("Error performing similarity search: {e}");
                Vec::new()
            }
        }
    }

    fn initialize_llms(&mut self) -> Result<(), String> {
        if self.llms.is_none() {
            let mut llms = HashMap::new();

            for provider in &self.llm_providers {
                let mut config = self.llm_configs.get(provider).cloned();

                // goose runs as an agent, so start it inside the kernel tree the
                // verdict is about; the prompt tells it to inspect that checkout
                if provider == "goose"
                    && let Some(repo) = &self.repo_path
                {
                    let obj = config.get_or_insert_with(|| serde_json::json!({}));
                    if let Some(map) = obj.as_object_mut() {
                        map.entry("cwd").or_insert_with(|| serde_json::json!(repo));
                    }
                }

                match commit_classifier::create_llm(provider, config) {
                    Ok(llm) => {
                        llms.insert(provider.clone(), llm);
                    },
                    Err(e) => {
                        return Err(format!("Failed to initialize {provider} LLM: {e}"));
                    }
                }
            }

            if llms.is_empty() {
                return Err("No valid LLM providers specified".to_string());
            }

            self.llms = Some(llms);
        }

        Ok(())
    }

    // Helper method to invoke an LLM with retry logic for common error cases
    async fn invoke_llm_with_retry(&self, llm: &dyn Llm, prompt: &str, provider: &str, verbose: bool)
        -> Result<(bool, String, String), String> {
        // Maximum number of retries for errors
        let max_retries = 3;
        let mut retry_count = 0;

        loop {
            match llm.invoke(prompt).await {
                Ok(response) => {
                    // Check if the response is just "yes" (case insensitive)
                    let trimmed_response = response.trim().to_lowercase();
                    if trimmed_response == "yes" && retry_count < max_retries {
                        retry_count += 1;
                        warn!("Received bare 'yes' response from {provider} LLM, retrying ({retry_count}/{max_retries})");

                        // Wait a bit before retrying
                        tokio::time::sleep(std::time::Duration::from_millis(1000 * retry_count)).await;
                        continue; // Try again
                    }

                    // Show full response when verbose is enabled
                    if verbose {
                        info!("===== FULL RESPONSE FROM {} =====", provider.to_uppercase());
                        info!("{response}");
                        info!("==========================================");
                    }

                    let (should_backport, explanation) = Self::parse_llm_response(&response);
                    return Ok((should_backport, explanation, response));
                },
                Err(e) => {
                    let err_str = e.to_string();

                    // Check if we still have retries available
                    if retry_count < max_retries {
                        retry_count += 1;

                        // Log specific error types differently
                        if err_str.contains("Failed to parse response: EOF while parsing a value at line 1 column 0") {
                            warn!("Encountered EOF parsing error with {provider} LLM, retrying ({retry_count}/{max_retries})");
                        } else {
                            warn!("Error invoking {provider} LLM: {err_str}, retrying ({retry_count}/{max_retries})");
                        }

                        // Wait a bit before retrying with exponential backoff
                        let backoff_ms = 1000 * retry_count;
                        tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                        continue; // Try again
                    }

                    // We've exhausted all retries
                    error!("Error invoking {provider} LLM after {max_retries} retries: {err_str}");
                    return Err(format!("Error: {err_str}"));
                }
            }
        }
    }

    pub async fn predict(&mut self, commit_info: &CommitFeatures, verbose: bool) -> Result<PredictionResult, String> {
        // Initialize LLMs if not already done
        self.initialize_llms()?;

        // Generate the vector database search for context
        let commit_text = Self::format_commit_info(commit_info);

        let similar_commits = if self.vectorstore.is_some() {
            // Ensure embeddings are initialized only if they aren't already
            if !self.embeddings.is_initialized() {
                info!("Initializing embeddings model for prediction");
                if let Err(e) = self.embeddings.initialize() {
                    return Err(format!("Failed to initialize embeddings model: {e}"));
                }
            }
            // Get similar commits for context
            self.find_similar_commits(&commit_text, 5)
        } else {
            // No RAG when vectorstore is not available
            info!("Running without RAG features - no similar commits will be used");
            Vec::new()
        };

        // Construct prompt with context and commit info
        let prompt = Self::construct_prompt(&commit_text, &similar_commits);

        // Show full prompt when verbose is enabled
        if verbose {
            info!("===== FULL PROMPT =====");
            // Print each line without extra indentation
            for line in prompt.lines() {
                info!("{line}");
            }
            info!("=======================");
        }

        // Process the prompt and get the prediction result
        let result = self.process_with_llms(&prompt, verbose).await?;

        // Return the consolidated results with commit-specific information
        Ok(PredictionResult {
            sha: commit_info.sha.clone(),
            subject: commit_info.message.lines().next().unwrap_or("Unknown").to_string(),
            should_select: result.should_select,
            vote_ratio: result.vote_ratio,
            provider_results: result.provider_results,
        })
    }

    // Extract a YES/NO verdict from the bold (**...**) spans of a single line
    // Matches a bare "**NO**" embedded in a sentence as well as a labelled span
    // such as "**Stable backport recommendation: NO**". The last verdict on the
    // line wins, since models tend to build up to their final answer
    //
    // A line holding both verdicts states no conclusion - it is a criterion split
    // per tree ("**NO** for 6.6, **YES** for 6.12"), or our own prompt echoed back
    // into the output ("must be EXACTLY **YES** or **NO**") - so it yields nothing
    fn bold_verdict_in_line(line: &str) -> Option<bool> {
        let mut verdict = None;
        let mut rest = line;

        while let Some(open) = rest.find("**") {
            let after_open = &rest[open + 2..];
            let Some(close) = after_open.find("**") else {
                break;
            };
            let span = &after_open[..close];
            rest = &after_open[close + 2..];

            match Self::verdict_from_span(span) {
                Some(true) if verdict == Some(false) => return None,
                Some(false) if verdict == Some(true) => return None,
                Some(v) => verdict = Some(v),
                None => {}
            }
        }

        verdict
    }

    // Read a verdict out of the text of one bold span
    fn verdict_from_span(span: &str) -> Option<bool> {
        let word_verdict = |word: &str| match word.trim().to_uppercase().as_str() {
            "YES" => Some(true),
            "NO" => Some(false),
            _ => None,
        };

        let span = span.trim().trim_end_matches(['.', '!', ':', ',']).trim();

        // Text after the last colon, so "Recommendation: NO" is recognised while a
        // checklist label like "Tested:" is not
        let tail = span.rsplit(':').next().unwrap_or(span);
        if let Some(verdict) = word_verdict(tail) {
            return Some(verdict);
        }

        // A whole span that states the verdict in words ("Backport recommendation
        // remains YES") ends on the answer. The keyword is what separates it from
        // an ordinary sentence that happens to end in yes or no
        let upper = span.to_uppercase();
        if ["DECISION", "VERDICT", "RECOMMENDATION", "ANSWER", "CONCLUSION", "ASSESSMENT",
            "DETERMINATION", "BACKPORT"].iter().any(|keyword| upper.contains(keyword))
        {
            let last_word = span.split_whitespace().next_back()?;
            return word_verdict(last_word.trim_matches(|c: char| !c.is_alphanumeric()));
        }

        None
    }

    // Check an uppercased line for "<label>: <verdict>", rejecting matches that
    // run into a longer word (e.g. "DECISION: NOT CLEAR" is not a NO)
    fn has_labeled_verdict(line_upper: &str, label: &str, verdict: &str) -> bool {
        for separator in [": ", ":"] {
            let pattern = format!("{label}{separator}{verdict}");
            for (idx, _) in line_upper.match_indices(&pattern) {
                let next = line_upper[idx + pattern.len()..].chars().next();
                if !next.is_some_and(|c| c.is_alphanumeric()) {
                    return true;
                }
            }
        }
        false
    }

    // Parse LLM response to extract decision and explanation
    // The prompt requires the response to END with **YES** or **NO** on the last line
    // but we have fallback patterns for LLMs that don't follow the format exactly
    pub fn parse_llm_response(response: &str) -> (bool, String) {
        // First, filter out any content inside <think> blocks
        let filtered_response = if response.contains("<think>") && response.contains("</think>") {
            // Extract content outside of think blocks
            let mut result = String::new();
            let mut in_think_block = false;

            for line in response.lines() {
                if line.contains("<think>") {
                    in_think_block = true;
                    continue;
                }
                if line.contains("</think>") {
                    in_think_block = false;
                    continue;
                }

                if !in_think_block {
                    result.push_str(line);
                    result.push('\n');
                }
            }

            result
        } else {
            response.to_string()
        };

        // If we got an error message back
        if filtered_response.starts_with("Error:") {
            return (false, format!("LLM Error: {filtered_response}"));
        }

        let trimmed = filtered_response.trim();

        // Find the last non-empty line - this is where the decision should be
        // Use trimmed version for finding the line
        let last_line_trimmed = trimmed.lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("")
            .trim();

        // PRIORITY 1: Check for exact format on last line (highest priority)
        // The prompt explicitly requires **YES** or **NO** on the last line
        // Bold patterns allow surrounding whitespace
        if last_line_trimmed == "**YES**" {
            return (true, filtered_response.to_string());
        }
        if last_line_trimmed == "**NO**" {
            return (false, filtered_response.to_string());
        }

        // For plain YES/NO, check the raw last line - must be at start of line (no leading whitespace)
        // This prevents false positives from indented text like "\t\tYES\t\t"
        let last_line_raw = filtered_response.lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("");

        if last_line_raw == "YES" || last_line_raw.starts_with("YES ") {
            return (true, filtered_response.to_string());
        }
        if last_line_raw == "NO" || last_line_raw.starts_with("NO ") {
            return (false, filtered_response.to_string());
        }

        // If we get here, the LLM didn't follow the format correctly
        // Log a warning and try fallback patterns
        warn!("LLM response did not end with **YES** or **NO** as required. Last line: '{}'", last_line_trimmed);

        // PRIORITY 2: Scan backwards for the last bold (**...**) verdict in the response
        // The verdict belongs on the last line, so when a model keeps talking - or the
        // agent harness appends a background-task notice right after it, with no newline
        // ("**NO**The background search ... finished") - the LAST bold verdict is still
        // the conclusion. Checklist items ("- **Tested:** YES") never match, because the
        // bold span itself has to be the verdict
        for line in trimmed.lines().rev() {
            let line_trimmed = line.trim();

            // Rules-check table rows ("| Fixes a real bug? | **NO** |") and checklist
            // bullets ("- **Applies to 6.6:** **NO**") answer one criterion each, not
            // the overall question. Note "* " needs the space: **NO** starts with '*'
            if line_trimmed.starts_with(['|', '>'])
                || line_trimmed.starts_with("- ")
                || line_trimmed.starts_with("* ")
                || line_trimmed.starts_with("+ ") {
                continue;
            }

            if let Some(verdict) = Self::bold_verdict_in_line(line_trimmed) {
                warn!("Found bold verdict in response but not alone on the last line");
                return (verdict, filtered_response.to_string());
            }
        }

        // PRIORITY 3: Check for markdown heading patterns (## YES / ## NO)
        for line in trimmed.lines() {
            let line_trimmed = line.trim();
            if line_trimmed.starts_with("## YES") || line_trimmed.starts_with("##YES") {
                warn!("Found ## YES markdown heading pattern");
                return (true, filtered_response.to_string());
            }
            if line_trimmed.starts_with("## NO") || line_trimmed.starts_with("##NO") {
                warn!("Found ## NO markdown heading pattern");
                return (false, filtered_response.to_string());
            }
        }

        // PRIORITY 4: Check for labelled verdicts (DECISION: YES, Recommendation: NO, ...)
        // Bold markers are stripped so "**Verdict: NO**" is matched the same way
        for line in trimmed.lines() {
            let line_upper = line.to_uppercase().replace("**", "");
            for label in ["DECISION", "RECOMMENDATION", "VERDICT", "ANSWER"] {
                if Self::has_labeled_verdict(&line_upper, label, "YES") {
                    warn!("Found {label}: YES pattern");
                    return (true, filtered_response.to_string());
                }
                if Self::has_labeled_verdict(&line_upper, label, "NO") {
                    warn!("Found {label}: NO pattern");
                    return (false, filtered_response.to_string());
                }
            }
        }

        // PRIORITY 5: Check for plain YES/NO at start of line (no leading whitespace allowed)
        // This is stricter than bold patterns to avoid false positives from indented text
        for line in filtered_response.lines() {
            // Check for YES at actual start of line (not indented), or YES followed by colon/comma/space
            // Don't match if there's leading whitespace - that suggests it's not the main answer
            if line == "YES"
                || line.starts_with("YES:")
                || line.starts_with("YES,")
                || line.starts_with("YES ") {
                warn!("Found plain YES pattern at start of line");
                return (true, filtered_response.to_string());
            }
            if line == "NO"
                || line.starts_with("NO:")
                || line.starts_with("NO,")
                || line.starts_with("NO ") {
                warn!("Found plain NO pattern at start of line");
                return (false, filtered_response.to_string());
            }
        }

        // PRIORITY 6: Check for keyword patterns (negative patterns first)
        let lower = filtered_response.to_lowercase();

        // Negative keywords (check first)
        if lower.contains("should not be backported") {
            warn!("Falling back to keyword 'should not be backported'");
            return (false, filtered_response.to_string());
        }
        if lower.contains("do not recommend backporting") || lower.contains("not recommend backporting") {
            warn!("Falling back to keyword 'do not recommend backporting'");
            return (false, filtered_response.to_string());
        }

        // Positive keywords
        if lower.contains("should be backported") {
            warn!("Falling back to keyword 'should be backported'");
            return (true, filtered_response.to_string());
        }
        if lower.contains("recommend backporting") {
            warn!("Falling back to keyword 'recommend backporting'");
            return (true, filtered_response.to_string());
        }

        // Give up - treat as NO to be conservative
        error!("Could not parse LLM response - no clear YES or NO found. Defaulting to NO.");
        (false, filtered_response.to_string())
    }

    // Shared method to process a prompt with all available LLMs
    async fn process_with_llms(&mut self, prompt: &str, verbose: bool) -> Result<PredictionResult, String> {
        // Initialize LLMs if not already done
        self.initialize_llms()?;

        // Clone the LLM providers for later use
        let llm_providers: Vec<String> = if let Some(llms) = &self.llms {
            llms.keys().cloned().collect()
        } else {
            return Err("LLMs not initialized".to_string());
        };

        // Extract the LLMs into a local variable to avoid borrowing issues
        let llms = if let Some(llms) = &self.llms {
            // Create a new HashMap with cloned keys and the original Box<dyn Llm> pointers
            let mut llm_map = HashMap::new();
            for (k, v) in llms {
                llm_map.insert(k.clone(), v);
            }
            llm_map
        } else {
            return Err("LLMs not initialized".to_string());
        };

        // Get LLM predictions
        let mut results = HashMap::new();
        let mut should_backport_votes = 0;
        let mut provider_count = 0;

        for provider in &llm_providers {
            if let Some(llm) = llms.get(provider) {
                // Deref Box<dyn Llm> to get a dyn Llm, then take a reference
                let llm_ref: &dyn Llm = llm.as_ref();
                match self.invoke_llm_with_retry(llm_ref, prompt, provider, verbose).await {
                    Ok((should_backport, explanation, response)) => {
                        results.insert(provider.clone(), ProviderResult {
                            should_select: should_backport,
                            explanation,
                            error: None,
                            raw_response: Some(response),
                        });

                        should_backport_votes += i32::from(should_backport);
                        provider_count += 1;
                    },
                    Err(e) => {
                        results.insert(provider.clone(), ProviderResult {
                            should_select: false,
                            explanation: String::new(),
                            error: Some(e),
                            raw_response: None,
                        });
                    }
                }
            }
        }

        // Calculate the consensus result
        let consensus_threshold = (provider_count + 1) / 2; // This rounds up for odd numbers
        let consensus_should_select = should_backport_votes >= consensus_threshold;

        // Return a generic prediction result (will be enhanced by specific methods)
        Ok(PredictionResult {
            sha: String::new(), // Will be overridden by the calling method
            subject: String::new(), // Will be overridden by the calling method
            should_select: consensus_should_select,
            vote_ratio: format!("{should_backport_votes}/{provider_count}"),
            provider_results: results,
        })
    }

    // Process a custom prompt directly without using embeddings or similar commit search
    pub async fn process_custom_prompt(&mut self, prompt: &str, verbose: bool) -> Result<PredictionResult, String> {
        // Initialize LLMs if needed
        self.initialize_llms()?;

        if verbose {
            info!("===== CUSTOM PROMPT =====");
            // Print each line without extra indentation
            for line in prompt.lines() {
                info!("{line}");
            }
            info!("=======================");
        }

        // Process the prompt and get the prediction result
        let result = self.process_with_llms(prompt, verbose).await?;

        // Return the consolidated results with custom values for sha and subject
        Ok(PredictionResult {
            sha: "custom_prompt".to_string(),
            subject: "Custom prompt".to_string(),
            should_select: result.should_select,
            vote_ratio: result.vote_ratio,
            provider_results: result.provider_results,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_llm_response_with_think_blocks() {
        // Test that content inside <think> blocks is filtered out
        let response = r#"<think>
This is some thinking that should be ignored.
It might contain YES or NO but should not affect the result.
</think>

The commit should not be backported because it introduces new features.

**NO**"#;

        let (should_backport, explanation) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);
        assert!(!explanation.contains("This is some thinking"));
        assert!(explanation.contains("The commit should not be backported"));
    }

    #[test]
    fn test_parse_llm_response_decision_at_end() {
        // Test bold YES at the end
        let response = "This commit fixes a critical bug.\n\n**YES**";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);

        // Test bold NO at the end
        let response = "This commit adds new features.\n\n**NO**";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);

        // Test plain YES at the end
        let response = "This should be backported.\n\nYES";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);

        // Test plain NO at the end
        let response = "This should not be backported.\n\nNO";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);
    }

    #[test]
    fn test_parse_llm_response_with_checklist_items() {
        // Test that checklist items with YES/NO are skipped, and final decision is used
        let response = "Analysis:\n- **Tested:** YES\n- **Real bug:** NO\n\nConclusion: This is a feature.\n\n**NO**";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);

        let response = "Rules check:\n- **Fixes bug:** YES\n- **Security:** NO\n\nThis should be backported.\n\n**YES**";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);
    }

    #[test]
    fn test_parse_llm_response_verdict_embedded_in_last_line() {
        // Bold verdict wrapped in prose on the last line
        let response = "Analysis of the refactor.\n\nThat reinforces the earlier conclusion: **NO** for stable backport.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);

        let response = "This fixes a NULL deref.\n\nSo the conclusion is **YES** for stable backport.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);

        // Labelled verdict inside a single bold span
        let response = "Only debugfs output is affected.\n\n**Stable backport recommendation: NO** - it only affects bootlog collection.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);

        let response = "Fixes a use-after-free.\n\n**Stable backport recommendation: YES** - it fixes a real crash.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);

        // Bold verdicts earlier in the response must not hijack the last line
        let response = "Step 1: is it a bug fix? **YES**\n\nBut the fix depends on a new API, so the answer is **NO**.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);
    }

    #[test]
    fn test_parse_llm_response_verdict_stated_in_words() {
        // The whole bold span is the verdict statement, with no colon before the answer
        let response = "Analysis of the rz-du driver.\n\n**Backport recommendation remains YES** for the trees where the driver exists.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);

        let response = "Analysis of the cleanup.\n\n**The stable backport verdict is NO** - no functional change.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);

        // An ordinary bold phrase ending in yes/no is not a verdict
        let response = "**YES** This commit should be backported. **Similar commits marked NO** were wrong.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);
    }

    #[test]
    fn test_parse_llm_response_verdict_glued_to_trailing_output() {
        // The agent harness can append a background-task notice straight after the
        // verdict with no newline in between
        let response = "- [Phase 8] No failure mode identified\n\n**NO**The background search finished successfully but returned no matches.\n\nThat fits the earlier conclusion: it is an isolated rename.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);

        let response = "- [Phase 8] NULL deref on probe\n\n**YES**The background grep finished; no further callers found.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);
    }

    #[test]
    fn test_parse_llm_response_ignores_checklist_and_table_verdicts() {
        // A rules-check table answers one criterion per row - none of them is the verdict
        let response = "| Criterion | Answer |\n| 2. Fixes a real bug? | **NO** |\n| 6. Applies to stable? | **NO** for 6.6; irrelevant for 6.18 |\n\nThis fix should be backported to all affected trees.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);

        // Same for checklist bullets carrying a standalone bold answer
        let response = "- **Can apply standalone to 6.6/6.12:** **NO** - will not compile.\n\nThis commit should be backported once the prerequisite lands.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);
    }

    #[test]
    fn test_parse_llm_response_labeled_verdicts() {
        let response = "Some analysis.\n\nRecommendation: YES\n\nTrailing notes about the subsystem.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);

        let response = "Some analysis.\n\nVerdict: NO\n\nTrailing notes about the subsystem.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);

        // A label followed by a longer word is not a verdict
        let response = "Some analysis.\n\nDECISION: NOTHING TO BACKPORT\n\nThis should be backported anyway.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);
    }

    #[test]
    fn test_parse_llm_response_keyword_patterns() {
        // Test "should be backported" keyword
        let response = "This commit should be backported to stable branches.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);

        // Test "recommend backporting" keyword
        let response = "I recommend backporting this fix.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);

        // Test "should not be backported" keyword
        // After the fix, negative patterns are checked first
        let response = "This commit should not be backported due to risk.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);

        // Test "not recommend backporting" keyword
        let response = "I do not recommend backporting this change.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);
    }

    #[test]
    fn test_parse_llm_response_error_handling() {
        // Test error message handling
        let response = "Error: Failed to analyze commit";
        let (should_backport, explanation) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);
        assert!(explanation.starts_with("LLM Error:"));
    }

    #[test]
    fn test_parse_llm_response_default_fallback() {
        // Test response with no clear indicator
        let response = "This commit makes some changes to the kernel.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);

        // Test empty response
        let response = "";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);

        // Test response with only whitespace
        let response = "   \n\n   ";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);
    }

    #[test]
    fn test_parse_llm_response_mixed_case() {
        // Test case insensitive keyword matching
        let response = "This commit SHOULD BE BACKPORTED immediately.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);

        let response = "I Recommend Backporting this fix.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);
    }

    #[test]
    fn test_parse_llm_response_complex_scenarios() {
        // Test response with YES at start of line after colon
        let response = "After analysis:\nYES\nThis is a critical fix.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);

        // Test multiline response with YES on its own line
        let response = "This commit analysis:\n\nYES\n\nIt fixes a security issue.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);

        // Test response where bold pattern overrides later plain text
        let response = "**YES**\n\nAlthough there are risks, I still say no to some aspects.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);
    }

    #[test]
    fn test_parse_llm_response_edge_cases() {
        // Test response with nested think blocks
        let response = r#"<think>
Nested <think>should still work</think> properly
</think>
**YES**"#;
        let (should_backport, explanation) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);
        assert!(!explanation.contains("Nested"));

        // Test malformed bold pattern (should not match bold, and not match plain YES either)
        let response = "*YES* (not bold)\n\nSome explanation.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        // This should not match bold pattern or plain YES pattern (since *YES* is not at start)
        assert!(!should_backport);

        // Test response with YES in code block (should still match)
        let response = "```\nYES\n```\nThis is the answer.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);
    }

    #[test]
    fn test_parse_llm_response_priority_order() {
        // Test that bold patterns take priority over DECISION patterns
        let response = "DECISION: NO\n\n**YES**\n\nActually, yes it should be backported.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);

        // Test that DECISION patterns take priority over plain patterns
        let response = "NO\n\nDECISION: YES\n\nAfter further analysis.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);

        // Test that plain patterns take priority over keywords
        let response = "YES\n\nThis should not be backported though.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);

        // Test that first bold pattern wins
        let response = "**NO** This commit should **NOT** be backported. **Similar commits marked YES** are different.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);

        // Test that first bold pattern wins (YES before NO)
        let response = "**YES** This commit should be backported. **Similar commits marked NO** were wrong.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);
    }

    #[test]
    fn test_parse_llm_response_think_block_edge_cases() {
        // Test unclosed think block (should NOT be filtered since no closing tag)
        let response = "<think>\nThis should be filtered\nNO\n\nThis should also be filtered";
        let (should_backport, explanation) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport); // NO is still found
        assert!(explanation.contains("<think>")); // The whole response is kept

        // Test multiple think blocks
        let response = "<think>\nFirst thought\n</think>\n\n**NO**\n\n<think>\nSecond thought\n</think>\n\nExplanation here.";
        let (should_backport, explanation) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);
        assert!(explanation.contains("**NO**"));
        assert!(explanation.contains("Explanation here"));
        assert!(!explanation.contains("First thought"));
        assert!(!explanation.contains("Second thought"));
    }

    #[test]
    fn test_parse_llm_response_special_formats() {
        // Test with leading/trailing whitespace
        let response = "   \n\n  **YES**  \n\n   ";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);

        // Test YES followed by colon and explanation on same line
        let response = "YES: This fixes a critical security vulnerability.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);

        // Test with tabs (doesn't match plain YES pattern)
        let response = "\t\tYES\t\t";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport); // Doesn't match because YES is not at start of line

        // Test answer in parentheses
        let response = "(YES) - This should be backported";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport); // Actually matches "should be backported" keyword

        // Test YES in quotes (should not match)
        let response = "\"YES\" is what I would say if this was backportable, but it's not.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);
    }

    #[test]
    fn test_parse_llm_response_markdown_heading_patterns() {
        // Test ## YES pattern
        let response = "## YES This commit should be backported.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport);

        // Test ## NO pattern
        let response = "## NO This commit should not be backported.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);

        // Test ## NO with bold NOT in explanation
        let response = "## NO This commit should **NOT** be backported to stable kernel trees.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);

        // Test that ## patterns take priority over keyword patterns
        let response = "## NO\n\nAlthough it says should be backported here.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);
    }

    #[test]
    fn test_parse_llm_response_conflicting_signals() {
        // Test response with both positive and negative keywords (negative should win)
        let response = "I recommend backporting this, but actually I do not recommend backporting after all.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);

        // Test response with YES pattern (which takes priority over negative keywords)
        let response = "Analysis complete.\n\nThis should not be backported.\n\nYES, I'm sure about this.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport); // Plain YES pattern takes priority over keywords

        // Test bold NO overrides everything
        let response = "YES\nDECISION: YES\nI recommend backporting\n\n**NO**\n\nActually, no.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport);
    }

    #[test]
    fn test_parse_llm_response_new_format_decision_first() {
        // Test the new format where decision comes first
        let response = r#"**YES**

This commit fixes a critical memory corruption bug that affects all users of the subsystem.

## Comprehensive Analysis

### TODO Items
- [x] Check for Cc: stable tag
- [x] Assess impact and severity
"#;
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport, "Should parse YES from first line");

        let response = r#"**NO**

This commit adds a new feature rather than fixing a bug, which violates stable kernel rules.

## Detailed Analysis

The commit introduces new functionality that is not appropriate for stable trees.
"#;
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(!should_backport, "Should parse NO from first line");

        // Test with extra text before the decision (should still work)
        let response = r#"**YES** - This should be backported

Brief summary: Fixes a critical bug affecting production systems.

## Analysis Details
..."#;
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert!(should_backport, "Should handle YES with extra text on first line");
    }
}
