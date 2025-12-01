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
            "max_tokens": 4000,
            "debug_logging": false
        }));

        // NVIDIA default config
        configs.insert("nvidia".to_string(), serde_json::json!({
            "model_type": "deepseek",
            "temperature": 0.6,
            "max_tokens": 4096
        }));

        // OpenAI default config
        configs.insert("openai".to_string(), serde_json::json!({
            "model_type": "gpt4o",
            "temperature": 0.2,
            "max_tokens": 4096,
            "debug_logging": false
        }));

        // Ollama default config
        configs.insert("ollama".to_string(), serde_json::json!({
            "model_type": "llama3",
            "temperature": 0.1,
            "max_tokens": 4096,
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

                if let Some(vs) = &mut self.vectorstore {
                    if let Err(e) = vs.add_texts(batch_texts, Some(batch_metadatas)) {
                        warn!("Failed to add batch to vectorstore: {e}");
                    }
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
        if let Some(vectorstore) = &self.vectorstore {
            if let Some(persist_dir) = &self.persist_directory {
                debug!("Saving vectorstore to {}", persist_dir.display());
                if let Err(e) = vectorstore.persist(persist_dir) {
                    return Err(format!("Failed to save vectorstore: {e}"));
                }
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
2. It must fix a real bug that affects users (not theoretical races or cleanup)
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
   - Suspend/resume quirks
   - USB quirks for non-compliant devices
   - PCI quirks for misconfigured hardware
   - Example: "USB: quirks: Add USB_QUIRK_DELAY_INIT for device 0x1234:0x5678"
   - These fix real-world hardware issues even though they add code

3. DEVICE TREE (DT) UPDATES:
   - DT binding additions for existing hardware
   - DT fixes for incorrect hardware descriptions
   - Enabling already-supported hardware on new boards
   - Usually safe because they only affect specific ARM/embedded platforms

4. BUILD FIXES:
   - Fixes for compilation errors or warnings
   - Kconfig dependency fixes
   - Include file fixes
   - These are critical for users who need to build the kernel

5. KCONFIG CHANGES:
   - Enabling already-existing features that were incorrectly disabled
   - Fixing Kconfig dependencies that prevent builds
   - Not for enabling experimental features

6. TRIVIAL SUBSYSTEM-SPECIFIC ADDITIONS:
   - New trivial driver support for widely-deployed hardware (rare, case-by-case)
   - New simple device support if the alternative is broken hardware
   - Usually requires maintainer blessing

7. DOCUMENTATION and COMMENT FIXES:
   - Fixing incorrect documentation
   - Updating obsolete comments
   - Zero risk of runtime regression

8. STABLE-SPECIFIC BACKPORTS:
   - Sometimes a mainline fix requires backporting with modifications
   - May need additional context or helper patches
   - Should still follow the spirit of stable rules

WHEN EXCEPTIONS DON'T APPLY:
- Adding a new driver entirely (almost never stable material)
- Adding new userspace-visible APIs or syscalls (never stable material)
- Adding new module parameters that change behavior (risky)
- Enabling experimental or staging features (not for stable)
- New features that aren't fixing broken hardware (not for stable)

WHAT MAKES A GOOD BACKPORT CANDIDATE:

STRONG YES signals:
- Fixes crashes, panics, or oopses that users can trigger
- Fixes data corruption or filesystem issues
- Fixes security vulnerabilities (especially with CVE)
- Fixes deadlocks or livelocks
- Fixes memory leaks in commonly-used code paths
- Has "Cc: stable@vger.kernel.org" tag (maintainer explicitly wants it in stable)
- Has "Fixes:" tag pointing to a commit that's in the stable tree
- Small, surgical fix with clear scope
- Already tested in mainline for several weeks/months
- Adds device IDs to existing drivers (PCI, USB, ACPI IDs)
- Adds hardware quirks/workarounds for broken devices
- Build fixes that prevent compilation
- Device tree updates for existing hardware

STRONG NO signals:
- Adds new features that aren't fixing real hardware issues
- Adds new APIs or changes userspace-visible behavior
- Adds entirely new drivers (use existing driver + device ID instead)
- Large refactoring or code reorganization
- Changes that require other commits to work properly (dependencies)
- Optimization-only changes that don't fix user-visible bugs
- Changes to new code that doesn't exist in the stable tree
- Theoretical race conditions with no known way to trigger them
- Code cleanup without fixing actual bugs
- Changes to staging drivers (usually not stable material)
- Adds new module parameters or sysfs attributes

BORDERLINE CASES (need deeper analysis):
- Performance fixes that address user complaints
- Fixes to rarely-used code paths
- Fixes that partially solve a problem (incomplete fixes)
- Changes that are large but fix critical issues
- Fixes that require minor backport adjustments

ANALYSIS WORKFLOW:

STEP 0: COMPREHENSIVE ANALYSIS FRAMEWORK
   Your analysis MUST be thorough and systematic. Work through each of these investigation areas carefully.
   Don't skip steps - comprehensive analysis is critical for making the right backporting decision.

   IMPORTANT: Tool usage is FREE and ENCOURAGED. Whenever you need more information, use the available tools:
   git commands (git log, git blame, git show, git grep), semantic code analysis (mcp__semcode__* tools), or
   any other tools that help understand the code better. Don't hesitate - investigate deeply!

   ESSENTIAL ANALYSIS AREAS (investigate all of these thoroughly):

   1. COMMIT MESSAGE ANALYSIS
      Parse the commit subject and body carefully. Look for keywords like "fix", "crash", "deadlock",
      "corruption", "oops", "panic", "regression". Identify what problem is being solved and why it matters.
      Look for "Cc: stable@vger.kernel.org" or "Fixes:" tags - these are strong signals.
      If you see a Fixes: tag, note the commit SHA - you MUST investigate it in the next step.

   2. DEEP CODE RESEARCH (MANDATORY - DO NOT SKIP)
      This is a REQUIRED step. You must conduct detailed research into the code and issue history:

      A. UNDERSTAND HOW THE BUG WAS INTRODUCED:
         - If there's a Fixes: tag, use "git show <commit-sha>" to examine the commit that introduced the bug
         - Use "git blame" on the affected files to see when the buggy code was added
         - Use "git log" on the affected files to understand the evolution of the code
         - Determine: When was this code introduced? What was the original intent? What changed?

      B. ANALYZE THE ACTUAL CODE IN DETAIL:
         - Read the diff carefully line by line - don't just skim it
         - Understand what the buggy code does and WHY it's wrong
         - Explain the technical mechanism of the bug (e.g., "the code dereferences a pointer before checking
           for NULL because...", "the race condition occurs when thread A modifies X while thread B reads it...")
         - Identify the root cause at the code level (logic error, missing check, wrong order of operations, etc.)

      C. EXPLAIN THE FIX:
         - What does the fix change in the code?
         - Why does this change solve the problem?
         - Are there any subtle implications or edge cases?

      D. USE SEMANTIC ANALYSIS TOOLS:
         - Use mcp__semcode__find_function to locate the functions being modified
         - Use mcp__semcode__find_callers to see where these functions are called from
         - Use mcp__semcode__find_calls to see what these functions call
         - This helps you understand the context and potential impact

      E. RESEARCH THE SUBSYSTEM:
         - What subsystem is this in? (networking, filesystem, driver, mm, scheduler, etc.)
         - How mature/stable is this subsystem?
         - Check git log on the directory to see how active development is

      Your analysis MUST include:
      - A clear technical explanation of what the bug is and how it manifests
      - How the bug was introduced (cite specific commits if possible)
      - Why the fix works at the code level
      - Whether this is well-understood or complex/subtle

      DO NOT PROCEED without completing this deep research. Use git commands and semantic tools liberally.
      This understanding is essential for assessing whether the fix is "obviously correct" (a stable requirement).

   3. SECURITY ASSESSMENT
      Search for "CVE-", "security", "vulnerability", "exploit", "UAF", "use-after-free", "buffer overflow",
      "integer overflow", "privilege escalation", "DoS", "memory safety". Any security issue is a STRONG YES signal.

   4. FEATURE VS BUG FIX CLASSIFICATION
      Is this fixing a bug or adding a feature? Keywords like "add support", "implement", "introduce", "enable new"
      suggest features (usually NO). Keywords like "fix", "correct", "prevent", "avoid" suggest fixes (possibly YES).
      BUT check for EXCEPTION CATEGORIES:
      - Device ID additions (PCI/USB/ACPI IDs like "0x1234:0x5678") - USUALLY YES
      - Hardware quirks/workarounds (look for "quirk", "workaround", "broken hardware") - USUALLY YES
      - Device tree updates (.dts/.dtsi files) - USUALLY YES
      - Build fixes (Kconfig, Makefile, #include, compilation errors) - USUALLY YES
      - Documentation fixes (.rst, .txt, Documentation/) - USUALLY YES

   5. CODE CHANGE SCOPE ASSESSMENT
      Count lines changed, files touched, complexity. Small surgical fixes are safer. Large refactoring is risky.
      The diff is in the "Changes:" section - analyze it carefully.

   6. BUG TYPE AND SEVERITY (if this is a bug fix)
      What kind of bug? Crash/panic/oops? Data corruption? Deadlock? Memory leak? Logic error? Race condition?
      How severe? CRITICAL (crash, data loss, security)? HIGH (service disruption)? MEDIUM (malfunction)? LOW (cosmetic)?
      Severity affects the risk/benefit calculation.

   7. USER IMPACT EVALUATION
      Who is affected? Core kernel paths and common drivers = HIGH impact. Obscure drivers = LOW impact.
      Consider using git log to see how active the affected files are. If you're unsure about how widely
      a function is used, use mcp__semcode__find_callers to see all call sites - this is very useful!

   8. REGRESSION RISK ANALYSIS
      Could this fix break something else? Consider size, complexity, subsystems touched, testing mentioned.
      Large changes to core subsystems = HIGH risk. Small localized fixes = LOW risk.

   9. MAINLINE STABILITY
      Has this been tested in mainline? Look for "Tested-by:", "Reviewed-by:", "Reported-by:" tags.
      Older commits that have been in mainline for months are safer than brand new commits.

   10. HISTORICAL COMMIT REVIEW
       Examine the "Historical similar commits" section. Look for commits with similar characteristics - were they
       backported or not? Learn from past decisions to maintain consistency.

   CONDITIONAL DEEPER INVESTIGATION (use when needed - tools are free!):

   - DEPENDENCY CHECK: Does this depend on other commits? Look for references to other commits in the message.
     If you see function calls or APIs that look unfamiliar, use mcp__semcode__find_function to locate them,
     or git log to see when they were introduced. Dependencies might mean the fix won't apply cleanly.

   - API/ABI CHANGES: New syscalls, ioctls, /proc or /sys changes, UAPI changes, new module parameters?
     These are generally NOT allowed in stable. Check include/uapi/ changes.

   - SUBSYSTEM EXISTENCE: Does the affected code even exist in stable trees? If a driver was added in 6.8,
     a fix for it can't go to 6.6 stable. Use git log --follow --diff-filter=A to see when files were added.
     Use git blame to see when specific code was introduced.

   - COMMIT HISTORY: For deeper context, use:
     * git log --oneline path/to/file (see recent changes)
     * git blame path/to/file (see when lines were modified)
     * git show <commit-sha> (examine Fixes: target or related commits)
     * git log --grep="keyword" (find related commits)

   - CALL CHAIN ANALYSIS: To understand code path importance and impact, use semantic analysis:
     * mcp__semcode__find_function: Locate function definitions
     * mcp__semcode__find_callers: See all places that call a function (very useful for impact assessment!)
     * mcp__semcode__find_calls: See what a function calls
     * mcp__semcode__find_callchain: Trace full call chains
     This helps you understand if a rarely-used or commonly-used code path is affected.

   - PRACTICAL VS THEORETICAL: Is this a bug users actually hit? Look for "Reported-by:", "Link:" to bug reports,
     or reproducers mentioned. Practical bugs with real-world impact are higher priority.

   - COMPLETENESS: Is this a partial fix? Look for "part 1 of N", follow-up patches mentioned, or related commits.
     Use git log --grep to find related commits.

   REMEMBER:
   - Be thorough. Use tools liberally to investigate - it costs nothing and improves accuracy.
   - Present your final answer as **YES** or **NO** in bold.
   - Provide a detailed explanation covering the key factors: what's fixed, severity, user impact, regression risk,
     and how it aligns with stable kernel rules.
   - Reference specific parts of the commit message or code that support your decision.

1. READ THE COMMIT MESSAGE CAREFULLY:
   - What problem does it solve?
   - Is there a "Fixes:" tag? If so, check if that commit is in stable trees
   - Is there a "Cc: stable@vger.kernel.org" tag? (Strong signal for backporting)
   - Does it mention a CVE, security issue, or bug report?
   - Does it mention "new feature", "adds support", or "enable"? (Often disqualifying)
   - Check for exception cases: device IDs, quirks, DT updates, build fixes

2. EXAMINE THE CODE CHANGES:
   - How many lines changed? (Smaller is better)
   - How many files affected? (Fewer is better)
   - Does it add new exports, new APIs, or new functionality?
   - Is it just adding PCI/USB/ACPI IDs? (Exception: usually OK)
   - Is it adding quirks for specific hardware? (Exception: usually OK)
   - Is it a build fix or documentation fix? (Exception: usually OK)
   - Is it fixing a logic error, off-by-one, use-after-free, null deref, etc?
   - Does it require other commits to make sense?

3. ASSESS IMPACT AND RISK:
   - How commonly is the affected code used?
   - Can users trigger the bug, or is it only in rare scenarios?
   - What's the severity if the bug hits? (crash vs minor issue)
   - What's the risk of the fix causing regressions?
   - Has this been in mainline long enough to be tested?

4. USE AVAILABLE TOOLS TO INVESTIGATE:
   You have access to the Linux kernel repository in the current working directory.

   Basic tools:
   - git log, git blame: Check history and related commits
   - grep: Search for patterns in code
   - git show: Examine specific commits

   Advanced tools (if available):
   - mcp__semcode__* tools: For semantic code analysis (find callers, trace call chains, etc.)
   - kernel-code-researcher agent: To investigate design decisions and history

   Use these tools to understand the code context, but remember: they are aids to your analysis, not replacements for
   understanding stable kernel backporting principles.

Historical similar commits and their backport status for reference:
{context}

IMPORTANT: Pay close attention to the Backport Status (YES/NO) of similar commits as they provide valuable reference points.
Commits with similar characteristics to those marked with "Backport Status: YES" are more likely to be suitable for backporting.

New Commit to analyze:
{commit_info}

Based on your analysis of both the commit message and code changes, should this commit be backported to stable kernel trees?

========================================
CRITICAL RESPONSE FORMAT REQUIREMENTS:
========================================

You MUST format your response EXACTLY as shown below. This format is REQUIRED for automated parsing.

YOUR RESPONSE STRUCTURE:
1. Provide your comprehensive analysis covering all the investigation areas mentioned above
2. Work through TODO items systematically
3. Include detailed explanation covering:
   - What problem the commit solves and why it matters to stable users
   - Whether it meets stable kernel rules (obviously correct, fixes real bug, small scope, no new features)
   - Risk vs benefit trade-off (severity of bug vs risk of regression)
   - Any concerns or caveats (dependencies, need for backport adjustments, affected versions)
   - References to specific parts of the code or commit message

FINAL LINE - Your decision (choose ONE):
After all your analysis, the VERY LAST LINE of your response must be EXACTLY:
**YES**
OR
**NO**

That's it - just **YES** or **NO** on the last line by itself. Nothing else after it.

CORRECT EXAMPLES:
-----------------
Example 1:
[comprehensive analysis here...]

This commit fixes a critical use-after-free bug that causes kernel panics in production.
The fix is small, surgical, and has been tested in mainline for 3 months without issues.
It meets all stable kernel criteria.

**YES**

Example 2:
[comprehensive analysis here...]

This commit adds a new feature rather than fixing a bug. While useful, it introduces
new functionality that is not appropriate for stable trees according to the documented rules.
There is no existing broken behavior being fixed.

**NO**

INCORRECT EXAMPLES (DO NOT DO THIS):
------------------------------------
- Ending with "Based on my analysis, **YES**" 
- Ending with "**Decision: YES**"
- Putting **YES** or **NO** in the middle of analysis
- Having text after the final **YES** or **NO**

The goal is to identify all applicable fixes that stable users need, while avoiding regressions and unsuitable changes.

REMEMBER: Last line MUST be exactly **YES** or **NO** - nothing else!"#;

        // Format the context from similar commits
        let mut context_parts = Vec::new();

        for (i, (content, was_backported)) in similar_commits.iter().enumerate() {
            if content.is_empty() {
                continue;  // Skip empty content
            }

            // Extract backport status
            let backport_status = if *was_backported { "YES" } else { "NO" };

            // Add a clear separator with the similarity ranking and status
            context_parts.push(format!("--- Similar Commit #{} (Status: {}) ---\n{}", i+1, backport_status, content));
        }

        let context_text = context_parts.join("\n\n");

        // Replace placeholders in the template
        if context_text.is_empty() {
            // When no similar commits, remove the context section from the prompt
            let prompt_without_context = prompt_template
                .lines()
                .filter(|line| !line.contains("Historical similar commits") && 
                              !line.contains("{context}") &&
                              !line.contains("IMPORTANT: Pay close attention to the Backport Status"))
                .collect::<Vec<&str>>()
                .join("\n");
            
            prompt_without_context.replace("{commit_info}", commit_text)
        } else {
            prompt_template
                .replace("{context}", &context_text)
                .replace("{commit_info}", commit_text)
        }
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
                let config = self.llm_configs.get(provider).cloned();
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

    // Parse LLM response to extract decision and explanation
    // The prompt requires the response to END with **YES** or **NO** on the last line
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
        let last_line = trimmed.lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("")
            .trim();

        // The prompt explicitly requires **YES** or **NO** on the last line
        // Check for exact format first
        if last_line == "**YES**" {
            return (true, filtered_response.to_string());
        }
        if last_line == "**NO**" {
            return (false, filtered_response.to_string());
        }
        
        // Fallback: Maybe the LLM used plain text without formatting
        if last_line == "YES" {
            return (true, filtered_response.to_string());
        }
        if last_line == "NO" {
            return (false, filtered_response.to_string());
        }

        // If we get here, the LLM didn't follow the format correctly
        // Log a warning and try to find YES/NO anywhere in the response
        warn!("LLM response did not end with **YES** or **NO** as required. Last line: '{}'", last_line);
        
        // Scan backwards from the end looking for **YES** or **NO**
        // Skip lines that look like checklist items (contain ":**")
        for line in trimmed.lines().rev() {
            let line_trimmed = line.trim();
            
            // Skip empty lines
            if line_trimmed.is_empty() {
                continue;
            }
            
            // Skip checklist items
            if line_trimmed.contains(":**") {
                continue;
            }
            
            if line_trimmed.contains("**YES**") {
                warn!("Found **YES** in response but not on last line");
                return (true, filtered_response.to_string());
            }
            if line_trimmed.contains("**NO**") {
                warn!("Found **NO** in response but not on last line");
                return (false, filtered_response.to_string());
            }
        }

        // Absolute last resort: check for keywords
        if filtered_response.to_lowercase().contains("should not be backported") {
            warn!("Falling back to keyword 'should not be backported'");
            return (false, filtered_response.to_string());
        }
        if filtered_response.to_lowercase().contains("should be backported") {
            warn!("Falling back to keyword 'should be backported'");
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
        assert_eq!(should_backport, false);
        assert!(!explanation.contains("This is some thinking"));
        assert!(explanation.contains("The commit should not be backported"));
    }

    #[test]
    fn test_parse_llm_response_decision_at_end() {
        // Test bold YES at the end
        let response = "This commit fixes a critical bug.\n\n**YES**";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);

        // Test bold NO at the end
        let response = "This commit adds new features.\n\n**NO**";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);

        // Test plain YES at the end
        let response = "This should be backported.\n\nYES";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);

        // Test plain NO at the end
        let response = "This should not be backported.\n\nNO";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);
    }

    #[test]
    fn test_parse_llm_response_with_checklist_items() {
        // Test that checklist items with YES/NO are skipped, and final decision is used
        let response = "Analysis:\n- **Tested:** YES\n- **Real bug:** NO\n\nConclusion: This is a feature.\n\n**NO**";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);

        let response = "Rules check:\n- **Fixes bug:** YES\n- **Security:** NO\n\nThis should be backported.\n\n**YES**";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);
    }

    #[test]
    fn test_parse_llm_response_keyword_patterns() {
        // Test "should be backported" keyword
        let response = "This commit should be backported to stable branches.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);

        // Test "recommend backporting" keyword
        let response = "I recommend backporting this fix.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);

        // Test "should not be backported" keyword
        // After the fix, negative patterns are checked first
        let response = "This commit should not be backported due to risk.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);

        // Test "not recommend backporting" keyword
        let response = "I do not recommend backporting this change.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);
    }

    #[test]
    fn test_parse_llm_response_error_handling() {
        // Test error message handling
        let response = "Error: Failed to analyze commit";
        let (should_backport, explanation) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);
        assert!(explanation.starts_with("LLM Error:"));
    }

    #[test]
    fn test_parse_llm_response_default_fallback() {
        // Test response with no clear indicator
        let response = "This commit makes some changes to the kernel.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);

        // Test empty response
        let response = "";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);

        // Test response with only whitespace
        let response = "   \n\n   ";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);
    }

    #[test]
    fn test_parse_llm_response_mixed_case() {
        // Test case insensitive keyword matching
        let response = "This commit SHOULD BE BACKPORTED immediately.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);

        let response = "I Recommend Backporting this fix.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);
    }

    #[test]
    fn test_parse_llm_response_complex_scenarios() {
        // Test response with YES at start of line after colon
        let response = "After analysis:\nYES\nThis is a critical fix.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);

        // Test multiline response with YES on its own line
        let response = "This commit analysis:\n\nYES\n\nIt fixes a security issue.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);

        // Test response where bold pattern overrides later plain text
        let response = "**YES**\n\nAlthough there are risks, I still say no to some aspects.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);
    }

    #[test]
    fn test_parse_llm_response_edge_cases() {
        // Test response with nested think blocks
        let response = r#"<think>
Nested <think>should still work</think> properly
</think>
**YES**"#;
        let (should_backport, explanation) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);
        assert!(!explanation.contains("Nested"));

        // Test malformed bold pattern (should not match bold, and not match plain YES either)
        let response = "*YES* (not bold)\n\nSome explanation.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        // This should not match bold pattern or plain YES pattern (since *YES* is not at start)
        assert_eq!(should_backport, false);

        // Test response with YES in code block (should still match)
        let response = "```\nYES\n```\nThis is the answer.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);
    }

    #[test]
    fn test_parse_llm_response_priority_order() {
        // Test that bold patterns take priority over DECISION patterns
        let response = "DECISION: NO\n\n**YES**\n\nActually, yes it should be backported.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);

        // Test that DECISION patterns take priority over plain patterns
        let response = "NO\n\nDECISION: YES\n\nAfter further analysis.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);

        // Test that plain patterns take priority over keywords
        let response = "YES\n\nThis should not be backported though.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);

        // Test that first bold pattern wins
        let response = "**NO** This commit should **NOT** be backported. **Similar commits marked YES** are different.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);

        // Test that first bold pattern wins (YES before NO)
        let response = "**YES** This commit should be backported. **Similar commits marked NO** were wrong.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);
    }

    #[test]
    fn test_parse_llm_response_think_block_edge_cases() {
        // Test unclosed think block (should NOT be filtered since no closing tag)
        let response = "<think>\nThis should be filtered\nNO\n\nThis should also be filtered";
        let (should_backport, explanation) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false); // NO is still found
        assert!(explanation.contains("<think>")); // The whole response is kept

        // Test multiple think blocks
        let response = "<think>\nFirst thought\n</think>\n\n**NO**\n\n<think>\nSecond thought\n</think>\n\nExplanation here.";
        let (should_backport, explanation) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);
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
        assert_eq!(should_backport, true);

        // Test YES followed by colon and explanation on same line
        let response = "YES: This fixes a critical security vulnerability.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);

        // Test with tabs (doesn't match plain YES pattern)
        let response = "\t\tYES\t\t";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false); // Doesn't match because YES is not at start of line

        // Test answer in parentheses
        let response = "(YES) - This should be backported";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true); // Actually matches "should be backported" keyword

        // Test YES in quotes (should not match)
        let response = "\"YES\" is what I would say if this was backportable, but it's not.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);
    }

    #[test]
    fn test_parse_llm_response_markdown_heading_patterns() {
        // Test ## YES pattern
        let response = "## YES This commit should be backported.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);

        // Test ## NO pattern
        let response = "## NO This commit should not be backported.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);

        // Test ## NO with bold NOT in explanation
        let response = "## NO This commit should **NOT** be backported to stable kernel trees.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);

        // Test that ## patterns take priority over keyword patterns
        let response = "## NO\n\nAlthough it says should be backported here.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);
    }

    #[test]
    fn test_parse_llm_response_conflicting_signals() {
        // Test response with both positive and negative keywords (negative should win)
        let response = "I recommend backporting this, but actually I do not recommend backporting after all.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);

        // Test response with YES pattern (which takes priority over negative keywords)
        let response = "Analysis complete.\n\nThis should not be backported.\n\nYES, I'm sure about this.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true); // Plain YES pattern takes priority over keywords

        // Test bold NO overrides everything
        let response = "YES\nDECISION: YES\nI recommend backporting\n\n**NO**\n\nActually, no.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);
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
        assert_eq!(should_backport, true, "Should parse YES from first line");

        let response = r#"**NO**

This commit adds a new feature rather than fixing a bug, which violates stable kernel rules.

## Detailed Analysis

The commit introduces new functionality that is not appropriate for stable trees.
"#;
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false, "Should parse NO from first line");

        // Test with extra text before the decision (should still work)
        let response = r#"**YES** - This should be backported

Brief summary: Fixes a critical bug affecting production systems.

## Analysis Details
..."#;
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true, "Should handle YES with extra text on first line");
    }
}
