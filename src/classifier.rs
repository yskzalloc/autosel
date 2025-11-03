// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use log::{info, debug, warn, error};
use regex::Regex;
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
Analyze both the commit message and the code changes carefully to identify if this commit is suitable for backporting.

IMPORTANT: You have access to the Linux kernel repository in the current working directory. You MUST use available tools to examine this repository to better understand the context of the commit.

AVAILABLE SEMANTIC CODE ANALYSIS TOOLS (when semcode MCP is available):
- mcp__semcode__find_function: Locate function definitions by name/pattern
- mcp__semcode__find_type: Find struct/enum/union/typedef definitions
- mcp__semcode__find_callers: Identify all functions that call a target function (critical for impact analysis)
- mcp__semcode__find_calls: Show all functions called by a specific function (dependency analysis)
- mcp__semcode__find_callchain: Trace complete call chains between functions (for understanding complex interactions)
- mcp__semcode__diff_functions: Semantic comparison between function versions (behavioral vs cosmetic changes)
- mcp__semcode__grep_functions: Search within function bodies with semantic understanding
- mcp__semcode__vgrep_functions: Variable-aware search that tracks data flow

RECOMMENDED ANALYSIS WORKFLOW:
1. Use mcp__semcode__find_function to locate changed functions in the kernel tree
2. Use mcp__semcode__diff_functions to understand semantic changes (behavioral vs refactoring)
3. Use mcp__semcode__find_callers to assess impact scope and determine if user-space can trigger the code
4. Use mcp__semcode__find_calls to identify dependencies that must exist in target kernel version
5. For security fixes: Use mcp__semcode__vgrep_functions to trace data flow from user input
6. Use mcp__semcode__find_callchain to verify exploit paths or understand complex interactions

AVAILABLE SPECIALIZED AGENTS (when applicable):
- kernel-code-researcher: For investigating the history, rationale, and design decisions behind kernel code
- Explore: For thorough codebase exploration when you need to understand broader context
- debugger: For complex issue diagnosis and root cause analysis

BASIC TOOLS (also available):
- grep, git log, git blame for text-based searches
- Any standard shell commands for repository examination

ANALYSIS PRIORITY: Prefer semantic code analysis tools (mcp__semcode__*) over basic text tools when available, as they provide superior accuracy by understanding code structure, call graphs, data flow, and semantic changes rather than just text matching.

ANALYSIS GUIDELINES - Use semantic tools to answer these questions:

1. IMPACT ANALYSIS: Does the commit fix a bug that affects users?
   → Use mcp__semcode__find_callers on affected functions to determine exposure
   → Use mcp__semcode__find_callchain to check if user-space can trigger the code path
   → Higher caller count + user-space reachability = higher priority for backport

2. SCOPE ANALYSIS: Is the fix relatively small and contained?
   → Use mcp__semcode__diff_functions to measure semantic change size
   → Use mcp__semcode__find_calls to check dependency complexity
   → Fewer changed functions + fewer new dependencies = better backport candidate

3. SIDE EFFECTS: Does the commit have clear side effects beyond fixing the issue?
   → Use mcp__semcode__find_callers to identify all affected call sites
   → Use mcp__semcode__diff_functions to distinguish behavioral vs cosmetic changes
   → Check if change affects function contracts or invariants

4. ARCHITECTURAL IMPACT: Does the commit include major architectural changes?
   → Use mcp__semcode__find_type to check if core data structures changed
   → Use mcp__semcode__find_callers on modified types to assess refactoring scope
   → Large structural changes typically should NOT be backported

5. SUBSYSTEM CRITICALITY: Does the commit touch critical kernel subsystems?
   → Use kernel-code-researcher agent to understand subsystem importance
   → Critical subsystems (mm, scheduler, locking) require extra scrutiny
   → Use debugger agent for complex critical path analysis

6. BACKPORT INDICATORS: Is there explicit mention of stable tree backporting?
   → Check commit message for "Cc: stable@vger.kernel.org" or "Fixes:" tags
   → Use kernel-code-researcher to investigate linked CVEs or bug reports
   → Explicit stable tags are strong signals for backporting

7. STABLE TREE COMPLIANCE: Does the change follow stable tree rules?
   → Bug fixes: YES → good candidate
   → New features: NO → poor candidate
   → Security fixes: YES → high priority candidate
   → Performance optimizations: MAYBE → evaluate risk vs benefit
   → Use mcp__semcode__diff_functions to categorize change type

Remember that good backport candidates typically:
- Fix important bugs (verify with mcp__semcode__find_callers for impact)
- Don't introduce new features (verify with mcp__semcode__diff_functions)
- Don't make architectural changes (verify with mcp__semcode__find_type)
- Have minimal risk of regression (verify with mcp__semcode__find_calls for dependencies)
- Are confined to a subsystem (verify scope with semantic tools)

CRITICAL: You MUST actually use the semantic code analysis tools to examine the kernel repository. Simply reading the commit diff is insufficient - you need to understand the broader code context, call graphs, and impact scope to make an informed decision.

Historical similar commits and their backport status for reference:
{context}

IMPORTANT: Pay close attention to the Backport Status (YES/NO) of similar commits as they provide valuable reference points.
Commits with similar characteristics to those marked with "Backport Status: YES" are more likely to be suitable for backporting.

EXAMPLE ANALYSIS SCENARIOS:

Scenario A - Null Pointer Dereference Fix:
1. Identify changed function: mcp__semcode__find_function <function_name>
2. Check impact: mcp__semcode__find_callers <function_name>
3. Verify user exposure: mcp__semcode__find_callchain <syscall_entry> <function_name>
4. If user-space can reach it → HIGH PRIORITY backport

Scenario B - Lock Ordering Fix:
1. Find lock usage: mcp__semcode__vgrep_functions "mutex_lock.*lock_name"
2. Check deadlock potential: mcp__semcode__find_callchain <lock_holder_a> <lock_holder_b>
3. Use debugger agent for complex lock dependency analysis
4. If deadlock exists → CRITICAL backport

Scenario C - Performance Optimization:
1. Semantic comparison: mcp__semcode__diff_functions <function> <old_commit> <new_commit>
2. Check if behavior changed or just optimized
3. Pure optimization without behavior change → LOWER PRIORITY
4. Optimization that fixes performance bug → HIGHER PRIORITY

Scenario D - API Addition:
1. Find new function: mcp__semcode__find_function <new_function>
2. Check usage: mcp__semcode__find_callers <new_function>
3. If no callers exist → DON'T backport (unused new feature)
4. If callers exist → evaluate if callers should also be backported

Scenario E - Security Vulnerability:
1. Use kernel-code-researcher to investigate CVE context and history
2. Trace data flow: mcp__semcode__vgrep_functions <vulnerable_variable>
3. Find exploit path: mcp__semcode__find_callchain <user_input> <vulnerable_code>
4. If exploit path exists → URGENT backport

New Commit to analyze:
{commit_info}

Based on your analysis of both the commit message AND code changes (using the semantic tools and agents described above to examine the kernel repository), should this commit be backported to stable kernel trees?

Provide your answer as YES or NO, followed by an extensive explanation that:
1. References specific parts of the code changes
2. Describes which semantic analysis tools you used (if any)
3. Summarizes findings from tool usage (call graphs, impact scope, dependency analysis, etc.)
4. Explains your reasoning based on the concrete data gathered from tools"#;

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

        // First priority: Check for markdown bold patterns at the start
        // These take precedence over plain text patterns
        // Check for patterns like **YES**, **Answer: YES**, **Answer: NO**, etc.
        if let Some(_caps) = Regex::new(r"^\*\*[^*]*\bYES\b[^*]*\*\*").unwrap().captures(trimmed) {
            return (true, filtered_response.to_string());
        }
        if let Some(_caps) = Regex::new(r"^\*\*[^*]*\bNO\b[^*]*\*\*").unwrap().captures(trimmed) {
            return (false, filtered_response.to_string());
        }

        // Check for bold patterns anywhere in the response using regex
        // Find the FIRST occurrence of either **YES** or **NO** pattern
        let bold_yes_pattern = Regex::new(r"\*\*[^*]*\bYES\b[^*]*\*\*").unwrap();
        let bold_no_pattern = Regex::new(r"\*\*[^*]*\bNO\b[^*]*\*\*").unwrap();

        // Find the position of first YES and NO patterns
        let yes_pos = bold_yes_pattern.find(&filtered_response).map(|m| m.start());
        let no_pos = bold_no_pattern.find(&filtered_response).map(|m| m.start());

        // Use whichever pattern appears first
        match (yes_pos, no_pos) {
            (Some(y), Some(n)) => {
                if y < n {
                    return (true, filtered_response.to_string());
                } else {
                    return (false, filtered_response.to_string());
                }
            },
            (Some(_), None) => return (true, filtered_response.to_string()),
            (None, Some(_)) => return (false, filtered_response.to_string()),
            (None, None) => {} // Continue to next pattern check
        }

        // Second priority: Look for "DECISION: YES" or "DECISION: NO" as reliable indicators
        if filtered_response.contains("DECISION: YES") {
            return (true, filtered_response.to_string());
        } else if filtered_response.contains("DECISION: NO") {
            return (false, filtered_response.to_string());
        }

        // Third priority: Check for markdown heading patterns ## YES or ## NO
        if trimmed.starts_with("## YES") {
            return (true, filtered_response.to_string());
        } else if trimmed.starts_with("## NO") {
            return (false, filtered_response.to_string());
        }

        // Fourth priority: Check for plain NO: pattern at the beginning
        if trimmed.starts_with("NO:") {
            return (false, filtered_response.to_string());
        }

        // Fifth priority: Look for plain YES or NO patterns
        // Use multiline mode (?m) and account for whitespace and colons
        let yes_pattern = Regex::new(r"(?m)^YES|^YES[,.:;]|\nYES\n|\nYES[,.:;]|^[ \t]*YES:|^[ \t]*YES[ \t]*:").unwrap();
        let no_pattern = Regex::new(r"(?m)^NO|^NO[,.:;]|\nNO\n|\nNO[,.:;]|^[ \t]*NO:|^[ \t]*NO[ \t]*:").unwrap();

        if yes_pattern.is_match(&filtered_response) {
            return (true, filtered_response.to_string());
        } else if no_pattern.is_match(&filtered_response) {
            return (false, filtered_response.to_string());
        }

        // If no clear YES/NO, look for keywords
        // Check negative patterns first since they are more specific
        let should_not_backport = filtered_response.to_lowercase().contains("should not be backported") ||
                                filtered_response.to_lowercase().contains("not recommend backporting");

        if should_not_backport {
            return (false, filtered_response.to_string());
        }

        // Then check positive patterns
        let should_backport = filtered_response.to_lowercase().contains("should be backported") ||
                            filtered_response.to_lowercase().contains("recommend backporting");

        if should_backport {
            return (true, filtered_response.to_string());
        }

        // Default fallback - consider it not backportable unless clearly indicated
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

**NO**

The commit should not be backported because it introduces new features."#;

        let (should_backport, explanation) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);
        assert!(!explanation.contains("This is some thinking"));
        assert!(explanation.contains("The commit should not be backported"));
    }

    #[test]
    fn test_parse_llm_response_bold_patterns_at_start() {
        // Test bold YES at the start
        let response = "**YES**\n\nThis commit fixes a critical bug.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);

        // Test bold NO at the start
        let response = "**NO**\n\nThis commit adds new features.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);

        // Test bold Answer: YES pattern
        let response = "**Answer: YES**\n\nThis should be backported.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);

        // Test bold Answer: NO pattern
        let response = "**Answer: NO**\n\nThis should not be backported.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);
    }

    #[test]
    fn test_parse_llm_response_bold_patterns_priority() {
        // Test that bold YES takes priority over plain NO
        let response = "Some analysis here.\n\n**YES**\n\nBut also mentions NO later.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);

        // Test that bold NO takes priority over plain YES
        let response = "YES might be mentioned here.\n\n**NO**\n\nBut the final answer is no.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);
    }

    #[test]
    fn test_parse_llm_response_decision_pattern() {
        // Test DECISION: YES pattern
        let response = "After careful analysis...\n\nDECISION: YES\n\nThis should be backported.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);

        // Test DECISION: NO pattern
        let response = "After careful analysis...\n\nDECISION: NO\n\nThis should not be backported.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);
    }

    #[test]
    fn test_parse_llm_response_plain_patterns() {
        // Test plain YES at start
        let response = "YES\n\nThis commit fixes a security issue.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);

        // Test plain NO at start
        let response = "NO\n\nThis commit is too risky.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);

        // Test NO: pattern at beginning
        let response = "NO: This commit introduces new functionality.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);

        // Test YES: pattern with spaces
        let response = "  YES: This is a critical fix.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);

        // Test YES with punctuation
        let response = "YES, this should be backported.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, true);

        // Test NO with punctuation
        let response = "NO, this is not suitable for stable.";
        let (should_backport, _) = BackportClassifier::parse_llm_response(response);
        assert_eq!(should_backport, false);
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
}
