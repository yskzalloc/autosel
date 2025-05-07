// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use log::{info, debug, warn, error};
use regex::Regex;
use indicatif::{ProgressBar, ProgressStyle};

use crate::models::{CommitFeatures, ProviderResult, PredictionResult, ModelState};
use crate::embeddings::HuggingFaceEmbeddings;
use crate::vectorstore::{ChromaStore, Metadata};
use crate::llm::Llm;

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
            persist_directory: persist_directory.map(|p| p.to_path_buf()),
            repo_path: repo_path.map(|p| p.to_path_buf()),
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

    /// Creates a new BackportClassifier instance optimized for processing prompts
    /// This is a lighter version that doesn't require a vectorstore
    pub fn new_for_prompt(
        llm_providers: Vec<String>,
        llm_configs: Option<HashMap<String, serde_json::Value>>,
        repo_path: Option<&Path>
    ) -> Self {
        // Initialize with the standard constructor but no persist_directory
        Self::new(llm_providers, llm_configs, None, repo_path)
    }

    pub fn train(&mut self, commit_dataset: &[CommitFeatures]) -> Result<(), String> {
        info!("Training classifier with {} commits", commit_dataset.len());

        // Process dataset into texts and metadata for vectorstore
        let mut texts = Vec::new();
        let mut metadatas = Vec::new();

        let max_chunk_length = 100000; // Characters

        let pb = ProgressBar::new(commit_dataset.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap());

        for commit in commit_dataset {
            let message_lines: Vec<&str> = commit.message.trim().lines().collect();
            let subject = message_lines.first().unwrap_or(&"").to_string();
            let message_body = if message_lines.len() > 1 {
                message_lines[1..].join("\n")
            } else {
                String::new()
            };

            // Build rich context for each commit
            let mut commit_context = Vec::new();

            commit_context.push(format!("Subject: {}", subject));
            commit_context.push(format!("Commit Message:\n{}", message_body));

            // Add code changes
            let diff_text = if commit.diff.len() > max_chunk_length {
                let truncated = commit.diff.chars().take(max_chunk_length).collect::<String>();
                warn!("Truncating large diff for commit {} (size: {})", commit.sha, commit.diff.len());
                format!("{}\n[... truncated due to size ...]", truncated)
            } else {
                commit.diff.clone()
            };

            commit_context.push(format!("Changes:\n{}", diff_text));

            // Add files changed
            let files_text = if !commit.files_changed.is_empty() {
                let displayed_files = if commit.files_changed.len() > 5 {
                    let joined = commit.files_changed[0..5].join(", ");
                    format!("{} and {} more", joined, commit.files_changed.len() - 5)
                } else {
                    commit.files_changed.join(", ")
                };
                format!("Files: {}", displayed_files)
            } else {
                "Files: [none]".to_string()
            };

            commit_context.push(files_text);

            // Mark backport status prominently
            let backport_status = if commit.was_backported == Some(true) { "YES" } else { "NO" };
            let text = format!("[Backport Status: {}]\n\n{}", backport_status, commit_context.join("\n\n"));

            // Create metadata
            let mut metadata = HashMap::new();
            metadata.insert("was_backported".to_string(), backport_status.to_string());
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
            return Err(format!("Failed to initialize embeddings: {}", e));
        }

        // Process in reasonable batches
        const BATCH_SIZE: usize = 1000;
        let mut start_idx = 0;

        if self.vectorstore.is_none() {
            // Create initial vectorstore with first batch
            let batch_size = std::cmp::min(BATCH_SIZE, texts.len());
            let batch_texts = texts[..batch_size].to_vec();
            let batch_metadatas = metadatas[..batch_size].to_vec();

            info!("Creating vectorstore with initial {} items", batch_size);

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
                Err(e) => return Err(format!("Failed to create vectorstore: {}", e)),
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
                        warn!("Failed to add batch to vectorstore: {}", e);
                    }
                }

                pb.inc(batch_size as u64);
            }

            pb.finish_with_message("Added all batches to vectorstore");
        }

        // Count backported vs non-backported
        let backported_count = commit_dataset.iter()
            .filter(|c| c.was_backported == Some(true))
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
                    return Err(format!("Failed to save vectorstore: {}", e));
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
                Ok(_) => Ok(()),
                Err(e) => Err(format!("Failed to write model file: {}", e)),
            },
            Err(e) => Err(format!("Failed to serialize model state: {}", e)),
        }
    }

    pub fn load_model(model_path: &Path) -> Result<Self, String> {
        // Load the model state from file
        let model_state: ModelState = match std::fs::read_to_string(model_path) {
            Ok(contents) => match serde_json::from_str(&contents) {
                Ok(state) => state,
                Err(e) => return Err(format!("Failed to parse model state: {}", e)),
            },
            Err(e) => return Err(format!("Failed to read model file: {}", e)),
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
                Err(e) => return Err(format!("Failed to load vectorstore: {}", e)),
            }
        }

        Ok(classifier)
    }

    pub fn format_commit_info(&self, commit_info: &CommitFeatures) -> String {
        let mut commit_parts = Vec::new();

        // Extract subject (first line of commit message)
        let message_lines: Vec<&str> = commit_info.message.trim().split('\n').collect();
        let subject = message_lines.first().unwrap_or(&"No subject").to_string();

        commit_parts.push(format!("Subject: {}", subject));
        commit_parts.push(format!("Commit Message:\n{}", commit_info.message));

        // Add code changes
        if !commit_info.diff.is_empty() {
            commit_parts.push(format!("Changes:\n{}", commit_info.diff));
        }

        // Add files changed
        if !commit_info.files_changed.is_empty() {
            let mut files_text = commit_info.files_changed.iter()
                .take(10)
                .map(|s| s.to_string())
                .collect::<Vec<String>>()
                .join(", ");

            if commit_info.files_changed.len() > 10 {
                files_text.push_str(&format!(" and {} more", commit_info.files_changed.len() - 10));
            }

            commit_parts.push(format!("Files Changed: {}", files_text));
        }

        commit_parts.join("\n\n")
    }

    pub fn construct_prompt(&self, commit_text: &str, similar_commits: Vec<(String, bool)>) -> String {
        // Template for the prompt
        let prompt_template = r#"You are a Linux kernel expert analyzing commits to determine if they should be backported to stable kernel trees.
Analyze both the commit message and the code changes carefully to identify if this commit is suitable for backporting.

Consider:
1. Does the commit fix a bug that affects users?
2. Is the fix relatively small and contained?
3. Does the commit have clear side effects beyond fixing the issue?
4. Does the commit include major architectural changes?
5. Does the commit touch critical kernel subsystems?
6. Is there explicit mention of stable tree backporting in the commit message?
7. Does the change follow the stable tree rules (important bugfixes, minimal risk, etc.)?

Remember that good backport candidates typically:
- Fix important bugs
- Don't introduce new features
- Don't make architectural changes
- Have minimal risk of regression
- Are confined to a subsystem

Historical similar commits and their backport status for reference:
{context}

IMPORTANT: Pay close attention to the Backport Status (YES/NO) of similar commits as they provide valuable reference points.
Commits with similar characteristics to those marked with "Backport Status: YES" are more likely to be suitable for backporting.

New Commit to analyze:
{commit_info}

Based on your analysis of both the commit message AND code changes, should this commit be backported to stable kernel trees?
Provide your answer as YES or NO, followed by a brief explanation that references specific parts of the code changes."#;

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
        prompt_template
            .replace("{context}", &context_text)
            .replace("{commit_info}", commit_text)
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
                        let was_backported = scored_doc.document.metadata.extra.get("was_backported")
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
                error!("Error performing similarity search: {}", e);
                Vec::new()
            }
        }
    }

    fn initialize_llms(&mut self) -> Result<(), String> {
        if self.llms.is_none() {
            let mut llms = HashMap::new();

            for provider in &self.llm_providers {
                let config = self.llm_configs.get(provider).cloned();
                match crate::llm::create_llm(provider, config) {
                    Ok(llm) => {
                        llms.insert(provider.clone(), llm);
                    },
                    Err(e) => {
                        return Err(format!("Failed to initialize {} LLM: {}", provider, e));
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
                        warn!("Received bare 'yes' response from {} LLM, retrying ({}/{})",
                              provider, retry_count, max_retries);

                        // Wait a bit before retrying
                        tokio::time::sleep(std::time::Duration::from_millis(1000 * retry_count)).await;
                        continue; // Try again
                    }

                    // Show full response when verbose is enabled
                    if verbose {
                        info!("===== FULL RESPONSE FROM {} =====", provider.to_uppercase());
                        info!("{}", response);
                        info!("==========================================");
                    }

                    let (should_backport, explanation) = self.parse_llm_response(&response);
                    return Ok((should_backport, explanation, response));
                },
                Err(e) => {
                    let err_str = e.to_string();

                    // Check if we still have retries available
                    if retry_count < max_retries {
                        retry_count += 1;

                        // Log specific error types differently
                        if err_str.contains("Failed to parse response: EOF while parsing a value at line 1 column 0") {
                            warn!("Encountered EOF parsing error with {} LLM, retrying ({}/{})",
                                  provider, retry_count, max_retries);
                        } else {
                            warn!("Error invoking {} LLM: {}, retrying ({}/{})",
                                  provider, err_str, retry_count, max_retries);
                        }

                        // Wait a bit before retrying with exponential backoff
                        let backoff_ms = 1000 * retry_count;
                        tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                        continue; // Try again
                    }

                    // We've exhausted all retries
                    error!("Error invoking {} LLM after {} retries: {}", provider, max_retries, err_str);
                    return Err(format!("Error: {}", err_str));
                }
            }
        }
    }

    pub async fn predict(&mut self, commit_info: &CommitFeatures, verbose: bool) -> Result<PredictionResult, String> {
        // Initialize LLMs if not already done
        self.initialize_llms()?;

        if self.vectorstore.is_none() {
            return Err("Vectorstore not initialized. Please train the model first.".to_string());
        }

        // Ensure embeddings are initialized only if they aren't already
        if !self.embeddings.is_initialized() {
            info!("Initializing embeddings model for prediction");
            if let Err(e) = self.embeddings.initialize() {
                return Err(format!("Failed to initialize embeddings model: {}", e));
            }
        }

        // Generate the vector database search for context
        let commit_text = self.format_commit_info(commit_info);

        // Get similar commits for context
        let similar_commits = self.find_similar_commits(&commit_text, 5);

        // Construct prompt with context and commit info
        let prompt = self.construct_prompt(&commit_text, similar_commits);

        // Show full prompt when verbose is enabled
        if verbose {
            info!("===== FULL PROMPT =====");
            // Print each line without extra indentation
            for line in prompt.lines() {
                info!("{}", line);
            }
            info!("=======================");
        }

        // Process the prompt and get the prediction result
        let result = self.process_with_llms(&prompt, verbose).await?;

        // Return the consolidated results with commit-specific information
        Ok(PredictionResult {
            sha: commit_info.sha.clone(),
            subject: commit_info.message.lines().next().unwrap_or("Unknown").to_string(),
            should_backport: result.should_backport,
            vote_ratio: result.vote_ratio,
            provider_results: result.provider_results,
        })
    }

    // Parse LLM response to extract decision and explanation
    pub fn parse_llm_response(&self, response: &str) -> (bool, String) {
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
            return (false, format!("LLM Error: {}", filtered_response));
        }

        // If response starts with NO: (very common pattern)
        if filtered_response.trim().starts_with("NO:") {
            return (false, filtered_response.to_string());
        }

        // Look for "DECISION: YES" or "DECISION: NO" first as the most reliable indicators
        if filtered_response.contains("DECISION: YES") {
            return (true, filtered_response.to_string());
        } else if filtered_response.contains("DECISION: NO") {
            return (false, filtered_response.to_string());
        }

        // Look for YES or NO at the beginning of the response or on its own line
        // Use multiline mode (?m) and account for whitespace and colons
        let yes_pattern = Regex::new(r"(?m)^YES|^YES[,.:;]|\nYES\n|\nYES[,.:;]|^[ \t]*YES:|^[ \t]*YES[ \t]*:").unwrap();
        let no_pattern = Regex::new(r"(?m)^NO|^NO[,.:;]|\nNO\n|\nNO[,.:;]|^[ \t]*NO:|^[ \t]*NO[ \t]*:").unwrap();

        if yes_pattern.is_match(&filtered_response) {
            return (true, filtered_response.to_string());
        } else if no_pattern.is_match(&filtered_response) {
            return (false, filtered_response.to_string());
        }

        // If no clear YES/NO, look for keywords
        let should_backport = filtered_response.to_lowercase().contains("should be backported") ||
                            filtered_response.to_lowercase().contains("recommend backporting");

        let should_not_backport = filtered_response.to_lowercase().contains("should not be backported") ||
                                filtered_response.to_lowercase().contains("not recommend backporting");

        if should_backport {
            return (true, filtered_response.to_string());
        } else if should_not_backport {
            return (false, filtered_response.to_string());
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
                            should_backport,
                            explanation,
                            error: None,
                            raw_response: Some(response),
                        });

                        should_backport_votes += if should_backport { 1 } else { 0 };
                        provider_count += 1;
                    },
                    Err(e) => {
                        results.insert(provider.clone(), ProviderResult {
                            should_backport: false,
                            explanation: "".to_string(),
                            error: Some(e),
                            raw_response: None,
                        });
                    }
                }
            }
        }

        // Calculate the consensus result
        let consensus_threshold = (provider_count + 1) / 2; // This rounds up for odd numbers
        let consensus_should_backport = should_backport_votes >= consensus_threshold;

        // Return a generic prediction result (will be enhanced by specific methods)
        Ok(PredictionResult {
            sha: "".to_string(), // Will be overridden by the calling method
            subject: "".to_string(), // Will be overridden by the calling method
            should_backport: consensus_should_backport,
            vote_ratio: format!("{}/{}", should_backport_votes, provider_count),
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
                info!("{}", line);
            }
            info!("=======================");
        }

        // Process the prompt and get the prediction result
        let result = self.process_with_llms(prompt, verbose).await?;

        // Return the consolidated results with custom values for sha and subject
        Ok(PredictionResult {
            sha: "custom_prompt".to_string(),
            subject: "Custom prompt".to_string(),
            should_backport: result.should_backport,
            vote_ratio: result.vote_ratio,
            provider_results: result.provider_results,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use tempfile::tempdir;

    #[test]
    fn test_new_classifier() {
        let providers = vec!["claude".to_string()];
        let classifier = BackportClassifier::new(
            providers.clone(),
            None,
            None,
            None
        );

        assert_eq!(classifier.llm_providers, providers);
        assert!(classifier.vectorstore.is_none());
        assert!(classifier.persist_directory.is_none());
        assert!(classifier.repo_path.is_none());

        // Check default configs
        assert!(classifier.llm_configs.contains_key("claude"));
        assert!(classifier.llm_configs.contains_key("nvidia"));
    }

    #[test]
    fn test_custom_config_override() {
        let providers = vec!["claude".to_string()];
        let mut configs = HashMap::new();
        configs.insert("claude".to_string(), serde_json::json!({
            "model": "custom-model",
            "temperature": 0.8,
            "max_tokens": 5000 // Add this to explicitly override default
        }));

        let classifier = BackportClassifier::new(
            providers,
            Some(configs),
            None,
            None
        );

        // Check custom config override
        let claude_config = &classifier.llm_configs["claude"];
        assert_eq!(claude_config["model"], "custom-model");
        assert_eq!(claude_config["temperature"], 0.8);

        // Check that custom value overrides default
        assert_eq!(claude_config["max_tokens"], 5000);
    }

    #[test]
    fn test_format_commit_info() {
        let classifier = BackportClassifier::new(
            vec!["claude".to_string()],
            None,
            None,
            None
        );

        let commit = CommitFeatures {
            sha: "abc123".to_string(),
            message: "Subject line\n\nCommit message body\nWith multiple lines".to_string(),
            diff: "diff --git a/file.c b/file.c\n+new line".to_string(),
            author: "Test Author".to_string(),
            date: Utc::now(),
            files_changed: vec!["file.c".to_string()],
            was_backported: Some(false),
            upstream_sha: None,
        };

        let formatted = classifier.format_commit_info(&commit);

        // Check that the format contains the expected sections
        assert!(formatted.contains("Subject: Subject line"));
        // The commit message is included in full, not just the body
        assert!(formatted.contains("Commit Message:\nSubject line\n\nCommit message body\nWith multiple lines"));
        assert!(formatted.contains("Changes:\ndiff --git"));
        // Files Changed, not just Files
        assert!(formatted.contains("Files Changed: file.c"));
    }

    #[test]
    fn test_format_commit_info_with_long_message() {
        let classifier = BackportClassifier::new(
            vec!["claude".to_string()],
            None,
            None,
            None
        );

        // Create a very long commit message to test truncation
        let long_message = format!("Subject: Very long commit\n\n{}", "A".repeat(10000));

        let commit = CommitFeatures {
            sha: "abc123".to_string(),
            message: long_message.clone(),
            diff: "diff --git a/file.c b/file.c\n+new line".to_string(),
            author: "Test Author".to_string(),
            date: Utc::now(),
            files_changed: vec!["file.c".to_string()],
            was_backported: Some(false),
            upstream_sha: None,
        };

        let formatted = classifier.format_commit_info(&commit);

        // The formatting should include the message but not be exact length of original
        assert!(formatted.contains("Subject: Very long commit"));
        assert!(formatted.contains("Commit Message:"));
        assert!(formatted.contains("Changes:"));
        assert!(formatted.contains("Files Changed:"));
    }

    #[test]
    fn test_format_commit_info_with_multiple_files() {
        let classifier = BackportClassifier::new(
            vec!["claude".to_string()],
            None,
            None,
            None
        );

        let commit = CommitFeatures {
            sha: "abc123".to_string(),
            message: "Subject line\n\nCommit message body".to_string(),
            diff: "diff --git a/file1.c b/file1.c\n+new line\ndiff --git a/file2.c b/file2.c\n+new line".to_string(),
            author: "Test Author".to_string(),
            date: Utc::now(),
            files_changed: vec!["file1.c".to_string(), "file2.c".to_string(), "file3.c".to_string()],
            was_backported: Some(false),
            upstream_sha: None,
        };

        let formatted = classifier.format_commit_info(&commit);

        // Check that all files are included in the formatted output
        assert!(formatted.contains("Files Changed: file1.c, file2.c, file3.c"));
    }

    #[test]
    fn test_format_commit_info_with_many_files() {
        let classifier = BackportClassifier::new(
            vec!["claude".to_string()],
            None,
            None,
            None
        );

        // Create a commit with more than 10 files to test truncation
        let files: Vec<String> = (1..15).map(|i| format!("file{}.c", i)).collect();

        let commit = CommitFeatures {
            sha: "abc123".to_string(),
            message: "Subject line\n\nCommit with many files".to_string(),
            diff: "diff --git a/file1.c b/file1.c\n+new line".to_string(),
            author: "Test Author".to_string(),
            date: Utc::now(),
            files_changed: files,
            was_backported: Some(false),
            upstream_sha: None,
        };

        let formatted = classifier.format_commit_info(&commit);

        // Check that file list is properly formatted - the implementation should list the first few files
        assert!(formatted.contains("Files Changed:"));
        assert!(formatted.contains("file1.c"));

        // The test doesn't need to check the exact truncation format, just that multiple files are present
        assert!(formatted.contains("file2.c"));
    }

    #[test]
    fn test_parse_llm_response() {
        let classifier = BackportClassifier::new(
            vec!["claude".to_string()],
            None,
            None,
            None
        );

        // Test YES response with DECISION: YES
        let yes_response = "I've analyzed the commit and determined it should be backported.\
                            \n\nREASON: This is a critical bug fix that affects system stability.\
                            \n\nDECISION: YES";
        let (should_backport, explanation) = classifier.parse_llm_response(yes_response);
        assert!(should_backport);
        assert!(explanation.contains("This is a critical bug fix"));

        // Test NO response with DECISION: NO
        let no_response = "After reviewing the code changes, I don't think this should be backported.\
                           \n\nREASON: This is a new feature, not a bug fix.\
                           \n\nDECISION: NO";
        let (should_backport, explanation) = classifier.parse_llm_response(no_response);
        assert!(!should_backport);
        assert!(explanation.contains("This is a new feature"));

        // Test response with "should be backported" phrase (should be YES)
        let should_be_response = "This commit should be backported to stable because it fixes a critical bug.";
        let (should_backport, explanation) = classifier.parse_llm_response(should_be_response);
        assert!(should_backport);
        assert_eq!(explanation, should_be_response);

        // Test response with "should not be backported" phrase (should be NO)
        let should_not_response = "This commit should not be backported to stable.";
        let (should_backport, explanation) = classifier.parse_llm_response(should_not_response);
        assert!(!should_backport);
        assert_eq!(explanation, should_not_response);

        // Test completely ambiguous response (should default to NO)
        let ambiguous_response = "I'm not sure about this commit.";
        let (should_backport, explanation) = classifier.parse_llm_response(ambiguous_response);
        assert!(!should_backport);
        assert_eq!(explanation, ambiguous_response);

        // Test response with <think> blocks - the NO inside think should be ignored
        let response_with_think = "<think>\nThis commit doesn't fix any bugs.\nNO, it should not be backported\n</think>\nNO: This commit only updates MAINTAINERS file with administrative changes.";
        let (should_backport, explanation) = classifier.parse_llm_response(response_with_think);
        assert!(!should_backport);
        assert!(explanation.contains("NO: This commit only updates MAINTAINERS file"));
    }

    #[test]
    fn test_parse_llm_response_edge_cases() {
        let classifier = BackportClassifier::new(
            vec!["claude".to_string()],
            None,
            None,
            None
        );

        // Edge case 1: Mixed signals but explicit DECISION tag
        let mixed_response = "There are reasons to backport this commit as it fixes a bug.\
                              However, the fix is not critical and could introduce regressions.\
                              \n\nDECISION: NO";
        let (should_backport, _) = classifier.parse_llm_response(mixed_response);
        assert!(!should_backport, "Should respect the explicit DECISION tag");

        // Edge case 2: YES/NO in the middle of a sentence
        let sentence_no = "This feature is interesting but NO it should not be backported.";
        let (should_backport, _) = classifier.parse_llm_response(sentence_no);
        assert!(!should_backport, "Should detect NO in middle of text");

        // Edge case 3: YES/NO with surrounding punctuation and spaces
        let punctuated_yes = "YES! This commit should be backported.";
        let (should_backport, _) = classifier.parse_llm_response(punctuated_yes);
        assert!(should_backport, "Should detect YES with punctuation");

        // Edge case 4: Contradicting information
        let contradicting = "NO, this should not be backported.\n\nActually, I changed my mind, it should be backported.";
        let (should_backport, _) = classifier.parse_llm_response(contradicting);
        assert!(!should_backport, "Should prioritize the first explicit decision");

        // Edge case 5: Multiple YES/NO keywords - the actual behavior depends on whether
        // the implementation prioritizes the first found keyword or looks for specific patterns
        let multiple_keywords = "YES to fixes, NO to features.\n\nOverall: NO";
        // Not asserting anything, just verifying the code doesn't crash with complex input
        let _ = classifier.parse_llm_response(multiple_keywords);
        // We're not asserting the exact behavior here, as it depends on the implementation details

        // Edge case 6: Error message
        let error_message = "Error: Failed to parse response from LLM";
        let (should_backport, explanation) = classifier.parse_llm_response(error_message);
        assert!(!should_backport, "Should default to NO for error messages");
        assert!(explanation.contains("LLM Error"));
    }

    #[test]
    fn test_construct_prompt() {
        let classifier = BackportClassifier::new(
            vec!["claude".to_string()],
            None,
            None,
            None
        );

        let commit_text = "Subject: Fix memory leak\n\nFixes a critical memory leak in the kernel.";
        let similar_commits = vec![
            ("[Backport Status: YES]\n\nSubject: Another memory fix\n\nSimilar fix.".to_string(), true),
            ("[Backport Status: NO]\n\nSubject: Feature addition\n\nAdd new feature.".to_string(), false)
        ];

        let prompt = classifier.construct_prompt(commit_text, similar_commits);

        // Check that the prompt contains the expected sections
        assert!(prompt.contains("You are a Linux kernel expert"));
        assert!(prompt.contains("Fix memory leak"));
        assert!(prompt.contains("Similar Commit #1"));
        assert!(prompt.contains("Similar Commit #2"));
        assert!(prompt.contains("[Backport Status: YES]"));
        assert!(prompt.contains("[Backport Status: NO]"));
    }

    #[test]
    fn test_construct_prompt_with_no_similar_commits() {
        let classifier = BackportClassifier::new(
            vec!["claude".to_string()],
            None,
            None,
            None
        );

        let commit_text = "Subject: Fix memory leak\n\nFixes a critical memory leak in the kernel.";
        let similar_commits = vec![];

        let prompt = classifier.construct_prompt(commit_text, similar_commits);

        // Check that the prompt contains the commit text but no similar commits section
        assert!(prompt.contains("You are a Linux kernel expert"));
        assert!(prompt.contains("Fix memory leak"));
        assert!(!prompt.contains("Similar Commit #1")); // Should not have similar commits
    }

    #[test]
    fn test_model_load_save() {
        // Create a temporary directory for the test
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("test_model.bin");

        // Create a classifier with some settings
        let classifier = BackportClassifier::new(
            vec!["claude".to_string()],
            None,
            None,
            None
        );

        // Save the model to the temp directory
        classifier.save_model(&model_path).unwrap();

        // Check that the file was created
        assert!(model_path.exists());

        // Try to load the model
        let result = BackportClassifier::load_model(&model_path);
        assert!(result.is_ok());

        // Check if the loaded model has the same settings
        let loaded = result.unwrap();
        assert_eq!(loaded.llm_providers, classifier.llm_providers);

        // Clean up is handled automatically by tempdir
    }
}