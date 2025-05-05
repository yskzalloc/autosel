// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use clap::{App, Arg};
use log::{info, debug, error, warn};
use std::collections::HashMap;
use std::path::Path;
use std::fs;
use indicatif::{ProgressBar, ProgressStyle};

mod embeddings;
mod vectorstore;
mod llm;
mod models;
mod collector;
mod classifier;
mod utils;
mod http;

use collector::BackportDataCollector;
use classifier::BackportClassifier;
use utils::setup_logging;
use models::ProviderResult;


fn main() {
    let matches = App::new("Linux Kernel Backport Classifier")
        .version("0.1.0")
        .author("Sasha Levin <sashal@kernel.org>")
        .about("Classifies if Linux kernel commits should be backported to stable trees")
        .arg(Arg::with_name("kernel-repo")
            .long("kernel-repo")
            .value_name("PATH")
            .help("Path to the Linux kernel repository")
            .default_value("~/linux")
            .takes_value(true))
        .arg(Arg::with_name("model-dir")
            .long("model-dir")
            .value_name("PATH")
            .help("Path to store the model")
            .default_value("./model")
            .takes_value(true))
        .arg(Arg::with_name("models")
            .long("models")
            .value_name("MODELS")
            .help("Comma-separated list of models to use (claude)")
            .takes_value(true))
        .arg(Arg::with_name("train")
            .long("train")
            .help("Train the model"))
        .arg(Arg::with_name("test")
            .long("test")
            .help("Test mode with limited dataset"))
        .arg(Arg::with_name("debug")
            .long("debug")
            .help("Enable debug logging"))
        .arg(Arg::with_name("commit")
            .long("commit")
            .value_name("SHA")
            .help("Commit SHA to analyze")
            .takes_value(true))
        .arg(Arg::with_name("commits")
            .long("commits")
            .value_name("SHAs")
            .help("List of commit SHAs to analyze")
            .multiple(true)
            .takes_value(true))
        .arg(Arg::with_name("output")
            .long("output")
            .value_name("FILE")
            .help("Output file for results (JSON format)")
            .takes_value(true))
        .arg(Arg::with_name("verbose")
            .long("verbose")
            .help("Enable verbose output"))
        .arg(Arg::with_name("batch")
            .long("batch")
            .value_name("EXPLANATION_DIR")
            .help("Output only commit ID and yes/no for each commit. If a directory path is provided, save LLM responses in that directory.")
            .takes_value(true))
        .arg(Arg::with_name("make-prompt")
            .long("make-prompt")
            .value_name("PROMPT_DIR")
            .help("Only generate LLM prompts for given commits and save them in the provided directory as $sha1.txt")
            .takes_value(true))
        .arg(Arg::with_name("send-prompt")
            .long("send-prompt")
            .value_name("FILE")
            .help("Skip embeddings initialization and send the prompt in the provided file to the selected LLMs")
            .takes_value(true))
        .get_matches();

    let batch_mode = matches.is_present("batch");

    setup_logging(matches.is_present("debug"), batch_mode);

    let kernel_repo = matches.value_of("kernel-repo").unwrap();
    let model_dir = matches.value_of("model-dir").unwrap();
    let test_mode = matches.is_present("test");
    let verbose = matches.is_present("verbose");
    let explanation_dir = matches.value_of("batch");
    let prompt_dir = matches.value_of("make-prompt");
    let prompt_file = matches.value_of("send-prompt");

    let model_dir_path = Path::new(model_dir);
    if !model_dir_path.exists() {
        if let Err(e) = fs::create_dir_all(model_dir_path) {
            error!("Failed to create model directory: {}", e);
            std::process::exit(1);
        }
    }

    let vectorstore_dir = model_dir_path.join("vectorstore");
    if !vectorstore_dir.exists() {
        if let Err(e) = fs::create_dir_all(&vectorstore_dir) {
            error!("Failed to create vectorstore directory: {}", e);
            std::process::exit(1);
        }
    }

    let llm_providers = if let Some(models) = matches.value_of("models") {
        models.split(',')
            .map(|s| s.trim().to_string())
            .collect::<Vec<String>>()
    } else {
        vec!["claude".to_string()]
    };

    if matches.is_present("commit") || matches.is_present("commits") {
        info!("Using LLM providers: {:?}", llm_providers);
    }

    let expanded_kernel_repo = shellexpand::tilde(kernel_repo);
    let kernel_repo_path = Path::new(expanded_kernel_repo.as_ref());

    let runtime = tokio::runtime::Runtime::new()
        .expect("Failed to create tokio runtime");

    if matches.is_present("train") {
        info!("Starting training process...");

        debug!("Initializing data collector with kernel repo: {}", kernel_repo_path.display());
        let collector = match BackportDataCollector::new(kernel_repo_path) {
            Ok(c) => c,
            Err(e) => {
                error!("Failed to initialize data collector: {}", e);
                std::process::exit(1);
            }
        };

        info!("Building dataset...");

        let max_workers = 48;

        let (max_backported, max_non_backported) = if test_mode {
            info!("Test mode: limiting dataset to 500 backported and 500 non-backported commits");
            (Some(500), Some(500))
        } else {
            (None, None)
        };

        let mut collector = collector;
        let dataset = collector.build_dataset(
            max_workers,
            max_backported,
            max_non_backported
        );

        let backported_count = dataset.iter().filter(|c| c.was_backported == Some(true)).count();
        info!("Dataset built with {} total commits:", dataset.len());
        info!("  - {} backported commits", backported_count);
        info!("  - {} non-backported commits", dataset.len() - backported_count);

        info!("Training classifier...");

        let mut classifier = BackportClassifier::new(
            Vec::new(),
            None,
            Some(&vectorstore_dir),
            Some(kernel_repo_path)
        );

        if let Err(e) = classifier.train(&dataset) {
                error!("Failed to train classifier: {}", e);
            std::process::exit(1);
        }

        let model_path = model_dir_path.join("backport_classifier.bin");
        info!("Saving model to {}", model_path.display());

        if let Err(e) = classifier.save_model(&model_path) {
            error!("Failed to save model: {}", e);
            std::process::exit(1);
        }

        info!("Training completed successfully");
    }

    if matches.is_present("commit") || matches.is_present("commits") {
        let model_path = model_dir_path.join("backport_classifier.bin");

        if !model_path.exists() {
            error!("Model file not found at {}. Please train the model first.", model_path.display());
            std::process::exit(1);
        }

        info!("Loading model from {}", model_path.display());

        let mut classifier = match BackportClassifier::load_model(&model_path) {
            Ok(c) => c,
            Err(e) => {
                error!("Failed to load model: {}", e);
                std::process::exit(1);
            }
        };

        classifier.repo_path = Some(kernel_repo_path.to_path_buf());

        if matches.is_present("models") {
            classifier.llm_providers = llm_providers.clone();
            classifier.llms = None;

            if verbose && classifier.llm_providers.contains(&"claude".to_string()) {
                if let Some(config) = classifier.llm_configs.get_mut("claude") {
                    if let Some(obj) = config.as_object_mut() {
                        obj.insert("debug_logging".to_string(), serde_json::json!(true));
                    }
                }
            }
        }

        let mut commits_to_analyze = Vec::new();
        if let Some(commit) = matches.value_of("commit") {
            commits_to_analyze.push(commit.to_string());
        }
        if let Some(commits) = matches.values_of("commits") {
            commits_to_analyze.extend(commits.map(|s| s.to_string()));
        }

        let collector = match BackportDataCollector::new(kernel_repo_path) {
            Ok(c) => c,
            Err(e) => {
                error!("Failed to initialize data collector: {}", e);
                std::process::exit(1);
            }
        };

        let mut results = HashMap::new();

        if commits_to_analyze.is_empty() {
            error!("No valid commits to analyze. Please provide valid commit SHAs with --commit SHA or --commits SHA1 SHA2 SHA3");
            std::process::exit(1);
        }

        let pb = ProgressBar::new(commits_to_analyze.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap());

        for sha in &commits_to_analyze {
            if sha.len() < 6 {
                warn!("Invalid commit SHA: {} (too short)", sha);
                continue;
            }

            match collector.get_commit_features(sha) {
                Some(features) => {
                    if let Some(dir) = prompt_dir {
                        if !classifier.embeddings.is_initialized() {
                            info!("Initializing embeddings model for prompt generation");
                            if let Err(e) = classifier.embeddings.initialize() {
                                error!("Failed to initialize embeddings model: {}", e);
                                continue;
                            }
                        }

                        let commit_text = classifier.format_commit_info(&features);

                        let similar_commits = classifier.find_similar_commits(&commit_text, 5);

                        let prompt = classifier.construct_prompt(&commit_text, similar_commits);

                        let prompt_dir_path = Path::new(dir);
                        if !prompt_dir_path.exists() {
                            if let Err(e) = fs::create_dir_all(prompt_dir_path) {
                                error!("Failed to create prompt directory: {}", e);
                                continue;
                            }
                        }

                        let filename = format!("{}.txt", sha);
                        let filepath = prompt_dir_path.join(&filename);

                        if let Err(e) = fs::write(&filepath, prompt) {
                            error!("Failed to write prompt file {}: {}", filepath.display(), e);
                        } else {
                            debug!("Saved prompt for {} to {}", sha, filepath.display());
                        }

                        pb.inc(1);
                        continue;
                    }

                    match runtime.block_on(classifier.predict(&features, verbose)) {
                        Ok(result) => {
                            if batch_mode {
                                let decision = if result.should_backport { "yes" } else { "no" };
                                println!("{} {}", sha, decision);

                                if let Some(dir) = explanation_dir {
                                    let explanation_dir_path = Path::new(dir);

                                    if !explanation_dir_path.exists() {
                                        if let Err(e) = fs::create_dir_all(explanation_dir_path) {
                                            error!("Failed to create explanation directory: {}", e);
                                        }
                                    }

                                    let filename = format!("{}.txt", sha);
                                    let filepath = explanation_dir_path.join(&filename);

                                    let mut full_explanation = String::new();
                                    for (provider, provider_result) in &result.provider_results {
                                        full_explanation.push_str(&format!("Provider: {}\n", provider));
                                        if let Some(raw_response) = &provider_result.raw_response {
                                            full_explanation.push_str(raw_response);
                                        } else {
                                            full_explanation.push_str(&provider_result.explanation);
                                        }
                                        full_explanation.push_str("\n\n");
                                    }

                                    if let Err(e) = fs::write(&filepath, full_explanation) {
                                        error!("Failed to write explanation file {}: {}", filepath.display(), e);
                                    } else {
                                        debug!("Saved explanation for {} to {}", sha, filepath.display());
                                    }
                                }
                            }
                            results.insert(sha.clone(), result);
                        },
                        Err(e) => {
                            error!("Failed to get prediction for commit {}: {}", sha, e);
                        }
                    }
                },
                None => {
                    warn!("Failed to get features for commit {}", sha);
                }
            }

            pb.inc(1);
        }

        pb.finish_with_message("Processed commits");

        if !results.is_empty() {
            if batch_mode {
                // Results already printed as they were processed
            } else {
                for (sha, result) in &results {
                    let subject = &result.subject;
                    let should_backport = result.should_backport;
                    let vote_ratio = &result.vote_ratio;

                    let status = if should_backport { "BACKPORT" } else { "DO NOT BACKPORT" };
                    info!("Commit {}: {} (Votes: {}) - {}", &sha[0..10], status, vote_ratio, subject);

                    if verbose {
                        for (provider, provider_result) in &result.provider_results {
                            let provider_decision = if provider_result.should_backport { "BACKPORT" } else { "DO NOT BACKPORT" };
                            info!("  - {}: {}", provider, provider_decision);

                            if let Some(error) = &provider_result.error {
                                info!("      Error: {}", error);
                            } else {
                                let explanation_lines: Vec<&str> = provider_result.explanation.lines().take(3).collect();
                                for line in &explanation_lines {
                                    info!("      {}", line);
                                }
                                if explanation_lines.len() < provider_result.explanation.lines().count() {
                                    info!("      ...");
                                }
                            }
                        }
                    }
                }
            }

            if let Some(output_file) = matches.value_of("output") {
                let output_path = Path::new(output_file);
                info!("Saving results to {}", output_path.display());

                match serde_json::to_string_pretty(&results) {
                    Ok(json) => {
                        match fs::write(output_path, json) {
                            Ok(_) => info!("Results saved successfully"),
                            Err(e) => error!("Error writing results file: {}", e),
                        }
                    },
                    Err(e) => error!("Error serializing results: {}", e),
                }
            }
        } else {
            warn!("No results to display");
        }
    }

    if let Some(prompt_file_path) = prompt_file {
        info!("Creating classifier for prompt processing");

        let config = if matches.is_present("models") && verbose {
            let mut configs = HashMap::new();
            if llm_providers.contains(&"claude".to_string()) {
                configs.insert("claude".to_string(), serde_json::json!({
                    "debug_logging": true
                }));
            }
            Some(configs)
        } else {
            None
        };

        let mut classifier = BackportClassifier::new_for_prompt(
            llm_providers.clone(),
            config,
            Some(kernel_repo_path)
        );

        let file_stem = Path::new(prompt_file_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("custom_prompt");

        let prompt = match fs::read_to_string(prompt_file_path) {
            Ok(content) => content,
            Err(e) => {
                error!("Failed to read prompt file {}: {}", prompt_file_path, e);
                std::process::exit(1);
            }
        };

        info!("Processing prompt from {}", prompt_file_path);
        match runtime.block_on(classifier.process_custom_prompt(&prompt, verbose)) {
            Ok(result) => {
                if batch_mode {
                    let decision = if result.should_backport { "yes" } else { "no" };
                    println!("{} {}", file_stem, decision);

                    if let Some(dir) = explanation_dir {
                        save_explanation(dir, file_stem, &result.provider_results);
                    }
                } else {
                    let status = if result.should_backport { "BACKPORT" } else { "DO NOT BACKPORT" };
                    info!("Result: {} (Votes: {})", status, &result.vote_ratio);

                    if verbose {
                        print_provider_details(&result.provider_results);
                    }
                }

                if let Some(output_file) = matches.value_of("output") {
                    let output_path = Path::new(output_file);
                    info!("Saving results to {}", output_path.display());

                    let mut results = HashMap::new();
                    results.insert(file_stem.to_string(), result);

                    if let Err(e) = fs::write(output_path,
                        serde_json::to_string_pretty(&results).unwrap_or_default()) {
                        error!("Error writing results file: {}", e);
                    } else {
                        info!("Results saved successfully");
                    }
                }
            },
            Err(e) => {
                error!("Failed to process prompt: {}", e);
            }
        }
    }

    if matches.is_present("make-prompt") {
        if let Some(dir) = prompt_dir {
            info!("All prompts saved to directory: {}", dir);
        }
    }

    if !matches.is_present("train") && !matches.is_present("commit") && !matches.is_present("commits") &&
       !matches.is_present("make-prompt") && !matches.is_present("send-prompt") {
        println!("{}", matches.usage());
    }
}

fn save_explanation(dir: &str, file_stem: &str, provider_results: &HashMap<String, ProviderResult>) {
    let explanation_dir_path = Path::new(dir);

    if !explanation_dir_path.exists() {
        if let Err(e) = fs::create_dir_all(explanation_dir_path) {
            error!("Failed to create explanation directory: {}", e);
            return;
        }
    }

    let filename = format!("{}.txt", file_stem);
    let filepath = explanation_dir_path.join(&filename);

    let mut full_explanation = String::new();
    for (provider, provider_result) in provider_results {
        full_explanation.push_str(&format!("Provider: {}\n", provider));
        if let Some(raw_response) = &provider_result.raw_response {
            full_explanation.push_str(raw_response);
        } else {
            full_explanation.push_str(&provider_result.explanation);
        }
        full_explanation.push_str("\n\n");
    }

    if let Err(e) = fs::write(&filepath, full_explanation) {
        error!("Failed to write explanation file {}: {}", filepath.display(), e);
    } else {
        debug!("Saved explanation for {} to {}", file_stem, filepath.display());
    }
}

fn print_provider_details(provider_results: &HashMap<String, ProviderResult>) {
    for (provider, provider_result) in provider_results {
        let provider_decision = if provider_result.should_backport { "BACKPORT" } else { "DO NOT BACKPORT" };
        info!("  - {}: {}", provider, provider_decision);

        if let Some(error) = &provider_result.error {
            info!("      Error: {}", error);
        } else {
            let explanation_lines: Vec<&str> = provider_result.explanation.lines().take(3).collect();
            for line in &explanation_lines {
                info!("      {}", line);
            }
            if explanation_lines.len() < provider_result.explanation.lines().count() {
                info!("      ...");
            }
        }
    }
}
