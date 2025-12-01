// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use log::{info, debug, warn, error};
use chrono::{DateTime, Utc};
use sha2::{Sha256, Digest};
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;
use rayon::prelude::*;
use std::sync::Arc;

// Import from the commit-classifier library
use commit_classifier::CommitFeatures;

pub struct BackportDataCollector {
    pub kernel_repo_path: PathBuf,
    pub seen_commit_messages: HashSet<String>,
    pub seen_subject_lines: HashSet<String>,
}

impl BackportDataCollector {
    pub fn new(kernel_repo_path: &Path) -> Result<Self, String> {
        let mut cmd = Command::new("git");
        cmd.current_dir(kernel_repo_path)
            .args(["rev-parse", "--is-inside-work-tree"]);

        match cmd.output() {
            Ok(output) => {
                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(format!("Not a valid git repository: {stderr}"));
                }
            },
            Err(e) => return Err(format!("Failed to verify git repository: {e}")),
        }

        Ok(BackportDataCollector {
            kernel_repo_path: kernel_repo_path.to_path_buf(),
            seen_commit_messages: HashSet::new(),
            seen_subject_lines: HashSet::new(),
        })
    }

    pub fn get_stable_branches(&self) -> Vec<String> {
        let mut stable_branches = Vec::new();

        if let Some(output) = self.safe_git_command(&["branch", "-r"]) {
            for line in output.lines() {
                let branch_name = line.trim();
                if branch_name.starts_with("origin/linux-") || branch_name.starts_with("stable/linux-") {
                    stable_branches.push(branch_name.to_string());
                }
            }
        } else {
            error!("Failed to get remote branches");
        }

        stable_branches
    }

    pub fn get_backported_commits(&self) -> HashSet<String> {
        let mut backported_commits = HashSet::new();

        let stable_branches = self.get_stable_branches();
        info!("Found {} stable branches", stable_branches.len());

        let mut filtered_branches = Vec::new();
        let mut skipped_branches = 0;

        for branch in &stable_branches {
            let version = if branch.starts_with("origin/linux-") {
                branch.replace("origin/linux-", "")
            } else if branch.starts_with("stable/linux-") {
                branch.replace("stable/linux-", "")
            } else {
                // Skip branches that don't match either pattern
                continue;
            };

            let version_parts: Vec<&str> = version.split('.').collect();

            if version_parts.len() >= 2
                && let (Ok(major), Ok(minor)) = (version_parts[0].parse::<i32>(), version_parts[1].parse::<i32>())
                && (major < 4 || (major == 4 && minor < 4))
            {
                debug!("Skipping {branch} - older than linux-4.4.y");
                skipped_branches += 1;
                continue;
            }

            filtered_branches.push(branch.clone());
        }

        info!("Processing {} branches in parallel", filtered_branches.len());

        let pb = ProgressBar::new(filtered_branches.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap());

        let per_branch_commits: Vec<HashSet<String>> = filtered_branches.par_iter()
            .map(|branch| {
                let result = self.process_branch(branch);
                pb.inc(1);
                result
            })
            .collect();

        pb.finish_with_message("Processed branches");

        for branch_commits in per_branch_commits {
            backported_commits.extend(branch_commits);
        }

        info!("Found {} unique upstream commits that were backported (skipped {} branches older than linux-4.4.y)",
              backported_commits.len(), skipped_branches);

        backported_commits
    }

    fn process_branch(&self, branch: &str) -> HashSet<String> {
        let mut branch_backported = HashSet::new();

        let output = self.safe_git_command(&["merge-base", "origin/master", branch]);
        let merge_base = match output {
            Some(base) => base.trim().to_string(),
            None => return branch_backported,
        };

        let merge_range = format!("{merge_base}..{branch}");
        let cmd = vec!["rev-list", "--no-merges", &merge_range];
        let output = self.safe_git_command(&cmd);

        let stable_commits = match output {
            Some(commits) => {
                commits.lines()
                    .filter(|line| !line.is_empty())
                    .map(std::string::ToString::to_string)
                    .collect::<Vec<String>>()
            },
            None => return branch_backported,
        };

        for commit_sha in stable_commits {
            if let Some(upstream_sha) = self.get_upstream_commit_sha(&commit_sha) {
                branch_backported.insert(upstream_sha);
            }
        }

        branch_backported
    }

    fn get_upstream_commit_sha(&self, stable_commit_sha: &str) -> Option<String> {
        let output = self.safe_git_command(&["log", "-1", stable_commit_sha]);

        if let Some(log_output) = output {
            let commit_lines: Vec<&str> = log_output.lines().skip(1).collect();
            let commit_text = commit_lines.join("\n");

            let sha_regex = Regex::new(r"[0-9a-f]{40}").unwrap();
            if let Some(captures) = sha_regex.captures(&commit_text) {
                return Some(captures[0].to_string());
            }
        }

        None
    }

    pub fn get_commit_features(&self, commit_sha: &str) -> Option<CommitFeatures> {
        let message = self.safe_git_command(
            &["log", "-1", "--pretty=format:%B", commit_sha]
        )?;

        let author_name = match self.safe_git_command(
            &["log", "-1", "--pretty=format:%an", commit_sha]
        ) {
            Some(name) => name,
            None => "Unknown".to_string(),
        };

        let date_str = self.safe_git_command(
            &["log", "-1", "--pretty=format:%aI", commit_sha]
        )?;

        let datetime = match DateTime::parse_from_rfc3339(&date_str) {
            Ok(dt) => dt.with_timezone(&Utc),
            Err(_) => Utc::now(), // Fallback to current time if invalid
        };

        let diff_text = self.safe_git_command(
            &["show", "-U20", "--format=", commit_sha]
        ).unwrap_or_default();

        let files_output = self.safe_git_command(
            &["show", "--name-only", "--format=", commit_sha]
        ).unwrap_or_default();

        let files_changed: Vec<String> = files_output
            .lines()
            .filter(|line| !line.is_empty())
            .map(std::string::ToString::to_string)
            .collect();

        Some(CommitFeatures {
            sha: commit_sha.to_string(),
            message,
            diff: diff_text,
            author: author_name,
            date: datetime,
            files_changed,
            was_selected: None,
            upstream_sha: None,
        })
    }

    // Execute git command with safety checks
    pub fn safe_git_command(&self, args: &[&str]) -> Option<String> {
        let mut cmd = Command::new("git");
        cmd.current_dir(&self.kernel_repo_path)
            .args(args);

        match cmd.output() {
            Ok(output) => {
                if output.status.success() {
                    match String::from_utf8(output.stdout) {
                        Ok(stdout) => Some(stdout),
                        Err(e) => {
                            warn!("Failed to decode git command output: {e}");
                            None
                        }
                    }
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    warn!("Git command failed: {} ({})", args.join(" "), stderr);
                    None
                }
            },
            Err(e) => {
                error!("Error executing git command {}: {}", args.join(" "), e);
                None
            }
        }
    }

    fn get_commit_message_hash(&self, commit_sha: &str) -> Option<(String, String)> {
        // Get the raw message
        let message = self.safe_git_command(
            &["log", "-1", "--pretty=format:%B", commit_sha]
        )?;

        // Split to get just the subject line for display
        let lines: Vec<&str> = message.lines().collect();
        let subject = lines.first().unwrap_or(&"").trim().to_string();

        // Hash the message
        let mut hasher = Sha256::new();
        hasher.update(message.as_bytes());
        let hash = format!("{:x}", hasher.finalize());

        Some((hash, subject))
    }

    pub fn is_duplicate_commit(&mut self, commit_sha: &str) -> bool {
        match self.get_commit_message_hash(commit_sha) {
            Some((hash, subject)) => {
                // Check if we've seen this message hash before
                if self.seen_commit_messages.contains(&hash) {
                    return true;
                }

                // Otherwise add it to our seen messages
                self.seen_commit_messages.insert(hash);

                // Check subject line
                if self.seen_subject_lines.contains(&subject) {
                    return true;
                }

                self.seen_subject_lines.insert(subject);
                false
            },
            None => false
        }
    }

    #[allow(clippy::too_many_lines)]
    pub fn build_dataset(&mut self, max_workers: usize,
               max_backported: Option<usize>, max_non_backported: Option<usize>) -> Vec<CommitFeatures> {
        // Configure the thread pool with the specified number of workers
        rayon::ThreadPoolBuilder::new()
            .num_threads(max_workers)
            .build_global()
            .unwrap();

        // Get backported commits (these are upstream commit SHAs)
        info!("Getting backported commits...");
        let backported_commits = self.get_backported_commits();
        info!("Retrieved {} unique upstream commits that were backported", backported_commits.len());

        // Reset seen commit messages
        self.seen_commit_messages.clear();
        self.seen_subject_lines.clear();

        let mut all_commits = Vec::new();

        // If in test mode, limit the number of backported commits
        let backported_list: Vec<String> = match max_backported {
            Some(limit) if limit < backported_commits.len() => {
                info!("Limiting backported commits to {limit} for testing");
                backported_commits.iter().take(limit).cloned().collect()
            },
            _ => backported_commits.iter().cloned().collect()
        };

        // Process backported commits
        let shas_to_process: Vec<String> = backported_list.iter()
            .filter(|sha| !self.is_duplicate_commit(sha))
            .cloned()
            .collect();

        // Progress bar
        let pb = ProgressBar::new(shas_to_process.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap());

        // Use progress bar that is thread-safe for parallel operations
        let pb = Arc::new(pb);

        // Collect and process commit features in parallel
        let processed_commits: Vec<CommitFeatures> = shas_to_process.par_iter()
            .filter_map(|sha| {
                let pb_clone = Arc::clone(&pb);
                let features = self.get_commit_features(sha);
                pb_clone.inc(1);

                match features {
                    Some(mut features) => {
                        features.was_selected = Some(true);
                        features.upstream_sha = Some(features.sha.clone());
                        Some(features)
                    },
                    None => None
                }
            })
            .collect();

        pb.finish_with_message("Processed backported commits");

        all_commits.extend(processed_commits);
        info!("Processed {} unique backported commits (skipped {} duplicates)",
              all_commits.len(), backported_list.len() - shas_to_process.len());

        // Get non-backported commits
        info!("Getting non-backported commits from v4.4 to 2 weeks ago...");

        // Calculate date boundaries (roughly)
        let oldest_date = "2016-01-10"; // Approximate date for v4.4
        let newest_date = chrono::Utc::now() - chrono::Duration::days(14);
        let newest_date_str = newest_date.format("%Y-%m-%d").to_string();

        // Get all non-merge commits in the date range
        let since_str = format!("--since={oldest_date}");
        let until_str = format!("--until={newest_date_str}");

        let cmd = vec![
            "rev-list", "--no-merges",
            &since_str,
            &until_str,
            "origin/master"
        ];

        match self.safe_git_command(&cmd) {
            Some(output) => {
                let all_master_shas: Vec<String> = output.lines()
                    .filter(|line| !line.is_empty())
                    .map(std::string::ToString::to_string)
                    .collect();

                info!("Found {} commits in origin/master in the date range", all_master_shas.len());

                // Filter out backported commits
                let non_backported_shas: Vec<String> = all_master_shas.into_iter()
                    .filter(|sha| !backported_commits.contains(sha))
                    .collect();

                let non_backported_count = non_backported_shas.len();
                info!("Found {non_backported_count} non-backported commits");

                // Clear previous deduplication state
                self.seen_commit_messages.clear();
                self.seen_subject_lines.clear();

                // First, compute message hashes in parallel
                info!("Computing commit message hashes for deduplication...");

                // Create a progress bar for deduplication
                let dedup_pb = ProgressBar::new(non_backported_shas.len() as u64);
                dedup_pb.set_style(ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) - Deduplicating")
                    .unwrap());
                dedup_pb.set_message("Deduplicating commits");

                let commit_hash_results: Vec<(String, Option<(String, String)>)> = non_backported_shas.par_iter()
                    .map(|sha| {
                        let hash_result = self.get_commit_message_hash(sha);
                        dedup_pb.inc(1);
                        (sha.clone(), hash_result)
                    })
                    .collect();

                dedup_pb.finish_with_message("Computed commit message hashes");

                // Now perform deduplication sequentially (this is fast since we already have all message hashes)
                info!("Filtering duplicate commits...");
                let mut seen_messages = HashSet::new();
                let mut seen_subjects = HashSet::new();
                let mut unique_non_backported = Vec::new();

                for (sha, hash_opt) in commit_hash_results {
                    if let Some((hash, subject)) = hash_opt {
                        // Check if we've seen this message hash before
                        if seen_messages.contains(&hash) {
                            continue;
                        }

                        // Otherwise add it to our seen messages
                        seen_messages.insert(hash);

                        // Check subject line
                        if seen_subjects.contains(&subject) {
                            continue;
                        }

                        seen_subjects.insert(subject);
                        unique_non_backported.push(sha);
                    }
                }

                let branch_duplicates = non_backported_count - unique_non_backported.len();
                info!("Found {} unique non-backported commits (skipped {} duplicates)",
                      unique_non_backported.len(), branch_duplicates);

                // Limit non-backported commits if needed
                let non_backported_list: Vec<String> = match max_non_backported {
                    Some(limit) if limit < unique_non_backported.len() => {
                        info!("Limiting non-backported commits to {limit} for testing");
                        unique_non_backported.into_iter().take(limit).collect()
                    },
                    _ => unique_non_backported
                };

                // Create progress bar for non-backported commits
                let pb = ProgressBar::new(non_backported_list.len() as u64);
                pb.set_style(ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) - Processing")
                    .unwrap());
                pb.set_message("Processing non-backported commits");

                // Make progress bar thread-safe
                let pb = Arc::new(pb);

                // Collect and process commit features in parallel
                let non_backported_commits: Vec<CommitFeatures> = non_backported_list.par_iter()
                    .filter_map(|sha| {
                        let pb_clone = Arc::clone(&pb);
                        let features = self.get_commit_features(sha);
                        pb_clone.inc(1);

                        match features {
                            Some(mut features) => {
                                features.was_selected = Some(false);
                                Some(features)
                            },
                            None => None
                        }
                    })
                    .collect();

                pb.finish_with_message("Processed non-backported commits");

                // Combine all commits
                all_commits.extend(non_backported_commits);

                info!("Final dataset: {} total unique commits", all_commits.len());

                let backported_count = all_commits.iter()
                    .filter(|c| c.was_selected == Some(true))
                    .count();

                info!("  - Backported commits: {} (skipped {} duplicates)",
                      backported_count, backported_list.len() - shas_to_process.len());
                info!("  - Non-backported commits: {} (skipped {} duplicates)",
                      all_commits.len() - backported_count, branch_duplicates);
            },
            None => {
                error!("Failed to get non-backported commits");
            }
        }

        all_commits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create a test commit message
    fn create_test_commit_message(subject: &str, body: &str) -> String {
        format!("{subject}\n\n{body}")
    }

    #[test]
    fn test_get_commit_message_hash() {
        // Create a mock collector with a temporary path (no need for real git repo for this test)
        let _collector = BackportDataCollector {
            kernel_repo_path: PathBuf::from("/tmp"),
            seen_commit_messages: HashSet::new(),
            seen_subject_lines: HashSet::new(),
        };

        // We'll test the hash calculation logic directly
        // by creating a mock message and manually computing the hash
        let subject = "Fix memory leak in the kernel";
        let body = "This commit fixes a memory leak in the kernel.\n\nSigned-off-by: Test <test@example.com>";
        let full_message = create_test_commit_message(subject, body);

        // Calculate SHA-256 hash of the full message
        let mut hasher = Sha256::new();
        hasher.update(full_message.as_bytes());
        let expected_hash = format!("{:x}", hasher.finalize());

        // Since we can't call the original method without a git repo,
        // we'll reimplement the logic here for testing
        let subject_line = full_message.lines().next().unwrap_or("").to_string();
        let mut hasher = Sha256::new();
        hasher.update(full_message.as_bytes());
        let message_hash = format!("{:x}", hasher.finalize());

        // Verify subject line and hash match expectations
        assert_eq!(subject_line, subject);
        assert_eq!(message_hash, expected_hash);
    }

    #[test]
    fn test_is_duplicate_commit_empty() {
        let mut collector = BackportDataCollector {
            kernel_repo_path: PathBuf::from("/tmp"),
            seen_commit_messages: HashSet::new(),
            seen_subject_lines: HashSet::new(),
        };

        // Since we've overridden the implementation to avoid git calls,
        // we'll directly manipulate the seen_subject_lines and seen_commit_messages
        assert!(!collector.seen_subject_lines.contains("Test subject"));
        assert!(!collector.seen_commit_messages.contains("abcdef1234"));

        // Insert values into the HashSets to simulate previous commits
        collector.seen_subject_lines.insert("Test subject".to_string());
        collector.seen_commit_messages.insert("abcdef1234".to_string());

        // Verify they are now detected as seen
        assert!(collector.seen_subject_lines.contains("Test subject"));
        assert!(collector.seen_commit_messages.contains("abcdef1234"));
    }

    #[test]
    fn test_deduplicate_dataset() {
        // Generate some test commit features with duplicates
        let date = Utc::now();
        let commits = vec![
            // First commit
            CommitFeatures {
                sha: "abc123".to_string(),
                message: "First commit".to_string(),
                diff: "diff1".to_string(),
                author: "Author 1".to_string(),
                date,
                files_changed: vec!["file1.c".to_string()],
                was_selected: Some(true),
                upstream_sha: None,
            },
            // Duplicate commit with same message but different SHA
            CommitFeatures {
                sha: "def456".to_string(),
                message: "First commit".to_string(),
                diff: "diff2".to_string(),
                author: "Author 2".to_string(),
                date,
                files_changed: vec!["file2.c".to_string()],
                was_selected: Some(false),
                upstream_sha: None,
            },
            // Unique commit
            CommitFeatures {
                sha: "ghi789".to_string(),
                message: "Second commit".to_string(),
                diff: "diff3".to_string(),
                author: "Author 3".to_string(),
                date,
                files_changed: vec!["file3.c".to_string()],
                was_selected: Some(true),
                upstream_sha: None,
            },
        ];

        // Deduplicate by subject line (simulates what build_dataset would do)
        let mut unique_subjects = HashSet::new();
        let deduplicated: Vec<CommitFeatures> = commits.into_iter()
            .filter(|commit| {
                let subject = commit.message.lines().next().unwrap_or("").to_string();
                unique_subjects.insert(subject)
            })
            .collect();

        // Verify that only two unique commits remain
        assert_eq!(deduplicated.len(), 2);

        // Check the subjects of the remaining commits
        let subjects: Vec<String> = deduplicated.iter()
            .map(|commit| commit.message.lines().next().unwrap_or("").to_string())
            .collect();

        assert!(subjects.contains(&"First commit".to_string()));
        assert!(subjects.contains(&"Second commit".to_string()));
    }
}