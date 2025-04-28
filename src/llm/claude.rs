// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use async_trait::async_trait;
use serde_json::Value;

use crate::http::AuthType;
use crate::llm::{Llm, LlmError, ResponseExtractor, ClaudeResponse, ModelType};
use crate::llm::provider::{LlmConfig, BaseLlmConfig};
use crate::llm::client::{LlmClient, ConfigHelper};
use crate::llm::claude_model::ClaudeModelType;

/// Claude configuration
#[derive(Debug, Clone)]
pub struct ClaudeConfig {
    pub model_type: ClaudeModelType,
    pub base: BaseLlmConfig,
}

impl Default for ClaudeConfig {
    fn default() -> Self {
        Self {
            model_type: ClaudeModelType::default_model(),
            base: BaseLlmConfig::default(),
        }
    }
}

impl LlmConfig for ClaudeConfig {
    fn from_json(json: Option<Value>) -> Result<Self, LlmError> {
        let json = json.unwrap_or(Value::Object(serde_json::Map::new()));

        let model_type = if let Some(mt) = json.get("model_type").and_then(|v| v.as_str()) {
            ClaudeModelType::from_str(mt)?
        } else if let Some(model) = json.get("model").and_then(|v| v.as_str()) {
            // Handle legacy "model" field for backward compatibility
            match model {
                "claude-3-7-sonnet-20250219" => ClaudeModelType::Sonnet,
                "claude-3-7-opus-20250501" => ClaudeModelType::Opus,
                "claude-3-haiku-20240307" => ClaudeModelType::Haiku,
                _ => ClaudeModelType::default_model(),
            }
        } else {
            ClaudeModelType::default_model()
        };

        let base = BaseLlmConfig::from_json(&json);

        Ok(Self {
            model_type,
            base,
        })
    }
}

/// Claude implementation
pub struct ClaudeLlm {
    config: ClaudeConfig,
    client: reqwest::Client,
}

impl ClaudeLlm {
    pub fn new(config: ClaudeConfig) -> Result<Self, LlmError> {
        let temp = Self {
            config: config.clone(),
            client: reqwest::Client::new(),
        };

        // Initialize the client with proper credentials
        let client = temp.get_http_client()?;

        // Return the fully initialized instance
        Ok(Self {
            config,
            client,
        })
    }
}

// Implementation of the ConfigHelper trait
impl crate::llm::client::ConfigHelper for ClaudeLlm {
    fn base_config(&self) -> &crate::llm::provider::BaseLlmConfig {
        &self.config.base
    }
}


#[async_trait]
impl LlmClient for ClaudeLlm {
    fn api_key_env_var(&self) -> &str {
        "ANTHROPIC_API_KEY"
    }

    fn create_auth(&self, api_key: String) -> AuthType {
        AuthType::ApiKey {
            header_name: "x-api-key".to_string(),
            key: api_key
        }
    }

    fn custom_headers(&self) -> Vec<(String, String)> {
        vec![
            ("anthropic-version".to_string(), "2023-06-01".to_string())
        ]
    }

    fn api_url(&self) -> &str {
        "https://api.anthropic.com/v1/messages"
    }

    fn model_name(&self) -> String {
        self.config.model_type.model_name()
    }

    // Use the helper trait methods for these common properties
    fn temperature(&self) -> f32 {
        self.get_temperature()
    }

    fn max_tokens(&self) -> usize {
        self.get_max_tokens()
    }

    fn debug_logging(&self) -> bool {
        self.get_debug_logging()
    }

    fn max_retries(&self) -> usize {
        self.get_max_retries()
    }

    fn initial_backoff_ms(&self) -> u64 {
        self.get_initial_backoff_ms()
    }
}

#[async_trait]
impl Llm for ClaudeLlm {
    async fn invoke(&self, prompt: &str) -> Result<String, LlmError> {
        // Create the request using the base implementation
        let request = self.create_request(prompt);

        // Send the request and get the response
        let response: ClaudeResponse = self.send_request(&self.client, &request).await?;

        // Extract the text using the response extractor trait
        response.extract_text()
    }
}