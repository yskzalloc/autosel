// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use async_trait::async_trait;
use serde_json::Value;
use serde::{Deserialize, Serialize};

use crate::http::AuthType;
use crate::llm::{Llm, LlmError};
use crate::llm::provider::{LlmConfig, BaseLlmConfig};
use crate::llm::client::{LlmClient, ConfigHelper};
use crate::llm::response::ResponseExtractor;

/// Ollama configuration
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    pub model: String,
    pub base: BaseLlmConfig,
    pub api_host: String,
    pub custom_model: Option<String>,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            model: "llama3".to_string(),
            base: BaseLlmConfig::default(),
            api_host: "http://localhost:11434".to_string(),
            custom_model: None,
        }
    }
}

impl LlmConfig for OllamaConfig {
    fn from_json(json: Option<Value>) -> Result<Self, LlmError> {
        let json = json.unwrap_or(Value::Object(serde_json::Map::new()));

        let model = json.get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("llama3")
            .to_string();

        let base = BaseLlmConfig::from_json(&json);

        let api_host = json.get("api_host")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "http://localhost:11434".to_string());

        let custom_model = json.get("custom_model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(Self {
            model,
            base,
            api_host,
            custom_model,
        })
    }
}

/// Ollama response format
#[derive(Deserialize, Debug)]
pub struct OllamaResponse {
    pub response: String,
    // These fields are needed for deserialization but aren't used in our code
    #[serde(default)]
    #[allow(dead_code)]
    pub model: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub created_at: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub done: bool,
    #[serde(default)]
    #[allow(dead_code)]
    pub done_reason: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub context: Vec<usize>,
    #[serde(default)]
    #[allow(dead_code)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub prompt_eval_count: Option<u64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub eval_count: Option<u64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub eval_duration: Option<u64>,
}

impl ResponseExtractor for OllamaResponse {
    fn extract_text(&self) -> Result<String, LlmError> {
        if self.response.is_empty() {
            return Err(LlmError::Api(String::from("Empty response in Ollama API response")));
        }

        // Remove the <think>...</think> blocks that appear in models like qwen3
        let response = remove_think_blocks(&self.response);

        Ok(response)
    }
}

// Helper function to remove <think>...</think> blocks from the text
fn remove_think_blocks(text: &str) -> String {
    let re = regex::Regex::new(r"(?s)<think>.*?</think>").unwrap_or_else(|_| {
        // If regex creation fails, use a simpler approach as fallback
        regex::Regex::new(r"<think>.*</think>").unwrap()
    });

    re.replace_all(text, "").trim().to_string()
}

/// Ollama request format
#[derive(Serialize, Debug)]
struct OllamaRequest {
    pub model: String,
    pub prompt: String,
    pub temperature: f32,
    pub num_predict: usize,
    pub stream: bool,
}

impl OllamaRequest {
    pub fn new(model: String, prompt: String, temperature: f32, max_tokens: usize) -> Self {
        Self {
            model,
            prompt,
            temperature,
            num_predict: max_tokens,
            stream: false, // We want the complete response, not streaming
        }
    }
}

/// Ollama implementation
pub struct OllamaLlm {
    config: OllamaConfig,
}

impl OllamaLlm {
    pub fn new(config: OllamaConfig) -> Result<Self, LlmError> {
        Ok(Self {
            config,
        })
    }

    fn get_full_api_url(&self) -> String {
        format!("{}/api/generate", self.config.api_host)
    }

    fn get_effective_model(&self) -> String {
        match &self.config.custom_model {
            Some(model) => model.clone(),
            None => self.config.model.clone(),
        }
    }
}

// Implementation of the ConfigHelper trait
impl crate::llm::client::ConfigHelper for OllamaLlm {
    fn base_config(&self) -> &crate::llm::provider::BaseLlmConfig {
        &self.config.base
    }
}

#[async_trait]
impl LlmClient for OllamaLlm {
    fn api_key_env_var(&self) -> &str {
        "OLLAMA_API_KEY" // Ollama might not require API keys for local deployments
    }

    fn create_auth(&self, api_key: String) -> AuthType {
        // Ollama typically doesn't use API keys for local deployments
        // but this could be customized for hosted instances
        // Always use BearerToken, even if empty
        AuthType::BearerToken(api_key)
    }

    fn custom_headers(&self) -> Vec<(String, String)> {
        vec![
            // Add any custom headers needed for Ollama
            ("Content-Type".to_string(), "application/json".to_string()),
        ]
    }

    fn api_url(&self) -> &str {
        // Dummy value since we override this with get_full_api_url
        ""
    }

    fn model_name(&self) -> String {
        self.get_effective_model()
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

    // Override HTTP client initialization since we may not need API keys
    fn initialize_client(&self) -> Result<reqwest::Client, LlmError> {
        let api_key = std::env::var(self.api_key_env_var()).unwrap_or_default();

        crate::http::create_llm_client(
            self.create_auth(api_key),
            self.custom_headers()
        ).map_err(|e| e.into())
    }

    // Override the create_request method since Ollama has a different format
    fn create_request(&self, prompt: &str) -> crate::llm::UniversalLlmRequest {
        // This implementation won't be used directly, but we need to provide it
        crate::llm::UniversalLlmRequest::new(
            self.model_name(),
            vec![crate::llm::Message::user(prompt.to_string())],
            self.temperature(),
            self.max_tokens()
        )
    }
}

#[async_trait]
impl Llm for OllamaLlm {
    async fn invoke(&self, prompt: &str) -> Result<String, LlmError> {
        // Create the Ollama-specific request
        let ollama_request = OllamaRequest::new(
            self.model_name(),
            prompt.to_string(),
            self.temperature(),
            self.max_tokens()
        );

        if self.debug_logging() {
            log::debug!("Sending prompt to Ollama (model: {}, length: {} chars)",
                self.model_name(), prompt.len());
        }

        let retry_config = self.create_retry_config();
        let client = self.get_http_client()?;

        // Send the request using the full URL
        let response: OllamaResponse = crate::http::post_json(
            &client,
            &self.get_full_api_url(),
            &ollama_request,
            Some(retry_config)
        ).await.map_err(|e: crate::http::HttpError| LlmError::from(e))?;

        // Extract the text directly
        response.extract_text()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_ollama_config_from_json() {
        // Test default config
        let config = OllamaConfig::from_json(None).unwrap();
        assert_eq!(config.model, "llama3");
        assert_eq!(config.api_host, "http://localhost:11434");
        assert_eq!(config.custom_model, None);

        // Test custom config
        let json = json!({
            "model_type": "mistral",
            "api_host": "http://my-ollama-server:11434",
            "custom_model": "my-custom-model",
            "temperature": 0.7,
            "max_tokens": 2000
        });

        let config = OllamaConfig::from_json(Some(json)).unwrap();
        assert_eq!(config.model, "mistral");
        assert_eq!(config.api_host, "http://my-ollama-server:11434");
        assert_eq!(config.custom_model, Some("my-custom-model".to_string()));
        assert_eq!(config.base.temperature, 0.7);
        assert_eq!(config.base.max_tokens, 2000);
    }

    #[test]
    fn test_ollama_response_extraction() {
        let response = OllamaResponse {
            response: "<think>This is thinking content that should be removed</think>\nThis is a test response from Ollama".to_string(),
            model: "qwen3:32b".to_string(),
            created_at: "2025-05-07T00:30:00Z".to_string(),
            done: true,
            done_reason: "stop".to_string(),
            context: vec![1, 2, 3],
            total_duration: Some(1000),
            load_duration: Some(100),
            prompt_eval_count: Some(10),
            prompt_eval_duration: Some(500),
            eval_count: Some(20),
            eval_duration: Some(800),
        };

        let text = response.extract_text().unwrap();
        assert_eq!(text, "This is a test response from Ollama");
    }

    #[test]
    fn test_remove_think_blocks() {
        // Test with standard think blocks
        let text = "<think>Some thinking content</think>\nActual response";
        assert_eq!(remove_think_blocks(text), "Actual response");

        // Test with multiline think blocks
        let text = "<think>\nLine 1\nLine 2\n</think>\nActual response";
        assert_eq!(remove_think_blocks(text), "Actual response");

        // Test with multiple think blocks
        let text = "<think>First thinking</think>\nSome response\n<think>More thinking</think>\nMore response";
        assert_eq!(remove_think_blocks(text).replace("\n\n", "\n"), "Some response\nMore response");

        // Test with no think blocks
        let text = "Just a normal response";
        assert_eq!(remove_think_blocks(text), "Just a normal response");
    }
}