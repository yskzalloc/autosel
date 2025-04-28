// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use async_trait::async_trait;
use serde_json::Value;

use crate::http::AuthType;
use crate::llm::{Llm, LlmError, ResponseExtractor, OpenAIResponse, ModelType};
use crate::llm::provider::{LlmConfig, BaseLlmConfig};
use crate::llm::client::{LlmClient, ConfigHelper};

/// OpenAI model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAIModelType {
    GPT4Turbo,
    GPT4O,
}

impl crate::llm::ModelType for OpenAIModelType {
    fn from_str(s: &str) -> Result<Self, LlmError> {
        match s.to_lowercase().as_str() {
            "gpt4" | "gpt4-turbo" => Ok(OpenAIModelType::GPT4Turbo),
            "gpt4o" => Ok(OpenAIModelType::GPT4O),
            _ => Err(crate::llm::ModelHelper::invalid_model_error(s, "OpenAI")),
        }
    }

    fn model_name(&self) -> String {
        match self {
            OpenAIModelType::GPT4Turbo => "gpt-4-turbo-2024-04-09".to_string(),
            OpenAIModelType::GPT4O => "gpt-4o-2024-05-13".to_string(),
        }
    }

    fn default_model() -> Self {
        OpenAIModelType::GPT4O
    }
}

/// OpenAI configuration
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    pub model_type: OpenAIModelType,
    pub base: BaseLlmConfig,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            model_type: OpenAIModelType::default_model(),
            base: BaseLlmConfig::default(),
        }
    }
}

impl LlmConfig for OpenAIConfig {
    fn from_json(json: Option<Value>) -> Result<Self, LlmError> {
        let json = json.unwrap_or(Value::Object(serde_json::Map::new()));

        let model_type = if let Some(mt) = json.get("model_type").and_then(|v| v.as_str()) {
            OpenAIModelType::from_str(mt)?
        } else {
            OpenAIModelType::default_model()
        };

        let base = BaseLlmConfig::from_json(&json);

        Ok(Self {
            model_type,
            base,
        })
    }
}

/// OpenAI implementation
pub struct OpenAILlm {
    config: OpenAIConfig,
    client: reqwest::Client,
}

impl OpenAILlm {
    pub fn new(config: OpenAIConfig) -> Result<Self, LlmError> {
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
impl crate::llm::client::ConfigHelper for OpenAILlm {
    fn base_config(&self) -> &crate::llm::provider::BaseLlmConfig {
        &self.config.base
    }
}


#[async_trait]
impl LlmClient for OpenAILlm {
    fn api_key_env_var(&self) -> &str {
        "OPENAI_API_KEY"
    }

    fn create_auth(&self, api_key: String) -> AuthType {
        AuthType::BearerToken(api_key)
    }

    fn custom_headers(&self) -> Vec<(String, String)> {
        vec![]
    }

    fn api_url(&self) -> &str {
        "https://api.openai.com/v1/chat/completions"
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
impl Llm for OpenAILlm {
    async fn invoke(&self, prompt: &str) -> Result<String, LlmError> {
        // Create the request using the base implementation
        let request = self.create_request(prompt);

        // Send the request and get the response
        let response: OpenAIResponse = self.send_request(&self.client, &request).await?;

        // Extract the text using the response extractor trait
        response.extract_text()
    }
}