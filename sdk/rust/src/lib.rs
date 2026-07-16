use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub sender: String,
    pub receiver: String,
    pub content: String,
    pub turn: usize,
    pub timestamp: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SimulationMetadata {
    pub simulation_id: String,
    pub scenario_name: String,
    pub timestamp: String,
    pub total_turns: usize,
    pub backend: String,
    pub temperature: f64,
    pub status: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SimulationRun {
    pub metadata: SimulationMetadata,
    pub messages: Vec<Message>,
    pub detector_scores: HashMap<String, f64>,
    pub detector_explanations: HashMap<String, String>,
}

#[derive(Serialize, Debug)]
struct SimulationRequest {
    scenario_name: String,
    backend: String,
    temperature: f64,
    turns: Option<usize>,
}

pub struct SwarmScopeClient {
    base_url: String,
    client: reqwest::Client,
}

impl SwarmScopeClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub async fn check_health(&self) -> Result<bool, reqwest::Error> {
        let url = format!("{}/health", self.base_url);
        let resp = self.client.get(&url).send().await?;
        if resp.status().is_success() {
            let body: serde_json::Value = resp.json().await?;
            if let Some(status) = body.get("status") {
                return Ok(status == "ok");
            }
        }
        Ok(false)
    }

    pub async fn run_simulation(
        &self,
        scenario_name: &str,
        backend: &str,
        temperature: f64,
        turns: Option<usize>,
    ) -> Result<SimulationRun, reqwest::Error> {
        let url = format!("{}/simulate", self.base_url);
        let req_body = SimulationRequest {
            scenario_name: scenario_name.to_string(),
            backend: backend.to_string(),
            temperature,
            turns,
        };

        let run: SimulationRun = self
            .client
            .post(&url)
            .json(&req_body)
            .send()
            .await?
            .json()
            .await?;

        Ok(run)
    }
}
