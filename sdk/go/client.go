package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type SimulationRequest struct {
	ScenarioName string  `json:"scenario_name"`
	Backend      string  `json:"backend"`
	Temperature  float64 `json:"temperature"`
	Turns        *int    `json:"turns,omitempty"`
}

type Message struct {
	Sender    string `json:"sender"`
	Receiver  string `json:"receiver"`
	Content   string `json:"content"`
	Turn      int    `json:"turn"`
	Timestamp string `json:"timestamp"`
}

type SimulationMetadata struct {
	SimulationID string  `json:"simulation_id"`
	ScenarioName string  `json:"scenario_name"`
	Timestamp    string  `json:"timestamp"`
	TotalTurns   int     `json:"total_turns"`
	Backend      string  `json:"backend"`
	Temperature  float64 `json:"temperature"`
	Status       string  `json:"status"`
}

type SimulationRun struct {
	Metadata             SimulationMetadata `json:"metadata"`
	Messages             []Message          `json:"messages"`
	DetectorScores       map[string]float64 `json:"detector_scores"`
	DetectorExplanations map[string]string  `json:"detector_explanations"`
}

type SwarmScopeClient struct {
	BaseURL    string
	HTTPClient *http.Client
}

func NewSwarmScopeClient(baseURL string) *SwarmScopeClient {
	return &SwarmScopeClient{
		BaseURL: strings.TrimSuffix(baseURL, "/"),
		HTTPClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

func (c *SwarmScopeClient) CheckHealth() (bool, error) {
	resp, err := c.HTTPClient.Get(c.BaseURL + "/health")
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return false, nil
	}

	var result map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return false, err
	}

	return result["status"] == "ok", nil
}

func (c *SwarmScopeClient) RunSimulation(scenarioName string, backend string, temperature float64) (*SimulationRun, error) {
	reqBody := SimulationRequest{
		ScenarioName: scenarioName,
		Backend:      backend,
		Temperature:  temperature,
	}

	jsonBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	resp, err := c.HTTPClient.Post(c.BaseURL+"/simulate", "application/json", bytes.NewBuffer(jsonBytes))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("server error: %s", string(bodyBytes))
	}

	var run SimulationRun
	if err := json.NewDecoder(resp.Body).Decode(&run); err != nil {
		return nil, err
	}

	return &run, nil
}

func main() {
	client := NewSwarmScopeClient("http://localhost:8000")
	fmt.Println("Testing Go client API wrapper...")
	ok, err := client.CheckHealth()
	if err != nil {
		fmt.Printf("Go Client healthcheck failed: %v\n", err)
		return
	}
	fmt.Printf("Server Health check: %t\n", ok)
}
