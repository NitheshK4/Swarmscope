export interface AgentConfig {
    name: string;
    role: string;
    goal: string;
    traits?: Record<string, number>;
}

export interface ScenarioConfig {
    name: string;
    description: string;
    agents: AgentConfig[];
    system_prompt: string;
    max_turns: number;
}

export interface Message {
    sender: string;
    receiver: string;
    content: string;
    turn: number;
    timestamp?: string;
}

export interface SimulationMetadata {
    simulation_id: string;
    scenario_name: string;
    timestamp: string;
    total_turns: number;
    backend: string;
    temperature: number;
    status: string;
}

export interface SimulationRun {
    metadata: SimulationMetadata;
    messages: Message[];
    detector_scores: Record<string, number>;
    detector_explanations: Record<string, string>;
}

export interface PredictionResult {
    scenario_name: string;
    backend: string;
    temperature: number;
    turns: number;
    failure_probability: number;
    risk_level: "Low" | "Medium" | "High";
}

export class SwarmScopeClient {
    private baseUrl: string;

    constructor(baseUrl: string = "http://localhost:8000") {
        this.baseUrl = baseUrl.replace(/\/$/, "");
    }

    async checkHealth(): Promise<boolean> {
        try {
            const response = await fetch(`${this.baseUrl}/health`);
            if (!response.ok) return false;
            const data = await response.json();
            return data.status === "ok";
        } catch {
            return false;
        }
    }

    async runSimulation(
        scenarioName: string,
        backend: string = "dummy",
        temperature: number = 0.7,
        turns?: number
    ): Promise<SimulationRun> {
        const response = await fetch(`${this.baseUrl}/simulate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                scenario_name: scenarioName,
                backend,
                temperature,
                turns
            })
        });

        if (!response.ok) {
            throw new Error(`Failed to execute simulation: ${response.statusText}`);
        }

        return (await response.json()) as SimulationRun;
    }

    async predictRisk(
        scenarioName: string,
        backend: string,
        temperature: number,
        turns: number
    ): Promise<PredictionResult> {
        const response = await fetch(`${this.baseUrl}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                scenario_name: scenarioName,
                backend,
                temperature,
                turns
            })
        });

        if (!response.ok) {
            throw new Error(`Failed to predict safety risk: ${response.statusText}`);
        }

        return (await response.json()) as PredictionResult;
    }

    async registerExternalRun(
        scenarioName: string,
        backend: string = "custom",
        temperature: number = 0.7
    ): Promise<string> {
        const response = await fetch(`${this.baseUrl}/runs`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                scenario_name: scenarioName,
                backend,
                temperature
            })
        });

        if (!response.ok) {
            throw new Error(`Failed to register external simulation run: ${response.statusText}`);
        }

        const data = await response.json();
        return data.simulation_id;
    }

    async postExternalMessage(
        simulationId: string,
        sender: string,
        receiver: string,
        content: string,
        turn: number
    ): Promise<any> {
        const response = await fetch(`${this.baseUrl}/runs/${simulationId}/messages`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                sender,
                receiver,
                content,
                turn
            })
        });

        if (!response.ok) {
            throw new Error(`Failed to post external agent message: ${response.statusText}`);
        }

        return await response.json();
    }
}
