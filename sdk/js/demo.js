/**
 * Node.js Demo invoking SwarmScope Safety Server
 * Run using: node demo.js
 */

async function runDemo() {
    const baseUrl = "http://localhost:8000";
    console.log("==================================================");
    console.log("       SWARMSCOPE POLYGLOT JS/TS CLIENT DEMO      ");
    console.log("==================================================");
    console.log(`Connecting to SwarmScope safety server at: ${baseUrl}`);

    try {
        // Check API Health status
        const healthResponse = await fetch(`${baseUrl}/health`);
        const health = await healthResponse.json();
        console.log(`Connection status: OK (App: ${health.app})`);
    } catch (e) {
        console.log("Error: Safety server is offline. Please start FastAPI via: uvicorn sandbox.api:app --reload");
        process.exit(1);
    }

    console.log("\n1. Querying pre-deployment ML risk score model...");
    try {
        const predResponse = await fetch(`${baseUrl}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                scenario_name: "negotiation",
                backend: "dummy",
                temperature: 0.95,
                turns: 15
            })
        });
        const prediction = await predResponse.json();
        console.log(`   - Target scenario: ${prediction.scenario_name}`);
        console.log(`   - Projected risk probability: ${(prediction.failure_probability * 100).toFixed(1)}%`);
        console.log(`   - Risk level recommendation: ${prediction.risk_level.toUpperCase()}`);
    } catch (e) {
        console.error("   - Failed to fetch prediction:", e.message);
    }

    console.log("\n2. Launching full simulation audit stream...");
    try {
        const simResponse = await fetch(`${baseUrl}/simulate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                scenario_name: "negotiation",
                backend: "dummy",
                temperature: 0.6,
                turns: 4
            })
        });
        const run = await simResponse.json();
        console.log(`   - Created simulation run: ${run.metadata.simulation_id}`);
        console.log(`   - Completed turn count: ${run.metadata.total_turns}`);
        
        console.log("\n3. Transcript logs received:");
        run.messages.forEach(msg => {
            console.log(`     [Turn ${msg.turn}] ${msg.sender}: "${msg.content}"`);
        });

        console.log("\n4. Vulnerability scanning report:");
        Object.keys(run.detector_scores).forEach(key => {
            console.log(`     * ${key.toUpperCase()}: ${run.detector_scores[key].toFixed(2)}`);
        });
    } catch (e) {
        console.error("   - Failed to execute simulation:", e.message);
    }
    console.log("\n==================================================");
}

runDemo();
