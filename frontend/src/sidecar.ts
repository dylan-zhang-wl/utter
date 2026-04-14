import { Command, Child } from "@tauri-apps/plugin-shell";

let serverProcess: Child | null = null;

export async function startBackend() {
  const command = Command.sidecar("../../backend/livescribe-server");

  command.on("error", (error) => {
    console.error("Backend error:", error);
  });

  command.stdout.on("data", (data) => {
    console.log("Backend:", data);
  });

  serverProcess = await command.spawn();
}

export async function stopBackend() {
  if (serverProcess) {
    await serverProcess.kill();
    serverProcess = null;
  }
}
