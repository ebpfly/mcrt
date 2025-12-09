import { BatchUpdateEvent, SimulationCompleteEvent, SimulationErrorEvent } from '../types';

type SSEEventHandler = {
  onBatchComplete?: (data: BatchUpdateEvent) => void;
  onSimulationComplete?: (data: SimulationCompleteEvent) => void;
  onError?: (data: SimulationErrorEvent) => void;
  onConnectionError?: (error: Event) => void;
};

export class SSEClient {
  private eventSource: EventSource | null = null;
  private handlers: SSEEventHandler = {};
  private sessionId: string | null = null;

  connect(streamUrl: string, sessionId: string, handlers: SSEEventHandler): void {
    this.disconnect();

    this.sessionId = sessionId;
    this.handlers = handlers;
    this.eventSource = new EventSource(streamUrl);

    this.eventSource.addEventListener('batch_complete', (event: MessageEvent) => {
      try {
        const data: BatchUpdateEvent = JSON.parse(event.data);
        this.handlers.onBatchComplete?.(data);
      } catch (e) {
        console.error('Failed to parse batch_complete event:', e);
      }
    });

    this.eventSource.addEventListener('simulation_complete', (event: MessageEvent) => {
      try {
        const data: SimulationCompleteEvent = JSON.parse(event.data);
        this.handlers.onSimulationComplete?.(data);
        this.disconnect();
      } catch (e) {
        console.error('Failed to parse simulation_complete event:', e);
      }
    });

    this.eventSource.addEventListener('error', (event: MessageEvent) => {
      try {
        const data: SimulationErrorEvent = JSON.parse(event.data);
        this.handlers.onError?.(data);
        this.disconnect();
      } catch (e) {
        console.error('Failed to parse error event:', e);
      }
    });

    this.eventSource.addEventListener('keepalive', () => {
      // Just acknowledge keepalive
    });

    this.eventSource.onerror = (error: Event) => {
      console.error('SSE connection error:', error);
      this.handlers.onConnectionError?.(error);
    };
  }

  disconnect(): void {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
      this.sessionId = null;
    }
  }

  isConnected(): boolean {
    return this.eventSource !== null && this.eventSource.readyState === EventSource.OPEN;
  }

  getSessionId(): string | null {
    return this.sessionId;
  }
}

export const sseClient = new SSEClient();
export default sseClient;
