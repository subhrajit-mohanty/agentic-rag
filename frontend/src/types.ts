
export type Framework = 'LangGraph' | 'CrewAI' | 'GeminiNative';

export interface SourceItem {
  arxiv_id: string;
  title: string;
  authors: string[];
  url: string;
  // Use 'number' instead of 'float' for TypeScript compatibility
  relevance_score: number;
}

export interface Persona {
  id: string;
  name: string;
  systemPrompt: string;
  temperature: number;
  allowedTools: string[];
}

export interface Connector {
  id: string;
  type: 'SharePoint' | 'GoogleDrive' | 'S3';
  name: string;
  status: 'connected' | 'error' | 'disconnected' | 'syncing';
  lastSync?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  reasoningSteps?: string[];
  framework: Framework;
  sources?: SourceItem[];
  retrievalAttempts?: number;
  executionTime?: number;
}

export interface ProviderKey {
  id: string;
  provider: string;
  apiKey: string;
  status: 'Active' | 'Revoked' | 'Rotating' | 'Validation Failed';
  lastRotated: string;
  rotationSchedule?: number;
  nextRotation?: string;
  autoRotate: boolean;
}

export interface LLMConfig {
  provider: 'OpenAI' | 'Anthropic' | 'Google' | 'vLLM';
  apiKey: string;
  endpoint?: string;
}

export interface VLLMEndpoint {
  id: string;
  name: string;
  url: string;
  apiKey: string;
  status: 'UP' | 'DOWN' | 'PENDING';
}