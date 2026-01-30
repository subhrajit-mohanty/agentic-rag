
import React, { useState, useRef, useEffect } from 'react';
import { Framework, ChatMessage, Persona } from '../types';

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [selectedFramework, setSelectedFramework] = useState<Framework>('LangGraph');
  const [selectedPersona, setSelectedPersona] = useState<string>('');
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [expandedTraces, setExpandedTraces] = useState<Record<string, boolean>>({});
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchPersonas();
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  const fetchPersonas = async () => {
    try {
      const response = await fetch('/api/v1/personas');
      if (response.ok) {
        const data = await response.json();
        setPersonas(data);
        if (data.length > 0) setSelectedPersona(data[0].id);
      }
    } catch (e) {
      console.error("Failed to load personas", e);
    }
  };

  const getStepIcon = (content: string) => {
    const text = content.toLowerCase();
    if (text.includes('guardrail') || text.includes('scope')) return 'fa-shield-halved';
    if (text.includes('retrieve') || text.includes('search') || text.includes('lookup')) return 'fa-magnifying-glass';
    if (text.includes('node') || text.includes('process') || text.includes('execution') || text.includes('orchestrat')) return 'fa-gears';
    if (text.includes('grade') || text.includes('relevant') || text.includes('evaluat')) return 'fa-square-check';
    if (text.includes('rewrite') || text.includes('optimiz')) return 'fa-pen-to-square';
    if (text.includes('generate') || text.includes('synthesizing') || text.includes('answer')) return 'fa-brain';
    if (text.includes('out_of_scope') || text.includes('rejection')) return 'fa-ban';
    if (text.includes('tool')) return 'fa-wrench';
    return 'fa-terminal';
  };

  const getStepColor = (content: string, isLast: boolean) => {
    const text = content.toLowerCase();
    if (isLast) return 'text-blue-400 border-blue-500/30 bg-blue-500/10';
    if (text.includes('guardrail')) return 'text-amber-400 border-amber-500/30 bg-amber-500/10';
    if (text.includes('retrieve') || text.includes('search')) return 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10';
    if (text.includes('grade')) return 'text-purple-400 border-purple-500/30 bg-purple-500/10';
    if (text.includes('rewrite')) return 'text-sky-400 border-sky-500/30 bg-sky-500/10';
    if (text.includes('out_of_scope')) return 'text-rose-400 border-rose-500/30 bg-rose-500/10';
    return 'text-slate-400 border-slate-700 bg-slate-800/50';
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date().toLocaleTimeString(),
      framework: selectedFramework,
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    try {
      const response = await fetch('/api/v1/ask-agentic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: input,
          framework: selectedFramework,
          persona_id: selectedPersona || null,
          use_hybrid: true,
          top_k: 3
        })
      });

      if (!response.ok) throw new Error(`Agent gateway error: ${response.status}`);

      const data = await response.json();

      const assistantMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer || data.content,
        timestamp: new Date().toLocaleTimeString(),
        reasoningSteps: data.reasoning_steps || data.trace,
        framework: selectedFramework,
        sources: data.sources,
        executionTime: data.execution_time,
        retrievalAttempts: data.retrieval_attempts
      };

      setMessages(prev => [...prev, assistantMsg]);
      setExpandedTraces(prev => ({ ...prev, [assistantMsg.id]: true }));
    } catch (error) {
      console.error("Query Error:", error);
      const errorMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "System Error: Failed to initialize LangGraph orchestration kernels.",
        timestamp: new Date().toLocaleTimeString(),
        framework: selectedFramework,
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="flex flex-col h-full max-w-5xl mx-auto border border-slate-800 bg-slate-900 rounded-2xl overflow-hidden shadow-2xl">
      {/* Header */}
      <div className="px-6 py-4 border-b border-slate-800 flex flex-wrap items-center justify-between bg-slate-900/80 backdrop-blur-md gap-4 z-10">
        <div className="flex items-center gap-3">
          <div className="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-pulse shadow-[0_0_10px_rgba(16,185,129,0.5)]"></div>
          <h2 className="text-sm font-black uppercase tracking-widest text-slate-300">Inference Core</h2>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-[9px] text-slate-500 font-bold uppercase">Persona</label>
            <select 
              value={selectedPersona}
              onChange={(e) => setSelectedPersona(e.target.value)}
              className="bg-slate-800 border border-slate-700 text-[11px] px-2 py-1 rounded outline-none text-slate-300 focus:border-blue-500 transition-all"
            >
              <option value="">Default Cluster</option>
              {personas.map(p => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-[9px] text-slate-500 font-bold uppercase">Pipeline</label>
            <select 
              value={selectedFramework}
              onChange={(e) => setSelectedFramework(e.target.value as Framework)}
              className="bg-slate-800 border border-slate-700 text-[11px] px-2 py-1 rounded outline-none text-slate-300 focus:border-blue-500 transition-all"
            >
              <option value="LangGraph">LangGraph (Stateful)</option>
              <option value="CrewAI">CrewAI (Multi-Agent)</option>
              <option value="GeminiNative">Gemini (Zero-Shot)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 space-y-10 bg-slate-950/20 scroll-smooth">
        {messages.map((msg) => (
          <div key={msg.id} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'} animate-fadeIn`}>
            {/* Meta Info */}
            <div className="flex items-center gap-2 mb-2 px-2">
              <span className={`text-[9px] font-black uppercase tracking-[0.2em] ${msg.role === 'user' ? 'text-blue-500' : 'text-emerald-500'}`}>
                {msg.role === 'user' ? 'Command Operator' : 'Enterprise Agent'}
              </span>
              <span className="text-[9px] text-slate-700 font-mono opacity-50">•</span>
              <span className="text-[9px] text-slate-600 font-mono uppercase">{msg.timestamp}</span>
              {msg.executionTime && (
                <span className="text-[9px] text-slate-500 font-mono bg-slate-800/50 px-1.5 py-0.5 rounded border border-slate-700/50 ml-1">
                  {msg.executionTime}s latency
                </span>
              )}
            </div>

            <div className={`max-w-[92%] rounded-2xl shadow-xl overflow-hidden transition-all ${
              msg.role === 'user' 
                ? 'bg-blue-600 text-white px-6 py-4 rounded-tr-none border border-blue-500/50' 
                : 'bg-slate-900/90 border border-slate-800 p-0 rounded-tl-none backdrop-blur-sm'
            }`}>
              {/* Message Content */}
              <div className={`text-sm leading-relaxed whitespace-pre-wrap ${msg.role === 'assistant' ? 'px-6 py-5' : ''}`}>
                {msg.content}
              </div>

              {/* Reasoning Trace Section */}
              {msg.reasoningSteps && msg.reasoningSteps.length > 0 && (
                <div className="border-t border-slate-800">
                  <button 
                    onClick={() => setExpandedTraces(prev => ({ ...prev, [msg.id]: !prev[msg.id] }))}
                    className="w-full flex items-center justify-between px-6 py-3 bg-slate-900/50 hover:bg-slate-800/80 transition-all group"
                  >
                    <div className="flex items-center gap-3">
                      <div className={`w-2 h-2 rounded-full ${expandedTraces[msg.id] ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 'bg-slate-700'}`}></div>
                      <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest group-hover:text-slate-200 transition-colors">
                        {expandedTraces[msg.id] ? 'Minimize Reasoning' : 'Inspect Execution Trace'}
                      </span>
                    </div>
                    <i className={`fas fa-chevron-${expandedTraces[msg.id] ? 'up' : 'down'} text-[10px] text-slate-600 group-hover:text-slate-400 transition-transform`}></i>
                  </button>
                  
                  {expandedTraces[msg.id] && (
                    <div className="p-8 bg-slate-950/80 border-t border-slate-800/50 space-y-8 animate-fadeIn relative">
                      {/* Internal Log Meta */}
                      <div className="flex items-center justify-between opacity-30 mb-2">
                        <span className="text-[8px] font-black text-slate-500 uppercase tracking-[0.3em]">Execution Path Analytics</span>
                        <span className="text-[8px] font-mono text-slate-600">ID: {msg.framework.toUpperCase()}</span>
                      </div>

                      {/* Timeline Structure */}
                      <div className="relative space-y-6">
                        {/* The Connecting Line (using the active-flow-line from index.html) */}
                        <div className="absolute left-[13px] top-4 bottom-4 w-[2px] bg-slate-800 active-flow-line"></div>
                        
                        {msg.reasoningSteps.map((step, idx) => {
                          const isLast = idx === msg.reasoningSteps!.length - 1;
                          const icon = getStepIcon(step);
                          const colorClasses = getStepColor(step, isLast);
                          
                          return (
                            <div 
                              key={idx} 
                              className="relative pl-12 animate-step-in" 
                              style={{ animationDelay: `${idx * 150}ms`, opacity: 0 }}
                            >
                              {/* Node Icon Circle */}
                              <div className="absolute left-0 top-1 z-10">
                                <div className={`w-[28px] h-[28px] rounded-lg border flex items-center justify-center transition-all duration-300 shadow-md ${colorClasses} ${
                                  isLast ? 'ring-2 ring-blue-500/20' : 'hover:scale-110'
                                }`}>
                                  <i className={`fas ${icon} text-[11px]`}></i>
                                </div>
                              </div>
                              
                              {/* Step Label/Body */}
                              <div className={`p-4 rounded-xl border transition-all ${
                                isLast 
                                  ? 'bg-blue-600/5 border-blue-500/20 shadow-[0_4px_12px_rgba(0,0,0,0.1)]' 
                                  : 'bg-slate-900/40 border-slate-800/60 hover:border-slate-700'
                              }`}>
                                <div className="flex items-start justify-between gap-4">
                                  <p className={`text-[12px] leading-relaxed font-medium ${isLast ? 'text-blue-100' : 'text-slate-300'}`}>
                                    {step}
                                  </p>
                                  {step.includes(':') && (
                                    <span className="text-[8px] font-black text-slate-600 bg-slate-800/50 px-1.5 py-0.5 rounded uppercase shrink-0 mt-0.5">
                                      {step.split(':')[0]}
                                    </span>
                                  )}
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      {/* Source Citations */}
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="pt-6 border-t border-slate-800/40 mt-4">
                          <h4 className="text-[9px] font-black text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                            <i className="fas fa-link text-[8px] text-blue-500/50"></i>
                            Grounded Knowledge Sources
                          </h4>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {msg.sources.map((src, idx) => (
                              <a 
                                key={idx} 
                                href={src.url} 
                                target="_blank" 
                                rel="noreferrer" 
                                className="flex items-center gap-3 px-4 py-3 bg-slate-900/60 hover:bg-slate-800 border border-slate-800 hover:border-blue-500/30 rounded-xl transition-all group overflow-hidden relative"
                              >
                                <div className="absolute inset-y-0 left-0 w-1 bg-blue-600/20 group-hover:bg-blue-600/60 transition-colors"></div>
                                <i className="fas fa-file-lines text-slate-500 group-hover:text-blue-400 text-sm transition-colors"></i>
                                <div className="flex flex-col min-w-0">
                                  <span className="text-[11px] text-slate-200 font-bold truncate pr-2">{src.title}</span>
                                  <div className="flex items-center gap-2 mt-0.5">
                                    <span className="text-[9px] text-emerald-500 font-mono uppercase font-black">Score: {(src.relevance_score * 100).toFixed(0)}%</span>
                                    <span className="text-[8px] text-slate-600 font-mono">ID: {src.arxiv_id}</span>
                                  </div>
                                </div>
                              </a>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {isTyping && (
          <div className="flex gap-4 items-center px-6 py-4 bg-slate-900/40 rounded-2xl rounded-tl-none border border-slate-800/50 animate-pulse w-fit">
            <div className="flex gap-1.5">
              <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce"></div>
              <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce [animation-delay:0.2s]"></div>
              <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce [animation-delay:0.4s]"></div>
            </div>
            <span className="text-[10px] text-slate-500 font-black uppercase tracking-[0.2em]">Executing Reasoning Loop...</span>
          </div>
        )}
      </div>

      {/* Input Section */}
      <div className="p-6 bg-slate-900/90 border-t border-slate-800/80 backdrop-blur-2xl">
        <div className="flex items-center gap-4 relative">
          <div className="absolute left-5 text-slate-600 pointer-events-none">
            <i className="fas fa-terminal text-[11px]"></i>
          </div>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Search internal docs or issue agent command..."
            className="flex-1 bg-slate-950 border border-slate-800 rounded-2xl pl-12 pr-6 py-4.5 text-sm focus:ring-1 focus:ring-blue-600/50 focus:border-blue-500/50 outline-none text-slate-200 placeholder:text-slate-700 transition-all shadow-inner"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isTyping}
            className="bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 text-white w-14 h-14 rounded-2xl flex items-center justify-center shadow-lg shadow-blue-600/10 active:scale-95 transition-all group relative overflow-hidden shrink-0"
          >
            <i className={`fas ${isTyping ? 'fa-circle-notch fa-spin' : 'fa-paper-plane'} text-lg group-hover:translate-x-0.5 transition-transform`}></i>
            <div className="absolute bottom-0 left-0 h-1 bg-white/20 w-full transform translate-y-full group-hover:translate-y-0 transition-transform"></div>
          </button>
        </div>
        <div className="mt-4 flex items-center justify-between px-2">
          <div className="flex gap-6">
             <div className="flex items-center gap-2 text-[10px] text-slate-600 font-bold uppercase tracking-widest">
               <i className="fas fa-fingerprint text-blue-900"></i> RSA-4096 Encrypted
             </div>
             <div className="flex items-center gap-2 text-[10px] text-slate-600 font-bold uppercase tracking-widest">
               <i className="fas fa-bolt text-emerald-900"></i> Hybrid Search On
             </div>
          </div>
          <div className="text-[10px] font-mono text-slate-700 tracking-tighter">
            NODE_ID: 0xFF-7A2
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
