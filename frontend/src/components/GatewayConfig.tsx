

import React, { useState, useEffect } from 'react';
import { VLLMEndpoint, ProviderKey } from '../types';

const GatewayConfig: React.FC = () => {
  const [endpoints, setEndpoints] = useState<VLLMEndpoint[]>([]);
  const [providerKeys, setProviderKeys] = useState<ProviderKey[]>([]);
  const [showAddForm, setShowAddForm] = useState(false);
  const [showKeyForm, setShowKeyForm] = useState(false);
  const [editingKey, setEditingKey] = useState<ProviderKey | null>(null);
  const [testingId, setTestingId] = useState<string | null>(null);
  
  const [newEndpoint, setNewEndpoint] = useState({ name: '', url: '', apiKey: '' });
  const [newKey, setNewKey] = useState({ provider: '', apiKey: '', rotationSchedule: 30, autoRotate: false });
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    fetchEndpoints();
    fetchKeys();
  }, []);

  const fetchEndpoints = async () => {
    try {
      const response = await fetch('/api/v1/gateway/vllm');
      if (response.ok) {
        const data = await response.json();
        setEndpoints(data);
      }
    } catch (err) {
      console.error("Failed to fetch endpoints:", err);
    }
  };

  const fetchKeys = async () => {
    try {
      const response = await fetch('/api/v1/gateway/keys');
      if (response.ok) {
        const data = await response.json();
        setProviderKeys(data);
      }
    } catch (err) {
      console.error("Failed to fetch keys:", err);
    }
  };

  const handleTestKey = async (id: string) => {
    setTestingId(id);
    try {
      const response = await fetch(`/api/v1/gateway/keys/${id}/test`, { method: 'POST' });
      const result = await response.json();
      if (response.ok) {
        fetchKeys();
        // Visual confirmation could be added here
      } else {
        alert(result.detail || "Connection test failed.");
      }
    } catch (err) {
      console.error("Test failed:", err);
    } finally {
      setTestingId(null);
    }
  };

  const handleAddKey = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newKey.provider || !newKey.apiKey) return;
    setIsLoading(true);
    try {
      const response = await fetch('/api/v1/gateway/keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newKey),
      });
      if (response.ok) {
        fetchKeys();
        setNewKey({ provider: '', apiKey: '', rotationSchedule: 30, autoRotate: false });
        setShowKeyForm(false);
      }
    } catch (err) {
      console.error("Failed to add key:", err);
    } finally {
      setIsLoading(false);
    }
  };

  // Fix: Implement missing handleAddEndpoint function
  const handleAddEndpoint = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newEndpoint.name || !newEndpoint.url) return;
    setIsLoading(true);
    try {
      const response = await fetch('/api/v1/gateway/vllm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newEndpoint),
      });
      if (response.ok) {
        fetchEndpoints();
        setNewEndpoint({ name: '', url: '', apiKey: '' });
        setShowAddForm(false);
      }
    } catch (err) {
      console.error("Failed to add endpoint:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRotateKey = async (id: string) => {
    try {
      const response = await fetch(`/api/v1/gateway/keys/${id}/rotate`, { method: 'POST' });
      if (response.ok) fetchKeys();
    } catch (err) {
      console.error("Failed to rotate key:", err);
    }
  };

  const handleUpdateSchedule = async (id: string, schedule: number, autoRotate: boolean) => {
    try {
      const response = await fetch(`/api/v1/gateway/keys/${id}/schedule`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rotationSchedule: schedule, autoRotate }),
      });
      if (response.ok) {
        fetchKeys();
        setEditingKey(null);
      }
    } catch (err) {
      console.error("Failed to update schedule:", err);
    }
  };

  const handleDeleteKey = async (id: string) => {
    if (!confirm('Are you sure you want to remove this provider key? This action is immediate and will disrupt active agent tasks.')) return;
    try {
      const response = await fetch(`/api/v1/gateway/keys/${id}`, { method: 'DELETE' });
      if (response.ok) fetchKeys();
    } catch (err) {
      console.error("Failed to delete key:", err);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Active': return 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.4)]';
      case 'Rotating': return 'bg-blue-500 animate-pulse';
      case 'Validation Failed': return 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.4)]';
      default: return 'bg-slate-500';
    }
  };

  return (
    <div className="space-y-8 max-w-6xl mx-auto pb-12 animate-fadeIn">
      <header className="flex justify-between items-end">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-3xl font-bold tracking-tight">Enterprise Gateway</h1>
            <span className="text-[9px] bg-slate-800 text-slate-400 border border-slate-700 px-2 py-0.5 rounded font-black uppercase tracking-widest">v2.4.0</span>
          </div>
          <p className="text-slate-400 mt-1">Manage inference clusters and secure API provider connectivity with automated lifecycle management.</p>
        </div>
        <div className="flex gap-2">
          <span className="text-[10px] bg-blue-600/10 text-blue-400 border border-blue-600/20 px-2 py-1 rounded font-bold uppercase tracking-widest flex items-center gap-2">
            <i className="fas fa-fingerprint"></i> Vault Encryption Active
          </span>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* API Key Management */}
        <section className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col min-h-[450px]">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-blue-600/10 rounded-xl flex items-center justify-center border border-blue-600/20">
                <i className="fas fa-key text-blue-500 text-lg"></i>
              </div>
              <div>
                <h3 className="text-lg font-bold">Cloud Provider Credentials</h3>
                <p className="text-xs text-slate-500">Secure storage and rotation policies.</p>
              </div>
            </div>
            {!showKeyForm && (
              <button 
                onClick={() => setShowKeyForm(true)}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-xl text-xs font-bold transition-all border border-slate-700 flex items-center gap-2"
              >
                <i className="fas fa-plus"></i> Add Key
              </button>
            )}
          </div>

          <div className="space-y-4 flex-1">
            {showKeyForm && (
              <form onSubmit={handleAddKey} className="bg-slate-800/80 border border-blue-600/30 p-5 rounded-xl space-y-4 mb-4 animate-slideIn">
                <div className="grid grid-cols-2 gap-4">
                  <div className="col-span-2">
                    <label className="text-[10px] font-black text-slate-500 uppercase ml-1 tracking-widest">Target Provider</label>
                    <input 
                      type="text" 
                      placeholder="e.g. OpenAI, Anthropic" 
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-sm focus:ring-1 focus:ring-blue-600 outline-none transition-all"
                      value={newKey.provider}
                      onChange={e => setNewKey({...newKey, provider: e.target.value})}
                    />
                  </div>
                  <div className="col-span-2">
                    <label className="text-[10px] font-black text-slate-500 uppercase ml-1 tracking-widest">Secret API Key</label>
                    <input 
                      type="password" 
                      placeholder="sk-..." 
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-sm focus:ring-1 focus:ring-blue-600 outline-none transition-all"
                      value={newKey.apiKey}
                      onChange={e => setNewKey({...newKey, apiKey: e.target.value})}
                    />
                  </div>
                  <div>
                    <label className="text-[10px] font-black text-slate-500 uppercase ml-1 tracking-widest">Rotation Frequency</label>
                    <select 
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-xs outline-none cursor-pointer"
                      value={newKey.rotationSchedule}
                      onChange={e => setNewKey({...newKey, rotationSchedule: parseInt(e.target.value)})}
                    >
                      <option value={0}>Manual (No Auto-Rotation)</option>
                      <option value={30}>Standard (30 Days)</option>
                      <option value={60}>Compliance (60 Days)</option>
                      <option value={90}>Extended (90 Days)</option>
                    </select>
                  </div>
                  <div className="flex items-center gap-3 pt-6 px-1">
                    <input 
                      type="checkbox" 
                      id="autoRotNew"
                      checked={newKey.autoRotate}
                      onChange={e => setNewKey({...newKey, autoRotate: e.target.checked})}
                      className="w-4 h-4 rounded border-slate-700 bg-slate-900 text-blue-600 focus:ring-offset-slate-900"
                    />
                    <label htmlFor="autoRotNew" className="text-xs font-bold text-slate-300 cursor-pointer">Enable Auto-Rotation</label>
                  </div>
                </div>
                <div className="flex justify-end gap-3 pt-2">
                  <button type="button" onClick={() => setShowKeyForm(false)} className="px-4 py-2 text-xs font-bold text-slate-400 hover:text-white">Discard</button>
                  <button type="submit" disabled={isLoading} className="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-xs font-black uppercase tracking-widest disabled:opacity-50 shadow-lg shadow-blue-600/20">Commit Key</button>
                </div>
              </form>
            )}

            {providerKeys.map((pk) => (
              <div key={pk.id} className="relative p-5 rounded-xl bg-slate-800/30 border border-slate-800/50 hover:border-slate-700 transition-all group overflow-hidden">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className={`w-2.5 h-2.5 rounded-full ${getStatusColor(pk.status)}`}></div>
                    <span className="text-base font-bold text-slate-100">{pk.provider}</span>
                    <span className={`text-[8px] font-black uppercase tracking-widest px-2 py-0.5 rounded border ${
                      pk.status === 'Validation Failed' ? 'bg-rose-500/10 text-rose-500 border-rose-500/20' : 'bg-slate-900 text-slate-500 border-slate-800'
                    }`}>
                      {pk.status}
                    </span>
                  </div>
                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-all">
                    <button onClick={() => handleTestKey(pk.id)} className="p-2 text-slate-400 hover:text-emerald-400 transition-colors" title="Test Connection">
                      <i className={`fas ${testingId === pk.id ? 'fa-spinner fa-spin' : 'fa-flask'} text-xs`}></i>
                    </button>
                    <button onClick={() => setEditingKey(pk)} className="p-2 text-slate-400 hover:text-blue-400 transition-colors" title="Rotation Policy">
                      <i className="fas fa-cog text-xs"></i>
                    </button>
                    <button onClick={() => handleRotateKey(pk.id)} className="p-2 text-slate-400 hover:text-blue-400 transition-colors" title="Immediate Rotation">
                      <i className="fas fa-sync-alt text-xs"></i>
                    </button>
                    <button onClick={() => handleDeleteKey(pk.id)} className="p-2 text-slate-400 hover:text-rose-400 transition-colors" title="Revoke Access">
                      <i className="fas fa-trash-alt text-xs"></i>
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                  <div className="col-span-2">
                    <label className="text-[9px] text-slate-500 uppercase font-black tracking-widest">Encrypted Reference</label>
                    <p className="text-[11px] text-slate-400 font-mono mt-1 tracking-tighter">{pk.apiKey}</p>
                  </div>
                  <div className="text-right col-span-2">
                    <label className="text-[9px] text-slate-500 uppercase font-black tracking-widest">Policy Schedule</label>
                    <p className={`text-[10px] font-bold mt-1 ${pk.autoRotate ? 'text-blue-400' : 'text-slate-500'}`}>
                      {pk.autoRotate ? <><i className="fas fa-robot mr-1"></i> Auto: {pk.nextRotation}</> : 'Manual Only'}
                    </p>
                  </div>
                </div>
                
                {/* Telemetry Footer */}
                <div className="flex items-center gap-4 pt-3 border-t border-slate-800/50">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[9px] text-slate-600 font-black uppercase">Tokens:</span>
                    <span className="text-[10px] text-slate-300 font-mono font-bold">{(pk as any).usage?.tokens || '0'}</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-[9px] text-slate-600 font-black uppercase">Health:</span>
                    <span className="text-[10px] text-emerald-500 font-mono font-bold">{(pk as any).usage?.success_rate || '100%'}</span>
                  </div>
                </div>

                {pk.status === 'Rotating' && (
                  <div className="absolute inset-0 bg-slate-900/80 backdrop-blur-sm flex items-center justify-center gap-3 z-10 animate-fadeIn">
                    <i className="fas fa-spinner fa-spin text-blue-500"></i>
                    <span className="text-xs font-black text-blue-400 uppercase tracking-[0.2em]">Executing Rotation...</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>

        {/* vLLM Management */}
        <section className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col min-h-[450px]">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-purple-600/10 rounded-xl flex items-center justify-center border border-purple-600/20">
                <i className="fas fa-server text-purple-500 text-lg"></i>
              </div>
              <div>
                <h3 className="text-lg font-bold">Private Inference Clusters</h3>
                <p className="text-xs text-slate-500">Local vLLM and HuggingFace endpoints.</p>
              </div>
            </div>
            {!showAddForm && (
              <button 
                onClick={() => setShowAddForm(true)}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-xl text-xs font-bold transition-all border border-slate-700 flex items-center gap-2"
              >
                <i className="fas fa-plus"></i> Add Cluster
              </button>
            )}
          </div>

          <div className="space-y-4 flex-1">
            {showAddForm && (
              <form onSubmit={handleAddEndpoint} className="bg-slate-800/80 border border-purple-600/30 p-5 rounded-xl space-y-4 mb-4 animate-slideIn">
                <div className="grid grid-cols-2 gap-3">
                  <div className="col-span-2">
                    <label className="text-[10px] font-black text-slate-500 uppercase ml-1 tracking-widest">Instance Label</label>
                    <input 
                      type="text" 
                      placeholder="e.g. Cluster B - Llama3" 
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-sm focus:ring-1 focus:ring-purple-600 outline-none transition-all"
                      value={newEndpoint.name}
                      onChange={e => setNewEndpoint({...newEndpoint, name: e.target.value})}
                    />
                  </div>
                  <div>
                    <label className="text-[10px] font-black text-slate-500 uppercase ml-1 tracking-widest">Base API URL</label>
                    <input 
                      type="text" 
                      placeholder="https://vllm.internal" 
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-sm focus:ring-1 focus:ring-purple-600 outline-none transition-all"
                      value={newEndpoint.url}
                      onChange={e => setNewEndpoint({...newEndpoint, url: e.target.value})}
                    />
                  </div>
                  <div>
                    <label className="text-[10px] font-black text-slate-500 uppercase ml-1 tracking-widest">Local Auth Token</label>
                    <input 
                      type="password" 
                      placeholder="••••••••" 
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-sm focus:ring-1 focus:ring-purple-600 outline-none transition-all"
                      value={newEndpoint.apiKey}
                      onChange={e => setNewEndpoint({...newEndpoint, apiKey: e.target.value})}
                    />
                  </div>
                </div>
                <div className="flex justify-end gap-3 pt-2">
                  <button type="button" onClick={() => setShowAddForm(false)} className="px-4 py-2 text-xs font-bold text-slate-400 hover:text-white">Cancel</button>
                  <button type="submit" disabled={isLoading} className="px-6 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-xl text-xs font-black uppercase tracking-widest shadow-lg shadow-purple-600/20 transition-all">Register Cluster</button>
                </div>
              </form>
            )}

            {endpoints.map((ep) => (
              <div key={ep.id} className="group p-5 bg-slate-800/40 border border-slate-700/50 rounded-xl hover:border-slate-600 transition-all shadow-sm">
                <div className="flex justify-between items-start">
                  <div className="space-y-2">
                    <div className="flex items-center gap-3">
                      <span className="font-bold text-base text-slate-100">{ep.name}</span>
                      <span className={`text-[8px] px-2 py-0.5 rounded-full font-black uppercase tracking-widest border ${
                        ep.status === 'UP' ? 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20' : 'bg-rose-500/10 text-rose-500 border-rose-500/20'
                      }`}>
                        {ep.status}
                      </span>
                    </div>
                    <code className="text-[10px] text-slate-400 font-mono block truncate max-w-[280px] bg-slate-900 px-2 py-1 rounded">
                      {ep.url}
                    </code>
                  </div>
                  <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-all">
                    <button className="p-2 text-slate-500 hover:text-white transition-colors">
                      <i className="fas fa-edit text-xs"></i>
                    </button>
                    <button className="p-2 text-slate-500 hover:text-rose-400 transition-colors">
                      <i className="fas fa-trash-alt text-xs"></i>
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>

      {/* Rotation Settings Modal */}
      {editingKey && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-xl p-4 animate-fadeIn">
          <div className="bg-slate-900 border border-slate-800 w-full max-w-md rounded-2xl shadow-2xl p-8 animate-slideIn">
            <div className="flex items-center gap-5 mb-8">
              <div className="w-14 h-14 bg-blue-600/10 rounded-2xl flex items-center justify-center border border-blue-600/20">
                <i className="fas fa-clock-rotate-left text-blue-500 text-2xl"></i>
              </div>
              <div>
                <h2 className="text-xl font-bold tracking-tight">{editingKey.provider} Lifecycle Policy</h2>
                <p className="text-xs text-slate-500">Configure automated credential rotation.</p>
              </div>
            </div>

            <div className="space-y-6">
              <div className="space-y-4 p-5 bg-slate-800/40 rounded-2xl border border-slate-800">
                <label className="text-[10px] font-black text-slate-500 uppercase block tracking-widest">Rotation Threshold</label>
                <div className="grid grid-cols-2 gap-3">
                  {[0, 30, 60, 90].map(days => (
                    <button
                      key={days}
                      onClick={() => setEditingKey({...editingKey, rotationSchedule: days})}
                      className={`px-4 py-2.5 rounded-xl text-xs font-black uppercase tracking-widest transition-all border ${
                        editingKey.rotationSchedule === days 
                          ? 'bg-blue-600 text-white border-blue-600 shadow-lg shadow-blue-600/30' 
                          : 'bg-slate-900 text-slate-400 border-slate-700 hover:border-slate-500'
                      }`}
                    >
                      {days === 0 ? 'Manual' : `${days} Days`}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex items-center justify-between p-5 bg-slate-800/40 rounded-2xl border border-slate-800">
                <div className="space-y-1">
                  <h4 className="text-sm font-bold">Autonomous Rotation</h4>
                  <p className="text-[10px] text-slate-500 italic">Allow background task orchestration.</p>
                </div>
                <button 
                  onClick={() => setEditingKey({...editingKey, autoRotate: !editingKey.autoRotate})}
                  className={`relative w-12 h-6 transition-all rounded-full outline-none focus:ring-2 focus:ring-blue-600/50 ${
                    editingKey.autoRotate ? 'bg-blue-600 shadow-[0_0_12px_rgba(37,99,235,0.4)]' : 'bg-slate-700'
                  }`}
                >
                  <div className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform duration-200 ${
                    editingKey.autoRotate ? 'translate-x-6' : ''
                  }`}></div>
                </button>
              </div>
            </div>

            <div className="mt-10 flex justify-end gap-4">
              <button 
                onClick={() => setEditingKey(null)}
                className="px-5 py-2.5 text-xs font-black uppercase tracking-widest text-slate-500 hover:text-white transition-colors"
              >
                Discard
              </button>
              <button 
                onClick={() => handleUpdateSchedule(editingKey.id, editingKey.rotationSchedule!, editingKey.autoRotate)}
                className="bg-blue-600 hover:bg-blue-500 text-white px-8 py-2.5 rounded-xl font-black uppercase tracking-widest transition-all shadow-lg shadow-blue-600/30"
              >
                Commit Policy
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="mt-8 p-6 bg-slate-900/60 border border-slate-800 rounded-2xl border-l-4 border-l-blue-600 shadow-xl flex items-center gap-8 backdrop-blur-sm">
        <div className="w-16 h-16 bg-blue-600/10 rounded-full flex items-center justify-center shrink-0 border border-blue-600/20">
          <i className="fas fa-shield-halved text-blue-500 text-2xl"></i>
        </div>
        <div>
          <h4 className="font-bold text-lg tracking-tight">Enterprise Compliance Vault</h4>
          <p className="text-sm text-slate-400 mt-1 max-w-4xl leading-relaxed">
            API keys are managed using a hybrid HSM (Hardware Security Module) logic. Automated rotation policies ensure compliance with SOC2 and ISO27001 credential management standards. 
            All rotation and access events are indexed for real-time security auditing.
          </p>
        </div>
      </div>
    </div>
  );
};

export default GatewayConfig;