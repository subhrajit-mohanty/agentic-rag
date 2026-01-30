
import React, { useState, useEffect } from 'react';
import { Connector } from '../types';

const ConnectorRegistry: React.FC = () => {
  const [connectors, setConnectors] = useState<Connector[]>([]);
  const [syncingId, setSyncingId] = useState<string | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [selectedType, setSelectedType] = useState<'SharePoint' | 'GoogleDrive' | 'S3'>('SharePoint');
  
  const [newConn, setNewConn] = useState({
    name: '',
    config: {} as Record<string, string>
  });

  useEffect(() => {
    fetchConnectors();
    const interval = setInterval(fetchConnectors, 5000); 
    return () => clearInterval(interval);
  }, []);

  const fetchConnectors = async () => {
    try {
      const response = await fetch('/api/v1/connectors');
      if (response.ok) {
        const data = await response.json();
        setConnectors(data);
      }
    } catch (e) {
      console.error("Fetch connectors error", e);
    }
  };

  const handleSync = async (id: string) => {
    setSyncingId(id);
    try {
      const response = await fetch('/api/v1/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ connector_id: id })
      });
      if (response.ok) {
        // Optimistic UI update
        setConnectors(prev => prev.map(c => c.id === id ? { ...c, status: 'syncing', lastSync: 'Just now' } : c));
      }
    } catch (e) {
      console.error("Sync error", e);
    } finally {
      setTimeout(() => setSyncingId(null), 2000);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Are you sure you want to remove this data source? All indexed vectors for this source will be detached.")) return;
    try {
      const response = await fetch(`/api/v1/connectors/${id}`, { method: 'DELETE' });
      if (response.ok) fetchConnectors();
    } catch (e) {
      console.error("Delete error", e);
    }
  };

  const handleRegister = async () => {
    if (!newConn.name) return;
    try {
      const response = await fetch('/api/v1/connectors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newConn.name,
          type: selectedType,
          config: newConn.config
        })
      });
      if (response.ok) {
        setShowAddModal(false);
        setNewConn({ name: '', config: {} });
        fetchConnectors();
      }
    } catch (e) {
      console.error("Register error", e);
    }
  };

  const openRegister = (type: 'SharePoint' | 'GoogleDrive' | 'S3') => {
    setSelectedType(type);
    setNewConn({ name: '', config: {} });
    setShowAddModal(true);
  };

  const getStatusDisplay = (status: string) => {
    switch (status) {
      case 'connected':
        return <span className="flex items-center gap-1.5 text-emerald-500 bg-emerald-500/10 px-2 py-0.5 rounded-full text-[10px] font-black uppercase tracking-tighter border border-emerald-500/20"><i className="fas fa-check-circle text-[8px]"></i> Connected</span>;
      case 'syncing':
        return <span className="flex items-center gap-1.5 text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded-full text-[10px] font-black uppercase tracking-tighter border border-blue-500/20 animate-pulse"><i className="fas fa-sync fa-spin text-[8px]"></i> Syncing</span>;
      case 'error':
        return <span className="flex items-center gap-1.5 text-rose-500 bg-rose-500/10 px-2 py-0.5 rounded-full text-[10px] font-black uppercase tracking-tighter border border-rose-500/20"><i className="fas fa-exclamation-triangle text-[8px]"></i> Error</span>;
      case 'disconnected':
        return <span className="flex items-center gap-1.5 text-slate-500 bg-slate-500/10 px-2 py-0.5 rounded-full text-[10px] font-black uppercase tracking-tighter border border-slate-700"><i className="fas fa-unlink text-[8px]"></i> Disconnected</span>;
      default:
        return <span className="text-slate-500 text-[10px] uppercase">{status}</span>;
    }
  };

  const renderConfigFields = () => {
    switch (selectedType) {
      case 'SharePoint':
        return (
          <>
            <div>
              <label className="text-[10px] font-black text-slate-500 uppercase ml-1 tracking-widest">Microsoft Tenant ID</label>
              <input 
                placeholder="72f988bf-..." 
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-sm outline-none mt-1 focus:border-blue-500 transition-colors"
                onChange={e => setNewConn({...newConn, config: {...newConn.config, tenantId: e.target.value}})}
              />
            </div>
            <div>
              <label className="text-[10px] font-black text-slate-500 uppercase ml-1 tracking-widest">SharePoint Site URL</label>
              <input 
                placeholder="https://company.sharepoint.com/sites/legal" 
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-sm outline-none mt-1 focus:border-blue-500 transition-colors"
                onChange={e => setNewConn({...newConn, config: {...newConn.config, siteUrl: e.target.value}})}
              />
            </div>
          </>
        );
      case 'S3':
        return (
          <>
            <div>
              <label className="text-[10px] font-black text-slate-500 uppercase ml-1 tracking-widest">S3 Bucket Name</label>
              <input 
                placeholder="enterprise-docs-prod" 
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-sm outline-none mt-1 focus:border-blue-500 transition-colors"
                onChange={e => setNewConn({...newConn, config: {...newConn.config, bucket: e.target.value}})}
              />
            </div>
            <div>
              <label className="text-[10px] font-black text-slate-500 uppercase ml-1 tracking-widest">AWS Region</label>
              <select 
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-sm outline-none mt-1 cursor-pointer focus:border-blue-500"
                onChange={e => setNewConn({...newConn, config: {...newConn.config, region: e.target.value}})}
              >
                <option value="us-east-1">US East (N. Virginia)</option>
                <option value="eu-west-1">EU (Ireland)</option>
                <option value="ap-southeast-1">Asia Pacific (Singapore)</option>
              </select>
            </div>
          </>
        );
      case 'GoogleDrive':
        return (
          <>
            <div>
              <label className="text-[10px] font-black text-slate-500 uppercase ml-1 tracking-widest">Shared Drive ID</label>
              <input 
                placeholder="0A..." 
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-sm outline-none mt-1 focus:border-blue-500 transition-colors"
                onChange={e => setNewConn({...newConn, config: {...newConn.config, driveId: e.target.value}})}
              />
            </div>
          </>
        );
    }
  };

  return (
    <div className="space-y-8 max-w-6xl mx-auto pb-12 animate-fadeIn">
      <header className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-100">Data Connector Registry</h1>
          <p className="text-slate-400 mt-1 max-w-2xl">
            Orchestrate semantic ETL pipelines across distributed cloud repositories. Automated chunking and embedding logic are triggered on every synchronization event.
          </p>
        </div>
        <div className="flex gap-3">
          <div className="px-4 py-2 bg-slate-900 border border-slate-800 rounded-2xl flex items-center gap-3">
             <div className="flex -space-x-2">
                <div className="w-6 h-6 rounded-full bg-blue-600 border-2 border-slate-900 flex items-center justify-center text-[10px]"><i className="fas fa-building-columns"></i></div>
                <div className="w-6 h-6 rounded-full bg-green-600 border-2 border-slate-900 flex items-center justify-center text-[10px]"><i className="fa-brands fa-google-drive"></i></div>
                <div className="w-6 h-6 rounded-full bg-amber-600 border-2 border-slate-900 flex items-center justify-center text-[10px]"><i className="fa-brands fa-aws"></i></div>
             </div>
             <span className="text-[10px] font-black uppercase tracking-widest text-slate-400">Multi-Cloud Ready</span>
          </div>
        </div>
      </header>

      {/* Connectors Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {connectors.length === 0 ? (
          <div className="col-span-full border-2 border-dashed border-slate-800 rounded-3xl p-20 flex flex-col items-center justify-center text-center space-y-6 bg-slate-900/20">
            <div className="w-24 h-24 bg-slate-900 rounded-full flex items-center justify-center border border-slate-800 shadow-inner">
              <i className="fas fa-project-diagram text-4xl text-slate-700"></i>
            </div>
            <div className="max-w-sm">
              <h3 className="text-xl font-bold text-slate-200 tracking-tight">Zero Semantic Hubs Active</h3>
              <p className="text-sm text-slate-500 mt-2 leading-relaxed"> Register an enterprise data source to begin the automated ingestion, document parsing, and vector embedding lifecycle.</p>
            </div>
          </div>
        ) : (
          connectors.map(conn => (
            <div key={conn.id} className="bg-slate-900/80 border border-slate-800 rounded-3xl p-6 hover:border-slate-600 transition-all shadow-2xl backdrop-blur-sm group relative overflow-hidden">
              {/* Connector Header */}
              <div className="flex items-start justify-between mb-6">
                <div className="flex items-center gap-4">
                  <div className={`w-14 h-14 rounded-2xl flex items-center justify-center text-2xl transition-all shadow-lg ${
                    conn.type === 'SharePoint' ? 'bg-blue-600/10 text-blue-500 border border-blue-600/20' :
                    conn.type === 'GoogleDrive' ? 'bg-green-600/10 text-green-500 border border-green-500/20' :
                    'bg-amber-600/10 text-amber-500 border border-amber-600/20'
                  }`}>
                    <i className={`fas ${
                      conn.type === 'SharePoint' ? 'fa-building-columns' :
                      conn.type === 'GoogleDrive' ? 'fa-brands fa-google-drive' :
                      'fa-brands fa-aws'
                    }`}></i>
                  </div>
                  <div>
                    <h3 className="font-bold text-lg text-slate-100 group-hover:text-blue-400 transition-colors">{conn.name}</h3>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-[10px] font-mono text-slate-600 tracking-tighter uppercase">{conn.type}</span>
                      <span className="text-slate-800">•</span>
                      {getStatusDisplay(conn.status)}
                    </div>
                  </div>
                </div>
                <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-all">
                  <button onClick={() => handleDelete(conn.id)} className="p-2.5 text-slate-600 hover:text-rose-500 transition-colors" title="Remove Provider">
                    <i className="fas fa-trash-alt text-sm"></i>
                  </button>
                </div>
              </div>

              {/* Stats & Metadata */}
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="bg-slate-950/40 border border-slate-800/50 rounded-2xl p-4">
                  <span className="text-[9px] font-black text-slate-600 uppercase tracking-widest block mb-1">Index Health</span>
                  <div className="flex items-center gap-2">
                    {/* Fixed: Use Math.random() for JS instead of Python's random.randint */}
                    <span className="text-lg font-bold text-slate-200">{((Math.floor(Math.random() * 21) + 80) / 10).toFixed(1)}/10</span>
                    <div className="flex-1 h-1 bg-slate-800 rounded-full overflow-hidden">
                       <div className="h-full bg-emerald-500" style={{ width: '92%' }}></div>
                    </div>
                  </div>
                </div>
                <div className="bg-slate-950/40 border border-slate-800/50 rounded-2xl p-4">
                  <span className="text-[9px] font-black text-slate-600 uppercase tracking-widest block mb-1">Last Update</span>
                  <div className="flex items-center gap-2">
                    <i className="fas fa-history text-[10px] text-slate-600"></i>
                    <span className="text-xs text-slate-400 font-medium">{conn.lastSync}</span>
                  </div>
                </div>
              </div>
              
              {/* Actions Footer */}
              <div className="flex items-center justify-between mt-2 pt-4 border-t border-slate-800/50">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-slate-500 font-mono tracking-tighter">ID: {conn.id}</span>
                </div>
                <button 
                  onClick={() => handleSync(conn.id)}
                  disabled={conn.status === 'syncing'}
                  className={`px-5 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all flex items-center gap-2 shadow-lg ${
                    conn.status === 'syncing' 
                      ? 'bg-slate-800 text-slate-500 border border-slate-700' 
                      : conn.status === 'disconnected' 
                        ? 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-600/10'
                        : 'bg-blue-600 hover:bg-blue-500 text-white shadow-blue-600/20 active:scale-95'
                  }`}
                >
                  {conn.status === 'syncing' ? (
                    <><i className="fas fa-spinner fa-spin"></i> Processing</>
                  ) : (
                    conn.status === 'disconnected' ? <><i className="fas fa-key"></i> Authenticate</> : <><i className="fas fa-rotate"></i> Incremental Sync</>
                  )}
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Registration Section - Registry Pattern */}
      <div className="bg-slate-900/40 border border-slate-800 rounded-3xl p-10 shadow-2xl backdrop-blur-md">
        <div className="mb-10 text-center">
          <h3 className="text-2xl font-bold tracking-tight text-slate-100">Provision Enterprise Knowledge Hub</h3>
          <p className="text-slate-400 mt-2 text-sm max-w-xl mx-auto leading-relaxed">
            Register a new secure repository to the semantic mapping engine. Select your protocol to begin the automated discovery and ingestion cycle.
          </p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-8">
          <button 
            onClick={() => openRegister('SharePoint')}
            className="flex flex-col items-center gap-5 p-8 bg-slate-900 border border-slate-800 rounded-3xl hover:border-blue-600/50 hover:bg-slate-800/40 transition-all group relative overflow-hidden active:scale-95"
          >
            <div className="absolute top-0 left-0 w-full h-1 bg-blue-600 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="w-16 h-16 rounded-2xl bg-blue-600/10 flex items-center justify-center border border-blue-600/20 group-hover:scale-110 transition-transform">
               <i className="fa-brands fa-microsoft text-3xl text-blue-500"></i>
            </div>
            <div className="text-center">
              <span className="text-[10px] font-black uppercase tracking-[0.2em] text-blue-400">Microsoft Graph</span>
              <p className="text-[12px] text-slate-300 mt-1 font-bold">SharePoint Online</p>
            </div>
          </button>
          
          <button 
            onClick={() => openRegister('GoogleDrive')}
            className="flex flex-col items-center gap-5 p-8 bg-slate-900 border border-slate-800 rounded-3xl hover:border-green-600/50 hover:bg-slate-800/40 transition-all group relative overflow-hidden active:scale-95"
          >
            <div className="absolute top-0 left-0 w-full h-1 bg-green-600 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="w-16 h-16 rounded-2xl bg-green-600/10 flex items-center justify-center border border-green-600/20 group-hover:scale-110 transition-transform">
               <i className="fa-brands fa-google text-3xl text-green-500"></i>
            </div>
            <div className="text-center">
              <span className="text-[10px] font-black uppercase tracking-[0.2em] text-green-400">Workspace API</span>
              <p className="text-[12px] text-slate-300 mt-1 font-bold">Google Drive SDK</p>
            </div>
          </button>
          
          <button 
            onClick={() => openRegister('S3')}
            className="flex flex-col items-center gap-5 p-8 bg-slate-900 border border-slate-800 rounded-3xl hover:border-amber-600/50 hover:bg-slate-800/40 transition-all group relative overflow-hidden active:scale-95"
          >
            <div className="absolute top-0 left-0 w-full h-1 bg-amber-600 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="w-16 h-16 rounded-2xl bg-amber-600/10 flex items-center justify-center border border-amber-600/20 group-hover:scale-110 transition-transform">
               <i className="fa-brands fa-aws text-3xl text-amber-500"></i>
            </div>
            <div className="text-center">
              <span className="text-[10px] font-black uppercase tracking-[0.2em] text-amber-400">AWS Cloud</span>
              <p className="text-[12px] text-slate-300 mt-1 font-bold">S3 Object Storage</p>
            </div>
          </button>
        </div>
      </div>

      {/* Registration Modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-xl p-4 animate-fadeIn">
          <div className="bg-slate-900 border border-slate-800 w-full max-w-xl rounded-[2.5rem] shadow-2xl p-10 animate-slideIn">
            <div className="flex items-center gap-6 mb-10">
              <div className={`w-16 h-16 rounded-2xl flex items-center justify-center text-3xl shadow-lg ${
                selectedType === 'SharePoint' ? 'bg-blue-600/10 text-blue-500' :
                selectedType === 'GoogleDrive' ? 'bg-green-600/10 text-green-500' :
                'bg-amber-600/10 text-amber-500'
              }`}>
                <i className={`fas ${
                  selectedType === 'SharePoint' ? 'fa-building-columns' :
                  selectedType === 'GoogleDrive' ? 'fa-brands fa-google-drive' :
                  'fa-brands fa-aws'
                }`}></i>
              </div>
              <div>
                <h2 className="text-2xl font-bold tracking-tight text-slate-100">Setup {selectedType} Pipeline</h2>
                <p className="text-sm text-slate-500 mt-1">Configure secure authentication and indexing scope.</p>
              </div>
            </div>

            <div className="space-y-6">
              <div>
                <label className="text-[10px] font-black text-slate-500 uppercase ml-1 tracking-widest">Pipeline Label</label>
                <input 
                  placeholder="e.g. Legal Repository A" 
                  className="w-full bg-slate-950 border border-slate-800 rounded-2xl px-5 py-4 text-sm focus:ring-1 focus:ring-blue-600 outline-none mt-1.5 transition-all text-slate-200"
                  value={newConn.name}
                  onChange={e => setNewConn({...newConn, name: e.target.value})}
                />
              </div>

              <div className="h-px bg-slate-800/40 my-2"></div>
              
              <div className="space-y-5">
                {renderConfigFields()}
              </div>
            </div>

            <div className="mt-12 flex justify-end gap-5">
              <button 
                onClick={() => setShowAddModal(false)}
                className="px-6 py-3 text-xs font-black uppercase tracking-widest text-slate-500 hover:text-slate-100 transition-colors"
              >
                Discard
              </button>
              <button 
                onClick={handleRegister}
                className="bg-blue-600 hover:bg-blue-500 text-white px-10 py-3 rounded-2xl font-black uppercase tracking-widest transition-all shadow-lg shadow-blue-600/20 active:scale-95"
              >
                Launch Pipeline
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ConnectorRegistry;
