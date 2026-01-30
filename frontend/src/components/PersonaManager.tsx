
import React, { useState, useEffect } from 'react';
import { Persona } from '../types';

const PersonaManager: React.FC = () => {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [showAddModal, setShowAddModal] = useState(false);
  const [newPersona, setNewPersona] = useState<Partial<Persona>>({
    name: '',
    systemPrompt: '',
    temperature: 0.1,
    allowedTools: ['Retrieval']
  });

  useEffect(() => {
    fetchPersonas();
  }, []);

  const fetchPersonas = async () => {
    try {
      const response = await fetch('/api/v1/personas');
      if (response.ok) {
        const data = await response.json();
        setPersonas(data);
      }
    } catch (e) {
      console.error("Failed to fetch personas", e);
    }
  };

  const handleAddPersona = async () => {
    try {
      const response = await fetch('/api/v1/personas', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newPersona)
      });
      if (response.ok) {
        setShowAddModal(false);
        fetchPersonas();
        setNewPersona({ name: '', systemPrompt: '', temperature: 0.1, allowedTools: ['Retrieval'] });
      }
    } catch (e) {
      console.error("Failed to add persona", e);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Personas</h1>
          <p className="text-slate-400 mt-1">Configure specialized agent roles.</p>
        </div>
        <button 
          onClick={() => setShowAddModal(true)}
          className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-xl flex items-center gap-2 transition-all shadow-lg shadow-blue-900/20"
        >
          <i className="fas fa-plus"></i> New Persona
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {personas.map(persona => (
          <div key={persona.id} className="bg-slate-900 border border-slate-800 rounded-2xl p-6 hover:border-blue-600/30 transition-all group">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold">{persona.name}</h3>
              <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <button className="p-2 text-slate-400 hover:text-white"><i className="fas fa-edit"></i></button>
                <button className="p-2 text-red-400 hover:text-red-300"><i className="fas fa-trash"></i></button>
              </div>
            </div>
            
            <div className="space-y-4">
              <div>
                <span className="text-xs font-bold text-slate-500 uppercase">System Prompt</span>
                <p className="text-sm text-slate-300 line-clamp-2 mt-1 italic">"{persona.systemPrompt}"</p>
              </div>

              <div className="flex items-center gap-4">
                <div>
                  <span className="text-xs font-bold text-slate-500 uppercase">Temp</span>
                  <div className="text-sm font-medium">{persona.temperature}</div>
                </div>
                <div className="flex-1">
                  <span className="text-xs font-bold text-slate-500 uppercase">Tools</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {persona.allowedTools.map(tool => (
                      <span key={tool} className="text-[10px] px-2 py-0.5 bg-blue-600/10 text-blue-400 border border-blue-600/20 rounded-full">
                        {tool}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {showAddModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-800 w-full max-w-lg rounded-2xl shadow-2xl p-8 animate-fadeIn">
            <h2 className="text-2xl font-bold mb-6">Create New Persona</h2>
            <div className="space-y-4">
              <div>
                <label className="text-xs font-bold text-slate-500 uppercase block mb-1">Name</label>
                <input 
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 outline-none focus:ring-1 focus:ring-blue-600"
                  value={newPersona.name}
                  onChange={e => setNewPersona({...newPersona, name: e.target.value})}
                />
              </div>
              <div>
                <label className="text-xs font-bold text-slate-500 uppercase block mb-1">System Prompt</label>
                <textarea 
                  rows={3}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 outline-none focus:ring-1 focus:ring-blue-600"
                  value={newPersona.systemPrompt}
                  onChange={e => setNewPersona({...newPersona, systemPrompt: e.target.value})}
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-bold text-slate-500 uppercase block mb-1">Temperature</label>
                  <input 
                    type="number" step="0.1" max="1" min="0"
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 outline-none"
                    value={newPersona.temperature}
                    onChange={e => setNewPersona({...newPersona, temperature: parseFloat(e.target.value)})}
                  />
                </div>
                <div>
                  <label className="text-xs font-bold text-slate-500 uppercase block mb-1">Primary Tool</label>
                  <select className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 outline-none">
                    <option>Retrieval</option>
                    <option>Search</option>
                  </select>
                </div>
              </div>
            </div>
            <div className="mt-8 flex justify-end gap-3">
              <button 
                onClick={() => setShowAddModal(false)}
                className="px-4 py-2 text-slate-400 hover:text-white"
              >
                Cancel
              </button>
              <button 
                onClick={handleAddPersona}
                className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-2 rounded-xl font-bold transition-all"
              >
                Create Persona
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PersonaManager;
