
import React from 'react';
import { NavLink } from 'react-router-dom';

interface SidebarProps {
  isOpen: boolean;
  toggle: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ isOpen, toggle }) => {
  const navItems = [
    { path: '/', icon: 'fa-chart-line', label: 'Dashboard' },
    { path: '/chat', icon: 'fa-comments', label: 'Agent Chat' },
    { path: '/personas', icon: 'fa-user-gear', label: 'Personas' },
    { path: '/connectors', icon: 'fa-link', label: 'Connectors' },
    { path: '/gateway', icon: 'fa-shield-halved', label: 'Gateway' },
  ];

  return (
    <aside 
      className={`fixed top-0 left-0 h-full bg-slate-900 border-r border-slate-800 transition-all duration-300 z-50 flex flex-col ${
        isOpen ? 'w-64' : 'w-16'
      }`}
    >
      <div className="p-4 flex items-center justify-between border-b border-slate-800">
        <div className={`flex items-center gap-3 overflow-hidden ${!isOpen && 'hidden'}`}>
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center font-bold text-white">R</div>
          <span className="font-bold text-lg whitespace-nowrap">AgenticRAG</span>
        </div>
        <button onClick={toggle} className="p-2 text-slate-400 hover:text-white transition-colors">
          <i className={`fas ${isOpen ? 'fa-chevron-left' : 'fa-bars'}`}></i>
        </button>
      </div>

      <nav className="flex-1 mt-6 px-2 space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `flex items-center gap-4 px-3 py-2.5 rounded-lg transition-all ${
                isActive 
                  ? 'bg-blue-600/10 text-blue-400 border border-blue-600/20' 
                  : 'text-slate-400 hover:bg-slate-800 hover:text-white'
              }`
            }
          >
            <i className={`fas ${item.icon} w-5 text-center`}></i>
            {isOpen && <span className="font-medium whitespace-nowrap">{item.label}</span>}
          </NavLink>
        ))}
      </nav>

      <div className="p-4 border-t border-slate-800">
        <div className={`flex items-center gap-3 ${!isOpen && 'justify-center'}`}>
          <div className="w-8 h-8 rounded-full bg-slate-700"></div>
          {isOpen && (
            <div className="flex flex-col">
              <span className="text-sm font-medium">Enterprise Admin</span>
              <span className="text-xs text-slate-500">Tier: Platinum</span>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
