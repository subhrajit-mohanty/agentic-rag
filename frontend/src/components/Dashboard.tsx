
import React, { useState, useEffect } from 'react';

interface Stats {
  total_vectors: string;
  agent_tasks: string;
  active_connectors: string;
  avg_latency: string;
  health: Record<string, string>;
}

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch('/api/v1/stats');
        if (response.ok) {
          const data = await response.json();
          setStats(data);
        }
      } catch (e) {
        console.error("Stats fetch error", e);
      }
    };
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  const statCards = [
    { label: 'Total Vectors', value: stats?.total_vectors || '...', icon: 'fa-database', color: 'text-blue-500' },
    { label: 'Agent Tasks', value: stats?.agent_tasks || '...', icon: 'fa-microchip', color: 'text-purple-500' },
    { label: 'Active Connectors', value: stats?.active_connectors || '...', icon: 'fa-plug', color: 'text-green-500' },
    { label: 'Avg Latency', value: stats?.avg_latency || '...', icon: 'fa-bolt', color: 'text-amber-500' },
  ];

  const recentActivity = [
    { event: 'S3 Sync Completed', time: '2 mins ago', status: 'Success' },
    { event: 'Milvus Collection Optimization', time: '15 mins ago', status: 'Processing' },
    { event: 'Persona "Legal Analyst" updated', time: '1 hour ago', status: 'Success' },
    { event: 'Gateway auth token rotated', time: '3 hours ago', status: 'Success' },
  ];

  return (
    <div className="space-y-8 animate-fadeIn">
      <header className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold">Enterprise Overview</h1>
          <p className="text-slate-400 mt-1">Platform metrics and operational status.</p>
        </div>
        <div className="flex gap-2">
          <div className="px-3 py-1 bg-slate-900 border border-slate-800 rounded-full flex items-center gap-2">
             <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"></div>
             <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400">System Live</span>
          </div>
        </div>
      </header>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statCards.map((stat) => (
          <div key={stat.label} className="bg-slate-900 border border-slate-800 p-6 rounded-2xl hover:border-slate-700 transition-colors shadow-sm">
            <div className="flex items-center justify-between">
              <i className={`fas ${stat.icon} text-2xl ${stat.color}`}></i>
              <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Live</span>
            </div>
            <div className="mt-4">
              <p className="text-slate-400 text-sm font-medium">{stat.label}</p>
              <h3 className="text-2xl font-bold mt-1">{stat.value}</h3>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Activity Feed */}
        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-bold">Recent Pipeline Activity</h3>
            <button className="text-blue-400 text-sm hover:underline font-semibold">Audit Logs</button>
          </div>
          <div className="space-y-4">
            {recentActivity.map((activity, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 rounded-xl bg-slate-800/50 hover:bg-slate-800 transition-all border border-transparent hover:border-slate-700 group">
                <div className="flex items-center gap-4">
                  <div className={`w-2 h-2 rounded-full ${activity.status === 'Success' ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.4)]' : 'bg-blue-500 animate-pulse'}`}></div>
                  <div>
                    <p className="text-sm font-medium">{activity.event}</p>
                    <p className="text-xs text-slate-500">{activity.time}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <span className={`text-[10px] font-bold px-2 py-1 rounded-full uppercase ${
                    activity.status === 'Success' ? 'bg-green-500/10 text-green-500' : 'bg-blue-500/10 text-blue-500'
                  }`}>
                    {activity.status}
                  </span>
                  <i className="fas fa-chevron-right text-[10px] text-slate-700 group-hover:text-slate-400 transition-colors"></i>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* System Health */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl h-fit">
          <h3 className="text-lg font-bold mb-6">Infrastructure Health</h3>
          <div className="space-y-6">
            <HealthMeter label="FastAPI Backend" status={stats?.health?.backend || "Healthy"} percent={98} />
            <HealthMeter label="Milvus Standalone" status={stats?.health?.milvus || "Optimal"} percent={92} />
            <HealthMeter label="Redis Cache" status={stats?.health?.redis || "Healthy"} percent={100} />
            <HealthMeter label="Agent Gateway" status={stats?.health?.gateway || "Healthy"} percent={99} />
          </div>
          <div className="mt-8 pt-6 border-t border-slate-800">
            <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
               <div className="flex justify-between items-center mb-1">
                 <span className="text-xs font-bold text-slate-400 uppercase">Resource Load</span>
                 <span className="text-[10px] font-mono text-blue-400">12% CPU</span>
               </div>
               <div className="h-1 w-full bg-slate-700 rounded-full">
                 <div className="h-full bg-blue-500 w-[12%] rounded-full"></div>
               </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const HealthMeter: React.FC<{ label: string, status: string, percent: number }> = ({ label, status, percent }) => (
  <div>
    <div className="flex justify-between items-center mb-2">
      <span className="text-sm font-medium text-slate-300">{label}</span>
      <span className={`text-xs font-semibold ${status === 'Healthy' || status === 'Optimal' ? 'text-green-500' : 'text-amber-500'}`}>
        {status}
      </span>
    </div>
    <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
      <div 
        className="h-full bg-blue-600 rounded-full transition-all duration-1000 shadow-[0_0_8px_rgba(37,99,235,0.3)]" 
        style={{ width: `${percent}%` }}
      ></div>
    </div>
  </div>
);

export default Dashboard;
