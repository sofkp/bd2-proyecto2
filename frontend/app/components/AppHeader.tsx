import { Layers } from "lucide-react";

export function AppHeader() {
  return (
    <header className="app-header">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="brand-mark flex items-center justify-center">
            <Layers className="h-5 w-5 text-white" />
          </div>
          <div>
            <span className="brand-title text-lg font-bold tracking-tight">TriModal Retrieval</span>
            <span className="brand-subtitle block text-xs font-medium">UTEC · Proyecto 2 · Ciclo 2026-1</span>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="status-text text-sm font-medium">PostgreSQL + pgvector Conectado</span>
        </div>
      </div>
    </header>
  );
}
