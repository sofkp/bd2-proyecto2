import { BarChart3 } from "lucide-react";
import type { Approach, Modality, UiStats } from "../types";

type StatsPanelProps = {
  approach: Approach;
  modality: Modality;
  hasSearched: boolean;
  stats: UiStats;
};

export function StatsPanel({ approach, modality, hasSearched, stats }: StatsPanelProps) {
  return (
    <div className="panel-card p-6 flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <BarChart3 className="h-4 w-4 text-purple-500" />
        <h2 className="panel-title text-sm font-bold uppercase tracking-wider">Métricas y Rendimiento</h2>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="stat-card p-4 flex flex-col gap-1">
          <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider">Latencia Total</span>
          <span className="text-xl font-bold text-purple-400">
            {hasSearched ? `${stats.time} ms` : "--"}
          </span>
          <span className="text-[9px] text-zinc-600">Red + query + render</span>
        </div>

        <div className="stat-card p-4 flex flex-col gap-1">
          <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider">Tiempo Query</span>
          <span className="text-xl font-bold text-emerald-400">
            {hasSearched ? `${stats.queryMs} ms` : "--"}
          </span>
          <span className="text-[9px] text-zinc-600">
            {approach === "custom" ? "Solo búsqueda en índice" : "Query GIN / HNSW en DB"}
          </span>
        </div>

        <div className="stat-card p-4 flex flex-col gap-1">
          <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider">
            {approach === "custom" ? "RAM Índice" : "Motor Índice"}
          </span>
          <span className="text-xl font-bold text-purple-400">
            {hasSearched
              ? approach === "custom"
                ? `${stats.indexMb} MB`
                : modality === "text" ? "GIN" : "HNSW"
              : "--"}
          </span>
          <span className="text-[9px] text-zinc-600">
            {approach === "custom" ? "Histogramas en memoria" : "pgvector en PostgreSQL"}
          </span>
        </div>

        <div className="stat-card p-4 flex flex-col gap-1">
          <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider">Comparaciones</span>
          <span className="text-xl font-bold text-emerald-400">
            {hasSearched ? stats.comparisons.toLocaleString() : "--"}
          </span>
          <span className="text-[9px] text-zinc-600">
            {approach === "custom"
              ? `Vectores comparados · dim ${hasSearched ? stats.vectorDim : "–"}`
              : "Resultados devueltos por DB"}
          </span>
        </div>
      </div>
    </div>
  );
}
