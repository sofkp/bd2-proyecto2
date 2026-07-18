import { Database, FileText, Music, Pause, Play, Search } from "lucide-react";
import type { Modality, ResultItem } from "../types";

type ResultsPanelProps = {
  modality: Modality;
  results: ResultItem[];
  hasSearched: boolean;
  isLoading: boolean;
  playingId: string | null;
  onPlayAudio: (id: string, audioUrl: string) => void;
  onSelectTextResult: (result: ResultItem) => void;
};

export function ResultsPanel({
  modality,
  results,
  hasSearched,
  isLoading,
  playingId,
  onPlayAudio,
  onSelectTextResult,
}: ResultsPanelProps) {
  return (
    <div className="panel-card p-6 flex-1 flex flex-col gap-4">
      <h2 className="panel-title text-sm font-bold uppercase tracking-wider flex items-center gap-2">
        <Database className="h-4 w-4 text-purple-500" />
        Resultados de la Recuperación
      </h2>

      {!hasSearched && !isLoading && (
        <div className="flex-1 flex flex-col items-center justify-center text-center p-8 gap-3 border border-dashed border-zinc-800 rounded-2xl">
          <Search className="h-8 w-8 text-zinc-700 animate-pulse" />
          <div>
            <span className="block text-sm font-semibold text-zinc-400">Sin consultas ejecutadas</span>
            <span className="block text-xs text-zinc-600 mt-1">Configura la query a la izquierda y presiona Ejecutar</span>
          </div>
        </div>
      )}

      {isLoading && (
        <div className="flex-1 flex flex-col items-center justify-center text-center p-8 gap-3">
          <div className="h-8 w-8 border-3 border-purple-500/20 border-t-purple-500 rounded-full animate-spin" />
          <span className="text-xs text-zinc-500">Consultando índice invertido y calculando distancias...</span>
        </div>
      )}

      {hasSearched && !isLoading && results.length === 0 && (
        <div className="flex-1 flex flex-col items-center justify-center text-center p-8 gap-3 border border-dashed border-zinc-800 rounded-2xl">
          <Search className="h-8 w-8 text-zinc-700" />
          <div>
            <span className="block text-sm font-semibold text-zinc-400">Sin resultados</span>
            <span className="block text-xs text-zinc-600 mt-1">
              No se encontró nada para esa consulta. Prueba con otros términos o revisa que el idioma coincida con los datos indexados (mayormente en inglés).
            </span>
          </div>
        </div>
      )}

      {hasSearched && !isLoading && results.length > 0 && (
        <div className="flex flex-col gap-4">
          {modality === "text" && (
            <TextResults results={results} onSelectTextResult={onSelectTextResult} />
          )}

          {modality === "image" && <ImageResults results={results} />}

          {modality === "audio" && (
            <AudioResults results={results} playingId={playingId} onPlayAudio={onPlayAudio} />
          )}
        </div>
      )}
    </div>
  );
}

function TextResults({
  results,
  onSelectTextResult,
}: {
  results: ResultItem[];
  onSelectTextResult: (result: ResultItem) => void;
}) {
  return results.map((item, idx) => {
    const rowId = `${item.id}-${idx}`;
    const fullText = item.fullText || item.snippet || "";
    const canOpen = fullText.length > 0;

    return (
      <div key={rowId} className="result-card p-5 hover:border-zinc-800 transition-all flex flex-col gap-2">
        <div className="flex items-center justify-between gap-4">
          <span className="rank-badge px-2 py-0.5">
            #{idx + 1} · {item.id}
          </span>
          <span className="score-text text-xs font-bold">
            {(item.similarity * 100).toFixed(2)}% Similitud
          </span>
        </div>
        <h3 className="content-title text-sm font-bold mt-1">{item.title}</h3>
        <span className="text-[10px] text-zinc-500 font-medium">{item.category}</span>
        <p className="text-xs text-zinc-400 leading-relaxed mt-2 border-l-2 border-purple-600 pl-3">
          {item.snippet}
        </p>
        {canOpen && (
          <button
            type="button"
            onClick={() => onSelectTextResult(item)}
            className="mt-1 inline-flex w-fit items-center gap-1.5 rounded-lg border border-zinc-800 bg-zinc-900 px-2.5 py-1 text-[10px] font-semibold text-zinc-300 hover:border-purple-700 hover:text-purple-300 transition-all"
          >
            <FileText className="h-3 w-3" />
            Ver texto completo
          </button>
        )}
      </div>
    );
  });
}

function ImageResults({ results }: { results: ResultItem[] }) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
      {results.map((item, idx) => (
        <div key={item.id} className="result-card rounded-xl overflow-hidden hover:border-purple-700 transition-all flex flex-col group">
          <div className="image-frame relative flex items-center justify-center">
            <img src={item.imgUrl} alt={item.title} />
            <span className="absolute top-2 left-2 text-[9px] bg-purple-600 text-white font-bold px-2 py-0.5 rounded-md shadow-md">
              Rank #{idx + 1}
            </span>
            <span className="score-text absolute bottom-2 right-2 text-[10px] font-bold bg-zinc-950/85 px-2 py-0.5 rounded-full">
              {(item.similarity * 100).toFixed(1)}%
            </span>
          </div>
          <div className="p-3 flex flex-col gap-0.5">
            <span className="text-[9px] text-zinc-500 uppercase font-bold tracking-wider">{item.category}</span>
            <h3 className="content-title text-xs font-bold line-clamp-1">{item.title}</h3>
            {item.price && (
              <span className="content-title text-xs font-black mt-0.5">{item.price}</span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

function AudioResults({
  results,
  playingId,
  onPlayAudio,
}: {
  results: ResultItem[];
  playingId: string | null;
  onPlayAudio: (id: string, audioUrl: string) => void;
}) {
  return results.map((item, idx) => (
    <div key={item.id} className="result-card p-4 hover:border-zinc-800 transition-all flex items-center justify-between gap-4">
      <div className="flex items-center gap-3">
        <div className="h-10 w-10 bg-purple-950/40 border border-purple-900/50 text-purple-400 rounded-xl flex items-center justify-center">
          <Music className="h-5 w-5" />
        </div>
        <div>
          <h3 className="content-title text-xs font-bold flex items-center gap-2">
            <span className="text-purple-400">#{idx + 1}</span> {item.title}
          </h3>
          <span className="text-[10px] text-zinc-500 mt-0.5 block">{item.artist} · {item.genres}</span>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <span className="text-[10px] text-zinc-600 font-mono">{item.duration}</span>
        <span className="score-text text-xs font-bold">
          {(item.similarity * 100).toFixed(1)}%
        </span>
        <button
          onClick={() => item.audio_url && onPlayAudio(item.id, item.audio_url)}
          disabled={!item.audio_url}
          className={`h-8 w-8 rounded-full flex items-center justify-center transition-colors ${
            playingId === item.id
              ? "bg-purple-600 text-white"
              : "bg-zinc-900 hover:bg-purple-600 hover:text-white text-zinc-400"
          } disabled:opacity-30`}
        >
          {playingId === item.id
            ? <Pause className="h-3.5 w-3.5 fill-current" />
            : <Play className="h-3.5 w-3.5 fill-current ml-0.5" />}
        </button>
      </div>
    </div>
  ));
}
