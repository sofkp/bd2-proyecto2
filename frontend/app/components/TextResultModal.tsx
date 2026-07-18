import { X } from "lucide-react";
import type { ResultItem } from "../types";

type TextResultModalProps = {
  result: ResultItem;
  onClose: () => void;
};

export function TextResultModal({ result, onClose }: TextResultModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 px-4 py-6 backdrop-blur-sm">
      <div className="flex max-h-[86vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl shadow-black/60">
        <div className="flex items-start justify-between gap-4 border-b border-zinc-900 bg-zinc-950 px-5 py-4">
          <div className="min-w-0">
            <span className="mb-2 inline-flex max-w-full items-center rounded-md border border-purple-900/50 bg-purple-950/40 px-2 py-0.5 text-[10px] font-bold text-purple-300">
              {result.id}
            </span>
            <h3 className="content-title text-sm font-bold">{result.title}</h3>
            <div className="mt-2 flex flex-wrap items-center gap-2 text-[10px] font-medium text-zinc-500">
              <span>{result.category || "Documento"}</span>
              <span className="h-1 w-1 rounded-full bg-zinc-700" />
              <span className="text-emerald-400">
                {(result.similarity * 100).toFixed(2)}% similitud
              </span>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-zinc-800 bg-zinc-900 text-zinc-400 transition-colors hover:border-purple-700 hover:text-white"
            aria-label="Cerrar vista de documento"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="min-h-0 overflow-y-auto px-5 py-4">
          <div className="border-l-2 border-purple-600 pl-4">
            <p className="whitespace-pre-wrap text-sm leading-7 text-zinc-300">
              {result.fullText || result.snippet}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
