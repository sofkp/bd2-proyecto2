import type React from "react";
import {
  FileText,
  Image as ImageIcon,
  Music,
  Pause,
  Play,
  Upload,
  Volume2,
  Zap,
} from "lucide-react";
import type { Approach, Modality } from "../types";

type QueryPanelProps = {
  modality: Modality;
  approach: Approach;
  queryText: string;
  topN: number;
  pdfFile: File | null;
  imagePreview: string | null;
  audioFile: string | null;
  audioFileRaw: File | null;
  isRecording: boolean;
  mfccReady: boolean;
  playingId: string | null;
  isLoading: boolean;
  onModalityChange: (modality: Modality) => void;
  onApproachChange: (approach: Approach) => void;
  onQueryTextChange: (value: string) => void;
  onTopNChange: (value: number) => void;
  onPdfSelected: (file: File) => void;
  onClearPdf: () => void;
  onImageUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onLoadSampleImage: (url: string, filename: string) => void;
  onClearImage: () => void;
  onAudioUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onClearAudio: () => void;
  onStartRecording: () => void;
  onStopRecording: () => void;
  onPlayAudio: (id: string, audioUrl: string) => void;
  onSearch: () => void;
};

export function QueryPanel({
  modality,
  approach,
  queryText,
  topN,
  pdfFile,
  imagePreview,
  audioFile,
  audioFileRaw,
  isRecording,
  mfccReady,
  playingId,
  isLoading,
  onModalityChange,
  onApproachChange,
  onQueryTextChange,
  onTopNChange,
  onPdfSelected,
  onClearPdf,
  onImageUpload,
  onLoadSampleImage,
  onClearImage,
  onAudioUpload,
  onClearAudio,
  onStartRecording,
  onStopRecording,
  onPlayAudio,
  onSearch,
}: QueryPanelProps) {
  const isSearchDisabled =
    isLoading ||
    (modality === "text" && !queryText && !pdfFile) ||
    (modality === "image" && !imagePreview) ||
    (modality === "audio" && !audioFileRaw);

  return (
    <section className="panel-card lg:col-span-4 flex flex-col gap-6 p-6 self-start">
      <div className="flex flex-col gap-1">
        <h2 className="panel-title text-lg font-bold">Panel de Consulta</h2>
        <p className="text-xs text-zinc-500">Configura los parámetros de búsqueda multimodal</p>
      </div>

      <div className="flex flex-col gap-2">
        <label className="field-label">Modalidad Primaria</label>
        <div className="segmented-control grid grid-cols-3 gap-2">
          <button
            onClick={() => onModalityChange("text")}
            className={`segmented-button ${modality === "text" ? "segmented-button--active" : "segmented-button--idle"}`}
          >
            <FileText className="h-4 w-4" />
            Texto
          </button>
          <button
            onClick={() => onModalityChange("image")}
            className={`segmented-button ${modality === "image" ? "segmented-button--active" : "segmented-button--idle"}`}
          >
            <ImageIcon className="h-4 w-4" />
            Imagen
          </button>
          <button
            onClick={() => onModalityChange("audio")}
            className={`segmented-button ${modality === "audio" ? "segmented-button--active" : "segmented-button--idle"}`}
          >
            <Volume2 className="h-4 w-4" />
            Audio
          </button>
        </div>
      </div>

      <div className="flex flex-col gap-2 min-h-[140px]">
        <label className="field-label">Entrada de Consulta</label>

        {modality === "text" && (
          <div className="flex flex-col gap-2">
            {pdfFile ? (
              <div className="surface-muted p-4 rounded-2xl flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <FileText className="h-5 w-5 text-purple-400" />
                  <div>
                    <p className="text-xs font-semibold text-zinc-300 max-w-[180px] truncate">{pdfFile.name}</p>
                    <p className="text-[10px] text-zinc-500">Se buscará por contenido del PDF</p>
                  </div>
                </div>
                <button onClick={onClearPdf} className="remove-button">
                  Remover
                </button>
              </div>
            ) : (
              <textarea
                value={queryText}
                onChange={(event) => onQueryTextChange(event.target.value)}
                placeholder="Ingresa texto, paper, o palabras clave a buscar..."
                className="surface-muted w-full min-h-[100px] rounded-2xl p-4 text-sm text-zinc-300 placeholder-zinc-500 focus:outline-none focus:border-purple-600 focus:ring-1 focus:ring-purple-600 resize-none transition-all"
              />
            )}
            <label className="flex items-center gap-2 cursor-pointer self-start">
              <div className="ghost-action flex items-center gap-1.5 text-[11px] px-3 py-1.5 rounded-lg">
                <FileText className="h-3.5 w-3.5" />
                {pdfFile ? "Cambiar PDF" : "Subir PDF"}
              </div>
              <input
                type="file"
                accept=".pdf"
                className="hidden"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) onPdfSelected(file);
                }}
              />
            </label>
          </div>
        )}

        {modality === "image" && (
          <div className="flex flex-col gap-3">
            {imagePreview ? (
              <div className="surface-muted relative h-32 w-full rounded-2xl overflow-hidden flex items-center justify-center">
                <img src={imagePreview} alt="Preview" className="h-full object-contain" />
                <button onClick={onClearImage} className="remove-button absolute top-2 right-2 backdrop-blur-sm">
                  Remover
                </button>
              </div>
            ) : (
              <label className="upload-zone p-6 flex flex-col items-center justify-center gap-2 group">
                <Upload className="h-8 w-8 text-zinc-600 group-hover:text-purple-500 transition-colors" />
                <span className="text-xs font-medium text-zinc-400">Arrastra o sube una imagen de referencia</span>
                <span className="text-[10px] text-zinc-600">JPG, PNG o WEBP</span>
                <input type="file" accept="image/*" className="hidden" onChange={onImageUpload} />
              </label>
            )}

            <div className="flex flex-col gap-1.5">
              <span className="text-[10px] text-zinc-500">¿No tienes imagen? Selecciona una de muestra:</span>
              <div className="flex gap-2">
                <button
                  onClick={() => onLoadSampleImage("https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&q=80", "zapatilla_roja.jpg")}
                  className="ghost-action text-[10px] py-1 px-2.5 rounded-lg"
                >
                  👟 Zapatilla Roja
                </button>
                <button
                  onClick={() => onLoadSampleImage("https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?w=400&q=80", "tacon_rosa.jpg")}
                  className="ghost-action text-[10px] py-1 px-2.5 rounded-lg"
                >
                  👠 Tacón Rosa
                </button>
              </div>
            </div>
          </div>
        )}

        {modality === "audio" && (
          <AudioInput
            approach={approach}
            audioFile={audioFile}
            audioFileRaw={audioFileRaw}
            isRecording={isRecording}
            mfccReady={mfccReady}
            playingId={playingId}
            onAudioUpload={onAudioUpload}
            onClearAudio={onClearAudio}
            onStartRecording={onStartRecording}
            onStopRecording={onStopRecording}
            onPlayAudio={onPlayAudio}
          />
        )}
      </div>

      <div className="flex flex-col gap-2">
        <label className="field-label">Enfoque de Búsqueda</label>
        <div className="flex flex-col gap-2">
          <ApproachButton
            active={approach === "custom"}
            title="Nuestra Implementación"
            description="Codebook + Histograma + Índice Invertido manual"
            onClick={() => onApproachChange("custom")}
          />
          <ApproachButton
            active={approach === "postgres"}
            title="PostgreSQL Nativo"
            description="GIN (Texto) / pgvector con HNSW (Imagen/Audio)"
            onClick={() => onApproachChange("postgres")}
          />
        </div>
      </div>

      <div className="flex flex-col gap-4 border-t border-zinc-900 pt-4">
        <div className="flex flex-col gap-1.5">
          <div className="flex justify-between items-center text-xs">
            <span className="font-semibold text-zinc-400">Tamaño K (Codewords)</span>
            <span className="font-bold text-purple-400">
              {modality === "text" ? "1000 words" : modality === "image" ? "100 clusters" : "512 clusters"}
            </span>
          </div>
          <span className="text-[10px] text-zinc-500">
            {modality === "text"
              ? "Top-1000 palabras por TF-IDF · fijo al indexar"
              : modality === "image"
              ? "100 visual words · KMeans sobre descriptores SIFT"
              : "512 acoustic words · MiniBatchKMeans sobre MFCC"}
          </span>
        </div>

        <div className="flex flex-col gap-1.5">
          <div className="flex justify-between items-center text-xs">
            <span className="font-semibold text-zinc-400">Top Resultados (N)</span>
            <span className="font-bold text-purple-400">{topN} elementos</span>
          </div>
          <input
            type="range"
            min="1"
            max="20"
            step="1"
            value={topN}
            onChange={(event) => onTopNChange(parseInt(event.target.value))}
            className="w-full accent-purple-600"
          />
        </div>
      </div>

      <button
        onClick={onSearch}
        disabled={isSearchDisabled}
        className="primary-action py-3.5 flex items-center justify-center gap-2"
      >
        {isLoading ? (
          <div className="h-5 w-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
        ) : (
          <>
            <Zap className="h-4 w-4" />
            Ejecutar Búsqueda
          </>
        )}
      </button>
    </section>
  );
}

function ApproachButton({
  active,
  title,
  description,
  onClick,
}: {
  active: boolean;
  title: string;
  description: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`approach-card ${active ? "approach-card--active" : "approach-card--idle"}`}
    >
      <div className={`mt-0.5 h-4 w-4 rounded-full border flex items-center justify-center ${
        active ? "border-purple-500" : "border-zinc-700"
      }`}>
        {active && <div className="h-2 w-2 rounded-full bg-purple-500" />}
      </div>
      <div>
        <span className="block text-xs font-bold">{title}</span>
        <span className="block text-[10px] text-zinc-500 mt-0.5">{description}</span>
      </div>
    </button>
  );
}

function AudioInput({
  approach,
  audioFile,
  audioFileRaw,
  isRecording,
  mfccReady,
  playingId,
  onAudioUpload,
  onClearAudio,
  onStartRecording,
  onStopRecording,
  onPlayAudio,
}: {
  approach: Approach;
  audioFile: string | null;
  audioFileRaw: File | null;
  isRecording: boolean;
  mfccReady: boolean;
  playingId: string | null;
  onAudioUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onClearAudio: () => void;
  onStartRecording: () => void;
  onStopRecording: () => void;
  onPlayAudio: (id: string, audioUrl: string) => void;
}) {
  const description = approach === "custom"
    ? `MFCC + KMeans codebook acústico — ${mfccReady ? "listo" : "esperando datos"}`
    : "MFCC + pgvector HNSW — búsqueda en PostgreSQL";

  return (
    <div className="flex flex-col gap-3">
      {audioFile && audioFileRaw ? (
        <div className="surface-muted p-4 rounded-2xl flex items-center justify-between gap-2">
          <div className="flex items-center gap-3 min-w-0">
            <Music className="h-5 w-5 text-purple-500 shrink-0" />
            <span className="text-xs font-medium text-zinc-300 truncate">{audioFile}</span>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={() => onPlayAudio("__query__", URL.createObjectURL(audioFileRaw))}
              className={`h-7 w-7 rounded-full flex items-center justify-center transition-colors ${
                playingId === "__query__"
                  ? "bg-purple-600 text-white"
                  : "bg-zinc-800 hover:bg-purple-600 hover:text-white text-zinc-400"
              }`}
            >
              {playingId === "__query__"
                ? <Pause className="h-3 w-3 fill-current" />
                : <Play className="h-3 w-3 fill-current ml-0.5" />}
            </button>
            <button onClick={onClearAudio} className="remove-button">
              Remover
            </button>
          </div>
        </div>
      ) : isRecording ? (
        <div className="border border-red-800 bg-red-950/20 p-5 rounded-2xl flex flex-col items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-red-500 animate-pulse" />
            <span className="text-xs font-semibold text-red-400">Grabando... tararea o canta</span>
          </div>
          <button
            onClick={onStopRecording}
            className="bg-red-600 hover:bg-red-500 text-white text-xs font-bold py-2 px-5 rounded-full transition-all"
          >
            Detener grabación
          </button>
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          <label className="upload-zone p-5 flex flex-col items-center justify-center gap-2 group">
            <Music className="h-7 w-7 text-zinc-600 group-hover:text-purple-500 transition-colors" />
            <span className="text-xs font-medium text-zinc-400">Sube un archivo de audio</span>
            <span className="text-[10px] text-zinc-600">WAV, MP3 u OGG</span>
            <input type="file" accept="audio/*" className="hidden" onChange={onAudioUpload} />
          </label>
          <button
            onClick={onStartRecording}
            className="ghost-action w-full flex items-center justify-center gap-2 text-xs font-medium py-2.5 rounded-2xl"
          >
            <span className="h-2 w-2 rounded-full bg-red-500" />
            Grabar desde micrófono
          </button>
        </div>
      )}
      <span className="text-[10px] text-zinc-500 flex items-center gap-1">
        <Music className="h-3 w-3" /> {description}
      </span>
    </div>
  );
}
