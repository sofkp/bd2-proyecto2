"use client";

import React, { useState } from "react";
import { 
  Search, 
  Image as ImageIcon, 
  Volume2, 
  Database, 
  Zap, 
  BarChart3, 
  Layers, 
  ArrowRight,
  Upload,
  Music,
  FileText,
  Play,
  Pause,
  AlertCircle
} from "lucide-react";

// --- DATOS SIMULADOS PARA DEMOSTRACIONES (FALLBACK) ---
const MOCK_TEXT_RESULTS = [
  {
    id: "arxiv_042",
    title: "Deep Multimodal Representation Learning for Image-Text Retrieval",
    similarity: 0.9421,
    category: "cs.CV (Computer Vision)",
    snippet: "We propose a novel framework for learning shared vector spaces (embeddings) across different modalities. By using a visual codebook combined with contrastive learning, our method outperforms state-of-the-art architectures in both latency and recall..."
  },
  {
    id: "arxiv_108",
    title: "Bag of Visual Words meets pgvector: Scalable Multimedia Search in PostgreSQL",
    similarity: 0.8872,
    category: "cs.DB (Databases)",
    snippet: "This paper analyzes the trade-offs of storing quantized image descriptors (codewords) in traditional inverted indexes vs storing raw high-dimensional embeddings in pgvector. Experimental results show that the codebook approach reduces disk I/O by 4x..."
  },
  {
    id: "arxiv_012",
    title: "Unifying Text, Image and Audio under a Shared Codebook Architecture",
    similarity: 0.8143,
    category: "cs.IR (Information Retrieval)",
    snippet: "A common paradigm for multimodal retrieval consists of splitting raw data into atomic chunks, extracting local SIFT/MFCC features, clustering them with K-Means to build a joint codebook, and query execution using inverted lists..."
  }
];

const MOCK_IMAGE_RESULTS = [
  {
    id: "fashion_001",
    title: "Zapatilla Nike Air Max Roja",
    similarity: 0.9634,
    price: "S/ 389.00",
    category: "Calzado Deportivo",
    imgUrl: "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&q=80"
  },
  {
    id: "fashion_002",
    title: "Zapatilla Running Confort Sport",
    similarity: 0.8912,
    price: "S/ 249.00",
    category: "Calzado Running",
    imgUrl: "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?w=400&q=80"
  },
  {
    id: "fashion_003",
    title: "Zapatillas Urbanas de Cuero Negro",
    similarity: 0.8245,
    price: "S/ 299.00",
    category: "Calzado Casual",
    imgUrl: "https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?w=400&q=80"
  }
];

const MOCK_AUDIO_RESULTS = [
  {
    id: "spotify_001",
    title: "Midnight City Groove",
    artist: "The Synthwave Project",
    similarity: 0.9512,
    duration: "3:42",
    genres: "Electronic, Synthwave"
  },
  {
    id: "spotify_002",
    title: "Electric Dreams",
    artist: "Retro Wave Orchestra",
    similarity: 0.8741,
    duration: "4:05",
    genres: "Electronic, Retro"
  },
  {
    id: "spotify_003",
    title: "Starlight Disco",
    artist: "Neon Horizon",
    similarity: 0.8105,
    duration: "3:18",
    genres: "Disco, Pop"
  }
];

export default function Home() {
  // Estados de Configuración
  const [modality, setModality] = useState<"text" | "image" | "audio">("text");
  const [approach, setApproach] = useState<"custom" | "postgres">("custom");
  const [queryText, setQueryText] = useState("");
  const [kValue, setKValue] = useState(500);
  const [topN, setTopN] = useState(5);
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [audioFile, setAudioFile] = useState<string | null>(null);
  const [audioFileRaw, setAudioFileRaw] = useState<File | null>(null);
  const [isRecording, setIsRecording] = useState(false);

  const [mfccReady, setMfccReady] = useState(false);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const audioPlayerRef = React.useRef<HTMLAudioElement | null>(null);
  const audioContextRef = React.useRef<AudioContext | null>(null);
  const streamRef = React.useRef<MediaStream | null>(null);
  const samplesRef = React.useRef<Float32Array[]>([]);
  const processorRef = React.useRef<ScriptProcessorNode | null>(null);

  // Encodea muestras Float32 a WAV (PCM 16-bit mono)
  const encodeWAV = (samples: Float32Array[], sampleRate: number): Blob => {
    const allSamples = new Float32Array(samples.reduce((n, s) => n + s.length, 0));
    let offset = 0;
    for (const chunk of samples) { allSamples.set(chunk, offset); offset += chunk.length; }
    const buf = new ArrayBuffer(44 + allSamples.length * 2);
    const view = new DataView(buf);
    const ws = (o: number, s: string) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
    ws(0, "RIFF"); view.setUint32(4, 36 + allSamples.length * 2, true); ws(8, "WAVE");
    ws(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    ws(36, "data"); view.setUint32(40, allSamples.length * 2, true);
    let o = 44;
    for (let i = 0; i < allSamples.length; i++) {
      const s = Math.max(-1, Math.min(1, allSamples[i]));
      view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7FFF, true); o += 2;
    }
    return new Blob([buf], { type: "audio/wav" });
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      streamRef.current = stream;
      const ctx = new AudioContext({ sampleRate: 22050 });
      audioContextRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      samplesRef.current = [];
      processor.onaudioprocess = (e) => {
        samplesRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      processor.connect(ctx.destination);
      processorRef.current = processor;
      setIsRecording(true);
    } catch {
      alert("No se pudo acceder al micrófono. Verifica los permisos del navegador.");
    }
  };

  const stopRecording = () => {
    processorRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    const sampleRate = audioContextRef.current?.sampleRate ?? 22050;
    audioContextRef.current?.close();
    const wav = encodeWAV(samplesRef.current, sampleRate);
    const file = new File([wav], "grabacion.wav", { type: "audio/wav" });
    setAudioFile("grabacion.wav");
    setAudioFileRaw(file);
    setIsRecording(false);
  };

  // Verificar estado del pipeline MFCC al montar
  React.useEffect(() => {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    fetch(`${API_URL}/pipeline/status`)
      .then((r) => r.json())
      .then((d) => setMfccReady(d?.audio_mfcc?.ready ?? false))
      .catch(() => {});
  }, []);

  // Limpiar audio al desmontar
  React.useEffect(() => {
    return () => { audioPlayerRef.current?.pause(); };
  }, []);

  const handlePlayAudio = (id: string, audioUrl: string) => {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    if (playingId === id) {
      audioPlayerRef.current?.pause();
      setPlayingId(null);
    } else {
      if (audioPlayerRef.current) {
        audioPlayerRef.current.pause();
      }
      // blob: URLs son locales, no necesitan el prefijo del backend
      const fullUrl = audioUrl.startsWith("blob:") ? audioUrl : `${API_URL}${audioUrl}`;
      audioPlayerRef.current = new Audio(fullUrl);
      audioPlayerRef.current.play();
      audioPlayerRef.current.onended = () => setPlayingId(null);
      setPlayingId(id);
    }
  };

  // Estados de Ejecución y Resultados
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [stats, setStats] = useState({
    time: 0,
    queryMs: 0,
    indexMb: 0,
    comparisons: 0,
    vectorDim: 0,
  });

  // Búsqueda real contra la API
  const handleSearch = async () => {
    setIsLoading(true);
    setHasSearched(false);

    const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const t0 = performance.now();

    const applyStats = (raw: { query_ms?: number; index_mb?: number; n_comparisons?: number; vector_dim?: number }, elapsed: number) => {
      setStats({
        time: elapsed,
        queryMs: raw.query_ms ?? 0,
        indexMb: raw.index_mb ?? 0,
        comparisons: raw.n_comparisons ?? 0,
        vectorDim: raw.vector_dim ?? 0,
      });
    };

    try {
      if (approach === "custom" && modality === "text" && pdfFile) {
        const form = new FormData();
        form.append("file", pdfFile);
        const res = await fetch(`${API_URL}/pipeline/search/text/pdf?k=${topN}`, { method: "POST", body: form });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const elapsed = parseFloat((performance.now() - t0).toFixed(2));
        setResults((data.results ?? data).map((r: any) => ({
          id: r.chunk_id, title: r.metadata.title || r.chunk_id, similarity: r.score,
          category: r.metadata.source || "arxiv", snippet: r.metadata.snippet || "",
        })));
        applyStats(data.stats ?? {}, elapsed);

      } else if (approach === "custom" && modality === "text") {
        const res = await fetch(`${API_URL}/pipeline/search/text`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: queryText, k: topN }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const elapsed = parseFloat((performance.now() - t0).toFixed(2));
        setResults((data.results ?? data).map((r: any) => ({
          id: r.chunk_id, title: r.metadata.title || r.chunk_id, similarity: r.score,
          category: r.metadata.source || "arxiv", snippet: r.metadata.snippet || "",
        })));
        applyStats(data.stats ?? {}, elapsed);

      } else if (approach === "custom" && modality === "audio" && audioFileRaw) {
        const form = new FormData();
        form.append("file", audioFileRaw);
        const res = await fetch(`${API_URL}/pipeline/search/audio-file?k=${topN}`, { method: "POST", body: form });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const elapsed = parseFloat((performance.now() - t0).toFixed(2));
        setResults((data.results ?? data).map((r: any) => ({
          id: r.chunk_id, title: r.metadata.title || r.chunk_id,
          artist: r.metadata.genre || "", similarity: r.score, duration: "–", genres: r.metadata.genre || "",
          audio_url: r.metadata.audio_url || null,
        })));
        applyStats(data.stats ?? {}, elapsed);

      } else if (approach === "custom" && modality === "image" && imageFile) {
        const form = new FormData();
        form.append("file", imageFile);
        const res = await fetch(`${API_URL}/pipeline/search/image?k=${topN}`, { method: "POST", body: form });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const elapsed = parseFloat((performance.now() - t0).toFixed(2));
        setResults((data.results ?? data).map((r: any) => ({
          id: r.chunk_id, title: r.metadata.title || r.chunk_id, similarity: r.score,
          category: "Imagen", imgUrl: `${API_URL}${r.metadata.image_url}`,
        })));
        applyStats(data.stats ?? {}, elapsed);

      } else {
        // Postgres → mock por ahora
        await new Promise((r) => setTimeout(r, 800));
        if (modality === "image") {
          setResults(MOCK_IMAGE_RESULTS.slice(0, topN));
        } else {
          setResults(MOCK_TEXT_RESULTS.slice(0, topN));
        }
        const elapsed = parseFloat((performance.now() - t0).toFixed(2));
        setStats({ time: elapsed, queryMs: 0, indexMb: 0, comparisons: 0, vectorDim: 0 });
      }
    } catch (err) {
      console.error("Error en búsqueda:", err);
      setResults([]);
    } finally {
      setIsLoading(false);
      setHasSearched(true);
    }
  };

  // Manejar Carga de Archivos Falsa
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const handleAudioUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setAudioFile(file.name);
      setAudioFileRaw(file);
    }
  };

  // Cargar imagen de muestra como File real para poder enviarla al backend
  const loadSampleImage = async (url: string, filename: string) => {
    try {
      const res = await fetch(url);
      const blob = await res.blob();
      const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
      setImageFile(file);
      setImagePreview(URL.createObjectURL(blob));
    } catch {
      setImagePreview(url); // fallback visual si hay error CORS
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-zinc-950 text-zinc-100 font-sans antialiased selection:bg-purple-600 selection:text-white">
      {/* HEADER */}
      <header className="border-b border-zinc-900 bg-zinc-950/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-purple-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-purple-900/30">
              <Layers className="h-5 w-5 text-white" />
            </div>
            <div>
              <span className="text-lg font-bold tracking-tight text-white">MUT RETRIEVAL BD2</span>
              <span className="block text-xs text-zinc-500 font-medium">UTEC · Proyecto 2 · Ciclo 2026-1</span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span className="text-sm font-medium text-zinc-400">PostgreSQL + pgvector Conectado</span>
          </div>
        </div>
      </header>

      {/* CUERPO PRINCIPAL */}
      <main className="max-w-7xl mx-auto px-6 py-8 flex-1 grid grid-cols-1 lg:grid-cols-12 gap-8 w-full">
        {/* PANEL DE CONTROL (IZQUIERDA) */}
        <section className="lg:col-span-4 flex flex-col gap-6 bg-zinc-900/40 border border-zinc-900 rounded-3xl p-6 shadow-xl backdrop-blur-sm self-start">
          <div className="flex flex-col gap-1">
            <h2 className="text-lg font-bold text-white">Panel de Consulta</h2>
            <p className="text-xs text-zinc-500">Configura los parámetros de búsqueda multimodal</p>
          </div>

          {/* Selector de Modalidad */}
          <div className="flex flex-col gap-2">
            <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Modalidad Primaria</label>
            <div className="grid grid-cols-3 gap-2 bg-zinc-950 p-1 rounded-xl border border-zinc-900">
              <button
                onClick={() => { setModality("text"); setHasSearched(false); }}
                className={`py-2 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-all ${
                  modality === "text"
                    ? "bg-purple-600 text-white shadow-md"
                    : "text-zinc-400 hover:text-white hover:bg-zinc-900"
                }`}
              >
                <FileText className="h-4 w-4" />
                Texto
              </button>
              <button
                onClick={() => { setModality("image"); setHasSearched(false); }}
                className={`py-2 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-all ${
                  modality === "image"
                    ? "bg-purple-600 text-white shadow-md"
                    : "text-zinc-400 hover:text-white hover:bg-zinc-900"
                }`}
              >
                <ImageIcon className="h-4 w-4" />
                Imagen
              </button>
              <button
                onClick={() => { setModality("audio"); setHasSearched(false); }}
                className={`py-2 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-all ${
                  modality === "audio"
                    ? "bg-purple-600 text-white shadow-md"
                    : "text-zinc-400 hover:text-white hover:bg-zinc-900"
                }`}
              >
                <Volume2 className="h-4 w-4" />
                Audio
              </button>
            </div>
          </div>

          {/* Campos según la Modalidad */}
          <div className="flex flex-col gap-2 min-h-[140px]">
            <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Entrada de Consulta</label>

            {/* Input de Texto */}
            {modality === "text" && (
              <div className="flex flex-col gap-2">
                {pdfFile ? (
                  <div className="border border-purple-800 bg-purple-950/20 p-4 rounded-2xl flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <FileText className="h-5 w-5 text-purple-400" />
                      <div>
                        <p className="text-xs font-semibold text-zinc-200 max-w-[180px] truncate">{pdfFile.name}</p>
                        <p className="text-[10px] text-zinc-500">Se buscará por contenido del PDF</p>
                      </div>
                    </div>
                    <button
                      onClick={() => { setPdfFile(null); setResults([]); setHasSearched(false); }}
                      className="bg-red-600/80 hover:bg-red-600 text-white text-[10px] font-bold py-1 px-2.5 rounded-full transition-all"
                    >
                      Remover
                    </button>
                  </div>
                ) : (
                  <textarea
                    value={queryText}
                    onChange={(e) => setQueryText(e.target.value)}
                    placeholder="Ingresa texto, paper, o palabras clave a buscar..."
                    className="w-full min-h-[100px] bg-zinc-950 border border-zinc-900 rounded-2xl p-4 text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-purple-600 focus:ring-1 focus:ring-purple-600 resize-none transition-all"
                  />
                )}
                <label className="flex items-center gap-2 cursor-pointer self-start">
                  <div className="flex items-center gap-1.5 text-[11px] text-zinc-500 hover:text-purple-400 border border-zinc-800 hover:border-purple-700 bg-zinc-950 px-3 py-1.5 rounded-lg transition-all">
                    <FileText className="h-3.5 w-3.5" />
                    {pdfFile ? "Cambiar PDF" : "Subir PDF"}
                  </div>
                  <input
                    type="file"
                    accept=".pdf"
                    className="hidden"
                    onChange={(e) => { if (e.target.files?.[0]) { setPdfFile(e.target.files[0]); setQueryText(""); } }}
                  />
                </label>
              </div>
            )}

            {/* Input de Imagen */}
            {modality === "image" && (
              <div className="flex flex-col gap-3">
                {imagePreview ? (
                  <div className="relative h-32 w-full rounded-2xl overflow-hidden border border-zinc-900 bg-zinc-950 flex items-center justify-center">
                    <img src={imagePreview} alt="Preview" className="h-full object-contain" />
                    <button
                      onClick={() => { setImagePreview(null); setImageFile(null); setResults([]); setHasSearched(false); }}
                      className="absolute top-2 right-2 bg-red-600/80 hover:bg-red-600 text-white text-[10px] font-bold py-1 px-2.5 rounded-full backdrop-blur-sm transition-all"
                    >
                      Remover
                    </button>
                  </div>
                ) : (
                  <label className="border border-dashed border-zinc-800 hover:border-purple-600 bg-zinc-950/50 hover:bg-zinc-950 p-6 rounded-2xl flex flex-col items-center justify-center gap-2 cursor-pointer transition-all group">
                    <Upload className="h-8 w-8 text-zinc-600 group-hover:text-purple-500 transition-colors" />
                    <span className="text-xs font-medium text-zinc-400">Arrastra o sube una imagen de referencia</span>
                    <span className="text-[10px] text-zinc-600">JPG, PNG o WEBP</span>
                    <input type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
                  </label>
                )}

                {/* Accesos rápidos de imágenes simuladas */}
                <div className="flex flex-col gap-1.5">
                  <span className="text-[10px] text-zinc-500">¿No tienes imagen? Selecciona una de muestra:</span>
                  <div className="flex gap-2">
                    <button
                      onClick={() => loadSampleImage("https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&q=80", "zapatilla_roja.jpg")}
                      className="text-[10px] bg-zinc-900 border border-zinc-800 hover:border-purple-600 text-zinc-300 py-1 px-2.5 rounded-lg transition-all"
                    >
                      👟 Zapatilla Roja
                    </button>
                    <button
                      onClick={() => loadSampleImage("https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?w=400&q=80", "tacon_rosa.jpg")}
                      className="text-[10px] bg-zinc-900 border border-zinc-800 hover:border-purple-600 text-zinc-300 py-1 px-2.5 rounded-lg transition-all"
                    >
                      👠 Tacón Rosa
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Input de Audio - Nuestra Implementación (MFCC) */}
            {modality === "audio" && approach === "custom" && (
              <div className="flex flex-col gap-3">
                {audioFile && approach === "custom" ? (
                  <div className="border border-zinc-900 bg-zinc-950 p-4 rounded-2xl flex items-center justify-between gap-2">
                    <div className="flex items-center gap-3 min-w-0">
                      <Music className="h-5 w-5 text-purple-500 shrink-0" />
                      <span className="text-xs font-medium text-zinc-300 truncate">{audioFile}</span>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <button
                        onClick={() => handlePlayAudio("__query__", URL.createObjectURL(audioFileRaw!))}
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
                      <button
                        onClick={() => { audioPlayerRef.current?.pause(); setPlayingId(null); setAudioFile(null); setAudioFileRaw(null); setResults([]); setHasSearched(false); }}
                        className="bg-red-600/80 hover:bg-red-600 text-white text-[10px] font-bold py-1 px-2.5 rounded-full transition-all"
                      >
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
                      onClick={stopRecording}
                      className="bg-red-600 hover:bg-red-500 text-white text-xs font-bold py-2 px-5 rounded-full transition-all"
                    >
                      Detener grabación
                    </button>
                  </div>
                ) : (
                  <div className="flex flex-col gap-2">
                    <label className="border border-dashed border-zinc-800 hover:border-purple-600 bg-zinc-950/50 p-5 rounded-2xl flex flex-col items-center justify-center gap-2 cursor-pointer transition-all group">
                      <Music className="h-7 w-7 text-zinc-600 group-hover:text-purple-500 transition-colors" />
                      <span className="text-xs font-medium text-zinc-400">Sube un archivo de audio</span>
                      <span className="text-[10px] text-zinc-600">WAV, MP3 u OGG</span>
                      <input type="file" accept="audio/*" className="hidden" onChange={handleAudioUpload} />
                    </label>
                    <button
                      onClick={startRecording}
                      className="w-full flex items-center justify-center gap-2 border border-zinc-800 hover:border-purple-600 bg-zinc-950/50 hover:bg-zinc-950 text-zinc-400 hover:text-purple-400 text-xs font-medium py-2.5 rounded-2xl transition-all"
                    >
                      <span className="h-2 w-2 rounded-full bg-red-500" />
                      Grabar desde micrófono
                    </button>
                  </div>
                )}
                <span className="text-[10px] text-zinc-500 flex items-center gap-1">
                  <Music className="h-3 w-3" /> MFCC + KMeans codebook acústico — {mfccReady ? "listo" : "esperando datos"}
                </span>
              </div>
            )}

            {modality === "audio" && approach === "postgres" && (
              <div className="flex flex-col gap-3">
                {audioFile ? (
                  <div className="border border-zinc-900 bg-zinc-950 p-4 rounded-2xl flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Music className="h-5 w-5 text-purple-500" />
                      <span className="text-xs font-medium text-zinc-300 max-w-[150px] truncate">{audioFile}</span>
                    </div>
                    <button
                      onClick={() => setAudioFile(null)}
                      className="bg-red-600/80 hover:bg-red-600 text-white text-[10px] font-bold py-1 px-2.5 rounded-full transition-all"
                    >
                      Remover
                    </button>
                  </div>
                ) : (
                  <label className="border border-dashed border-zinc-800 hover:border-purple-600 bg-zinc-950/50 p-6 rounded-2xl flex flex-col items-center justify-center gap-2 cursor-pointer transition-all group">
                    <Music className="h-8 w-8 text-zinc-600 group-hover:text-purple-500 transition-colors" />
                    <span className="text-xs font-medium text-zinc-400">Arrastra o sube un archivo de audio</span>
                    <span className="text-[10px] text-zinc-600">MP3, WAV o OGG</span>
                    <input type="file" accept="audio/*" className="hidden" onChange={handleAudioUpload} />
                  </label>
                )}
              </div>
            )}
          </div>

          {/* Selector de Enfoque/Algoritmo */}
          <div className="flex flex-col gap-2">
            <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Enfoque de Búsqueda</label>
            <div className="flex flex-col gap-2">
              <button
                onClick={() => setApproach("custom")}
                className={`w-full p-3 rounded-xl border text-left flex items-start gap-3 transition-all ${
                  approach === "custom"
                    ? "bg-purple-950/20 border-purple-600 text-white"
                    : "bg-zinc-950/50 border-zinc-900 hover:border-zinc-800 text-zinc-400"
                }`}
              >
                <div className={`mt-0.5 h-4 w-4 rounded-full border flex items-center justify-center ${
                  approach === "custom" ? "border-purple-500" : "border-zinc-700"
                }`}>
                  {approach === "custom" && <div className="h-2 w-2 rounded-full bg-purple-500" />}
                </div>
                <div>
                  <span className="block text-xs font-bold">Nuestra Implementación</span>
                  <span className="block text-[10px] text-zinc-500 mt-0.5">Codebook + Histograma + Índice Invertido manual</span>
                </div>
              </button>
              <button
                onClick={() => setApproach("postgres")}
                className={`w-full p-3 rounded-xl border text-left flex items-start gap-3 transition-all ${
                  approach === "postgres"
                    ? "bg-purple-950/20 border-purple-600 text-white"
                    : "bg-zinc-950/50 border-zinc-900 hover:border-zinc-800 text-zinc-400"
                }`}
              >
                <div className={`mt-0.5 h-4 w-4 rounded-full border flex items-center justify-center ${
                  approach === "postgres" ? "border-purple-500" : "border-zinc-700"
                }`}>
                  {approach === "postgres" && <div className="h-2 w-2 rounded-full bg-purple-500" />}
                </div>
                <div>
                  <span className="block text-xs font-bold">PostgreSQL Nativo</span>
                  <span className="block text-[10px] text-zinc-500 mt-0.5">GIN (Texto) / pgvector con HNSW (Imagen/Audio)</span>
                </div>
              </button>
            </div>
          </div>

          {/* Parámetros Adicionales */}
          <div className="flex flex-col gap-4 border-t border-zinc-900 pt-4">
            <div className="flex flex-col gap-1.5">
              <div className="flex justify-between items-center text-xs">
                <span className="font-semibold text-zinc-400">Tamaño K (Codewords)</span>
                <span className="font-bold text-purple-400">
                  {modality === "text" ? "200 words" : modality === "image" ? "100 clusters" : "50 clusters"}
                </span>
              </div>
              <div className="w-full bg-zinc-900 rounded-full h-1.5">
                <div
                  className="bg-purple-600 h-1.5 rounded-full transition-all"
                  style={{ width: modality === "text" ? "10%" : modality === "image" ? "5%" : "2.5%" }}
                />
              </div>
              <span className="text-[10px] text-zinc-600">
                {modality === "text"
                  ? "Top-200 palabras por TF-IDF · fijo al indexar"
                  : modality === "image"
                  ? "100 visual words · KMeans sobre descriptores SIFT"
                  : "50 acoustic words · MiniBatchKMeans sobre MFCC"}
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
                onChange={(e) => setTopN(parseInt(e.target.value))}
                className="w-full accent-purple-600"
              />
            </div>
          </div>

          {/* Botón Ejecutar */}
          <button
            onClick={handleSearch}
            disabled={isLoading || (modality === "text" && !queryText && !pdfFile) || (modality === "image" && !imagePreview) || (modality === "audio" && approach === "postgres" && !audioFile) || (modality === "audio" && approach === "custom" && !audioFileRaw)}
            className="w-full py-3.5 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 disabled:from-zinc-800 disabled:to-zinc-800 disabled:text-zinc-600 text-white font-bold rounded-2xl flex items-center justify-center gap-2 shadow-lg shadow-purple-950/20 active:scale-[0.98] transition-all cursor-pointer"
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

        {/* CONTENIDO DERECHA: ESTADÍSTICAS Y RESULTADOS */}
        <section className="lg:col-span-8 flex flex-col gap-6">
          {/* PANEL DE ESTADÍSTICAS */}
          <div className="bg-zinc-900/40 border border-zinc-900 rounded-3xl p-6 shadow-xl backdrop-blur-sm flex flex-col gap-4">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-purple-500" />
              <h2 className="text-sm font-bold text-white uppercase tracking-wider">Métricas y Rendimiento</h2>
            </div>
            
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="bg-zinc-950 p-4 rounded-2xl border border-zinc-900/80 flex flex-col gap-1">
                <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider">Latencia Total</span>
                <span className="text-xl font-bold text-purple-400">
                  {hasSearched ? `${stats.time} ms` : "--"}
                </span>
                <span className="text-[9px] text-zinc-600">Red + query + render</span>
              </div>
              <div className="bg-zinc-950 p-4 rounded-2xl border border-zinc-900/80 flex flex-col gap-1">
                <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider">Tiempo Query</span>
                <span className="text-xl font-bold text-emerald-400">
                  {hasSearched && approach === "custom" ? `${stats.queryMs} ms` : "--"}
                </span>
                <span className="text-[9px] text-zinc-600">Solo búsqueda en índice</span>
              </div>
              <div className="bg-zinc-950 p-4 rounded-2xl border border-zinc-900/80 flex flex-col gap-1">
                <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider">RAM Índice</span>
                <span className="text-xl font-bold text-purple-400">
                  {hasSearched && approach === "custom" ? `${stats.indexMb} MB` : "--"}
                </span>
                <span className="text-[9px] text-zinc-600">Histogramas en memoria</span>
              </div>
              <div className="bg-zinc-950 p-4 rounded-2xl border border-zinc-900/80 flex flex-col gap-1">
                <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider">Comparaciones</span>
                <span className="text-xl font-bold text-emerald-400">
                  {hasSearched && approach === "custom" ? stats.comparisons.toLocaleString() : "--"}
                </span>
                <span className="text-[9px] text-zinc-600">Vectores comparados · dim {hasSearched && approach === "custom" ? stats.vectorDim : "–"}</span>
              </div>
            </div>
          </div>

          {/* PANEL DE RESULTADOS */}
          <div className="bg-zinc-900/40 border border-zinc-900 rounded-3xl p-6 shadow-xl flex-1 backdrop-blur-sm flex flex-col gap-4">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <Database className="h-4 w-4 text-purple-500" />
              Resultados de la Recuperación
            </h2>

            {/* Estado Inicial */}
            {!hasSearched && !isLoading && (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-8 gap-3 border border-dashed border-zinc-800 rounded-2xl">
                <Search className="h-8 w-8 text-zinc-700 animate-pulse" />
                <div>
                  <span className="block text-sm font-semibold text-zinc-400">Sin consultas ejecutadas</span>
                  <span className="block text-xs text-zinc-600 mt-1">Configura la query a la izquierda y presiona Ejecutar</span>
                </div>
              </div>
            )}

            {/* Estado de Carga */}
            {isLoading && (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-8 gap-3">
                <div className="h-8 w-8 border-3 border-purple-500/20 border-t-purple-500 rounded-full animate-spin" />
                <span className="text-xs text-zinc-500">Consultando índice invertido y calculando distancias...</span>
              </div>
            )}

            {/* Mostrar Resultados */}
            {hasSearched && !isLoading && (
              <div className="flex flex-col gap-4">
                {/* Resultados Texto */}
                {modality === "text" && results.map((item, idx) => (
                  <div key={item.id} className="bg-zinc-950 p-5 rounded-2xl border border-zinc-900 hover:border-zinc-800 transition-all flex flex-col gap-2">
                    <div className="flex items-center justify-between gap-4">
                      <span className="text-[10px] bg-purple-950/40 text-purple-400 font-bold px-2 py-0.5 rounded-md border border-purple-900/50">
                        #{idx + 1} · {item.id}
                      </span>
                      <span className="text-xs font-bold text-emerald-400">
                        {(item.similarity * 100).toFixed(2)}% Similitud
                      </span>
                    </div>
                    <h3 className="text-sm font-bold text-white mt-1">{item.title}</h3>
                    <span className="text-[10px] text-zinc-500 font-medium">{item.category}</span>
                    <p className="text-xs text-zinc-400 leading-relaxed mt-2 border-l-2 border-purple-600 pl-3">
                      {item.snippet}
                    </p>
                  </div>
                ))}

                {/* Resultados Imagen */}
                {modality === "image" && (
                  <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
                    {results.map((item, idx) => (
                      <div key={item.id} className="bg-zinc-950 rounded-xl overflow-hidden border border-zinc-800 hover:border-purple-700 transition-all flex flex-col group">
                        {/* Contenedor de imagen: fondo blanco, tamaño fijo 200px que no estira thumbnails pequeños */}
                        <div className="relative bg-white flex items-center justify-center" style={{ height: "200px" }}>
                          <img
                            src={item.imgUrl}
                            alt={item.title}
                            style={{ maxWidth: "100%", maxHeight: "100%", width: "auto", height: "auto" }}
                          />
                          <span className="absolute top-2 left-2 text-[9px] bg-purple-600 text-white font-bold px-2 py-0.5 rounded-md shadow-md">
                            Rank #{idx + 1}
                          </span>
                          <span className="absolute bottom-2 right-2 text-[10px] font-bold text-emerald-400 bg-zinc-950/85 px-2 py-0.5 rounded-full">
                            {(item.similarity * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="p-3 flex flex-col gap-0.5">
                          <span className="text-[9px] text-zinc-500 uppercase font-bold tracking-wider">{item.category}</span>
                          <h3 className="text-xs font-bold text-white line-clamp-1">{item.title}</h3>
                          {item.price && (
                            <span className="text-xs font-black text-white mt-0.5">{item.price}</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Resultados Audio */}
                {modality === "audio" && results.map((item, idx) => (
                  <div key={item.id} className="bg-zinc-950 p-4 rounded-2xl border border-zinc-900 hover:border-zinc-800 transition-all flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                      <div className="h-10 w-10 bg-purple-950/40 border border-purple-900/50 text-purple-400 rounded-xl flex items-center justify-center">
                        <Music className="h-5 w-5" />
                      </div>
                      <div>
                        <h3 className="text-xs font-bold text-white flex items-center gap-2">
                          <span className="text-purple-400">#{idx + 1}</span> {item.title}
                        </h3>
                        <span className="text-[10px] text-zinc-500 mt-0.5 block">{item.artist} · {item.genres}</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <span className="text-[10px] text-zinc-600 font-mono">{item.duration}</span>
                      <span className="text-xs font-bold text-emerald-400">
                        {(item.similarity * 100).toFixed(1)}%
                      </span>
                      <button
                        onClick={() => item.audio_url && handlePlayAudio(item.id, item.audio_url)}
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
                ))}
              </div>
            )}
          </div>
        </section>
      </main>

      {/* FOOTER */}
      <footer className="border-t border-zinc-900 bg-zinc-950 py-6 mt-12">
        <div className="max-w-7xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-zinc-600">
          <span>© 2026 UTEC · Escuela de Ciencias de la Computación</span>
          <span>Desarrollado para la asignatura de Base de Datos II</span>
        </div>
      </footer>
    </div>
  );
}
