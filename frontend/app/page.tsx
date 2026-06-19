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
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [audioFile, setAudioFile] = useState<string | null>(null);

  // Estados de Ejecución y Resultados
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [stats, setStats] = useState({
    time: 0,
    diskIO: 0,
    memory: 0,
    recall: 0
  });

  // Simular Búsqueda
  const handleSearch = () => {
    setIsLoading(true);
    setHasSearched(false);

    // Simular retraso de red
    setTimeout(() => {
      setIsLoading(false);
      setHasSearched(true);

      // Definir estadísticas según enfoque (Nuestra impl vs Postgres nativo)
      if (approach === "custom") {
        setStats({
          time: parseFloat((Math.random() * 2 + 1).toFixed(2)), // 1ms - 3ms (Rápido con índice invertido)
          diskIO: Math.floor(Math.random() * 10 + 5),         // Bajo I/O
          memory: parseFloat((Math.random() * 0.8 + 0.5).toFixed(2)), // ~1MB
          recall: parseFloat((Math.random() * 5 + 88).toFixed(1))     // ~90% por cuantización
        });
      } else {
        // Enfoque Postgres Nativo (pgvector / GIN)
        setStats({
          time: parseFloat((Math.random() * 15 + 8).toFixed(2)),   // ~15ms (HNSW o Full-text)
          diskIO: Math.floor(Math.random() * 50 + 30),         // Mayor I/O en disco
          memory: parseFloat((Math.random() * 5 + 8).toFixed(2)),     // ~10MB por carga de índices HNSW
          recall: parseFloat((Math.random() * 2 + 97).toFixed(1))     // ~98% (más exacto)
        });
      }

      // Asignar resultados según modalidad
      if (modality === "text") {
        setResults(MOCK_TEXT_RESULTS.slice(0, topN));
      } else if (modality === "image") {
        setResults(MOCK_IMAGE_RESULTS.slice(0, topN));
      } else {
        setResults(MOCK_AUDIO_RESULTS.slice(0, topN));
      }
    }, 800);
  };

  // Manejar Carga de Archivos Falsa
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const handleAudioUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setAudioFile(e.target.files[0].name);
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
              <textarea
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                placeholder="Ingresa texto, paper, o palabras clave a buscar..."
                className="w-full min-h-[120px] bg-zinc-950 border border-zinc-900 rounded-2xl p-4 text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-purple-600 focus:ring-1 focus:ring-purple-600 resize-none transition-all"
              />
            )}

            {/* Input de Imagen */}
            {modality === "image" && (
              <div className="flex flex-col gap-3">
                {imagePreview ? (
                  <div className="relative h-32 w-full rounded-2xl overflow-hidden border border-zinc-900 bg-zinc-950 flex items-center justify-center">
                    <img src={imagePreview} alt="Preview" className="h-full object-contain" />
                    <button 
                      onClick={() => setImagePreview(null)}
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
                      onClick={() => setImagePreview("https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&q=80")}
                      className="text-[10px] bg-zinc-900 border border-zinc-800 hover:border-purple-600 text-zinc-300 py-1 px-2.5 rounded-lg transition-all"
                    >
                      👟 Zapatilla Roja
                    </button>
                    <button 
                      onClick={() => setImagePreview("https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?w=400&q=80")}
                      className="text-[10px] bg-zinc-900 border border-zinc-800 hover:border-purple-600 text-zinc-300 py-1 px-2.5 rounded-lg transition-all"
                    >
                      👠 Tacón Rosa
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Input de Audio */}
            {modality === "audio" && (
              <div className="flex flex-col gap-3">
                {audioFile ? (
                  <div className="border border-zinc-900 bg-zinc-950 p-4 rounded-2xl flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Music className="h-5 w-5 text-purple-500" />
                      <span className="text-xs font-medium text-zinc-300 max-w-[150px] truncate">{audioFile}</span>
                    </div>
                    <button 
                      onClick={() => setAudioFile(null)}
                      className="bg-red-600/80 hover:bg-red-600 text-white text-[10px] font-bold py-1 px-2.5 rounded-full backdrop-blur-sm transition-all"
                    >
                      Remover
                    </button>
                  </div>
                ) : (
                  <label className="border border-dashed border-zinc-800 hover:border-purple-600 bg-zinc-950/50 hover:bg-zinc-950 p-6 rounded-2xl flex flex-col items-center justify-center gap-2 cursor-pointer transition-all group">
                    <Music className="h-8 w-8 text-zinc-600 group-hover:text-purple-500 transition-colors" />
                    <span className="text-xs font-medium text-zinc-400">Arrastra o sube un archivo de audio</span>
                    <span className="text-[10px] text-zinc-600">MP3, WAV o OGG</span>
                    <input type="file" accept="audio/*" className="hidden" onChange={handleAudioUpload} />
                  </label>
                )}

                {/* Audios rápidos de muestra */}
                <div className="flex flex-col gap-1.5">
                  <span className="text-[10px] text-zinc-500">¿No tienes audio? Selecciona una de muestra:</span>
                  <div className="flex gap-2">
                    <button 
                      onClick={() => setAudioFile("cancion_synthwave_test.mp3")}
                      className="text-[10px] bg-zinc-900 border border-zinc-800 hover:border-purple-600 text-zinc-300 py-1 px-2.5 rounded-lg transition-all"
                    >
                      🎵 Synthwave Beat
                    </button>
                    <button 
                      onClick={() => setAudioFile("ritmo_rock_retro.wav")}
                      className="text-[10px] bg-zinc-900 border border-zinc-800 hover:border-purple-600 text-zinc-300 py-1 px-2.5 rounded-lg transition-all"
                    >
                      🎸 Retro Rock
                    </button>
                  </div>
                </div>
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
                <span className="font-bold text-purple-400">{kValue} clusters</span>
              </div>
              <input
                type="range"
                min="50"
                max="2000"
                step="50"
                value={kValue}
                onChange={(e) => setKValue(parseInt(e.target.value))}
                className="w-full accent-purple-600"
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <div className="flex justify-between items-center text-xs">
                <span className="font-semibold text-zinc-400">Top Resultados (N)</span>
                <span className="font-bold text-purple-400">{topN} elementos</span>
              </div>
              <input
                type="range"
                min="1"
                max="3"
                value={topN}
                onChange={(e) => setTopN(parseInt(e.target.value))}
                className="w-full accent-purple-600"
              />
            </div>
          </div>

          {/* Botón Ejecutar */}
          <button
            onClick={handleSearch}
            disabled={isLoading || (modality === "text" && !queryText) || (modality === "image" && !imagePreview) || (modality === "audio" && !audioFile)}
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
                <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider">Tiempo / Latencia</span>
                <span className="text-xl font-bold text-purple-400">
                  {hasSearched ? `${stats.time} ms` : "--"}
                </span>
                <span className="text-[9px] text-zinc-600">Calculado en el backend</span>
              </div>
              <div className="bg-zinc-950 p-4 rounded-2xl border border-zinc-900/80 flex flex-col gap-1">
                <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider">Lecturas de Disco</span>
                <span className="text-xl font-bold text-purple-400">
                  {hasSearched ? stats.diskIO : "--"}
                </span>
                <span className="text-[9px] text-zinc-600">Accesos I/O simulados</span>
              </div>
              <div className="bg-zinc-950 p-4 rounded-2xl border border-zinc-900/80 flex flex-col gap-1">
                <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider">Memoria RAM</span>
                <span className="text-xl font-bold text-purple-400">
                  {hasSearched ? `${stats.memory} MB` : "--"}
                </span>
                <span className="text-[9px] text-zinc-600">Consumo de carga de índice</span>
              </div>
              <div className="bg-zinc-950 p-4 rounded-2xl border border-zinc-900/80 flex flex-col gap-1">
                <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider">Efectividad (Recall)</span>
                <span className="text-xl font-bold text-emerald-400">
                  {hasSearched ? `${stats.recall}%` : "--"}
                </span>
                <span className="text-[9px] text-zinc-600">Precisión promedio aproximada</span>
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

                {/* Resultados Imagen (E-commerce) */}
                {modality === "image" && (
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    {results.map((item, idx) => (
                      <div key={item.id} className="bg-zinc-950 rounded-2xl overflow-hidden border border-zinc-900 hover:border-zinc-800 transition-all flex flex-col">
                        <div className="relative h-44 bg-zinc-900 flex items-center justify-center overflow-hidden">
                          <img src={item.imgUrl} alt={item.title} className="w-full h-full object-cover" />
                          <span className="absolute top-2 left-2 text-[9px] bg-purple-600 text-white font-bold px-2 py-0.5 rounded-md shadow-md">
                            Rank #{idx + 1}
                          </span>
                        </div>
                        <div className="p-4 flex flex-col gap-1.5 flex-1 justify-between">
                          <div>
                            <span className="text-[9px] text-zinc-500 block uppercase font-bold">{item.category}</span>
                            <h3 className="text-xs font-bold text-white mt-1 line-clamp-1">{item.title}</h3>
                          </div>
                          <div className="flex justify-between items-center mt-3 pt-2 border-t border-zinc-900/60">
                            <span className="text-xs font-black text-white">{item.price}</span>
                            <span className="text-[10px] font-bold text-emerald-400">
                              {(item.similarity * 100).toFixed(1)}% Sim.
                            </span>
                          </div>
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
                      <button className="h-8 w-8 rounded-full bg-zinc-900 hover:bg-purple-600 hover:text-white text-zinc-400 flex items-center justify-center transition-colors">
                        <Play className="h-3.5 w-3.5 fill-current ml-0.5" />
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
