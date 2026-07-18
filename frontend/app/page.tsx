"use client";

import React, { useState } from "react";
import { AppFooter } from "./components/AppFooter";
import { AppHeader } from "./components/AppHeader";
import { QueryPanel } from "./components/QueryPanel";
import { ResultsPanel } from "./components/ResultsPanel";
import { StatsPanel } from "./components/StatsPanel";
import { TextResultModal } from "./components/TextResultModal";
import type { ApiResult, Approach, Modality, ResultItem, SearchResponse } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [modality, setModality] = useState<Modality>("text");
  const [approach, setApproach] = useState<Approach>("custom");
  const [queryText, setQueryText] = useState("");
  const [topN, setTopN] = useState(5);
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [audioFile, setAudioFile] = useState<string | null>(null);
  const [audioFileRaw, setAudioFileRaw] = useState<File | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [mfccReady, setMfccReady] = useState(false);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [results, setResults] = useState<ResultItem[]>([]);
  const [selectedTextResult, setSelectedTextResult] = useState<ResultItem | null>(null);
  const [stats, setStats] = useState({
    time: 0,
    queryMs: 0,
    indexMb: 0,
    comparisons: 0,
    vectorDim: 0,
  });

  const audioPlayerRef = React.useRef<HTMLAudioElement | null>(null);
  const audioContextRef = React.useRef<AudioContext | null>(null);
  const streamRef = React.useRef<MediaStream | null>(null);
  const samplesRef = React.useRef<Float32Array[]>([]);
  const processorRef = React.useRef<ScriptProcessorNode | null>(null);

  React.useEffect(() => {
    fetch(`${API_URL}/pipeline/status`)
      .then((response) => response.json())
      .then((data) => setMfccReady(data?.audio_mfcc?.ready ?? false))
      .catch(() => {});
  }, []);

  React.useEffect(() => {
    return () => {
      audioPlayerRef.current?.pause();
    };
  }, []);

  const resetSearch = () => {
    setResults([]);
    setHasSearched(false);
    setSelectedTextResult(null);
  };

  const handleModalityChange = (nextModality: Modality) => {
    setModality(nextModality);
    setHasSearched(false);
    setSelectedTextResult(null);
  };

  const handlePdfSelected = (file: File) => {
    setPdfFile(file);
    setQueryText("");
    resetSearch();
  };

  const clearPdf = () => {
    setPdfFile(null);
    resetSearch();
  };

  const clearImage = () => {
    setImagePreview(null);
    setImageFile(null);
    resetSearch();
  };

  const clearAudio = () => {
    audioPlayerRef.current?.pause();
    setPlayingId(null);
    setAudioFile(null);
    setAudioFileRaw(null);
    resetSearch();
  };

  const encodeWAV = (samples: Float32Array[], sampleRate: number): Blob => {
    const allSamples = new Float32Array(samples.reduce((n, sample) => n + sample.length, 0));
    let offset = 0;
    for (const chunk of samples) {
      allSamples.set(chunk, offset);
      offset += chunk.length;
    }

    const buffer = new ArrayBuffer(44 + allSamples.length * 2);
    const view = new DataView(buffer);
    const writeString = (position: number, text: string) => {
      for (let i = 0; i < text.length; i += 1) {
        view.setUint8(position + i, text.charCodeAt(i));
      }
    };

    writeString(0, "RIFF");
    view.setUint32(4, 36 + allSamples.length * 2, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, allSamples.length * 2, true);

    let writeOffset = 44;
    for (let i = 0; i < allSamples.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, allSamples[i]));
      view.setInt16(writeOffset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      writeOffset += 2;
    }

    return new Blob([buffer], { type: "audio/wav" });
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      streamRef.current = stream;

      const context = new AudioContext({ sampleRate: 22050 });
      audioContextRef.current = context;
      await context.resume();

      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);
      samplesRef.current = [];
      processor.onaudioprocess = (event) => {
        samplesRef.current.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      processor.connect(context.destination);
      processorRef.current = processor;
      setIsRecording(true);
    } catch {
      alert("No se pudo acceder al micrófono. Verifica los permisos del navegador.");
    }
  };

  const stopRecording = () => {
    processorRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    const sampleRate = audioContextRef.current?.sampleRate ?? 22050;
    audioContextRef.current?.close();

    const totalSamples = samplesRef.current.reduce((n, sample) => n + sample.length, 0);
    if (totalSamples < sampleRate * 0.5) {
      alert("La grabación quedó vacía o muy corta. Verifica el micrófono e intenta de nuevo.");
      setIsRecording(false);
      return;
    }

    const wav = encodeWAV(samplesRef.current, sampleRate);
    const file = new File([wav], "grabacion.wav", { type: "audio/wav" });
    setAudioFile("grabacion.wav");
    setAudioFileRaw(file);
    setIsRecording(false);
  };

  const handlePlayAudio = (id: string, audioUrl: string) => {
    if (playingId === id) {
      audioPlayerRef.current?.pause();
      setPlayingId(null);
      return;
    }

    audioPlayerRef.current?.pause();
    const fullUrl = audioUrl.startsWith("blob:") ? audioUrl : `${API_URL}${audioUrl}`;
    const player = new Audio(fullUrl);
    audioPlayerRef.current = player;
    player.onended = () => setPlayingId(null);
    player.onerror = () => {
      console.error("No se pudo reproducir el audio:", fullUrl, player.error);
      alert(`No se pudo reproducir el audio (código ${player.error?.code}). Revisa la consola.`);
      setPlayingId(null);
    };
    player.play().catch((error) => {
      console.error("play() rechazado:", error);
      alert(`No se pudo iniciar la reproducción: ${error.name} - ${error.message}`);
      setPlayingId(null);
    });
    setPlayingId(id);
  };

  const applyStats = (
    raw: { query_ms?: number; index_mb?: number; n_comparisons?: number; vector_dim?: number },
    elapsed: number,
  ) => {
    setStats({
      time: elapsed,
      queryMs: raw.query_ms ?? 0,
      indexMb: raw.index_mb ?? 0,
      comparisons: raw.n_comparisons ?? 0,
      vectorDim: raw.vector_dim ?? 0,
    });
  };

  const mapTextResults = (data: SearchResponse) => (data.results ?? []).map((result: ApiResult) => ({
    id: result.chunk_id,
    title: result.metadata.title || result.chunk_id,
    similarity: result.score,
    category: result.metadata.source || "AG News",
    snippet: result.metadata.snippet || "",
    fullText: result.metadata.content || result.metadata.snippet || "",
  }));

  const mapImageResults = (data: SearchResponse) => (data.results ?? []).map((result: ApiResult) => ({
    id: result.chunk_id,
    title: result.metadata.title || result.chunk_id,
    similarity: result.score,
    category: "Imagen",
    imgUrl: `${API_URL}${result.metadata.image_url}`,
  }));

  const mapAudioResults = (data: SearchResponse) => (data.results ?? []).map((result: ApiResult) => ({
    id: result.chunk_id,
    title: result.metadata.title || result.chunk_id,
    artist: result.metadata.genre || "",
    similarity: result.score,
    duration: "–",
    genres: result.metadata.genre || "",
    audio_url: result.metadata.audio_url || null,
  }));

  const handleSearch = async () => {
    setIsLoading(true);
    setHasSearched(false);
    setSelectedTextResult(null);
    const startedAt = performance.now();

    try {
      if (approach === "custom" && modality === "text" && pdfFile) {
        const form = new FormData();
        form.append("file", pdfFile);
        const response = await fetch(`${API_URL}/pipeline/search/text/pdf?k=${topN}`, { method: "POST", body: form });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = (await response.json()) as SearchResponse;
        setResults(mapTextResults(data));
        applyStats(data.stats ?? {}, elapsedMs(startedAt));
      } else if (approach === "custom" && modality === "text") {
        const response = await fetch(`${API_URL}/pipeline/search/text`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: queryText, k: topN }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = (await response.json()) as SearchResponse;
        setResults(mapTextResults(data));
        applyStats(data.stats ?? {}, elapsedMs(startedAt));
      } else if (approach === "custom" && modality === "image" && imageFile) {
        const form = new FormData();
        form.append("file", imageFile);
        const response = await fetch(`${API_URL}/pipeline/search/image?k=${topN}`, { method: "POST", body: form });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = (await response.json()) as SearchResponse;
        setResults(mapImageResults(data));
        applyStats(data.stats ?? {}, elapsedMs(startedAt));
      } else if (approach === "custom" && modality === "audio" && audioFileRaw) {
        const form = new FormData();
        form.append("file", audioFileRaw);
        const response = await fetch(`${API_URL}/pipeline/search/audio-file?k=${topN}`, { method: "POST", body: form });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = (await response.json()) as SearchResponse;
        setResults(mapAudioResults(data));
        applyStats(data.stats ?? {}, elapsedMs(startedAt));
      } else if (approach === "postgres" && modality === "text") {
        const response = await fetch(`${API_URL}/postgres/search/text`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: queryText, k: topN }),
        });
        const data = (await response.json()) as SearchResponse;
        setResults(mapTextResults(data));
        applyStats({ query_ms: data.stats?.query_ms, n_comparisons: data.stats?.n_comparisons }, elapsedMs(startedAt));
      } else if (approach === "postgres" && modality === "image" && imageFile) {
        const form = new FormData();
        form.append("file", imageFile);
        const response = await fetch(`${API_URL}/postgres/search/image?k=${topN}`, { method: "POST", body: form });
        const data = (await response.json()) as SearchResponse;
        setResults(mapImageResults(data));
        applyStats({ query_ms: data.stats?.query_ms, n_comparisons: data.stats?.n_comparisons }, elapsedMs(startedAt));
      } else if (approach === "postgres" && modality === "audio" && audioFileRaw) {
        const form = new FormData();
        form.append("file", audioFileRaw);
        const response = await fetch(`${API_URL}/postgres/search/audio?k=${topN}`, { method: "POST", body: form });
        const data = (await response.json()) as SearchResponse;
        setResults(mapAudioResults(data));
        applyStats({ query_ms: data.stats?.query_ms, n_comparisons: data.stats?.n_comparisons }, elapsedMs(startedAt));
      } else {
        setResults([]);
        setStats({ time: elapsedMs(startedAt), queryMs: 0, indexMb: 0, comparisons: 0, vectorDim: 0 });
      }
    } catch (error) {
      console.error("Error en búsqueda:", error);
      setResults([]);
    } finally {
      setIsLoading(false);
      setHasSearched(true);
    }
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    resetSearch();
  };

  const handleAudioUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setAudioFile(file.name);
    setAudioFileRaw(file);
    resetSearch();
  };

  const loadSampleImage = async (url: string, filename: string) => {
    try {
      const response = await fetch(url);
      const blob = await response.blob();
      const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
      setImageFile(file);
      setImagePreview(URL.createObjectURL(blob));
    } catch {
      setImagePreview(url);
    }
    resetSearch();
  };

  return (
    <div className="app-shell flex flex-col font-sans antialiased selection:bg-purple-600 selection:text-white">
      <AppHeader />

      <main className="max-w-7xl mx-auto px-6 py-8 flex-1 grid grid-cols-1 lg:grid-cols-12 gap-8 w-full">
        <QueryPanel
          modality={modality}
          approach={approach}
          queryText={queryText}
          topN={topN}
          pdfFile={pdfFile}
          imagePreview={imagePreview}
          audioFile={audioFile}
          audioFileRaw={audioFileRaw}
          isRecording={isRecording}
          mfccReady={mfccReady}
          playingId={playingId}
          isLoading={isLoading}
          onModalityChange={handleModalityChange}
          onApproachChange={setApproach}
          onQueryTextChange={setQueryText}
          onTopNChange={setTopN}
          onPdfSelected={handlePdfSelected}
          onClearPdf={clearPdf}
          onImageUpload={handleImageUpload}
          onLoadSampleImage={loadSampleImage}
          onClearImage={clearImage}
          onAudioUpload={handleAudioUpload}
          onClearAudio={clearAudio}
          onStartRecording={startRecording}
          onStopRecording={stopRecording}
          onPlayAudio={handlePlayAudio}
          onSearch={handleSearch}
        />

        <section className="lg:col-span-8 flex flex-col gap-6">
          <StatsPanel
            approach={approach}
            modality={modality}
            hasSearched={hasSearched}
            stats={stats}
          />
          <ResultsPanel
            modality={modality}
            results={results}
            hasSearched={hasSearched}
            isLoading={isLoading}
            playingId={playingId}
            onPlayAudio={handlePlayAudio}
            onSelectTextResult={setSelectedTextResult}
          />
        </section>
      </main>

      <AppFooter />

      {selectedTextResult && (
        <TextResultModal
          result={selectedTextResult}
          onClose={() => setSelectedTextResult(null)}
        />
      )}
    </div>
  );
}

function elapsedMs(startedAt: number) {
  return parseFloat((performance.now() - startedAt).toFixed(2));
}
