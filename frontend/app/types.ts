export type Modality = "text" | "image" | "audio";

export type Approach = "custom" | "postgres";

export type ApiMetadata = {
  title?: string;
  source?: string;
  snippet?: string;
  content?: string;
  image_url?: string;
  genre?: string;
  audio_url?: string | null;
};

export type ApiResult = {
  chunk_id: string;
  score: number;
  metadata: ApiMetadata;
};

export type ApiStats = {
  query_ms?: number;
  index_mb?: number;
  n_comparisons?: number;
  vector_dim?: number;
};

export type SearchResponse = {
  results?: ApiResult[];
  stats?: ApiStats;
};

export type ResultItem = {
  id: string;
  title: string;
  similarity: number;
  category?: string;
  snippet?: string;
  fullText?: string;
  imgUrl?: string;
  price?: string;
  artist?: string;
  duration?: string;
  genres?: string;
  audio_url?: string | null;
};

export type UiStats = {
  time: number;
  queryMs: number;
  indexMb: number;
  comparisons: number;
  vectorDim: number;
};
