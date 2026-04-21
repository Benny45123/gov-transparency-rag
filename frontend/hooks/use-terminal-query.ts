'use client';

import { useState, useCallback } from 'react';

export interface Source {
  source_file: string;
  source_url: string;
  chunk_index: number;
  score: number;
  preview: string;
}

interface QueryResponse {
  answer: string;
  sources: Source[];
  cached: boolean;
  query: string;
  error: string | null;
}

export function useTerminalQuery(baseUrl: string = 'http://127.0.0.1:8000') {
  const [isLoading, setIsLoading] = useState(false);
  const [streamContent, setStreamContent] = useState('');
  const [sources, setSources] = useState<Source[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<string[]>([]);

  const streamQuery = useCallback(
    async (question: string, isDeep: boolean = false) => {
      setIsLoading(true);
      setError(null);
      setStreamContent('');
      setSources([]); // Clear citations for the new query

      // Add to local history
      setHistory((prev) => {
        if (prev.includes(question)) return prev;
        return [question, ...prev].slice(0, 20);
      });

      try {
        // --- PATH 1: The Stream (Always execute for the left pane) ---
        const streamPromise = (async () => {
          const queryParams = new URLSearchParams({ q: question });
          const response = await fetch(`${baseUrl}/query/stream?${queryParams}`);
          
          if (!response.ok) throw new Error('Stream path failed');

          const reader = response.body?.getReader();
          const decoder = new TextDecoder();
          let accumulated = '';

          if (reader) {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              accumulated += decoder.decode(value);
              setStreamContent(accumulated);
            }
          }
        })();

        // --- PATH 2: The Deep Study (Only execute if isDeep is true) ---
        let deepPromise = Promise.resolve();
        if (isDeep) {
          deepPromise = fetch(`${baseUrl}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, skip_cache: false }),
          })
          .then(async (res) => {
            if (!res.ok) throw new Error('Deep Study path failed');
            const data: QueryResponse = await res.json();
            setSources(data.sources || []);
            // If the stream was somehow slower, we sync the final answer here
            if (data.answer) setStreamContent(data.answer);
          });
        }

        // Wait for both to complete
        await Promise.all([streamPromise, deepPromise]);

      } catch (err) {
        setError(err instanceof Error ? err.message : 'Connection interrupted');
      } finally {
        setIsLoading(false);
      }
    },
    [baseUrl]
  );

  const fetchHistory = useCallback(async (limit: number = 20) => {
    try {
      const response = await fetch(`${baseUrl}/history?limit=${limit}`);
      if (response.ok) {
        const data = await response.json();
        setHistory(data.history || []);
      }
    } catch (err) {
      console.error("History fetch failed");
    }
  }, [baseUrl]);

  const clearHistory = useCallback(() => setHistory([]), []);

  return {
    isLoading,
    streamContent,
    sources,
    error,
    history,
    streamQuery,
    fetchHistory,
    clearHistory,
  };
}