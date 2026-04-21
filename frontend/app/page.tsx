'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Header } from '@/components/header';
import { QueryInput } from '@/components/query-input';
import { StreamWindow } from '@/components/stream-window';
import { CitationsPane } from '@/components/citations-pane';
import { HistorySidebar } from '@/components/history-sidebar';
import { GeometricGreeting } from '@/components/geometric-greeting';
import { useTerminalQuery } from '@/hooks/use-terminal-query';

export default function Home() {
  const [apiUrl] = useState('http://127.0.0.1:8000');
  const [showGreeting, setShowGreeting] = useState(true);
  
  // State for Dual-Role Selection
  const [searchMode, setSearchMode] = useState<'stream' | 'deep'>('stream');

  const { isLoading, streamContent, sources, error, history, streamQuery, clearHistory } =
    useTerminalQuery(apiUrl);

  const handleQuerySubmit = async (query: string) => {
    // Pass the searchMode to your hook to dictate which backend path to prioritize
    await streamQuery(query, searchMode === 'deep');
  };

  return (
    <main className="min-h-screen bg-black relative overflow-hidden">
      {/* CRT Scanline overlay */}
      <div className="fixed inset-0 pointer-events-none z-40 opacity-[0.03]">
        <div className="absolute inset-0 bg-repeat" style={{
          backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 1px, rgba(0, 0, 0, 1) 1px, rgba(0, 0, 0, 1) 2px)'
        }} />
      </div>

      {showGreeting && (
        <GeometricGreeting onInteraction={() => setShowGreeting(false)} />
      )}

      <Header />

      <div className="flex h-[calc(100vh-80px)]">
        <HistorySidebar
          history={history}
          onSelectQuery={(q) => streamQuery(q, searchMode === 'deep')}
          onClearHistory={clearHistory}
        />

        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-shrink-0 border-b border-cyan-500/20 p-6 space-y-6 bg-gradient-to-b from-black/80 to-black/40">
            
            {/* DUAL ROLE SELECTOR */}
            <div className="flex items-center justify-between">
              <div className="flex bg-gray-900/50 p-1 rounded-md border border-gray-800">
                <button
                  onClick={() => setSearchMode('stream')}
                  className={`px-4 py-1.5 font-terminal text-[10px] uppercase tracking-tighter transition-all rounded ${
                    searchMode === 'stream' 
                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50' 
                    : 'text-gray-500 hover:text-gray-300'
                  }`}
                >
                  ⚡ Stream_Response
                </button>
                <button
                  onClick={() => setSearchMode('deep')}
                  className={`px-4 py-1.5 font-terminal text-[10px] uppercase tracking-tighter transition-all rounded ${
                    searchMode === 'deep' 
                    ? 'bg-amber-500/20 text-amber-400 border border-amber-500/50' 
                    : 'text-gray-500 hover:text-gray-300'
                  }`}
                >
                  🔍 Deep_Study
                </button>
              </div>

              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full animate-pulse ${searchMode === 'deep' ? 'bg-amber-500' : 'bg-green-500'}`} />
                <span className="text-gray-500 font-terminal text-xs">
                  MODE: {searchMode.toUpperCase()}
                </span>
              </div>
            </div>

            <QueryInput
              onSubmit={handleQuerySubmit}
              isLoading={isLoading}
              isLoadingSources={isLoading && searchMode === 'deep'}
              placeholder={searchMode === 'deep' ? "Deep analysis of case records..." : "Quick stream query..."}
            />
          </div>

          {/* DUAL PANE VIEW */}
          <div className="flex-1 overflow-hidden flex gap-4 p-6">
            <div className={`flex-1 transition-all duration-500 ${searchMode === 'deep' ? 'opacity-40 scale-[0.98]' : 'opacity-100 scale-100'}`}>
              <StreamWindow isLoading={isLoading && searchMode === 'stream'} content={streamContent} />
            </div>

            <div className={`flex-1 transition-all duration-500 ${searchMode === 'stream' ? 'opacity-40 scale-[0.98]' : 'opacity-100 scale-100'}`}>
              <CitationsPane sources={sources} isLoading={isLoading && searchMode === 'deep'} />
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}