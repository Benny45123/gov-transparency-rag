'use client';

import { motion, AnimatePresence } from 'framer-motion';

export interface Source {
  source_file: string;
  source_url: string;
  chunk_index: number;
  score: number;
  preview: string;
}

interface CitationsPaneProps {
  sources: Source[];
  isLoading: boolean;
}

export function CitationsPane({ sources, isLoading }: CitationsPaneProps) {
  return (
    <motion.div
      className="flex flex-col h-full bg-black/40 border border-amber-500/40 rounded-lg overflow-hidden"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4 }}
    >
      {/* Header */}
      <div className="border-b border-amber-500/20 px-6 py-4 flex items-center gap-2 bg-black/60">
        <motion.div
          className="w-3 h-3 rounded-full bg-amber-400"
          animate={{
            opacity: isLoading ? [1, 0.5, 1] : 1,
            boxShadow: [
              '0 0 6px rgba(251, 191, 36, 0.5)',
              '0 0 12px rgba(251, 191, 36, 0.8)',
              '0 0 6px rgba(251, 191, 36, 0.5)',
            ],
          }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
        <span className="text-amber-400 font-terminal text-sm uppercase tracking-wider">
          DEEP_STUDY
        </span>
        <span className="text-gray-600 font-terminal text-xs ml-auto">
          {sources.length} source{sources.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Content area */}
      <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-amber-500/20 scrollbar-track-transparent">
        <div className="px-6 py-6 space-y-4">
          <AnimatePresence mode="popLayout">
            {sources.length > 0 ? (
              sources.map((source, index) => (
                <motion.div
                  key={`${source.source_file}-${source.chunk_index}`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ delay: index * 0.08 }}
                  className="group relative"
                >
                  {/* Glowing border effect */}
                  <motion.div
                    className="absolute inset-0 bg-gradient-to-r from-amber-500/0 via-amber-500/10 to-amber-500/0 rounded-lg pointer-events-none"
                    animate={{
                      opacity: [0, 0.5, 0],
                    }}
                    transition={{
                      duration: 2,
                      delay: index * 0.1,
                      repeat: Infinity,
                    }}
                  />

                  <motion.div
                    className="border border-amber-500/30 rounded-lg p-4 bg-black/40 hover:bg-amber-500/5 transition group-hover:border-amber-500/60"
                    whileHover={{ scale: 1.02 }}
                  >
                    {/* Relevance score */}
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <p className="text-amber-400 font-terminal text-xs font-bold uppercase tracking-wider mb-1">
                          {source.source_file.replace('.pdf', '')}
                        </p>
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 w-12 bg-gray-700 rounded-full overflow-hidden">
                            <motion.div
                              className="h-full bg-gradient-to-r from-amber-500 to-amber-400"
                              initial={{ width: 0 }}
                              animate={{ width: `${Math.round(source.score * 100)}%` }}
                              transition={{ delay: index * 0.1 + 0.3, duration: 0.6 }}
                            />
                          </div>
                          <span className="text-gray-500 font-terminal text-xs">
                            {Math.round(source.score * 100)}%
                          </span>
                        </div>
                      </div>
                      <span className="text-gray-600 font-terminal text-xs">
                        Chunk {source.chunk_index}
                      </span>
                    </div>

                    {/* Preview */}
                    <p className="text-gray-300 font-terminal text-xs leading-relaxed mb-4 line-clamp-4">
                      {source.preview.replace(/\[CONTEXT:[^\]]*\]/g, '').trim()}
                    </p>

                    {/* Link button */}
                    <motion.a
                      href={source.source_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-2 text-amber-400 hover:text-amber-300 font-terminal text-xs font-bold uppercase tracking-wider transition"
                      whileHover={{ x: 2 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      VIEW_DOCUMENT
                      <svg
                        width="12"
                        height="12"
                        viewBox="0 0 12 12"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.5"
                      >
                        <path d="M2 10L10 2M10 2H2M10 2V10" />
                      </svg>
                    </motion.a>
                  </motion.div>
                </motion.div>
              ))
            ) : isLoading ? (
              <motion.div
                className="space-y-4"
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                {[1, 2, 3].map((i) => (
                  <div
                    key={i}
                    className="h-32 bg-gradient-to-r from-amber-500/10 via-transparent to-amber-500/10 rounded-lg"
                  />
                ))}
              </motion.div>
            ) : (
              <motion.div
                className="text-center py-12 text-gray-600 font-terminal text-xs"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <div className="mb-2">NO_SOURCES</div>
                <div className="text-gray-700">Submit a query to view citations</div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-amber-500/20 px-6 py-3 bg-black/60 text-xs text-gray-500 font-terminal">
        <span>
          {isLoading ? 'LOADING_CITATIONS...' : 'CITATIONS_READY'}
        </span>
      </div>
    </motion.div>
  );
}
