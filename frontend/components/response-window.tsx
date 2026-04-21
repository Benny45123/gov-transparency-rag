'use client';

import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { useEffect, useRef } from 'react';

interface ResponseWindowProps {
  isLoading: boolean;
  response: string;
  error?: string | null;
}

export function ResponseWindow({
  isLoading,
  response,
  error,
}: ResponseWindowProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when response updates
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [response]);

  return (
    <motion.div
      className="w-full max-w-4xl mx-auto mt-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.5, duration: 0.4 }}
    >
      <div className="glass rounded-lg border border-cyan-500/30 overflow-hidden">
        {/* Header */}
        <div className="border-b border-cyan-500/20 px-6 py-4 flex items-center justify-between bg-black/40">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-cyan-500 animate-pulse" />
            <span className="text-cyan-400 font-terminal text-sm uppercase tracking-wider">
              {isLoading ? 'Decrypting...' : 'Response'}
            </span>
          </div>
          <span className="text-gray-500 font-terminal text-xs">
            {response.length} chars
          </span>
        </div>

        {/* Content area */}
        <div
          ref={scrollRef}
          className="h-96 overflow-y-auto px-6 py-6 font-terminal text-sm scrollbar-thin scrollbar-thumb-cyan-500/20 scrollbar-track-transparent"
        >
          {error ? (
            <motion.div
              className="text-red-400 font-terminal"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <div className="text-red-500 mb-2">[ERROR]</div>
              <p>{error}</p>
            </motion.div>
          ) : isLoading && !response ? (
            <motion.div
              className="space-y-2"
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <div className="text-amber-400">[INITIALIZING DATABASE QUERY...]</div>
              <div className="text-gray-500">&gt; Accessing classified files</div>
              <div className="text-gray-500">&gt; Cross-referencing documents</div>
              <div className="text-gray-500">&gt; Compiling results</div>
              <div className="mt-4 inline-block">
                <motion.span
                  animate={{ opacity: [1, 0.3, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                  className="text-cyan-400"
                >
                  _
                </motion.span>
              </div>
            </motion.div>
          ) : response ? (
            <motion.div
              className="prose prose-invert max-w-none"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
            >
              <div className="text-green-400 mb-4 font-terminal text-xs">
                [QUERY EXECUTED SUCCESSFULLY]
              </div>
              <div className="text-gray-100 space-y-4 font-terminal leading-relaxed text-xs">
                <ReactMarkdown
                  components={{
                    p: ({ children }) => (
                      <p className="text-gray-300 mb-3">{children}</p>
                    ),
                    h1: ({ children }) => (
                      <h1 className="text-amber-400 text-lg font-bold mb-2 mt-4">
                        {children}
                      </h1>
                    ),
                    h2: ({ children }) => (
                      <h2 className="text-cyan-400 text-base font-bold mb-2 mt-3">
                        {children}
                      </h2>
                    ),
                    h3: ({ children }) => (
                      <h3 className="text-green-400 font-bold mb-1 mt-2">
                        {children}
                      </h3>
                    ),
                    ul: ({ children }) => (
                      <ul className="list-disc list-inside space-y-1 mb-3">
                        {children}
                      </ul>
                    ),
                    li: ({ children }) => <li className="text-gray-300">{children}</li>,
                    code: ({ children }) => (
                      <code className="bg-black/60 px-2 py-1 rounded text-green-400 text-xs font-bold">
                        {children}
                      </code>
                    ),
                    blockquote: ({ children }) => (
                      <blockquote className="border-l-2 border-amber-500 pl-4 italic text-gray-400 my-3">
                        {children}
                      </blockquote>
                    ),
                  }}
                >
                  {response}
                </ReactMarkdown>
              </div>
            </motion.div>
          ) : (
            <motion.div
              className="text-center text-gray-500 font-terminal py-12"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <div className="text-lg mb-2">AWAITING INPUT</div>
              <div className="text-xs">Submit a query to begin analysis</div>
            </motion.div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-cyan-500/20 px-6 py-3 bg-black/40 flex items-center justify-between text-xs text-gray-500 font-terminal">
          <span>
            {isLoading
              ? 'Processing...'
              : response
                ? 'Complete'
                : 'Ready'}
          </span>
          <span className="text-cyan-600">▓▓▓▓▓▓▓▓▓▓</span>
        </div>
      </div>
    </motion.div>
  );
}
