'use client';

import { motion } from 'framer-motion';
import { useEffect, useRef } from 'react';

interface StreamWindowProps {
  isLoading: boolean;
  content: string;
}

export function StreamWindow({ isLoading, content }: StreamWindowProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [content]);

  return (
    <motion.div
      className="flex flex-col h-full bg-black/40 border border-cyan-500/40 rounded-lg overflow-hidden"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4 }}
    >
      {/* Header */}
      <div className="border-b border-cyan-500/20 px-6 py-4 flex items-center gap-2 bg-black/60">
        <motion.div
          className="w-3 h-3 rounded-full bg-cyan-500"
          animate={{
            opacity: isLoading ? [1, 0.5, 1] : 1,
            boxShadow: isLoading
              ? [
                  '0 0 6px rgba(6, 182, 212, 0.5)',
                  '0 0 12px rgba(6, 182, 212, 0.8)',
                  '0 0 6px rgba(6, 182, 212, 0.5)',
                ]
              : '0 0 6px rgba(6, 182, 212, 0.5)',
          }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
        <span className="text-cyan-400 font-terminal text-sm uppercase tracking-wider">
          {isLoading ? 'STREAMING...' : 'STREAM_COMPLETE'}
        </span>
      </div>

      {/* Content area */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-cyan-500/20 scrollbar-track-transparent"
      >
        <div className="px-6 py-6 font-terminal text-sm text-gray-100 space-y-1 leading-relaxed whitespace-pre-wrap">
          {content ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.2 }}
            >
              {content}
              {isLoading && (
                <motion.span
                  animate={{ opacity: [1, 0.3, 1] }}
                  transition={{ duration: 0.8, repeat: Infinity }}
                  className="inline-block text-cyan-400"
                >
                  ▌
                </motion.span>
              )}
            </motion.div>
          ) : isLoading ? (
            <motion.div
              className="space-y-2 text-amber-400"
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <div>[DECRYPTING DATABASE...]</div>
              <div className="text-gray-500">&gt; Loading query results</div>
              <div className="text-gray-500">&gt; Parsing documents</div>
              <div className="text-gray-500">&gt; Streaming response</div>
            </motion.div>
          ) : (
            <div className="text-gray-600">&gt; Awaiting query...</div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-cyan-500/20 px-6 py-3 bg-black/60 flex items-center justify-between text-xs text-gray-500 font-terminal">
        <span>{content.length} chars</span>
        <span className="text-cyan-600">
          <motion.span
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            ▓▓▓▓▓▓▓▓▓▓
          </motion.span>
        </span>
      </div>
    </motion.div>
  );
}
