'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';

interface HistoryEntry {
  query: string;
  timestamp: string;
}

interface HistorySidebarProps {
  history: string[];
  onSelectQuery: (query: string) => void;
  onClearHistory: () => void;
  isCollapsed?: boolean;
}

export function HistorySidebar({
  history,
  onSelectQuery,
  onClearHistory,
  isCollapsed = false,
}: HistorySidebarProps) {
  const [collapsed, setCollapsed] = useState(isCollapsed);

  return (
    <motion.aside
      className="bg-black/80 border-r border-cyan-500/30 flex flex-col h-full"
      animate={{ width: collapsed ? 60 : 280 }}
      transition={{ duration: 0.3 }}
    >
      {/* Header */}
      <motion.div className="border-b border-cyan-500/20 px-4 py-4 flex items-center justify-between">
        {!collapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
          >
            <h3 className="text-amber-400 font-terminal text-xs font-bold uppercase tracking-wider">
              LEADS_HISTORY
            </h3>
            <p className="text-gray-600 font-terminal text-xs mt-1">
              {history.length} queries
            </p>
          </motion.div>
        )}
        <motion.button
          onClick={() => setCollapsed(!collapsed)}
          className="text-cyan-400 hover:text-cyan-300 transition"
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d={collapsed ? 'M6 4 L10 8 L6 12' : 'M10 4 L6 8 L10 12'} />
          </svg>
        </motion.button>
      </motion.div>

      {/* History list */}
      <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-cyan-500/20 scrollbar-track-transparent">
        <AnimatePresence mode="popLayout">
          {!collapsed && history.length > 0 ? (
            history.map((query, index) => (
              <motion.button
                key={`${query}-${index}`}
                onClick={() => onSelectQuery(query)}
                className="w-full text-left px-4 py-3 border-b border-cyan-500/10 hover:bg-cyan-500/10 transition group relative overflow-hidden"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ delay: index * 0.05 }}
                whileHover={{ x: 4 }}
              >
                {/* Glitch effect on hover */}
                <motion.div
                  className="absolute inset-0 bg-cyan-500/5 pointer-events-none"
                  initial={{ opacity: 0 }}
                  whileHover={{ opacity: 1 }}
                  transition={{ duration: 0.2 }}
                />

                {/* Cyan glow border */}
                <motion.div
                  className="absolute inset-0 border border-cyan-500/0 pointer-events-none"
                  whileHover={{ borderColor: 'rgba(6, 182, 212, 0.5)' }}
                  transition={{ duration: 0.2 }}
                />

                <p className="text-gray-300 font-terminal text-xs leading-tight relative z-10 group-hover:text-cyan-300 transition line-clamp-2">
                  {query}
                </p>
              </motion.button>
            ))
          ) : !collapsed ? (
            <motion.div
              className="px-4 py-8 text-center text-gray-600 font-terminal text-xs"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              No query history
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>

      {/* Footer */}
      {!collapsed && (
        <motion.div
          className="border-t border-cyan-500/20 px-4 py-4 space-y-2"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          <motion.button
            onClick={onClearHistory}
            disabled={history.length === 0}
            className="w-full px-3 py-2 bg-red-500/20 hover:bg-red-500/40 disabled:opacity-50 disabled:cursor-not-allowed text-red-400 hover:text-red-300 font-terminal text-xs font-bold uppercase tracking-wider rounded transition"
            whileHover={{ scale: history.length > 0 ? 1.02 : 1 }}
            whileTap={{ scale: history.length > 0 ? 0.98 : 1 }}
          >
            CLEAR_LOGS
          </motion.button>
          <div className="flex items-center gap-2 px-2">
            <motion.div
              className="w-2 h-2 rounded-full bg-green-500"
              animate={{
                opacity: [1, 0.5, 1],
                boxShadow: [
                  '0 0 4px rgba(34, 197, 94, 0.5)',
                  '0 0 8px rgba(34, 197, 94, 0.8)',
                  '0 0 4px rgba(34, 197, 94, 0.5)',
                ],
              }}
              transition={{ duration: 2, repeat: Infinity }}
            />
            <span className="text-gray-500 font-terminal text-xs">
              SYSTEM READY
            </span>
          </div>
        </motion.div>
      )}
    </motion.aside>
  );
}
