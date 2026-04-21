'use client';

import { motion } from 'framer-motion';

export function Header() {
  return (
    <motion.header
      className="w-full border-b border-amber-500/20 bg-black/80 backdrop-blur-md z-30"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <div className="max-w-[1600px] mx-auto px-6 py-4">
        <div className="flex items-start justify-between">
          
          {/* Left Side: Brand & Subtitle */}
          <div className="flex items-start gap-4">
            {/* Animated Logo Icon */}
            <motion.div
              className="relative w-12 h-12 border-2 border-amber-500 rounded flex items-center justify-center bg-amber-500/5 mt-1"
              animate={{
                boxShadow: [
                  'inset 0 0 0px rgba(251, 191, 36, 0)',
                  'inset 0 0 10px rgba(251, 191, 36, 0.2)',
                  'inset 0 0 0px rgba(251, 191, 36, 0)',
                ],
              }}
              transition={{ duration: 4, repeat: Infinity }}
            >
              <div className="w-6 h-6 border border-amber-500/50 rounded-sm rotate-45" />
              <div className="absolute inset-0 border border-amber-500/20 scale-75 animate-pulse" />
            </motion.div>

            <div className="space-y-1">
              <div className="flex items-center gap-3">
                <h1 className="text-2xl font-bold text-amber-500 font-terminal uppercase tracking-[0.15em]">
                  Research Terminal
                </h1>
                <span className="px-2 py-0.5 border border-amber-500/30 rounded text-[9px] text-amber-500 font-terminal">
                  v2.0.4-LOCKED
                </span>
              </div>
              
              <div className="flex items-center gap-3 text-gray-500 font-terminal text-[11px] uppercase tracking-widest">
                <span className="text-amber-500/60 font-bold">INTEL_NODE: 0812</span>
                <span className="opacity-30">•</span>
                <span>High-fidelity investigative database</span>
                <span className="opacity-30">•</span>
                <span>Real-time document analysis</span>
              </div>
            </div>
          </div>

          {/* Right Side: System Status */}
          <div className="flex flex-col items-end space-y-2">
            <div className="flex items-center gap-3 bg-green-500/5 border border-green-500/20 px-3 py-1 rounded-sm">
              <div className="relative">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-ping absolute opacity-75" />
                <div className="w-2 h-2 rounded-full bg-green-500 relative" />
              </div>
              <span className="text-green-500 font-terminal text-[10px] font-bold uppercase tracking-widest">
                System_Ready // Connected
              </span>
            </div>
            
            <div className="flex gap-4 text-[9px] font-terminal text-gray-600 uppercase">
              <span>LATENCY: 24MS</span>
              <span>BUFFER: 100%</span>
            </div>
          </div>

        </div>
      </div>
      
      {/* Decorative Bottom Bar */}
      <div className="h-[1px] w-full bg-gradient-to-r from-transparent via-amber-500/20 to-transparent" />
    </motion.header>
  );
}