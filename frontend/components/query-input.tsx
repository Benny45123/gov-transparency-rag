'use client';

import { motion } from 'framer-motion';
import { useState } from 'react';

interface QueryInputProps {
  onSubmit: (query: string) => void;
  isLoading: boolean;
  placeholder?: string;
  isLoadingSources?: boolean;
}

export function QueryInput({
  onSubmit,
  isLoading,
  placeholder = 'Ask about the case files...',
  isLoadingSources = false,
}: QueryInputProps) {
  const [input, setInput] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSubmit(input);
      setInput('');
    }
  };

  return (
    <motion.form
      onSubmit={handleSubmit}
      className="w-full max-w-3xl mx-auto"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4, duration: 0.4 }}
    >
      <div 
        className={`relative glass rounded-lg overflow-hidden group transition-all duration-300 ${
          isLoadingSources 
            ? 'border border-cyan-500/50 shadow-[0_0_15px_rgba(6,182,212,0.1)]' 
            : 'border border-amber-500/30 hover:border-amber-500/50 shadow-[0_0_15px_rgba(251,191,36,0.05)]'
        }`}
      >
        {/* Container for the Left-side Prompt elements */}
        <div className="absolute left-4 top-1/2 -translate-y-1/2 flex items-center gap-2 pointer-events-none z-10">
          <span className="text-amber-400 font-terminal text-lg font-bold leading-none">
            &gt;
          </span>
          {/* Blinking Cursor - Only shows when input is empty */}
          {!input && (
            <motion.span 
              animate={{ opacity: [1, 0, 1] }}
              transition={{ duration: 0.8, repeat: Infinity }}
              className="text-amber-400 font-terminal text-lg leading-none mt-1"
            >
              _
            </motion.span>
          )}
        </div>
  
        {/* Input field - Padding adjusted to start exactly after the cursor */}
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={input ? "" : placeholder} // Hide placeholder if user starts typing
          disabled={isLoading}
          className="w-full bg-black/20 pl-14 pr-24 py-5 text-gray-100 placeholder-gray-700 focus:outline-none font-terminal text-sm tracking-tight"
        />
  
        {/* Action Buttons Container */}
        <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-3">
          {isLoading && (
            <motion.div
              className="w-4 h-4"
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
            >
              <div className="w-full h-full border-2 border-transparent border-t-cyan-400 border-r-cyan-400 rounded-full" />
            </motion.div>
          )}
          
          <motion.button
            type="submit"
            disabled={isLoading || !input.trim()}
            className={`p-2 rounded font-terminal font-bold text-xs tracking-tighter transition-colors ${
              isLoadingSources 
                ? 'text-cyan-400 hover:text-cyan-200' 
                : 'text-amber-500 hover:text-amber-300'
            }`}
            whileHover={!isLoading ? { scale: 1.05 } : {}}
            whileTap={!isLoading ? { scale: 0.95 } : {}}
          >
            {isLoading ? 'PROCESSING' : 'SEARCH'}
          </motion.button>
        </div>
      </div>
  
      {/* Footer Help Text */}
      <motion.p
        className="text-center text-[10px] text-gray-600 mt-3 font-terminal uppercase tracking-[0.2em]"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
      >
        Enter to send <span className="mx-2 text-gray-800">•</span> Shift+Enter for details <span className="mx-2 text-gray-800">•</span> Ctrl+K to clear
      </motion.p>
    </motion.form>
  );
}
