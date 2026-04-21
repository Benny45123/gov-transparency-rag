'use client';

import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';

interface GeometricGreetingProps {
  onInteraction?: () => void;
}

export function GeometricGreeting({ onInteraction }: GeometricGreetingProps) {
  const [isVisible, setIsVisible] = useState(true);

  const handleInteraction = () => {
    setIsVisible(false);
    onInteraction?.();
  };

  useEffect(() => {
    const handleClick = () => handleInteraction();
    const handleKeyPress = () => handleInteraction();

    window.addEventListener('click', handleClick);
    window.addEventListener('keydown', handleKeyPress);

    return () => {
      window.removeEventListener('click', handleClick);
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, []);

  if (!isVisible) return null;

  return (
    <motion.div
      className="fixed inset-0 bg-black/95 flex items-center justify-center z-50"
      initial={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 1 }}
      onClick={handleInteraction}
    >
      <div className="absolute inset-0 flex items-center justify-center">
        <motion.svg
          width="400"
          height="400"
          viewBox="0 0 400 400"
          className="text-amber-400"
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 2, ease: 'easeOut' }}
        >
          {/* Dodecahedron wireframe */}
          <defs>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Main geometric structure - Network grid */}
          <motion.g
            animate={{ rotateZ: 360, rotateX: 20, rotateY: 20 }}
            transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
            style={{ perspective: '1200px' }}
          >
            {/* Outer pentagon wireframe */}
            <motion.line
              x1="200"
              y1="50"
              x2="350"
              y2="130"
              stroke="currentColor"
              strokeWidth="2"
              filter="url(#glow)"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 3, ease: 'easeInOut' }}
              style={{ filter: 'drop-shadow(0 0 8px rgba(251, 191, 36, 0.8))' }}
            />
            <motion.line
              x1="350"
              y1="130"
              x2="310"
              y2="310"
              stroke="currentColor"
              strokeWidth="2"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 3, ease: 'easeInOut', delay: 0.3 }}
              style={{ filter: 'drop-shadow(0 0 8px rgba(251, 191, 36, 0.8))' }}
            />
            <motion.line
              x1="310"
              y1="310"
              x2="90"
              y2="310"
              stroke="currentColor"
              strokeWidth="2"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 3, ease: 'easeInOut', delay: 0.6 }}
              style={{ filter: 'drop-shadow(0 0 8px rgba(251, 191, 36, 0.8))' }}
            />
            <motion.line
              x1="90"
              y1="310"
              x2="50"
              y2="130"
              stroke="currentColor"
              strokeWidth="2"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 3, ease: 'easeInOut', delay: 0.9 }}
              style={{ filter: 'drop-shadow(0 0 8px rgba(251, 191, 36, 0.8))' }}
            />
            <motion.line
              x1="50"
              y1="130"
              x2="200"
              y2="50"
              stroke="currentColor"
              strokeWidth="2"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 3, ease: 'easeInOut', delay: 1.2 }}
              style={{ filter: 'drop-shadow(0 0 8px rgba(251, 191, 36, 0.8))' }}
            />

            {/* Inner geometric structure */}
            <motion.line
              x1="200"
              y1="200"
              x2="200"
              y2="50"
              stroke="currentColor"
              strokeWidth="1.5"
              opacity="0.6"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 3, ease: 'easeInOut', delay: 0.15 }}
            />
            <motion.line
              x1="200"
              y1="200"
              x2="350"
              y2="130"
              stroke="currentColor"
              strokeWidth="1.5"
              opacity="0.6"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 3, ease: 'easeInOut', delay: 0.45 }}
            />
            <motion.line
              x1="200"
              y1="200"
              x2="310"
              y2="310"
              stroke="currentColor"
              strokeWidth="1.5"
              opacity="0.6"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 3, ease: 'easeInOut', delay: 0.75 }}
            />
            <motion.line
              x1="200"
              y1="200"
              x2="90"
              y2="310"
              stroke="currentColor"
              strokeWidth="1.5"
              opacity="0.6"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 3, ease: 'easeInOut', delay: 1.05 }}
            />
            <motion.line
              x1="200"
              y1="200"
              x2="50"
              y2="130"
              stroke="currentColor"
              strokeWidth="1.5"
              opacity="0.6"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 3, ease: 'easeInOut', delay: 1.35 }}
            />

            {/* Corner nodes with pulsing glow */}
            {[
              [200, 50],
              [350, 130],
              [310, 310],
              [90, 310],
              [50, 130],
              [200, 200],
            ].map((pos, i) => (
              <motion.circle
                key={`node-${i}`}
                cx={pos[0]}
                cy={pos[1]}
                r="6"
                fill="currentColor"
                animate={{
                  r: [6, 8, 6],
                  filter: [
                    'drop-shadow(0 0 4px rgba(251, 191, 36, 0.5))',
                    'drop-shadow(0 0 12px rgba(251, 191, 36, 1))',
                    'drop-shadow(0 0 4px rgba(251, 191, 36, 0.5))',
                  ],
                }}
                transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
              />
            ))}

            {/* Cyan accent for network feel */}
            <motion.circle
              cx="200"
              cy="200"
              r="40"
              stroke="rgba(6, 182, 212, 0.3)"
              strokeWidth="1"
              fill="none"
              animate={{ r: [40, 60, 40] }}
              transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
            />
          </motion.g>
        </motion.svg>
      </div>

      {/* Text overlay */}
      <motion.div
        className="absolute bottom-20 text-center pointer-events-none"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 2, duration: 1 }}
      >
        <p className="text-amber-400 font-terminal text-lg font-bold tracking-widest">
          SYSTEM INITIALIZING
        </p>
        <p className="text-gray-500 font-terminal text-xs mt-2">
          Click or press any key to continue
        </p>
      </motion.div>
    </motion.div>
  );
}
