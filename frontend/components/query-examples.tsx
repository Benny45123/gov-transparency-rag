'use client';

import { motion } from 'framer-motion';

interface Example {
  category: string;
  queries: string[];
  icon: string;
}

const examples: Example[] = [
  {
    category: 'People',
    icon: 'PER',
    queries: [
      'Who flew on the Lolita Express most frequently?',
      'What is known about Ghislaine Maxwell\'s role?',
    ],
  },
  {
    category: 'Documents',
    icon: 'DOC',
    queries: [
      'What documents mention Les Wexner?',
      'Which court filings describe the plea deal?',
    ],
  },
  {
    category: 'Locations',
    icon: 'LOC',
    queries: [
      'What happened at the Palm Beach property?',
      'What do we know about Little St. James Island?',
    ],
  },
];

interface QueryExamplesProps {
  onSelectQuery: (query: string) => void;
  isLoading?: boolean;
}

export function QueryExamples({
  onSelectQuery,
  isLoading = false,
}: QueryExamplesProps) {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15,
        delayChildren: 0.3,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.4, ease: 'easeOut' },
    },
  };

  const queryVariants = {
    hidden: { opacity: 0, x: -10 },
    visible: {
      opacity: 1,
      x: 0,
      transition: { duration: 0.3 },
    },
  };

  return (
    <motion.div
      className="w-full max-w-4xl mx-auto mt-12 mb-8"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {examples.map((example) => (
          <motion.div key={example.category} variants={itemVariants}>
            <div className="glass rounded-lg border border-amber-500/20 p-6 hover:border-amber-500/50 transition-colors">
              {/* Header */}
              <div className="flex items-center gap-3 mb-4">
                <div className="px-3 py-1 rounded bg-amber-500/20 border border-amber-500/40">
                  <span className="text-amber-400 font-terminal font-bold text-xs uppercase">
                    {example.icon}
                  </span>
                </div>
                <h3 className="text-gray-200 font-terminal font-bold uppercase text-sm">
                  {example.category}
                </h3>
              </div>

              {/* Queries */}
              <div className="space-y-3">
                {example.queries.map((query, idx) => (
                  <motion.button
                    key={idx}
                    onClick={() => !isLoading && onSelectQuery(query)}
                    disabled={isLoading}
                    variants={queryVariants}
                    whileHover={!isLoading ? { x: 5 } : {}}
                    className="w-full text-left p-3 rounded border border-gray-700/50 hover:border-cyan-500/50 bg-black/40 hover:bg-black/60 transition-all disabled:opacity-50 disabled:cursor-not-allowed group"
                  >
                    <p className="text-gray-300 text-xs font-terminal leading-relaxed group-hover:text-cyan-400 transition-colors">
                      {query}
                    </p>
                  </motion.button>
                ))}
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
